functions {
  // Gamma distribution-shaped curve with mode (peak) at (x, y) = (loc, 1) and
  // standard deviation sigma. Both loc and sigma must be > 0. Note that the
  // curve is *not* a valid statistical distribution because its integral is not
  // guaranteed to be 1. To obtain a curve with peak at y = A, simply multiply
  // the returned value by A. See also gamma_c() to compute the required
  // argument c from loc and sigma.
  real gamma_loc_sigma(real x, real loc, real c) {
    if (x == 0.0) {
      return 0.0;
    } else {
      return exp(c * (log(x) - log(loc) - x/loc + 1));
    }
  }

  // Version for a vector argument.
  vector gamma_loc_sigma(vector x, real loc, real c) {
    real log_loc = log(loc);
    int n = num_elements(x);
    vector[n] res;
    for (i in 1:n) {
      if (x[i] == 0.0) {
        res[i] = 0.0;
      } else {
        res[i] = exp(c * (log(x[i]) - log_loc - x[i]/loc + 1));
      }
    }
    return res;
  }

  // Version for per-scan loc and c (one value per observation).
  vector gamma_loc_sigma(vector x, vector loc, vector c) {
    return exp(c .* (log(x) - log(loc) - x ./ loc + 1));
  }

  // Compute c for gamma_loc_sigma() given loc, the location of the mode (peak),
  // and sigma, the standard deviation of the curve.
  real gamma_c(real loc, real sigma) {
    real loc2 = square(loc);
    real sigma2 = square(sigma);
    return 0.5 * (loc2 + loc*sqrt(loc2 + 4*sigma2)) / sigma2;
  }
}

//single-gamma dip only; tests the impact that gender has on beta (the amplitude)
//scans are modeled individually (not averaged), with scan-level random effects
//  on amplitude nested within patient. Dip timing is COMPLETELY POOLED: a single
//  tp_dip shared across all patients and scans, parameterized on the natural
//  (TR) scale like width_dip.
//residual noise is now PER PATIENT: sigma_p[p], hierarchical on the log scale.
data {
  int<lower=0> n_obs;
  int<lower=1> n_ppl;
  int<lower=1> n_scans;                                  // total scans across all patients; ragged OK

  array[n_obs] int<lower=1, upper=n_scans> scan;         // obs -> scan
  array[n_scans] int<lower=1, upper=n_ppl> scan_patient; // scan -> patient

  vector[n_obs] y;
  vector<lower=0>[n_obs] x;   // TR index; t == 0 handled by gamma_loc_sigma()
  vector[n_ppl] gender;       // boy = .5, girl = -.5 (per patient, aligned to patient index)
}

transformed data {
  // obs -> patient, composed from the two maps already passed in. Derived here
  // rather than passed from Python so the two cannot get out of sync.
  array[n_obs] int obs_patient;
  for (i in 1:n_obs) obs_patient[i] = scan_patient[scan[i]];
}

parameters {
  //timing of the dip -> (a-1)/b, in TRs. Shared across patients and scans.
  real<lower=0> tp_dip;

  //width of dip -> SD of the curve in TRs, shared across patients and scans.
  //  Converted to the gamma's shape parameter c via gamma_c(tp_dip, width_dip).
  real<lower=0> width_dip;

  //amplitude, log scale
  real mu_beta_log;     //mean amplitude
  real beta_gender;                // boy-girl difference in amplitude (gender coded +-0.5)
  real<lower=0> sigma_beta;        //sd in amplitude
  vector[n_ppl] beta_log;          // patient level, centered: sampled directly
  real<lower=0> sigma_beta_scan;   // scan-to-scan SD within a patient, shared across patients
  vector[n_scans] z_beta;          // scan level, non-centered: standardized offsets

  //per-patient residual noise, hierarchical on the log scale.
  //  log scale keeps sigma_p > 0 without a hard boundary, and makes the spread
  //  multiplicative (a quiet patient is "half as noisy", not "0.9 units less").
  real mu_log_sigma;                // population median log noise
  real<lower=0> sigma_log_sigma;    // spread of log noise across patients
  vector[n_ppl] z_sigma;            // non-centered: standardized offsets
}

transformed parameters {
  //expand patient values out to one per scan, then offset each scan.
  //  scans from the same patient share the patient term and differ only
  //  by sigma_beta_scan * z_beta
  vector[n_scans] beta_log_s = beta_log[scan_patient] + sigma_beta_scan * z_beta;
  vector[n_scans] beta_s     = exp(beta_log_s);

  //one shape parameter for the whole dataset, since both tp and width are pooled
  real c_dip = gamma_c(tp_dip, width_dip);

  //per-patient noise SD, back on the natural scale
  vector<lower=0>[n_ppl] sigma_p = exp(mu_log_sigma + sigma_log_sigma * z_sigma);
}

model{
  //tp dip prior (natural scale: ~11 TR, +-30% at 1sd)
  tp_dip        ~ normal(11, 3);

  //width of the dip (SD of the curve, in TRs)
  width_dip     ~ normal(5, 2);

  //beta priors (log scale: exp(0.7) ~ 2 BOLD units)
  mu_beta_log     ~ normal(0.7, 0.5);
  beta_gender     ~ normal(0, 0.2);     // multiplicative: ~ +-40% depth at 2sd
  sigma_beta      ~ normal(0, 0.1);     // fractional SD across patients
  beta_log        ~ normal(mu_beta_log + beta_gender * gender, sigma_beta);
  sigma_beta_scan ~ normal(0, 0.1);     // fractional SD across scans within patient; PRIOR IS A GUESS
  z_beta          ~ std_normal();       // non-centered; real prior is normal(beta_log, sigma_beta_scan)

  //per-patient noise priors.
  //  exp(0.6) ~ 1.8, the pooled-sigma posterior from the previous fit.
  //  sigma_log_sigma of 0.5 allows a ~2.7x spread between quiet and noisy
  //  patients at 1sd.
  mu_log_sigma     ~ normal(0.6, 0.5);
  sigma_log_sigma  ~ normal(0, 0.5);
  z_sigma          ~ std_normal();      // non-centered; real prior is lognormal(mu_log_sigma, sigma_log_sigma)

  vector[n_obs] mu = -beta_s[scan] .* gamma_loc_sigma(x, tp_dip, c_dip);

  //likelihood -- each observation scored against its own patient's noise SD.
  //  sigma_p[obs_patient] gathers the length-n_ppl vector out to length n_obs;
  //  still a single vectorized normal_lpdf call.
  y ~ normal(mu, sigma_p[obs_patient]);
}

generated quantities {
  // boy-girl difference in dip depth, BOLD units.
  // positive = boys dip deeper (beta is magnitude; sign applied in mu).
  // unaffected by the scan level: gender is a patient attribute, and both
  // terms are population-level.
  real mem_gender = exp(mu_beta_log + 0.5*beta_gender)
                  - exp(mu_beta_log - 0.5*beta_gender);

  // median patient's noise SD, for comparison against the pooled-sigma fit.
  // NB this is the MEDIAN, not the mean: the mean of a lognormal sits above it.
  real sigma_median = exp(mu_log_sigma);
}