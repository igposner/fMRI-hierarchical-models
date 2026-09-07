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

  // Vectorized version: elementwise gamma_c for per-scan (loc, sigma) pairs.
  // loc and sigma must be the same length.
  vector gamma_c(vector loc, vector sigma) {
    vector[num_elements(loc)] loc2 = square(loc);
    vector[num_elements(sigma)] sigma2 = square(sigma);
    return 0.5 * (loc2 + loc .* sqrt(loc2 + 4*sigma2)) ./ sigma2;
  }
}

//single-gamma dip only; tests the impact that gender has on beta (the amplitude)
//scans are modeled individually (not averaged), with scan-level random effects
//  on amplitude AND width nested within patient.
//tp_dip is now PARTIALLY POOLED at the patient level (non-centered, log scale).
//  Previously a single pooled scalar at 10.28 TR, but the empirical per-scan
//  trough in TR 4-30 has median 13.0 and IQR [10.0, 19.8] -- one number cannot
//  cover that spread. No scan level on timing yet; that is a separate question.
//width_dip has a 3-level structure: population -> patient (centered) -> scan
//  (non-centered), on the log scale.
//beta is NON-CENTERED at BOTH the patient and scan levels. The previous
//  centered patient level funnelled: divergences concentrated at small
//  sigma_beta (0.023 divergent vs 0.145 clean), and one chain slid toward
//  sigma_beta -> 0.
//residual noise is PER PATIENT: sigma_p[p], hierarchical on the log scale.
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
  //timing of the dip -> (a-1)/b, in TRs. Hierarchical: population mean ->
  //  patient offset (non-centered), log scale. Log scale matches beta and
  //  width, keeps tp > 0 without a hard boundary, and makes the patient
  //  spread multiplicative.
  real mu_tp_log;                   // population mean log timing
  real<lower=0> sigma_tp;           // patient-to-patient SD, log scale
  vector<lower=log(4), upper=log(25)>[n_ppl] tp_log;

  //width of dip -> SD of the curve in TRs. Hierarchical: population mean
  //  -> patient offset (centered) -> scan offset (non-centered), log scale.
  real mu_width_log;                // population mean log width
  real<lower=0> sigma_width;        // patient-to-patient SD, log scale
  vector[n_ppl] z_width_p;          // patient level, non-centered: standardized offsets
  real<lower=0> sigma_width_scan;   // scan-to-scan SD within a patient, log scale
  vector[n_scans] z_width;          // scan level, non-centered: standardized offsets

  //amplitude, log scale
  real mu_beta_log;     //mean amplitude
  real beta_gender;                // boy-girl difference in amplitude (gender coded +-0.5)
  real<lower=0> sigma_beta;        //sd in amplitude between patients
  vector[n_ppl] z_beta_p;          // patient level, non-centered: standardized offsets
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
  //patient-level dip timing, non-centered, then back to the natural (TR) scale.
  vector<lower=0>[n_ppl] tp_dip_p = exp(tp_log);

  vector[n_ppl] width_log = mu_width_log + sigma_width * z_width_p;

  //patient-level amplitude, built non-centered from the population mean, the
  //  gender regression, and the standardized offsets.
  vector[n_ppl] beta_log = mu_beta_log + beta_gender * gender + sigma_beta * z_beta_p;

  //expand patient values out to one per scan, then offset each scan.
  //  scans from the same patient share the patient term and differ only
  //  by sigma_beta_scan * z_beta
  vector[n_scans] beta_log_s = beta_log[scan_patient] + sigma_beta_scan * z_beta;
  vector[n_scans] beta_s     = exp(beta_log_s);

  //same expansion for width: patient term + scan-level non-centered offset,
  //  then back to the natural (TR) scale.
  vector[n_scans] width_log_s = width_log[scan_patient] + sigma_width_scan * z_width;
  vector<lower=0>[n_scans] width_dip_s = exp(width_log_s);

  //per-scan gamma shape parameter. Both arguments now vary: timing is gathered
  //  from patient to scan, width is already per scan.
  vector[n_scans] c_dip_s = gamma_c(tp_dip_p[scan_patient], width_dip_s);

  //per-patient noise SD, back on the natural scale
  vector<lower=0>[n_ppl] sigma_p = exp(mu_log_sigma + sigma_log_sigma * z_sigma);
}

model{
  //tp dip priors. exp(2.4) ~ 11 TR, matching the old pooled normal(11, 3).
  //  sigma_tp of 0.2 allows roughly 9-13 TR at 1sd. THIS IS A GUESS and may be
  //  too tight: the empirical per-scan trough spans [10.0, 19.8] at the IQR.
  //  If the posterior presses against the top of this prior, widen it.
  mu_tp_log ~ normal(2.4, 0.3);
  sigma_tp  ~ normal(0, 0.3);              // was 0.05
  tp_log    ~ normal(mu_tp_log, sigma_tp); // replaces z_tp_p ~ std_normal()

  //width priors, mirroring beta's 3-level structure.
  //  mu_width_log centered so exp(mu_width_log) ~ 5 TR, matching the old
  //  pooled width_dip ~ normal(5, 2) prior. sigma_width / sigma_width_scan
  //  are first-pass guesses -- validate with prior predictive before trusting.
  mu_width_log      ~ normal(1.6, 0.3);
  sigma_width       ~ normal(0, 0.3);
  z_width_p         ~ std_normal();
  sigma_width_scan  ~ normal(0, 0.3);
  z_width           ~ std_normal();     // non-centered; real prior is normal(width_log, sigma_width_scan)

  //beta priors
  mu_beta_log     ~ normal(0.55, 0.28);  // dip depth between 1 and 3 BOLD units
  beta_gender     ~ normal(0, 0.2);      // the prior on difference between genders
  sigma_beta      ~ normal(0, 0.2);      // fractional SD across patients
  z_beta_p        ~ std_normal();        // non-centered; real prior is normal(mu_beta_log + beta_gender*gender, sigma_beta)
  sigma_beta_scan ~ normal(0, 0.5);      // fractional SD across scans within patient; PRIOR IS A GUESS
  z_beta          ~ std_normal();        // non-centered; real prior is normal(beta_log, sigma_beta_scan)

  //per-patient noise priors.
  //  exp(0.6) ~ 1.8, the pooled-sigma posterior from the previous fit.
  //  sigma_log_sigma of 0.5 allows a ~2.7x spread between quiet and noisy
  //  patients at 1sd.
  mu_log_sigma     ~ normal(0.6, 0.5);
  sigma_log_sigma  ~ normal(0, 0.5);
  z_sigma          ~ std_normal();      // non-centered; real prior is lognormal(mu_log_sigma, sigma_log_sigma)

  //loc is gathered from patient to obs; c_dip_s from scan to obs.
  vector[n_obs] mu = -beta_s[scan] .* gamma_loc_sigma(x, tp_dip_p[obs_patient], c_dip_s[scan]);

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

  // population median dip width, in TRs -- for comparison against the old
  // pooled width_dip posterior.
  real width_median = exp(mu_width_log);

  // population median dip timing, in TRs -- for comparison against the old
  // pooled tp_dip posterior (10.28).
  real tp_median = exp(mu_tp_log);
}