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
  vector gamma_c(vector loc, vector sigma) {
    vector[num_elements(loc)] loc2 = square(loc);
    vector[num_elements(sigma)] sigma2 = square(sigma);
    return 0.5 * (loc2 + loc .* sqrt(loc2 + 4*sigma2)) ./ sigma2;
  }
}

//tp:    population -> scan (centered, bounded 4-28 TR)
//width: population -> scan (non-centered)
//beta:  population -> patient (non-centered) -> scan (non-centered)
//noise: per patient, hierarchical on the log scale.
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
  //timing, log scale. Population -> scan directly; no patient level.
  real mu_tp_log;
  real<lower=0> sigma_tp_scan;                           // scan-to-scan
  vector[n_scans] tp_log_s; // scan level, bounded

  //width -> SD of the curve in TRs, log scale. Population -> scan directly.
  real mu_width_log;
  real<lower=0> sigma_width_scan;   // scan-to-scan
  vector[n_scans] z_width;          // scan level, non-centered

  //amplitude, log scale.
  real mu_beta_log;
  real beta_gender;                 // boy-girl difference (gender coded +-0.5)
  real<lower=0> sigma_beta;         // between patients
  vector[n_ppl] z_beta_p;           // patient level, non-centered
  real<lower=0> sigma_beta_scan;    // within patient
  vector[n_scans] z_beta;           // scan level, non-centered

  //per-patient residual noise, log scale.
  real mu_log_sigma;
  real<lower=0> sigma_log_sigma;
  vector[n_ppl] z_sigma;
}

transformed parameters {
  //scan-level timing. This is what the model uses.
  vector<lower=0>[n_scans] tp_dip_s = exp(tp_log_s);

  vector[n_ppl] beta_log = mu_beta_log + beta_gender * gender + sigma_beta * z_beta_p;

  //expand patient depth to one per scan, then offset each scan.
  vector[n_scans] beta_log_s = beta_log[scan_patient] + sigma_beta_scan * z_beta;
  vector[n_scans] beta_s     = exp(beta_log_s);

  //width goes straight from population to scan.
  vector[n_scans] width_log_s = mu_width_log + sigma_width_scan * z_width;
  vector<lower=0>[n_scans] width_dip_s = exp(width_log_s);

  //per-scan gamma shape. Both arguments are per scan.
  vector[n_scans] c_dip_s = gamma_c(tp_dip_s, width_dip_s);

  vector<lower=0>[n_ppl] sigma_p = exp(mu_log_sigma + sigma_log_sigma * z_sigma);
}

model{
  //tp priors. exp(2.4) ~ 11 TR.
  mu_tp_log     ~ normal(2.4, 0.3);
  sigma_tp_scan ~ normal(0, 1/sqrt(n_scans));
  tp_log_s      ~ normal(mu_tp_log, sigma_tp_scan);

  //width priors. exp(1.6) ~ 5 TR.
  mu_width_log      ~ normal(1.6, 0.3);
  sigma_width_scan  ~ normal(0, 0.3);
  z_width           ~ std_normal();

  //beta priors. exp(0.55) ~ 1.7 BOLD units.
  mu_beta_log     ~ normal(0.55, 0.28);
  beta_gender     ~ normal(0, 0.2);
  sigma_beta      ~ normal(0, 0.2);
  z_beta_p        ~ std_normal();
  sigma_beta_scan ~ normal(0, 0.5);     // PRIOR IS A GUESS
  z_beta          ~ std_normal();

  //noise priors. exp(0.6) ~ 1.8.
  mu_log_sigma     ~ normal(0.6, 0.5);
  sigma_log_sigma  ~ normal(0, 0.5);
  z_sigma          ~ std_normal();

  vector[n_obs] mu = -beta_s[scan] .* gamma_loc_sigma(x, tp_dip_s[scan], c_dip_s[scan]);

  y ~ normal(mu, sigma_p[obs_patient]);
}

generated quantities {
  // boy-girl difference in dip depth, BOLD units. Positive = boys dip deeper.
  real mem_gender = exp(mu_beta_log + 0.5*beta_gender)
                  - exp(mu_beta_log - 0.5*beta_gender);

  real sigma_median = exp(mu_log_sigma);   // MEDIAN patient noise SD, not mean
  real width_median = exp(mu_width_log);   // population median width, TRs
  real tp_median    = exp(mu_tp_log);      // population median timing, TRs
}