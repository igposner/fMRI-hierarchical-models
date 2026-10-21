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

//tp:    population -> scan (centered)
//width: population -> scan (non-centered)
//beta:  population -> patient (non-centered) -> scan (non-centered)
//noise: two independent per-patient hierarchies, log scale.
//       inside TR 5-30: sigma_p with AR(1). outside: sigma_out_p, iid.
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
  // obs -> patient, composed from the two maps already passed in.
  array[n_obs] int obs_patient;
  for (i in 1:n_obs) obs_patient[i] = scan_patient[scan[i]];

  // OU window, inclusive
  real ou_lo = 5;
  real ou_hi = 30;

  // Requires rows sorted by scan, then by x within scan.
  array[n_obs] int is_start;      // 1 = first obs of its scan
  array[n_obs] int in_win;        // 1 = inside OU window
  vector[n_obs] dt;               // gap in TRs from previous obs; 0 at scan starts
  array[n_scans] int n_starts = rep_array(0, n_scans);

  for (i in 1:n_obs) {
    is_start[i] = (i == 1 || scan[i] != scan[i - 1]);
    in_win[i]   = (x[i] >= ou_lo && x[i] <= ou_hi);
    if (is_start[i]) {
      dt[i] = 0;
      n_starts[scan[i]] += 1;
    } else {
      dt[i] = x[i] - x[i - 1];
      if (dt[i] <= 0) reject("x not increasing within scan at obs ", i);
    }
  }

  for (s in 1:n_scans) {
    if (n_starts[s] != 1) reject("scan ", s, " has ", n_starts[s], " blocks; sort rows by scan");
  }

  // 0 = outside (iid), 1 = window chain start (marginal), 2 = window conditional (AR)
  array[n_obs] int cls;
  int n_out = 0;
  int n_wstart = 0;
  int n_wcond = 0;
  for (i in 1:n_obs) {
    if (!in_win[i]) {
      cls[i] = 0;
    } else if (is_start[i]) {
      cls[i] = 1;
    } else if (!in_win[i - 1]) {
      cls[i] = 1;
    } else {
      cls[i] = 2;
    }
    if (cls[i] == 0) n_out += 1;
    if (cls[i] == 1) n_wstart += 1;
    if (cls[i] == 2) n_wcond += 1;
  }

  array[n_out] int idx_out;
  array[n_wstart] int idx_wstart;
  array[n_wcond] int idx_wcond;
  array[n_wcond] int idx_wprev;
  vector[n_wcond] dt_wcond;
  {
    int a = 1;
    int b = 1;
    int c = 1;
    for (i in 1:n_obs) {
      if (cls[i] == 0) {
        idx_out[a] = i;
        a += 1;
      } else if (cls[i] == 1) {
        idx_wstart[b] = i;
        b += 1;
      } else {
        idx_wcond[c] = i;
        idx_wprev[c] = i - 1;
        dt_wcond[c]  = dt[i];
        c += 1;
      }
    }
  }
  array[n_out] int pat_out = obs_patient[idx_out];
}

parameters {
  real<lower=0, upper=1> phi;

  //timing, log scale. Population -> scan directly; no patient level.
  real mu_tp_log;
  real<lower=0> sigma_tp_scan;      // scan-to-scan
  vector[n_scans] tp_log_s;

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

  //per-patient noise inside the window, log scale.
  real mu_log_sigma;
  real<lower=0> sigma_log_sigma;
  vector[n_ppl] z_sigma;

  //per-patient noise outside the window, log scale. Independent of inside.
  real mu_log_sigma_out;
  real<lower=0> sigma_log_sigma_out;
  vector[n_ppl] z_sigma_out;
}

transformed parameters {
  vector<lower=0>[n_scans] tp_dip_s = exp(tp_log_s);

  vector[n_ppl] beta_log = mu_beta_log + beta_gender * gender + sigma_beta * z_beta_p;

  vector[n_scans] beta_log_s = beta_log[scan_patient] + sigma_beta_scan * z_beta;
  vector[n_scans] beta_s     = exp(beta_log_s);

  vector[n_scans] width_log_s = mu_width_log + sigma_width_scan * z_width;
  vector<lower=0>[n_scans] width_dip_s = exp(width_log_s);

  vector[n_scans] c_dip_s = gamma_c(tp_dip_s, width_dip_s);

  vector<lower=0>[n_ppl] sigma_p     = exp(mu_log_sigma + sigma_log_sigma * z_sigma);
  vector<lower=0>[n_ppl] sigma_out_p = exp(mu_log_sigma_out + sigma_log_sigma_out * z_sigma_out);
}

model {
  phi ~ beta(8, 2);

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

  //inside noise priors. exp(0.6) ~ 1.8.
  mu_log_sigma     ~ normal(0.6, 0.5);
  sigma_log_sigma  ~ normal(0, 0.5);
  z_sigma          ~ std_normal();

  //outside noise priors. exp(1.0) ~ 2.7, believed bigger than inside.
  mu_log_sigma_out     ~ normal(1.0, 0.5);
  sigma_log_sigma_out  ~ normal(0, 0.5);
  z_sigma_out          ~ std_normal();

  vector[n_obs] mu = -beta_s[scan] .* gamma_loc_sigma(x, tp_dip_s[scan], c_dip_s[scan]);
  vector[n_obs] e  = y - mu;
  vector[n_obs] s_in = sigma_p[obs_patient];

  //AR(1) inside window; chain restarts at each scan and on re-entry
  real log_phi = log(phi);
  vector[n_wcond] rho = exp(dt_wcond * log_phi);                                         // phi^dt
  vector[n_wcond] sd_innov = s_in[idx_wcond] .* sqrt(-expm1(2 * dt_wcond * log_phi));   // s * sqrt(1 - phi^(2dt))

  target += normal_lpdf(e[idx_wstart] | 0, s_in[idx_wstart]);
  target += normal_lpdf(e[idx_wcond] | rho .* e[idx_wprev], sd_innov);

  //iid outside window
  target += normal_lpdf(e[idx_out] | 0, sigma_out_p[pat_out]);
}

generated quantities {
  // boy-girl difference in dip depth, BOLD units. Positive = boys dip deeper.
  real mem_gender = exp(mu_beta_log + 0.5*beta_gender)
                  - exp(mu_beta_log - 0.5*beta_gender);

  real sigma_median     = exp(mu_log_sigma);       // MEDIAN patient noise SD, inside window
  real sigma_out_median = exp(mu_log_sigma_out);   // MEDIAN patient noise SD, outside window
  real width_median     = exp(mu_width_log);       // population median width, TRs
  real tp_median        = exp(mu_tp_log);          // population median timing, TRs
}