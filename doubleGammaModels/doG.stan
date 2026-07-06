data {
  int<lower=0> n_obs;
  int<lower=1> n_ppl;
  array[n_obs] int<lower=1, upper=n_ppl> patient;
  vector[n_obs] y;
  vector<lower=0>[n_obs] x;   // t==0 rows must be dropped in Python (log(0) -> -inf)
}

parameters {
  real mu_tp_dip;
  real<lower=0> sigma_tp_dip;
  vector[n_ppl] tp_dip_raw;

  real mu_tp_bump;
  real<lower=0> sigma_tp_bump;
  vector[n_ppl] tp_bump_raw;

  real log_b_dip;    // shared across patients
  real log_b_bump;   // shared across patients

  real mu_log_c;
  real<lower=0> sigma_log_c;
  vector[n_ppl] log_c_raw;

  real mu_beta;
  real<lower=0> sigma_beta;
  vector[n_ppl] beta_raw;

  real<lower=0> sigma;
}

transformed parameters {
  vector[n_ppl] tp_dip  = mu_tp_dip  + sigma_tp_dip  * tp_dip_raw;
  vector[n_ppl] tp_bump = mu_tp_bump + sigma_tp_bump * tp_bump_raw;

  real b_dip  = exp(log_b_dip);
  real b_bump = exp(log_b_bump);

  vector[n_ppl] c    = exp(mu_log_c + sigma_log_c * log_c_raw);
  vector[n_ppl] beta = mu_beta + sigma_beta * beta_raw;

  // a = tp*b + 1.01 (a > 1 guaranteed). True mode at tp + 0.01/b, ~0.045 TR off tp.
  vector[n_ppl] a_dip  = fmax(tp_dip  * b_dip,  0.01) + 1.0;
  vector[n_ppl] a_bump = fmax(tp_bump * b_bump, 0.01) + 1.0;

  // Per-patient log peak heights (unit-peak normalization). Precomputed here so the
  // likelihood doesn't recompute them per observation.
  vector[n_ppl] lpk_dip  = (a_dip  - 1) .* log((a_dip  - 1) / b_dip)  - (a_dip  - 1)
                           + a_dip  * log(b_dip)  - lgamma(a_dip);
  vector[n_ppl] lpk_bump = (a_bump - 1) .* log((a_bump - 1) / b_bump) - (a_bump - 1)
                           + a_bump * log(b_bump) - lgamma(a_bump);
}

model {
  mu_tp_dip     ~ normal(13, 3);
  sigma_tp_dip  ~ normal(0, 1);
  tp_dip_raw    ~ normal(0, 1);

  mu_tp_bump    ~ normal(27, 3);
  sigma_tp_bump ~ normal(0, 1);
  tp_bump_raw   ~ normal(0, 1);

  log_b_dip  ~ normal(-1.5, 0.5);
  log_b_bump ~ normal(-1.5, 0.5);

  mu_log_c     ~ normal(0, 0.5);
  sigma_log_c  ~ normal(0, 0.2);
  log_c_raw    ~ normal(0, 1);

  mu_beta      ~ normal(0, 10);
  sigma_beta   ~ normal(0, 5);
  beta_raw     ~ normal(0, 1);

  sigma ~ normal(0, 5);

  // Vectorized unit-peak double gamma; per-obs shape via [patient] indexing.
  vector[n_obs] a_b = a_bump[patient];
  vector[n_obs] a_d = a_dip[patient];
  vector[n_obs] lg_bump = (a_b - 1) .* log(x) - b_bump * x
                          + a_b * log(b_bump) - lgamma(a_b) - lpk_bump[patient];
  vector[n_obs] lg_dip  = (a_d - 1) .* log(x) - b_dip  * x
                          + a_d * log(b_dip)  - lgamma(a_d) - lpk_dip[patient];
  vector[n_obs] mu = beta[patient] .* (exp(lg_bump) - c[patient] .* exp(lg_dip));

  y ~ normal(mu, sigma);
}