functions {
  // Double gamma HRF shape, each component normalized to UNIT PEAK analytically.
  // g(t) = exp((a-1)log t - b t + a log b - lgamma(a)); its peak is at the mode
  // t* = (a-1)/b. Dividing by g(t*) makes each gamma peak at exactly 1, with NO
  // dependence on the data -> fully vectorizable, no per-patient loop.
  // Returns h = g_bump(unit) - c * g_dip(unit): a unit bump minus c*(unit dip).
  // beta then carries amplitude; c carries the dip/bump ratio.
  vector double_gamma(vector t, real a_bump, real b_bump, real a_dip, real b_dip, real c) {
    int N = num_elements(t);
    vector[N] h;
    // Analytic peak heights at each gamma's mode t* = (a-1)/b.
    // log g(t*) = (a-1)*log((a-1)/b) - (a-1) + a*log(b) - lgamma(a)
    real lpk_bump = (a_bump - 1) * log((a_bump - 1) / b_bump) - (a_bump - 1)
                    + a_bump * log(b_bump) - lgamma(a_bump);
    real lpk_dip  = (a_dip  - 1) * log((a_dip  - 1) / b_dip)  - (a_dip  - 1)
                    + a_dip  * log(b_dip)  - lgamma(a_dip);
    for (n in 1:N) {
      if (t[n] <= 0) {
        h[n] = 0;
      } else {
        // log gamma value minus log peak height => unit-peak gamma after exp.
        real g_bump = exp((a_bump - 1) * log(t[n]) - b_bump * t[n]
                          + a_bump * log(b_bump) - lgamma(a_bump) - lpk_bump);
        real g_dip  = exp((a_dip  - 1) * log(t[n]) - b_dip  * t[n]
                          + a_dip  * log(b_dip)  - lgamma(a_dip)  - lpk_dip);
        h[n] = g_bump - c * g_dip;
      }
    }
    return h;
  }
}

data {
  int<lower=0> n_obs;
  int<lower=1> n_ppl;
  array[n_obs] int<lower=1, upper=n_ppl> patient;
  vector[n_obs] y;
  vector[n_obs] x;
}

parameters {
  // Each shape feature is per-patient, partially pooled, non-centered.
  // Population means (mu_*), between-patient SDs (sigma_*), z-score offsets (*_raw).

  // Dip peak time
  real mu_tp_dip;
  real<lower=0> sigma_tp_dip;
  vector[n_ppl] tp_dip_raw;

  // Bump peak time
  real mu_tp_bump;
  real<lower=0> sigma_tp_bump;
  vector[n_ppl] tp_bump_raw;

  // Dip width (rate), log-space for positivity
  real mu_log_b_dip;
  real<lower=0> sigma_log_b_dip;
  vector[n_ppl] log_b_dip_raw;

  // Bump width (rate), log-space for positivity
  real mu_log_b_bump;
  real<lower=0> sigma_log_b_bump;
  vector[n_ppl] log_b_bump_raw;

  // Dip/bump ratio c, in log-space for positivity
  real mu_log_c;
  real<lower=0> sigma_log_c;
  vector[n_ppl] log_c_raw;

  // Amplitude
  real mu_beta;
  real<lower=0> sigma_beta;
  vector[n_ppl] beta_raw;

  // Observation noise
  real<lower=0> sigma;
}

transformed parameters {
  // Rebuild per-patient values from non-centered offsets.
  vector[n_ppl] tp_dip  = mu_tp_dip  + sigma_tp_dip  * tp_dip_raw;
  vector[n_ppl] tp_bump = mu_tp_bump + sigma_tp_bump * tp_bump_raw;

  // Widths via log-space (positive by construction).
  vector[n_ppl] b_dip  = exp(mu_log_b_dip  + sigma_log_b_dip  * log_b_dip_raw);
  vector[n_ppl] b_bump = exp(mu_log_b_bump + sigma_log_b_bump * log_b_bump_raw);

  // Dip/bump ratio (positive) and amplitude.
  vector[n_ppl] c    = exp(mu_log_c + sigma_log_c * log_c_raw);
  vector[n_ppl] beta = mu_beta + sigma_beta * beta_raw;

  // Per-patient gamma shapes from peak time + rate: a = tp*b + 1 (a > 1 guaranteed).
  vector[n_ppl] a_dip  = tp_dip  .* b_dip  + 1.01;
  vector[n_ppl] a_bump = tp_bump .* b_bump + 1.01;
}

model {
  // --- Hyperpriors: timing tight & separated so dip stays before bump ---
  mu_tp_dip     ~ normal(13, 3);
  sigma_tp_dip  ~ normal(0, 1);
  tp_dip_raw    ~ normal(0, 1);

  mu_tp_bump    ~ normal(27, 3);
  sigma_tp_bump ~ normal(0, 1);
  tp_bump_raw   ~ normal(0, 1);

  // --- Widths ---
  mu_log_b_dip     ~ normal(-1.5, 0.5);
  sigma_log_b_dip  ~ normal(0, 0.2);
  log_b_dip_raw    ~ normal(0, 1);

  mu_log_b_bump    ~ normal(-1.5, 0.5);
  sigma_log_b_bump ~ normal(0, 0.2);
  log_b_bump_raw   ~ normal(0, 1);

  // --- Dip/bump ratio ---
  mu_log_c     ~ normal(0, 0.5);  // c centered near 1
  sigma_log_c  ~ normal(0, 0.2);
  log_c_raw    ~ normal(0, 1);

  // --- Amplitude ---
  mu_beta      ~ normal(0, 10);
  sigma_beta   ~ normal(0, 5);
  beta_raw     ~ normal(0, 1);

  sigma ~ normal(0, 5);

  // --- Vectorized likelihood: per-observation shape via [patient] indexing ---
  vector[n_obs] mu;
  for (i in 1:n_obs) {
    int p = patient[i];
    real tt = x[i];
    if (tt <= 0) {
      mu[i] = 0;
    } else {
      real lpk_bump = (a_bump[p]-1)*log((a_bump[p]-1)/b_bump[p]) - (a_bump[p]-1)
                      + a_bump[p]*log(b_bump[p]) - lgamma(a_bump[p]);
      real lpk_dip  = (a_dip[p]-1)*log((a_dip[p]-1)/b_dip[p]) - (a_dip[p]-1)
                      + a_dip[p]*log(b_dip[p]) - lgamma(a_dip[p]);
      // log of each unit-peak gamma; clamp to avoid exp() overflow -> inf -> NaN
      real lg_bump = (a_bump[p]-1)*log(tt) - b_bump[p]*tt + a_bump[p]*log(b_bump[p]) - lgamma(a_bump[p]) - lpk_bump;
      real lg_dip  = (a_dip[p]-1)*log(tt) - b_dip[p]*tt + a_dip[p]*log(b_dip[p]) - lgamma(a_dip[p]) - lpk_dip;
      real g_bump = exp(fmin(lg_bump, 20));   // exp(20)~5e8, safely below overflow
      real g_dip  = exp(fmin(lg_dip, 20));
      mu[i] = beta[p] * (g_bump - c[p] * g_dip);
    }
  }
  y ~ normal(mu, sigma);
}