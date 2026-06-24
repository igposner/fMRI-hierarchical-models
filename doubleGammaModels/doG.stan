functions {
  // Double gamma HRF shape, h(t) = g1(t) - c*g2(t)
  // a1,b1: peak gamma; a2,b2: undershoot gamma; c: undershoot weight
  vector double_gamma(vector t, real a1, real b1, real a2, real b2, real c) {
    vector[num_elements(t)] g1;
    vector[num_elements(t)] g2;
    // gamma pdf form: t^(a-1) * exp(-b*t) * b^a / Gamma(a)
    g1 = exp((a1 - 1) * log(t) - b1 * t + a1 * log(b1) - lgamma(a1));
    g2 = exp((a2 - 1) * log(t) - b2 * t + a2 * log(b2) - lgamma(a2));
    return g1 - c * g2;
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
  // --- Shared shape parameters (global, one value across all patients) ---
  real<lower=0> a1;          // peak gamma shape
  real<lower=0> b1;          // peak gamma rate
  real<lower=0> a2;          // undershoot gamma shape
  real<lower=0> b2;          // undershoot gamma rate
  real<lower=0> c;           // undershoot weight (height of g2 relative to g1)

  // --- Amplitude: partially pooled per-patient, non-centered ---
  real mu_beta;              // population-mean amplitude
  real<lower=0> sigma_beta;  // between-patient SD of amplitude
  vector[n_ppl] beta_raw;    // standardized per-patient offsets (z-scores)

  // --- Observation noise ---
  real<lower=0> sigma;       // shared residual SD across all observations
}

transformed parameters {
  // Non-centered amplitude: rebuild real per-patient amplitudes from
  // standard-normal offsets. beta_raw[p] ~ N(0,1) is what the sampler
  // explores; this line rescales it to N(mu_beta, sigma_beta).
  // Saved to output so we get per-patient amplitudes directly.
  vector[n_ppl] beta = mu_beta + sigma_beta * beta_raw;
}

model {
  // --- Priors (wide / weakly-informative, all positive via half-normal) ---
  a1 ~ normal(0, 10);        // peak gamma shape   (>0 by declaration)
  b1 ~ normal(0, 10);        // peak gamma rate
  a2 ~ normal(0, 10);        // undershoot gamma shape
  b2 ~ normal(0, 10);        // undershoot gamma rate
  c  ~ normal(0, 5);         // undershoot weight

  // --- Hierarchy on amplitude ---
  mu_beta     ~ normal(0, 10);   // population-mean amplitude
  sigma_beta  ~ normal(0, 5);    // between-patient SD (half-normal, >0)
  beta_raw    ~ normal(0, 1);    // standard-normal offsets (non-centered)

  // --- Observation noise ---
  sigma ~ normal(0, 5);          // residual SD (half-normal, >0)

  // --- Likelihood ---
  vector[n_obs] mu = beta[patient] .* double_gamma(x, a1, b1, a2, b2, c);
  y ~ normal(mu, sigma);
}
