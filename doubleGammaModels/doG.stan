functions {
  // Double gamma HRF shape, h(t) = g1(t) - c*g2(t)
  // a1,b1: peak gamma; a2,b2: undershoot gamma; c: undershoot weight
  vector double_gamma(vector t, real a1, real b1, real a2, real b2, real c) {
    int N = num_elements(t);
    vector[N] h;
    for (n in 1:N) {
      if (t[n] <= 0) {
        h[n] = 0;                 // gamma is 0 at the origin; skip log(0)
      } else {
        // gamma pdf form: t^(a-1) * exp(-b*t) * b^a / Gamma(a)
        real g1 = exp((a1 - 1) * log(t[n]) - b1 * t[n] + a1 * log(b1) - lgamma(a1));
        real g2 = exp((a2 - 1) * log(t[n]) - b2 * t[n] + a2 * log(b2) - lgamma(a2));
        h[n] = g1 - c * g2;
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
  // --- Shared shape, parameterized by peak TIME ---
  // Data shows the DIP first (~t=13) then the BUMP (~t=27).
  // In h(t) = g1 - c*g2, g2 is subtracted, so g2 is the early dip and g1 the later bump.
  positive_ordered[2] tp;    // peak times, tp[1] < tp[2], tp[1] = early dip  -> drives g2 (subtracted), tp[2] = later bump -> drives g1 (added)
  real<lower=0> b_dip;       // dip gamma rate   (g2 width)
  real<lower=0> b_bump;      // bump gamma rate  (g1 width)
  real<lower=0> c;           // dip weight (height of subtracted g2)

  // --- Amplitude: partially pooled per-patient, non-centered ---
  real mu_beta;
  real<lower=0> sigma_beta;
  vector[n_ppl] beta_raw;

  // --- Observation noise ---
  real<lower=0> sigma;
}

transformed parameters {
  // Recover gamma shapes from peak time and rate: mode = (a-1)/b => a = tp*b + 1.
  // a > 1 by construction, so each gamma rises from 0 with a real interior peak.
  // tp[1] (early) -> dip gamma g2; tp[2] (late) -> bump gamma g1.
  real a_dip  = tp[1] * b_dip  + 1;   // subtracted gamma, peaks early (~13)
  real a_bump = tp[2] * b_bump + 1;   // added gamma,      peaks later (~27)

  // Non-centered amplitude (unchanged).
  vector[n_ppl] beta = mu_beta + sigma_beta * beta_raw;
}

model {
  // --- Priors: centers read off the mean-signal plot ---
  tp[1] ~ normal(13, 4);     // early dip   (subtracted g2)
  tp[2] ~ normal(27, 5);     // later bump  (added g1)
  b_dip  ~ normal(0, 5);     // rates; smaller => wider gamma
  b_bump ~ normal(0, 5);
  c      ~ normal(0, 5);     // dip weight

  // --- Hierarchy on amplitude (unchanged) ---
  mu_beta    ~ normal(0, 10);
  sigma_beta ~ normal(0, 5);
  beta_raw   ~ normal(0, 1);

  // --- Observation noise ---
  sigma ~ normal(0, 5);

  // --- Likelihood ---
  // double_gamma(t, a1, b1, a2, b2, c) computes g1 - c*g2.
  // g1 = bump (added), g2 = dip (subtracted), so pass bump params as 1, dip as 2.
  vector[n_obs] mu = beta[patient] .* double_gamma(x, a_bump, b_bump, a_dip, b_dip, c);
  y ~ normal(mu, sigma);
}
