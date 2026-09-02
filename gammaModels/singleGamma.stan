//single-gamma dip only; tests the impact that gender has on beta (the amplitude)
data {
  int<lower=0> n_obs;
  int<lower=1> n_ppl;
  array[n_obs] int<lower=1, upper=n_ppl> patient;
  vector[n_obs] y;
  vector<lower=0>[n_obs] x;   // t==0 rows must be dropped in Python (log(0) -> -inf)
  vector[n_ppl] gender;       // boy = .5, girl = -.5 (per patient, aligned to patient index)
}

parameters {
  //timing of the dip -> (a-1)/b
  real mu_tp_dip;
  real<lower=0> sigma_tp_dip;
  vector<lower=0>[n_ppl] tp_dip;   // centered: sampled directly

  //width of dip (dispersion)
  real<lower=0.01> b_dip;    // shared across patients

  //amplitudes
  real<lower=0> mu_beta;     //mean amplitude
  real beta_gender;          // boy-girl difference in amplitude (gender coded +-0.5)
  real<lower=0> sigma_beta;  //sd in amplitude
  vector[n_ppl] beta;        // centered: sampled directly

  //population noise
  real<lower=1e-6> sigma_early;
  real<lower=1e-6> sigma_late;
}

model {

  //tp dip priors
  mu_tp_dip     ~ normal(9, 3);   //informed by the mean curve
  sigma_tp_dip  ~ normal(0, 1);
  tp_dip        ~ normal(mu_tp_dip, sigma_tp_dip);

  //dispersion
  b_dip         ~ lognormal(-1.5, 0.5);

  //beta priors, this is what the predictor is attatched - normalized
  mu_beta       ~ normal(0, 10);
  beta_gender   ~ normal(0, 5);
  sigma_beta    ~ normal(0, 5);
  beta          ~ normal(mu_beta + beta_gender * gender, sigma_beta);
  //population noise
  sigma_early   ~ normal(0, 5);
  sigma_late    ~ normal(0, 5);

  //making a vector of length n_obs that has the tps
  vector[n_obs] td = tp_dip[patient];

  //unit-peak gamma: peaks at exactly 1 when x == td
  vector[n_obs] lg_dip = (td * b_dip) .* (log(x) - log(td)) - b_dip * (x - td);
  vector[n_obs] mu = -beta[patient] .* exp(lg_dip);   // beta > 0 -> downward dip

  //likelyhood
  // HARDCODED breakpoint: dip sits near t~10, so this splits signal from tail.
  // Promote to a data field before comparing runs across different values.
  real t_split = 50.0;
  vector[n_obs] sig;
  for (i in 1:n_obs) sig[i] = x[i] > t_split ? sigma_late : sigma_early;
  y ~ normal(mu, sig);
}

generated quantities {
  vector[n_obs] td_debug = tp_dip[patient];
  vector[n_obs] lg_dip_debug = (td_debug * b_dip) .* (log(x) - log(td_debug))
                               - b_dip * (x - td_debug);
  vector[n_obs] mu_debug = -beta[patient] .* exp(lg_dip_debug);
}