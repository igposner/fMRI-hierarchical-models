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

  // Version for per-patient loc and c (one value per observation).
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
data {
  int<lower=0> n_obs;
  int<lower=1> n_ppl;
  array[n_obs] int<lower=1, upper=n_ppl> patient;
  vector[n_obs] y;
  vector<lower=0>[n_obs] x;   // TR index; t == 0 handled by gamma_loc_sigma()
  vector[n_ppl] gender;       // boy = .5, girl = -.5 (per patient, aligned to patient index)
}

parameters {
  //timing of the dip -> (a-1)/b, in log scale
  real mu_tp_dip_log;
  real<lower=0> sigma_tp_dip;
  vector[n_ppl] tp_dip_log;   // centered: sampled directly

  //width of dip -> SD of the curve in TRs, shared across patients.
  //  Converted to the gamma's shape parameter c via gamma_c(tp_dip, width_dip),
  //  so each patient gets a dip of the same width in TRs regardless of tp.
  real<lower=0> width_dip;   

  //amplitude, log scale
  real mu_beta_log;     //mean amplitude
  real beta_gender;          // boy-girl difference in amplitude (gender coded +-0.5)
  real<lower=0> sigma_beta;  //sd in amplitude
  vector[n_ppl] beta_log;    // centered: sampled directly

  //population noise
  real<lower=1e-6> sigma;
}

transformed parameters {
  vector[n_ppl] tp_dip = exp(tp_dip_log);
  vector[n_ppl] beta_p = exp(beta_log);

  vector[n_ppl] c_dip;
  for (j in 1:n_ppl) c_dip[j] = gamma_c(tp_dip[j], width_dip);
}

model{
  //tp dip priors (log scale: exp(2.4) ~ 11 TR)
  mu_tp_dip_log ~ normal(2.4, 0.3);
  sigma_tp_dip  ~ normal(0, 0.15);      // fractional SD across patients
  tp_dip_log    ~ normal(mu_tp_dip_log, sigma_tp_dip);

  //width of the dip (SD of the curve, in TRs)
  width_dip     ~ normal(5, 2);

  //beta priors (log scale: exp(0.7) ~ 2 BOLD units)
  mu_beta_log   ~ normal(0.7, 0.5);
  beta_gender   ~ normal(0, 0.2);      // multiplicative: ~ +-40% depth at 2sd
  sigma_beta    ~ normal(0, 0.1);      // fractional SD across patients
  beta_log      ~ normal(mu_beta_log + beta_gender * gender, sigma_beta);

  //population noise
  sigma ~ normal(0, 5);

  vector[n_obs] mu = -beta_p[patient] .* gamma_loc_sigma(x, tp_dip[patient], c_dip[patient]);

  //likelihood
  y ~ normal(mu, sigma);
}

generated quantities {
  // boy-girl difference in dip depth, BOLD units.
  // positive = boys dip deeper (beta is magnitude; sign applied in mu).
  real mem_gender = exp(mu_beta_log + 0.5*beta_gender)
                  - exp(mu_beta_log - 0.5*beta_gender);
}