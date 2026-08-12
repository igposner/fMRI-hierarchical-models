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
    }
    else {
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
      }
      else {
        res[i] = exp(c * (log(x[i]) - log_loc - x[i]/loc + 1));
      }
    }
    return res;
  }

  // Compute c for gamma_loc_sigma() given x, the location of the mode (peak),
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
  vector<lower=0>[n_obs] x;   // time; x==0 rows must be dropped in Python (log(0) -> -inf)
  vector[n_ppl] gender;       // boy = .5, girl = -.5 (per patient, aligned to patient index)
}

parameters {
  //timing of the dip -> (a-1)/b
  real<lower=0> mu_tp_dip;

  //width of dip (dispersion)
  real<lower=0> sigma_dip;    // shared across patients

  //amplitudes
  real mu_beta_log;     //mean amplitude
  real beta_gender_log;          // boy-girl difference in amplitude (gender coded +-0.5)

  //population noise
  real<lower=0> sigma_early;
  real<lower=0> sigma_late;
}


transformed parameters {
  // 2. Stan computes the necessary shape (c) dynamically for every HMC step
  real c_dip = gamma_c(mu_tp_dip, sigma_dip);
  
  // If you need b_dip elsewhere in the model, it is simply c_dip + 1
  real b_dip = c_dip + 1; 
}

model {
  //priors:

  //tp dip priors
  mu_tp_dip     ~ normal(9, 3);   //informed by the mean curve

  //dispersion
  sigma_dip         ~ lognormal(2.3, 0.5);

  //beta priors, this is what the predictor is attatched - normalized
  mu_beta_log       ~ normal(0, 10);
  beta_gender_log   ~ normal(0, 5);
  //population noise
  sigma_early   ~ exponential(0.2);
  sigma_late    ~ exponential(0.2);

  // unit-peak gamma: peaks at exactly 1 when x == td
  vector[n_obs] mu = -exp(mu_beta_log + gender * beta_gender_log)[patient] .* gamma_loc_sigma(x, mu_tp_dip, c_dip);

  // likelihood
  // HARDCODED breakpoint: dip sits near t~10, so this splits signal from tail.
  // Promote to a data field before comparing runs across different values.
  real t_split = 30.0;
  vector[n_obs] sig;
  for (i in 1:n_obs) sig[i] = x[i] > t_split ? sigma_late : sigma_early;
  y ~ normal(mu, sig);
}
