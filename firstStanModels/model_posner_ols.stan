data {
  int<lower=0> n_obs;
  int<lower=1> n_pred;
  int<lower=1> n_schools;
  array[n_obs] int<lower=1, upper=n_schools> school;
  matrix[n_obs, n_pred] x;
  vector[n_obs] y;
  vector<lower=0>[n_pred] x_scale;
  real<lower=0> y_scale;
}

parameters {
  vector[n_pred] gamma;
  vector<lower=0>[n_pred] tau;
  cholesky_factor_corr[n_pred] L_Omega;   // Cholesky factor, not full corr_matrix
  matrix[n_pred, n_schools] z;
  real<lower=0> sigma;
}

transformed parameters {
  array[n_schools] vector[n_pred] beta;
  {
    // diag_pre_multiply(tau, L_Omega) is the Cholesky factor of the covariance,
    // computed directly without an expensive decomposition each iteration.
    matrix[n_pred, n_pred] L = diag_pre_multiply(tau, L_Omega);
    for (j in 1:n_schools)
      beta[j] = gamma + L * z[, j];
  }
}

model {
  gamma ~ normal(0, 5 * (y_scale ./ x_scale));
  tau   ~ normal(0, y_scale ./ x_scale);
  L_Omega ~ lkj_corr_cholesky(2);          // prior on the Cholesky factor
  sigma ~ normal(0, y_scale);
  to_vector(z) ~ normal(0, 1);

  {
  vector[n_obs] mu;
  for (n in 1:n_obs)
    mu[n] = x[n] * beta[school[n]];   // just compute the means in the loop
  y ~ normal(mu, sigma);              // ONE vectorized sampling statement
  }
}

generated quantities {
  // Reconstruct the readable correlation matrix for inspection.
  matrix[n_pred, n_pred] Omega = multiply_lower_tri_self_transpose(L_Omega);
}