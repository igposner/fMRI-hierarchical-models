data {
  int<lower=0> n_obs;                     // num students (observations)
  int<lower=1> n_pred;                    // num predictors (incl intercept)
  int<lower=1> n_schools;                 // num schools (groups)

  array[n_obs] int<lower=1, upper=n_schools> school;  // school id for each student
  matrix[n_obs, n_pred] x;                // student predictors (design matrix)
  vector[n_obs] y;                        // SAT outcomes
}

parameters {
  corr_matrix[n_pred] Omega;              // correlation among coefficients
  vector<lower=0>[n_pred] tau;            // sd for coefficients
  array[n_schools] vector[n_pred] beta;   // school specific coefficients
  real<lower=0> sigma;                    // noise sd
  vector[n_pred] gamma;                   // beta mean
}

model {
  tau ~ cauchy(0, 2.5);
  Omega ~ lkj_corr(2);
  gamma ~ normal(0, 5);
  sigma ~ cauchy(0, 2.5);
  for (j in 1:n_schools) {
    beta[j] ~ multi_normal(gamma, quad_form_diag(Omega, tau));
  }

  for (n in 1:n_obs) {
    y[n] ~ normal(x[n] * beta[school[n]], sigma);
  }
}