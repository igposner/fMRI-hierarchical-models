functions {
  // Returns coeffs[4, K-1] where segment j uses:
  // y(x) = a_j + b_j*dx + c_j*dx^2 + d_j*dx^3,  dx = x - x_knot[j]
  matrix not_a_knot_coeffs(vector x_knot, vector y_knot) {
    int K = rows(x_knot);
    int n = K - 1;
    int M = 4 * n;

    vector[n] h;
    matrix[M, M] A = rep_matrix(0.0, M, M);
    vector[M] rhs = rep_vector(0.0, M);
    vector[M] z;
    matrix[4, n] coeffs;

    if (K < 3) reject("Need at least 3 knots.");
    for (k in 2:K)
      if (x_knot[k] <= x_knot[k-1]) reject("x_knot must be strictly increasing.");

    for (j in 1:n) h[j] = x_knot[j+1] - x_knot[j];

    for (j in 1:n) {
      A[j, j] = 1.0;
      rhs[j] = y_knot[j];
    }

    for (j in 1:n) {
      int row = n + j;
      A[row, j]       = 1.0;
      A[row, n + j]   = h[j];
      A[row, 2*n + j] = h[j]^2;
      A[row, 3*n + j] = h[j]^3;
      rhs[row] = y_knot[j+1];
    }

    for (j in 1:(n-1)) {
      int row = 2*n + j;
      A[row, n + j]       =  1.0;
      A[row, n + (j+1)]   = -1.0;
      A[row, 2*n + j]     =  2.0 * h[j];
      A[row, 3*n + j]     =  3.0 * h[j]^2;
      rhs[row] = 0.0;
    }

    for (j in 1:(n-1)) {
      int row = (3*n - 1) + j;
      A[row, 2*n + j]       =  1.0;
      A[row, 2*n + (j+1)]   = -1.0;
      A[row, 3*n + j]       =  3.0 * h[j];
      rhs[row] = 0.0;
    }

    {
      int row1 = M - 1;
      int row2 = M;

      if (n < 3) reject("Need at least 4 knots (K>=4) for both not-a-knot end constraints.");

      A[row1, 3*n + 1] =  1.0;
      A[row1, 3*n + 2] = -1.0;
      rhs[row1] = 0.0;

      A[row2, 3*n + (n-1)] =  1.0;
      A[row2, 3*n + n]     = -1.0;
      rhs[row2] = 0.0;
    }

    z = mdivide_left(A, rhs);

    for (j in 1:n) {
      coeffs[1, j] = z[j];
      coeffs[2, j] = z[n + j];
      coeffs[3, j] = z[2*n + j];
      coeffs[4, j] = z[3*n + j];
    }
    return coeffs;
  }

  int spline_segment(vector x_knot, real x) {
    int K = rows(x_knot);
    int n = K - 1;

    if (x < x_knot[1] || x > x_knot[K]) reject("x out of range.");
    if (x == x_knot[1]) return 1;
    if (x == x_knot[K]) return n;

    for (j in 2:K)
      if (x <= x_knot[j]) return j - 1;

    return n;
  }

  vector spline_eval(matrix coeffs, vector x_knot, real x_new) {
    int N = 1;
    int K = rows(x_knot);
    int n = K - 1;
    vector[N] y;

    if (cols(coeffs) != n) reject("coeffs has wrong number of segments.");

    for (i in 1:N) {
      int seg = spline_segment(x_knot, x_new);
      real dx = x_new - x_knot[seg];
      y[i] =
        coeffs[1, seg]
        + coeffs[2, seg] * dx
        + coeffs[3, seg] * dx^2
        + coeffs[4, seg] * dx^3;
    }
    return y;
  }
}

data {
  int<lower=0> n_obs;
  int<lower=1> n_ppl;
  array[n_obs] int<lower=1, upper=n_ppl> patient;
  vector[n_obs] y;
  vector[n_obs] x;
  int<lower=1> n_grid;
  vector[n_grid] x_plot;

  int<lower=4> n_knots;
  vector[n_knots] x_knot;          // FIXED, shared — last entry is the cutoff (~60)
}

transformed data {
  array[n_obs] int seg_obs;
  vector[n_obs] dx_obs;
  array[n_obs] int<lower=0, upper=1> is_active;

  for (j in 1:n_obs) {
    if (x[j] < x_knot[n_knots]) {
      is_active[j] = 1;
      seg_obs[j]   = spline_segment(x_knot, x[j]);
      dx_obs[j]    = x[j] - x_knot[seg_obs[j]];
    } else {
      is_active[j] = 0;
      seg_obs[j]   = 1;
      dx_obs[j]    = 0;
    }
  }
}

parameters {
  vector[n_knots] mu_yknot;
  vector<lower=0>[n_knots] sigma_yknot;
  array[n_ppl] vector[n_knots] y_knot;
  vector<lower=0.01>[n_ppl] sigma;
}

transformed parameters {
  array[n_ppl] matrix[4, n_knots - 1] coeffs;
  for (i in 1:n_ppl)
    coeffs[i] = not_a_knot_coeffs(x_knot, y_knot[i]);   // x_knot is constant — basis never reshapes
}

model {
  mu_yknot[1] ~ normal( 1.0, 0.5);   // onset, ~+1
  mu_yknot[2] ~ normal(-2.0, 0.7);   // the dip — this is the big one
  mu_yknot[3] ~ normal( 0.2, 0.7);   // crossing zero, rising
  mu_yknot[4] ~ normal( 0.8, 0.7);   // overshoot shoulder
  mu_yknot[5] ~ normal( 0.0, 0.5);   // settled near zero   // last knot near zero (dip has decayed)
  sigma_yknot               ~ normal(0, 0.5);

  for (i in 1:n_ppl)
    y_knot[i] ~ normal(mu_yknot, sigma_yknot);

  sigma ~ normal(2.5, 2.5);

  vector[n_obs] mu_obs;

  for (j in 1:n_obs) {
    int i = patient[j];
    if (is_active[j]) {
      int s   = seg_obs[j];
      real dx = dx_obs[j];
      mu_obs[j] = coeffs[i][1, s]
                + coeffs[i][2, s] * dx
                + coeffs[i][3, s] * dx^2
                + coeffs[i][4, s] * dx^3;
    } else {
      mu_obs[j] = 0;
    }
  }

  y ~ normal(mu_obs, sigma[patient]);
}

generated quantities {
  array[n_ppl] vector[n_grid] y_plot;
  for (i in 1:n_ppl)
    for (g in 1:n_grid)
      y_plot[i][g] = (x_plot[g] < x_knot[n_knots])
                     ? spline_eval(coeffs[i], x_knot, x_plot[g])[1]
                     : 0;

  vector[n_grid] y_plot_mu;
  {
    matrix[4, n_knots - 1] coeffs_mu = not_a_knot_coeffs(x_knot, mu_yknot);
    for (g in 1:n_grid)
      y_plot_mu[g] = (x_plot[g] < x_knot[n_knots])
                     ? spline_eval(coeffs_mu, x_knot, x_plot[g])[1]
                     : 0;
  }
}