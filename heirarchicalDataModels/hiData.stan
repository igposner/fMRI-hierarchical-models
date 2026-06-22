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

    // Unknown ordering: [a_1..a_n, b_1..b_n, c_1..c_n, d_1..d_n]

    // 1) Left endpoint interpolation: a_j = y_j
    for (j in 1:n) {
      A[j, j] = 1.0;
      rhs[j] = y_knot[j];
    }

    // 2) Right endpoint interpolation:
    // a_j + b_j*h + c_j*h^2 + d_j*h^3 = y_{j+1}
    for (j in 1:n) {
      int row = n + j;
      A[row, j]       = 1.0;
      A[row, n + j]   = h[j];
      A[row, 2*n + j] = h[j]^2;
      A[row, 3*n + j] = h[j]^3;
      rhs[row] = y_knot[j+1];
    }

    // 3) 1st-derivative continuity at interior knots:
    // b_j - b_{j+1} + 2*c_j*h_j + 3*d_j*h_j^2 = 0
    for (j in 1:(n-1)) {
      int row = 2*n + j;
      A[row, n + j]       =  1.0;
      A[row, n + (j+1)]   = -1.0;
      A[row, 2*n + j]     =  2.0 * h[j];
      A[row, 3*n + j]     =  3.0 * h[j]^2;
      rhs[row] = 0.0;
    }

    // 4) 2nd-derivative continuity (divided by 2):
    // c_j - c_{j+1} + 3*d_j*h_j = 0
    for (j in 1:(n-1)) {
      int row = (3*n - 1) + j; // rows 3n .. 4n-2
      A[row, 2*n + j]       =  1.0;
      A[row, 2*n + (j+1)]   = -1.0;
      A[row, 3*n + j]       =  3.0 * h[j];
      rhs[row] = 0.0;
    }

    // 5) Not-a-knot constraints (same spirit as your Python: equalize end cubic pieces)
    // d_1 = d_2 and d_{n-1} = d_n
    {
      int row1 = M - 1;
      int row2 = M;

      if (n < 3) reject("Need at least 4 knots (K>=4) for both not-a-knot end constraints.");

      // d_1 - d_2 = 0
      A[row1, 3*n + 1] =  1.0;
      A[row1, 3*n + 2] = -1.0;
      rhs[row1] = 0.0;

      // d_{n-1} - d_n = 0
      A[row2, 3*n + (n-1)] =  1.0;
      A[row2, 3*n + n]     = -1.0;
      rhs[row2] = 0.0;
    }

    z = mdivide_left(A, rhs);

    for (j in 1:n) {
      coeffs[1, j] = z[j];         // a_j
      coeffs[2, j] = z[n + j];     // b_j
      coeffs[3, j] = z[2*n + j];   // c_j
      coeffs[4, j] = z[3*n + j];   // d_j
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

  vector spline_evalVector(matrix coeffs, vector x_knot, vector x_new) {
    int N = rows(x_new);
    int K = rows(x_knot);
    int n = K - 1;
    vector[N] y;

    if (cols(coeffs) != n) reject("coeffs has wrong number of segments.");

    for (i in 1:N) {
      int seg = spline_segment(x_knot, x_new[i]);
      real dx = x_new[i] - x_knot[seg];
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
  int<lower=0> n_obs;                     // num obs
  int<lower=1> n_ppl;                 // num patients

  array[n_obs] int<lower=1, upper=n_ppl> patient;  // patient id for each
  vector[n_obs] y;                    // fmri outcomes
  vector[n_obs] x;                    // t value
  int<lower=1> n_grid;
}

parameters {
  vector<lower=20, upper=80>[n_ppl] switchPoint; // switchpoints for each person
  real<lower=0.01> sigma; //each has similar noise
  array[n_ppl] vector[5] y_knot; //each patient gets its own spline points
  array[n_ppl] simplex[4] x_props; //each pateint gets own simplex
}

transformed parameters {
  array[n_ppl] vector[5] x_knot;   // knot locations per patient
  array[n_ppl] vector[4] x_diff;   // gaps between knots per patient

  for (i in 1:n_ppl) {
    x_diff[i] = x_props[i] * (switchPoint[i] - x[1]);

    x_knot[i][1] = x[1];
    x_knot[i][5] = switchPoint[i];
    for (k in 2:4) {
      x_knot[i][k] = x_knot[i][k-1] + x_diff[i][k-1];
    }
  }
}

model {
  // ---- Priors ----
  switchPoint ~ normal(40, 10);   // vectorized across all patients
  sigma ~ exponential(2);

  for (i in 1:n_ppl) {
    x_props[i] ~ dirichlet(rep_vector(5, 4));

    y_knot[i][1]   ~ normal(1.0, 0.5);
    y_knot[i][2:4] ~ normal(-0.5, 2);
    y_knot[i][5]   ~ normal(-0.3, 0.3);
  }

  // ---- Likelihood ----
  // Pre-compute spline coefficients once per patient
  array[n_ppl] matrix[4, 4] coeffs;
  for (i in 1:n_ppl) {
    coeffs[i] = not_a_knot_coeffs(x_knot[i], y_knot[i]);
  }

  for (j in 1:n_obs) {
    int i = patient[j];
    if (x[j] < switchPoint[i]) {
      y[j] ~ normal(spline_eval(coeffs[i], x_knot[i], x[j])[1], sigma);
    } else {
      y[j] ~ normal(y_knot[i][5], sigma);
    }
  }
}

generated quantities {
  // Per-patient spline coefficients for this draw
  array[n_ppl] matrix[4, 4] coeffs_gq;
  for (i in 1:n_ppl) {
    coeffs_gq[i] = not_a_knot_coeffs(x_knot[i], y_knot[i]);
  }

  // Plotting grid: same x-grid for every patient, easier to compare curves later
  vector[n_grid] x_plot;
  real x_lo = min(x);                     // earliest observed x across all patients
  real x_hi = max(x);                     // latest observed x across all patients
  for (g in 1:n_grid) {
    real frac = (g - 1.0) / (n_grid - 1.0);
    x_plot[g] = x_lo + frac * (x_hi - x_lo);
  }

  // y_plot[i, g] = patient i's mean curve at grid point g (noise-free)
  // y_pred[i, g] = patient i's posterior predictive draw at grid point g (with noise)
  array[n_ppl] vector[n_grid] y_plot;
  array[n_ppl] vector[n_grid] y_pred;

  for (i in 1:n_ppl) {
    for (g in 1:n_grid) {
      real mu_ig;
      if (x_plot[g] < switchPoint[i]) {
        mu_ig = spline_eval(coeffs_gq[i], x_knot[i], x_plot[g])[1];
      } else {
        mu_ig = 0;
      }
      y_plot[i][g] = mu_ig;
      y_pred[i][g] = normal_rng(mu_ig, sigma);
    }
  }
}