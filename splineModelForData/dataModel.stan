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
    int<lower=2> N; //number of observations
    vector[N] x; //x values of the observations
    vector[N] y; //y values of the observations
}

parameters {
  real<lower=20, upper=80> switchPoint;
  real<lower=0.01> sigma;
  simplex[4] x_props;
  vector[5] y_knot;
}

transformed parameters {
  vector[5] x_knot; //this is the location of the knots
  vector[4] x_diff; //this is the difference between
  
  x_diff = x_props[1:4] * (switchPoint - x[1]); 
  
  x_knot[1] = x[1];
  x_knot[5] = switchPoint;
  for (k in 2:4) {
    x_knot[k] = x_knot[k-1] + x_diff[k-1];
  }
}

model {
  matrix[4, 4] coeffs = not_a_knot_coeffs(x_knot, y_knot);

  switchPoint ~ normal(40, 10);
  sigma ~ exponential(2);
  x_props ~ dirichlet(rep_vector(5, 4));

  y_knot[1] ~ normal(1.0, 0.5);
  y_knot[2:4] ~ normal(-0.5, 2);
  y_knot[5] ~ normal(-0.3, 0.3);

  for (i in 1:N) {
    if (x[i] < switchPoint) {
      y[i] ~ normal(spline_eval(coeffs, x_knot, x[i])[1], sigma);
    } else {
      y[i] ~ normal(y_knot[5], sigma);
    }
  }
}

generated quantities {
  matrix[4, 4] coeffs_gq = not_a_knot_coeffs(x_knot, y_knot);  // recompute spline coefficients from this draw's knots, so we can evaluate the fitted curve

  vector[N] x_plot;  // x-values we'll evaluate the fitted curve at (same grid as data, for easy plotting)
  vector[N] y_plot;  // mean curve f(x) at each x_plot — no observation noise, just the underlying function
  vector[N] y_pred;  // posterior predictive draws — f(x) plus simulated noise, representing possible new observations

  // build an evenly-spaced x grid spanning the full data range
  for (i in 1:N) {
    real frac = (i - 1.0) / (N - 1.0);       // fraction from 0 to 1 as i goes from 1 to N
    x_plot[i] = x[1] + frac * (x[N] - x[1]); // linearly interpolate between first and last data x
  }

  // evaluate the mean curve and draw a predictive sample at each x_plot point
  for (i in 1:N) {
    real mu_i;                                          // mean of y at this x, under the current posterior draw
    if (x_plot[i] < switchPoint) {                      // before the switchpoint: we're in the spline region
      mu_i = spline_eval(coeffs_gq, x_knot, x_plot[i])[1];  // evaluate the cubic spline at this x; [1] pulls the scalar out of the length-1 vector
    } else {                                            // after the switchpoint: we're in the flat/baseline region
      mu_i = y_knot[5];                                 // baseline value is just the 5th y_knot (the post-switchpoint level)
    }
    y_plot[i] = mu_i;                                   // store the noise-free mean curve value
    y_pred[i] = normal_rng(mu_i, sigma);                // draw one simulated observation: mean mu_i, std dev sigma — this is what makes it a "predictive" draw
  }
}
