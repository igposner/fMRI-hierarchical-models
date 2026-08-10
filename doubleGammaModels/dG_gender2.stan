
//model tests the impact that gender has on beta (the amplitude)
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

  //timing of the bump 
  real<lower=0> tp_gap_pop; //modeling with the mean time gap for the whole population parameterized so that it is ordered
  real<lower=0> sigma_tp_bump;
  vector[n_ppl] tp_bump_raw;
  

  //width of dip and bump (dispersion)
  real<lower=0.01> b_dip;    // shared across patients
  real<lower=0.01> b_bump;   // shared across patients

  //scale of bump, using log space so that I can keep it normalized and keep c positive
  real mu_log_c;
  real<lower=0> sigma_log_c;
  vector[n_ppl] log_c_raw;

  //amplitudes
  real<lower=0> mu_beta; //mean amplitude
  real beta_gender;     // boy-girl difference in amplitude (gender coded +-0.5)
  real<lower=0> sigma_beta; //sd in amplitude
  vector[n_ppl] beta_raw; //normalized distance each patient is from  mu_beta

  //population noise
  real<lower=1e-6> sigma_early;
  real<lower=1e-6> sigma_late;

}

transformed parameters {
  //computed time points
  real mu_tp_bump = tp_gap_pop + mu_tp_dip; //this makes it so that the time points are ordered. A prior is placed later on tp_gap so the ordering is strong
  vector[n_ppl] tp_bump = mu_tp_bump + sigma_tp_bump * tp_bump_raw; //bump

  //computed c (scale of gammas)
  vector[n_ppl] c = exp(mu_log_c + sigma_log_c * log_c_raw);

  //computed amplitude beta
  vector[n_ppl] beta = (mu_beta + beta_gender * gender) + sigma_beta * beta_raw;
}

model {
  //priors:

  //tp dip priors - normalized
  mu_tp_dip     ~ normal(9, 1); //informed by the mean curve
  sigma_tp_dip  ~ normal(0, 1);
  tp_dip        ~ normal(mu_tp_dip, sigma_tp_dip);

  //tp bump priors - normalized
  tp_gap_pop    ~ normal(35, 1); //informed by the mean curve
  sigma_tp_bump ~ normal(0, 1);
  tp_bump_raw   ~ std_normal();

  //just priors on the dispersion (right now in log beacuse of how I used to have in log space)
  b_dip         ~ lognormal(-1.5, 0.5);
  b_bump        ~ lognormal(-1.5, 0.5);

  //c priors, still in log space so I can have lower = 0 and have normalized
  mu_log_c      ~ normal(log(1.0/6.0), 0.2);
  sigma_log_c   ~ normal(0, 0.3);
  log_c_raw     ~ std_normal();

  //beta priors, this is what the predictor is attatched - normalized
  mu_beta       ~ normal(0, 3); //
  beta_gender   ~ normal(0, 3);
  sigma_beta    ~ normal(0, 5);
  beta_raw      ~ std_normal();

  //population noise
  sigma_early   ~ normal(0, 3);
  sigma_late    ~ normal(0, 8);

  //making a vector of length n_obs that has the tps
  vector[n_obs] td = tp_dip[patient];
  vector[n_obs] tb = tp_bump[patient];

  //making the gamma first draft - used AI for uncommented second draft 
  //vector[n_obs] g_dip  = pow(x ./ td, td * b_dip)  .* exp(-b_dip  * (x - td));
  //vector[n_obs] g_bump = pow(x ./ tb, tb * b_bump) .* exp(-b_bump * (x - tb));
  //vector[n_obs] mu = beta[patient] .* (g_bump - c[patient] .* g_dip);

  vector[n_obs] lg_dip  = (td * b_dip)  .* (log(x) - log(td)) - b_dip  * (x - td);
  vector[n_obs] lg_bump = (tb * b_bump) .* (log(x) - log(tb)) - b_bump * (x - tb);
  vector[n_obs] mu = beta[patient] .* (c[patient] .* exp(lg_bump) - exp(lg_dip));

  //likelyhood
  // HARDCODED breakpoint: sits past the bump (mu_tp_bump ~ 50).
  real t_split = 50.0;
  vector[n_obs] sig;
  for (i in 1:n_obs) sig[i] = x[i] > t_split ? sigma_late : sigma_early;
  y ~ normal(mu, sig);
}

generated quantities {
  vector[n_obs] td_debug = tp_dip[patient];
  vector[n_obs] tb_debug = tp_bump[patient];

  vector[n_obs] lg_dip_debug  = (td_debug * b_dip)  .* (log(x) - log(td_debug)) - b_dip  * (x - td_debug);
  vector[n_obs] lg_bump_debug = (tb_debug * b_bump) .* (log(x) - log(tb_debug)) - b_bump * (x - tb_debug);
  vector[n_obs] mu_debug = beta[patient] .* (c[patient] .* exp(lg_bump_debug) - exp(lg_dip_debug));
}