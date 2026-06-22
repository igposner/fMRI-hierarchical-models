# %% Set up environment
import cmdstanpy
import logging
import matplotlib
import matplotlib.pyplot
import numpy
import subprocess
from lib.plotting import draw_stacked_density_histograms

# Uncomment to install cmdstan.
# Should only need to run installation once.
# cmdstanpy.install_cmdstan()

# Configure stan
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

# Detect number of available CPU cores.
ncpus = int(subprocess.run('nproc', stdout=subprocess.PIPE).stdout)
print(f"{ncpus} CPU cores available")

# %% Simulate schools data.

# Outcome variable:
# SAT score
# Predictor variables:
# GPA
# Age
# Height

# Model:
# SAT ~ 1 + GPA + age + height

# Number of observations, total and per school.
n_obs = 2000
n_schools = 10
n_school_obs = numpy.array([150, 200, 250, 175, 205, 220, 300, 180, 160, 160])
assert n_school_obs.shape[0] == n_schools
assert numpy.sum(n_school_obs) == n_obs

# Simulate a design matrix with columns as per above.
# First column is intercept (ones).
x = numpy.ones([n_obs, 4])
pos = 0
for i, n in enumerate(n_school_obs):
    x[pos:pos+n,1:] = numpy.random.multivariate_normal(
        [2.5 + numpy.random.normal(0, 0.3, 1)[0], 17.5, 163.0],
        [
            [0.4, 0.1, 0],
            [0.1, 0.2, 0],
            [0, 0, 7.5],
        ],
        n,
    )
    pos = pos + n

# Clamp impossible GPAs.
x[numpy.where(x[:,1] < 0),1] = 0
x[numpy.where(x[:,1] > 4),1] = 4.

# Simulate beta values.
b = numpy.random.multivariate_normal(
    [140, 160., 25., 0],
    [
        [10000., 800., -4.5, 0],
        [800., 1600.0, 2.5, 0],
        [-4.5, 2.5, 16.0, 0],
        [0, 0, 0, 0.01],
    ],
    n_schools,
)

# Simulate observed data, y.
y = numpy.zeros(n_obs)
pos = 0
for i, n in enumerate(n_school_obs):
    y[pos:pos+n] = x[pos:pos+n,:] @ b[i,:]
    pos = pos + n
y += numpy.random.normal(0, 10, n_obs)

# Clamp impossible SAT scores.
y[numpy.where(y < 400)] = 400
y[numpy.where(y > 1600)] = 1600

# %% Visualize schools data.

# Histogram of GPA.
fig, ax = matplotlib.pyplot.subplots()
ax.hist(x[:,1], color='black')
ax.set_xlabel('GPA')
ax.set_ylabel('Number of Students')

# SAT as a function of GPA, color-coded by school.
fig, ax = matplotlib.pyplot.subplots()
pos = 0
for i, n in enumerate(n_school_obs):
    ax.scatter(x[pos:pos+n,1], y[pos:pos+n])
    pos = pos + n
ax.set_xlabel('GPA')
ax.set_ylabel('SAT')

# %% Package schools data for stan.

# Center x, convey scales as priors.
x_centered = numpy.ones([n_obs, x.shape[1]])
x_centered[:,1:] = x[:,1:] - numpy.mean(x[:,1:], axis=0)
x_scale = numpy.concatenate([numpy.array([1]), numpy.std(x[:,1:], axis=0)])
school = numpy.repeat(numpy.arange(1, 11), n_school_obs)
stan_data = {
    'n_obs': n_obs,
    'y': y,
    'n_pred': x.shape[1],
    'x': x_centered,
    'x_scale': x_scale,
    'y_scale': numpy.std(y),
    'school': school,
    'n_schools': n_schools,
}

print("mean SAT:", numpy.mean(y))          # gamma[1] should land near this
print("intercept col all ones?", numpy.allclose(x_centered[:,0], 1.0))
print("cols 1-3 centered (means ~0)?", numpy.mean(x_centered[:,1:], axis=0))
print("x_scale:", x_scale)
print("y_scale:", numpy.std(y))

# %% Compile the model.

model_ols = cmdstanpy.CmdStanModel(
    stan_file='model_posner_ols.stan',
    cpp_options={'STAN_THREADS': 'TRUE'}
)

# %% Compile the model 2.

model_ols = cmdstanpy.CmdStanModel(
    stan_file='stan_models/posner_model.stan',
    cpp_options={'STAN_THREADS': 'TRUE'}
)
# %% Fit the model to the data.

fit = model_ols.sample(
    data=stan_data,           # REQUIRED — the data to fit
    parallel_chains=ncpus,    # run chains in parallel across your cores
    iter_warmup=500,
    iter_sampling=500,
    adapt_delta=0.9,
    max_treedepth=12,
    show_progress=True,
)
print(fit.diagnose())

# Uncomment for more detailed diagnostics.
print(fit.summary().to_string())

# %% Inspect the posterior distributions of the betas.
fig, ax = draw_stacked_density_histograms(fit.stan_variable('gamma')[:])

# %% Pair plot to identify divergences.
divergences = fit.method_variables()['divergent__'].flatten(order='F')
fig, ax = matplotlib.pyplot.subplots()
ax.scatter(fit.stan_variable('beta')[numpy.argwhere(divergences==0),0], fit.stan_variable('sigma')[numpy.argwhere(divergences==0)])
ax.scatter(fit.stan_variable('beta')[numpy.argwhere(divergences==1),0], fit.stan_variable('sigma')[numpy.argwhere(divergences==1)], color='red')

# %% Check posterior predictive accuracy.

# Estimated beta hat.
bhat = numpy.mean(fit.stan_variable('beta'), axis=0)
ghat = numpy.mean(fit.stan_variable('gamma'), axis=0)

# The simulated data.
fig, ax = matplotlib.pyplot.subplots()
pos = 0
for i, n in enumerate(n_school_obs):
    ax.scatter(x[pos:pos+n,1], y[pos:pos+n])
    pos = pos + n
ax.set_xlabel('GPA')
ax.set_ylabel('SAT')

# The ground-truth fit.
fig, ax = matplotlib.pyplot.subplots()
pos = 0
for i, n in enumerate(n_school_obs):
    ax.scatter(x[pos:pos+n,1], x[pos:pos+n,:] @ b[i,:])
    pos = pos + n
ax.set_xlabel('GPA')
ax.set_ylabel('SAT')

# The fit from the OLS model. 
fig, ax = matplotlib.pyplot.subplots()
# pos = 0
for i in range(0, n_schools):
    # ax.scatter(x[pos:pos+n,1], x_centered[pos:pos+n,:] @ bhat)
    ax.axline((0, bhat[i, 0]), (4, 4*bhat[i,1]+bhat[i,0]))
    # pos = pos + n
ax.axline((0,ghat[0]), (4, 4*ghat[1]+ghat[0]), color = "black")
ax.set_xlabel('GPA')
ax.set_ylabel('SAT')

# %%
