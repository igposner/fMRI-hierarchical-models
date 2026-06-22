# %%
import numpy as np
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel

# %% 
# ---- 1) Your data / knots (input here, like you wanted) ----
x_knot = np.array([0., 8., 30., 75., 100.1], dtype=float)

# IMPORTANT: y_knot must match length of x_knot
# (Your earlier snippet had 6 y values; that won't work.)
y_knot = np.array([1., -1.8, 0., 0., -0.4], dtype=float)

# sort knots (Stan requires increasing)
order = np.argsort(x_knot)
x_knot = x_knot[order]
y_knot = y_knot[order]

# ---- 2) Compile Stan model ----
model = CmdStanModel(stan_file="splineCode.stan")

# ---- 3) Run Stan in fixed_param mode (no sampling; just compute outputs once) ----
stan_data = {
    "K": len(x_knot),
    "x_knot": x_knot,
    "y_knot": y_knot,
    "N_plot": 200,
}

fit = model.sample(
    data=stan_data,
    chains=1,
    iter_sampling=1,
    iter_warmup=0,
    fixed_param=True,
    adapt_engaged=False

)

# ---- 4) Extract results ----
coeffs = fit.stan_variable("coeffs")[0]   # shape (4, K-1)
x_plot = fit.stan_variable("x_plot")[0]
y_plot = fit.stan_variable("y_plot")[0]

print("coeffs (rows: a,b,c,d; cols: segments):")
print(coeffs)

# ---- 5) Plot knots + spline ----
plt.figure()
plt.scatter(x_knot, y_knot, label="knots")
plt.plot(x_plot, y_plot, label="not-a-knot spline")
plt.legend()
plt.show()



# %% Make a noisy version of the spline for testing
y_noisy = y_plot + np.random.normal(0, 1, y_plot.size)

plt.figure()
plt.scatter(x_knot, y_knot, label="knots")
plt.plot(x_plot, y_plot, label="not-a-knot spline")
plt.plot(x_plot, y_noisy, label="noise", color = "tab:red")
plt.legend()
plt.show()

# %%
model = CmdStanModel(stan_file="splineCode2.stan")

# ---- 3) Run Stan in fixed_param mode (no sampling; just compute outputs once) ----
stan_data = {
    "K": len(x_knot),
    "x": x_plot,
    "y": y_noisy,
    "N_plot": 200,
    "N" : x_plot.size
}

fit = model.sample(
    data=stan_data,
    show_console=True,
    adapt_delta=0.95
)
print(fit.diagnose())
# %% Pair plots with divergent transitions

# Which samples were divergent transitions?
divergences = fit.method_variables()['divergent__'].flatten(order='F')

# Plot a pair of parameters
import matplotlib
import numpy
fig, ax = matplotlib.pyplot.subplots()
ax.scatter(fit.stan_variable('x_knot')[numpy.argwhere(divergences==0), 2], fit.stan_variable('y_knot')[numpy.argwhere(divergences==0), 2])
ax.scatter(fit.stan_variable('x_knot')[numpy.argwhere(divergences==1), 2], fit.stan_variable('y_knot')[numpy.argwhere(divergences==1), 2], color='red')

# %% AI generated code to plot means for the estimated knots. then plot the spline that runs through the means
# Compute mean knot positions
x_knot_mean = fit.stan_variable('x_knot').mean(axis=0)
y_knot_mean = fit.stan_variable('y_knot').mean(axis=0)

# Use splineCode.stan to evaluate spline at mean knots
model_plot = CmdStanModel(stan_file="splineCode.stan")

stan_data_mean = {
    "K": len(x_knot_mean),
    "x_knot": x_knot_mean,
    "y_knot": y_knot_mean,
    "N_plot": 200,
}

fit_mean = model_plot.sample(
    data=stan_data_mean,
    chains=1,
    iter_sampling=1,
    iter_warmup=0,
    fixed_param=True,
    adapt_engaged=False
)

x_plot_mean = fit_mean.stan_variable("x_plot")[0]
y_plot_mean = fit_mean.stan_variable("y_plot")[0]

# Plot everything
plt.figure()
plt.plot(x_plot, y_plot, label="true spline", color="tab:blue")
plt.plot(x_plot, y_noisy, label="noisy data", color="tab:red", alpha=0.5)
plt.plot(x_plot_mean, y_plot_mean, label="estimated spline", color="tab:green")
plt.scatter(x_knot_mean, y_knot_mean, label="estimated knots", color="tab:green", zorder=5)
plt.scatter(x_knot, y_knot, label="true knots", color="tab:orange", zorder=5)
plt.legend()
plt.show()
# %%
