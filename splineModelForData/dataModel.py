# %%
import numpy as np
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel

# %%
data = np.genfromtxt(
    'main_a1run4.csv',
    delimiter = ',',
    skip_header = 1,
    missing_values = 'NA'
)

data = data[:,1:]

# %% make graph of schwarzlose avg data
yavg = np.nanmean(data, axis=1)

figure, ax = plt.subplots()

t = np.linspace(0, data.shape[0] - 1, data.shape[0])
# 1st column = 1st participant
ax.plot(t, yavg)
ax.set_xlim([t[0], t[-1]])
ax.set_ylim([-3, 2])


# %% Compile and fit the Stan model
model = CmdStanModel(stan_file="dataModel.stan")

mask = ~np.isnan(yavg)
t_clean = t[mask]
yavg_clean = yavg[mask]

stan_data = {
    "x": t_clean,
    "y": yavg_clean,
    "N": yavg_clean.size
}

rng = np.random.default_rng()   # add seed=42 for reproducibility

inits = [{"switchPoint": 40.0 + rng.uniform(-10, 10)} for _ in range(4)]

fit = model.sample(
    data=stan_data,
    chains=4,
    show_console=True,
    adapt_delta=0.95,
    inits=inits
)

print(fit.diagnose())

# %% Extract posterior means
x_knot_mean = fit.stan_variable('x_knot').mean(axis=0)
y_knot_mean = fit.stan_variable('y_knot').mean(axis=0)
switchPoint_mean = fit.stan_variable('switchPoint').mean()
x_plot_mean = fit.stan_variable('x_plot').mean(axis=0)
y_plot_mean = fit.stan_variable('y_plot').mean(axis=0)

# %% Extract posterior means and draws
x_knot_mean = fit.stan_variable('x_knot').mean(axis=0)         # posterior mean of each knot x-position (average across all draws)
y_knot_mean = fit.stan_variable('y_knot').mean(axis=0)         # posterior mean of each knot y-value
switchPoint_mean = fit.stan_variable('switchPoint').mean()     # posterior mean of the switchpoint (where spline ends and baseline begins)
x_plot_mean = fit.stan_variable('x_plot').mean(axis=0)         # x-grid for plotting — same across draws since it's deterministic, but .mean() is harmless
y_plot_mean = fit.stan_variable('y_plot').mean(axis=0)         # posterior mean of the fitted curve at each x_plot point

# full draws for intervals — we need the whole distribution, not just the mean, to compute percentiles
y_plot_draws = fit.stan_variable('y_plot')   # shape: (num_draws, N) — the noise-free mean curve for every posterior draw
y_pred_draws = fit.stan_variable('y_pred')   # shape: (num_draws, N) — the simulated new observations (mean curve + noise) for every draw

# 95% posterior predictive interval — chops off bottom 2.5% and top 2.5% of y_pred draws at each x
# axis=0 means "take the percentile down the rows (across draws), separately at each x"
pred_lo, pred_hi = np.percentile(y_pred_draws, [2.5, 97.5], axis=0)

# 95% credible interval for the mean curve — same operation, but on the noise-free curves
# this band will be narrower since it excludes observation noise
mean_lo, mean_hi = np.percentile(y_plot_draws, [2.5, 97.5], axis=0)

# %% Plot results
plt.figure()   # create a new figure (so we don't overwrite the earlier yavg plot)

# 95% posterior predictive band — drawn FIRST so it sits at the bottom of the z-order
# alpha=0.15 makes it very transparent since it's the widest band and we don't want it to dominate
plt.fill_between(x_plot_mean, pred_lo, pred_hi,
                 color="tab:orange", alpha=0.15,
                 label="95% posterior predictive")

# 95% credible band for the mean curve — drawn ON TOP of the predictive band
# alpha=0.30 makes it slightly more opaque so it's visually distinguishable as a nested, narrower band
plt.fill_between(x_plot_mean, mean_lo, mean_hi,
                 color="tab:orange", alpha=0.30,
                 label="95% mean credible interval")

# raw observed data — plotted as a line since it's a time series
# alpha=0.5 so the bands underneath are still visible through it
plt.plot(t_clean, yavg_clean, label="data", color="tab:blue", alpha=0.5)

# reconstruct a smooth fitted spline for display purposes
from scipy.interpolate import CubicSpline                            # scipy's cubic spline interpolator (separate from Stan's spline — just for drawing a smooth line)
y_knot_on_curve = np.interp(x_knot_mean, x_plot_mean, y_plot_mean)   # find y-value of the fitted curve at each knot x (so the plotted spline passes through the right points)
cs = CubicSpline(x_knot_mean, y_knot_on_curve)                       # build an interpolating spline through (knot_x, knot_y_on_curve)
x_spline = np.linspace(x_knot_mean[0], x_knot_mean[-1], 200)         # dense x-grid from first to last knot for a smooth-looking line
plt.plot(x_spline, cs(x_spline), label="fitted spline",              # draw the smooth spline
         color="tab:orange", linewidth=2)

# horizontal line for the flat/baseline region after the switchpoint
# starts at the switchpoint, extends to the last data point, at the baseline y-value (last knot)
plt.hlines(y_knot_on_curve[-1], switchPoint_mean, t_clean[-1],
           color="tab:orange", linewidth=2, label="baseline")

# green dots marking where the estimated knots are on the fitted curve
# zorder=5 forces them to be drawn on top of everything else so they're always visible
plt.scatter(x_knot_mean, y_knot_on_curve, label="estimated knots",
            color="tab:green", zorder=5)

# vertical dashed line showing where the switchpoint is
plt.axvline(x=switchPoint_mean, color='red', linestyle='--', label="switchpoint")

plt.xlabel("Time")                            # x-axis label
plt.ylabel("Response")                        # y-axis label
plt.title("Spline + Flat Model Fit")          # figure title
plt.legend()                                  # show the legend (auto-populated from the label= arguments above)
plt.show()                                    # render the figure

# %% Knot distributions
x_knot_draws = fit.stan_variable('x_knot')   # shape: (draws, 5)
y_knot_draws = fit.stan_variable('y_knot')   # shape: (draws, 5)

fig, axes = plt.subplots(2, 5, figsize=(15, 5), dpi=300)

for k in range(5):
    # x-position of knot k
    axes[0, k].hist(x_knot_draws[:, k], bins=40, color="tab:blue", alpha=0.7)
    axes[0, k].set_title(f"x_knot[{k+1}]")
    axes[0, k].axvline(x_knot_draws[:, k].mean(), color="red", linestyle="--")

    # y-value of knot k
    axes[1, k].hist(y_knot_draws[:, k], bins=40, color="tab:orange", alpha=0.7)
    axes[1, k].set_title(f"y_knot[{k+1}]")
    axes[1, k].axvline(y_knot_draws[:, k].mean(), color="red", linestyle="--")

plt.tight_layout()
plt.show()

# %% Distances between consecutive knots
x_knot_draws = fit.stan_variable('x_knot')   # shape: (draws, 5)

# compute consecutive differences: shape (draws, 4)
dx_draws = np.diff(x_knot_draws, axis=1)

fig, axes = plt.subplots(1, 4, figsize=(14, 3))

for k in range(4):
    axes[k].hist(dx_draws[:, k], bins=40, color="tab:purple", alpha=0.7)
    axes[k].set_title(f"knot {k+2} − knot {k+1}")
    axes[k].axvline(dx_draws[:, k].mean(), color="red", linestyle="--",
                    label=f"mean = {dx_draws[:, k].mean():.2f}")
    axes[k].legend(fontsize=8)
    axes[k].set_xlabel("distance")

plt.tight_layout()
plt.show()

# print summary
print("Distances between consecutive knots (median [2.5%, 97.5%]):")
for k in range(4):
    lo, med, hi = np.percentile(dx_draws[:, k], [2.5, 50, 97.5])
    print(f"  knot {k+2} − knot {k+1}:  {med:.2f}  [{lo:.2f}, {hi:.2f}]")
    
# %% 
import glob
logs = glob.glob("/tmp/tmp9hy11q_8/dataModelhwr1kgus/*stdout.txt")
for f in logs:
    print(open(f).read())

