# %%
import numpy as np
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel

# %%
data = np.genfromtxt(
    'data/main_a1run4.csv',
    delimiter = ',',
    skip_header = 1,
    missing_values = 'NA'
)

data = data[:,1:]

# %% get rid of patients that have at least 50% empty data
keep = np.mean(~np.isnan(data), axis=0) >= 0.5
data = data[:, keep]
original_indices = np.where(keep)[0]
print(f"Kept {keep.sum()} of {len(keep)} patients")

# %% subset to 10 patients for fast iteration
# data = data[:, :10]
# print(f"Using {data.shape[1]} patients for debugging")

# %% mean patient data
mean_signal = np.nanmean(data, axis=1)
sem_signal = np.nanstd(data, axis=1) / np.sqrt(np.sum(~np.isnan(data), axis=1))
t = np.linspace(0, data.shape[0] - 1, data.shape[0])

figure, ax = plt.subplots(figsize=(10, 4))
ax.plot(t, mean_signal, color='steelblue', linewidth=1.5, label='Mean BOLD')
ax.fill_between(t, mean_signal - sem_signal, mean_signal + sem_signal,
                alpha=0.3, color='steelblue', label='±1 SEM')
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlim([t[0], t[-1]])
ax.set_xlabel('Time (TRs)')
ax.set_ylabel('BOLD')
ax.set_title(f'Mean signal across {data.shape[1]} patients')
ax.legend()
plt.tight_layout()
plt.show()

# %% patient 2 graph
p1 = data[:, 1]  #just chage the 0 to a 1

figure, ax = plt.subplots()
ax.plot(t, p1)
ax.set_xlim([t[0], t[-1]])
ax.set_ylim([-3, 2])
ax.set_xlabel('time')
ax.set_ylabel('BOLD')
ax.set_title('Patient 2')
plt.show() # note that it is completely empty because there is no data for this patient

# %% Compile the Stan model
model = CmdStanModel(stan_file="doG.stan")

# %% Build stan_data
n_timepoints, n_patients = data.shape

y_long = data.flatten(order='F')
x_long = np.tile(t, n_patients)
patient_long = np.repeat(np.arange(n_patients) + 1, n_timepoints)

mask = ~np.isnan(y_long)

stan_data = {
    "n_obs":    int(mask.sum()),
    "n_ppl":    n_patients,
    "y":        y_long[mask],
    "x":        x_long[mask],
    "patient":  patient_long[mask].astype(int),
}

# %% Fit
rng = np.random.default_rng()

fit = model.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_sampling=500,
    iter_warmup=500,
    show_console=True,
    adapt_delta=0.9,
)

print(fit.diagnose())

# %% Pull out posterior draws
import numpy as np
import matplotlib.pyplot as plt

# Shape: (n_draws, n_ppl, n_grid)
y_plot_samples = fit.stan_variable("y_plot")
x_plot = stan_data["x_plot"]   # x_plot is the same across draws, grab the first

# Posterior summaries per patient
y_plot_mean = y_plot_samples.mean(axis=0)                       # (n_ppl, n_grid)
y_plot_lo   = np.quantile(y_plot_samples, 0.025, axis=0)        # 2.5% credible band
y_plot_hi   = np.quantile(y_plot_samples, 0.975, axis=0)        # 97.5%

# %% Plot 4 patients
patients_to_show = [0, 2, 5, 8]   # python 0-indexed; stan patient = python idx + 1

fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi = 300)

for ax, p_idx in zip(axes.flat, patients_to_show):
    stan_id = p_idx + 1   # because patient index in stan_data is 1-indexed

    # raw observations for this patient
    obs_mask = stan_data["patient"] == stan_id
    x_obs = stan_data["x"][obs_mask]
    y_obs = stan_data["y"][obs_mask]

    # fitted curve
    ax.plot(x_plot, y_plot_mean[p_idx], color="C0", label="posterior mean")
    ax.fill_between(x_plot, y_plot_lo[p_idx], y_plot_hi[p_idx],
                    color="C0", alpha=0.3, label="95% CI")

    # raw data
    ax.scatter(x_obs, y_obs, s=8, color="k", alpha=0.5, label="data")

    # switchpoint
    ax.axvline(60, color="red", ls="--", alpha=0.7, label="cutoff = 60")

    ax.set_title(f"Patient {stan_id}")
    ax.set_xlabel("time")
    ax.set_ylabel("BOLD")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# y_plot_mu has shape (n_draws, n_grid) — one population-mean curve per posterior draw
y_plot_mu = fit.stan_variable("y_plot_mu")   # CmdStanPy: pulls the GQ variable across all draws

x_plot = np.asarray(stan_data["x_plot"])     # the grid the curve was evaluated on

# summarize across draws
mean_curve = y_plot_mu.mean(axis=0)
lo, hi = np.percentile(y_plot_mu, [2.5, 97.5], axis=0)   # 95% credible band on the mean curve

plt.figure(figsize=(9, 4))
plt.plot(x_plot, mean_curve, color="C0", lw=2, label="population posterior mean")
plt.fill_between(x_plot, lo, hi, color="C0", alpha=0.25, label="95% CI")
plt.axhline(0, color="k", ls="--", lw=0.8)
plt.axvline(stan_data["x_knot"][-1], color="r", ls="--", lw=1, label="cutoff = 60")
plt.xlabel("time")
plt.ylabel("BOLD")
plt.title("Population mean curve (y_plot_mu)")
plt.legend()
plt.tight_layout()
plt.show()
# %%
