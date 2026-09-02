# %%
import numpy as np
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel

# %%
data = np.genfromtxt(
    '../data/main_a1run4.csv',
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
data = data[:, :10]
print(f"Using {data.shape[1]} patients for debugging")

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
model = CmdStanModel(stan_file="doG.stan", cpp_options = {'STAN_THREADS': 'TRUE'})

# %% Build stan_data
n_timepoints, n_patients = data.shape

y_long = data.flatten(order='F')
x_long = np.tile(t, n_patients)
patient_long = np.repeat(np.arange(n_patients) + 1, n_timepoints)

mask = ~np.isnan(y_long) & (x_long > 0)

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
    iter_sampling=1000,
    iter_warmup=1000,
    show_console=True,
    adapt_delta=0.8,
)

print(fit.diagnose())

# %% Pull out posterior draws and reconstruct fitted curves in Python
import numpy as np
import matplotlib.pyplot as plt

a_bump = fit.stan_variable("a_bump")   # (n_draws, n_ppl) now
b_bump = fit.stan_variable("b_bump")
a_dip  = fit.stan_variable("a_dip")
b_dip  = fit.stan_variable("b_dip")
c      = fit.stan_variable("c")        # (n_draws, n_ppl)
beta   = fit.stan_variable("beta")     # (n_draws, n_ppl)

n_draws, n_ppl = beta.shape
x_plot = np.linspace(t[0], t[-1], 200)

from scipy.special import gammaln
def double_gamma(tt, a_bump, b_bump, a_dip, b_dip, c):
    # analytic unit-peak normalization, matches doG.stan
    tt = np.asarray(tt, float)
    h = np.zeros_like(tt)
    pos = tt > 0
    logt = np.log(tt[pos])
    lpk_b = (a_bump-1)*np.log((a_bump-1)/b_bump) - (a_bump-1) + a_bump*np.log(b_bump) - gammaln(a_bump)
    lpk_d = (a_dip -1)*np.log((a_dip -1)/b_dip ) - (a_dip -1) + a_dip *np.log(b_dip ) - gammaln(a_dip)
    g_b = np.exp((a_bump-1)*logt - b_bump*tt[pos] + a_bump*np.log(b_bump) - gammaln(a_bump) - lpk_b)
    g_d = np.exp((a_dip -1)*logt - b_dip *tt[pos] + a_dip *np.log(b_dip ) - gammaln(a_dip ) - lpk_d)
    h[pos] = g_b - c*g_d
    return h

# Per-patient curves: shape now differs by patient. (n_draws, n_ppl, n_grid)
y_plot_samples = np.empty((n_draws, n_ppl, x_plot.size))
for d in range(n_draws):
    for p in range(n_ppl):
        h = double_gamma(x_plot, a_bump[d,p], b_bump[d], a_dip[d,p], b_dip[d], c[d,p])
        y_plot_samples[d, p] = beta[d, p] * h

y_plot_mean = y_plot_samples.mean(axis=0)
y_plot_lo   = np.quantile(y_plot_samples, 0.025, axis=0)
y_plot_hi   = np.quantile(y_plot_samples, 0.975, axis=0)

# %% Plot 4 patients
patients_to_show = [0, 2, 5, 8]   # python 0-indexed; stan patient = python idx + 1

fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)

for ax, p_idx in zip(axes.flat, patients_to_show):
    stan_id = p_idx + 1

    obs_mask = stan_data["patient"] == stan_id
    x_obs = stan_data["x"][obs_mask]
    y_obs = stan_data["y"][obs_mask]

    ax.plot(x_plot, y_plot_mean[p_idx], color="C0", label="posterior mean")
    ax.fill_between(x_plot, y_plot_lo[p_idx], y_plot_hi[p_idx],
                    color="C0", alpha=0.3, label="95% CI")
    ax.scatter(x_obs, y_obs, s=8, color="k", alpha=0.5, label="data")

    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_title(f"Patient {stan_id}")
    ax.set_xlabel("time")
    ax.set_ylabel("BOLD")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

# %% Population mean curve: average of per-patient fitted curves
pop_mean = y_plot_mean.mean(axis=0)                      # mean across patients
pop_lo   = np.quantile(y_plot_samples.mean(axis=1), 0.025, axis=0)
pop_hi   = np.quantile(y_plot_samples.mean(axis=1), 0.975, axis=0)

plt.figure(figsize=(9, 4))
plt.plot(x_plot, pop_mean, color="C0", lw=2, label="population mean (avg of patients)")
plt.fill_between(x_plot, pop_lo, pop_hi, color="C0", alpha=0.25, label="95% CI")
plt.axhline(0, color="k", ls="--", lw=0.8)
plt.xlabel("time")
plt.ylabel("BOLD")
plt.title("Population mean curve")
plt.legend()
plt.tight_layout()
plt.show()

# %% Patient 4 posterior summary
import numpy as np

p = 3   # python index; stan patient id = 4

print("Patient 4 posterior (per-patient parameters):\n")
for name in ["beta", "tp_dip", "tp_bump", "b_dip", "b_bump", "c"]:
    draws = fit.stan_variable(name)[:, p]   # (n_draws,) for patient 4
    print(f"{name:8s}  mean={draws.mean():8.3f}  sd={draws.std():7.3f}  "
          f"q2.5={np.quantile(draws,0.025):8.3f}  q97.5={np.quantile(draws,0.975):8.3f}")



# %% Patient 8 data check
m = stan_data["patient"] == 8
x8, y8 = stan_data["x"][m], stan_data["y"][m]
print(f"Patient 8: {m.sum()} obs, x range {x8.min():.0f}-{x8.max():.0f}")
plt.scatter(x8, y8, s=10, color="k", alpha=0.5); plt.axhline(0, ls="--", lw=0.8)
plt.xlim(0,60); plt.show()


# %%
