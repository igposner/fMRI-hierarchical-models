# %%
import numpy as np
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel

rng = np.random.default_rng(0)

T_SPLIT = 50.0   # must match the hardcoded t_split in dG_dip.stan

# %% Import data

# Import the fMRI timeseries into a time x participant (scanning day) matrix
data = np.genfromtxt(
    '../data/main_a1avg.csv',
    delimiter=',',
    skip_header=1,
    missing_values='NA'
)                                    # (420, 65)

# Import the covariates into a participant x predictor matrix
cov = np.genfromtxt('../data/MainStudy_withsexageanxiety.csv',
                    delimiter=',', skip_header=1)
# Pull out gender as our only covariate of interest
gender_all = cov[:, 1]
# Also pull out the participant IDs for merging covariates and fMRI data together
ids = cov[:, 0]

# Some of the participants had multiple days of scanning
# Their covariate data is blank for the additional days
# Populate the blank gender data for extra scan days
for i in np.where(np.isnan(gender_all))[0]:
    match = (ids == ids[i]) & ~np.isnan(gender_all)
    assert match.sum() == 1, f"no unique source for patid {ids[i]}"
    gender_all[i] = gender_all[match][0]

assert len(gender_all) == data.shape[1], f"Number of rows in covariate matrix {len(gender_all)} does not match number of columns in fMRI matrix {data.shape[1]}"
assert not np.isnan(gender_all).any(), "Some gender data is missing"

# %% full cohort, first 100 timepoints
data = data[:100]
print(f"{data.shape[1]} patients, {data.shape[0]} timepoints")
print(f"{(gender_all > 0).sum()} boys, {(gender_all < 0).sum()} girls")

# %% subset to 10 participants for fast iteration
data = data[:, :10]
gender_all = gender_all[:10]
print(f"Using {data.shape[1]} patients, gender: {gender_all}")

# %% Plot mean timeseries, averaging across participants
mean_signal = np.nanmean(data, axis=1)
sem_signal = np.nanstd(data, axis=1)
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

# read the dip off this curve -> mu_tp_dip prior center
dip_i = np.nanargmin(mean_signal)
print(f"empirical dip at t={t[dip_i]:.0f}, depth {mean_signal[dip_i]:.4f}")

# %% Compile the Stan model
model = CmdStanModel(stan_file="singleGamma.stan",
                     cpp_options={'STAN_THREADS': 'TRUE'})

# %% Build stan_data
n_timepoints, n_patients = data.shape

# Stack fMRI data columns into a single vector
y_long = data.flatten(order='F')
x_long = np.tile(t, n_patients)
patient_long = np.repeat(np.arange(n_patients) + 1, n_timepoints)

# Exclude censored/missing fMRI data
mask = ~np.isnan(y_long)

stan_data = {
    "n_obs":    int(mask.sum()),
    "n_ppl":    n_patients,
    "y":        y_long[mask],
    "x":        x_long[mask],
    "patient":  patient_long[mask].astype(int),
    "gender":   gender_all,
}

# %% Fit
fit = model.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_sampling=500,
    iter_warmup=500,
    show_console=True,
    adapt_delta=0.8,
)

fit.save_csvfiles(dir="fits")
print(fit.diagnose())

# %% Draws
tp_dip      = fit.stan_variable("tp_dip")        # (D, P)
b_dip       = fit.stan_variable("b_dip")         # (D,)  shared
beta        = fit.stan_variable("beta")          # (D, P)
sigma_early = fit.stan_variable("sigma_early")   # (D,)
sigma_late  = fit.stan_variable("sigma_late")    # (D,)

mu_tp_dip   = fit.stan_variable("mu_tp_dip")
beta_gender = fit.stan_variable("beta_gender")
mu_beta     = fit.stan_variable("mu_beta")

td_debug     = fit.stan_variable("td_debug")     # (D, n_obs)
lg_dip_debug = fit.stan_variable("lg_dip_debug")
mu_debug     = fit.stan_variable("mu_debug")

for name, arr in [("td", td_debug), ("lg_dip", lg_dip_debug), ("mu", mu_debug)]:
    print(f"{name:8s} min={np.nanmin(arr):12.4f}  max={np.nanmax(arr):12.4f}  "
          f"n_nan={np.isnan(arr).sum():d}  n_inf={np.isinf(arr).sum():d}")

print(f"min tp_dip across all draws/patients: {tp_dip.min():.3f}")
print(f"mu_tp_dip   mean={mu_tp_dip.mean():6.2f}")
print(f"b_dip       mean={b_dip.mean():6.3f}")
print(f"implied a_dip (= tp*b + 1) mean={(tp_dip * b_dip[:, None]).mean() + 1:.3f}")
print(f"implied dip width (sqrt(a)/b) = "
      f"{np.sqrt((tp_dip * b_dip[:, None]).mean() + 1) / b_dip.mean():.2f} TR")
print(f"sigma_early mean={sigma_early.mean():6.3f}   "
      f"sigma_late mean={sigma_late.mean():6.3f}")

# %% Reconstruct fitted curves (vectorized, log space for numerical safety)
n_draws, n_ppl = beta.shape
n_keep = min(500, n_draws)                       # thin for memory on the cluster
keep = rng.choice(n_draws, n_keep, replace=False)

x_plot = np.linspace(0.5, t[-1], 200)            # start > 0; log(0) is -inf


def unit_gamma(x, tp, b):
    """Gamma curve normalized to peak exactly 1 at tp.

    x  : (X,)          tp : (D, P)        b : (D,)
    returns (D, P, X)
    """
    tp_ = tp[:, :, None]
    b_ = b[:, None, None]
    x_ = x[None, None, :]
    lg = (tp_ * b_) * (np.log(x_) - np.log(tp_)) - b_ * (x_ - tp_)
    return np.exp(lg)


g_dip = unit_gamma(x_plot, tp_dip[keep], b_dip[keep])
mu_curves = -beta[keep][:, :, None] * g_dip      # (D, P, X)

# posterior predictive: heteroskedastic noise, sigma_early before T_SPLIT
sig_x = np.where(x_plot > T_SPLIT,
                 sigma_late[keep][:, None],
                 sigma_early[keep][:, None])     # (D, X)
y_pred = mu_curves + rng.normal(0.0, 1.0, mu_curves.shape) * sig_x[:, None, :]

y_plot_mean = mu_curves.mean(axis=0)             # (P, X)
ppi_lo = np.quantile(y_pred, 0.025, axis=0)
ppi_hi = np.quantile(y_pred, 0.975, axis=0)

# %% gender effect on amplitude
print(f"beta_gender  mean={beta_gender.mean():7.3f}  sd={beta_gender.std():6.3f}  "
      f"q2.5={np.quantile(beta_gender, 0.025):7.3f}  q97.5={np.quantile(beta_gender, 0.975):7.3f}")
print(f"P(beta_gender > 0) = {(beta_gender > 0).mean():.3f}")
print(f"mu_beta      mean={mu_beta.mean():7.3f}  sd={mu_beta.std():6.3f}")
print(f"sigma_beta   mean={fit.stan_variable('sigma_beta').mean():7.3f}"
      f"   (compare beta_gender against this, not against zero)")

boys, girls = gender_all > 0, gender_all < 0
print(f"\nsubset: {boys.sum()} boys, {girls.sum()} girls")
print(f"implied boy amplitude:  {(mu_beta + 0.5 * beta_gender).mean():7.3f}")
print(f"implied girl amplitude: {(mu_beta - 0.5 * beta_gender).mean():7.3f}")

plt.figure(figsize=(7, 3))
plt.hist(beta_gender, bins=60, color="C0", alpha=0.8)
plt.axvline(0, color="k", ls="--", lw=1)
plt.xlabel("beta_gender (boy - girl amplitude)")
plt.ylabel("posterior draws")
plt.tight_layout()
plt.show()

# %% dip curve by gender (credible bands -- decomposition, no observation noise)
plt.figure(figsize=(8, 4), dpi=120)
for sel, lab, col in [(boys, "boys", "C0"), (girls, "girls", "C3")]:
    if sel.sum() == 0:
        continue
    dips = mu_curves[:, sel, :].mean(axis=1)     # (D, X)
    plt.plot(x_plot, dips.mean(axis=0), color=col, lw=2, label=f"{lab} (n={sel.sum()})")
    plt.fill_between(x_plot, np.quantile(dips, 0.025, axis=0),
                     np.quantile(dips, 0.975, axis=0), color=col, alpha=0.2)
plt.axhline(0, color="k", ls="--", lw=0.8)
plt.axvline(mu_tp_dip.mean(), color="grey", ls=":", lw=0.8)
plt.xlabel("time (TR)")
plt.ylabel("BOLD")
plt.title("dip component by gender")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% Plot 4 patients
patients_to_show = [0, 2, 5, 8]   # python 0-indexed; stan patient = python idx + 1

fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)

for ax, p_idx in zip(axes.flat, patients_to_show):
    stan_id = p_idx + 1

    obs_mask = stan_data["patient"] == stan_id
    x_obs = stan_data["x"][obs_mask]
    y_obs = stan_data["y"][obs_mask]

    ax.plot(x_plot, y_plot_mean[p_idx], color="C0", label="posterior mean")
    ax.fill_between(x_plot, ppi_lo[p_idx], ppi_hi[p_idx],
                    color="C0", alpha=0.3, label="95% PPI")
    ax.scatter(x_obs, y_obs, s=8, color="k", alpha=0.5, label="data")

    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_title(f"Patient {stan_id}")
    ax.set_xlabel("time")
    ax.set_ylabel("BOLD")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

# %% coverage check: what fraction of observations fall inside the 95% PPI?
inside = 0
total = 0
for p_idx in range(n_ppl):
    obs_mask = stan_data["patient"] == p_idx + 1
    x_obs = stan_data["x"][obs_mask]
    y_obs = stan_data["y"][obs_mask]
    lo = np.interp(x_obs, x_plot, ppi_lo[p_idx])
    hi = np.interp(x_obs, x_plot, ppi_hi[p_idx])
    inside += np.sum((y_obs >= lo) & (y_obs <= hi))
    total += y_obs.size
print(f"PPI coverage: {inside / total:.3f} of observations inside the 95% band")

# %% Population mean curve
# the data being predicted here is the average over n_ppl patients, so the
# predictive noise shrinks by sqrt(n_ppl)
mu_pop = mu_curves.mean(axis=1)                  # (D, X)
y_pop_pred = mu_pop + rng.normal(0.0, 1.0, mu_pop.shape) * (sig_x / np.sqrt(n_ppl))

pop_mean = mu_pop.mean(axis=0)
pop_lo = np.quantile(y_pop_pred, 0.025, axis=0)
pop_hi = np.quantile(y_pop_pred, 0.975, axis=0)

plt.figure(figsize=(9, 4))
plt.plot(x_plot, pop_mean, color="C0", lw=2, label="population mean (avg of patients)")
plt.fill_between(x_plot, pop_lo, pop_hi, color="C0", alpha=0.25, label="95% PPI")
plt.plot(t, mean_signal, color="k", lw=1, ls="--", alpha=0.7, label="observed mean")
plt.axhline(0, color="k", ls="--", lw=0.8)
plt.xlabel("time")
plt.ylabel("BOLD")
plt.title("Population mean curve")
plt.legend()
plt.tight_layout()
plt.show()
# %%
