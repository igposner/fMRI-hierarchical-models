# %%
import numpy as np
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel

rng = np.random.default_rng(0)

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

# %% full cohort, first 65 timepoints
# The dip is over by ~TR 30. Cutting at 65 removes the late troughs that
# compete with it for tp_dip, without touching the dip itself.
data = data[:65]
print(f"{data.shape[1]} patients, {data.shape[0]} timepoints")
print(f"{(gender_all > 0).sum()} boys, {(gender_all < 0).sum()} girls")

# %% subset to exclude patient 20 and 61
drop = [20, 61]   # 0-indexed: stan patients 21 and 62
keep_p = np.setdiff1d(np.arange(data.shape[1]), drop)
data = data[:, keep_p]
gender_all = gender_all[keep_p]

# %% subset to 10 participants for fast iteration
# data = data[:, :10]
# gender_all = gender_all[:10]
# print(f"Using {data.shape[1]} patients, gender: {gender_all}")

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

# %% Read the three population priors off the mean curve.
#    tp -> mu_tp_dip_log, depth -> mu_beta_log, FWHM/2.35 -> width_dip.
dip_i = np.nanargmin(mean_signal)
tp_emp = t[dip_i]
depth_emp = abs(mean_signal[dip_i])

# half-depth crossings, for the width
half = mean_signal[dip_i] / 2.0
sign_change = np.where(np.diff(np.sign(mean_signal - half)))[0]

print(f"empirical dip at t={tp_emp:.0f} TR, depth {mean_signal[dip_i]:.4f}")
print(f"  -> mu_tp_dip_log ~ normal({np.log(tp_emp):.2f}, 0.3)")
print(f"  -> mu_beta_log   ~ normal({np.log(depth_emp):.2f}, 0.5)")
if len(sign_change) >= 2:
    fwhm = t[sign_change[-1]] - t[sign_change[0]]
    print(f"  FWHM ~ {fwhm:.1f} TR")
    print(f"  -> width_dip     ~ normal({fwhm / 2.35:.1f}, 2)")
else:
    print("  could not locate two half-depth crossings; set width_dip by eye")

# %% Compile the Stan model
model = CmdStanModel(stan_file="singleGamma2.stan",
                     cpp_options={'STAN_THREADS': 'TRUE'})

# %% Build stan_data
n_timepoints, n_patients = data.shape

# Stack fMRI data columns into a single vector
y_long = data.flatten(order='F')
x_long = np.tile(t, n_patients)
patient_long = np.repeat(np.arange(n_patients) + 1, n_timepoints)

# Exclude censored/missing fMRI data AND t == 0.
# The gamma is evaluated as log(x); the t=0 rows have no guard in the
# vector-vector-vector overload, and the gradient through log(0) is NaN.
mask = ~np.isnan(y_long) & (x_long > 0)
print(f"dropped {(~mask).sum()} rows ({np.isnan(y_long).sum()} NaN, "
      f"{(x_long == 0).sum()} at t=0)")

stan_data = {
    "n_obs":    int(mask.sum()),
    "n_ppl":    n_patients,
    "y":        y_long[mask],
    "x":        x_long[mask],
    "patient":  patient_long[mask].astype(int),
    "gender":   gender_all,
}

# %% Fit
# width_dip init matters: Stan's default draws it from [0.1, 7] on the
# constrained scale, and small values make c_dip enormous (~6700 at 0.14),
# which flattens the gradient and can stall warmup.

fit = model.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_sampling=1000,
    iter_warmup=1000,
    show_console=True,
    adapt_delta=0.8,
)

fit.save_csvfiles(dir=f"fits/sg2_noncentered_{fit.metadata.cmdstan_config['seed']}")
print(fit.diagnose())

# %% Draws
tp_dip    = fit.stan_variable("tp_dip")      # (D, P)  = exp(tp_dip_log)
beta_p    = fit.stan_variable("beta_p")      # (D, P)  = exp(beta_log)
c_dip     = fit.stan_variable("c_dip")       # (D, P)  = gamma_c(tp_dip, width_dip)
width_dip = fit.stan_variable("width_dip")   # (D,)    shared, curve SD in TRs

mu_tp_dip_log = fit.stan_variable("mu_tp_dip_log")
mu_beta_log   = fit.stan_variable("mu_beta_log")
beta_gender   = fit.stan_variable("beta_gender")   # LOG RATIO, not a difference
sigma_tp_dip  = fit.stan_variable("sigma_tp_dip")
sigma_beta    = fit.stan_variable("sigma_beta")
mem_gender    = fit.stan_variable("mem_gender")    # boy-girl depth diff, BOLD units

sigma = fit.stan_variable("sigma")                 # (D,)  single pooled noise SD

# %% Sanity checks
print(f"tp_dip     min={tp_dip.min():7.3f}  mean={tp_dip.mean():7.3f}  max={tp_dip.max():7.3f}")
print(f"beta_p     min={beta_p.min():7.3f}  mean={beta_p.mean():7.3f}  max={beta_p.max():7.3f}")
print(f"width_dip  mean={width_dip.mean():7.3f}  q2.5={np.quantile(width_dip, 0.025):6.3f}"
      f"  q97.5={np.quantile(width_dip, 0.975):6.3f}   (prior normal(5, 2))")
print(f"c_dip      min={c_dip.min():9.2f}  mean={c_dip.mean():9.2f}  max={c_dip.max():9.2f}")
print(f"implied a  (= c + 1) mean={c_dip.mean() + 1:.2f}")

# population values back on the natural scale
print(f"\npop tp_dip (exp mu_tp_dip_log) = {np.exp(mu_tp_dip_log).mean():6.2f} TR")
print(f"pop depth  (exp mu_beta_log)   = {np.exp(mu_beta_log).mean():6.2f} BOLD")
print(f"sigma      mean={sigma.mean():6.3f}  q2.5={np.quantile(sigma, 0.025):6.3f}"
      f"  q97.5={np.quantile(sigma, 0.975):6.3f}")
print("  (was sigma_early 1.02 / sigma_late 1.11 under the split -- "
      "the split was buying nothing)")

# the two fractional SDs are guesses -- check them against the fitted spread
print(f"\nsigma_tp_dip mean={sigma_tp_dip.mean():.3f}"
      f"   vs spread of log(tp_dip) across patients: "
      f"{np.log(tp_dip.mean(axis=0)).std():.3f}   (prior normal(0, 0.3))")
print(f"sigma_beta   mean={sigma_beta.mean():.3f}"
      f"   vs spread of log(beta_p) across patients: "
      f"{np.log(beta_p.mean(axis=0)).std():.3f}   (prior normal(0, 0.5))")

# %% Per-patient multimodality check.
# Chains that disagree about WHICH trough is the dip show up here as
# chain means tens of TR apart. This is the R-hat failure mode, and it is
# per-patient, not global -- population chain means agree closely.
d = fit.draws_pd()
n_ppl_ = tp_dip.shape[1]
spread = np.zeros(n_ppl_)
for p in range(n_ppl_):
    cm = d.groupby("chain__")[f"tp_dip[{p+1}]"].mean().values
    spread[p] = cm.max() - cm.min()

flagged = np.where(spread > 5.0)[0]
print(f"{len(flagged)} patients with >5 TR disagreement between chain means:")
for p in flagged:
    cm = d.groupby("chain__")[f"tp_dip[{p+1}]"].mean().values
    print(f"  patient {p+1:3d}: {np.round(cm, 1)}")

# %% Reconstruct fitted curves (vectorized, log space for numerical safety)
n_draws, n_ppl = beta_p.shape
n_keep = min(500, n_draws)                       # thin for memory on the cluster
keep = rng.choice(n_draws, n_keep, replace=False)

x_plot = np.linspace(0.5, t[-1], 200)            # start > 0; log(0) is -inf


def unit_gamma(x, tp, c):
    """Gamma curve normalized to peak exactly 1 at tp.

    Mirrors gamma_loc_sigma() in singleGamma2.stan.

    x  : (X,)          tp : (D, P)        c : (D, P)
    returns (D, P, X)
    """
    tp_ = tp[:, :, None]
    c_ = c[:, :, None]
    x_ = x[None, None, :]
    lg = c_ * (np.log(x_) - np.log(tp_) - x_ / tp_ + 1)
    return np.exp(lg)


g_dip = unit_gamma(x_plot, tp_dip[keep], c_dip[keep])
mu_curves = -beta_p[keep][:, :, None] * g_dip    # (D, P, X)  dip is downward

# posterior predictive: single homoskedastic noise term now
y_pred = mu_curves + rng.normal(0.0, 1.0, mu_curves.shape) * sigma[keep][:, None, None]

y_plot_mean = mu_curves.mean(axis=0)             # (P, X)
ppi_lo = np.quantile(y_pred, 0.025, axis=0)
ppi_hi = np.quantile(y_pred, 0.975, axis=0)

# %% gender effect on amplitude
# beta_gender is MULTIPLICATIVE: exp(beta_gender) is the boy/girl depth ratio.
pct = 100 * (np.exp(beta_gender) - 1)
print(f"beta_gender (log ratio)  mean={beta_gender.mean():7.3f}  sd={beta_gender.std():6.3f}  "
      f"q2.5={np.quantile(beta_gender, 0.025):7.3f}  q97.5={np.quantile(beta_gender, 0.975):7.3f}")
print(f"  -> boys dip {pct.mean():+.1f}% deeper than girls "
      f"[{np.quantile(pct, 0.025):+.1f}%, {np.quantile(pct, 0.975):+.1f}%]")
print(f"P(beta_gender > 0) = {(beta_gender > 0).mean():.3f}")
print(f"mem_gender (BOLD units)  mean={mem_gender.mean():7.3f}  "
      f"q2.5={np.quantile(mem_gender, 0.025):7.3f}  q97.5={np.quantile(mem_gender, 0.975):7.3f}")
print(f"sigma_beta  mean={sigma_beta.mean():7.3f}"
      f"   (compare beta_gender against this, not against zero)")

boys, girls = gender_all > 0, gender_all < 0
print(f"\nsubset: {boys.sum()} boys, {girls.sum()} girls")
print(f"implied boy depth:  {np.exp(mu_beta_log + 0.5 * beta_gender).mean():7.3f}")
print(f"implied girl depth: {np.exp(mu_beta_log - 0.5 * beta_gender).mean():7.3f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 3))
axes[0].hist(beta_gender, bins=60, color="C0", alpha=0.8)
axes[0].axvline(0, color="k", ls="--", lw=1)
axes[0].set_xlabel("beta_gender (log boy/girl depth ratio)")
axes[0].set_ylabel("posterior draws")
axes[1].hist(mem_gender, bins=60, color="C0", alpha=0.8)
axes[1].axvline(0, color="k", ls="--", lw=1)
axes[1].set_xlabel("mem_gender (boy - girl depth, BOLD)")
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
plt.axvline(np.exp(mu_tp_dip_log).mean(), color="grey", ls=":", lw=0.8)
plt.xlabel("time (TR)")
plt.ylabel("BOLD")
plt.title("dip curve by gender")
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
    ax.set_title(f"Patient {stan_id}  (tp={tp_dip[:, p_idx].mean():.1f}, "
                 f"depth={beta_p[:, p_idx].mean():.2f})")
    ax.set_xlabel("time")
    ax.set_ylabel("BOLD")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

# %% Plot the flagged (multimodal) patients, if any.
# NB the posterior MEAN curve for these is an average over chains sitting on
# different troughs -- it can show two dips, a shape no single draw has.
if len(flagged) > 0:
    n_show = min(9, len(flagged))
    ncol = 3
    nrow = int(np.ceil(n_show / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 3 * nrow), dpi=110)
    for ax, p_idx in zip(np.atleast_1d(axes).flat, flagged[:n_show]):
        obs_mask = stan_data["patient"] == p_idx + 1
        ax.plot(stan_data["x"][obs_mask], stan_data["y"][obs_mask],
                "k.", ms=4, alpha=0.6)
        ax.plot(x_plot, y_plot_mean[p_idx], color="C0", lw=1.5)
        ax.axhline(0, color="k", ls="--", lw=0.6)
        ax.set_title(f"patient {p_idx + 1}  (chain spread {spread[p_idx]:.0f} TR)",
                     fontsize=9)
    for ax in np.atleast_1d(axes).flat[n_show:]:
        ax.axis("off")
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
y_pop_pred = mu_pop + rng.normal(0.0, 1.0, mu_pop.shape) * (
    sigma[keep][:, None] / np.sqrt(n_ppl))

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

# %% Residuals.
# A single gamma decays monotonically to zero and cannot produce the positive
# overshoot the mean curve shows around TR 30-55. Systematic positive residual
# there is structural, not a bug -- and sigma absorbs it, so sigma is not a
# clean read on measurement noise.
resid = mean_signal - np.interp(t, x_plot, pop_mean)
plt.figure(figsize=(9, 3))
plt.plot(t, resid, color="C3", lw=1)
plt.axhline(0, color="k", ls="--", lw=0.8)
plt.axvspan(30, 55, color="grey", alpha=0.15, label="unmodelled overshoot")
plt.xlabel("time (TR)")
plt.ylabel("observed - fitted")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %%