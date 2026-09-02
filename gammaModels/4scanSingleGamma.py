# %%
import numpy as np
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel

rng = np.random.default_rng(0)

# %% Import data

# Import the fMRI timeseries, one file per run, into a run x time x patient array
run_files = [f'../data/main_a1run{r}.csv' for r in (1, 2, 3, 4)]
runs = np.stack([
    np.genfromtxt(f, delimiter=',', skip_header=1, missing_values='NA')
    for f in run_files
])                                   # (4, 420, 66)
n_runs = runs.shape[0]

# R's write.csv wrote row names as an unnamed, quoted first column ("", "1", "2", ...).
# Quoted strings parse as NaN, so it looks like a 66th all-NaN patient. Drop it.
# Without this the column survives to the missingness report as a patient with
# zero scans, and every patient after it is shifted one position against gender.
assert np.isnan(runs[:, :, 0]).all(), "expected an all-NaN row-name column at index 0"
runs = runs[:, :, 1:]                # (4, 420, 65), columns are V1..V65

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

assert len(gender_all) == runs.shape[2], f"Number of rows in covariate matrix {len(gender_all)} does not match number of columns in fMRI matrix {runs.shape[2]}"
assert not np.isnan(gender_all).any(), "Some gender data is missing"

# %% full cohort, first 65 timepoints
# The dip is over by ~TR 30. Cutting at 65 removes the late troughs that
# compete with it for tp_dip, without touching the dip itself.
runs = runs[:, :65, :]
t = np.arange(runs.shape[1], dtype=float)
print(f"{runs.shape[2]} patients, {n_runs} runs, {runs.shape[1]} timepoints")
print(f"{(gender_all > 0).sum()} boys, {(gender_all < 0).sum()} girls")

# %% subset to exclude patient 20 and 61
# NOT USED for the scan-level fit -- all 65 patients retained.
# drop = [20, 61]   # 0-indexed: stan patients 21 and 62
# keep_p = np.setdiff1d(np.arange(runs.shape[2]), drop)
# runs = runs[:, :, keep_p]
# gender_all = gender_all[keep_p]

# %% Missingness report, evaluated on the analysis window only.
# all-NaN column  -> scan absent, dropped entirely (this is the ragged case)
# partial NaN     -> scan kept, only the NaN timepoints dropped by the mask
# NB the all-NaN test runs AFTER the 65-TR cut, so a scan with data only past
# TR 65 counts as missing. Re-run this report if the window changes.
n_timepoints, n_patients = runs.shape[1], runs.shape[2]
allnan = np.isnan(runs).all(axis=1)              # (n_runs, n_patients)
anynan = np.isnan(runs).any(axis=1)
partial = anynan & ~allnan

for r in range(n_runs):
    print(f"run{r+1}: {allnan[r].sum():3d} missing scans, "
          f"{partial[r].sum():3d} partial ({np.isnan(runs[r][:, partial[r]]).sum()} NaN cells)")

per_patient = (~allnan).sum(axis=0)              # scans kept per patient
print(f"\ntotal scans: {per_patient.sum()} of {n_runs * n_patients} possible")
for k in range(n_runs + 1):
    if (per_patient == k).sum():
        print(f"  {(per_patient == k).sum():3d} patients with {k} scan(s)")
if (per_patient == 0).any():
    print(f"  WARNING: patients with no scans: {np.where(per_patient == 0)[0] + 1}")

# %% Build the scan index.
# Scans are numbered patient-major, so a patient's scans are contiguous.
# Ragged handled by absence: a dropped scan simply never gets an id.
scan_cols = [(r, p) for p in range(n_patients)
                    for r in range(n_runs) if not allnan[r, p]]
n_scans = len(scan_cols)

scan_patient = np.array([p + 1 for (_, p) in scan_cols], dtype=int)
scan_run = np.array([r + 1 for (r, _) in scan_cols], dtype=int)   # not passed to Stan
Y = np.column_stack([runs[r, :, p] for (r, p) in scan_cols])      # (T, n_scans)

# gender expanded to one value per scan, for grouping scan-level quantities
gender_scan = gender_all[scan_patient - 1]

print(f"n_scans={n_scans}, Y shape {Y.shape}")
print(f"scans per run: {[int((scan_run == r + 1).sum()) for r in range(n_runs)]}")
print(f"scan_patient runs {scan_patient.min()}..{scan_patient.max()}, "
      f"{len(np.unique(scan_patient))} distinct patients")
print(f"first 8 scans (patient, run): "
      f"{list(zip(scan_patient[:8], scan_run[:8]))}")
print(f"patients with <{n_runs} scans: "
      f"{[int(p) for p in np.unique(scan_patient) if (scan_patient == p).sum() < n_runs]}")

# %% Plot mean timeseries, averaging across all scans
mean_signal = np.nanmean(Y, axis=1)
sem_signal = np.nanstd(Y, axis=1)

figure, ax = plt.subplots(figsize=(10, 4))
ax.plot(t, mean_signal, color='steelblue', linewidth=1.5, label='Mean BOLD')
ax.fill_between(t, mean_signal - sem_signal, mean_signal + sem_signal,
                alpha=0.3, color='steelblue', label='±1 SD')
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlim([t[0], t[-1]])
ax.set_xlabel('Time (TRs)')
ax.set_ylabel('BOLD')
ax.set_title(f'Mean signal across {n_scans} scans ({n_patients} patients)')
ax.legend()
plt.tight_layout()
plt.show()

# %% Read the three population priors off the mean curve.
#    tp -> mu_tp_dip_log, depth -> mu_beta_log, FWHM/2.35 -> width_dip.
# NB this is now the mean over unaveraged scans, so the curve is noisier than
# the avg-file version. Depth should be similar; the width read may be shakier.
dip_i = np.nanargmin(mean_signal)
tp_emp = t[dip_i]
depth_emp = abs(mean_signal[dip_i])

# half-depth crossings, for the width.
# Restrict to finite timepoints first: sign(NaN) is NaN, diff propagates it,
# and np.where counts NaN as truthy -- an all-NaN TR 0 fakes a crossing at t=0
# and inflates the FWHM.
finite = np.isfinite(mean_signal)
ms_f, t_f = mean_signal[finite], t[finite]
half = mean_signal[dip_i] / 2.0
sign_change = np.where(np.diff(np.sign(ms_f - half)))[0]

print(f"empirical dip at t={tp_emp:.0f} TR, depth {mean_signal[dip_i]:.4f}")
print(f"  -> mu_tp_dip_log ~ normal({np.log(tp_emp):.2f}, 0.3)")
print(f"  -> mu_beta_log   ~ normal({np.log(depth_emp):.2f}, 0.5)")
print(f"  half-depth {half:.3f}, crossings at TR {np.round(t_f[sign_change], 1)}")
if len(sign_change) >= 2:
    fwhm = t_f[sign_change[-1]] - t_f[sign_change[0]]
    print(f"  FWHM ~ {fwhm:.1f} TR")
    print(f"  -> width_dip     ~ normal({fwhm / 2.35:.1f}, 2)")
    if len(sign_change) > 2:
        print("  NB >2 crossings -- curve re-crosses half depth; FWHM uses "
              "first and last, which may overstate the width")
else:
    print("  could not locate two half-depth crossings; set width_dip by eye")

# %% Compile the Stan model
model = CmdStanModel(stan_file="4scanSingleGamma.stan",
                     cpp_options={'STAN_THREADS': 'TRUE'})

# %% Build stan_data

# Stack scan columns into a single vector
y_long = Y.flatten(order='F')
x_long = np.tile(t, n_scans)
scan_long = np.repeat(np.arange(n_scans) + 1, n_timepoints)

# Exclude censored/missing fMRI data AND t == 0.
# The gamma is evaluated as log(x); the t=0 rows have no guard in the
# vector-vector-vector overload, and the gradient through log(0) is NaN.
mask = ~np.isnan(y_long) & (x_long > 0)
print(f"dropped {(~mask).sum()} rows ({np.isnan(y_long).sum()} NaN, "
      f"{(x_long == 0).sum()} at t=0)")

stan_data = {
    "n_obs":        int(mask.sum()),
    "n_ppl":        n_patients,
    "n_scans":      n_scans,
    "y":            y_long[mask],
    "x":            x_long[mask],
    "scan":         scan_long[mask].astype(int),
    "scan_patient": scan_patient,
    "gender":       gender_all,
}
print(f"n_obs={stan_data['n_obs']}, n_scans={n_scans}, n_ppl={n_patients}")

# %% Fit
# width_dip init matters: Stan's default draws it from [0.1, 7] on the
# constrained scale, and small values make c_dip enormous (~6700 at 0.14),
# which flattens the gradient and can stall warmup.
#
# adapt_delta held at 0.8 deliberately for this first scan-level run: raising
# it would mask divergences caused by the new level. Divergences here are
# diagnostic, not automatically a failure.

fit = model.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_sampling=500,
    iter_warmup=500,
    show_console=True,
    adapt_delta=0.8,
)

fit.save_csvfiles(dir=f"fits/sg2_scanlevel_{fit.metadata.cmdstan_config['seed']}")
print(fit.diagnose())

# %% Draws
# scan level (what enters the likelihood)
tp_dip_s  = fit.stan_variable("tp_dip_s")    # (D, S)  = exp(tp_dip_log_s)
beta_s    = fit.stan_variable("beta_s")      # (D, S)  = exp(beta_log_s)
c_dip     = fit.stan_variable("c_dip")       # (D, S)  = gamma_c(tp_dip_s, width_dip)
width_dip = fit.stan_variable("width_dip")   # (D,)    shared, curve SD in TRs

# patient level (where the gender regression lives)
tp_dip_log = fit.stan_variable("tp_dip_log")  # (D, P)
beta_log   = fit.stan_variable("beta_log")    # (D, P)

mu_tp_dip_log = fit.stan_variable("mu_tp_dip_log")
mu_beta_log   = fit.stan_variable("mu_beta_log")
beta_gender   = fit.stan_variable("beta_gender")    # LOG RATIO, not a difference
sigma_tp_dip  = fit.stan_variable("sigma_tp_dip")   # between patients
sigma_beta    = fit.stan_variable("sigma_beta")     # between patients
sigma_tp_scan = fit.stan_variable("sigma_tp_scan")  # between scans within patient
sigma_beta_scan = fit.stan_variable("sigma_beta_scan")
mem_gender    = fit.stan_variable("mem_gender")     # boy-girl depth diff, BOLD units

sigma = fit.stan_variable("sigma")                  # (D,)  within-scan noise SD

# %% Sanity checks
print(f"tp_dip_s   min={tp_dip_s.min():7.3f}  mean={tp_dip_s.mean():7.3f}  max={tp_dip_s.max():7.3f}")
print(f"beta_s     min={beta_s.min():7.3f}  mean={beta_s.mean():7.3f}  max={beta_s.max():7.3f}")
print(f"width_dip  mean={width_dip.mean():7.3f}  q2.5={np.quantile(width_dip, 0.025):6.3f}"
      f"  q97.5={np.quantile(width_dip, 0.975):6.3f}   (prior normal(5, 2))")
print(f"c_dip      min={c_dip.min():9.2f}  mean={c_dip.mean():9.2f}  max={c_dip.max():9.2f}")
print(f"implied a  (= c + 1) mean={c_dip.mean() + 1:.2f}")

# population values back on the natural scale
print(f"\npop tp_dip (exp mu_tp_dip_log) = {np.exp(mu_tp_dip_log).mean():6.2f} TR")
print(f"pop depth  (exp mu_beta_log)   = {np.exp(mu_beta_log).mean():6.2f} BOLD")
print(f"sigma      mean={sigma.mean():6.3f}  q2.5={np.quantile(sigma, 0.025):6.3f}"
      f"  q97.5={np.quantile(sigma, 0.975):6.3f}")
print("  sigma is now WITHIN-SCAN residual, not within-average.")
print("  expect ~2x the averaged-data posterior (averaging 4 scans cut noise by sqrt(4)).")

# the patient-level fractional SDs are guesses -- check against the fitted spread
print(f"\nsigma_tp_dip mean={sigma_tp_dip.mean():.3f}"
      f"   vs spread of tp_dip_log across patients: "
      f"{tp_dip_log.mean(axis=0).std():.3f}   (prior normal(0, 0.15))")
print(f"sigma_beta   mean={sigma_beta.mean():.3f}"
      f"   vs spread of beta_log across patients: "
      f"{beta_log.mean(axis=0).std():.3f}   (prior normal(0, 0.1))")

# %% Variance decomposition: the whole point of the scan level.
# sigma_*_dip  = how much PATIENTS differ from each other
# sigma_*_scan = how much one patient's SCANS differ from each other
# These two split a fixed total, so they trade off. A strong negative
# correlation is a ridge -- the data cannot cleanly separate them. Watch the
# scan SD in particular: if it piles up at 0, scan-to-scan variation is
# negligible and the model has effectively reduced to replicates-only.
for lab, between, within in [("tp", sigma_tp_dip, sigma_tp_scan),
                             ("beta", sigma_beta, sigma_beta_scan)]:
    r = np.corrcoef(between, within)[0, 1]
    frac = within.mean() / (within.mean() + between.mean())
    print(f"{lab:5s} between={between.mean():.4f}  within={within.mean():.4f}"
          f"  within/(within+between)={frac:.2f}  corr={r:+.3f}"
          f"  P(within<0.01)={(within < 0.01).mean():.3f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=110)
for ax, lab, between, within in [(axes[0], "tp", sigma_tp_dip, sigma_tp_scan),
                                 (axes[1], "beta", sigma_beta, sigma_beta_scan)]:
    ax.scatter(between, within, s=2, alpha=0.15, color="C0")
    ax.set_xlabel(f"sigma_{lab}_dip (between patients)")
    ax.set_ylabel(f"sigma_{lab}_scan (within patient)")
    ax.set_title(f"{lab}: ridge check")
plt.tight_layout()
plt.show()

# %% Multimodality check, at the scan level.
# Chains that disagree about WHICH trough is the dip show up here as chain
# means tens of TR apart. This is the R-hat failure mode.
d = fit.draws_pd()
spread = np.zeros(n_scans)
for s in range(n_scans):
    cm = d.groupby("chain__")[f"tp_dip_s[{s+1}]"].mean().values
    spread[s] = cm.max() - cm.min()

flagged = np.where(spread > 5.0)[0]
print(f"{len(flagged)} scans with >5 TR disagreement between chain means:")
for s in flagged:
    cm = d.groupby("chain__")[f"tp_dip_s[{s+1}]"].mean().values
    print(f"  scan {s+1:3d} (patient {scan_patient[s]:3d}, run {scan_run[s]}): {np.round(cm, 1)}")

if len(flagged):
    bad_p, bad_n = np.unique(scan_patient[flagged], return_counts=True)
    print(f"\naffected patients (scans flagged / scans total):")
    for p, n in zip(bad_p, bad_n):
        print(f"  patient {p:3d}: {n}/{(scan_patient == p).sum()}")

# %% Reconstruct fitted curves (vectorized, log space for numerical safety)
n_draws = beta_s.shape[0]
n_keep = min(200, n_draws)     # 200 kept: (200, n_scans, 200) is already ~80 MB/array
keep = rng.choice(n_draws, n_keep, replace=False)

x_plot = np.linspace(0.5, t[-1], 200)            # start > 0; log(0) is -inf


def gamma_c(loc, sigma_w):
    """Mirrors gamma_c() in the Stan model. Elementwise."""
    loc2 = loc ** 2
    s2 = sigma_w ** 2
    return 0.5 * (loc2 + loc * np.sqrt(loc2 + 4 * s2)) / s2


def unit_gamma(x, tp, c):
    """Gamma curve normalized to peak exactly 1 at tp.

    Mirrors gamma_loc_sigma() in the Stan model.

    x  : (X,)          tp : (D, N)        c : (D, N)
    returns (D, N, X)
    """
    tp_ = tp[:, :, None]
    c_ = c[:, :, None]
    x_ = x[None, None, :]
    lg = c_ * (np.log(x_) - np.log(tp_) - x_ / tp_ + 1)
    return np.exp(lg)


# scan-level curves: what the likelihood actually predicts
g_dip = unit_gamma(x_plot, tp_dip_s[keep], c_dip[keep])
mu_curves = -beta_s[keep][:, :, None] * g_dip    # (D, S, X)  dip is downward

# patient-level curves: the scan effects averaged out. Used for the gender
# contrast, which lives at the patient level.
tp_dip_p = np.exp(tp_dip_log[keep])              # (D, P)
beta_p = np.exp(beta_log[keep])                  # (D, P)
c_dip_p = gamma_c(tp_dip_p, width_dip[keep][:, None])
mu_curves_p = -beta_p[:, :, None] * unit_gamma(x_plot, tp_dip_p, c_dip_p)

# posterior predictive: single homoskedastic noise term, within-scan
y_pred = mu_curves + rng.normal(0.0, 1.0, mu_curves.shape) * sigma[keep][:, None, None]

y_plot_mean = mu_curves.mean(axis=0)             # (S, X)
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
print(f"  scans: {(gender_scan > 0).sum()} from boys, {(gender_scan < 0).sum()} from girls")
print(f"implied boy depth:  {np.exp(mu_beta_log + 0.5 * beta_gender).mean():7.3f}")
print(f"implied girl depth: {np.exp(mu_beta_log - 0.5 * beta_gender).mean():7.3f}")
print("NB mu_beta_log is the MIDPOINT of the two groups, not the cohort average --")
print("   with unequal group sizes these differ. (open question for Kay)")

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
# built from PATIENT-level curves: gender is a patient attribute, and this
# keeps scan-to-scan jitter out of the contrast.
plt.figure(figsize=(8, 4), dpi=120)
for sel, lab, col in [(boys, "boys", "C0"), (girls, "girls", "C3")]:
    if sel.sum() == 0:
        continue
    dips = mu_curves_p[:, sel, :].mean(axis=1)   # (D, X)
    plt.plot(x_plot, dips.mean(axis=0), color=col, lw=2, label=f"{lab} (n={sel.sum()})")
    plt.fill_between(x_plot, np.quantile(dips, 0.025, axis=0),
                     np.quantile(dips, 0.975, axis=0), color=col, alpha=0.2)
plt.axhline(0, color="k", ls="--", lw=0.8)
plt.axvline(np.exp(mu_tp_dip_log).mean(), color="grey", ls=":", lw=0.8)
plt.xlabel("time (TR)")
plt.ylabel("BOLD")
plt.title("dip curve by gender (patient level)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% Plot 4 patients, all of their scans overlaid.
# The dashed black line is the patient-level curve; the coloured lines are the
# per-scan fits. The gap between them is what sigma_*_scan is estimating.
patients_to_show = [1, 3, 6, 9]   # stan patient ids

fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)

for ax, stan_p in zip(axes.flat, patients_to_show):
    scans_of_p = np.where(scan_patient == stan_p)[0]

    for k, s in enumerate(scans_of_p):
        obs_mask = stan_data["scan"] == s + 1
        ax.scatter(stan_data["x"][obs_mask], stan_data["y"][obs_mask],
                   s=6, color=f"C{k}", alpha=0.35)
        ax.plot(x_plot, y_plot_mean[s], color=f"C{k}", lw=1.2,
                label=f"run {scan_run[s]} (tp={tp_dip_s[:, s].mean():.1f})")

    ax.plot(x_plot, mu_curves_p[:, stan_p - 1, :].mean(axis=0),
            color="k", ls="--", lw=2, label="patient level")
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_title(f"Patient {stan_p}  ({len(scans_of_p)} scans, "
                 f"depth={np.exp(beta_log[:, stan_p - 1]).mean():.2f})")
    ax.set_xlabel("time")
    ax.set_ylabel("BOLD")
    ax.legend(fontsize=7)

plt.tight_layout()
plt.show()

# %% Plot the flagged (multimodal) scans, if any.
# NB the posterior MEAN curve for these is an average over chains sitting on
# different troughs -- it can show two dips, a shape no single draw has.
if len(flagged) > 0:
    n_show = min(9, len(flagged))
    ncol = 3
    nrow = int(np.ceil(n_show / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 3 * nrow), dpi=110)
    for ax, s in zip(np.atleast_1d(axes).flat, flagged[:n_show]):
        obs_mask = stan_data["scan"] == s + 1
        ax.plot(stan_data["x"][obs_mask], stan_data["y"][obs_mask],
                "k.", ms=4, alpha=0.6)
        ax.plot(x_plot, y_plot_mean[s], color="C0", lw=1.5)
        ax.axhline(0, color="k", ls="--", lw=0.6)
        ax.set_title(f"scan {s + 1} (pt {scan_patient[s]}, run {scan_run[s]}, "
                     f"spread {spread[s]:.0f} TR)", fontsize=9)
    for ax in np.atleast_1d(axes).flat[n_show:]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# %% coverage check: what fraction of observations fall inside the 95% PPI?
inside = 0
total = 0
for s in range(n_scans):
    obs_mask = stan_data["scan"] == s + 1
    x_obs = stan_data["x"][obs_mask]
    y_obs = stan_data["y"][obs_mask]
    lo = np.interp(x_obs, x_plot, ppi_lo[s])
    hi = np.interp(x_obs, x_plot, ppi_hi[s])
    inside += np.sum((y_obs >= lo) & (y_obs <= hi))
    total += y_obs.size
print(f"PPI coverage: {inside / total:.3f} of observations inside the 95% band")

# %% Population mean curve
# averaged over SCANS, to match mean_signal above (which is also a scan mean --
# with ragged data, patients with more scans get more weight in both).
mu_pop = mu_curves.mean(axis=1)                  # (D, X)
y_pop_pred = mu_pop + rng.normal(0.0, 1.0, mu_pop.shape) * (
    sigma[keep][:, None] / np.sqrt(n_scans))

pop_mean = mu_pop.mean(axis=0)
pop_lo = np.quantile(y_pop_pred, 0.025, axis=0)
pop_hi = np.quantile(y_pop_pred, 0.975, axis=0)

plt.figure(figsize=(9, 4))
plt.plot(x_plot, pop_mean, color="C0", lw=2, label="population mean (avg of scans)")
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