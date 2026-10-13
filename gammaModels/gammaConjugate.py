# %% ===== CELL 1: IMPORTS =====
# Run once per kernel session. Nothing here depends on the model or the data.

import numpy as np
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel

rng = np.random.default_rng(0)


# %% ===== CELL 2: CONFIG + LOAD/PREP DATA =====
# Everything that touches the raw CSVs. Rerun when the window, the exclusion
# list, or the source files change -- not when the model changes.
# Produces: Y, t, scan_patient, scan_run, gender_all, gender_scan, n_* counts.

WINDOW  = 65        # timepoints kept; dip is over by ~TR 30
EXCLUDE = []        # 0-indexed patients to drop, e.g. [20, 61] for stan ids 21/62
VERBOSE = True      # missingness + scan-index detail

RUN_FILES = [f'../data/main_a1run{r}.csv' for r in (1, 2, 3, 4)]
COV_FILE  = '../data/MainStudy_withsexageanxiety.csv'

# --- fMRI timeseries: (run, time, patient) ---
runs = np.stack([np.genfromtxt(f, delimiter=',', skip_header=1, missing_values='NA')
                 for f in RUN_FILES])
n_runs = runs.shape[0]

# R's write.csv wrote row names as an unnamed quoted first column, which parses
# as an all-NaN 66th patient. Dropping it keeps patients aligned with gender.
assert np.isnan(runs[:, :, 0]).all(), "expected an all-NaN row-name column at index 0"
runs = runs[:, :, 1:]

# --- covariates ---
cov = np.genfromtxt(COV_FILE, delimiter=',', skip_header=1)
gender_all, ids = cov[:, 1], cov[:, 0]

# participants scanned on multiple days have blank covariates on extra rows
for i in np.where(np.isnan(gender_all))[0]:
    match = (ids == ids[i]) & ~np.isnan(gender_all)
    assert match.sum() == 1, f"no unique source for patid {ids[i]}"
    gender_all[i] = gender_all[match][0]

assert len(gender_all) == runs.shape[2], "covariate rows != fMRI columns"
assert not np.isnan(gender_all).any(), "some gender data is missing"

# --- window + exclusions ---
runs = runs[:, :WINDOW, :]
if EXCLUDE:
    keep_p = np.setdiff1d(np.arange(runs.shape[2]), EXCLUDE)
    runs, gender_all = runs[:, :, keep_p], gender_all[keep_p]

t = np.arange(runs.shape[1], dtype=float)
n_timepoints, n_patients = runs.shape[1], runs.shape[2]

# --- missingness, evaluated on the analysis window only ---
# all-NaN column -> scan absent, dropped (the ragged case)
# partial NaN    -> scan kept, NaN timepoints dropped by the mask in cell 4
allnan  = np.isnan(runs).all(axis=1)
partial = np.isnan(runs).any(axis=1) & ~allnan
per_patient = (~allnan).sum(axis=0)

# --- scan index: patient-major, so a patient's scans are contiguous ---
scan_cols = [(r, p) for p in range(n_patients)
                    for r in range(n_runs) if not allnan[r, p]]
n_scans      = len(scan_cols)
scan_patient = np.array([p + 1 for (_, p) in scan_cols], dtype=int)
scan_run     = np.array([r + 1 for (r, _) in scan_cols], dtype=int)
Y            = np.column_stack([runs[r, :, p] for (r, p) in scan_cols])
gender_scan  = gender_all[scan_patient - 1]

if VERBOSE:
    for r in range(n_runs):
        print(f"run{r+1}: {allnan[r].sum():3d} missing, {partial[r].sum():3d} partial "
              f"({np.isnan(runs[r][:, partial[r]]).sum()} NaN cells)")
    for k in range(n_runs + 1):
        if (per_patient == k).sum():
            print(f"  {(per_patient == k).sum():3d} patients with {k} scan(s)")
    if (per_patient == 0).any():
        print(f"  WARNING: patients with no scans: {np.where(per_patient == 0)[0] + 1}")
    ragged = [int(p) for p in np.unique(scan_patient) if (scan_patient == p).sum() < n_runs]
    print(f"  patients with <{n_runs} scans: {ragged}")

print(f"{n_patients} patients ({(gender_all > 0).sum()} boys, {(gender_all < 0).sum()} girls), "
      f"{n_scans} scans of {n_runs * n_patients} possible, {n_timepoints} timepoints")

# --- mean signal across all scans, for eyeballing the dip before fitting ---
mean_signal = np.nanmean(Y, axis=1)
sd_signal   = np.nanstd(Y, axis=1)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t, mean_signal, color='steelblue', lw=1.5, label='Mean BOLD')
ax.fill_between(t, mean_signal - sd_signal, mean_signal + sd_signal,
                alpha=.3, color='steelblue', label='±1 SD')
ax.axhline(0, color='k', lw=.8, ls='--')
ax.set(xlim=[t[0], t[-1]], xlabel='Time (TRs)', ylabel='BOLD',
       title=f'Mean signal across {n_scans} scans ({n_patients} patients)')
ax.legend(); plt.tight_layout(); plt.show()

# %% ===== CELL 2.5 (OPTIONAL): SUBSET TO N PATIENTS =====
# Fast iteration on a small cohort. Skip this cell for the full run.
# Rebuilds the scan index, so run it BEFORE cell 4 -- and rerun cell 2 first
# if you want to go back to the full cohort.

SUBSET_N = 10       # keep the first N patients

runs        = runs[:, :, :SUBSET_N]
gender_all  = gender_all[:SUBSET_N]
n_patients  = runs.shape[2]

allnan    = np.isnan(runs).all(axis=1)
scan_cols = [(r, p) for p in range(n_patients)
                    for r in range(n_runs) if not allnan[r, p]]
n_scans      = len(scan_cols)
scan_patient = np.array([p + 1 for (_, p) in scan_cols], dtype=int)
scan_run     = np.array([r + 1 for (r, _) in scan_cols], dtype=int)
Y            = np.column_stack([runs[r, :, p] for (r, p) in scan_cols])
gender_scan  = gender_all[scan_patient - 1]

mean_signal = np.nanmean(Y, axis=1)
sd_signal   = np.nanstd(Y, axis=1)

print(f"SUBSET: {n_patients} patients ({(gender_all > 0).sum()} boys, "
      f"{(gender_all < 0).sum()} girls), {n_scans} scans")


# %% ===== CELL 3: COMPILE =====
# Rerun after editing the .stan file. Independent of the data.

STAN_FILE = "gammaConjugate.stan"

model = CmdStanModel(stan_file=STAN_FILE, cpp_options={'STAN_THREADS': 'TRUE'})


# %% ===== CELL 4: BUILD stan_data =====
# Flatten scans to long format and mask. obs_patient is derived inside the
# model's transformed data block; the copy here is only for grouping diagnostics.

y_long    = Y.flatten(order='F')
x_long    = np.tile(t, n_scans)
scan_long = np.repeat(np.arange(n_scans) + 1, n_timepoints)

# Drop NaN and t == 0. The t=0 drop is belt-and-braces (gamma_loc_sigma guards
# x == 0 internally) but keeps n_obs comparable across fits.
mask = ~np.isnan(y_long) & (x_long > 0)

stan_data = {
    "n_obs":        int(mask.sum()),
    "n_ppl":        n_patients,
    "n_scans":      n_scans,
    "y":            y_long[mask],
    "x":            x_long[mask],
    "scan":         scan_long[mask].astype(int),
    "scan_patient": scan_patient,
    "gender":       gender_all,
    "ess_ratio":    1.0,
    "s0_tp":        0.10
}
obs_patient = scan_patient[stan_data["scan"] - 1]

print(f"n_obs={stan_data['n_obs']}, n_scans={n_scans}, n_ppl={n_patients}, "
      f"dropped {(~mask).sum()} rows ({np.isnan(y_long).sum()} NaN, "
      f"{(x_long == 0).sum()} at t=0)")


# %% ===== CELL 5: FIT =====
# adapt_delta held at 0.8 deliberately: raising it would mask divergences.

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

# %% ===== CELL 6: EXTRACT DRAWS + RECONSTRUCT CURVES =====
# Silent; builds objects only. Timing and width have no patient level:
# both go population -> scan. Only depth goes population -> patient -> scan.

# timing: population -> scan
mu_tp_log     = fit.stan_variable("mu_tp_log")            # (D,)
sigma_tp_scan = fit.stan_variable("sigma_tp_scan")        # (D,)   scan-to-scan
tp_log_s      = fit.stan_variable("tp_log_s")             # (D, S)
tp_dip_s      = fit.stan_variable("tp_dip_s")             # (D, S) TR scale
tp_median     = fit.stan_variable("tp_median")            # (D,)

# width: population -> scan
mu_width_log     = fit.stan_variable("mu_width_log")      # (D,)
sigma_width_scan = fit.stan_variable("sigma_width_scan")  # (D,)  scan-to-scan
width_dip_s      = fit.stan_variable("width_dip_s")       # (D, S) TR scale
c_dip_s          = fit.stan_variable("c_dip_s")           # (D, S) gamma shape
width_median     = fit.stan_variable("width_median")      # (D,)

# amplitude: population -> patient -> scan
mu_beta_log     = fit.stan_variable("mu_beta_log")
beta_gender     = fit.stan_variable("beta_gender")       # LOG RATIO, not a difference
sigma_beta      = fit.stan_variable("sigma_beta")
sigma_beta_scan = fit.stan_variable("sigma_beta_scan")
beta_log        = fit.stan_variable("beta_log")          # (D, P)
beta_s          = fit.stan_variable("beta_s")            # (D, S)
mem_gender      = fit.stan_variable("mem_gender")

# per-patient residual noise
sigma_p         = fit.stan_variable("sigma_p")           # (D, P)
mu_log_sigma    = fit.stan_variable("mu_log_sigma")
sigma_log_sigma = fit.stan_variable("sigma_log_sigma")
sigma_median    = fit.stan_variable("sigma_median")


def gamma_c(loc, sigma):
    """Mirrors gamma_c() in Stan: (location, SD) -> gamma shape parameter c."""
    loc2, sigma2 = loc ** 2, sigma ** 2
    return 0.5 * (loc2 + loc * np.sqrt(loc2 + 4 * sigma2)) / sigma2


def unit_gamma(x, tp, c):
    """Gamma curve normalized to peak exactly 1 at tp. Mirrors gamma_loc_sigma().
    x: (X,)   tp: (D, N)   c: (D, N)   ->  (D, N, X)"""
    tp_, c_, x_ = tp[:, :, None], c[:, :, None], x[None, None, :]
    return np.exp(c_ * (np.log(x_) - np.log(tp_) - x_ / tp_ + 1))


n_keep = min(200, beta_s.shape[0])   # (200, S, 200) float64 ~ 80 MB per array
keep   = rng.choice(beta_s.shape[0], n_keep, replace=False)
x_plot = np.linspace(0.5, t[-1], 200)          # start > 0; log(0) is -inf

# scan-level curves: what the likelihood predicts. Timing, width and depth all
# come from the scan itself.
tp_scan   = tp_dip_s[keep]                                  # (D, S)
mu_curves = -beta_s[keep][:, :, None] * unit_gamma(x_plot, tp_scan, c_dip_s[keep])

# patient-level curves: scan effects averaged out. Depth is the patient's own;
# timing and width are POPULATION, since neither has a patient level.
beta_p      = np.exp(beta_log[keep])                        # (D, P)
tp_p        = np.repeat(tp_median[keep][:, None],   beta_p.shape[1], axis=1)
width_p     = np.repeat(width_median[keep][:, None], beta_p.shape[1], axis=1)
c_p         = gamma_c(tp_p, width_p)
mu_curves_p = -beta_p[:, :, None] * unit_gamma(x_plot, tp_p, c_p)

# posterior predictive: noise is the owning patient's sigma
sigma_scan = sigma_p[keep][:, scan_patient - 1]             # (D, S)
y_pred = mu_curves + rng.normal(0.0, 1.0, mu_curves.shape) * sigma_scan[:, :, None]

y_plot_mean = mu_curves.mean(axis=0)                        # (S, X)
ppi_lo = np.quantile(y_pred, 0.025, axis=0)
ppi_hi = np.quantile(y_pred, 0.975, axis=0)

# %% ===== CELL 7: DIAGNOSTICS REPORT =====
# All text. Sections: parameter sanity -> identifiability -> scan-level spread
# -> per-patient noise -> variance decomposition -> chain agreement ->
# residual autocorrelation -> gender -> PPI coverage.

sig_p_mean   = sigma_p.mean(axis=0)
width_s_mean = width_dip_s.mean(axis=0)
tp_p_mean    = tp_median.mean()           # scalar: no patient timing level
w_p_mean     = width_median.mean()        # scalar: no patient width level
tp_s_mean    = tp_dip_s.mean(axis=0)
boys, girls  = gender_all > 0, gender_all < 0

print("=== parameters ===")
print(f"tp_median     mean={tp_median.mean():7.3f}  [{np.quantile(tp_median,.025):6.3f},"
      f" {np.quantile(tp_median,.975):6.3f}]   (pooled fit gave 10.28)")
print(f"tp_dip_s      min={tp_s_mean.min():7.3f}  median={np.median(tp_s_mean):7.3f}"
      f"  max={tp_s_mean.max():7.3f}   (bounded 4-28)")
print(f"  at floor (<4.2) {np.sum(tp_s_mean < 4.2):3d} scans"
      f"   at ceiling (>27.5) {np.sum(tp_s_mean > 27.5):3d} scans  of {n_scans}")
print(f"width_median  mean={width_median.mean():7.3f}  [{np.quantile(width_median,.025):6.3f},"
      f" {np.quantile(width_median,.975):6.3f}]")
print(f"width_dip_s   min={width_s_mean.min():7.3f}  median={np.median(width_s_mean):7.3f}"
      f"  max={width_s_mean.max():7.3f}")
print(f"c_dip_s       min={c_dip_s.mean(0).min():9.2f}  mean={c_dip_s.mean():9.2f}"
      f"  max={c_dip_s.mean(0).max():9.2f}")
print(f"beta_s        min={beta_s.min():7.3f}  mean={beta_s.mean():7.3f}  max={beta_s.max():7.3f}")
print(f"pop depth (exp mu_beta_log) = {np.exp(mu_beta_log).mean():.2f} BOLD")

print("\n=== identifiability ===")
print("per-parameter posterior SD vs the population SD it was drawn from.")
print("ratio near 1 = data adds nothing, parameter is sampling its prior.")
for lab, pat, pop in [("tp_log_s  (scan)   ", tp_log_s,   sigma_tp_scan),
                      ("beta_log  (patient)", beta_log,   sigma_beta)]:
    post_sd = np.median(pat.std(axis=0))
    print(f"{lab}  post sd {post_sd:.3f}  vs sigma {pop.mean():.3f}"
          f"  ratio {post_sd / pop.mean():.2f}"
          f"   | fitted spread {pat.mean(axis=0).std():.3f}")
print("NB patient levels for tp and width were REMOVED: ratios 1.27 and 1.19,")
print("   fitted spreads 0.017 and 0.026 across 65 patients. Depth is the only")
print("   patient level left, kept because beta_gender sits on it.")

print("\n=== scan-level spread ===")
print(f"sigma_tp_scan     mean={sigma_tp_scan.mean():.4f}  "
      f"[{np.quantile(sigma_tp_scan,.025):.4f}, {np.quantile(sigma_tp_scan,.975):.4f}]"
      f"   (prior normal(0, 0.1))")
print(f"  P(sigma_tp_scan > 0.20) = {(sigma_tp_scan > 0.20).mean():.3f}")
if np.quantile(sigma_tp_scan, 0.9) > 0.20:
    print("  WARNING: pressing against its prior. Drift is believed to be NOISE,")
    print("           not real timing variation -- do not widen. The fix is a")
    print("           temporal error model, not a looser prior.")
print(f"  implied scan drift at tp=11: +-{11*(np.exp(sigma_tp_scan.mean())-1):.1f} TR at 1sd")
print(f"sigma_width_scan  mean={sigma_width_scan.mean():.4f}   (was 0.351 WITH a patient level)")
print(f"sigma_beta_scan   mean={sigma_beta_scan.mean():.4f}   (was 1.19 WITHOUT scan timing)")
print(f"  -> {100*(np.exp(sigma_beta_scan.mean())-1):.0f}% depth variation at 1sd."
      f" fMRI test-retest literature is 14-23% between sessions.")

print("\n=== per-patient noise ===")
print(f"sigma_median   mean={sigma_median.mean():6.3f}  [{np.quantile(sigma_median,.025):6.3f},"
      f" {np.quantile(sigma_median,.975):6.3f}]   (pooled-sigma fit gave 1.83)")
print(f"sigma_log_sigma mean={sigma_log_sigma.mean():6.3f} -> "
      f"{np.exp(sigma_log_sigma.mean()):.2f}x noise ratio at 1sd")
print(f"sigma_p        min={sig_p_mean.min():6.3f}  median={np.median(sig_p_mean):6.3f}"
      f"  max={sig_p_mean.max():6.3f}  ratio={sig_p_mean.max()/sig_p_mean.min():.2f}x")
order = np.argsort(sig_p_mean)
print(f"  quietest: {[f'p{p+1}={sig_p_mean[p]:.2f}' for p in order[:5]]}")
print(f"  noisiest: {[f'p{p+1}={sig_p_mean[p]:.2f}' for p in order[-5:][::-1]]}")

print("\n=== variance decomposition ===")
print("between/within split a fixed total, so they trade off. Strong negative")
print("correlation = ridge (data can't separate them); two clusters = bimodal.")
print("tp and width are omitted: no between-patient level to decompose.")
r    = np.corrcoef(sigma_beta, sigma_beta_scan)[0, 1]
frac = sigma_beta_scan.mean() / (sigma_beta_scan.mean() + sigma_beta.mean())
print(f"beta   between={sigma_beta.mean():.4f}  within={sigma_beta_scan.mean():.4f}"
      f"  within/total={frac:.2f}  corr={r:+.3f}"
      f"  P(within<0.01)={(sigma_beta_scan < 0.01).mean():.3f}")

print("\n=== chain agreement on population means ===")
d = fit.draws_pd()
for v in ["mu_tp_log", "sigma_tp_scan", "mu_width_log", "sigma_width_scan",
          "mu_beta_log", "sigma_beta", "sigma_beta_scan", "mu_log_sigma"]:
    cm = d.groupby("chain__")[v].mean().values
    print(f"{v:17s} {np.round(cm, 4)}   within-chain sd {d[v].std():.4f}")

div = d["divergent__"].values.astype(bool)
if div.sum():
    print(f"\ndivergences by chain: {d.groupby('chain__')['divergent__'].sum().values}")
    for v in ["sigma_tp_scan", "sigma_width_scan", "sigma_beta", "sigma_beta_scan"]:
        print(f"  {v:18s} divergent {d[v][div].mean():.4f}  vs clean {d[v][~div].mean():.4f}")

print("\n=== residual autocorrelation ===")
print("the likelihood assumes independent residuals; BOLD noise is not.")
acs = []
for s in range(n_scans):
    m  = stan_data["scan"] == s + 1
    rs = stan_data["y"][m] - np.interp(stan_data["x"][m], x_plot, y_plot_mean[s])
    if rs.size > 10:
        acs.append(np.corrcoef(rs[:-1], rs[1:])[0, 1])
acs = np.array(acs)
print(f"lag-1: mean {acs.mean():.3f}  median {np.median(acs):.3f}  "
      f"[q10 {np.quantile(acs,.1):.3f}, q90 {np.quantile(acs,.9):.3f}]")
print(f"  -> effective n ~ {n_timepoints*(1-acs.mean())/(1+acs.mean()):.0f} "
      f"of {n_timepoints} timepoints per scan")

print("\n=== run effect (open question for Kay) ===")
b_s_mean = beta_s.mean(axis=0)
for r in np.unique(scan_run):
    m = scan_run == r
    print(f"  run {r}: n={m.sum():3d}  depth {b_s_mean[m].mean():6.3f}"
          f"  width {width_s_mean[m].mean():6.3f}  tp {tp_s_mean[m].mean():6.2f}")
r3 = np.array([b_s_mean[(scan_patient==p) & (scan_run==3)].mean() -
               b_s_mean[(scan_patient==p) & (scan_run!=3)].mean()
               for p in np.unique(scan_patient)])
print(f"  per-patient (run3 - others): mean {np.nanmean(r3):+.3f}"
      f"  frac negative {np.nanmean(r3 < 0):.2f}")

print("\n=== gender (multiplicative on amplitude) ===")
pct = 100 * (np.exp(beta_gender) - 1)
print(f"beta_gender  mean={beta_gender.mean():7.3f}  sd={beta_gender.std():6.3f}"
      f"  [{np.quantile(beta_gender,.025):7.3f}, {np.quantile(beta_gender,.975):7.3f}]")
print(f"  -> boys dip {pct.mean():+.1f}% deeper "
      f"[{np.quantile(pct,.025):+.1f}%, {np.quantile(pct,.975):+.1f}%], "
      f"P(>0)={(beta_gender > 0).mean():.3f}")
print(f"mem_gender   mean={mem_gender.mean():7.3f}  "
      f"[{np.quantile(mem_gender,.025):7.3f}, {np.quantile(mem_gender,.975):7.3f}] BOLD")
print(f"compare against sigma_beta={sigma_beta.mean():.3f}, not against zero")
print(f"{boys.sum()} boys / {girls.sum()} girls; scans "
      f"{(gender_scan > 0).sum()} / {(gender_scan < 0).sum()}")
print("NB mu_beta_log is the MIDPOINT of the groups, not the cohort average --")
print("   with unequal group sizes these differ. (open question for Kay)")
print("NB mem_gender is a difference of MEDIANS. The difference of means is")
print("   larger by exp(sigma_beta_scan^2/2). (open question for Kay)")
print("NB intervals here do NOT account for the residual autocorrelation above,")
print("   so they are substantially too narrow.")

print("\n=== PPI coverage ===")
cov_p, tot_p = np.zeros(n_patients), np.zeros(n_patients)
for s in range(n_scans):
    m = stan_data["scan"] == s + 1
    x_obs, y_obs = stan_data["x"][m], stan_data["y"][m]
    ok = np.sum((y_obs >= np.interp(x_obs, x_plot, ppi_lo[s])) &
                (y_obs <= np.interp(x_obs, x_plot, ppi_hi[s])))
    cov_p[scan_patient[s] - 1] += ok
    tot_p[scan_patient[s] - 1] += y_obs.size
cov_frac = cov_p / np.maximum(tot_p, 1)
quiet = sig_p_mean < np.median(sig_p_mean)
print(f"overall {cov_p.sum()/tot_p.sum():.3f} inside the 95% band  "
      f"(quiet half {cov_frac[quiet].mean():.3f}, noisy half {cov_frac[~quiet].mean():.3f})")

# %% ===== CELL 8: FIGURES =====
# 1 observed mean  2 noise  3 scan-level spreads  4 gender posteriors
# 5 dip by gender  6 example patients  7 population curve  8 residual
# 9 fitted vs empirical timing  10 fitted vs empirical depth

mean_signal = np.nanmean(Y, axis=1)
sd_signal   = np.nanstd(Y, axis=1)

mu_pop     = mu_curves.mean(axis=1)
sigma_pop  = sigma_scan.mean(axis=1) / np.sqrt(n_scans)
y_pop_pred = mu_pop + rng.normal(0.0, 1.0, mu_pop.shape) * sigma_pop[:, None]
pop_mean   = mu_pop.mean(axis=0)

PATIENTS_TO_SHOW = [1, 3, 6, 9]      # stan patient ids

# --- 1. observed mean signal ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t, mean_signal, color='steelblue', lw=1.5, label='Mean BOLD')
ax.fill_between(t, mean_signal - sd_signal, mean_signal + sd_signal,
                alpha=.3, color='steelblue', label='±1 SD')
ax.axhline(0, color='k', lw=.8, ls='--')
ax.set(xlim=[t[0], t[-1]], xlabel='Time (TRs)', ylabel='BOLD',
       title=f'Mean signal across {n_scans} scans ({n_patients} patients)')
ax.legend(); plt.tight_layout(); plt.show()

# --- 2. per-patient noise ---
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), dpi=110)
axes[0].hist(sig_p_mean, bins=25, color="C0", alpha=.8)
axes[0].axvline(sigma_median.mean(), color="k", ls="--", lw=1, label="median patient")
axes[0].set(xlabel="sigma_p (posterior mean)", ylabel="patients"); axes[0].legend(fontsize=8)
axes[1].hist(sigma_log_sigma, bins=60, color="C0", alpha=.8)
axes[1].set(xlabel="sigma_log_sigma", ylabel="posterior draws")
plt.tight_layout(); plt.show()

# --- 3. scan-level spreads. Only amplitude still has a between level, so only
#        it gets a scatter; tp and width are shown against their priors.
fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=110)
axes[0].scatter(sigma_beta, sigma_beta_scan, s=2, alpha=.15, color="C0")
axes[0].set(xlabel="sigma_beta (between)", ylabel="sigma_beta_scan (within)",
            title="amplitude")
axes[1].hist(sigma_width_scan, bins=60, color="C1", alpha=.8)
axes[1].axvline(0.3, color="k", ls="--", lw=1, label="prior scale")
axes[1].set(xlabel="sigma_width_scan", ylabel="posterior draws",
            title="width  (scan only; prior normal(0, 0.3))")
axes[1].legend(fontsize=8)
axes[2].hist(sigma_tp_scan, bins=60, color="C2", alpha=.8)
axes[2].axvline(0.1, color="k", ls="--", lw=1, label="prior scale")
axes[2].set(xlabel="sigma_tp_scan", ylabel="posterior draws",
            title="timing  (scan only; prior normal(0, 0.1))")
axes[2].legend(fontsize=8)
plt.tight_layout(); plt.show()

# --- 4. gender posteriors ---
fig, axes = plt.subplots(1, 2, figsize=(11, 3))
axes[0].hist(beta_gender, bins=60, color="C0", alpha=.8); axes[0].axvline(0, color="k", ls="--", lw=1)
axes[0].set(xlabel="beta_gender (log boy/girl ratio)", ylabel="posterior draws")
axes[1].hist(mem_gender, bins=60, color="C0", alpha=.8); axes[1].axvline(0, color="k", ls="--", lw=1)
axes[1].set(xlabel="mem_gender (boy - girl depth, BOLD)")
plt.tight_layout(); plt.show()

# --- 5. dip by gender, patient level (no observation noise) ---
# Patients now differ ONLY in depth, so these curves differ only in scale.
plt.figure(figsize=(8, 4), dpi=120)
for sel, lab, col in [(boys, "boys", "C0"), (girls, "girls", "C3")]:
    if sel.sum() == 0:
        continue
    dips = mu_curves_p[:, sel, :].mean(axis=1)
    plt.plot(x_plot, dips.mean(axis=0), color=col, lw=2, label=f"{lab} (n={sel.sum()})")
    plt.fill_between(x_plot, np.quantile(dips, .025, axis=0),
                     np.quantile(dips, .975, axis=0), color=col, alpha=.2)
plt.axhline(0, color="k", ls="--", lw=.8)
plt.axvline(tp_p_mean, color="grey", ls=":", lw=.8)
plt.xlabel("time (TR)"); plt.ylabel("BOLD"); plt.title("dip curve by gender (patient level)")
plt.legend(fontsize=8); plt.tight_layout(); plt.show()

# --- 6. example patients, each scan with its own fit ---
# No averaging of scans that peak at different times. x limited to the dip.
fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
for ax, stan_p in zip(axes.flat, PATIENTS_TO_SHOW):
    for k, s in enumerate(np.where(scan_patient == stan_p)[0]):
        m = stan_data["scan"] == s + 1
        ax.scatter(stan_data["x"][m], stan_data["y"][m], s=8, color=f"C{k}", alpha=.30)
        ax.plot(x_plot, y_plot_mean[s], color=f"C{k}", lw=1.3,
                label=f"run {scan_run[s]}: d={beta_s[:, s].mean():.2f}, "
                      f"tp={tp_s_mean[s]:.1f}, w={width_s_mean[s]:.2f}")
    ax.plot(x_plot, mu_curves_p[:, stan_p - 1, :].mean(axis=0),
            color="k", ls="--", lw=2, label="patient level (pop tp & width)")
    ax.axhline(0, color="k", ls="--", lw=.8)
    ax.set(xlim=(0, 32), xlabel="time (TR)", ylabel="BOLD",
           title=f"Patient {stan_p} ({(scan_patient == stan_p).sum()} scans, "
                 f"depth={np.exp(beta_log[:, stan_p-1]).mean():.2f}, "
                 f"sigma={sig_p_mean[stan_p-1]:.2f})")
    ax.legend(fontsize=7)
plt.tight_layout(); plt.show()

# --- 7. population mean curve ---
plt.figure(figsize=(9, 4))
plt.plot(x_plot, pop_mean, color="C0", lw=2, label="population mean (avg of scans)")
plt.fill_between(x_plot, np.quantile(y_pop_pred, .025, axis=0),
                 np.quantile(y_pop_pred, .975, axis=0), color="C0", alpha=.25, label="95% PPI")
plt.plot(t, mean_signal, color="k", lw=1, ls="--", alpha=.7, label="observed mean")
plt.axhline(0, color="k", ls="--", lw=.8)
plt.xlabel("time"); plt.ylabel("BOLD"); plt.title("Population mean curve")
plt.legend(); plt.tight_layout(); plt.show()

# --- 8. residual ---
# A single gamma decays monotonically and cannot produce the positive overshoot
# around TR 25-60; that residual is structural, and sigma_p absorbs it.
plt.figure(figsize=(9, 3))
plt.plot(t, mean_signal - np.interp(t, x_plot, pop_mean), color="C3", lw=1)
plt.axhline(0, color="k", ls="--", lw=.8)
plt.axvspan(25, 60, color="grey", alpha=.15, label="unmodelled overshoot")
plt.xlabel("time (TR)"); plt.ylabel("observed - fitted")
plt.legend(fontsize=8); plt.tight_layout(); plt.show()

# --- 9. fitted vs empirical timing, per scan ---
# Empirical trough restricted to TR 4-30: over the full window the global
# minimum finds slow baseline wander rather than the response.
from scipy.ndimage import uniform_filter1d

trough_t  = np.full(n_scans, np.nan)
emp_depth = np.full(n_scans, np.nan)
for s in range(n_scans):
    m = (stan_data["scan"] == s + 1) & (stan_data["x"] >= 4) & (stan_data["x"] <= 30)
    xs, ys = stan_data["x"][m], stan_data["y"][m]
    if xs.size >= 15:
        sm = uniform_filter1d(ys, size=5, mode="nearest")
        trough_t[s]  = xs[np.argmin(sm)]
        emp_depth[s] = -sm.min()

ok = np.isfinite(trough_t) & (trough_t > 5) & (trough_t < 29)   # drop window-edge hits

fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=110)
axes[0].hist(trough_t[ok], bins=20, color="grey", alpha=.6, label="empirical, per scan")
axes[0].hist(tp_s_mean, bins=25, color="C2", alpha=.7, label="fitted, per scan")
axes[0].axvline(tp_p_mean, color="k", ls="--", lw=1, label="population")
axes[0].set(xlabel="dip time (TR)", ylabel="count"); axes[0].legend(fontsize=8)

axes[1].scatter(trough_t[ok], tp_s_mean[ok], s=12, alpha=.5, color="C2")
lim = [min(trough_t[ok].min(), tp_s_mean.min()) - 1,
       max(trough_t[ok].max(), tp_s_mean.max()) + 1]
axes[1].plot(lim, lim, color="k", ls="--", lw=.8)
axes[1].set(xlim=lim, ylim=lim, xlabel="empirical trough (per scan)",
            ylabel="fitted tp_dip_s",
            title=f"corr {np.corrcoef(trough_t[ok], tp_s_mean[ok])[0,1]:+.2f}  "
                  f"(n={ok.sum()} of {n_scans})")
plt.tight_layout(); plt.show()

# --- 10. fitted vs empirical depth, per scan ---
# The direct check on the quantity the model exists to estimate. NB both
# measures read the same noisy trough, so some agreement is shared error.
b_s_mean = beta_s.mean(axis=0)
okd = np.isfinite(emp_depth)
plt.figure(figsize=(5.5, 5), dpi=120)
plt.scatter(emp_depth[okd], b_s_mean[okd], s=14, alpha=.5, color="C0")
lim = [0, np.nanpercentile(np.r_[emp_depth[okd], b_s_mean[okd]], 99) * 1.1]
plt.plot(lim, lim, "k--", lw=.8)
plt.xlim(lim); plt.ylim(lim)
plt.xlabel("empirical trough depth"); plt.ylabel("fitted beta_s")
plt.title(f"corr {np.corrcoef(emp_depth[okd], b_s_mean[okd])[0,1]:+.2f}")
plt.tight_layout(); plt.show()

# %% ===== CELL 9 (OPTIONAL): PRIOR CALIBRATION =====
# Reads tp / depth / width off the mean curve to set population priors. Run
# ONLY when the analysis window or cohort changes -- the numbers get hand-copied
# into the .stan file, so this is not part of the fit loop.

dip_i     = np.nanargmin(mean_signal)
depth_emp = abs(mean_signal[dip_i])
finite    = np.isfinite(mean_signal)
ms_f, t_f = mean_signal[finite], t[finite]
half      = mean_signal[dip_i] / 2.0
# restrict to finite points first: sign(NaN) propagates through diff and fakes
# a crossing at t=0, which inflates the FWHM
crossings = np.where(np.diff(np.sign(ms_f - half)))[0]

print(f"empirical dip at t={t[dip_i]:.0f} TR, depth {mean_signal[dip_i]:.4f}")
print(f"  -> tp_dip       ~ normal({t[dip_i]:.0f}, 3)")
print(f"  -> mu_beta_log  ~ normal({np.log(depth_emp):.2f}, 0.5)")
print(f"  half-depth {half:.3f}, crossings at TR {np.round(t_f[crossings], 1)}")
if len(crossings) >= 2:
    fwhm = t_f[crossings[-1]] - t_f[crossings[0]]
    print(f"  FWHM ~ {fwhm:.1f} TR")
    print(f"  -> mu_width_log ~ normal({np.log(fwhm / 2.35):.2f}, 0.3)   "
          f"(exp = {fwhm/2.35:.1f} TR)")
    if len(crossings) > 2:
        print("  NB >2 crossings -- FWHM uses first and last, may overstate width")
else:
    print("  could not locate two half-depth crossings; set width by eye")