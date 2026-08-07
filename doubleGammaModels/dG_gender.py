# %%
import numpy as np
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel

# %%
data = np.genfromtxt(
    '../data/main_a1avg.csv',
    delimiter=',',
    skip_header=1,
    missing_values='NA'
)                                    # (420, 65)

cov = np.genfromtxt('../data/MainStudy_withsexageanxiety.csv',
                    delimiter=',', skip_header=1)
gender_all = cov[:, 1]
ids = cov[:, 0]

# two-day patients have a blank second row; gender is the same person
for i in np.where(np.isnan(gender_all))[0]:
    match = (ids == ids[i]) & ~np.isnan(gender_all)
    assert match.sum() == 1, f"no unique source for patid {ids[i]}"
    gender_all[i] = gender_all[match][0]

assert len(gender_all) == data.shape[1], (len(gender_all), data.shape[1])
assert not np.isnan(gender_all).any()

# %% full cohort, first 100 timepoints
data = data[:100]
print(f"{data.shape[1]} patients, {data.shape[0]} timepoints")
print(f"{(gender_all > 0).sum()} boys, {(gender_all < 0).sum()} girls")

# %% subset to 10 for fast iteration
data = data[:, :10]
gender_all = gender_all[:10]
print(f"Using {data.shape[1]} patients, gender: {gender_all}")

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

# %% Compile the Stan model
model = CmdStanModel(stan_file="dG_gender.stan",
                     cpp_options={'STAN_THREADS': 'TRUE'})

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
    "gender":  gender_all,
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
def double_gamma(tt, a_bump, b_bump, a_dip, b_dip, c, t0=25.0):
    tt = np.asarray(tt, float)
    h = np.zeros_like(tt)
    pos = tt > 0
    ttb = tt - t0
    posb = ttb > 0
    lpk_b = (a_bump-1)*np.log((a_bump-1)/b_bump) - (a_bump-1) + a_bump*np.log(b_bump) - gammaln(a_bump)
    lpk_d = (a_dip -1)*np.log((a_dip -1)/b_dip ) - (a_dip -1) + a_dip *np.log(b_dip ) - gammaln(a_dip)
    g_b = np.zeros_like(tt)
    g_b[posb] = np.exp((a_bump-1)*np.log(ttb[posb]) - b_bump*ttb[posb]
                       + a_bump*np.log(b_bump) - gammaln(a_bump) - lpk_b)
    g_d = np.zeros_like(tt)
    g_d[pos] = np.exp((a_dip-1)*np.log(tt[pos]) - b_dip*tt[pos]
                      + a_dip*np.log(b_dip) - gammaln(a_dip) - lpk_d)
    h = g_b - c*g_d
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

# %% gender effect on amplitude
b_g  = fit.stan_variable("b_gender")
mu_b = fit.stan_variable("mu_beta")

print(f"b_gender  mean={b_g.mean():7.3f}  sd={b_g.std():6.3f}  "
      f"q2.5={np.quantile(b_g,0.025):7.3f}  q97.5={np.quantile(b_g,0.975):7.3f}")
print(f"P(b_gender > 0) = {(b_g > 0).mean():.3f}")
print(f"mu_beta   mean={mu_b.mean():7.3f}  sd={mu_b.std():6.3f}")

boys, girls = gender_all > 0, gender_all < 0
print(f"\nsubset: {boys.sum()} boys, {girls.sum()} girls")
print(f"implied boy amplitude:  {(mu_b + 0.5*b_g).mean():7.3f}")
print(f"implied girl amplitude: {(mu_b - 0.5*b_g).mean():7.3f}")

plt.figure(figsize=(7, 3))
plt.hist(b_g, bins=60, color="C0", alpha=0.8)
plt.axvline(0, color="k", ls="--", lw=1)
plt.xlabel("b_gender (boy - girl amplitude)")
plt.ylabel("posterior draws")
plt.tight_layout(); plt.show()

# %% dip and bump components by gender
tp_dip_d  = fit.stan_variable("tp_dip")
tp_bump_d = fit.stan_variable("tp_bump")
b_dip_d   = fit.stan_variable("b_dip")
b_bump_d  = fit.stan_variable("b_bump")
c_d       = fit.stan_variable("c")
beta_d    = fit.stan_variable("beta")

T0 = 25.0
x_g = np.linspace(0.5, t[-1], 400)

def unit_gamma(tt, a, b):
    out = np.zeros_like(tt)
    pos = tt > 0
    lpk = (a-1)*np.log((a-1)/b) - (a-1) + a*np.log(b) - gammaln(a)
    out[pos] = np.exp((a-1)*np.log(tt[pos]) - b*tt[pos] + a*np.log(b) - gammaln(a) - lpk)
    return out

def group_components(sel):
    n = tp_dip_d.shape[0]
    dips  = np.empty((n, x_g.size))
    bumps = np.empty((n, x_g.size))
    for d in range(n):
        acc_d = np.zeros_like(x_g); acc_b = np.zeros_like(x_g)
        for p in np.where(sel)[0]:
            a_d = max(tp_dip_d[d, p]  * b_dip_d[d],  0.01) + 1.0
            a_b = max(tp_bump_d[d, p] * b_bump_d[d], 0.01) + 1.0
            acc_d += beta_d[d, p] * c_d[d, p] * unit_gamma(x_g, a_d, b_dip_d[d])
            acc_b += beta_d[d, p] * unit_gamma(x_g - T0, a_b, b_bump_d[d])
        dips[d]  = acc_d / sel.sum()
        bumps[d] = acc_b / sel.sum()
    return dips, bumps

fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=120)
for sel, lab, col in [(boys, "boys", "C0"), (girls, "girls", "C3")]:
    if sel.sum() == 0:
        continue
    dips, bumps = group_components(sel)
    for ax, comp, title in [(axes[0], -dips, "dip component"),
                            (axes[1], bumps, "bump component"),
                            (axes[2], bumps - dips, "combined")]:
        m = comp.mean(axis=0)
        ax.plot(x_g, m, color=col, lw=2, label=f"{lab} (n={sel.sum()})")
        ax.fill_between(x_g, np.quantile(comp, 0.025, axis=0),
                        np.quantile(comp, 0.975, axis=0), color=col, alpha=0.2)
        ax.set_title(title)

for ax in axes:
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.axvline(T0, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("time (TR)"); ax.set_ylabel("BOLD")
    ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

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

