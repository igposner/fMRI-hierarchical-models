# fMRI Hierarchical Models

Bayesian hierarchical spline models of the fMRI BOLD response across a cohort
of ~65 patients. The BOLD response is expected to show an early transient dip
followed by a relatively flat regime, and the model is built to characterize
that trend rather than to enforce it.

## Modeling approach

The current model fits a cubic spline to each patient's BOLD time course, with
partial pooling across patients on the spline knot heights. Key architectural
decisions:

- **Knot positions are fixed data, not parameters.** They are set to
  `np.linspace(0, 60, 5)` and shared across all patients. Free knot positions
  cause the spline basis to reshape under the sampler, producing ill-conditioned
  posterior geometry (curved ridges, max-treedepth saturation, very low E-BFMI).
  Fixing the knots is the single most important constraint in this model.
- **No switchpoint.** Earlier versions parameterized the last knot as a
  per-patient switchpoint, which reintroduced sampler pathology. The knots are
  now fixed, with the last one at t=60 — the end of the modeled time window.
  t=60 is where the data ends and the fit stops; it reflects the observed extent
  of the response rather than imposing any boundary behavior.
- **Partial pooling on knot heights only.** Each patient's knot heights are drawn
  from a population distribution: `y_knot[i] ~ normal(mu_yknot, sigma_yknot)`.
  Observation noise (`sigma`) is currently estimated per patient, not pooled.

## Toolchain

- **Stan** (via CmdStanPy) for model definition and sampling
- **Python** (NumPy, CmdStanPy) for data prep and fit management
- Runs on a remote compute cluster (CHPC) via Slurm

## Repository structure

- `heirarchicalDataModels/` — the current working model lineage (`hiData*.stan`).
  `hiData5.stan` is the latest; `hData.py` / `hDataSmall.py` prepare data and
  manage fits. Lower-numbered `hiData*.stan` files are earlier versions kept for
  reference.
- `splineModelForData/` — spline model fitted to real data (pre-hierarchical).
- `firstSplineModel/`, `miscSplineWork/` — early spline experiments.
- `firstStanModels/` — initial Stan models (Posner OLS).
- `quick-job.sh` — launches an interactive Slurm session on the cluster.

## Data

Patient data is **not** tracked in this repository (see `.gitignore`). Input CSVs
live locally in `data/` on the cluster.

## Running a fit

1. Start a compute session: `bash quick-job.sh`
2. Activate the environment: `source venv/bin/activate`
3. Run the fit script, e.g. `python heirarchicalDataModels/hData.py`

Test on a small patient subset (`hDataSmall.py`) before scaling to the full
cohort.
