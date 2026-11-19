# Full-likelihood pyhf baseline, no signal uncertainty (ATLAS-SUSY-2018-16)

A ground-truth reference to validate a trained no-uncertainty surrogate
against, once one exists — computed independently of this project's own
sampling pipeline (`sampling/`), via SModelS's own pyhf interface
(`smodels-utils/validation/runValidation.py`), not `sample.py`/`likelihood.py`.
Two independent implementations of the same physics agreeing is a much
stronger check than re-running our own code.

## Where this came from

`/projects/smodels-database/13TeV/ATLAS/ATLAS-SUSY-2018-16-orig/validation_nouncert/`
— generated 2026-06-30 by `runValidation.py` with signal uncertainty forced to
0 (see that directory's own `README`: *"this is the validation folder for the
case of signal uncertainties == 0"*). It decomposes the official SModelS mass
grid for each topology through the full-likelihood pyhf statistical model and
records, per surviving point, the same quantities our own pipeline computes:

| This baseline's field | Our pipeline's analogue |
|---|---|
| `nll_SM`  | `nLL_obs_mu0` (background-only, constant per topology) |
| `nll_min` | `nLL_obs_max` (profiled best-fit nLL) |
| `nll`     | `nLL_obs_mu1`-like (nLL at the point's full predicted signal) |
| `signal` / `UL` | predicted cross-section / pyhf 95% CL UL on it — ratio is the r-value |

**Confirmed apples-to-apples**: `Sleptons_bkgonly.json` here is byte-identical
to `data/1911.12606/Sleptons_bkgonly.json`. `EWKinos_bkgonly_SR2lOnly.json`
(44 channels) is the exact same statistical model as
`data/1911.12606/EWKinos_bkgonly.json` (50 channels) with the 6
`SR_eMLL*_Onelep1track_cuts` channels removed — i.e. identical to what
`remove_channels` in `parameters.yaml`/the `1911.12606*.yaml` cards already
produces (checked field-by-field: same `channels[].samples`, same
`observations`, same `measurements`; verified 2026-11-19).

## Files

- `EWKinos/smodels_baseline/EWKinos_nounc_pyhf_baseline.csv` — 195 valid
  points (of 234 in the full TChiWZoff grid; the rest fail SModelS
  decomposition, unrelated to any of this).
- `Sleptons/smodels_baseline/Sleptons_nounc_pyhf_baseline.csv` — 109 valid
  points (of 844 in the full TSlepSlep grid).
- `*_x_x-y_combined.py` next to each CSV — the raw `runValidation.py` output
  (includes the decomposition-failed points too, with an `"error"` key, for
  anyone who wants the full picture).
- `SModelS_ExclusionLines.json` — the resulting full-likelihood exclusion
  contour in the mass plane for both topologies, from the same run.

Columns in the CSVs: `x`, `y` (mass-plane axes, GeV), `slhafile`, `StatModel`,
`dataset`, `signal`, `UL`, `eUL`, `eUL_m1`, `eUL_p1`, `nll_SM`, `nll`,
`nll_min`, `condition`, `kfactor`, `t` (wall time, seconds).

## Using this later

Once a model is trained on the no-uncertainty data (`train/data/1911.12606-
nounc-*`), validating against this baseline needs the per-bin **signal
yields** at these same `(x, y)` points to feed the surrogate — this baseline
only has the aggregate `signal`/`UL`/`nll*` numbers, not per-bin yields. Join
on `slhafile`/mass point against the official benchmark-grid yields already
in the repo:

- `sampling/starting_points/ewkinos_grid_and_patchset_yields.csv`
- `sampling/starting_points/sleptons_grid_and_patchset_yields.csv`

then compare the surrogate's predicted `nLL_obs_mu1 - nLL_obs_mu0` (or the
full r-value, reconstructing `UL` from the surrogate the way `plotRatio.py`
does for the with-uncertainty models in `results/smodels_validation/`) against
this baseline's `nll - nll_SM`, the same way the existing `ratios_*.png` /
`ratios.txt` comparisons were built for the 22%-uncertainty models.
