# EWKinos / Sleptons "no-22%-uncertainty" results (2026-10-12)

Surrogate models trained on EWKino and Sleptons scan data generated **without the
standard 22% signal relative uncertainty** (`sig_rel_unc`), unlike every other
dataset in this project. Same architecture and training recipe as the main
`output_weights` sweep (see `results/results_summary/`), applied here to this
separate data family.

**Update (2026-11-19)**: a full-likelihood pyhf baseline (no signal
uncertainty) for validating a future model against is now available — see
`smodels_baseline_README.md`. It predates these two models (an independent
SModelS/`runValidation.py` run from 2026-06-30) and was not used to validate
`equal_weights`/`weight10` below; the EWKinos training data for those models is
also being extended (see the low-delta sweep around `nLL(mu=1)-nLL(mu=0) ~ 0`,
`sampling/ewkinos_lowdelta_sweep.sh`), so a from-scratch retrain against the
new baseline is the plan rather than validating these two as-is.

## Source data

- EWKinos: `1911.12606-nounc-EWKinos-merged.csv` (1,670,311 rows)
- Sleptons: `1911.12606-nounc-sleptons-merged.csv` (990,965 rows)

## Cleaning

- EWKinos: `correct_csv.py --nll 400 --delta 100`, then `--z 5` outlier cut.
  1,670,311 -> **1,534,532** rows (91.9% kept).
- Sleptons: `correct_csv.py --delta 150` only, no nLL cut.
  990,965 -> **990,965** rows (100% kept).

## Directory layout

```
EWKinos/
  scan_raw/       plot_scan.py (MLlikelihoods) on the raw merged data
  scan_cleaned/    plot_scan.py on the cleaned/trained-on data
  models/
    equal_weights/   training.output_weights=null
    weight10/         training.output_weights=[1,10,1,1]
  training_data/
    raw/              1911.12606-nounc-EWKinos-merged.csv+json
    cleaned/            the CSV+JSON both EWKino models were trained on

Sleptons/
  scan_raw/ , scan_cleaned/
  models/
    equal_weights/    training.output_weights=null
  training_data/
    raw/               1911.12606-nounc-sleptons-merged.csv+json
    cleaned/             the CSV+JSON the Sleptons model was trained on
```

Each `models/<name>/` folder has `model.onnx` (ONNX export, `model_author` =
"Rafal Maselek"), `config_0.yaml` (full training config), and
`olltrain_validation_plots/` (residual plots at several percentile cuts).

## Training recipe

256-wide/6-layer μP MuMLP, MAE-to-heteroscedastic loss, AdamW lr=2.018308e-05,
batchsize=2048, L2 reg (lambda=4.752559e-04), CosineAnnealingLR — identical to every
model in `results/results_summary/`. `output_weights` is the only thing that differs
between the two EWKino models.

## Training closure (ONNX export self-consistency, not accuracy)

| Model | Closure med. rel. err [exp,obs,expA,obsA] | Wall time |
|---|---|---|
| EWKinos equal_weights | 0.08%, 0.30%, 0.08%, 0.10% | 71.6 min |
| EWKinos weight10 | 0.18%, 0.14%, 0.18%, 0.18% | 71.8 min |
| Sleptons equal_weights | 0.06%, 0.11%, 0.06%, 0.07% | 50.3 min |

No SModelS obs/exp ratio numbers here since SModelS validation was not run for this
batch — only the ONNX export closure (`train/tools/validation_plots.py` residuals)
confirms the exported model matches the trained PyTorch model.
