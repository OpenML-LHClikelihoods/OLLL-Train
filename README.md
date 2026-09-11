# OLLL-Train

All you need to produce a new ML-likelihood: neural-network surrogates for
ATLAS SUSY-search negative-log-likelihoods (nLLs).

Everything lives in two directories:

- [`train/`](train/README.md) — the training pipeline. Turn sampled data into a
  model, then export it to ONNX. Entry point: `python run.py` (Hydra).
- [`sampling/`](sampling/README.md) — MCMC data generation. Sample
  `(yields → nLL)` training rows from ATLAS `pyhf` workspaces. Entry point:
  `python sample.py parameters.yaml`.

Workflow: `sampling/ → data/ → train/ → train/runs/... → ONNX → nnAdapter.py`.

- `data/` — raw data tarballs, one per ATLAS analysis (keyed by arXiv ID).
- `train/README.md` — full training + ONNX export documentation.
- `sampling/README.md` — full sampling documentation (cards, scan limits, parameters).