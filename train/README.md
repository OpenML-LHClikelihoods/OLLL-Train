# likelihoods

Neural-network surrogates for **ATLAS SUSY-search likelihoods**. Given a point's
signal-region **yields**, an MLP predicts the four **negative-log-likelihood (nLL)
deltas** `[exp, obs, expA, obsA]` — the `μ=1 − μ=0` differences of the profiled
likelihood for {expected, observed, Asimov-expected, Asimov-observed}. Models are
trained per ATLAS analysis (datasets named by arXiv ID) and exported to **ONNX**
with all pre/post-processing carried as metadata, so a consumer only needs the
`.onnx` file and `nnAdapter.py`.

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install torch hydra-core omegaconf mup torch_geometric \
            onnx onnxruntime numpy scipy scikit-learn matplotlib
```

`IntrinsicDimDeep/` (intrinsic-dimension diagnostics used when
`training.get_ID=true`) is vendored in-tree; no extra setup needed.

## Quickstart

Training is driven by [Hydra](https://hydra.cc). The entry point is `run.py`; the
default config tree is under `config/` (`config_name="nLLs"`).

```bash
# train the μP MLP on one dataset
python run.py model=mup_mlp data.dataset=[1908.08215-400k-fluct20_] training.lr=2e-4

# override any config field from the CLI
python run.py model=mup_mlp training.loss=HETEROSC training.iterations=100000
```

A run executes `init_physics → init_data → init_model → train → evaluate → plot`
and writes checkpoints, predictions, and plots under `runs/<exp_name>/<run_name>/`.

## Data

Datasets are `.npy` files in `data/`, named `<arXivID>-<channel>-<size>-<tag>.npy`.
Each row is `[ yields… | 8 nLL columns ]`; the 8 nLL columns are baseline-
subtracted into the **4 delta targets** `[exp, obs, expA, obsA]`. Inputs and
targets are preprocessed with a signed-log (`sign(x)·log1p(|x|)`) followed by
standardization.

## Deployment (ONNX)

```bash
# export a trained run to ONNX (preprocessing stored as metadata)
python export_onnx_from_run.py <run_dir> <reference_rafal_onnx> <out.onnx>
```

Inference is done through `nnAdapter.py` (`NNAdapter`), which reapplies the
metadata-stored preprocessing, runs onnxruntime, and reconstructs the absolute nLL
as `nll(μ=1) = nLL_mu0 + delta`.

## Layout

| Path | What |
|------|------|
| `run.py` | Hydra entry point |
| `experiment.py` | likelihoods experiment: data, loss, eval, plots |
| `base_experiment.py` | generic train loop, optimizer/scheduler, μP, checkpointing |
| `models/` | model implementations (μP MLP is the maintained one) |
| `wrappers.py`, `preprocessing.py`, `losses.py`, `dataset.py` | model wrapper, preprocessing, losses, dataset |
| `config/` | Hydra configs |
| `nnAdapter.py`, `export_*.py`, `update_metadata.py` | ONNX export + inference |
| `IntrinsicDimDeep/` | intrinsic-dimension diagnostics |
