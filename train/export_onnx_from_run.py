import os
import gzip
import inspect
import json
import datetime
import numpy as np
import torch
import onnx

from omegaconf import OmegaConf
from experiment import nLLsExperiment
from misc import get_device


def load_model_gz(path, device):
    with gzip.open(path, "rb") as f:
        return torch.load(f, map_location=device)


class ExportAdapter(torch.nn.Module):
    """(features, global_token) -> nLLs, the published ONNX interface.

    The trained wrapper's forward is (inputs, type_token, global_token, attn_mask)
    and ignores everything but `inputs`. The old TorchScript exporter tolerated
    being handed fewer args than the signature declares; torch>=2.6 exports via
    torch.export by default and binds arguments strictly, so it fails with
    "missing a required argument: 'global_token'". Adapt the signature here
    instead of feeding a third input the consumers don't send. (global_token is
    unused, so the tracer prunes it and the graph keeps `features` alone, exactly
    as in the published models_onnx/ files.)
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features, global_token):
        return self.model(features, global_token, global_token)


def main(run_dir, rafal_onnx_path, out_onnx=None, run_idx=0):
    device = get_device()
    
    # Set default output path in the run's models folder
    if out_onnx is None:
        out_onnx = os.path.join(run_dir, "models", "model_with_metadata.onnx")
    
    # ------------------------------------------------------------------
    # 1. Load config from existing run
    # ------------------------------------------------------------------
    cfg_path = os.path.join(run_dir, f"config_{run_idx}.yaml")
    if not os.path.exists(cfg_path):
        # Retraining runs start at index 1+; find the lowest available config
        candidates = sorted(
            int(f[len("config_"):-len(".yaml")])
            for f in os.listdir(run_dir)
            if f.startswith("config_") and f.endswith(".yaml")
        )
        assert candidates, f"No config_*.yaml found in {run_dir}"
        run_idx = candidates[0]
        cfg_path = os.path.join(run_dir, f"config_{run_idx}.yaml")
        print(f"[INFO] config_0.yaml not found; using config_{run_idx}.yaml")
    cfg = OmegaConf.load(cfg_path)
    
    # ------------------------------------------------------------------
    # 2. Recreate experiment JUST enough to get preprocessing + model
    # ------------------------------------------------------------------
    exp = nLLsExperiment(cfg, device)
    # minimal attributes normally set by BaseExperiment._init()
    exp.warm_start = False
    exp.dtype = torch.float32
    exp.init_physics()
    exp.init_data()
    # Same order as BaseExperiment.full_run (init_data -> _init_dataloader ->
    # init_model). _init_dataloader is what derives the training-target range,
    # and init_model needs it to pin the output clamp; skipping it exported a
    # model whose clamp constants were still +-inf, i.e. silently unbounded.
    exp._init_dataloader()
    exp.init_model()
    
    # ------------------------------------------------------------------
    # 3. Load trained weights
    # ------------------------------------------------------------------
    model_path_gz = os.path.join(run_dir, "models", f"model_run{run_idx}.pt.gz")
    model_path_pt = os.path.join(run_dir, "models", f"model_run{run_idx}.pt")
    if os.path.exists(model_path_gz):
        model_path = model_path_gz
        checkpoint = load_model_gz(model_path, device)
    elif os.path.exists(model_path_pt):
        model_path = model_path_pt
        checkpoint = torch.load(model_path, map_location=device)
    else:
        raise AssertionError(f"Missing {model_path_gz} and {model_path_pt}")
    state_dict = checkpoint["model"]
    exp.model.load_state_dict(state_dict)
    exp.model.eval()
    
    # ------------------------------------------------------------------
    # 4. Export ONNX
    # ------------------------------------------------------------------
    batch_size = 1
    global_token = torch.zeros(batch_size, dtype=torch.long, device=device)
    dummy_features = torch.zeros(
        1,
        cfg.model.net.n_features,
        device=device,
    )
    dummy_global_token = torch.zeros(
        1,
        dtype=torch.long,
        device=device,
    )
    
    export_kwargs = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        # torch>=2.6 defaults to the dynamo exporter. Stay on the TorchScript
        # path the published models_onnx/ files were produced with.
        export_kwargs["dynamo"] = False

    torch.onnx.export(
        ExportAdapter(exp.model),
        (dummy_features, dummy_global_token),
        "_tmp.onnx",
        opset_version=17,
        input_names=["features", "global_token"],
        output_names=["nLLs"],
        dynamic_axes={
            "features": {0: "batch"},
            "global_token": {0: "batch"},
            "nLLs": {0: "batch"},
        },
        **export_kwargs,
    )
    
    # ------------------------------------------------------------------
    # 5. Read Rafal's ONNX metadata
    # ------------------------------------------------------------------
    rafal_onnx = onnx.load(rafal_onnx_path)
    rafal_metadata = {p.key: p.value for p in rafal_onnx.metadata_props}

    # Print Rafal's ONNX size
    rafal_size_mb = os.path.getsize(rafal_onnx_path) / (1024 * 1024)
    print(f"Rafal's ONNX size: {rafal_size_mb:.2f} MB")

    # ------------------------------------------------------------------
    # 5b. Compute x/y min/max from training split
    # ------------------------------------------------------------------
    train_frac = cfg.data.train_test_val[0]
    subsample = cfg.data.get("subsample")
    features_raw = exp.features[0]   # raw (unpreprocessed) features
    nLLs_raw = exp.nLLs[0]           # raw delta-nLLs (4 differences, unpreprocessed)
    N = features_raw.shape[0]
    n_train = int(N * train_frac)
    if subsample is not None:
        n_train = min(int(subsample), n_train)
    x_train = features_raw[:n_train]
    y_train = nLLs_raw[:n_train]
    x_min = x_train.min(axis=0).tolist()
    x_max = x_train.max(axis=0).tolist()
    y_min = y_train.min(axis=0).tolist()
    y_max = y_train.max(axis=0).tolist()

    # ------------------------------------------------------------------
    # 6. Attach metadata to new ONNX
    # ------------------------------------------------------------------
    new_onnx = onnx.load("_tmp.onnx")

    # Which reference keys are dropped, and which ones only we may write, is
    # defined once in update_metadata.py (the publish-time normalizer) and
    # imported here — the two stages disagreeing is what let stale keys through.
    #
    # MODEL_OWNED_KEYS must cover EVERY key this script writes below. Anything
    # written but not listed there ends up in the file *twice* — once inherited
    # from the reference, once ours — and which one a consumer sees depends on
    # whether it parses first- or last-match. That silently mismatched
    # `standardization`/`preprocessing` against the wrong pipeline on real
    # published models; first-match decoding gave 34-47% relative error where the
    # correct spec gives <1%. The mu=0 baselines are in that set for the same
    # reason: consumers rebuild the physical value as nLL_*_mu0 + delta, so
    # inheriting the reference's baselines offsets every absolute nLL. The
    # 2018-16 scans sit +0.918939 (=0.5*ln(2*pi)) above the older data because
    # spey's normalisation changed.
    from update_metadata import REFERENCE_KEYS_TO_DROP, MODEL_OWNED_KEYS

    # Rafal's metadata — strip prefix, skip dropped/replaced keys
    for k, v in rafal_metadata.items():
        clean_key = k[len("rafal::"):] if k.startswith("rafal::") else k
        if clean_key in REFERENCE_KEYS_TO_DROP or clean_key in MODEL_OWNED_KEYS:
            continue
        p = new_onnx.metadata_props.add()
        p.key = clean_key
        p.value = v

    # Computed min/max from Joaquin's training data
    for key, val in [("x_min", x_min), ("x_max", x_max), ("y_min", y_min), ("y_max", y_max)]:
        p = new_onnx.metadata_props.add()
        p.key = key
        p.value = json.dumps(val)

    # Run config
    p = new_onnx.metadata_props.add()
    p.key = "run_config"
    p.value = OmegaConf.to_yaml(cfg)

    # Standardization stats
    stats = {
        "nLLs_mean": [m.tolist() for m in exp.prepd_mean],
        "nLLs_std": [s.tolist() for s in exp.prepd_std],
        "features_mean": [m.tolist() for m in exp.prepd_mean_features],
        "features_std": [s.tolist() for s in exp.prepd_std_features],
    }
    bounds = exp.prepd_nll_bounds[0] if exp.prepd_nll_bounds else {}
    if "lo" in bounds:  # logit_bounded only; per-output spec carries no lo/hi
        stats["nLLs_lo"] = np.asarray(bounds["lo"]).tolist()
        stats["nLLs_hi"] = np.asarray(bounds["hi"]).tolist()
    p = new_onnx.metadata_props.add()
    p.key = "standardization"
    p.value = json.dumps(stats)

    # Preprocessing pipeline description (derived from the actual training config)
    trafos = OmegaConf.to_container(cfg.data.get("trafos") or {}, resolve=True)
    feat_pipeline = []
    for trafo_fns in trafos.values():
        if isinstance(trafo_fns, list):
            feat_pipeline.extend(trafo_fns)
    # keep the container as-is: a flat list, or the per-output mapping
    # {"per_output": [...], "asinh_scale": s} (list(dict) would drop to keys)
    nll_pipeline = OmegaConf.to_container(cfg.data.get("nLL_trafos") or [], resolve=True)

    p = new_onnx.metadata_props.add()
    p.key = "preprocessing"
    p.value = json.dumps({
        "features_pipeline": feat_pipeline,
        "nLLs_pipeline": nll_pipeline,
        "note": (
            "log_w_negatives = log(|x|+1)*sign(x). "
            "Standardization parameters (mean, std per feature/output) "
            "are stored in the 'standardization' key."
        ),
    })

    # Model identity, derived from THIS run's config rather than inherited. Without
    # this the file keeps the reference model's architecture/optimizer/batchsize,
    # which describes a different network entirely.
    net = cfg.model.net
    act = str(net.get("activation", "gelu"))
    model_params = {
        "architecture": str(net._target_).rsplit(".", 1)[-1],
        "hidden_channels": net.get("hidden_channels"),
        "hidden_layers": net.get("hidden_layers"),
        "activation": act,
        "out_shape": 4,
        "loss": str(cfg.training.get("loss")),
        "optimizer": str(cfg.training.get("optimizer")),
        "batchsize": cfg.training.get("batchsize"),
        "lr": cfg.training.get("lr"),
        "regularization": str(cfg.training.get("regularization")),
        "regularization_lambda": cfg.training.get("regularization_lambda"),
        "iterations": cfg.training.get("iterations"),
    }
    if isinstance(nll_pipeline, dict) and "asinh_scale" in nll_pipeline:
        model_params["asinh_scale"] = nll_pipeline["asinh_scale"]
    # mu=0 baselines measured from THIS run's data (init_data keeps them before
    # subtracting). Constant per analysis, so the median is exact.
    if getattr(exp, "nLL_mu0", None):
        base = np.asarray(exp.nLL_mu0[0]).ravel()
        for key, val in zip(["nLL_exp_mu0", "nLL_obs_mu0",
                             "nLLA_exp_mu0", "nLLA_obs_mu0"], base):
            p = new_onnx.metadata_props.add()
            p.key = key
            p.value = repr(float(val))

    ident = {
        "model_author": "Joaquin Iturriaga",
        "model_name": f"{model_params['architecture']}_c{net.get('hidden_channels')}"
                      f"_l{net.get('hidden_layers')}_{act}",
        "model_parameters": json.dumps(model_params),
        "training_date": datetime.datetime.fromtimestamp(
            os.path.getmtime(model_path)).strftime("%Y-%m-%d %H:%M:%S"),
    }
    for k, v in ident.items():
        p = new_onnx.metadata_props.add()
        p.key = k
        p.value = str(v)

    # Guard: the duplication bug above is easy to reintroduce by adding a write
    # without updating MODEL_OWNED_KEYS, and it fails silently. Fail loudly here.
    seen = [p.key for p in new_onnx.metadata_props]
    dups = {k for k in seen if seen.count(k) > 1}
    assert not dups, f"duplicate metadata keys {sorted(dups)} — add them to MODEL_OWNED_KEYS"

    onnx.save(new_onnx, out_onnx)
    os.remove("_tmp.onnx")
    
    # Print final model size
    final_size_mb = os.path.getsize(out_onnx) / (1024 * 1024)
    print(f"Final ONNX size: {final_size_mb:.2f} MB")
    print(f"Saved to: {out_onnx}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python export_onnx_from_run.py <run_dir> <reference_onnx> [run_idx]")
        print()
        print("  run_dir        runs/<exp>/<run>/ — must still hold config_<idx>.yaml")
        print("                 and models/model_run<idx>.pt(.gz)")
        print("  reference_onnx one of Rafal's ONNX models for the SAME analysis; it is")
        print("                 read only as a metadata donor (analysis/channels/yields/")
        print("                 patchsets/license). Nothing of its network is used.")
        print("  run_idx        checkpoint index, default 0")
        print()
        print("Writes <run_dir>/models/model_with_metadata.onnx")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], run_idx=int(sys.argv[3]) if len(sys.argv) > 3 else 0)
