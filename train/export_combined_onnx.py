"""
Export two partial-model runs into a single 4-output ONNX.

Usage:
  python export_combined_onnx.py \\
      --run_a <run_dir_A> --indices_a 0 2 3 \\
      --run_b <run_dir_B> --indices_b 1 \\
      --rafal_onnx <rafal.onnx> \\
      --out <combined.onnx>

The resulting ONNX has the same input/output interface as every other model:
  inputs:  features, global_token
  outputs: nLLs  (shape [batch, 4] for MSE, [batch, 8] for HETEROSC)

Outputs are assembled from the two sub-models and placed at the correct indices
in the combined output vector.
"""

import argparse
import datetime
import gzip
import inspect
import json
import os

import numpy as np
import onnx
import torch
import torch.nn as nn
from omegaconf import OmegaConf, open_dict

from experiment import nLLsExperiment


# ---------------------------------------------------------------------------
# Helper: load a model from a run directory
# ---------------------------------------------------------------------------

def _load_run(run_dir, run_idx, device):
    cfg_path = os.path.join(run_dir, f"config_{run_idx}.yaml")
    if not os.path.exists(cfg_path):
        candidates = sorted(
            int(f[len("config_"):-len(".yaml")])
            for f in os.listdir(run_dir)
            if f.startswith("config_") and f.endswith(".yaml")
        )
        assert candidates, f"No config_*.yaml in {run_dir}"
        run_idx = candidates[0]
        cfg_path = os.path.join(run_dir, f"config_{run_idx}.yaml")

    cfg = OmegaConf.load(cfg_path)

    exp = nLLsExperiment(cfg, device)
    exp.warm_start = False
    exp.dtype = torch.float32
    exp.init_physics()
    exp.init_data()
    exp.init_model()

    gz = os.path.join(run_dir, "models", f"model_run{run_idx}.pt.gz")
    pt = os.path.join(run_dir, "models", f"model_run{run_idx}.pt")
    if os.path.exists(gz):
        with gzip.open(gz, "rb") as f:
            ckpt = torch.load(f, map_location=device)
    elif os.path.exists(pt):
        ckpt = torch.load(pt, map_location=device)
    else:
        raise FileNotFoundError(f"No checkpoint found in {run_dir}/models/")

    exp.model.load_state_dict(ckpt["model"])
    exp.model.eval()
    return exp


# ---------------------------------------------------------------------------
# Combined wrapper
# ---------------------------------------------------------------------------

class CombinedNLLModel(nn.Module):
    """Routes two partial models into a single 4-output (or 8-output) model."""

    def __init__(self, model_a, indices_a, model_b, indices_b,
                 target_indices_a=None, target_indices_b=None,
                 n_outputs=4, heterosc=False):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b
        self.indices_a = indices_a
        self.indices_b = indices_b
        self.n_outputs = n_outputs
        self.heterosc = heterosc

        # Map global output indices → local position in each model's output tensor.
        # A model trained on target_indices=[0,1,2,3] outputs means in that order;
        # one trained on target_indices=[1] outputs only one mean at local position 0.
        ta = target_indices_a if target_indices_a is not None else list(range(n_outputs))
        tb = target_indices_b if target_indices_b is not None else list(range(n_outputs))
        self.local_a = [ta.index(idx) for idx in indices_a]
        self.local_b = [tb.index(idx) for idx in indices_b]
        self.n_out_a = len(ta)
        self.n_out_b = len(tb)

    def forward(self, inputs, type_token, global_token, attn_mask=None):
        out_a = self.model_a(inputs, type_token, global_token, attn_mask)
        out_b = self.model_b(inputs, type_token, global_token, attn_mask)

        n = self.n_outputs
        out = torch.zeros(inputs.shape[0], 2 * n if self.heterosc else n,
                          dtype=out_a.dtype, device=out_a.device)

        if self.heterosc:
            means_a, sigs_a = out_a[:, :self.n_out_a], out_a[:, self.n_out_a:]
            means_b, sigs_b = out_b[:, :self.n_out_b], out_b[:, self.n_out_b:]
            for local_i, idx in zip(self.local_a, self.indices_a):
                out[:, idx] = means_a[:, local_i]
                out[:, n + idx] = sigs_a[:, local_i]
            for local_i, idx in zip(self.local_b, self.indices_b):
                out[:, idx] = means_b[:, local_i]
                out[:, n + idx] = sigs_b[:, local_i]
        else:
            for local_i, idx in zip(self.local_a, self.indices_a):
                out[:, idx] = out_a[:, local_i]
            for local_i, idx in zip(self.local_b, self.indices_b):
                out[:, idx] = out_b[:, local_i]

        return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(run_dir_a, indices_a, run_dir_b, indices_b, rafal_onnx_path, out_onnx, run_idx=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model A from {run_dir_a} (indices {indices_a})")
    exp_a = _load_run(run_dir_a, run_idx, device)
    print(f"Loading model B from {run_dir_b} (indices {indices_b})")
    exp_b = _load_run(run_dir_b, run_idx, device)

    cfg_a = exp_a.cfg
    heterosc = cfg_a.training.loss in ("HETEROSC", "MAE_TO_HETEROSC")

    def _target_indices(cfg, n_outputs=4):
        ti = cfg.data.get("target_indices")
        return list(OmegaConf.to_container(ti, resolve=True)) if ti else list(range(n_outputs))

    combined = CombinedNLLModel(
        model_a=exp_a.model,
        indices_a=indices_a,
        model_b=exp_b.model,
        indices_b=indices_b,
        target_indices_a=_target_indices(cfg_a),
        target_indices_b=_target_indices(exp_b.cfg),
        n_outputs=4,
        heterosc=heterosc,
    )
    combined.eval()

    n_features = cfg_a.model.net.n_features
    dummy_features = torch.zeros(1, n_features, device=device)
    dummy_token = torch.zeros(1, dtype=torch.long, device=device)

    export_kwargs = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        # torch>=2.6 defaults to the dynamo exporter; stay on the TorchScript
        # path the published models were produced with.
        export_kwargs["dynamo"] = False

    torch.onnx.export(
        combined,
        (dummy_features, dummy_token, dummy_token),
        "_tmp_combined.onnx",
        opset_version=17,
        input_names=["features", "type_token", "global_token"],
        output_names=["nLLs"],
        dynamic_axes={
            "features": {0: "batch"},
            "type_token": {0: "batch"},
            "global_token": {0: "batch"},
            "nLLs": {0: "batch"},
        },
        **export_kwargs,
    )

    # -----------------------------------------------------------------------
    # Metadata: reassemble full 4-output preprocessing stats
    # -----------------------------------------------------------------------
    # prepd_mean/std from each exp cover only their target indices, ordered by
    # the training target_indices list.  Use local_a/local_b (the same mapping
    # used in CombinedNLLModel) so the standardization params stay aligned with
    # the model outputs.
    full_mean = np.zeros(4)
    full_std  = np.ones(4)
    for local_i, idx in zip(combined.local_a, indices_a):
        m = exp_a.prepd_mean[0]
        s = exp_a.prepd_std[0]
        full_mean[idx] = m[local_i] if hasattr(m, '__len__') else float(m)
        full_std[idx]  = s[local_i] if hasattr(s, '__len__') else float(s)
    for local_i, idx in zip(combined.local_b, indices_b):
        m = exp_b.prepd_mean[0]
        s = exp_b.prepd_std[0]
        full_mean[idx] = m[local_i] if hasattr(m, '__len__') else float(m)
        full_std[idx]  = s[local_i] if hasattr(s, '__len__') else float(s)

    # x/y min-max from run A training split (features are the same for both)
    features_raw = exp_a.features[0]
    nLLs_a = exp_a.nLLs[0]  # shape [N, len(indices_a)]
    nLLs_b = exp_b.nLLs[0]  # shape [N, len(indices_b)]
    n_train_a = int(features_raw.shape[0] * cfg_a.data.train_test_val[0])
    x_train = features_raw[:n_train_a]

    # reassemble full 4-output nLLs for y min/max
    full_nLLs = np.zeros((n_train_a, 4))
    for i, idx in enumerate(indices_a):
        full_nLLs[:, idx] = nLLs_a[:n_train_a, i]
    for i, idx in enumerate(indices_b):
        full_nLLs[:, idx] = nLLs_b[:n_train_a, i]

    x_min = x_train.min(axis=0).tolist()
    x_max = x_train.max(axis=0).tolist()
    y_min = full_nLLs.min(axis=0).tolist()
    y_max = full_nLLs.max(axis=0).tolist()

    # -----------------------------------------------------------------------
    # Attach metadata
    # -----------------------------------------------------------------------
    rafal_onnx = onnx.load(rafal_onnx_path)
    rafal_metadata = {p.key: p.value for p in rafal_onnx.metadata_props}

    new_onnx = onnx.load("_tmp_combined.onnx")

    # Shared with export_onnx_from_run.py — see update_metadata.py, which owns
    # both sets. MODEL_OWNED_KEYS must cover EVERY key written below, or the file
    # gets it twice (inherited + ours) and consumers disagree depending on first-
    # vs last-match.
    from update_metadata import REFERENCE_KEYS_TO_DROP, MODEL_OWNED_KEYS

    for k, v in rafal_metadata.items():
        clean_key = k[len("rafal::"):] if k.startswith("rafal::") else k
        if clean_key in REFERENCE_KEYS_TO_DROP or clean_key in MODEL_OWNED_KEYS:
            continue
        if clean_key in ("run_config_a", "run_config_b"):
            continue
        p = new_onnx.metadata_props.add()
        p.key = clean_key
        p.value = v

    for key, val in [("x_min", x_min), ("x_max", x_max), ("y_min", y_min), ("y_max", y_max)]:
        p = new_onnx.metadata_props.add()
        p.key = key
        p.value = json.dumps(val)

    stats = {
        "nLLs_mean": [full_mean.tolist()],
        "nLLs_std":  [full_std.tolist()],
        "features_mean": [m.tolist() for m in exp_a.prepd_mean_features],
        "features_std":  [s.tolist() for s in exp_a.prepd_std_features],
        "model_a_indices": indices_a,
        "model_b_indices": indices_b,
    }
    bounds_a = exp_a.prepd_nll_bounds[0] if exp_a.prepd_nll_bounds else {}
    if bounds_a:
        full_lo = np.zeros(4)
        full_hi = np.ones(4)
        for i, idx in enumerate(indices_a):
            full_lo[idx] = bounds_a["lo"][i]
            full_hi[idx] = bounds_a["hi"][i]
        bounds_b = exp_b.prepd_nll_bounds[0] if exp_b.prepd_nll_bounds else {}
        if bounds_b:
            for i, idx in enumerate(indices_b):
                full_lo[idx] = bounds_b["lo"][i]
                full_hi[idx] = bounds_b["hi"][i]
        stats["nLLs_lo"] = full_lo.tolist()
        stats["nLLs_hi"] = full_hi.tolist()

    p = new_onnx.metadata_props.add()
    p.key = "standardization"
    p.value = json.dumps(stats)

    trafos_a = OmegaConf.to_container(cfg_a.data.get("trafos") or {}, resolve=True)
    feat_pipeline = []
    for fns in trafos_a.values():
        if isinstance(fns, list):
            feat_pipeline.extend(fns)
    # Keep the container as-is: a flat list, or the per-output mapping
    # {"per_output": [...], "asinh_scale": s}. list(dict) would collapse it to its
    # KEYS and silently discard the pipeline description.
    nll_pipeline = OmegaConf.to_container(cfg_a.data.get("nLL_trafos") or [], resolve=True)

    p = new_onnx.metadata_props.add()
    p.key = "preprocessing"
    p.value = json.dumps({
        "features_pipeline": feat_pipeline,
        "nLLs_pipeline": nll_pipeline,
        "note": (
            "Combined model: model_a covers output indices "
            f"{indices_a}, model_b covers {indices_b}. "
            "log_w_negatives = log(|x|+1)*sign(x). "
            "Standardization parameters are in the 'standardization' key."
        ),
    })

    p = new_onnx.metadata_props.add()
    p.key = "run_config"
    p.value = OmegaConf.to_yaml(cfg_a)
    p = new_onnx.metadata_props.add()
    p.key = "run_config_a"
    p.value = OmegaConf.to_yaml(cfg_a)
    p = new_onnx.metadata_props.add()
    p.key = "run_config_b"
    p.value = OmegaConf.to_yaml(exp_b.cfg)

    # mu=0 baselines, scattered into global output order like the bounds above.
    # These used to be inherited from the reference model, which offsets every
    # absolute nLL a consumer reconstructs as nLL_*_mu0 + delta.
    # Each experiment's nLL_mu0 is ordered by its own target_indices, so a global
    # output index maps to a local position exactly as CombinedNLLModel does it —
    # assuming position i holds global output i is only right for a full 4-output
    # run, and silently permutes the baselines otherwise.
    full_mu0 = np.zeros(4)
    have_mu0 = True
    for exp, indices, targets in ((exp_a, indices_a, _target_indices(cfg_a)),
                                  (exp_b, indices_b, _target_indices(exp_b.cfg))):
        if not getattr(exp, "nLL_mu0", None):
            have_mu0 = False
            break
        mu0 = np.asarray(exp.nLL_mu0[0]).ravel()
        for idx in indices:
            full_mu0[idx] = mu0[targets.index(idx)]
    if have_mu0:
        for key, val in zip(["nLL_exp_mu0", "nLL_obs_mu0",
                             "nLLA_exp_mu0", "nLLA_obs_mu0"], full_mu0):
            p = new_onnx.metadata_props.add()
            p.key = key
            p.value = repr(float(val))
    else:
        print("[WARN] no mu=0 baselines recovered — absolute nLLs will be unusable")

    # Identity of THIS combined model, not the reference's.
    net = cfg_a.model.net
    act = str(net.get("activation", "gelu"))
    arch = str(net._target_).rsplit(".", 1)[-1]
    ident = {
        "model_author": "Joaquin Iturriaga",
        "model_name": f"{arch}_c{net.get('hidden_channels')}"
                      f"_l{net.get('hidden_layers')}_{act}_combined",
        "model_parameters": json.dumps({
            "architecture": arch,
            "hidden_channels": net.get("hidden_channels"),
            "hidden_layers": net.get("hidden_layers"),
            "activation": act,
            "out_shape": 4,
            "combined": {"model_a_indices": indices_a, "model_b_indices": indices_b},
            "loss_a": str(cfg_a.training.get("loss")),
            "loss_b": str(exp_b.cfg.training.get("loss")),
        }),
        "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    for k, v in ident.items():
        p = new_onnx.metadata_props.add()
        p.key = k
        p.value = str(v)

    seen = [p.key for p in new_onnx.metadata_props]
    dups = {k for k in seen if seen.count(k) > 1}
    assert not dups, f"duplicate metadata keys {sorted(dups)} — add them to MODEL_OWNED_KEYS"

    onnx.save(new_onnx, out_onnx)
    os.remove("_tmp_combined.onnx")

    final_size_mb = os.path.getsize(out_onnx) / (1024 * 1024)
    print(f"Saved combined ONNX ({final_size_mb:.2f} MB) to: {out_onnx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_a", required=True)
    parser.add_argument("--indices_a", nargs="+", type=int, required=True)
    parser.add_argument("--run_b", required=True)
    parser.add_argument("--indices_b", nargs="+", type=int, required=True)
    parser.add_argument("--rafal_onnx", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run_idx", type=int, default=0)
    args = parser.parse_args()

    main(args.run_a, args.indices_a, args.run_b, args.indices_b,
         args.rafal_onnx, args.out, args.run_idx)
