"""
Publish-time metadata normalizer for every ONNX model in models_onnx/.

Each file carries two kinds of metadata: keys inherited from the reference model
of the same ATLAS analysis (analysis/channels/yields/patchsets — properties of
the published workspace, true regardless of who generated the scan), and keys
that describe *our* trained network and *our* training data. This script keeps
the first kind, drops the parts of it that only described the reference scan,
and rebuilds the second kind from the run config embedded in the file:

  1. Classify keys as reference vs. ours — by name, not by a 'rafal::' prefix
     (export_onnx_from_run.py already strips that prefix, so prefix-based
     classification silently matched nothing and rebuilt neither set)
  2. Drop reference keys that describe the reference sampling run or its
     statistical-model configuration — false for our data (REFERENCE_KEYS_TO_DROP)
  3. Rebuild model_author/name/parameters/training_date/training_duration
  4. Recompute x_min/x_max/y_min/y_max from our training split (mmap, fast)
  5. Recompute the mu=0 baselines from our data — inheriting them offsets every
     absolute nLL a consumer reconstructs
  6. Derive 'preprocessing' from the run config rather than assuming the default
     pipeline (per-output/asinh runs do not use it)

Every key ends up written exactly once; a guard at the end enforces that.
"""

import os
import sys
import json
import datetime
import numpy as np
import onnx
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ONNX_DIR = os.path.join(BASE_DIR, "models_onnx")

MODEL_AUTHOR = "Joaquin Iturriaga"

# Reference-model keys that never survive publication. export_onnx_from_run.py
# imports this set, so the two stages cannot drift apart.
#
# Three groups:
#   - the reference pipeline's plumbing (paths, buffers, verbosity, parallelism)
#   - how the REFERENCE scan was sampled: point counts, seeds, the sampled yield
#     box, the signal-leakage and SR/CR/VR smearing knobs. We generate our own
#     scans, so every one of these describes a run that produced none of our data.
#   - how the reference likelihood was CONFIGURED (fit_bkg, removeCRsVRs,
#     remove_channels, merged). These are not sampling knobs — they change what
#     the nLL targets mean — so carrying them over asserts something about our
#     targets that we never checked.
# standardization_mean/std are the reference's plain linear standardization,
# superseded by our 'standardization' key.
REFERENCE_KEYS_TO_DROP = {
    # pipeline plumbing
    "starting_points", "standardization_mean", "standardization_std",
    "folder_name", "input_folder", "output_folder",
    "buffer_size", "keep_files", "bkg_unc_samples", "low_lim_samples",
    "processes", "spey_verbose_lvl",
    # the reference sampling run ("start method" is a typo duplicate of start_method)
    "start method", "start_method",
    "points", "total_points", "scans", "seed", "cluster", "scan_criterion",
    "filtering_applied", "model_version", "modified",
    "signal_leakage_CR", "signal_leakage_CR_spread", "signal_leakage_CR_sign",
    "signal_leakage_VR", "signal_leakage_VR_spread", "signal_leakage_VR_sign",
    "SR_sigma", "CR_sigma", "VR_sigma", "CR_center", "VR_center",
    "lower_limits", "upper_limits", "initial_lower_limits",
    "nLL_exp_max", "nLL_obs_max", "nLLA_exp_max", "nLLA_obs_max",
    # the reference likelihood configuration. NOT remove_channels/removeCRsVRs:
    # nnAdapter._parseMetaData prunes smYields by remove_channels and _getSROrder
    # derives the canonical input ordering from what is left, so dropping it
    # silently lengthens the expected yield vector (by 6 SRs on the 2018-16
    # EWkinos models, 3 CRs on 2018-04). It describes the input contract, which
    # our scans share, not the reference sampling.
    "fit_bkg", "merged",
    # the reference's own training details
    "optimizer", "batch_size", "early_stopping_used",
}

# Keys whose value must come from our run. Never inherited: either rebuilt here
# or carried over verbatim from what the exporter wrote (see OWN_KEYS_KEPT).
MODEL_OWNED_KEYS = {
    "x_min", "x_max", "y_min", "y_max",
    "model_author", "model_name", "model_parameters",
    "training_date", "training_duration",
    "standardization", "preprocessing", "run_config",
    "nLL_exp_mu0", "nLL_obs_mu0", "nLLA_exp_mu0", "nLLA_obs_mu0",
}

# Of those, the ones this script cannot recompute and copies through unchanged.
OWN_KEYS_KEPT = {"standardization", "run_config"}


def describe_preprocessing(cfg: dict) -> str:
    """Preprocessing description derived from the run config.

    Assuming the default log+standardize pipeline here would mislabel the
    per-output runs (e.g. the asinh models), whose nLL pipeline is a
    {"per_output": [...], "asinh_scale": s} mapping.
    """
    feat_pipeline = []
    for trafo_fns in (cfg.get("data", {}).get("trafos") or {}).values():
        if isinstance(trafo_fns, list):
            feat_pipeline.extend(trafo_fns)
    return json.dumps({
        "features_pipeline": feat_pipeline,
        "nLLs_pipeline": cfg.get("data", {}).get("nLL_trafos") or [],
        "note": (
            "log_w_negatives = log(|x|+1)*sign(x). "
            "Standardization parameters (mean, std per feature/output) "
            "are stored in the 'standardization' key."
        ),
    })


def compute_training_stats(cfg: dict) -> tuple[list, list, list, list, list]:
    """
    Replicate experiment.py data loading + split to get per-feature and per-output
    min/max over the training portion of the dataset, plus the mu=0 baselines.

    Uses mmap_mode + permutation to avoid loading the full array into RAM:
    np.random.permutation with seed 1234 produces the same row ordering as
    np.random.seed(1234); np.random.shuffle(data), so we can index directly.
    """
    dataset = cfg["data"]["dataset"]
    if isinstance(dataset, list):
        dataset = dataset[0]

    data_path = cfg["data"]["data_path"]
    if not os.path.isabs(data_path):
        data_path = os.path.join(BASE_DIR, data_path)

    train_frac = cfg["data"]["train_test_val"][0]
    subsample = cfg["data"].get("subsample")

    npy_path = os.path.join(data_path, f"{dataset}.npy")
    val_path = os.path.join(data_path, f"{dataset}_val.npy")
    test_path = os.path.join(data_path, f"{dataset}_test.npy")

    if os.path.exists(val_path):
        # Concatenated split files — load them individually with mmap then concat indices
        train_raw = np.load(npy_path, mmap_mode="r", allow_pickle=True)
        val_raw   = np.load(val_path, mmap_mode="r", allow_pickle=True)
        test_raw  = np.load(test_path, mmap_mode="r", allow_pickle=True)
        N_total = len(train_raw) + len(val_raw) + len(test_raw)
        np.random.seed(1234)
        perm = np.random.permutation(N_total)
        n_train = int(N_total * train_frac)
        if subsample is not None:
            n_train = min(int(subsample), n_train)
        train_idx = perm[:n_train]
        # Build the concatenated array only for the training rows we need
        # (split across the three files by offset)
        n0, n1 = len(train_raw), len(val_raw)
        rows = []
        for i in sorted(train_idx):
            if i < n0:
                rows.append(train_raw[i])
            elif i < n0 + n1:
                rows.append(val_raw[i - n0])
            else:
                rows.append(test_raw[i - n0 - n1])
        data_train = np.stack(rows)
    else:
        data_mmap = np.load(npy_path, mmap_mode="r", allow_pickle=True)
        N_total = len(data_mmap)
        np.random.seed(1234)
        perm = np.random.permutation(N_total)
        n_train = int(N_total * train_frac)
        if subsample is not None:
            n_train = min(int(subsample), n_train)
        train_idx = np.sort(perm[:n_train])   # sort for sequential disk reads
        data_train = data_mmap[train_idx]

    features = data_train[:, :-8]
    nLLs_raw = data_train[:, -8:].astype(float)
    # mu=0 baselines, as experiment.init_data computes them (median of the even
    # columns) and before the subtraction below overwrites the odd ones. Constant
    # per analysis, so the median over the training rows is the exact value.
    nLL_mu0 = np.median(nLLs_raw[:, 0::2], axis=0)
    for i in range(4):
        nLLs_raw[:, 2 * i + 1] -= nLLs_raw[:, 2 * i]
    nLLs = nLLs_raw[:, 1::2]   # 4 delta-nLL columns

    return (
        features.min(axis=0).tolist(),
        features.max(axis=0).tolist(),
        nLLs.min(axis=0).tolist(),
        nLLs.max(axis=0).tolist(),
        nLL_mu0.tolist(),
    )


def extract_model_info(cfg: dict) -> dict:
    """Derive model_name, model_parameters, training_date, training_duration from run_config."""
    net_cfg      = cfg.get("model", {}).get("net", {})
    training_cfg = cfg.get("training", {})

    arch = net_cfg.get("_target_", "").rsplit(".", 1)[-1]   # e.g. "MuMLP"
    hc   = net_cfg.get("hidden_channels", "?")
    hl   = net_cfg.get("hidden_layers", "?")
    act  = net_cfg.get("activation", "?")
    model_name = f"{arch}_c{hc}_l{hl}_{act}"

    model_parameters = json.dumps({
        "architecture":          arch,
        "hidden_channels":       hc,
        "hidden_layers":         hl,
        "activation":            act,
        "out_shape":             net_cfg.get("out_shape"),
        "loss":                  training_cfg.get("loss"),
        "optimizer":             training_cfg.get("optimizer"),
        "lr":                    training_cfg.get("lr"),
        "regularization":        training_cfg.get("regularization"),
        "regularization_lambda": training_cfg.get("regularization_lambda"),
        "iterations":            training_cfg.get("iterations"),
    })

    # Parse start time from run_name: "20260203_060932_MuMLP_8815"
    run_name = cfg.get("run_name", "")
    training_date = "unknown"
    start_dt = None
    if run_name:
        try:
            d, t = run_name.split("_")[:2]
            training_date = f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:4]}:{t[4:6]}"
            start_dt = datetime.datetime.strptime(training_date, "%Y-%m-%d %H:%M:%S")
        except (ValueError, IndexError):
            pass

    # Estimate training duration from model checkpoint mtime vs start time
    training_duration = "unknown"
    run_dir = cfg.get("run_dir", "")
    if run_dir:
        if not os.path.isabs(run_dir):
            run_dir = os.path.join(BASE_DIR, run_dir)
        model_ckpt = os.path.join(run_dir, "models", "model_run0.pt.gz")
        if os.path.exists(model_ckpt):
            end_dt = datetime.datetime.fromtimestamp(os.path.getmtime(model_ckpt))
            if start_dt:
                total_s = max(0, int((end_dt - start_dt).total_seconds()))
                h, rem  = divmod(total_s, 3600)
                m, s    = divmod(rem, 60)
                training_duration = f"{h} hours {m} minutes {s} seconds"
            if training_date == "unknown":
                # Not every run_config carries run_name; fall back to the
                # checkpoint mtime, which is what the exporter records.
                training_date = end_dt.strftime("%Y-%m-%d %H:%M:%S")

    return {
        "model_name":         model_name,
        "model_parameters":   model_parameters,
        "training_date":      training_date,
        "training_duration":  training_duration,
    }


def update_model(onnx_path: str) -> None:
    m = onnx.load(onnx_path)

    # Classify by NAME, not by the 'rafal::' prefix. The exporter strips that
    # prefix, so prefix-based classification found no reference keys at all: the
    # drop list never fired and every key was re-emitted alongside its rebuilt
    # twin. Files written before the prefix was dropped still carry it, hence
    # both branches. Later duplicates win — in those older files our value was
    # appended after the inherited one.
    reference, own = {}, {}
    for p in m.metadata_props:
        if p.key.startswith("rafal::"):
            reference[p.key[len("rafal::"):]] = p.value
        else:
            own[p.key] = p.value
    inherited = {**reference, **own}

    run_config_str = own.get("run_config", "")
    if not run_config_str:
        print(f"  WARNING: no run_config in {os.path.basename(onnx_path)}, skipping")
        return

    cfg = yaml.safe_load(run_config_str)

    print("  Computing training stats ...")
    x_min, x_max, y_min, y_max, nLL_mu0 = compute_training_stats(cfg)
    print(f"  x_min={x_min}")
    print(f"  x_max={x_max}")
    print(f"  y_min={y_min}")
    print(f"  y_max={y_max}")

    model_info = extract_model_info(cfg)
    print(f"  model_name={model_info['model_name']}")
    print(f"  training_date={model_info['training_date']}")
    print(f"  training_duration={model_info['training_duration']}")

    # mu=0 baselines. The exporter took its median over the whole dataset while
    # this runs over the training rows, so the two can disagree in the last
    # digits on datasets where mu0 is not bit-for-bit constant. Keep the stored
    # value when it agrees to within MU0_TOL and only overwrite a genuinely wrong
    # one — an inherited baseline is off by ~0.92 (spey's changed normalisation),
    # orders of magnitude outside the tolerance.
    MU0_TOL = 1e-3
    mu0_keys = ["nLL_exp_mu0", "nLL_obs_mu0", "nLLA_exp_mu0", "nLLA_obs_mu0"]
    mu0_final = {}
    for key, computed in zip(mu0_keys, nLL_mu0):
        stored = own.get(key, inherited.get(key))
        try:
            stored_val = float(str(stored).strip('"'))
        except (TypeError, ValueError):
            stored_val = None
        if stored_val is not None and abs(stored_val - computed) <= MU0_TOL:
            mu0_final[key] = stored
        else:
            mu0_final[key] = repr(float(computed))
            print(f"  NOTE: {key} {stored} -> {mu0_final[key]} (recomputed from our data)")

    # Everything the reference contributes, minus what only described its own run
    # and minus anything we own.
    rebuilt = {
        k: v for k, v in inherited.items()
        if k not in REFERENCE_KEYS_TO_DROP and k not in MODEL_OWNED_KEYS
    }

    for k in OWN_KEYS_KEPT:
        if k in own:
            rebuilt[k] = own[k]
        else:
            print(f"  WARNING: '{k}' missing — the model cannot be decoded without it")

    rebuilt["model_author"]      = MODEL_AUTHOR
    rebuilt["model_name"]        = model_info["model_name"]
    rebuilt["model_parameters"]  = model_info["model_parameters"]
    # Prefer what the exporter recorded: it read the checkpoint mtime at export
    # time, whereas re-reading it now picks up any later touch of the file (a
    # re-gzip moved one of these dates by five months). Recompute only to fill a
    # missing or unknown value.
    for key in ("training_date", "training_duration"):
        stored = own.get(key, "").strip('"')
        rebuilt[key] = stored if stored and stored != "unknown" else model_info[key]
    rebuilt["preprocessing"]     = describe_preprocessing(cfg)
    for key, val in [("x_min", x_min), ("x_max", x_max),
                     ("y_min", y_min), ("y_max", y_max)]:
        rebuilt[key] = json.dumps(val)
    rebuilt.update(mu0_final)

    # Clear all metadata and rebuild
    while len(m.metadata_props) > 0:
        m.metadata_props.pop()
    for k, v in rebuilt.items():
        p = m.metadata_props.add()
        p.key   = k
        p.value = str(v)

    seen = [p.key for p in m.metadata_props]
    dups = {k for k in seen if seen.count(k) > 1}
    assert not dups, f"duplicate metadata keys {sorted(dups)}"

    onnx.save(m, onnx_path)
    print(f"  Saved ({len(m.metadata_props)} metadata keys)")


def main() -> None:
    onnx_files = sorted(
        f for f in os.listdir(MODELS_ONNX_DIR)
        if f.endswith(".onnx") and not f.startswith("._")
    )
    if not onnx_files:
        print("No ONNX files found.")
        sys.exit(1)

    for fname in onnx_files:
        print(f"\n=== {fname} ===")
        update_model(os.path.join(MODELS_ONNX_DIR, fname))

    print("\nDone.")


if __name__ == "__main__":
    main()
