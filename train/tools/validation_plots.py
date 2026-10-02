"""
validation_plots.py

For each ONNX model in models_onnx/, loads the test split of the training data,
runs the model, inverts preprocessing, and generates a 2x2 residual plot:

    (Δ_pred - Δ_truth)  vs  Δ_truth

for the four delta-nLL outputs: Expected, Observed, Expected Asimov, Observed Asimov.

Usage:
    python validation_plots.py [--max-events N] [--out-dir DIR] [model1.onnx ...]

If no model files are given, all *.onnx in models_onnx/ are processed.
"""

import os
import sys
import json
import argparse
import numpy as np
import onnx
import onnxruntime as ort
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
import yaml

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (script lives in tools/)
MODELS_ONNX   = os.path.join(BASE_DIR, "models_onnx")

SUBPLOT_TITLES = ["Expected", "Observed", "Expected Asimov", "Observed Asimov"]

# Max test-set events used for plotting (keeps plots fast; set None to use all)
DEFAULT_MAX_EVENTS = None


# ---------------------------------------------------------------------------
# Preprocessing helpers  (must match experiment.py exactly)
# ---------------------------------------------------------------------------

def log_w_negatives(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log(np.abs(x) + 1.0)

def inv_log_w_negatives(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * (np.exp(np.abs(x)) - 1.0)

def preprocess_features(features: np.ndarray, mean, std, pipeline=None) -> np.ndarray:
    """Apply the feature preprocessing pipeline used during training.

    pipeline: ordered list of transform names, e.g. ["standardization"] or
              ["log_w_negatives", "standardization"].  Extracted from the
              run_config stored in the ONNX metadata (cfg['data']['trafos']).
              Defaults to the old hard-coded behaviour if None.
    """
    if pipeline is None:
        pipeline = ["log_w_negatives", "standardization"]
    x = features.astype(np.float64)
    for step in pipeline:
        if step == "log_w_negatives":
            x = log_w_negatives(x)
        elif step == "standardization":
            x = (x - np.array(mean)) / np.array(std)
        # Unknown steps are silently skipped; add cases here as needed.
    return x.astype(np.float32)

def inverse_preprocess_nLLs(prepd: np.ndarray, mean, std,
                             pipeline=None, nll_bounds=None) -> np.ndarray:
    if pipeline is None:
        pipeline = ["log_w_negatives", "standardization"]
    # per-output spec: {"per_output": [[log, standardization], [asinh, ...], ...],
    # "asinh_scale": s} — each output column has its own pipeline.
    if isinstance(pipeline, dict) and "per_output" in pipeline:
        per_output = [list(p) for p in pipeline["per_output"]]
        scale = float(pipeline.get("asinh_scale", 1.0))
        mean = np.asarray(mean); std = np.asarray(std)
        x = prepd.astype(np.float64).copy()
        for c in range(x.shape[1]):
            col = x[:, c]
            for step in reversed(per_output[c]):
                if step == "standardization":
                    col = col * std[c] + mean[c]
                elif step == "asinh":
                    col = scale * np.sinh(col)
                elif step == "log":
                    col = np.exp(col)
                elif step == "log_w_negatives":
                    col = inv_log_w_negatives(col)
            x[:, c] = col
        return x
    nll_bounds = nll_bounds or {}
    x = prepd.astype(np.float64)
    for step in reversed(pipeline):
        if step == "standardization":
            x = x * np.array(std) + np.array(mean)
        elif step == "log_w_negatives":
            x = inv_log_w_negatives(x)
        elif step == "logit_bounded":
            lo = np.array(nll_bounds["lo"])
            hi = np.array(nll_bounds["hi"])
            x = lo + (hi - lo) / (1.0 + np.exp(-x))
    return x


# ---------------------------------------------------------------------------
# Data loading  (mmap + sorted indices — avoids loading the full file)
# ---------------------------------------------------------------------------

def load_test_split(cfg: dict, standardization: dict, max_events: int | None):
    """
    Replicates experiment.py init_data() + _init_dataloader() for the test split.
    Returns:
        features_prepd  (N, n_features)  float32  — preprocessed, ready for ONNX
        nLLs_raw        (N, 4)           float64  — raw delta-nLLs (ground truth)
    """
    dataset   = cfg["data"]["dataset"]
    if isinstance(dataset, list):
        dataset = dataset[0]

    data_path = cfg["data"]["data_path"]
    if not os.path.isabs(data_path):
        data_path = os.path.join(BASE_DIR, data_path)

    train_frac, _, val_frac = cfg["data"]["train_test_val"]
    subsample = cfg["data"].get("subsample")

    npy_path  = os.path.join(data_path, f"{dataset}.npy")
    val_path  = os.path.join(data_path, f"{dataset}_val.npy")
    test_path = os.path.join(data_path, f"{dataset}_test.npy")

    if os.path.exists(val_path):
        # <dataset>{,_val,_test}.npy is a FIXED on-disk split: _test.npy IS the
        # test set. Matches experiment.init_data, which honours the split rather
        # than concatenating and reshuffling (the train block may oversample a
        # region, so a reshuffle would leak duplicated train rows into test).
        data_test = np.array(np.load(test_path, allow_pickle=True))
    else:
        mmap = np.load(npy_path, mmap_mode="r", allow_pickle=True)
        N    = len(mmap)
        np.random.seed(1234)
        perm = np.random.permutation(N)

        n_train = int(N * train_frac)
        if subsample:
            n_train = min(int(subsample), n_train)
        n_val     = max(int(n_train * val_frac / train_frac), 1)
        test_idx  = np.sort(perm[n_train + n_val:])
        data_test = mmap[test_idx]

    if max_events and len(data_test) > max_events:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(data_test), max_events, replace=False)
        data_test = data_test[np.sort(idx)]

    features = data_test[:, :-8]
    nLLs_all = data_test[:, -8:].astype(np.float64)
    nLL0     = nLLs_all[:, ::2].copy()   # (N, 4) baseline nLL values (even cols, before subtraction)
    for i in range(4):
        nLLs_all[:, 2 * i + 1] -= nLLs_all[:, 2 * i]
    nLLs_raw = nLLs_all[:, 1::2]   # (N, 4) delta-nLLs

    feat_mean = standardization["features_mean"][0]
    feat_std  = standardization["features_std"][0]

    # Build the feature pipeline from the stored config, preserving the
    # exact sequence of transforms applied during training.  A single trafos
    # group is assumed (multi-group would require per-group mean/std storage).
    trafos = cfg.get("data", {}).get("trafos") or {}
    feat_pipeline = []
    for trafo_fns in trafos.values():
        if isinstance(trafo_fns, list):
            feat_pipeline.extend(trafo_fns)

    features_prepd = preprocess_features(
        features, feat_mean, feat_std,
        pipeline=feat_pipeline if feat_pipeline else None,
    )

    return features_prepd, nLLs_raw, nLL0


# ---------------------------------------------------------------------------
# ONNX inference
# ---------------------------------------------------------------------------

def run_onnx(session: ort.InferenceSession, features: np.ndarray,
             batch_size: int = 4096) -> np.ndarray:
    """Run ONNX model in batches, return first 4 outputs (mean predictions).
    AmplitudeMLPWrapper ignores global_token so the exported model only has 'features'."""
    input_names = {i.name for i in session.get_inputs()}
    all_preds = []
    for start in range(0, len(features), batch_size):
        chunk = features[start:start + batch_size]
        feed  = {"features": chunk}
        if "global_token" in input_names:
            feed["global_token"] = np.zeros(len(chunk), dtype=np.int64)
        out = session.run(None, feed)[0]
        all_preds.append(out[:, :4])   # first 4 = mean predictions (HETEROSC has 8 outputs)
    return np.concatenate(all_preds, axis=0)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# exp (0) + expA (2) share x limits; obs (1) + obsA (3) share x limits.
_EXP_IDX = [0, 2]
_OBS_IDX = [1, 3]
_FONTSIZE   = 15


PERCENTILE_CUTS = [None, 99.999, 99.99, 99.9, 99, 95]

# Per-dataset final cuts: (pct_exp, pct_obs). Pattern matched against dataset name.
# pct=None → no cut (full range). Checked in order; first match wins.
FINAL_CUTS = [
    ('1908',              (99.999, 99.95)),   # ATLAS-SUSY-2018-32 (arXiv:1908.08215)
    ('1909',              (99.95,  99.95)),   # ATLAS-SUSY-2019-08 (arXiv:1909.09226)
    ('offshell-higgsino', (None,   None)),    # ATLAS-SUSY-2021-06 offshell higgsinos
    ('offshell-winobino', (None,   None)),    # ATLAS-SUSY-2021-06 offshell wino/bino
    ('onshell-winobino',  (99.99,  99.9)),    # ATLAS-SUSY-2021-06 onshell wino/bino
    ('sleptons',          (99.99,  99.99)),   # ATLAS-SUSY-2019-02 (arXiv:1911.12606)
]


def _get_final_cuts(dataset: str):
    ds = dataset.lower()
    for pattern, cuts in FINAL_CUTS:
        if pattern in ds:
            return cuts
    return (None, None)


def _pct_suffix(pct):
    """None → 'nocut', 99.9 → 'p99p9', 99 → 'p99', 95 → 'p95'."""
    if pct is None:
        return "nocut"
    return "p" + f"{pct}".replace(".", "p")


def _finite_range(arr, pct=None):
    """pct=None → full [min, max]; pct=99 → [1st, 99th percentile]."""
    f = arr[np.isfinite(arr)]
    if pct is None:
        return float(f.min()), float(f.max())
    tail = 100.0 - pct
    return float(np.percentile(f, tail)), float(np.percentile(f, pct))


def _sci_formatter():
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((0, 0))
    return fmt


def _savefig_pair(fig, out_path: str) -> None:
    """Write the figure as both .pdf and .png (CLAUDE.md: figures ship as a pair)."""
    stem = out_path[:-4] if out_path.lower().endswith((".pdf", ".png")) else out_path
    for ext in (".pdf", ".png"):
        fig.savefig(stem + ext, dpi=150, bbox_inches="tight")


def _apply_style(ax, xlabel, ylabel, subtitle):
    ax.xaxis.set_major_formatter(_sci_formatter())
    ax.yaxis.set_major_formatter(_sci_formatter())
    ax.set_xlabel(xlabel, fontsize=_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=_FONTSIZE)
    ax.set_title(subtitle, fontsize=_FONTSIZE)
    ax.tick_params(labelsize=_FONTSIZE - 1)


def _make_plot(fig, axes, xs, ys, xlabel, ylabel, title, pct=None, pct_exp=None, pct_obs=None):
    #fig.suptitle(title, fontsize=_FONTSIZE + 2, y=1.01)

    cmap = plt.cm.plasma.copy()
    cmap.set_bad("white")

    _pe = pct_exp if pct_exp is not None else pct
    _po = pct_obs if pct_obs is not None else pct

    exp_abs = max(abs(_finite_range(xs[i], _pe)[j]) for i in _EXP_IDX for j in (0, 1))
    obs_abs = max(abs(_finite_range(xs[i], _po)[j]) for i in _OBS_IDX for j in (0, 1))
    x_limits = {i: (-exp_abs, exp_abs) for i in _EXP_IDX}
    x_limits.update({i: (-obs_abs, obs_abs) for i in _OBS_IDX})

    for i, (ax, subtitle) in enumerate(zip(axes.flat, SUBPLOT_TITLES)):
        x, y = xs[i], ys[i]

        x_lo, x_hi = x_limits[i]
        _pi = _pe if i in _EXP_IDX else _po
        y_lo, y_hi = _finite_range(y, _pi)

        n_bins = max(40, min(80, int(np.sqrt(len(x) / 10))))
        counts, xedges, yedges = np.histogram2d(
            x, y, bins=[n_bins, n_bins],
            range=[[x_lo, x_hi], [y_lo, y_hi]],
        )
        counts_ma = np.ma.masked_where(counts == 0, counts)

        im = ax.pcolormesh(
            xedges, yedges, counts_ma.T,
            norm=LogNorm(vmin=1, vmax=counts.max()),
            cmap=cmap, rasterized=True,
        )
        cb = plt.colorbar(im, ax=ax)
        cb.set_label("counts", fontsize=_FONTSIZE - 1)
        cb.ax.tick_params(labelsize=_FONTSIZE - 1)

        ax.axvline(0, color="black", lw=0.6, ls="--")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        _apply_style(ax, xlabel, ylabel, subtitle)


def _symlog_bins(lo: float, hi: float, linthresh: float,
                 n_log: int = 60, n_lin: int = 30) -> np.ndarray:
    """Bins that are uniform in symlog space — bars look equal-width on a symlog axis."""
    parts = []
    if lo < -linthresh:
        parts.append(-np.geomspace(-lo, linthresh, n_log + 1)[:-1])
    parts.append(np.linspace(max(lo, -linthresh), min(hi, linthresh), n_lin + 1))
    if hi > linthresh:
        parts.append(np.geomspace(linthresh, hi, n_log + 1)[1:])
    return np.unique(np.concatenate(parts))


def make_histogram_plots(nLLs_truth: np.ndarray, title: str, out_dir: str, stem: str,
                         pct=None) -> None:
    """Two 2×2 grids of delta-nLL histograms: log-y/linear-x and log-y/symlog-x."""
    # Pre-compute shared x-limits per group (exp+expA and obs+obsA)
    def _group_range(indices):
        lo = min(_finite_range(nLLs_truth[np.isfinite(nLLs_truth[:, i]), i], pct)[0]
                 for i in indices)
        hi = max(_finite_range(nLLs_truth[np.isfinite(nLLs_truth[:, i]), i], pct)[1]
                 for i in indices)
        return lo, hi

    group_lims = {i: _group_range(_EXP_IDX) for i in _EXP_IDX}
    group_lims.update({i: _group_range(_OBS_IDX) for i in _OBS_IDX})

    for xscale, suffix in [("linear", "linx"), ("symlog", "symlogx")]:
        fig, axes = plt.subplots(2, 2, figsize=(11, 10))
        # fig.suptitle(f"{title} — nLL$_1$ distribution ({xscale} x)",
        #              fontsize=_FONTSIZE + 2, y=1.01)

        for i, (ax, subtitle) in enumerate(zip(axes.flat, SUBPLOT_TITLES)):
            vals   = nLLs_truth[:, i]
            finite = vals[np.isfinite(vals)]
            lo, hi = group_lims[i]

            if xscale == "symlog":
                # linthresh at the median |val| so the linear region covers
                # the dense central band; at least 0.01 to avoid log(0).
                nonzero = np.abs(finite[finite != 0.0])
                linthresh = float(np.median(nonzero)) if len(nonzero) else 1.0
                linthresh = max(linthresh, 1e-2)
                bins = _symlog_bins(lo, hi, linthresh)
                ax.set_xscale("symlog", linthresh=linthresh)
            else:
                bins = 200

            ax.hist(finite, bins=bins, range=(lo, hi) if xscale == "linear" else None,
                    color="steelblue", alpha=0.85)
            ax.set_xlim(lo, hi)
            ax.set_yscale("log")
            ax.axvline(0, color="black", lw=0.8, ls="--")
            ax.set_xlabel(r"nLL$_1$", fontsize=_FONTSIZE)
            ax.set_ylabel("counts", fontsize=_FONTSIZE)
            ax.set_title(subtitle, fontsize=_FONTSIZE)
            ax.tick_params(labelsize=_FONTSIZE - 1)
            ax.yaxis.set_major_formatter(_sci_formatter())

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{stem}_hist_{suffix}.pdf")
        _savefig_pair(fig, out_path)
        plt.close(fig)
        print(f"  Saved: {out_path}")


def make_residual_plot(nLLs_truth: np.ndarray, nLLs_pred: np.ndarray,
                       title: str, out_path: str, pct=None, pct_exp=None, pct_obs=None) -> None:
    residuals = nLLs_pred - nLLs_truth
    xs = [residuals[:, i] for i in range(4)]
    ys = [nLLs_truth[:, i] for i in range(4)]

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    _make_plot(fig, axes, xs, ys,
               xlabel=r"$\mathrm{nLL}_{1,\mathrm{pred}} - \mathrm{nLL}_{1,\mathrm{truth}}$",
               ylabel=r"$\mathrm{nLL}_{1,\mathrm{truth}}$",
               title=title, pct=pct, pct_exp=pct_exp, pct_obs=pct_obs)
    fig.tight_layout()
    _savefig_pair(fig, out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def make_relative_residual_plot(nLLs_truth: np.ndarray, nLLs_pred: np.ndarray,
                                title: str, out_path: str, pct=None, pct_exp=None, pct_obs=None) -> None:
    """Same layout but x-axis shows (pred - truth) / truth (relative residual)."""
    MIN_TRUTH = 1e-3
    xs, ys = [], []
    for i in range(4):
        truth = nLLs_truth[:, i]
        pred  = nLLs_pred[:, i]
        mask  = np.abs(truth) >= MIN_TRUTH
        xs.append((pred[mask] - truth[mask]) / truth[mask])
        ys.append(truth[mask])

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    _make_plot(fig, axes, xs, ys,
               xlabel=r"$(\mathrm{nLL}_{1,\mathrm{pred}} - \mathrm{nLL}_{1,\mathrm{truth}})\,/\,\mathrm{nLL}_{1,\mathrm{truth}}$",
               ylabel=r"$\mathrm{nLL}_{1,\mathrm{truth}}$",
               title=title, pct=pct, pct_exp=pct_exp, pct_obs=pct_obs)
    fig.tight_layout()
    _savefig_pair(fig, out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_model(onnx_path: str, out_dir: str, max_events: int | None) -> None:
    model_name = os.path.splitext(os.path.basename(onnx_path))[0]
    print(f"\n=== {model_name} ===")

    # --- metadata ---
    m  = onnx.load(onnx_path)
    md = {p.key: p.value for p in m.metadata_props}

    run_config_str = md.get("run_config", "")
    if not run_config_str:
        print("  No run_config — skipping.")
        return

    cfg            = yaml.safe_load(run_config_str)
    standardization = json.loads(md["standardization"])
    nLLs_mean = np.array(standardization["nLLs_mean"][0])
    nLLs_std  = np.array(standardization["nLLs_std"][0])

    nll_pipeline = cfg.get("data", {}).get("nLL_trafos") or None
    nll_lo = standardization.get("nLLs_lo")
    nll_hi = standardization.get("nLLs_hi")
    nll_bounds = ({"lo": nll_lo, "hi": nll_hi} if nll_lo is not None else {})

    # --- load test data ---
    print("  Loading test data ...")
    features_prepd, nLLs_truth, nLL0 = load_test_split(cfg, standardization, max_events=None)
    print(f"  Test events: {len(features_prepd)}")

    # Convert deltas to absolute nLL1 = nLL0 + delta for plots
    nLL1_truth = nLL0 + nLLs_truth

    analysis = md.get("analysis_altname", model_name).strip('"')

    # --- data distribution histograms (in nLL1 space) ---
    #make_histogram_plots(nLL1_truth, analysis, out_dir, model_name)

    # --- run model ---
    print("  Running ONNX inference ...")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    preds_prepd = run_onnx(session, features_prepd)

    # --- inverse preprocessing → predicted deltas → absolute nLL1 ---
    nLLs_pred = inverse_preprocess_nLLs(preds_prepd, nLLs_mean, nLLs_std,
                                         pipeline=nll_pipeline, nll_bounds=nll_bounds)
    nLL1_pred = nLL0 + nLLs_pred

    # --- final plots with per-dataset cuts ---
    dataset_cfg = cfg["data"]["dataset"]
    if isinstance(dataset_cfg, list):
        dataset_cfg = dataset_cfg[0]
    pct_exp_final, pct_obs_final = _get_final_cuts(dataset_cfg)
    make_residual_plot(
        nLL1_truth, nLL1_pred, analysis,
        os.path.join(out_dir, f"{model_name}_residuals_final.pdf"),
        pct_exp=pct_exp_final, pct_obs=pct_obs_final,
    )
    make_relative_residual_plot(
        nLL1_truth, nLL1_pred, analysis,
        os.path.join(out_dir, f"{model_name}_relative_residuals_final.pdf"),
        pct_exp=pct_exp_final, pct_obs=pct_obs_final,
    )

    # --- residual plots (in nLL1 space) — one file per percentile cut ---
    for pct in PERCENTILE_CUTS:
        suf = _pct_suffix(pct)
        make_residual_plot(
            nLL1_truth, nLL1_pred, analysis,
            os.path.join(out_dir, f"{model_name}_residuals_{suf}.pdf"),
            pct=pct,
        )
        make_relative_residual_plot(
            nLL1_truth, nLL1_pred, analysis,
            os.path.join(out_dir, f"{model_name}_relative_residuals_{suf}.pdf"),
            pct=pct,
        )
        #make_histogram_plots(nLL1_truth, analysis, out_dir, f"{model_name}_{suf}", pct=pct)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate residual validation plots for ONNX models.")
    parser.add_argument("models", nargs="*",
                        help="ONNX files to process (default: all in models_onnx/)")
    parser.add_argument("--max-events", type=int, default=DEFAULT_MAX_EVENTS,
                        help="Max test events per model (default: %(default)s; 0 = all)")
    parser.add_argument("--out-dir", default=os.path.join(BASE_DIR, "validation_plots"),
                        help="Output directory for plots (default: %(default)s)")
    args = parser.parse_args()

    max_events = args.max_events or None

    os.makedirs(args.out_dir, exist_ok=True)

    onnx_files = args.models or sorted(
        os.path.join(MODELS_ONNX, f)
        for f in os.listdir(MODELS_ONNX)
        if f.endswith(".onnx") and not f.startswith("._")
    )

    if not onnx_files:
        print("No ONNX files found.")
        sys.exit(1)

    for path in onnx_files:
        try:
            process_model(path, args.out_dir, max_events)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    print("\nAll done.")


if __name__ == "__main__":
    main()
