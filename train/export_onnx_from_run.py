"""
Export a trained run to ONNX with OLLL v0.1 metadata.

    python export_onnx_from_run.py RUN_DIR [--generation GEN.json] [--analysis ATLAS-SUSY-...]
                                   [--label NAME] [--filtering TEXT] [--nll-max JSON]
                                   [--run-idx N] [--out OUT.onnx]

Example (a run trained on data/1911.12606-70k-uncertainty.npy, with the
generator's data/1911.12606-70k-uncertainty.json next to it):

    python export_onnx_from_run.py runs/my_exp/20261001_120000_MuMLP_1234 --label EWKinos

Run it from the repository root, where the run directory and the training data
are (it rebuilds the experiment and reads the data). No GPU needed.

Inputs
  RUN_DIR       runs/<exp>/<run>/: config_<idx>.yaml and models/model_run<idx>.pt[.gz]
                (the lowest config index present if <idx> = --run-idx is missing).
  --generation  GEN.json, the record the data-generation pipeline (sampling/ in
                OLLL-Train) writes next to every CSV it produces. Default:
                <data_path>/<dataset>.json of the run's config. It must hold
                bkgfiles, channels, obs_yields, bkg_yields, bkg_unc, removeCRsVRs,
                remove_channels and nLL_{exp,obs}_max, nLLA_{exp,obs}_max as
                [mu_hat, nLL]. No previously published model is needed.
  --analysis    ATLAS-SUSY-YYYY-NN. Default: the record's analysis_altname. A new
                analysis needs one entry (arXiv, INSPIRE, HEPData DOI) in
                olll_metadata.ANALYSES.
  --label       the model within the analysis (EWKinos, Sleptons, Offshell, ...);
                goes into model_name, e.g. ATLAS-SUSY-2018-16_EWKinos_MuMLP-5x512.
  --filtering   how the training dataset was derived from the generated scan
                (e.g. "rows with delta nLL_obs > 40 removed"); default "none".
  --nll-max     '{"nLL_exp_max": [mu_hat, nll], ...}' to supply *_max values the
                record lacks. A null *_max in the record means the generator's
                maximum-likelihood fit failed (sampling/likelihood.py,
                calculate_Lmax); re-running it is better than typing values in.
  --out         default RUN_DIR/models/model_with_metadata.onnx.

Where each part of the metadata comes from (olll_metadata.py builds it):
  - network, run_config, preprocessing, standardization  <- this run
  - bounds x/y_min/max, mu=0 baselines                   <- the run's training data
                                                            (same train split as the run)
  - source (paper, HEPData record)                       <- olll_metadata.ANALYSES
  - statistical model, *_max, generation settings        <- GEN.json
    (incl. sig_rel_unc, the signal uncertainty the generator put in the likelihood)

The inputs are declared as total yields (background + signal) per active bin,
in the order of bkg_yields minus remove_channels: the training data must be in
that form. A record that merges several scans can flag more than one patchset
as used; the model is then the one whose channel map holds exactly the regions
of bkg_yields, and only its patchset is recorded.

The file is only written if it checks out: decoded from its own metadata it
must reproduce held-out rows of the training data (median relative error on
nLL(mu=1) below 1%), the architecture it states must be the one in the graph,
S. Kraml's validator (tools/olll_validate.py) must report no errors, and the
OLLL adapter (hep_olll) must load it. Otherwise it stops and says why. With
jsonschema installed, hep_olll's own metadata validator runs too.
"""

import argparse
import gzip
import inspect
import json
import math
import os
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml
from omegaconf import OmegaConf

from experiment import nLLsExperiment
from misc import get_device
import olll_metadata as om

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from olll_training_stats import training_stats  # noqa: E402
from olll_publish import CLOSURE_TOL, closure, graph_architecture  # noqa: E402

# what the generation record must provide
RECORD_KEYS = ["bkgfiles", "channels", "obs_yields", "bkg_yields", "bkg_unc",
               "removeCRsVRs", "remove_channels"]
MU0_TOL = 1e-3


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


def load_run(run_dir, run_idx, device):
    """(cfg, experiment with the trained weights loaded, run_idx actually used)."""
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

    # Recreate the experiment just enough to get preprocessing + model
    exp = nLLsExperiment(cfg, device)
    exp.warm_start = False   # normally set by BaseExperiment._init()
    exp.dtype = torch.float32
    exp.init_physics()
    exp.init_data()
    # Same order as BaseExperiment.full_run (init_data -> _init_dataloader ->
    # init_model). _init_dataloader is what derives the training-target range,
    # and init_model needs it to pin the output clamp; skipping it exported a
    # model whose clamp constants were still +-inf, i.e. silently unbounded.
    exp._init_dataloader()
    exp.init_model()

    gz = os.path.join(run_dir, "models", f"model_run{run_idx}.pt.gz")
    pt = os.path.join(run_dir, "models", f"model_run{run_idx}.pt")
    if os.path.exists(gz):
        checkpoint = load_model_gz(gz, device)
    elif os.path.exists(pt):
        checkpoint = torch.load(pt, map_location=device)
    else:
        raise FileNotFoundError(f"Missing {gz} and {pt}")
    exp.model.load_state_dict(checkpoint["model"])
    exp.model.eval()
    return cfg, exp, run_idx


def export_graph(exp, n_features, device, path):
    export_kwargs = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        # torch>=2.6 defaults to the dynamo exporter. Stay on the TorchScript
        # path the published models_onnx/ files were produced with.
        export_kwargs["dynamo"] = False
    torch.onnx.export(
        ExportAdapter(exp.model),
        (torch.zeros(1, n_features, device=device), torch.zeros(1, dtype=torch.long, device=device)),
        path,
        opset_version=17,
        input_names=["features", "global_token"],
        output_names=["nLLs"],
        dynamic_axes={"features": {0: "batch"}, "global_token": {0: "batch"}, "nLLs": {0: "batch"}},
        **export_kwargs,
    )


def load_record(path, nll_max):
    """The generation record, checked for what the metadata needs."""
    with open(path) as f:
        rec = json.load(f)
    missing = [k for k in RECORD_KEYS if k not in rec]
    if missing:
        sys.exit(f"{path}: no {missing}; not a generation record")
    if nll_max:
        rec.update(json.loads(nll_max))
    bad = [k for k in om.MAX_KEYS
           if not (isinstance(rec.get(k), list) and len(rec[k]) == 2
                   and all(isinstance(v, (int, float)) and math.isfinite(v) for v in rec[k]))]
    if bad:
        sys.exit(f"{path}: {bad} not recorded ({[rec.get(k) for k in bad]}). OLLL v0.1 requires "
                 "each as [mu_hat, nLL at mu_hat]; the generator computes them on its first "
                 "scan point (sampling/likelihood.py calculate_Lmax), and a null means that fit "
                 "failed. Re-run it, or pass the values with --nll-max.")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--generation", help="generation record (default: <data_path>/<dataset>.json)")
    ap.add_argument("--analysis", choices=sorted(om.ANALYSES),
                    help="ATLAS analysis ID (default: the record's analysis_altname)")
    ap.add_argument("--label", help="model within the analysis, e.g. EWKinos, used in model_name")
    ap.add_argument("--filtering", default="none",
                    help="how the training dataset was derived from the generated scan")
    ap.add_argument("--nll-max", help='JSON {"nLL_exp_max": [mu_hat, nll], ...} for a record without them')
    ap.add_argument("--run-idx", type=int, default=0)
    ap.add_argument("--out", help="default: RUN_DIR/models/model_with_metadata.onnx")
    args = ap.parse_args()

    device = get_device()
    cfg, exp, run_idx = load_run(args.run_dir, args.run_idx, device)
    run_config = OmegaConf.to_yaml(cfg)   # as stored by the run, interpolations kept
    cfgd = OmegaConf.to_container(cfg, resolve=True)
    datasets = list(cfgd["data"]["dataset"])
    assert len(datasets) == 1, f"one dataset per published model, run has {datasets}"
    dataset, data_dir = datasets[0], cfgd["data"]["data_path"]
    n_features = int(cfgd["model"]["net"]["n_features"])
    out = args.out or os.path.join(args.run_dir, "models", "model_with_metadata.onnx")
    log = []

    # -- the analysis side, from the generation record
    gen_path = args.generation or os.path.join(data_dir, f"{dataset}.json")
    rec = load_record(gen_path, args.nll_max)
    analysis = args.analysis or rec.get("analysis_altname")
    if analysis not in om.ANALYSES:
        sys.exit(f"analysis {analysis!r} unknown: pass --analysis, and add it to olll_metadata.ANALYSES "
                 "if it is new")
    if rec.get("analysis"):
        assert rec["analysis"] == om.ANALYSES[analysis]["arxiv"], (rec["analysis"], analysis)
    removed = set(rec["remove_channels"] or [])
    active = [b[0] for b in rec["bkg_yields"] if b[0].rsplit("-", 1)[0] not in removed]
    assert len(active) == n_features, \
        f"{gen_path}: {len(active)} active bins, the network takes {n_features} inputs"
    log.append(f"generation record {gen_path}: model {om.active_statistical_model(rec)[0]}, "
               f"{n_features} input bins")

    # -- the data side, from the training data (same split as the run)
    stats = training_stats(data_dir, dataset, cfgd["data"]["train_test_val"][0],
                           cfgd["data"].get("subsample"))
    assert stats["n_features"] == n_features, (stats["n_features"], n_features)
    mu0 = stats["mu0"]
    run_mu0 = np.asarray(exp.nLL_mu0[0]).ravel()
    assert np.allclose(run_mu0, mu0, atol=MU0_TOL), f"mu=0 baselines: run {run_mu0} vs data {mu0}"
    if isinstance(rec.get("y_min"), list) and len(rec["y_min"]) == 8:
        # the generator's y_min covers the 8 absolute nLL columns; the mu=0 ones are constant
        rec_mu0 = rec["y_min"][0::2]
        if not np.allclose(rec_mu0, mu0, atol=MU0_TOL):
            sys.exit(f"mu=0 baselines: record {rec_mu0} vs training data {mu0}: "
                     "the record is not the one of this dataset")
    if stats["n_sentinel_train_rows"]:
        log.append(f"bounds exclude {stats['n_sentinel_train_rows']} failed-fit rows (|nLL| >= 1e9)")
    if isinstance(rec.get("lower_limits"), list):
        lo = [v for b, v in zip(rec["bkg_yields"], rec["lower_limits"])
              if b[0].rsplit("-", 1)[0] not in removed]
        below = int(np.sum(np.asarray(stats["x_min"]) < np.asarray(lo) - 1e-3))
        if below:
            log.append(f"NOTE {below} input bins have training yields below the record's lower_limits "
                       "(expected only for regions with signal leakage)")

    # -- the network
    with tempfile.TemporaryDirectory() as tmp:
        tmp_onnx = os.path.join(tmp, "graph.onnx")
        export_graph(exp, n_features, device, tmp_onnx)
        model = onnx.load(tmp_onnx)
    n_in, width, depth, n_out = graph_architecture(model)
    net = cfgd["model"]["net"]
    assert (n_in, width, depth, n_out) == (n_features, net["hidden_channels"], net["hidden_layers"], 8), \
        f"graph {(n_in, width, depth, n_out)} vs run_config {net}"

    std = {
        "nLLs_mean": [m.tolist() for m in exp.prepd_mean],
        "nLLs_std": [s.tolist() for s in exp.prepd_std],
        "features_mean": [m.tolist() for m in exp.prepd_mean_features],
        "features_std": [s.tolist() for s in exp.prepd_std_features],
    }
    nll_bounds = exp.prepd_nll_bounds[0] if exp.prepd_nll_bounds else {}
    if "lo" in nll_bounds:  # logit_bounded only; per-output spec carries no lo/hi
        std["nLLs_lo"] = np.asarray(nll_bounds["lo"]).tolist()
        std["nLLs_hi"] = np.asarray(nll_bounds["hi"]).tolist()

    run_log = os.path.join(args.run_dir, f"out_{run_idx}.log")
    if os.path.exists(run_log):
        date, duration = om.run_log_times(run_log)
    else:
        date, duration = om.training_date_from_run_name(cfgd.get("run_name")), None
        log.append(f"no {run_log}: training_duration omitted")

    meta = om.build_metadata(
        analysis_id=analysis,
        model_name=om.model_name(analysis, args.label, depth, width),
        run_config=run_config,
        standardization=std,
        reference=rec,
        bounds={k: stats[k] for k in ("x_min", "x_max", "y_min", "y_max")},
        mu0=mu0,
        generation=om.normalize_generation(rec),
        training_dataset={"name": dataset, "n_rows": stats["n_rows"], "filtering": args.filtering},
        training_date=date,
        training_duration=duration,
    )
    om.write_metadata(model, meta)

    # -- checks on the file as a consumer sees it, before it is written out
    with tempfile.TemporaryDirectory() as tmp:
        cand = os.path.join(tmp, "model.onnx")
        onnx.save(model, cand)
        wm = {p.key: p.value for p in onnx.load(cand).metadata_props}
        sample = np.asarray(stats["sample"], dtype=np.float64)
        err = closure(ort.InferenceSession(cand), yaml.safe_load(wm["run_config"]),
                      json.loads(wm["standardization"]), [float(wm[k]) for k in om.MU0_KEYS], sample)
        log.append(f"closure on {len(sample)} held-out rows: median rel err nLL(mu=1) {np.round(err, 6).tolist()}")
        if np.max(err) >= CLOSURE_TOL:
            sys.exit("\n".join(log) + f"\nthe file does not reproduce its training data (tolerance {CLOSURE_TOL})")

        import olll_validate
        from pathlib import Path
        report = olll_validate.Report(cand)
        data, m2 = olll_validate.load_input(Path(cand), report)
        olll_validate.validate(data, report, model=m2)
        res = report.as_dict()
        log.append(f"olll_validate: {res['status']} ({res['errors']} errors, {res['warnings']} warnings)")
        log.extend(f"  [{i['severity']}] {i['code']}: {i['message']}" for i in res["issues"])
        if res["errors"]:
            sys.exit("\n".join(log))

        from hep_olll.nnAdapter import NNAdapter
        try:
            import jsonschema  # noqa: F401  (the adapter's own validator needs it)
            check_meta = True
        except ImportError:
            check_meta = False
            log.append("hep_olll metadata validator skipped: jsonschema not installed")
        adapter = NNAdapter(cand, validate_metadata=check_meta)
        adapter.predict({s: 0.0 for s in adapter.srOrder})
        log.append(f"hep_olll: loads and predicts ({len(adapter.srOrder)} inputs)")

        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        onnx.save(model, out)

    print(f"== {out}")
    for line in log:
        print("  " + line)


if __name__ == "__main__":
    main()
