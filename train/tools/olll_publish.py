"""
Republish an existing ONNX model with clean OLLL v0.1 metadata.

    python tools/olll_publish.py IN.onnx OUT.onnx --analysis ATLAS-SUSY-2018-16 \
        --label EWKinos --stats stats/<dataset>.json

--stats is the JSON printed by tools/olll_training_stats.py for the model's
training dataset (it runs where the data lives; see that file).

Nothing is taken on trust from the input file's metadata:
  1. Every (run_config, standardization) pairing present in the file is run
     through the network on held-out rows of the training dataset; the one
     that reproduces the truth is the one that belongs to these weights, and it
     must win clearly. A file whose metadata cannot reproduce its own data is
     refused.
  2. The architecture in that run_config must match the weight shapes in the
     graph.
  3. Bounds and mu=0 baselines come from the training data (--stats); stored
     baselines that disagree are reported.
  4. With --reference-onnx (candidates from the data-generation pipeline, R.
     Maselek's ML_LHClikelihoods/models), the statistical-model and generation
     fields come from the one candidate that (a) describes the same statistical
     model and inputs as IN, (b) whose recorded sampling lower_limits equal the
     training data's minima, and (c) whose *_max share the training data's nLL
     normalisation. Needed when IN was exported without them.
  5. Metadata is rebuilt by olll_metadata.build_metadata; the graph must come
     out byte-identical.
  6. The written file is decoded again from its own metadata alone and must
     pass the same closure test, then S. Kraml's validator (tools/olll_validate.py)
     with no errors.
"""

import argparse
import json
import os
import re
import sys
from collections import OrderedDict

import numpy as np
import onnx
import onnxruntime as ort
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import olll_metadata as om  # noqa: E402
from olll_transforms import preprocess_features, undo_preprocess_nLLs  # noqa: E402

# closure tolerance: median relative error on nLL(mu=1) over held-out rows
CLOSURE_TOL = 1e-2
# a competing metadata pairing must be at least this much worse to rule it out
WIN_FACTOR = 5.0
MU0_TOL = 1e-3

REFERENCE_KEYS = ["channels", "bkgfiles", "patchsets", "obs_yields", "bkg_yields",
                  "bkg_unc", "removeCRsVRs", "remove_channels", "analysis_altname",
                  "analysis"] + om.MAX_KEYS + om.GENERATION_KEYS + list(om.LEGACY_SPELLINGS)


def multimap(model):
    out = OrderedDict()
    for p in model.metadata_props:
        out.setdefault(p.key.replace("rafal::", ""), []).append(p.value)
    return out


def predict_nll1(sess, cfg, std, mu0, X):
    """nLL(mu=1) for rows X, decoding with (cfg, std, mu0) the way the adapters do."""
    d = cfg["data"]
    Xp, _, _ = preprocess_features(X, trafos=d["trafos"],
                                   mean=np.asarray(std["features_mean"][0]),
                                   std=np.asarray(std["features_std"][0]))
    out = sess.run(None, {"features": Xp.astype(np.float32)})[0]
    deltas = np.array([undo_preprocess_nLLs(o[:4].astype(np.float64),
                                            mean=np.asarray(std["nLLs_mean"][0]),
                                            std=np.asarray(std["nLLs_std"][0]),
                                            trafos=d["nLL_trafos"]) for o in out])
    return np.asarray(mu0)[None, :] + deltas


def closure(sess, cfg, std, mu0, sample):
    X, T = sample[:, :-8], sample[:, -8:]
    t1 = T[:, 1::2]
    p1 = predict_nll1(sess, cfg, std, mu0, X)
    return np.median(np.abs(p1 - t1) / np.abs(t1), axis=0)


def graph_architecture(model):
    """(n_inputs, width, depth, n_outputs) from the Gemm weight shapes."""
    shapes = [tuple(i.dims) for i in model.graph.initializer if len(i.dims) == 2]
    widths = [s[0] for s in shapes[:-1]]
    assert len(set(widths)) == 1, f"non-uniform hidden widths {shapes}"
    return shapes[0][1], widths[0], len(shapes) - 1, shapes[-1][0]


def decode_reference(meta):
    reference = {}
    for key in REFERENCE_KEYS:
        vals = set(meta.get(key, []))
        assert len(vals) <= 1, f"{key}: {len(vals)} conflicting reference values"
        if vals:
            v = vals.pop()
            try:
                reference[key] = json.loads(v)
            except ValueError:
                reference[key] = v
    return reference


# fields that fix the statistical model and the input contract
MODEL_FIELDS = ["bkg_yields", "obs_yields", "bkg_unc", "remove_channels", "removeCRsVRs"]


def pick_reference(paths, own, stats, log, nll_offset=0.0, check_generation=True):
    """The one candidate whose record belongs to this model's training data."""
    passing = []
    for path in paths:
        try:
            ref = decode_reference(multimap(onnx.load(path, load_external_data=False)))
        except Exception as exc:  # noqa: BLE001  (a truncated/corrupt candidate)
            log.append(f"reference {path}: unreadable ({type(exc).__name__})")
            continue
        why = []
        for k in MODEL_FIELDS:
            if k in own and ref.get(k) != own[k]:
                why.append(f"{k} differs")
        if own.get("channels") is not None and ref.get("channels") != own["channels"]:
            why.append("channels differ")
        if not check_generation:
            pass   # data from another generator: its sampling box is not this one
        elif "lower_limits" not in ref:
            why.append("no lower_limits")
        else:
            # lower_limits bound the sampled yields BEFORE signal leakage: with
            # signal_leakage_CR/VR on, that region's yields are then varied by up
            # to +-spread and can fall below it. Compare only regions without
            # leakage (always the SRs).
            names = [b[0] for b in ref["bkg_yields"]]
            rm = set(ref.get("remove_channels") or [])
            chan = ref["channels"]
            chan = {k: v for d in (chan if isinstance(chan, list) else [chan]) for k, v in d.items()}
            leaky = {c for c, t in chan.items()
                     if (t == "CR" and ref.get("signal_leakage_CR")) or (t == "VR" and ref.get("signal_leakage_VR"))}
            active = [(n, v) for n, v in zip(names, ref["lower_limits"]) if n.rsplit("-", 1)[0] not in rm]
            if len(active) != len(stats["x_min"]):
                why.append(f"{len(active)} active bins vs {len(stats['x_min'])} inputs")
            else:
                cmp = [(v, x) for (n, v), x in zip(active, stats["x_min"]) if n.rsplit("-", 1)[0] not in leaky]
                diff = max(abs(v - x) for v, x in cmp)
                if diff > 1e-3:
                    why.append(f"lower_limits != training-data minima on non-leakage bins (max diff {diff:.3g})")
                elif len(cmp) < len(active):
                    log.append(f"reference {path}: lower_limits compared on {len(cmp)} of "
                               f"{len(active)} bins ({len(active) - len(cmp)} with signal leakage)")
        for key, mu0, idx in (("nLL_exp_max", stats["mu0"][0], 1), ("nLLA_exp_max", stats["mu0"][2], 1)):
            if key not in ref:
                why.append(f"no {key}")
            elif abs(ref[key][idx] + nll_offset - mu0) > MU0_TOL:
                why.append(f"{key} nLL {ref[key][idx]:.6f} (+{nll_offset}) vs data mu0 {mu0:.6f}: other normalisation")
        log.append(f"reference {path}: " + ("MATCH" if not why else "; ".join(why)))
        if not why:
            passing.append((path, ref))
    assert passing, "no --reference-onnx candidate matches the training data:\n  " + \
        "\n  ".join(l for l in log if l.startswith("reference "))
    record = lambda r: (json.dumps(om.normalize_generation(r), sort_keys=True)
                        + json.dumps([r.get(k) for k in om.MAX_KEYS]))

    def union_if_consistent(cands):
        """One record from files that agree on every key they share, else None.
        (The same scan recorded more or less completely, e.g. one file adds merged.)"""
        merged = {}
        for _, r in cands:
            for k, v in r.items():
                if k in merged and merged[k] != v:
                    return None
                merged[k] = v
        return merged

    if len({record(r) for _, r in passing}) > 1:
        union = union_if_consistent(passing)
        if union is not None:
            log.append(f"{len(passing)} candidates match and agree on every shared key; "
                       f"using their union ({', '.join(p for p, _ in passing)})")
            return union
        # One scan grown in steps (100k, 200k, 400k, ...) leaves one file per
        # step, all consistent with the data; the pipeline's folder_name names
        # the step, and our dataset keeps that name ('%' -> '_' on disk).
        keys = sorted(set().union(*(r.keys() for _, r in passing)))
        for k in keys:
            vals = [json.dumps(r.get(k))[:60] for _, r in passing]
            if len(set(vals)) > 1:
                log.append(f"candidates differ on {k}: " + " || ".join(vals))
        # (or its prefix up to a '-': '2106.01676-offshell-winobino-plus' for
        # '2106.01676-offshell-winobino-plus-fluct20_-300k'; the offshell scans
        # share one statistical model and differ only in the signal patchset).
        def names_dataset(folder):
            folder = str(folder or "").replace("%", "_")
            return bool(folder) and (stats["dataset"] == folder or stats["dataset"].startswith(folder + "-"))
        named = [(p, r) for p, r in passing if names_dataset(r.get("folder_name"))]
        log.append(f"{len(passing)} candidates match; {len(named)} whose folder_name names {stats['dataset']}")
        passing = named
    assert passing and len({record(r) for _, r in passing}) == 1, \
        "ambiguous: no single generation record for this dataset:\n  " + \
        "\n  ".join(l for l in log if ": MATCH" in l or "candidates" in l)
    path, ref = passing[0]
    log.append(f"statistical-model {'' if not check_generation else 'and generation '}metadata from {path}")
    if nll_offset:
        ref = dict(ref)
        for key in om.MAX_KEYS:
            ref[key] = [ref[key][0], ref[key][1] + nll_offset]
        log.append(f"*_max nLL values shifted by +{nll_offset} to the training data's normalisation (mu_hat unchanged)")
    if "patchsets" in own:
        log.append("input file's own patchsets " + ("agree" if own["patchsets"] == ref.get("patchsets")
                                                   else f"DIFFER: {json.dumps(own['patchsets'])}"))
    return ref


def duration_hms(text):
    """'1 hours 1 minutes 8 seconds' -> '01:01:08'."""
    m = re.match(r"^\s*(\d+) hours (\d+) minutes (\d+) seconds\s*$", str(text).strip('"'))
    return f"{int(m.group(1)):02d}:{int(m.group(2)):02d}:{int(m.group(3)):02d}" if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--analysis", required=True, choices=sorted(om.ANALYSES))
    ap.add_argument("--label", default=None, help="model within the analysis, e.g. EWKinos")
    ap.add_argument("--stats", required=True)
    ap.add_argument("--training-date", default=None,
                    help="YYYY-MM-DD, for runs whose run_config has no timestamp (read it from the run log)")
    ap.add_argument("--training-duration", default=None, help="HH:MM:SS, likewise")
    ap.add_argument("--run-log", default=None,
                    help="the run's out_<idx>.log: training date and duration from its first "
                         "line and its 'Finished experiment' line")
    ap.add_argument("--reference-onnx", action="append", default=[],
                    help="candidate generation-pipeline ONNX to take statistical-model and "
                         "generation metadata from (repeatable; exactly one must match)")
    ap.add_argument("--nll-offset", type=float, default=0.0,
                    help="constant by which the training data's nLLs sit above the reference's "
                         "(a likelihood-normalisation change, e.g. 0.5*ln(2*pi) for newer spey); "
                         "checked against the mu=0 baselines and added to the reference *_max")
    ap.add_argument("--generation-not-recorded", action="store_true",
                    help="the training data was NOT produced by the reference's generation "
                         "pipeline: take only the statistical-model fields from it, and write "
                         "every generation key as null")
    ap.add_argument("--drop-generation-key", action="append", default=[],
                    help="a recorded generation setting that does not hold for the training "
                         "dataset (e.g. a filter record for a filter it never went through)")
    ap.add_argument("--filtering", default=None,
                    help="how the training dataset was derived from the generated scan "
                         "(our filtering); omit if used as generated")
    args = ap.parse_args()

    model = onnx.load(args.inp)
    graph_before = model.graph.SerializeToString()
    meta = multimap(model)
    stats = json.load(open(args.stats))
    sample = np.asarray(stats["sample"], dtype=np.float64)
    sess = ort.InferenceSession(args.inp)
    log = []

    # -- reference (statistical-model) fields: must be unambiguous as they are
    reference = decode_reference(meta)
    if args.reference_onnx:
        reference = pick_reference(args.reference_onnx, reference, stats, log,
                                   nll_offset=args.nll_offset,
                                   check_generation=not args.generation_not_recorded)
    alt = reference.get("analysis_altname")
    assert alt in (None, args.analysis), f"file says {alt}, --analysis says {args.analysis}"
    if "analysis" in reference:
        assert reference["analysis"] == om.ANALYSES[args.analysis]["arxiv"], reference["analysis"]

    # -- which run_config/standardization belong to these weights
    rcs, sts = meta.get("run_config", []), meta.get("standardization", [])
    mu0 = stats["mu0"]
    scored = []
    for i, rc in enumerate(rcs):
        cfg = yaml.safe_load(rc)
        for j, st in enumerate(sts):
            err = closure(sess, cfg, json.loads(st), mu0, sample)
            scored.append((float(np.max(err)), i, j, err))
            log.append(f"closure run_config#{i} x standardization#{j}: "
                       f"median rel err nLL(mu=1) {np.round(err, 6).tolist()}")
    scored.sort(key=lambda s: s[0])
    best, i_rc, i_st, best_err = scored[0]
    assert best < CLOSURE_TOL, f"no metadata pairing reproduces the data (best {best:.3g})"
    distinct_rivals = [s for s in scored[1:] if not (rcs[s[1]] == rcs[i_rc] and sts[s[2]] == sts[i_st])]
    for s in distinct_rivals:
        assert s[0] > WIN_FACTOR * best, f"ambiguous: pairing {s[1:3]} scores {s[0]:.3g} vs best {best:.3g}"
    run_config, std = rcs[i_rc], json.loads(sts[i_st])
    cfg = yaml.safe_load(run_config)
    if len(scored) > 1:
        log.append(f"chose run_config#{i_rc} x standardization#{i_st} "
                   f"(run_dir {cfg.get('run_dir') or cfg.get('run_name')})")

    # -- the architecture described must be the one in the graph
    n_in, width, depth, n_out = graph_architecture(model)
    net = cfg["model"]["net"]
    assert (n_in, width, depth, n_out) == (net["n_features"], net["hidden_channels"],
                                          net["hidden_layers"], 8), \
        f"graph {(n_in, width, depth, n_out)} vs run_config {net}"
    assert stats["n_features"] == n_in, (stats["n_features"], n_in)
    assert stats["dataset"] in cfg["data"]["dataset"], (stats["dataset"], cfg["data"]["dataset"])

    # -- baselines: data is authoritative; report disagreement with the file
    for key, val in zip(om.MU0_KEYS, mu0):
        stored = {float(v) for v in meta.get(key, [])}
        if any(abs(s - val) > MU0_TOL for s in stored):
            log.append(f"NOTE {key}: stored {sorted(stored)} -> {val} (from training data)")

    # -- identity: only what the chosen run can vouch for
    date = om.training_date_from_run_name(cfg.get("run_name"))
    duration = None
    if date:
        for d_val, dur in zip(meta.get("training_date", []), meta.get("training_duration", [])):
            if d_val.strip('"').startswith(date):
                duration = duration_hms(dur)
    elif args.run_log:
        date, duration = om.run_log_times(args.run_log)
        log.append(f"training_date/duration from {args.run_log}: {date} {duration}")
    elif args.training_date:
        date, duration = args.training_date, args.training_duration
        log.append(f"training_date/duration from the command line: {date} {duration}")
    else:
        log.append("training_date/duration omitted: the chosen run_config has no run timestamp")
    model_name = om.model_name(args.analysis, args.label, depth, width)

    bounds = {k: stats[k] for k in ("x_min", "x_max", "y_min", "y_max")}
    if stats.get("n_sentinel_train_rows"):
        log.append(f"bounds exclude {stats['n_sentinel_train_rows']} failed-fit rows (|nLL|>=1e9)")

    generation = om.normalize_generation(reference)
    if args.generation_not_recorded:
        keep = ("analysis", "analysis_altname", "analyses")   # the analysis, not the sampling
        generation = {k: v for k, v in generation.items() if k in keep}
        log.append("generation keys written as null: the training data is not from the reference's generation pipeline")
    for k in args.drop_generation_key:
        assert k in generation, f"--drop-generation-key {k}: not in the file"
        log.append(f"dropped generation key {k}={json.dumps(generation.pop(k))} (does not hold for the training dataset)")
    training_dataset = {"name": stats["dataset"], "n_rows": stats["n_rows"],
                        "filtering": args.filtering or "none recorded in the training code"}
    new_meta = om.build_metadata(
        analysis_id=args.analysis, model_name=model_name, run_config=run_config,
        standardization=std, reference=reference, bounds=bounds, mu0=mu0,
        generation=generation, training_dataset=training_dataset,
        training_date=date, training_duration=duration)
    dropped = sorted(set(meta) - set(new_meta) - set(om.LEGACY_SPELLINGS))
    log.append(f"not carried over: {dropped}")
    om.write_metadata(model, new_meta)
    assert model.graph.SerializeToString() == graph_before, "graph changed"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    onnx.save(model, args.out)

    # -- the written file, decoded from its own metadata only
    written = onnx.load(args.out)
    wm = {p.key: p.value for p in written.metadata_props}
    err = closure(ort.InferenceSession(args.out), yaml.safe_load(wm["run_config"]),
                  json.loads(wm["standardization"]),
                  [float(wm[k]) for k in om.MU0_KEYS], sample)
    assert np.max(err) < CLOSURE_TOL, f"written file fails closure: {err}"
    log.append(f"written file closure: median rel err nLL(mu=1) {np.round(err, 6).tolist()}")

    import olll_validate
    report = olll_validate.Report(args.out)
    data, m2 = olll_validate.load_input(__import__("pathlib").Path(args.out), report)
    olll_validate.validate(data, report, model=m2)
    res = report.as_dict()
    log.append(f"olll_validate: {res['status']} ({res['errors']} errors, {res['warnings']} warnings)")
    for issue in res["issues"]:
        log.append(f"  [{issue['severity']}] {issue['code']}: {issue['message']}")

    print(f"== {args.out}")
    for line in log:
        print("  " + line)
    sys.exit(1 if res["errors"] else 0)


if __name__ == "__main__":
    main()
