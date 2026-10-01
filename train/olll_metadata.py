"""
OLLL metadata, schema v0.1: the single place that decides what an ONNX file of
this project says about itself.

Every exporter (export_onnx_from_run.py, export_combined_onnx.py) and the
republisher (tools/olll_publish.py) build their metadata through
`build_metadata` and write it through `write_metadata`, so key names, key
order, value formats and the author line cannot drift between them. The schema
is S. Kraml's OLLL v0.1 (template + validator vendored in tools/).

What goes in, and where each value comes from:
  - model identity/parameters/run_config  <- the run that produced the weights
  - source                                <- ANALYSES below (HEPData records)
  - channels/yields/removal/*_max         <- the statistical model, as recorded
                                             by R. Maselek's data-generation pipeline
  - GENERATION_KEYS                       <- that pipeline's settings for the scan
                                             our training data was drawn from
                                             (he generated the data; we filtered it)
  - training_dataset                      <- our dataset and the filtering we applied
  - bounds, *_mu0                         <- our training data
  - preprocessing/standardization         <- the run that produced the weights
Not carried over: the reference file's own network/training description (a
different model) and the generation pipeline's plumbing (folders, buffers,
process count, verbosity), which says nothing about the data.
"""

import json
import re
from collections import OrderedDict

SCHEMA_VERSION = "0.1"
MODEL_AUTHOR = "Joaquin Iturriza Ramirez"
MODEL_TYPE = "Regressor"
MODEL_LICENSE = "CC BY 4.0"

OUTPUT_NAMES = ["nLL_exp", "nLL_obs", "nLLA_exp", "nLLA_obs"]
MU0_KEYS = ["nLL_exp_mu0", "nLL_obs_mu0", "nLLA_exp_mu0", "nLLA_obs_mu0"]
MAX_KEYS = ["nLL_exp_max", "nLL_obs_max", "nLLA_exp_max", "nLLA_obs_max"]

# Source documentation per ATLAS analysis. `reference` is the DOI of the HEPData
# resource holding the full-likelihood archive (checked against hepdata.net);
# the file inside it is recorded per model (`filename`, its path inside the
# archive, checked against the archive listing), since one archive carries
# several background-only models.
ANALYSES = {
    "ATLAS-SUSY-2018-04": {"arxiv": "1911.06660", "inspire_id": "1765529",
                           "doi": "10.17182/hepdata.92006.v2/r2"},
    "ATLAS-SUSY-2018-16": {"arxiv": "1911.12606", "inspire_id": "1767649",
                           "doi": "10.17182/hepdata.91374.v5/r6",
                           # the background-only files sit in this folder of the archive
                           "archive_dir": "statistical_models/"},
    "ATLAS-SUSY-2018-32": {"arxiv": "1908.08215", "inspire_id": "1750597",
                           "doi": "10.17182/hepdata.89413.v4/r5"},
    "ATLAS-SUSY-2019-08": {"arxiv": "1909.09226", "inspire_id": "1755298",
                           "doi": "10.17182/hepdata.90607.v4/r3"},
    "ATLAS-SUSY-2019-09": {"arxiv": "2106.01676", "inspire_id": "1866951",
                           "doi": "10.17182/hepdata.95751.v2/r3"},
}
# Data-generation settings recorded by the generation pipeline, kept verbatim.
# start_method is how the sampler draws its starting points
# (default/random/fine-tune/edges; ML_LHClikelihoods sampling/utils.py), not the
# multiprocessing start method (hard-coded 'spawn' in sampling/sample.py).
# 'start method' (with a space) is a stale entry of sampling/default_params.py
# that the sampler never reads: start_method wins when both are present.
# 'scan' is an older spelling of scans.
# Every file carries every one of these keys, null where the pipeline did not
# record it for that scan (OLLL: the same fields in every published file).
# Not included: bkgfiles (source.statistical_model.filename says it; the list
# named every model of the archive) and 'modified' (written by no known code,
# meaning unknown). patchsets is reduced to the patchset the scan used, as its
# path in the archive.
GENERATION_KEYS = [
    "analysis", "analysis_altname", "analyses", "patchsets", "merged",
    "fit_bkg", "scan_criterion", "scans", "points", "total_points", "seed",
    "start_method", "cluster", "bkg_unc_samples", "low_lim_samples",
    "SR_sigma", "CR_sigma", "VR_sigma", "CR_center", "VR_center",
    "signal_leakage_CR", "signal_leakage_CR_spread", "signal_leakage_CR_sign",
    "signal_leakage_VR", "signal_leakage_VR_spread", "signal_leakage_VR_sign",
    "lower_limits", "upper_limits", "initial_lower_limits",
    "folder_name", "filtering_applied",
]
LEGACY_SPELLINGS = {"start method": "start_method", "scan": "scans"}


def normalize_generation(raw):
    """Generation settings keyed by GENERATION_KEYS, legacy spellings folded in.

    `raw` maps key -> decoded value. The current spelling always wins; a 'scan'
    that disagrees with 'scans' is an error, not something to choose silently.
    """
    out = {}
    if "start method" in raw:
        out["start_method"] = raw["start method"]   # only if start_method is absent
    if "scan" in raw:
        if "scans" in raw:
            assert raw["scan"] == raw["scans"], f"scan={raw['scan']!r} vs scans={raw['scans']!r}"
        out["scans"] = raw["scan"]
    for k in GENERATION_KEYS:
        if k in raw:
            out[k] = raw[k]
    return out


SQRT_S_TEV = 13.0
LUMINOSITY_IFB = 139.0

PREPROCESSING_NOTE = (
    "Inputs: total yields (background + signal) of the active bins, in the order "
    "of obs_yields after dropping remove_channels. Outputs: one tensor [batch, 8] "
    "= preprocessed deltas nLL(mu=1)-nLL(mu=0) for [nLL_exp, nLL_obs, nLLA_exp, "
    "nLLA_obs], then their uncertainties in the same order. Operations: "
    "log_w_negatives(x) = sign(x)*log(1+|x|); log(x) = ln(x); "
    "asinh(x) = arcsinh(x/asinh_scale); standardization(x) = (x-mean)/std with "
    "the per-column mean/std in 'standardization'. Undo the nLL pipeline in "
    "reverse order, then add *_mu0 to obtain nLL(mu=1)."
)


def _dumps(value):
    return json.dumps(value)


def features_pipeline(data_cfg):
    """Flat list of feature trafos, in application order."""
    pipe = []
    for fns in (data_cfg.get("trafos") or {}).values():
        if isinstance(fns, list):
            pipe.extend(fns)
    return pipe


def nlls_pipeline(data_cfg):
    """nLL trafos, as a list (shared by all outputs) or keyed by output name.

    The training config writes per-output pipelines as
    {"per_output": [[...], ...], "asinh_scale": s}; here each list is keyed by
    the output it applies to, which reads unambiguously and is what the v0.1
    validator accepts. The run_config keeps the training form, which is what
    the adapters decode.
    """
    spec = data_cfg.get("nLL_trafos") or []
    if isinstance(spec, dict) and "per_output" in spec:
        out = {"per_output": {name: list(p) for name, p in zip(OUTPUT_NAMES, spec["per_output"])}}
        if "asinh_scale" in spec:
            out["asinh_scale"] = float(spec["asinh_scale"])
        return out
    return list(spec)


def model_parameters(cfg):
    net, tr = cfg["model"]["net"], cfg["training"]
    width, depth = int(net["hidden_channels"]), int(net["hidden_layers"])
    params = OrderedDict([
        ("architecture", net["_target_"].rsplit(".", 1)[-1]),
        ("parametrization", "muP"),
        ("n_inputs", int(net["n_features"])),
        ("hidden_layers", [width] * depth),
        ("activation", str(net.get("activation", "gelu")).upper()),
        ("out_shape", 4),
        ("graph_outputs", 8),
        ("loss", tr.get("loss")),
        ("optimizer", tr.get("optimizer")),
        ("learning_rate", tr.get("lr")),
        ("batch_size", tr.get("batchsize")),
        ("iterations", tr.get("iterations")),
        ("regularization", tr.get("regularization")),
        ("regularization_lambda", tr.get("regularization_lambda")),
    ])
    return params


def training_date_from_run_name(run_name):
    """'20260203_060932_MuMLP_8815' -> '2026-02-03'; None if not a timestamp."""
    m = re.match(r"^(\d{4})(\d{2})(\d{2})_\d{6}", str(run_name or ""))
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def active_statistical_model(reference):
    """(bkg-only filename, its channel map) of the model the scan was drawn from.

    Multi-model analyses record every background file, patchset and channel map
    of the archive in parallel lists, with the patchset used flagged true. Only
    that entry describes this model.
    """
    bkgfiles = reference["bkgfiles"]
    channels = reference["channels"]
    patchsets = reference.get("patchsets") or []
    flags = [bool(p[1]) for p in patchsets if isinstance(p, list) and len(p) == 2]
    if flags:
        assert sum(flags) == 1, f"expected exactly one active patchset, got {patchsets}"
        idx = flags.index(True)
    else:
        assert len(bkgfiles) == 1, f"no patchset flag to choose among {bkgfiles}"
        idx = 0
    chan = channels[idx] if isinstance(channels, list) else channels
    return bkgfiles[idx], dict(chan)


def build_metadata(*, analysis_id, model_name, run_config, standardization, reference,
                   bounds, mu0, generation, training_dataset,
                   training_date=None, training_duration=None):
    """OLLL v0.1 metadata as an ordered {key: string} map.

    run_config       YAML text of the run config that produced the weights,
                     stored verbatim (the adapters decode nLL_trafos from it)
    standardization  {"features_mean","features_std","nLLs_mean","nLLs_std"[,...]}
    reference        decoded reference metadata: channels, bkgfiles, patchsets,
                     obs_yields, bkg_yields, bkg_unc, removeCRsVRs,
                     remove_channels, nLL_*_max
    bounds           {"x_min","x_max","y_min","y_max"} over the training block
    mu0              four mu=0 baselines, in OUTPUT_NAMES order
    generation       normalize_generation(...) of the generation pipeline's record
    training_dataset {"name", "n_rows", "filtering"}: what we trained on and how
                     it was derived from the generated scan
    """
    import yaml

    cfg = yaml.safe_load(run_config)
    src = ANALYSES[analysis_id]
    bkgfile, channels = active_statistical_model(reference)
    data_cfg = cfg["data"]

    std = {k: standardization[k] for k in
           ("features_mean", "features_std", "nLLs_mean", "nLLs_std", "nLLs_lo", "nLLs_hi")
           if k in standardization}

    meta = OrderedDict()
    meta["olll_schema_version"] = SCHEMA_VERSION
    meta["model_type"] = MODEL_TYPE
    meta["model_name"] = model_name
    meta["model_author"] = MODEL_AUTHOR
    meta["model_license"] = MODEL_LICENSE
    meta["model_parameters"] = _dumps(model_parameters(cfg))
    if training_date:
        meta["training_date"] = training_date
    if training_duration:
        meta["training_duration"] = training_duration
    meta["run_config"] = run_config
    meta["source"] = _dumps(OrderedDict([
        ("analysis_id", analysis_id),
        ("arxiv", src["arxiv"]),
        ("inspire_id", src["inspire_id"]),
        ("experiment", "ATLAS"),
        ("sqrt_s", SQRT_S_TEV),
        ("luminosity", LUMINOSITY_IFB),
        ("statistical_model", OrderedDict([
            ("type", "histfactory"),
            ("reference", f"https://doi.org/{src['doi']}"),
            ("filename", src.get("archive_dir", "") + bkgfile),
        ])),
    ]))
    meta["input_type"] = "total_yields"
    meta["channels"] = _dumps(channels)
    for key in ("obs_yields", "bkg_yields", "bkg_unc"):
        meta[key] = _dumps(reference[key])
    meta["removeCRsVRs"] = _dumps(bool(reference["removeCRsVRs"]))
    meta["remove_channels"] = _dumps(list(reference["remove_channels"]))
    for key in ("x_min", "x_max", "y_min", "y_max"):
        meta[key] = _dumps([float(v) for v in bounds[key]])
    meta["preprocessing"] = _dumps(OrderedDict([
        ("features_pipeline", features_pipeline(data_cfg)),
        ("nLLs_pipeline", nlls_pipeline(data_cfg)),
        ("note", PREPROCESSING_NOTE),
    ]))
    meta["standardization"] = _dumps(std)
    for key, val in zip(MU0_KEYS, mu0):
        meta[key] = repr(float(val))
    for key in MAX_KEYS:
        mu_hat, nll = reference[key]
        meta[key] = _dumps([float(mu_hat), float(nll)])
    meta["training_dataset"] = _dumps(training_dataset)
    for key in GENERATION_KEYS:
        value = generation.get(key)
        if key == "patchsets" and isinstance(value, list):
            # the patchset(s) the scan used, as paths inside the archive (like
            # source.statistical_model.filename). The pipeline wrote either plain
            # names or [name, used] pairs over every patchset of the archive.
            used = [p[0] for p in value if isinstance(p, list) and len(p) == 2 and p[1]]
            names = used or [p for p in value if isinstance(p, str)]
            value = [src.get("archive_dir", "") + n for n in names]
        meta[key] = _dumps(value)
    return meta


def write_metadata(model, meta):
    """Replace the model's metadata_props with `meta`, in order, each key once."""
    del model.metadata_props[:]
    for k, v in meta.items():
        p = model.metadata_props.add()
        p.key, p.value = k, str(v)
    keys = [p.key for p in model.metadata_props]
    assert len(keys) == len(set(keys)), "duplicate metadata keys"
    return model
