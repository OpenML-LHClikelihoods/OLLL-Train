import numpy as np
import torch
from scipy.stats import boxcox
from sklearn.preprocessing import QuantileTransformer


def logit_bounded(nLLs, margin=0.01):
    """Map each output column from [lo, hi] -> (-inf, inf) via logit.

    Bounds are derived from the data with a small margin so that the
    training maximum/minimum does not sit exactly at the logit singularity.
    Returns (transformed, lo, hi) so the caller can store the bounds for
    the inverse transform.
    """
    lo = nLLs.min(axis=0) - margin
    hi = nLLs.max(axis=0) + margin
    p = np.clip((nLLs - lo) / (hi - lo), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)), lo, hi


def undo_logit_bounded(nLLs, lo, hi):
    """Inverse of logit_bounded: sigmoid rescaled to [lo, hi]."""
    return lo + (hi - lo) / (1.0 + np.exp(-nLLs))


def _is_per_output(trafos):
    """True if `trafos` is a per-output spec (a mapping carrying 'per_output')."""
    try:
        return "per_output" in trafos
    except TypeError:
        return False


def _apply_col_trafo(col, fn_str, scale):
    """Forward per-column nLL trafo (standardization handled by the caller)."""
    if fn_str == "log":
        return np.log(col)
    elif fn_str == "log_w_negatives":
        return log_with_negatives(col)
    elif fn_str == "asinh":
        return np.arcsinh(col / scale)
    else:
        raise ValueError(f"unsupported per-output nLL trafo: {fn_str!r}")


def _undo_col_trafo(col, fn_str, scale):
    """Exact inverse of _apply_col_trafo."""
    if fn_str == "log":
        return np.exp(col)
    elif fn_str == "log_w_negatives":
        return undo_log_with_negatives(col)
    elif fn_str == "asinh":
        return scale * np.sinh(col)
    else:
        raise ValueError(f"unsupported per-output nLL trafo: {fn_str!r}")


def _preprocess_nLLs_per_output(nLLs, spec, fixed_mean=None, fixed_std=None, n_fit=None):
    """Apply a separate trafo pipeline to each nLL output column.

    `spec` is a mapping with:
      - per_output: list (length = n outputs) of trafo-lists, e.g.
        [[log, standardization], [asinh, standardization], ...] applied to the
        columns in order (after any target_indices restriction).
      - asinh_scale: scalar scale s used by the 'asinh' trafo (default 1.0);
        asinh(x/s) is linear for |x|<<s and logarithmic for |x|>>s.

    standardization, where it appears, is applied per column at its position in
    the pipeline; the per-column mean/std are returned for exact inversion.

    If `fixed_mean`/`fixed_std` (arrays of length n) are given, the standardization
    step uses them instead of refitting from the data — needed for finetuning, so
    the target normalization stays pinned to the pretrained model's stats.
    """
    per_output = [list(p) for p in spec["per_output"]]
    scale = float(spec.get("asinh_scale", 1.0))
    n = nLLs.shape[1]
    assert len(per_output) == n, (
        f"per_output length {len(per_output)} != number of nLL outputs {n}"
    )
    fixed = fixed_mean is not None and fixed_std is not None
    if fixed:
        fixed_mean = np.asarray(fixed_mean, dtype=np.float64).ravel()
        fixed_std = np.asarray(fixed_std, dtype=np.float64).ravel()
    out = np.array(nLLs, dtype=np.float64)
    mean = np.zeros(n)
    std = np.ones(n)
    for c in range(n):
        col = out[:, c]
        for fn_str in per_output[c]:
            if fn_str == "standardization":
                if fixed:
                    m, s = fixed_mean[c], float(np.clip(fixed_std[c], 1e-6, None))
                else:
                    fit = col[:n_fit] if n_fit is not None else col
                    m = fit.mean()
                    s = np.clip(fit.std(), 1e-6, None)
                col = (col - m) / s
                mean[c], std[c] = m, s
            else:
                col = _apply_col_trafo(col, fn_str, scale)
        out[:, c] = col
    nll_bounds = {"per_output": per_output, "asinh_scale": scale}
    assert np.isfinite(out).all()
    return out, mean, std, nll_bounds


def _undo_preprocess_nLLs_per_output(nLLs, mean, std, spec, nll_bounds):
    """Exact inverse of _preprocess_nLLs_per_output."""
    nll_bounds = nll_bounds or {}
    per_output = [list(p) for p in spec["per_output"]]
    scale = float(spec.get("asinh_scale", nll_bounds.get("asinh_scale", 1.0)))
    n = nLLs.shape[1]
    out = np.array(nLLs, dtype=np.float64)
    for c in range(n):
        col = out[:, c]
        for fn_str in reversed(per_output[c]):
            if fn_str == "standardization":
                col = col * std[c] + mean[c]
            else:
                col = _undo_col_trafo(col, fn_str, scale)
        out[:, c] = col
    assert np.isfinite(out).all()
    return out


def preprocess_nLLs(nLLs, trafos=None, fixed_mean=None, fixed_std=None, n_fit=None):
    mean, std = 0.0, 1.0
    nll_bounds = {}

    if trafos and _is_per_output(trafos):
        return _preprocess_nLLs_per_output(nLLs, trafos, fixed_mean, fixed_std, n_fit)

    if trafos:
        for fn_str in trafos:
            print("Applying nLL transformation:", fn_str)

            if fn_str == "standardization":
                nLLs, mean, std = standardization(
                    nLLs, return_mean_std=True, clip=False, n_fit=n_fit
                )

            elif fn_str == "log_w_negatives":
                nLLs = log_with_negatives(nLLs)

            elif fn_str == "logit_bounded":
                nLLs, lo, hi = logit_bounded(nLLs)
                nll_bounds["lo"] = lo
                nll_bounds["hi"] = hi

            else:
                fn = get_fn(fn_str)
                nLLs = fn(nLLs, None)

    assert np.isfinite(nLLs).all()
    return nLLs, mean, std, nll_bounds


def log_with_negatives(nLLs):
    """
    Signed log1p transform.
    Works for negative values and is exactly invertible.
    """
    return np.sign(nLLs) * np.log1p(np.abs(nLLs))

def undo_log_with_negatives(nLLs):
    """
    Exact inverse of log_with_negatives.
    """
    return np.sign(nLLs) * (np.expm1(np.abs(nLLs)))



def undo_preprocess_nLLs(nLLs, mean, std, trafos=None, nll_bounds=None):
    if trafos and _is_per_output(trafos):
        return _undo_preprocess_nLLs_per_output(nLLs, mean, std, trafos, nll_bounds)

    nll_bounds = nll_bounds or {}
    if trafos:
        for fn_str in reversed(trafos):

            if fn_str == "standardization":
                nLLs = nLLs * std + mean

            elif fn_str == "log_w_negatives":
                nLLs = undo_log_with_negatives(nLLs)

            elif fn_str == "logit_bounded":
                nLLs = undo_logit_bounded(nLLs, nll_bounds["lo"], nll_bounds["hi"])

            else:
                inv_fn = get_inv_fn(fn_str)
                nLLs = inv_fn(nLLs, None)

    assert np.isfinite(nLLs).all()
    return nLLs



def preprocess_features(
    features_raw,
    type_tokens=None,
    trafos=None,
    incl_fvs=True,
    mean=None,
    std=None,
    eps_std=1e-2,
    return_dict=False,
    n_fit=None,
):
    assert np.isfinite(features_raw).all()
    
    feature_sets = {}
    if trafos:
        for t_name, t_fns in trafos.items():
            transformed_features = features_raw
            for fn_str in t_fns:
                if fn_str == "standardization":
                    transformed_features, mean, std = standardization(
                        transformed_features, return_mean_std=True, clip=False,
                        use_previous_mean_std=(mean is not None and std is not None),
                        mean=mean, std=std, n_fit=n_fit,
                    )
                else:
                    fn = get_fn(fn_str)
                    transformed_features = fn(transformed_features, type_tokens)
            feature_sets[t_name] = transformed_features

    # if incl_fvs:
    #     feature_sets["fvs_raw"] = sort_features(features_raw, type_tokens, "E")
    #     feature_sets["fvs_raw"] /= np.max(
    #         np.abs(feature_sets["fvs_raw"][:, ::4])
    #     )  # normalize by max energy of features
    #     # feature_sets["fvs_raw"] = features_raw

    if return_dict:
        return feature_sets, mean, std
    else:
        return np.concatenate(list(feature_sets.values()), axis=1), mean, std


def standardization(features, return_mean_std=False, clip=True, use_previous_mean_std=False, mean=None, std=None, n_fit=None):
    """standardize features

    n_fit: fit mean/std on the first n_fit rows only (the training block; rows
    are already shuffled and ordered train|val|test by init_data) but apply the
    transform to all rows. None fits on everything, which leaks val/test column
    statistics into the normalisation.
    """
    if use_previous_mean_std:
        assert mean is not None and std is not None, "Mean and std must be provided if use_previous_mean_std is True"
    else:
        fit = features[:n_fit] if n_fit is not None else features
        mean = fit.mean(axis=0)
        std = fit.std(axis=0)
        # A column that is exactly constant carries no information; (x-mean)/std
        # is 0/0 there. std=1 maps it to 0 and stays exactly invertible. Only
        # triggers on std == 0, which otherwise divides by zero and trips the
        # assert below, so this cannot change any run that currently works.
        # Small scans hit this: a signal region no scanned mass point populates
        # is identically 0 across the whole set.
        std = np.where(std == 0.0, 1.0, std)
    if clip:
        std = np.clip(std, a_min=1e-6, a_max=None)  # avoid std=0.
    features = (features - mean) / std
    assert np.isfinite(features).all()
    if return_mean_std:
        return features, mean, std
    else:
        return features


def compute_invariants(features, eps=1e-4, incl_diag_invariants=False, reshape=True):
    """compute Lorentz invariants out of four-vectors"""
    if reshape:
        ps = features.reshape(features.shape[0], features.shape[1] // 4, 4)
    else:
        ps = features

    # compute matrix of all inner products
    def inner_product(p1, p2):
        return p1[..., 0] * p2[..., 0] - (p1[..., 1:] * p2[..., 1:]).sum(axis=-1)

    if incl_diag_invariants:
        offset = 0
    else:
        offset = 1

    idxs = np.triu_indices(ps.shape[-2], k=offset)
    invariants = inner_product(ps[..., idxs[0], :], ps[..., idxs[1], :])
    if type(invariants) == np.ndarray:
        invariants = np.clip(invariants, a_min=eps, a_max=None)
    else:
        invariants = torch.clip(invariants, eps, None)
    return invariants


def sort_features(features, type_tokens, sort_key):
    """sort features according to some property"""
    ps = features.reshape(features.shape[0], features.shape[1] // 4, 4)
    particle_types = np.unique(type_tokens)
    sorted_ps = []
    for particle_type in particle_types:
        mask = type_tokens == particle_type
        # sort according to energy, which is the first entry of the four-vector
        match sort_key:
            case "E":
                sorted_indices = np.argsort(ps[:, mask][:, :, 0])
            case "pt":
                pts = np.linalg.norm(ps[:, mask][:, :, 1:3], axis=-1)
                sorted_indices = np.argsort(pts)
            case _:
                raise ValueError(f"Unknown sort key {sort_key}")
        sorted_ps.append(
            np.take_along_axis(ps[:, mask], sorted_indices[:, :, np.newaxis], axis=1)
        )
    return np.concatenate(sorted_ps, axis=1).reshape(features.shape)


def apply_boxcox(features):
    ps = features.transpose()
    res = np.zeros(ps.shape)
    for i in range(ps.shape[0]):
        try:
            res[i] = boxcox(ps[i])[0]
        except:
            res[i] = ps[i]
    return res.transpose()


def apply_quantile_transform(features):
    return QuantileTransformer(output_distribution="normal").fit_transform(features)


def get_fn(fn_str):
    # get functions from string
    match fn_str:
        case "log":
            return lambda p, t: np.log(p)
        case "exp":
            return lambda p, t: np.exp(p)
        case "sqrt":
            return lambda p, t: np.sqrt(p)
        case "inverse":
            return lambda p, t: 1 / p
        case "boxcox":
            return lambda p, t: apply_boxcox(p)
        case "quantile_transform":
            return lambda p, t: apply_quantile_transform(p)
        case "invs":
            return lambda p, t: compute_invariants(p)
        case "standardization":
            return lambda p, t: standardization(p)
        case "log_w_negatives":
            return lambda p, t: log_with_negatives(p)
        case "sort_E":
            return lambda p, t: sort_features(p, t, "E")
        case "sort_pt":
            return lambda p, t: sort_features(p, t, "pt")
        case _:
            raise ValueError(f"Unknown transformation function {fn_str}")


def get_inv_fn(fn_str):
    # get inverse functions from string
    match fn_str:
        case "log":
            return lambda p, t: np.exp(p)
        case "exp":
            return lambda p, t: np.log(p)
        case "sqrt":
            return lambda p, t: p**2
        case "inverse":
            return lambda p, t: 1 / p
        case _:
            raise ValueError(f"Unknown inverse transformation function {fn_str}")
