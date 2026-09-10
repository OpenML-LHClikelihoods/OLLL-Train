#!/usr/bin/env python3
"""
.. module:: preprocessing_nnAdapter
   :synopsis: Preprocessing and inverse-preprocessing functions for the
   NNAdapter. Handles feature standardisation and nLL postprocessing,
   including uncertainty propagation through arbitrary transformation chains.

"""

__all__ = [
    "preprocess_features",
    "undo_preprocess_nLLs",
    "undo_preprocess_nLLs_errors",
]

import numpy as np
import autograd
import autograd.numpy as anp
from typing import Optional


# ---------------------------------------------------------------------------
# Elementary transforms (forward and inverse)
# Forward transforms use plain numpy (no need to differentiate through them).
# Inverse transforms use autograd.numpy so that undo_preprocess_nLLs_errors
# can differentiate through them with autograd.elementwise_grad.
# ---------------------------------------------------------------------------

def _log_with_negatives(x: np.ndarray) -> np.ndarray:
    """Signed log1p transform, works for negative values, exactly invertible."""
    return np.sign(x) * np.log1p(np.abs(x))


def _undo_log_with_negatives(x) -> np.ndarray:
    """Exact inverse of _log_with_negatives. Uses anp for autograd compatibility."""
    return anp.sign(x) * anp.expm1(anp.abs(x))


def _standardize(x, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply standardisation using pre-computed mean and std."""
    return (x - mean) / std


def _undo_standardize(x, mean: np.ndarray, std: np.ndarray):
    """Invert standardisation. Uses anp for autograd compatibility."""
    return x * std + mean


# ---------------------------------------------------------------------------
# Feature preprocessing (forward, used at inference time)
# ---------------------------------------------------------------------------

def _get_fn(fn_str: str):
    """Return a named scalar transform (forward direction, plain numpy)."""
    match fn_str:
        case "log":
            return np.log
        case "exp":
            return np.exp
        case "sqrt":
            return np.sqrt
        case "inverse":
            return lambda x: 1.0 / x
        case "log_w_negatives":
            return _log_with_negatives
        case _:
            raise ValueError(f"Unknown transform '{fn_str}'.")


def preprocess_features(
    features_raw: np.ndarray,
    trafos: Optional[dict] = None,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Apply feature preprocessing using saved mean/std from training.

    :param features_raw: raw input features, shape (batch, n_features)
    :param trafos: dict mapping set-name to list of transform strings,
        e.g. ``{"fvs_standardized": ["log_w_negatives", "standardization"]}``
    :param mean: per-feature mean saved at training time (required when
        "standardization" is in trafos)
    :param std: per-feature std saved at training time (required when
        "standardization" is in trafos)
    :returns: (scaled_features, mean, std)
    """
    assert np.isfinite(features_raw).all(), "Non-finite values in features_raw"

    feature_sets = {}
    if trafos:
        for set_name, fn_list in trafos.items():
            transformed = features_raw.copy()
            for fn_str in fn_list:
                if fn_str == "standardization":
                    assert mean is not None and std is not None, \
                        "mean and std must be provided for standardization at inference time"
                    transformed = _standardize(transformed, mean, std)
                else:
                    transformed = _get_fn(fn_str)(transformed)
            feature_sets[set_name] = transformed

    scaled = np.concatenate(list(feature_sets.values()), axis=1) if feature_sets else features_raw
    return scaled, mean, std


# ---------------------------------------------------------------------------
# nLL postprocessing (inverse transforms)
# ---------------------------------------------------------------------------

def _get_inv_fn(fn_str: str):
    """Return the inverse of a named scalar transform. Uses anp for autograd compatibility."""
    match fn_str:
        case "log":
            return anp.exp
        case "exp":
            return anp.log
        case "sqrt":
            return lambda x: x ** 2
        case "inverse":
            return lambda x: 1.0 / x
        case "log_w_negatives":
            return _undo_log_with_negatives
        case "logit_bounded":
            raise ValueError(
                "'logit_bounded' requires per-output bounds — pass nll_bounds to undo_preprocess_nLLs."
            )
        case _:
            raise ValueError(f"No known inverse for transform '{fn_str}'.")


def _is_per_output(trafos) -> bool:
    """True if `trafos` is a per-output spec (a mapping carrying 'per_output')."""
    try:
        return "per_output" in trafos
    except TypeError:
        return False


def _undo_preprocess_nLLs_per_output(nLLs, mean, std, spec, nll_bounds):
    """Undo a per-output nLL preprocessing spec (mirror of preprocessing.py).

    `spec` carries ``per_output`` (a list, one trafo-list per output column) and
    ``asinh_scale``. `nLLs`, `mean`, `std` are length-n arrays (per output).
    Uses autograd.numpy so undo_preprocess_nLLs_errors can differentiate through
    it (each column depends only on its own input → elementwise).
    """
    per_output = [list(p) for p in spec["per_output"]]
    scale = float(spec.get("asinh_scale", (nll_bounds or {}).get("asinh_scale", 1.0)))
    cols = []
    for c in range(len(per_output)):
        col = nLLs[c]
        for fn_str in reversed(per_output[c]):
            if fn_str == "standardization":
                col = col * std[c] + mean[c]
            elif fn_str == "asinh":
                col = scale * anp.sinh(col)
            elif fn_str == "log":
                col = anp.exp(col)
            elif fn_str == "log_w_negatives":
                col = _undo_log_with_negatives(col)
            else:
                col = _get_inv_fn(fn_str)(col)
        cols.append(col)
    nLLs = anp.stack(cols)
    assert np.isfinite(nLLs).all(), "Non-finite values after undo_preprocess_nLLs"
    return nLLs


def undo_preprocess_nLLs(
    nLLs: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    trafos: Optional[list] = None,
    nll_bounds: Optional[dict] = None,
) -> np.ndarray:
    """Undo nLL preprocessing in reverse order.

    :param nLLs: preprocessed nLL deltas output by the NN, shape (4,)
    :param mean: per-output mean saved at training time
    :param std: per-output std saved at training time
    :param trafos: list of transform strings applied during training,
        e.g. ``["log_w_negatives", "standardization"]``, or a per-output mapping
        ``{"per_output": [[log, standardization], [asinh, standardization], ...],
        "asinh_scale": s}`` (each output column gets its own pipeline)
    :param nll_bounds: dict with ``"lo"`` and ``"hi"`` arrays required when
        ``"logit_bounded"`` is in trafos
    :returns: unpreprocessed nLL deltas, same shape as nLLs
    """
    if trafos and _is_per_output(trafos):
        return _undo_preprocess_nLLs_per_output(nLLs, mean, std, trafos, nll_bounds)

    nll_bounds = nll_bounds or {}
    if trafos:
        for fn_str in reversed(trafos):
            if fn_str == "standardization":
                nLLs = _undo_standardize(nLLs, mean, std)
            elif fn_str == "logit_bounded":
                lo = np.asarray(nll_bounds["lo"])
                hi = np.asarray(nll_bounds["hi"])
                nLLs = lo + (hi - lo) / (1.0 + anp.exp(-nLLs))
            else:
                nLLs = _get_inv_fn(fn_str)(nLLs)
    assert np.isfinite(nLLs).all(), "Non-finite values after undo_preprocess_nLLs"
    return nLLs


def undo_preprocess_nLLs_errors(
    errors: np.ndarray,
    nLLs_prepd: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    trafos: Optional[list] = None,
    nll_bounds: Optional[dict] = None,
) -> np.ndarray:
    """Propagate heteroskedastic errors through the inverse nLL preprocessing.

    Uses ``autograd.elementwise_grad`` to compute the exact derivative
    ``|d(undo_preprocess)/d(nLL)|`` at the predicted values, then multiplies
    by the NN-predicted uncertainties: ``sigma_out = |df/dx| * sigma_in``.
    This is exact for any composition of elementwise transforms.

    :param errors: NN-predicted uncertainties on the preprocessed deltas,
        shape (4,) — these are the errors on nLLs_prepd
    :param nLLs_prepd: the central (mean) preprocessed nLL deltas, shape (4,)
    :param mean: per-output mean saved at training time
    :param std: per-output std saved at training time
    :param trafos: same transform list used in undo_preprocess_nLLs
    :param nll_bounds: same bounds dict used in undo_preprocess_nLLs
    :returns: propagated uncertainties on the unpreprocessed nLL deltas,
        shape (4,)
    """
    grad_fn = autograd.elementwise_grad(
        lambda x: undo_preprocess_nLLs(x, mean, std, trafos, nll_bounds)
    )
    deriv = np.abs(grad_fn(nLLs_prepd.astype(float)))
    return deriv * errors
