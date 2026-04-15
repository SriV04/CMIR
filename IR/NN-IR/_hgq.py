"""HGQ quantizer accessors used by the NN-IR builder.

HGQ layers expose three quantizer attributes at inference time: `iq` (input),
`kq` (kernel), `bq` (bias). Each quantizer holds Keras Variables whose names
end with a short tag: `b` is the integer-bit count and `f` is the fractional
bit count. The effective per-parameter bitwidth is the absolute value of those
tensors (HGQ stores signed values).
"""

from __future__ import annotations

import numpy as np


def _bw_variable(quantizer):
    if quantizer is None or not hasattr(quantizer, "variables"):
        return None
    for v in quantizer.variables:
        tag = v.name.split("/")[-1]
        if tag in ("b", "f"):
            return v
    return None


def bw_array(quantizer) -> np.ndarray | None:
    """Return the full per-parameter bitwidth array, or None."""
    v = _bw_variable(quantizer)
    if v is None:
        return None
    return np.abs(np.array(v.value)).astype(float)


def avg_bw(quantizer) -> float | None:
    """Return the mean absolute bitwidth across all parameters, or None."""
    arr = bw_array(quantizer)
    if arr is None:
        return None
    return float(arr.mean())


def max_bw(quantizer) -> float | None:
    """Return the max absolute bitwidth (worst-case wire width), or None."""
    arr = bw_array(quantizer)
    if arr is None:
        return None
    return float(arr.max())


def sparsity(kernel, tol: float = 1e-12) -> float | None:
    """Fraction of entries in `kernel` that are (near) zero."""
    if kernel is None:
        return None
    arr = np.array(kernel)
    if arr.size == 0:
        return None
    return float((np.abs(arr) <= tol).sum()) / float(arr.size)
