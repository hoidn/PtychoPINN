"""Shared visualization defaults for PDEBench image-suite comparison figures."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


_CFD_CNS_FIELD_CMAPS = {
    "density": "cividis",
    "Vx": "RdBu_r",
    "Vy": "PuOr",
    "pressure": "magma",
}
_CFD_CNS_SIGNED_FIELDS = {"Vx", "Vy"}
_DEFAULT_ERROR_CMAP = "magma"
_DEFAULT_SCALAR_CMAP = "viridis"


def cfd_cns_field_visual_spec(
    field_name: str,
    arrays: Sequence[np.ndarray],
    *,
    is_error: bool = False,
) -> dict[str, Any]:
    if not arrays:
        raise ValueError("arrays must contain at least one image")
    normalized_field = str(field_name)
    stack = [np.asarray(item, dtype=np.float32) for item in arrays]
    if is_error:
        vmax = float(max(np.max(item) for item in stack))
        return {"cmap": _DEFAULT_ERROR_CMAP, "vmin": 0.0, "vmax": vmax}
    if normalized_field in _CFD_CNS_SIGNED_FIELDS:
        vmax = float(max(np.max(np.abs(item)) for item in stack))
        return {
            "cmap": _CFD_CNS_FIELD_CMAPS.get(normalized_field, _DEFAULT_SCALAR_CMAP),
            "vmin": -vmax,
            "vmax": vmax,
        }
    return {
        "cmap": _CFD_CNS_FIELD_CMAPS.get(normalized_field, _DEFAULT_SCALAR_CMAP),
        "vmin": float(min(np.min(item) for item in stack)),
        "vmax": float(max(np.max(item) for item in stack)),
    }
