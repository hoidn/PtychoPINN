"""One scoring door for the reconstruction metric contract.

Every study scores a stitched canvas through :func:`score_canvas` (or through
``evaluate_reconstruction_quality``, which prepares the comparison the same
way): exposure support from ``probe_exposure_support`` unless an explicit
support is given, anchor alignment and the global gauge from
``prepare_anchor_aligned``, then the maintained metric functions. Study
scripts must not re-list the metric bundle or carry their own FRC.

Contract: roadmap "Reconstruction metric contract", clause 3, and
``docs/superpowers/specs/2026-09-04-reconstruction-global-gauge-design.md``.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

import ptycho_torch.reconstruction_evaluation as evaluation

PRIMARY_METRIC_KEYS = (
    "amp_mae",
    "phase_wrapped_mae",
    "amplitude_ssim",
    "phase_ssim",
    "amplitude_frc50_frequency",
    "phase_frc50_frequency",
)
SCORE_KEYS = (
    *PRIMARY_METRIC_KEYS,
    "absolute_amp_mae",
    "amp_mean_ratio",
    "amplitude_frc50_pixels",
    "phase_frc50_pixels",
    "frc_curves",
    "gauge",
    "valid_pixel_count",
)


def score_canvas(
    canvas: Any,
    weights: Any,
    anchor: Mapping[str, Any],
    truth: Any,
    probe: Any | None = None,
    *,
    metric_crop_border: int = 0,
    metric_support: Any | None = None,
) -> dict[str, Any]:
    """Score one stitched canvas against its truth on the exposure support.

    ``weights`` is the stitch's aggregate-exposure canvas (``canvas_weights``);
    the support is ``evaluation.probe_exposure_support(weights, probe)`` unless
    ``metric_support`` is passed (a boolean canvas-shaped array, used when two
    rows must share one support). The reconstruction is aligned and gauged by
    ``prepare_anchor_aligned``; ``absolute_amp_mae`` and ``amp_mean_ratio``
    describe the amplitude scale the gauge removed.
    """
    if metric_support is None:
        if probe is None:
            raise ValueError("score_canvas needs a probe or an explicit metric_support")
        support = evaluation.probe_exposure_support(weights, probe)
    else:
        support = np.asarray(metric_support)
    prepared = evaluation.prepare_anchor_aligned(
        canvas,
        weights,
        anchor,
        truth,
        metric_crop_border=metric_crop_border,
        metric_support=support,
    )
    if prepared.gauge is None:
        raise ValueError("score_canvas requires the global gauge")
    absolute = evaluation.absolute_scale_metrics(prepared)
    frc = evaluation.frc50_metrics(prepared)
    return {
        "amp_mae": float(absolute["amp_mae"]),
        "phase_wrapped_mae": float(evaluation.phase_wrapped_mae(prepared)),
        "amplitude_ssim": float(evaluation.amplitude_ssim(prepared)),
        "phase_ssim": float(evaluation.phase_ssim(prepared)),
        "amplitude_frc50_frequency": frc["amplitude_frc50_frequency"],
        "phase_frc50_frequency": frc["phase_frc50_frequency"],
        "absolute_amp_mae": float(absolute["absolute_amp_mae"]),
        "amp_mean_ratio": float(absolute["amp_mean_ratio"]),
        "amplitude_frc50_pixels": frc["amplitude_frc50_pixels"],
        "phase_frc50_pixels": frc["phase_frc50_pixels"],
        "frc_curves": frc["frc_curves"],
        "gauge": prepared.gauge.to_jsonable(prepared.reconstruction.shape),
        "valid_pixel_count": int(np.count_nonzero(prepared.common_mask)),
    }


__all__ = ["PRIMARY_METRIC_KEYS", "SCORE_KEYS", "score_canvas"]
