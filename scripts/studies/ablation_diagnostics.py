"""Pure post-hoc checks for the VarPro/probe ablation harness (Task 1.5, F3/F4).

No torch, no config objects -- numpy + stdlib only, so these run instantly in
tests without training and can also gate the real runner at the end of every
arm (``validate_arm_outputs`` is called by ``run_arm`` and raises loudly on
any problem -- see ``varpro_probe_ablation_runner.py``).

Two independent checks:
  - ``validate_arm_outputs`` (F3): artifact completeness, metric finiteness,
    and probe-vs-uniform canvas non-triviality for one arm's on-disk outputs.
  - ``canvas_rail_diagnostics`` (F4): degeneracy diagnostics for a single
    reconstruction canvas, so a saturated/degenerate checkpoint (e.g. the
    1-epoch smoke arm) is visible from ``metrics.json`` alone. The rails
    correspond to the decoder heads' hardwired activations
    (``ptycho_torch/model.py:430-449``): ``Decoder_last_Amp`` uses
    ``ScaledTanh(scale=1.0, offset=0.2)`` -> real part in ``[-0.8, 1.2]``;
    ``Decoder_last_Phase`` uses ``ScaledTanh(scale=1.2)`` -> imag part in
    ``[-1.2, 1.2]``. A fully-saturated decoder emits real/imag values sitting
    almost exactly on those rails everywhere.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

REQUIRED_VARIANT_FILES = ("canvas.npz", "metrics.json", "recon_panel.png", "error.png")

_AMP_RAILS = (-0.8, 1.2)
_PHASE_RAILS = (-1.2, 1.2)
_RAIL_ATOL = 1e-3


def validate_arm_outputs(arm_dir: Path, variants: List[str]) -> List[str]:
    """Check one arm's on-disk artifact contract. Returns problem strings
    (empty list == clean). Does not raise -- callers decide how to react."""
    problems: List[str] = []
    canvases: Dict[str, np.ndarray] = {}

    for variant in variants:
        variant_dir = arm_dir / variant
        for fname in REQUIRED_VARIANT_FILES:
            if not (variant_dir / fname).exists():
                problems.append(f"{variant}: missing {fname}")

        metrics_path = variant_dir / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not np.isfinite(value):
                    problems.append(f"{variant}: metrics['{key}'] is non-finite ({value})")

        canvas_path = variant_dir / "canvas.npz"
        if canvas_path.exists():
            with np.load(canvas_path) as data:
                canvases[variant] = data["canvas"]

    probe_variants = [v for v in canvases if v.startswith("probe")]
    uniform_variants = [v for v in canvases if v.startswith("uniform")]
    for probe_variant in probe_variants:
        for uniform_variant in uniform_variants:
            if np.array_equal(canvases[probe_variant], canvases[uniform_variant]):
                problems.append(
                    f"{probe_variant} and {uniform_variant} canvases are identical "
                    "(probe vs. uniform weighting must produce different reconstructions)"
                )
    return problems


def canvas_rail_diagnostics(canvas: np.ndarray) -> Dict[str, float]:
    """Pure-numpy degeneracy diagnostics for one reconstruction canvas.

    Returns ``rail_fraction_real``/``rail_fraction_imag`` (fraction of pixels
    within ``1e-3`` of the amp/phase decoder heads' tanh rails) and
    ``canvas_phase_std``/``canvas_amp_std`` (std of angle/|canvas| restricted
    to pixels with ``|canvas| > 0.1 * max(|canvas|)``, so background/empty
    canvas regions don't dilute the phase statistic).
    """
    canvas = np.asarray(canvas)
    real, imag, amp = canvas.real, canvas.imag, np.abs(canvas)

    real_rail = np.isclose(real, _AMP_RAILS[0], atol=_RAIL_ATOL) | np.isclose(real, _AMP_RAILS[1], atol=_RAIL_ATOL)
    imag_rail = np.isclose(imag, _PHASE_RAILS[0], atol=_RAIL_ATOL) | np.isclose(imag, _PHASE_RAILS[1], atol=_RAIL_ATOL)

    amp_max = float(amp.max()) if amp.size else 0.0
    mask = amp > 0.1 * amp_max if amp_max > 0 else np.zeros_like(amp, dtype=bool)

    return {
        "rail_fraction_real": float(np.mean(real_rail)),
        "rail_fraction_imag": float(np.mean(imag_rail)),
        "canvas_phase_std": float(np.std(np.angle(canvas[mask]))) if mask.any() else 0.0,
        "canvas_amp_std": float(np.std(amp[mask])) if mask.any() else 0.0,
    }
