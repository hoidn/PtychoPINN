"""
Patch Statistics Instrumentation Module (FIX-PYTORCH-FORWARD-PARITY-001 Phase A)

This module provides optional patch-level diagnostics for PyTorch forward inference,
emitting JSON statistics and normalized PNG grids when enabled via CLI flags.

Usage:
    from ptycho_torch.patch_stats_instrumentation import PatchStatsLogger

    logger = PatchStatsLogger(
        output_dir=Path("analysis"),
        enabled=True,
        limit=2
    )
    logger.log_batch(amp_tensor, "train", batch_idx=0)
    logger.finalize()  # Write JSON + PNG

References:
    - Phase A plan: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md §A1
    - Brief: input.md (2025-11-16)
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import torch

logger = logging.getLogger(__name__)


class PatchStatsLogger:
    """
    Captures per-patch amplitude/phase statistics during training/inference.

    Attributes:
        enabled: Whether to collect stats (default: False)
        limit: Maximum number of batches to instrument (default: None = unlimited)
        output_dir: Directory for JSON + PNG artifacts
        batch_count: Running count of instrumented batches
        stats: Accumulated statistics per batch
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        enabled: bool = False,
        limit: Optional[int] = None,
    ):
        self.enabled = enabled
        self.limit = limit
        self.output_dir = Path(output_dir) if output_dir else None
        self.batch_count = 0
        self.stats: List[Dict[str, Any]] = []

        if self.enabled and self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def should_log(self) -> bool:
        """Check if current batch should be instrumented."""
        if not self.enabled:
            return False
        if self.limit is None:
            return True
        return self.batch_count < self.limit

    def log_batch(
        self,
        amplitude_tensor: torch.Tensor,
        phase: str = "unknown",
        batch_idx: int = 0,
    ) -> None:
        """
        Record statistics for a single batch of amplitude patches.

        Args:
            amplitude_tensor: Reconstructed amplitude tensor (B, C, H, W)
            phase: Training phase identifier ("train", "val", "inference")
            batch_idx: Batch index in current epoch
        """
        if not self.should_log():
            return

        with torch.no_grad():
            amp_abs = torch.abs(amplitude_tensor.detach())

            # Global stats
            mean_val = float(torch.mean(amp_abs).cpu())
            std_val = float(torch.std(amp_abs).cpu())

            # Zero-mean variance (per-patch normalization check)
            zero_mean = amp_abs - torch.mean(amp_abs, dim=(-2, -1), keepdim=True)
            var_zero_mean = float(torch.mean(zero_mean ** 2).cpu())

            # Per-patch stats (first 4 patches for inspection)
            batch_size = amp_abs.shape[0]
            per_patch = []
            for i in range(min(4, batch_size)):
                patch = amp_abs[i]
                per_patch.append({
                    "patch_idx": i,
                    "mean": float(torch.mean(patch).cpu()),
                    "std": float(torch.std(patch).cpu()),
                    "min": float(torch.min(patch).cpu()),
                    "max": float(torch.max(patch).cpu()),
                })

            # Record
            entry = {
                "phase": phase,
                "batch_idx": batch_idx,
                "batch_size": batch_size,
                "global_mean": mean_val,
                "global_std": std_val,
                "var_zero_mean": var_zero_mean,
                "per_patch": per_patch,
            }
            self.stats.append(entry)
            self.batch_count += 1

            # Console log
            msg = (
                f"Torch patch stats ({phase} batch {batch_idx}): "
                f"mean={mean_val:.6f} std={std_val:.6f} var_zero_mean={var_zero_mean:.6f}"
            )
            logger.info(msg)
            print(msg)

    def finalize(self) -> None:
        """Write accumulated stats to JSON and generate normalized PNG grid."""
        if not self.enabled or not self.output_dir or not self.stats:
            return

        # Write JSON
        json_path = self.output_dir / "torch_patch_stats.json"
        with open(json_path, "w") as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f"Wrote patch stats JSON to {json_path}")

        # Generate PNG grid (placeholder for now; Phase A focuses on JSON)
        # Full implementation would normalize first batch and tile patches
        png_path = self.output_dir / "torch_patch_grid.png"
        _write_placeholder_png(png_path)
        logger.info(f"Wrote patch grid placeholder PNG to {png_path}")


def _write_placeholder_png(path: Path) -> None:
    """Write a minimal placeholder PNG until full grid rendering is implemented."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(
            0.5, 0.5,
            "Patch grid placeholder\n(Phase A skeleton)",
            ha='center', va='center',
            fontsize=12
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close(fig)
    except ImportError:
        # Fallback if matplotlib unavailable
        path.write_text("PNG placeholder (matplotlib not available)\n")
