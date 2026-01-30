"""MLflow intermediate reconstruction logging callback (Torch Lightning)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np


@dataclass
class PtychoReconLoggingCallback:
    """Minimal MLflow recon logging callback (stub for TDD)."""

    every_n_epochs: int
    num_patches: int = 4
    fixed_indices: Optional[List[int]] = None
    log_stitch: bool = False
    artifact_root: str = "recon"

    def _get_patch_batch(self, index: int):
        raise NotImplementedError

    def _log_image(self, trainer, key: str, image: np.ndarray) -> None:
        if trainer.logger is None:
            return
        experiment = getattr(trainer.logger, "experiment", None)
        if experiment is None:
            return
        experiment.log_image(image, key)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        indices: Iterable[int] = self.fixed_indices or list(range(self.num_patches))
        for idx in indices:
            _ = self._get_patch_batch(idx)
            dummy = np.zeros((8, 8), dtype=np.float32)
            self._log_image(trainer, f"{self.artifact_root}/patch_{idx}/amp_pred", dummy)
