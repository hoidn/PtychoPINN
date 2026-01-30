"""MLflow intermediate reconstruction logging callback (Torch Lightning)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import torch


@dataclass
class PtychoReconLoggingCallback:
    """MLflow recon logging callback (patch logging scaffold)."""

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

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        array = tensor.detach().cpu().numpy()
        return np.squeeze(array)

    def _log_patch_bundle(self, trainer, pl_module, idx: int, batch) -> None:
        data, probe, _scale = batch
        images = data["images"]
        coords = data["coords_relative"]
        rms_scale = data["rms_scaling_constant"]
        experiment_ids = data.get("experiment_id")

        pred_diffraction, amp_pred, phase_pred = pl_module.forward(
            images,
            coords,
            probe,
            rms_scale,
            rms_scale,
            experiment_ids=experiment_ids,
        )

        diff_pred_log = torch.log1p(torch.abs(pred_diffraction))
        diff_obs_log = torch.log1p(torch.abs(images))

        prefix = f"{self.artifact_root}/patch_{idx}"
        self._log_image(trainer, f"{prefix}/amp_pred", self._to_numpy(amp_pred))
        self._log_image(trainer, f"{prefix}/phase_pred", self._to_numpy(phase_pred))
        self._log_image(trainer, f"{prefix}/diff_pred_logI", self._to_numpy(diff_pred_log))
        self._log_image(trainer, f"{prefix}/diff_obs_logI", self._to_numpy(diff_obs_log))

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        indices: Iterable[int] = self.fixed_indices or list(range(self.num_patches))
        for idx in indices:
            batch = self._get_patch_batch(idx)
            self._log_patch_bundle(trainer, pl_module, idx, batch)
