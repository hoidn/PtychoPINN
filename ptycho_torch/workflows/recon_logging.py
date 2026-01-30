"""MLflow intermediate reconstruction logging callback (Torch Lightning).

Logs patch-level and optionally stitched reconstructions to MLflow every N epochs.
Design: docs/plans/2026-01-30-mlflow-intermediate-recon-logging-design.md
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List, Optional

import lightning as L
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless logging
import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)


def _resolve_val_dataloader(trainer: L.Trainer):
    """Extract a single DataLoader from trainer.val_dataloaders.

    Lightning may provide a single DataLoader, a list of DataLoaders,
    or a CombinedLoader. We always want the first one.
    Returns (dataloader, dataset) or (None, None).
    """
    val_dl = trainer.val_dataloaders
    if val_dl is None:
        return None, None

    # Lightning may wrap in a list or CombinedLoader
    if isinstance(val_dl, (list, tuple)):
        if len(val_dl) == 0:
            return None, None
        val_dl = val_dl[0]

    # CombinedLoader has .flattened attribute
    if hasattr(val_dl, 'flattened'):
        flattened = val_dl.flattened
        if flattened:
            val_dl = flattened[0]

    dataset = getattr(val_dl, 'dataset', None)
    if dataset is None:
        return None, None
    return val_dl, dataset


def _to_2d(array: np.ndarray) -> np.ndarray:
    """Squeeze a tensor to 2D for imshow. Takes first channel if multi-channel."""
    arr = np.squeeze(array)
    if arr.ndim == 3:
        # Multi-channel (gridsize > 1): show first channel
        return arr[0]
    return arr


class PtychoReconLoggingCallback(L.Callback):
    """Log intermediate reconstructions to MLflow during training.

    Patch-level: logs amp/phase predictions, diffraction (log-scale),
    and GT + error maps when supervised labels are available.

    Stitched: logs full-resolution stitched reconstructions when enabled.
    Requires a val dataloader with test data.

    Args:
        every_n_epochs: Log every N epochs.
        num_patches: Number of fixed patch indices to sample.
        fixed_indices: Explicit indices (overrides num_patches auto-select).
        log_stitch: Enable stitched full-resolution logging.
        max_stitch_samples: Cap on stitched samples (None = no limit).
    """

    def __init__(
        self,
        every_n_epochs: int = 5,
        num_patches: int = 4,
        fixed_indices: Optional[List[int]] = None,
        log_stitch: bool = False,
        max_stitch_samples: Optional[int] = None,
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_patches = num_patches
        self.fixed_indices = fixed_indices
        self.log_stitch = log_stitch
        self.max_stitch_samples = max_stitch_samples
        self._selected_indices: Optional[List[int]] = None

    def _select_indices(self, dataset_len: int) -> List[int]:
        """Select fixed patch indices deterministically.

        If fixed_indices was provided, use those (clamped to dataset size).
        Otherwise pick num_patches evenly spaced indices.
        """
        if self._selected_indices is not None:
            return self._selected_indices

        if self.fixed_indices is not None:
            self._selected_indices = [i for i in self.fixed_indices if i < dataset_len]
        else:
            n = min(self.num_patches, dataset_len)
            if n == 0:
                self._selected_indices = []
            else:
                step = max(1, dataset_len // n)
                self._selected_indices = [i * step for i in range(n)]

        logger.info("Recon logging: selected patch indices %s (dataset size %d)",
                     self._selected_indices, dataset_len)
        return self._selected_indices

    def _should_log(self, trainer: L.Trainer) -> bool:
        """Check whether to log this epoch."""
        if not trainer.is_global_zero:
            return False
        if trainer.logger is None:
            return False
        epoch = trainer.current_epoch + 1  # 1-indexed for human readability
        return epoch % self.every_n_epochs == 0

    def _get_mlflow_logger(self, trainer: L.Trainer):
        """Return the MLflow experiment client, or None."""
        mlflow_logger = trainer.logger
        if mlflow_logger is None:
            return None
        experiment = getattr(mlflow_logger, 'experiment', None)
        return experiment

    def _log_figure(self, trainer: L.Trainer, artifact_path: str, filename: str, fig) -> None:
        """Log a matplotlib figure to MLflow as a PNG artifact.

        Args:
            artifact_path: MLflow artifact subdirectory (e.g. "epoch_0005/patch_00").
            filename: PNG filename within the artifact path (e.g. "amp_pred.png").
        """
        experiment = self._get_mlflow_logger(trainer)
        if experiment is None:
            return

        run_id = trainer.logger.run_id
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / filename
            fig.savefig(fpath, dpi=100, bbox_inches='tight')
            experiment.log_artifact(run_id, str(fpath), artifact_path)

    def _make_image_fig(self, array: np.ndarray, title: str, cmap: str = 'viridis'):
        """Create a single-image matplotlib figure."""
        img = _to_2d(array)
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
        return fig

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy().squeeze()

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self._should_log(trainer):
            return

        val_dl, dataset = _resolve_val_dataloader(trainer)
        if val_dl is None or dataset is None:
            logger.debug("Recon logging: no val dataloader, skipping.")
            return

        epoch = trainer.current_epoch + 1
        epoch_str = f"epoch_{epoch:04d}"

        dataset_len = len(dataset)
        indices = self._select_indices(dataset_len)
        if not indices:
            return

        pl_module.eval()
        try:
            self._log_patches(trainer, pl_module, dataset, indices, epoch_str)
            if self.log_stitch:
                self._log_stitched(trainer, pl_module, val_dl, epoch_str)
        finally:
            pl_module.train()

    def _log_patches(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        dataset,
        indices: List[int],
        epoch_str: str,
    ) -> None:
        """Log patch-level reconstructions for fixed indices."""
        for patch_idx in indices:
            sample = dataset[patch_idx]
            # dataset returns (dict, probe, scale) tuple
            data_dict, probe, _scale = sample

            # Add batch dimension and move to model device
            device = pl_module.device
            images = data_dict['images'].unsqueeze(0).to(device)
            coords = data_dict['coords_relative'].unsqueeze(0).to(device)
            rms_scale = data_dict['rms_scaling_constant'].unsqueeze(0).to(device)
            exp_ids = data_dict['experiment_id'].unsqueeze(0).to(device)
            probe_t = probe.unsqueeze(0).to(device) if probe.dim() < 4 else probe.to(device)

            artifact_base = f"{epoch_str}/patch_{patch_idx:02d}"

            with torch.no_grad():
                # Match training: use rms_scale for both input and output scaling
                pred_diff, amp_pred, phase_pred = pl_module(
                    images, coords, probe_t, rms_scale, rms_scale, exp_ids,
                )

            # Amp/phase predictions
            amp_np = self._to_numpy(amp_pred)
            phase_np = self._to_numpy(phase_pred)

            fig = self._make_image_fig(amp_np, 'Amplitude (pred)')
            self._log_figure(trainer, artifact_base, "amp_pred.png", fig)
            plt.close(fig)

            fig = self._make_image_fig(phase_np, 'Phase (pred)', cmap='twilight')
            self._log_figure(trainer, artifact_base, "phase_pred.png", fig)
            plt.close(fig)

            # Diffraction log-scale
            diff_obs_log = self._to_numpy(torch.log1p(torch.abs(images)))
            diff_pred_log = self._to_numpy(torch.log1p(torch.abs(pred_diff)))

            fig = self._make_image_fig(diff_obs_log, 'Observed diff (log)')
            self._log_figure(trainer, artifact_base, "diff_obs_logI.png", fig)
            plt.close(fig)

            fig = self._make_image_fig(diff_pred_log, 'Predicted diff (log)')
            self._log_figure(trainer, artifact_base, "diff_pred_logI.png", fig)
            plt.close(fig)

            # GT + error maps if supervised
            if 'label_amp' in data_dict:
                amp_gt = self._to_numpy(data_dict['label_amp'])
                phase_gt = self._to_numpy(data_dict['label_phase'])

                fig = self._make_image_fig(amp_gt, 'Amplitude (GT)')
                self._log_figure(trainer, artifact_base, "amp_gt.png", fig)
                plt.close(fig)

                fig = self._make_image_fig(phase_gt, 'Phase (GT)', cmap='twilight')
                self._log_figure(trainer, artifact_base, "phase_gt.png", fig)
                plt.close(fig)

                amp_err = np.abs(_to_2d(amp_np) - _to_2d(amp_gt))
                phase_err = np.abs(_to_2d(phase_np) - _to_2d(phase_gt))

                fig = self._make_image_fig(amp_err, 'Amplitude error', cmap='hot')
                self._log_figure(trainer, artifact_base, "amp_error.png", fig)
                plt.close(fig)

                fig = self._make_image_fig(phase_err, 'Phase error', cmap='hot')
                self._log_figure(trainer, artifact_base, "phase_error.png", fig)
                plt.close(fig)

            # Diffraction error
            diff_err = np.abs(_to_2d(diff_obs_log) - _to_2d(diff_pred_log))
            fig = self._make_image_fig(diff_err, 'Diffraction error (log)', cmap='hot')
            self._log_figure(trainer, artifact_base, "diff_error_logI.png", fig)
            plt.close(fig)

    def _log_stitched(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        val_dl,
        epoch_str: str,
    ) -> None:
        """Log stitched full-resolution reconstructions.

        Requires params.cfg to have nimgs_test, outer_offset_test, N, gridsize
        populated (via update_legacy_dict). Gracefully skips if metadata is missing.
        """
        try:
            from ptycho.workflows.grid_lines_workflow import stitch_predictions
            from ptycho import params as p
        except ImportError:
            logger.warning("Recon logging: stitching import failed, skipping.")
            return

        # Verify required metadata exists in params.cfg
        required_keys = ['nimgs_test', 'N', 'gridsize']
        missing = [k for k in required_keys if p.get(k) is None and k not in p.cfg]
        if missing:
            logger.warning("Recon logging: stitching requires %s in params.cfg, skipping.", missing)
            return

        gridsize = p.cfg['gridsize']

        # Collect all predictions from val set
        all_amp = []
        all_phase = []
        device = pl_module.device
        count = 0

        with torch.no_grad():
            for batch in val_dl:
                data_dict, probe, _scale = batch
                images = data_dict['images'].to(device)
                coords = data_dict['coords_relative'].to(device)
                rms_scale = data_dict['rms_scaling_constant'].to(device)
                exp_ids = data_dict['experiment_id'].to(device)
                probe_t = probe.to(device)

                # Use full forward (matching training scaling) for amp/phase
                _pred_diff, amp_pred, phase_pred = pl_module(
                    images, coords, probe_t, rms_scale, rms_scale, exp_ids,
                )
                # Channel-first (B, C, H, W) → channel-last (B, H, W, C) for stitching
                all_amp.append(amp_pred.permute(0, 2, 3, 1).cpu().numpy())
                all_phase.append(phase_pred.permute(0, 2, 3, 1).cpu().numpy())

                count += images.shape[0]
                if self.max_stitch_samples is not None and count >= self.max_stitch_samples:
                    break

        if not all_amp:
            return

        amp_all = np.concatenate(all_amp, axis=0)   # (N_total, H, W, C)
        phase_all = np.concatenate(all_phase, axis=0)

        # For gridsize>1, C = gridsize^2; stitch_predictions handles reshaping.
        # For gridsize=1, C = 1; shape is already (N, H, W, 1).
        try:
            norm_Y_I = 1.0
            amp_stitched = stitch_predictions(amp_all, norm_Y_I=norm_Y_I, part="amp")
            phase_stitched = stitch_predictions(phase_all, norm_Y_I=norm_Y_I, part="phase")
        except Exception as e:
            logger.warning("Recon logging: stitching failed (%s), skipping.", e)
            return

        artifact_base = f"{epoch_str}/stitched"

        fig = self._make_image_fig(amp_stitched.squeeze(), 'Stitched amplitude')
        self._log_figure(trainer, artifact_base, "amp_pred.png", fig)
        plt.close(fig)

        fig = self._make_image_fig(phase_stitched.squeeze(), 'Stitched phase', cmap='twilight')
        self._log_figure(trainer, artifact_base, "phase_pred.png", fig)
        plt.close(fig)

        # Stitched GT + error if validation dataset has labels
        val_dl_resolved, dataset = _resolve_val_dataloader(trainer)
        if dataset is not None and len(dataset) > 0:
            sample_dict = dataset[0][0]
            if 'label_amp' in sample_dict:
                # Collect GT labels for stitching
                all_gt_amp = []
                all_gt_phase = []
                for batch in val_dl:
                    data_dict, _, _ = batch
                    all_gt_amp.append(data_dict['label_amp'].permute(0, 2, 3, 1).numpy())
                    all_gt_phase.append(data_dict['label_phase'].permute(0, 2, 3, 1).numpy())
                    if self.max_stitch_samples is not None and sum(a.shape[0] for a in all_gt_amp) >= self.max_stitch_samples:
                        break

                gt_amp = np.concatenate(all_gt_amp, axis=0)
                gt_phase = np.concatenate(all_gt_phase, axis=0)

                try:
                    gt_amp_stitched = stitch_predictions(gt_amp, norm_Y_I=norm_Y_I, part="amp")
                    gt_phase_stitched = stitch_predictions(gt_phase, norm_Y_I=norm_Y_I, part="phase")
                except Exception as e:
                    logger.warning("Recon logging: GT stitching failed (%s), skipping GT.", e)
                    return

                fig = self._make_image_fig(gt_amp_stitched.squeeze(), 'Stitched amplitude (GT)')
                self._log_figure(trainer, artifact_base, "amp_gt.png", fig)
                plt.close(fig)

                fig = self._make_image_fig(gt_phase_stitched.squeeze(), 'Stitched phase (GT)', cmap='twilight')
                self._log_figure(trainer, artifact_base, "phase_gt.png", fig)
                plt.close(fig)

                amp_err = np.abs(amp_stitched.squeeze() - gt_amp_stitched.squeeze())
                phase_err = np.abs(phase_stitched.squeeze() - gt_phase_stitched.squeeze())

                fig = self._make_image_fig(amp_err, 'Stitched amplitude error', cmap='hot')
                self._log_figure(trainer, artifact_base, "amp_error.png", fig)
                plt.close(fig)

                fig = self._make_image_fig(phase_err, 'Stitched phase error', cmap='hot')
                self._log_figure(trainer, artifact_base, "phase_error.png", fig)
                plt.close(fig)
