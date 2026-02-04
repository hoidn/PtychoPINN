"""MLflow intermediate reconstruction logging callback (Torch Lightning).

Logs patch-level and optionally stitched reconstructions to MLflow every N epochs.
Design: docs/plans/2026-01-30-mlflow-intermediate-recon-logging-design.md
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

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


def _extract_stitch_fields(
    metadata: Optional[dict],
    dataset_len: Optional[int] = None,
) -> Optional[dict]:
    if not metadata:
        return None

    if "physics_parameters" in metadata:
        physics = metadata.get("physics_parameters", {})
        additional = metadata.get("additional_parameters", {})
        N = physics.get("N") or metadata.get("N")
        gridsize = physics.get("gridsize") or metadata.get("gridsize")
        offset = additional.get("offset", metadata.get("offset", 0))
        outer_offset_test = additional.get("outer_offset_test", metadata.get("outer_offset_test", offset))
        nimgs_test = additional.get("nimgs_test", metadata.get("nimgs_test", dataset_len))
    else:
        N = metadata.get("N")
        gridsize = metadata.get("gridsize")
        offset = metadata.get("offset", 0)
        outer_offset_test = metadata.get("outer_offset_test", offset)
        nimgs_test = metadata.get("nimgs_test", dataset_len)

    if N is None or gridsize is None:
        return None

    return {
        "N": N,
        "gridsize": gridsize,
        "offset": offset,
        "outer_offset_test": outer_offset_test,
        "nimgs_test": nimgs_test if nimgs_test is not None else dataset_len,
    }


def _reorder_grid_channels(patches: np.ndarray, gridsize: int) -> np.ndarray:
    """Reorder grid channels into spatial layout for stitching."""
    if patches.ndim == 3:
        return patches[..., np.newaxis]
    if gridsize <= 1:
        return patches if patches.shape[-1] == 1 else patches[..., np.newaxis]
    if patches.ndim == 4 and patches.shape[-1] == gridsize**2:
        batch, height, width, _ = patches.shape
        reshaped = patches.reshape(batch, height, width, gridsize, gridsize)
        reshaped = reshaped.transpose(0, 3, 4, 1, 2)
        return reshaped.reshape(batch * gridsize**2, height, width, 1)
    return patches


def _select_offsets(data_dict: dict) -> Optional[torch.Tensor]:
    for key in ("global_offsets", "coords_relative", "coords_nominal", "coords_absolute", "coords"):
        if key in data_dict:
            return data_dict[key]
    if "xcoords" in data_dict and "ycoords" in data_dict:
        return torch.stack([data_dict["xcoords"], data_dict["ycoords"]], dim=-1)
    return None


def _normalize_offsets(offsets: torch.Tensor) -> Optional[torch.Tensor]:
    if offsets is None:
        return None
    if offsets.ndim == 3 and offsets.shape[-1] == 2:
        return offsets.unsqueeze(2)
    if offsets.ndim == 4 and offsets.shape[-1] == 2:
        return offsets
    if offsets.ndim == 4 and offsets.shape[-2] == 2:
        return offsets.permute(0, 1, 3, 2).contiguous()
    return None


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
        metadata_path: Optional NPZ path for MetadataManager lookup.
    """

    def __init__(
        self,
        every_n_epochs: int = 5,
        num_patches: int = 4,
        fixed_indices: Optional[List[int]] = None,
        log_stitch: bool = False,
        max_stitch_samples: Optional[int] = None,
        metadata_path: Optional[Path] = None,
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_patches = num_patches
        self.fixed_indices = fixed_indices
        self.log_stitch = log_stitch
        self.max_stitch_samples = max_stitch_samples
        self.metadata_path = Path(metadata_path) if metadata_path is not None else None
        self._selected_indices: Optional[List[int]] = None
        self._metadata_context: Optional[dict] = None

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

    def _load_metadata_context(self) -> dict:
        if self._metadata_context is not None:
            return self._metadata_context

        context = {
            "metadata": None,
            "norm_Y_I": None,
            "YY_ground_truth": None,
        }
        if self.metadata_path is None:
            self._metadata_context = context
            return context

        try:
            from ptycho.metadata import MetadataManager
            data, metadata = MetadataManager.load_with_metadata(str(self.metadata_path))
            context["metadata"] = metadata
            if "norm_Y_I" in data:
                context["norm_Y_I"] = float(np.array(data["norm_Y_I"]).squeeze())
            if "YY_ground_truth" in data:
                context["YY_ground_truth"] = data["YY_ground_truth"]
            elif "YY_full" in data:
                context["YY_ground_truth"] = data["YY_full"]
        except Exception as e:
            logger.warning("Recon logging: failed to load metadata from %s (%s).",
                           self.metadata_path, e)

        self._metadata_context = context
        return context

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self._should_log(trainer):
            return

        val_dl, dataset = _resolve_val_dataloader(trainer)
        if val_dl is None or dataset is None:
            # Fallback to train dataloader for train-only workflows
            train_dl = getattr(trainer, 'train_dataloader', None)
            if callable(train_dl):
                train_dl = train_dl()
            if train_dl is not None:
                logger.warning("Recon logging: no val dataloader, falling back to train loader.")
                dataset = getattr(train_dl, 'dataset', None)
                val_dl = train_dl
            if val_dl is None or dataset is None:
                logger.debug("Recon logging: no dataloader available, skipping.")
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

        Uses ptycho.image.stitching.reassemble_patches with a config dict
        derived from the dataset metadata. Falls back gracefully when
        required metadata (N, gridsize, offset, nimgs_test) is unavailable.
        """
        try:
            from ptycho.image.stitching import reassemble_patches
        except ImportError:
            logger.warning("Recon logging: stitching import failed, skipping.")
            return

        metadata_context = self._load_metadata_context()
        norm_Y_I = metadata_context.get("norm_Y_I", None)
        if norm_Y_I is None:
            norm_Y_I = 1.0

        # Build stitch config from dataset metadata or metadata file
        stitch_config = self._build_stitch_config(val_dl)
        use_grid_stitch = stitch_config is not None

        # Collect all predictions from val set
        all_amp: List[torch.Tensor] = []
        all_phase: List[torch.Tensor] = []
        all_offsets: List[torch.Tensor] = []
        all_gt_amp: List[torch.Tensor] = []
        all_gt_phase: List[torch.Tensor] = []
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

                _pred_diff, amp_pred, phase_pred = pl_module(
                    images, coords, probe_t, rms_scale, rms_scale, exp_ids,
                )
                all_amp.append(amp_pred.detach().cpu())
                all_phase.append(phase_pred.detach().cpu())

                if not use_grid_stitch:
                    offsets = _normalize_offsets(_select_offsets(data_dict))
                    if offsets is not None:
                        all_offsets.append(offsets.detach().cpu())

                if 'label_amp' in data_dict:
                    all_gt_amp.append(data_dict['label_amp'].detach().cpu())
                    all_gt_phase.append(data_dict['label_phase'].detach().cpu())

                count += images.shape[0]
                if self.max_stitch_samples is not None and count >= self.max_stitch_samples:
                    break

        if not all_amp:
            return

        amp_all = torch.cat(all_amp, dim=0)
        phase_all = torch.cat(all_phase, dim=0)
        complex_all = torch.polar(amp_all, phase_all)

        if use_grid_stitch:
            try:
                complex_np = complex_all.permute(0, 2, 3, 1).numpy()
                complex_np = _reorder_grid_channels(complex_np, stitch_config["gridsize"])
                stitch_config['nimgs_test'] = complex_np.shape[0]
                amp_stitched = reassemble_patches(complex_np, stitch_config, norm_Y_I=norm_Y_I, part="amp")
                phase_stitched = reassemble_patches(complex_np, stitch_config, norm_Y_I=norm_Y_I, part="phase")
            except Exception as e:
                logger.warning("Recon logging: stitching failed (%s), skipping.", e)
                return
        else:
            if not all_offsets:
                logger.warning("Recon logging: no offsets available for position-based stitching, skipping.")
                return
            offsets_all = torch.cat(all_offsets, dim=0)
            offsets_all = _normalize_offsets(offsets_all)
            if offsets_all is None:
                logger.warning("Recon logging: offsets format unsupported for position-based stitching, skipping.")
                return
            data_config = getattr(pl_module, "data_config", None)
            model_config = getattr(pl_module, "model_config", None)
            if data_config is None or model_config is None:
                logger.warning("Recon logging: missing data/model config for position-based stitching, skipping.")
                return
            try:
                from ptycho_torch.helper import reassemble_patches_position_real
                complex_cpu = complex_all.cpu()
                offsets_cpu = offsets_all.cpu()
                stitched_complex = reassemble_patches_position_real(
                    complex_cpu, offsets_cpu, data_config, model_config
                )
                amp_stitched = self._to_numpy(torch.abs(stitched_complex))
                phase_stitched = self._to_numpy(torch.angle(stitched_complex))
            except Exception as e:
                logger.warning("Recon logging: position-based stitching failed (%s), skipping.", e)
                return

        artifact_base = f"{epoch_str}/stitched"

        fig = self._make_image_fig(amp_stitched.squeeze(), 'Stitched amplitude')
        self._log_figure(trainer, artifact_base, "amp_pred.png", fig)
        plt.close(fig)

        fig = self._make_image_fig(phase_stitched.squeeze(), 'Stitched phase', cmap='twilight')
        self._log_figure(trainer, artifact_base, "phase_pred.png", fig)
        plt.close(fig)

        # Stitched GT + error (metadata GT preferred for grid stitching)
        gt_logged = False
        gt_full = metadata_context.get("YY_ground_truth", None)
        if gt_full is not None:
            gt_amp = np.abs(gt_full)
            gt_phase = np.angle(gt_full)
            if gt_amp.shape == amp_stitched.squeeze().shape:
                fig = self._make_image_fig(gt_amp.squeeze(), 'Stitched amplitude (GT)')
                self._log_figure(trainer, artifact_base, "amp_gt.png", fig)
                plt.close(fig)
                fig = self._make_image_fig(gt_phase.squeeze(), 'Stitched phase (GT)', cmap='twilight')
                self._log_figure(trainer, artifact_base, "phase_gt.png", fig)
                plt.close(fig)
                amp_err = np.abs(amp_stitched.squeeze() - gt_amp.squeeze())
                phase_err = np.abs(phase_stitched.squeeze() - gt_phase.squeeze())
                fig = self._make_image_fig(amp_err, 'Stitched amplitude error', cmap='hot')
                self._log_figure(trainer, artifact_base, "amp_error.png", fig)
                plt.close(fig)
                fig = self._make_image_fig(phase_err, 'Stitched phase error', cmap='hot')
                self._log_figure(trainer, artifact_base, "phase_error.png", fig)
                plt.close(fig)
                gt_logged = True

        if not gt_logged and use_grid_stitch and all_gt_amp:
            try:
                gt_amp = torch.cat(all_gt_amp, dim=0)
                gt_phase = torch.cat(all_gt_phase, dim=0)
                gt_complex = torch.polar(gt_amp, gt_phase)
                gt_np = gt_complex.permute(0, 2, 3, 1).numpy()
                gt_np = _reorder_grid_channels(gt_np, stitch_config["gridsize"])
                gt_config = dict(stitch_config)
                gt_config['nimgs_test'] = gt_np.shape[0]
                gt_amp_stitched = reassemble_patches(gt_np, gt_config, norm_Y_I=norm_Y_I, part="amp")
                gt_phase_stitched = reassemble_patches(gt_np, gt_config, norm_Y_I=norm_Y_I, part="phase")
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

    def _build_stitch_config(self, val_dl) -> Optional[dict]:
        """Build a stitch config dict from dataset metadata.

        Attempts to extract N, gridsize, offset from the dataset's metadata
        or from params.cfg as a fallback. Returns None if insufficient metadata.
        """
        dataset = getattr(val_dl, 'dataset', None)
        dataset_len = len(dataset) if dataset is not None else None

        metadata_context = self._load_metadata_context()
        metadata = metadata_context.get("metadata")
        config = _extract_stitch_fields(metadata, dataset_len)
        if config is not None:
            return config

        # Try to get metadata from the dataset (PtychoDataset stores it)
        metadata = getattr(dataset, 'metadata', None) if dataset is not None else None
        config = _extract_stitch_fields(metadata, dataset_len)
        if config is not None:
            return config

        # Try params.cfg as fallback
        try:
            from ptycho import params as p
            cfg = getattr(p, 'cfg', {})
            N = cfg.get('N')
            gridsize = cfg.get('gridsize')
            offset = cfg.get('offset', 0)
            outer_offset_test = cfg.get('outer_offset_test', offset)
            if N is None or gridsize is None:
                return None
            return {
                'N': N,
                'gridsize': gridsize,
                'offset': offset,
                'outer_offset_test': outer_offset_test,
                'nimgs_test': dataset_len,
            }
        except ImportError:
            return None
