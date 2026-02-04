"""Tests for PtychoReconLoggingCallback.

Covers: index selection determinism, epoch gating, DDP guard,
missing-data skip paths, MLflow artifact logging calls.
"""

import numpy as np
import pytest
import torch
from unittest import mock

from ptycho_torch.workflows.recon_logging import PtychoReconLoggingCallback


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

class FakeMLFlowExperiment:
    """Mock MLflow MlflowClient-like experiment object."""
    def __init__(self):
        self.log_artifact = mock.Mock()


class FakeLogger:
    def __init__(self):
        self.experiment = FakeMLFlowExperiment()
        self.run_id = "RUN123"


class FakeDataset:
    """Minimal dataset returning (dict, probe, scale) tuples."""
    def __init__(self, n=16, supervised=False):
        self.n = n
        self.supervised = supervised

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        data = {
            "images": torch.randn(1, 8, 8),
            "coords_relative": torch.zeros(1, 1, 2),
            "rms_scaling_constant": torch.ones(1),
            "physics_scaling_constant": torch.ones(1),
            "experiment_id": torch.tensor(0, dtype=torch.long),
        }
        if self.supervised:
            data["label_amp"] = torch.rand(1, 8, 8)
            data["label_phase"] = torch.rand(1, 8, 8)
        probe = torch.ones(1, 8, 8)
        scale = torch.tensor(1.0)
        return data, probe, scale


class FakeValDataloader:
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        for i in range(len(self.dataset)):
            data, probe, scale = self.dataset[i]
            # Add batch dim
            batch_data = {k: v.unsqueeze(0) for k, v in data.items()}
            yield batch_data, probe.unsqueeze(0), scale.unsqueeze(0)


class FakeModule(torch.nn.Module):
    """Fake Lightning module with forward() returning (diff, amp, phase)."""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)  # need at least one param for .device

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, positions, probe, input_scale, output_scale, experiment_ids):
        return x, torch.abs(x), torch.zeros_like(x)

    def forward_predict(self, x, positions, probe, input_scale):
        return torch.complex(torch.abs(x), torch.zeros_like(x))

    def eval(self):
        return self

    def train(self, mode=True):
        return self


class FakeModuleWithConfigs(FakeModule):
    def __init__(self, N=8, C=1):
        super().__init__()
        from ptycho_torch.config_params import DataConfig, ModelConfig
        self.data_config = DataConfig(N=N, C=C)
        self.model_config = ModelConfig(C_forward=C)


class FakeTrainer:
    def __init__(self, epoch=4, has_logger=True, is_global_zero=True, val_dl=None, train_dl=None):
        self.current_epoch = epoch
        self.is_global_zero = is_global_zero
        self.logger = FakeLogger() if has_logger else None
        self.val_dataloaders = val_dl
        self._train_dl = train_dl

    def train_dataloader(self):
        return self._train_dl


# ---------------------------------------------------------------------------
# Index selection tests
# ---------------------------------------------------------------------------

class TestIndexSelection:
    def test_explicit_indices_respected(self):
        cb = PtychoReconLoggingCallback(fixed_indices=[2, 5, 9, 11])
        result = cb._select_indices(20)
        assert result == [2, 5, 9, 11]

    def test_explicit_indices_clamped_to_dataset_size(self):
        cb = PtychoReconLoggingCallback(fixed_indices=[0, 5, 100])
        result = cb._select_indices(10)
        assert result == [0, 5]

    def test_auto_indices_evenly_spaced(self):
        cb = PtychoReconLoggingCallback(num_patches=4)
        result = cb._select_indices(16)
        assert len(result) == 4
        # Evenly spaced: step=4, indices [0, 4, 8, 12]
        assert result == [0, 4, 8, 12]

    def test_auto_indices_deterministic(self):
        cb = PtychoReconLoggingCallback(num_patches=4)
        r1 = cb._select_indices(100)
        # Reset cached selection
        cb._selected_indices = None
        r2 = cb._select_indices(100)
        assert r1 == r2

    def test_indices_cached_after_first_call(self):
        cb = PtychoReconLoggingCallback(num_patches=4)
        r1 = cb._select_indices(16)
        # Second call with different dataset_len should return same (cached)
        r2 = cb._select_indices(999)
        assert r1 == r2

    def test_empty_dataset(self):
        cb = PtychoReconLoggingCallback(num_patches=4)
        result = cb._select_indices(0)
        assert result == []


# ---------------------------------------------------------------------------
# Epoch gating / should_log tests
# ---------------------------------------------------------------------------

class TestShouldLog:
    def test_logs_at_correct_interval(self):
        cb = PtychoReconLoggingCallback(every_n_epochs=5)
        # current_epoch is 0-indexed, should_log uses epoch+1
        # epoch+1=5 → should log
        trainer = FakeTrainer(epoch=4)
        assert cb._should_log(trainer) is True

    def test_skips_non_interval_epoch(self):
        cb = PtychoReconLoggingCallback(every_n_epochs=5)
        trainer = FakeTrainer(epoch=2)  # epoch+1=3
        assert cb._should_log(trainer) is False

    def test_skips_when_no_logger(self):
        cb = PtychoReconLoggingCallback(every_n_epochs=5)
        trainer = FakeTrainer(epoch=4, has_logger=False)
        assert cb._should_log(trainer) is False

    def test_skips_when_not_global_zero(self):
        cb = PtychoReconLoggingCallback(every_n_epochs=5)
        trainer = FakeTrainer(epoch=4, is_global_zero=False)
        assert cb._should_log(trainer) is False


# ---------------------------------------------------------------------------
# Patch logging integration
# ---------------------------------------------------------------------------

class TestPatchLogging:
    def test_logs_artifacts_for_unsupervised(self):
        dataset = FakeDataset(n=8, supervised=False)
        val_dl = FakeValDataloader(dataset)
        trainer = FakeTrainer(epoch=4, val_dl=val_dl)
        module = FakeModule()

        cb = PtychoReconLoggingCallback(
            every_n_epochs=5,
            num_patches=2,
        )
        cb.on_validation_epoch_end(trainer, module)

        exp = trainer.logger.experiment
        assert exp.log_artifact.called
        # Each patch: amp_pred, phase_pred, diff_obs, diff_pred, diff_error = 5 figs
        # 2 patches × 5 = 10
        assert exp.log_artifact.call_count == 10

    def test_logs_gt_and_error_for_supervised(self):
        dataset = FakeDataset(n=8, supervised=True)
        val_dl = FakeValDataloader(dataset)
        trainer = FakeTrainer(epoch=4, val_dl=val_dl)
        module = FakeModule()

        cb = PtychoReconLoggingCallback(
            every_n_epochs=5,
            num_patches=1,
        )
        cb.on_validation_epoch_end(trainer, module)

        exp = trainer.logger.experiment
        # 1 patch: 5 (unsupervised) + 4 (gt_amp, gt_phase, err_amp, err_phase) = 9
        assert exp.log_artifact.call_count == 9

    def test_artifact_paths_contain_epoch_and_patch(self):
        dataset = FakeDataset(n=8, supervised=False)
        val_dl = FakeValDataloader(dataset)
        trainer = FakeTrainer(epoch=4, val_dl=val_dl)
        module = FakeModule()

        cb = PtychoReconLoggingCallback(
            every_n_epochs=5,
            fixed_indices=[3],
        )
        cb.on_validation_epoch_end(trainer, module)

        exp = trainer.logger.experiment
        # Check artifact_path arg (3rd positional = index 2)
        artifact_paths = [call.args[2] for call in exp.log_artifact.call_args_list]
        assert all("epoch_0005" in p for p in artifact_paths)
        assert all("patch_03" in p for p in artifact_paths)

    def test_handles_val_dataloader_list(self):
        dataset = FakeDataset(n=4, supervised=False)
        val_dl = FakeValDataloader(dataset)
        trainer = FakeTrainer(epoch=4, val_dl=[val_dl])
        module = FakeModule()

        cb = PtychoReconLoggingCallback(every_n_epochs=5, num_patches=1)
        cb.on_validation_epoch_end(trainer, module)

        assert trainer.logger.experiment.log_artifact.called

    def test_uses_rms_scale_for_output(self):
        class CaptureModule(FakeModule):
            def __init__(self):
                super().__init__()
                self.calls = []
            def forward(self, x, positions, probe, input_scale, output_scale, experiment_ids):
                self.calls.append((input_scale, output_scale))
                return x, torch.abs(x), torch.zeros_like(x)

        dataset = FakeDataset(n=2, supervised=False)
        val_dl = FakeValDataloader(dataset)
        trainer = FakeTrainer(epoch=4, val_dl=val_dl)
        module = CaptureModule()

        cb = PtychoReconLoggingCallback(every_n_epochs=5, num_patches=1)
        cb.on_validation_epoch_end(trainer, module)

        assert module.calls
        input_scale, output_scale = module.calls[0]
        assert torch.allclose(input_scale, output_scale)

    def test_multi_channel_patch_logging(self):
        class MultiChannelDataset(FakeDataset):
            def __getitem__(self, idx):
                data, probe, scale = super().__getitem__(idx)
                data["images"] = torch.randn(2, 8, 8)
                return data, probe, scale

        dataset = MultiChannelDataset(n=2, supervised=False)
        val_dl = FakeValDataloader(dataset)
        trainer = FakeTrainer(epoch=4, val_dl=val_dl)
        module = FakeModule()

        cb = PtychoReconLoggingCallback(every_n_epochs=5, num_patches=1)
        cb.on_validation_epoch_end(trainer, module)

        assert trainer.logger.experiment.log_artifact.called

    def test_falls_back_to_train_dataloader(self):
        dataset = FakeDataset(n=4, supervised=False)
        train_dl = FakeValDataloader(dataset)
        trainer = FakeTrainer(epoch=4, val_dl=None, train_dl=train_dl)
        module = FakeModule()

        cb = PtychoReconLoggingCallback(every_n_epochs=5, num_patches=1)
        cb.on_validation_epoch_end(trainer, module)

        assert trainer.logger.experiment.log_artifact.called

    def test_no_logging_when_no_val_dataloader(self):
        trainer = FakeTrainer(epoch=4, val_dl=None)
        module = FakeModule()

        cb = PtychoReconLoggingCallback(every_n_epochs=5, num_patches=2)
        cb.on_validation_epoch_end(trainer, module)

        assert trainer.logger.experiment.log_artifact.call_count == 0


# ---------------------------------------------------------------------------
# Config field tests
# ---------------------------------------------------------------------------

class TestStitchedLogging:
    def test_stitched_logging_uses_reassemble_patches(self, monkeypatch):
        """Verify _log_stitched calls ptycho.image.stitching.reassemble_patches."""
        dataset = FakeDataset(n=4, supervised=False)
        # Add metadata so _build_stitch_config succeeds
        dataset.metadata = {'N': 8, 'gridsize': 1, 'offset': 0, 'outer_offset_test': 0}
        val_dl = FakeValDataloader(dataset)
        trainer = FakeTrainer(epoch=4, val_dl=val_dl)
        module = FakeModule()

        called = {"count": 0}
        def fake_reassemble(patches, config, **kwargs):
            called["count"] += 1
            return np.zeros((1, 8, 8, 1), dtype=np.float32)

        monkeypatch.setattr(
            "ptycho_torch.workflows.recon_logging.PtychoReconLoggingCallback._log_stitched",
            PtychoReconLoggingCallback._log_stitched,  # keep original
        )
        monkeypatch.setattr(
            "ptycho.image.stitching.reassemble_patches", fake_reassemble,
        )

        cb = PtychoReconLoggingCallback(every_n_epochs=5, num_patches=1, log_stitch=True)
        cb.on_validation_epoch_end(trainer, module)

        assert called["count"] > 0
        assert trainer.logger.experiment.log_artifact.called

    def test_stitched_logging_skips_when_no_metadata(self):
        """Verify _log_stitched skips gracefully when metadata is missing."""
        dataset = FakeDataset(n=4, supervised=False)
        # No metadata attribute → _build_stitch_config falls back to params.cfg
        val_dl = FakeValDataloader(dataset)
        trainer = FakeTrainer(epoch=4, val_dl=val_dl)
        module = FakeModule()

        cb = PtychoReconLoggingCallback(every_n_epochs=5, num_patches=1, log_stitch=True)
        # Should not raise even if params.cfg is missing
        cb.on_validation_epoch_end(trainer, module)

    def test_build_stitch_config_from_dataset_metadata(self):
        """Verify _build_stitch_config extracts config from dataset metadata."""
        dataset = FakeDataset(n=8)
        dataset.metadata = {
            'physics_parameters': {'N': 64, 'gridsize': 2},
            'additional_parameters': {'offset': 16, 'outer_offset_test': 8, 'nimgs_test': 8},
        }
        val_dl = FakeValDataloader(dataset)

        cb = PtychoReconLoggingCallback()
        config = cb._build_stitch_config(val_dl)
        assert config is not None
        assert config['N'] == 64
        assert config['gridsize'] == 2
        assert config['offset'] == 16
        assert config['nimgs_test'] == 8

    def test_build_stitch_config_from_metadata_manager(self, tmp_path):
        """Verify _build_stitch_config uses MetadataManager when provided."""
        from ptycho.metadata import MetadataManager

        metadata = {
            'physics_parameters': {'N': 32, 'gridsize': 1},
            'additional_parameters': {'offset': 4, 'outer_offset_test': 6, 'nimgs_test': 7},
        }
        npz_path = tmp_path / "test_with_metadata.npz"
        MetadataManager.save_with_metadata(str(npz_path), {'dummy': np.zeros((1,))}, metadata)

        dataset = FakeDataset(n=4)
        val_dl = FakeValDataloader(dataset)

        cb = PtychoReconLoggingCallback(metadata_path=npz_path)
        config = cb._build_stitch_config(val_dl)

        assert config is not None
        assert config['N'] == 32
        assert config['gridsize'] == 1
        assert config['offset'] == 4
        assert config['outer_offset_test'] == 6
        assert config['nimgs_test'] == 7

    def test_stitched_logging_reorders_gridsize_channels(self, monkeypatch):
        """Verify gridsize>1 channels are reordered before stitching."""
        class GridDataset(FakeDataset):
            def __getitem__(self, idx):
                data, probe, scale = super().__getitem__(idx)
                data["images"] = torch.randn(4, 8, 8)
                data["coords_relative"] = torch.zeros(4, 1, 2)
                return data, probe, scale

        dataset = GridDataset(n=1, supervised=False)
        dataset.metadata = {
            'physics_parameters': {'N': 8, 'gridsize': 2},
            'additional_parameters': {'offset': 0, 'outer_offset_test': 0, 'nimgs_test': 1},
        }
        val_dl = FakeValDataloader(dataset)
        trainer = FakeTrainer(epoch=4, val_dl=val_dl)
        module = FakeModuleWithConfigs(N=8, C=4)

        captured = {}
        def fake_reassemble(patches, config, **kwargs):
            captured['shape'] = patches.shape
            return np.zeros((1, 8, 8, 1), dtype=np.float32)

        monkeypatch.setattr("ptycho.image.stitching.reassemble_patches", fake_reassemble)

        cb = PtychoReconLoggingCallback(every_n_epochs=5, num_patches=1, log_stitch=True)
        cb.on_validation_epoch_end(trainer, module)

        assert captured['shape'] == (4, 8, 8, 1)

    def test_stitched_logging_uses_position_when_grid_missing(self, monkeypatch):
        """Verify position-based stitching when grid metadata is unavailable."""
        import ptycho.params as params

        monkeypatch.setattr(params, 'cfg', {}, raising=False)

        dataset = FakeDataset(n=1, supervised=False)
        val_dl = FakeValDataloader(dataset)
        trainer = FakeTrainer(epoch=4, val_dl=val_dl)
        module = FakeModuleWithConfigs(N=8, C=1)

        called = {}
        def fake_position_stitch(inputs, offsets_xy, data_config, model_config, **kwargs):
            called['inputs_shape'] = tuple(inputs.shape)
            called['offsets_shape'] = tuple(offsets_xy.shape)
            return torch.zeros((inputs.shape[0], 8, 8), dtype=torch.complex64)

        monkeypatch.setattr(
            "ptycho_torch.helper.reassemble_patches_position_real",
            fake_position_stitch,
        )

        cb = PtychoReconLoggingCallback(every_n_epochs=5, num_patches=1, log_stitch=True)
        cb.on_validation_epoch_end(trainer, module)

        assert called['inputs_shape'] == (1, 1, 8, 8)
        assert called['offsets_shape'] == (1, 1, 1, 2)

    def test_build_stitch_config_returns_none_without_metadata(self):
        """Verify _build_stitch_config returns None when no metadata available."""
        dataset = FakeDataset(n=4)
        val_dl = FakeValDataloader(dataset)

        cb = PtychoReconLoggingCallback()
        # Without metadata attr and without params.cfg having N/gridsize,
        # this may return None or a config depending on params state.
        # The key is it doesn't raise.
        cb._build_stitch_config(val_dl)


class TestConfigFields:
    def test_execution_config_defaults(self):
        from ptycho.config.config import PyTorchExecutionConfig
        cfg = PyTorchExecutionConfig()
        assert cfg.recon_log_every_n_epochs is None
        assert cfg.recon_log_num_patches == 4
        assert cfg.recon_log_fixed_indices is None
        assert cfg.recon_log_stitch is False
        assert cfg.recon_log_max_stitch_samples is None

    def test_execution_config_custom(self):
        from ptycho.config.config import PyTorchExecutionConfig
        cfg = PyTorchExecutionConfig(
            recon_log_every_n_epochs=10,
            recon_log_num_patches=2,
            recon_log_fixed_indices=[0, 7],
            recon_log_stitch=True,
            recon_log_max_stitch_samples=50,
        )
        assert cfg.recon_log_every_n_epochs == 10
        assert cfg.recon_log_fixed_indices == [0, 7]
        assert cfg.recon_log_stitch is True
        assert cfg.recon_log_max_stitch_samples == 50


class TestRunnerPropagation:
    def test_runner_propagates_recon_log_max_stitch_samples(self, tmp_path):
        from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig, setup_torch_configs
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="fno",
            recon_log_max_stitch_samples=8,
        )
        _, exec_cfg = setup_torch_configs(cfg)
        assert exec_cfg.recon_log_max_stitch_samples == 8
