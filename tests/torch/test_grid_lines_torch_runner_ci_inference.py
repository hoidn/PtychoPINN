"""Focused CI train/inference handoff contract tests."""

import numpy as np
import pytest

from ptycho.config.config import ModelConfig, TrainingConfig
from ptycho.metadata import MetadataManager
from ptycho_torch.scaling_contract import CI_SCALE_CONTRACT, COUNT_INTENSITY
from scripts.studies.grid_lines_torch_runner import (
    TorchRunnerConfig,
    load_cached_dataset,
    run_torch_training,
    setup_torch_configs,
)

@pytest.fixture
def synthetic_npz(tmp_path):
    """Create synthetic NPZ files for testing."""
    N = 64
    n_samples = 4
    gridsize = 1

    # Create synthetic data matching expected contract
    data = {
        'diffraction': np.random.rand(n_samples, N, N, gridsize**2).astype(np.float32),
        'Y_I': np.random.rand(n_samples, N, N, gridsize**2).astype(np.float32),
        'Y_phi': np.random.rand(n_samples, N, N, gridsize**2).astype(np.float32),
        'coords_nominal': np.random.rand(n_samples * gridsize**2, 2).astype(np.float32),
        'coords_true': np.random.rand(n_samples * gridsize**2, 2).astype(np.float32),
        'YY_full': np.random.rand(1, N * 2, N * 2).astype(np.complex64),
        'YY_ground_truth': np.random.rand(1, N * 2, N * 2).astype(np.complex64),
    }

    # Compute correct YY_ground_truth shape for stitching params
    outer_offset_test = 20
    bordersize = (N - outer_offset_test / 2) / 2
    borderleft = int(np.ceil(bordersize))
    borderright = int(np.floor(bordersize))
    tile_size = N - (borderleft + borderright)
    data["YY_ground_truth"] = np.random.rand(tile_size, tile_size, 1).astype(np.complex64)
    data["norm_Y_I"] = np.array(1.0, dtype=np.float32)

    train_path = tmp_path / "train.npz"
    test_path = tmp_path / "test.npz"

    cfg_for_meta = TrainingConfig(model=ModelConfig(N=N, gridsize=gridsize))
    metadata = MetadataManager.create_metadata(
        cfg_for_meta,
        script_name="test_fixture",
        nimgs_test=n_samples,
        outer_offset_test=outer_offset_test,
    )
    MetadataManager.save_with_metadata(str(train_path), data, metadata)
    MetadataManager.save_with_metadata(str(test_path), data, metadata)

    return train_path, test_path



class TestCIPreparedInference:
    def test_ci_inference_rejects_raw_amplitude_without_prepared_contract(
        self,
        synthetic_npz,
        tmp_path,
    ):
        from scripts.studies.grid_lines_torch_runner import run_torch_inference
        import torch

        _, test_path = synthetic_npz
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=test_path,
            output_dir=tmp_path,
            architecture="cnn",
            scale_contract_version=CI_SCALE_CONTRACT,
            measurement_domain=COUNT_INTENSITY,
            physics_forward_mode="rectangular_scaled",
            torch_loss_mode="poisson",
        )

        class UnusedModel:
            def eval(self):
                return self

            def forward_predict(self, *args, **kwargs):
                raise AssertionError("raw CI input must fail before model execution")

        raw_test_data = dict(np.load(test_path, allow_pickle=True))
        with pytest.raises(ValueError, match="CI inference requires the prepared"):
            run_torch_inference(UnusedModel(), raw_test_data, cfg)

    def test_amplitude_mode_does_not_activate_ci_from_default_data_pair(self, tmp_path):
        from scripts.studies.grid_lines_torch_runner import run_torch_inference
        import torch

        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path,
            architecture="cnn",
            N=4,
            physics_forward_mode="amplitude",
        )
        raw = {
            "diffraction": np.ones((1, 4, 4, 1), dtype=np.float32),
            "probeGuess": np.ones((4, 4), dtype=np.complex64),
            "coords_relative": np.zeros((1, 1, 2, 1), dtype=np.float32),
        }

        class SpyModel:
            def eval(self):
                return self

            def forward_predict(self, x, positions, probe, input_scale_factor):
                return torch.zeros((x.shape[0], 4, 4), dtype=torch.complex64)

        prediction = run_torch_inference(SpyModel(), raw, cfg)

        assert prediction.shape == (1, 4, 4)

    def test_ci_inference_uses_prepared_count_input_and_frozen_rms_scale(self, tmp_path):
        """CI inference must reuse the representation prepared for validation."""
        from scripts.studies.grid_lines_torch_runner import run_torch_inference
        import torch

        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path,
            architecture="cnn",
            N=4,
            scale_contract_version=CI_SCALE_CONTRACT,
            measurement_domain=COUNT_INTENSITY,
            physics_forward_mode="rectangular_scaled",
            torch_loss_mode="poisson",
        )
        count_input = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4, 1)
        raw_amplitude = np.full_like(count_input, 99.0)
        prepared_probe = np.full((4, 4), 2.0 + 3.0j, dtype=np.complex64)
        raw_probe = np.full((4, 4), 11.0 + 13.0j, dtype=np.complex64)
        prepared_test_data = {
            "X": count_input,
            "diffraction": raw_amplitude,
            "coords_relative": np.zeros((2, 1, 2, 1), dtype=np.float32),
            "probe_training": prepared_probe,
            "probeGuess": raw_probe,
            "rms_input_scale": np.array(0.25, dtype=np.float32),
        }

        class SpyModel:
            def __init__(self):
                self.calls = []

            def eval(self):
                return self

            def forward_predict(self, x, positions, probe, input_scale_factor):
                self.calls.append(
                    (
                        x.detach().cpu(),
                        positions.detach().cpu(),
                        probe.detach().cpu(),
                        input_scale_factor.detach().cpu(),
                    )
                )
                return torch.zeros(
                    x.shape[0], x.shape[2], x.shape[3], dtype=torch.complex64
                )

        model = SpyModel()
        run_torch_inference(model, prepared_test_data, cfg)

        assert len(model.calls) == 1
        x, _, probe, input_scale = model.calls[0]
        torch.testing.assert_close(
            x,
            torch.from_numpy(count_input).permute(0, 3, 1, 2),
        )
        torch.testing.assert_close(probe, torch.from_numpy(prepared_probe))
        torch.testing.assert_close(
            input_scale,
            torch.full((2, 1, 1, 1), 0.25, dtype=torch.float32),
        )


def test_grouped_runner_enables_full_support_probe(tmp_path):
    cfg = TorchRunnerConfig(
        train_npz=tmp_path / "train.npz",
        test_npz=tmp_path / "test.npz",
        output_dir=tmp_path,
        architecture="cnn",
        gridsize=2,
    )

    training_config, _ = setup_torch_configs(cfg)

    assert training_config.model.probe_big is True


class TestCITrainingHandoff:
    def test_ci_training_returns_validation_container_for_matching_inference(
        self,
        synthetic_npz,
        tmp_path,
        monkeypatch,
    ):
        """The CI validation representation must survive the training boundary."""
        from unittest.mock import MagicMock

        train_path, test_path = synthetic_npz
        cfg = TorchRunnerConfig(
            train_npz=train_path,
            test_npz=test_path,
            output_dir=tmp_path,
            architecture="cnn",
            physics_forward_mode="rectangular_scaled",
            torch_loss_mode="poisson",
            scale_contract_version=CI_SCALE_CONTRACT,
            measurement_domain=COUNT_INTENSITY,
        )
        train_data = load_cached_dataset(train_path)
        test_data = load_cached_dataset(test_path)
        captured = {}

        def fake_train(train_container, test_container, config, execution_config=None, overrides=None):
            captured["test_container"] = test_container
            return {
                "history": {"train_loss": []},
                "models": {"diffraction_to_obj": MagicMock()},
            }

        monkeypatch.setattr(
            "ptycho_torch.workflows.components._train_with_lightning",
            fake_train,
        )

        result = run_torch_training(cfg, train_data, test_data)

        prepared = result["_ci_inference_container"]
        assert prepared is captured["test_container"]
        assert prepared["X"] is not test_data["diffraction"]
        np.testing.assert_allclose(
            prepared["X"],
            prepared["measured_intensity"].detach().cpu().numpy(),
        )
        assert "rms_input_scale" in prepared
        assert "probe_training" in prepared


def test_library_run_routes_ci_prepared_validation_container_to_inference(
    tmp_path,
    monkeypatch,
):
    from scripts.studies import grid_lines_torch_runner as runner

    class FakeModel:
        def parameters(self):
            return []

        def state_dict(self):
            return {"weights": np.array([1.0], dtype=np.float32)}

    raw_test_data = {
        "diffraction": np.full((1, 4, 4, 1), 99.0, dtype=np.float32),
        "YY_ground_truth": np.ones((4, 4), dtype=np.complex64),
    }
    prepared_test_data = {
        "X": np.full((1, 4, 4, 1), 7.0, dtype=np.float32),
        "rms_input_scale": np.array(0.25, dtype=np.float32),
    }
    captured = {}

    monkeypatch.setattr(
        runner,
        "load_cached_dataset_with_metadata",
        lambda path: (raw_test_data, None),
    )
    monkeypatch.setattr(
        runner,
        "run_torch_training",
        lambda *args, **kwargs: {
            "history": {"train_loss": [0.1]},
            "model": FakeModel(),
            "_ci_inference_container": prepared_test_data,
        },
    )

    def capture_inference(model, test_data, cfg, metadata=None):
        captured["test_data"] = test_data
        return np.ones((4, 4), dtype=np.complex64)

    monkeypatch.setattr(runner, "run_torch_inference", capture_inference)
    monkeypatch.setattr(
        runner,
        "compute_metrics",
        lambda pred, gt, label: {"mae": [0.1, 0.2], "mse": [0.01, 0.02]},
    )
    monkeypatch.setattr(
        "ptycho.workflows.grid_lines_workflow.save_recon_artifact",
        lambda output_dir, model_id, recon: output_dir / "recons" / model_id / "recon.npz",
    )
    monkeypatch.setattr(
        "ptycho.workflows.grid_lines_workflow.render_grid_lines_visuals",
        lambda output_dir, order: {},
    )

    cfg = TorchRunnerConfig(
        train_npz=tmp_path / "train.npz",
        test_npz=tmp_path / "test.npz",
        output_dir=tmp_path,
        architecture="cnn",
        epochs=1,
        N=4,
        scale_contract_version=CI_SCALE_CONTRACT,
        measurement_domain=COUNT_INTENSITY,
        physics_forward_mode="rectangular_scaled",
        torch_loss_mode="poisson",
    )
    cfg.train_npz.write_bytes(b"stub")
    cfg.test_npz.write_bytes(b"stub")

    runner.run_grid_lines_torch(cfg)

    assert captured["test_data"] is prepared_test_data
