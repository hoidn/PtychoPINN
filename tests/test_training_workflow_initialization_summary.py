"""Training-workflow propagation of rectangular initialization identity."""

import argparse
from unittest.mock import MagicMock

import numpy as np


def test_materialize_backend_container_preserves_exact_grouped_raw_counts():
    from ptycho.config import ModelConfig, TrainingConfig
    from ptycho.workflows import training as training_workflow

    counts = np.arange(2 * 64 * 64, dtype=np.float32).reshape(2, 64, 64, 1)
    grouped = {
        "diffraction": counts,
        "X_full": np.full_like(counts, 0.25),
        "Y": None,
        "coords_relative": np.zeros((2, 1, 2, 1), dtype=np.float32),
        "coords_offsets": np.zeros((2, 1, 2, 1), dtype=np.float64),
        "nn_indices": np.arange(2, dtype=np.int32).reshape(-1, 1),
    }
    raw = argparse.Namespace(
        probeGuess=np.ones((64, 64), dtype=np.complex64),
        metadata=None,
    )
    config = TrainingConfig(
        model=ModelConfig(N=64, gridsize=1),
        backend="pytorch",
    )

    container = training_workflow._materialize_backend_container(
        grouped,
        raw,
        config,
    )

    np.testing.assert_array_equal(container.raw_grouped_diffraction, counts)
    assert container.raw_grouped_diffraction.flags.c_contiguous
    torch_x = container.X.detach().cpu().numpy()
    np.testing.assert_array_equal(torch_x, grouped["X_full"])


def test_pytorch_workflow_returns_initialization_and_summary_path(
    tmp_path,
    monkeypatch,
):
    from ptycho.config import (
        DataConfig as PublicDataConfig,
        ModelConfig,
        SamplingConfig,
        TrainingConfig,
    )
    from ptycho.workflows import training as training_workflow
    from ptycho_torch.config_params import (
        DataConfig,
        ModelConfig as TorchModelConfig,
        TrainingConfig as TorchTrainingConfig,
    )

    train_path = tmp_path / "train.npz"
    train_path.touch()
    output_dir = tmp_path / "out"
    config = TrainingConfig(
        model=ModelConfig(N=64, gridsize=1),
        data=PublicDataConfig(train_data_file=train_path),
        sampling=SamplingConfig(training_groups=1),
        backend="pytorch",
        nepochs=1,
        output_dir=output_dir,
    )

    class FakeRaw:
        probeGuess = np.ones((64, 64), dtype=np.complex64)
        metadata = None

        def generate_grouped_data(self, **_kwargs):
            return {
                "nn_indices": np.zeros((1, 1), dtype=np.int32),
                "diffraction": np.full(
                    (1, 64, 64, 1), 7.0, dtype=np.float32
                ),
                "Y": None,
                "X_full": np.full(
                    (1, 64, 64, 1), 0.25, dtype=np.float32
                ),
                "coords_relative": np.zeros(
                    (1, 1, 2, 1), dtype=np.float32
                ),
                "coords_offsets": np.zeros(
                    (1, 1, 2, 1), dtype=np.float64
                ),
            }

    payload = argparse.Namespace(
        tf_training_config=config,
        pt_data_config=DataConfig(N=64, gridsize=1),
        pt_model_config=TorchModelConfig(),
        pt_training_config=TorchTrainingConfig(),
        model_spec=object(),
    )
    initialization = {
        "schema_version": "rect-s1s2-initialization-v2",
        "mode": "ones",
        "solved_gauge": 1.0,
        "method": "unit_default_no_solve",
        "sampled_patterns": 0,
    }
    training_summary_path = output_dir / "training_summary.json"
    dispatch = MagicMock(
        return_value=(
            None,
            None,
            {
                "backend": "pytorch",
                "bundle_path": output_dir / "bundle.zip",
                "rect_s1s2_initialization": initialization,
                "training_summary_path": training_summary_path,
            },
        )
    )
    monkeypatch.setattr(training_workflow, "_resolve_public_config", lambda _r: config)
    monkeypatch.setattr(training_workflow, "load_data", lambda *_a, **_k: FakeRaw())
    monkeypatch.setattr(
        training_workflow,
        "_legacy_execution_and_patch",
        lambda *_args: (None, {}),
    )
    monkeypatch.setattr(
        training_workflow,
        "resolve_training_payload",
        lambda **_kwargs: payload,
    )
    monkeypatch.setattr(
        training_workflow,
        "run_cdi_example_with_backend",
        dispatch,
    )
    monkeypatch.setattr(
        training_workflow,
        "_persist_tensorflow_outputs",
        MagicMock(),
    )

    result = training_workflow.run_training_workflow(
        training_workflow.TrainingWorkflowRequest(
            legacy_args=argparse.Namespace(config=None, do_stitching=False),
        )
    )

    assert result.rect_s1s2_initialization == initialization
    assert result.training_summary_path == training_summary_path
    dispatched_train_container = dispatch.call_args.args[0]
    np.testing.assert_array_equal(
        dispatched_train_container.raw_grouped_diffraction,
        np.full((1, 64, 64, 1), 7.0, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        dispatched_train_container.X.detach().cpu().numpy(),
        np.full((1, 64, 64, 1), 0.25, dtype=np.float32),
    )
