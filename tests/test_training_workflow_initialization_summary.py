"""Training-workflow propagation of rectangular initialization identity."""

import argparse
from unittest.mock import MagicMock

import numpy as np


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
        sampling=SamplingConfig(n_groups=1),
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
                "diffraction": np.ones((1, 64, 64, 1), dtype=np.float32),
                "Y": None,
                "X_full": np.ones((1, 64, 64, 1), dtype=np.float32),
            }

    payload = argparse.Namespace(
        tf_training_config=config,
        pt_data_config=DataConfig(N=64, C=1, grid_size=(1, 1)),
        pt_model_config=TorchModelConfig(C_model=1, C_forward=1),
        pt_training_config=TorchTrainingConfig(),
        model_spec=object(),
    )
    initialization = {
        "schema_version": "rect-s1s2-initialization-v1",
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
        "_materialize_backend_container",
        lambda grouped, *_args: grouped,
    )
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
