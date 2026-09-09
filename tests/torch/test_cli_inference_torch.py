"""CLI runtime routing, retained corruption guards, and real strict-load parity."""

import pytest
from pathlib import Path
from unittest.mock import patch


@pytest.fixture
def minimal_inference_args(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "wts.h5.zip").touch()
    test_file = tmp_path / "test.npz"
    test_file.touch()
    return [
        "--model_path", str(model_dir),
        "--test_data", str(test_file),
        "--output_dir", str(tmp_path / "inference_outputs"),
    ]


@pytest.mark.parametrize("flags, expected", [
    (["--accelerator", "cpu"], {"accelerator": "cpu"}),
    (["--num-workers", "4"], {"num_workers": 4}),
    (["--inference-batch-size", "32"], {"inference_batch_size": 32}),
    (["--accelerator", "gpu", "--num-workers", "8", "--inference-batch-size", "64"],
     {"accelerator": "gpu", "num_workers": 8, "inference_batch_size": 64}),
    (["--quiet"], {"enable_progress_bar": False}),
])
def test_execution_request_roundtrip(minimal_inference_args, monkeypatch, flags, expected):
    from ptycho_torch.inference import cli_main

    monkeypatch.setattr("sys.argv", ["inference.py", *minimal_inference_args, *flags])
    with patch(
        "ptycho_torch.execution_request.resolve_runtime_execution_request",
        side_effect=RuntimeError("captured runtime request"),
    ) as runtime, pytest.raises(RuntimeError, match="captured runtime request"):
        cli_main()
    request = runtime.call_args.args[0]
    assert all(request.values[name] == value for name, value in expected.items())
    assert set(expected) <= request.explicit_fields


def test_native_inference_execution_request_preserves_explicit_options(
    minimal_inference_args, monkeypatch
):
    from ptycho_torch.cli.shared import build_execution_request_from_args
    from ptycho_torch.inference import cli_main

    argv = minimal_inference_args + [
        "--accelerator=cpu", "--device", "cuda", "--num-workers=2",
        "--inference-batch-size", "8", "--quiet",
    ]
    monkeypatch.setattr("sys.argv", ["inference.py", *argv])
    with patch(
        "ptycho_torch.cli.shared.build_execution_request_from_args",
        wraps=build_execution_request_from_args,
    ) as builder, patch(
        "ptycho_torch.execution_request.resolve_runtime_execution_request",
        side_effect=RuntimeError("captured runtime request"),
    ) as runtime, pytest.raises(RuntimeError, match="captured runtime request"):
        cli_main()
    assert builder.call_args.kwargs == {
        "mode": "inference", "explicit_options": tuple(argv), "lane": "native-inference",
    }
    request = runtime.call_args.args[0]
    assert request.explicit_fields == frozenset({
        "accelerator", "num_workers", "inference_batch_size", "enable_progress_bar",
    })
    assert request.values["accelerator"] == "cpu"
    assert request.values["num_workers"] == 2
    assert request.values["inference_batch_size"] == 8
    assert request.values["enable_progress_bar"] is False


def test_cli_validates_paths_and_saves_arrays(minimal_inference_args, monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from ptycho_torch.cli.shared import validate_paths
    from ptycho_torch.inference import cli_main

    amplitude, phase = np.ones((8, 8)), np.zeros((8, 8))
    monkeypatch.setattr("sys.argv", [
        "inference.py", *minimal_inference_args, "--accelerator", "cpu", "--quiet",
    ])
    with patch(
        "ptycho_torch.cli.shared.validate_paths", wraps=validate_paths,
    ) as validate, patch(
        "ptycho_torch.inference.reconstruct",
        return_value=SimpleNamespace(amplitude=amplitude, phase=phase),
    ) as reconstruct, patch(
        "ptycho_torch.inference.save_individual_reconstructions",
    ) as save:
        assert cli_main() == 0
    validate.assert_called_once_with(
        train_file=None, test_file=Path(minimal_inference_args[3]),
        output_dir=Path(minimal_inference_args[5]),
    )
    assert reconstruct.call_args.kwargs["quiet"] is True
    assert save.call_args.args[0] is amplitude
    assert save.call_args.args[1] is phase
    assert save.call_args.args[2] == Path(minimal_inference_args[5])


def test_real_checkpoint_cli_matches_direct_reconstruction(
    tmp_path, synthetic_ptycho_npz, monkeypatch
):
    import numpy as np
    import torch
    from tests.torch.era_fixtures import v5_bundle
    from ptycho_torch import inference
    from ptycho_torch.config_params import InferenceConfig
    from ptycho_torch.workflows import bundle_io
    from scripts.inference import inference as unified

    if not torch.cuda.is_available():
        pytest.skip("real model inference requires CUDA")
    model_dir = v5_bundle(tmp_path)
    _, test_data = synthetic_ptycho_npz
    expected = inference.reconstruct(
        model_dir, test_data, work_dir=tmp_path / "direct",
        inference_config=InferenceConfig(patch_weighting="uniform", varpro_scaling=False),
        device="cuda", precision="32-true", num_workers=0,
        inference_batch_size=2, quiet=True,
    )
    real_reconstruct = inference.reconstruct
    for door in ("native", "unified"):
        output = tmp_path / door
        if door == "native":
            argv = [
                "inference.py", "--model_path", str(model_dir),
                "--test_data", str(test_data), "--output_dir", str(output),
                "--accelerator", "gpu", "--inference-batch-size", "2", "--quiet",
            ]
        else:
            argv = [
                "ptycho_inference", "--backend", "pytorch", "--model_path", str(model_dir),
                "--test_data", str(test_data), "--output_dir", str(output),
                "--torch-accelerator", "gpu", "--torch-inference-batch-size", "2",
            ]
        monkeypatch.setattr("sys.argv", argv)
        observed = []

        def capture(*args, **kwargs):
            result = real_reconstruct(*args, **kwargs)
            observed.append(result)
            return result

        with patch.object(inference, "reconstruct", side_effect=capture), patch.object(
            bundle_io, "load_inference_bundle_torch", wraps=bundle_io.load_inference_bundle_torch,
        ) as loader:
            if door == "native":
                assert inference.cli_main() == 0
            else:
                with pytest.raises(SystemExit) as exited:
                    unified.main()
                assert exited.value.code == 0
        loader.assert_called_once()
        assert len(observed) == 1
        np.testing.assert_allclose(observed[0].amplitude, expected.amplitude, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(observed[0].phase, expected.phase, rtol=1e-5, atol=1e-6)
        for name in ("reconstructed_amplitude.png", "reconstructed_phase.png"):
            assert (output / name).stat().st_size > 0


class TestCorruptionRetainedBoundaries:
    """Phase 4 Task 4: each deleted check leaves a retained boundary that still
    rejects the same corruption."""

    def test_corruption_channel_join_rejected_by_retained_decode(self):
        """The inference-side C-join is gone; decode still rejects a broken join."""
        from ptycho.config.config import ModelConfig as CanonicalModelConfig
        from ptycho_torch.artifact_schema import (
            decode_artifact_identity,
            encode_artifact_identity,
        )
        from ptycho_torch.config_params import (
            DataConfig,
            InferenceConfig,
            ModelConfig,
            TrainingConfig,
        )
        from ptycho_torch.model_spec import derive_model_spec

        data = DataConfig(N=64, gridsize=1, probe_scale=4.0)
        model = ModelConfig(
            object_layout="single_patch",
            training_canvas="independent",
            training_patch_weighting="uniform",
            object_big=None,
            amp_activation="silu",
        )
        canonical = CanonicalModelConfig(
            N=64,
            gridsize=1,
            object_layout="single_patch",
            training_canvas="independent",
            training_patch_weighting="uniform",
            object_big=None,
            amp_activation="swish",
        )
        spec = derive_model_spec(canonical, model, data)
        payload = encode_artifact_identity(
            spec, data, TrainingConfig(torch_loss_mode="poisson"), InferenceConfig()
        )
        payload["data_config"]["C"] = 2
        with pytest.raises(ValueError, match="field set is not exact"):
            decode_artifact_identity(payload)

    def test_corruption_scale_contract_rejected_by_retained_validation(self):
        """The inference-side scale-contract check is gone; construction still rejects a broken pair."""
        from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
        from ptycho_torch.scaling_contract import validate_scale_contract

        data = DataConfig(
            N=64,
            gridsize=1,
            scale_contract_version="ci_intensity_v2",
            measurement_domain="normalized_amplitude",
        )
        model = ModelConfig(
            physics_forward_mode="rectangular_scaled"
        )
        with pytest.raises(ValueError, match="scale contract"):
            validate_scale_contract(
                data, model, TrainingConfig(torch_loss_mode="poisson")
            )
