import json
import dill
import io
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from lightning.pytorch import Trainer

from ptycho_torch.config_factory import (
    create_inference_payload,
    create_training_payload,
)
from ptycho_torch.config_params import (
    DataConfig,
    DatagenConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
)
from ptycho_torch.lightning_utils import load_checkpoint_with_configs
from ptycho_torch.model import PtychoPINN_Lightning
from ptycho_torch.scaling_contract import (
    CI_SCALE_CONTRACT,
    COUNT_INTENSITY,
    LEGACY_SCALE_CONTRACT,
    NORMALIZED_AMPLITUDE,
)
from ptycho_torch.utils import config_to_json_serializable_dict
from ptycho_torch.model_manager import save_torch_bundle
from ptycho_torch.workflows import components
from ptycho.config.config import ModelConfig as CanonicalModelConfig
from ptycho.config.config import TrainingConfig as CanonicalTrainingConfig
from ptycho import params as legacy_params


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_npz(path: Path, n_images: int = 8) -> None:
    rng = np.random.default_rng(17)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        diff3d=rng.poisson(4.0, size=(n_images, 64, 64)).astype(np.float32),
        probeGuess=np.ones((64, 64), dtype=np.complex64),
        objectGuess=np.ones((96, 96), dtype=np.complex64),
        xcoords=np.arange(n_images, dtype=np.float32),
        ycoords=np.arange(n_images, dtype=np.float32),
    )


def _tiny_configs(*, legacy: bool = False):
    if legacy:
        data_config = DataConfig(
            N=64,
            C=1,
            grid_size=(1, 1),
            scale_contract_version=LEGACY_SCALE_CONTRACT,
            measurement_domain=NORMALIZED_AMPLITUDE,
        )
    else:
        data_config = DataConfig(N=64, C=1, grid_size=(1, 1))
    model_config = ModelConfig(
        C_model=1,
        C_forward=1,
        object_big=False,
        probe_big=False,
        n_filters_scale=1,
        physics_forward_mode="rectangular_scaled",
        cnn_output_mode="real_imag",
        rect_s1s2_trainable=False,
    )
    return (
        data_config,
        model_config,
        TrainingConfig(device="cpu", torch_loss_mode="poisson"),
        InferenceConfig(),
        DatagenConfig(),
    )


def _save_checkpoint_run(
    root: Path,
    *,
    legacy: bool = False,
    ci_statistics=None,
    metadata_free_legacy: bool = False,
) -> Path:
    configs = _tiny_configs(legacy=legacy)
    data_config, model_config, training_config, inference_config, datagen_config = configs
    model = PtychoPINN_Lightning(
        model_config,
        data_config,
        training_config,
        inference_config,
    )
    if ci_statistics is not None:
        model.register_ci_statistics(ci_statistics)

    config_dir = root / "configs"
    checkpoint_dir = root / "checkpoints"
    config_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    for name, config in zip(
        (
            "data_config",
            "model_config",
            "training_config",
            "inference_config",
            "datagen_config",
        ),
        configs,
    ):
        (config_dir / f"{name}.json").write_text(
            json.dumps(config_to_json_serializable_dict(config))
        )

    checkpoint_path = checkpoint_dir / "best-checkpoint.ckpt"
    trainer = Trainer(
        max_epochs=0,
        enable_checkpointing=True,
        logger=False,
        enable_progress_bar=False,
        accelerator="cpu",
        default_root_dir=root,
    )
    trainer.strategy._lightning_module = model
    trainer.save_checkpoint(checkpoint_path)

    if metadata_free_legacy:
        data_path = config_dir / "data_config.json"
        payload = json.loads(data_path.read_text())
        payload.pop("scale_contract_version")
        payload.pop("measurement_domain")
        payload["probe_ramp_removal"] = False
        data_path.write_text(json.dumps(payload))

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_data = checkpoint["hyper_parameters"]["data_config"]
        saved_data.pop("scale_contract_version")
        saved_data.pop("measurement_domain")
        saved_data["probe_ramp_removal"] = False
        torch.save(checkpoint, checkpoint_path)

    return checkpoint_path


def test_rectangular_factory_defaults_to_ci_profile(tmp_path):
    train_npz = tmp_path / "train.npz"
    _write_npz(train_npz)

    payload = create_training_payload(
        train_data_file=train_npz,
        output_dir=tmp_path / "output",
        overrides={
            "n_groups": 4,
            "gridsize": 1,
            "physics_forward_mode": "rectangular_scaled",
            "cnn_output_mode": "real_imag",
        },
    )

    assert payload.pt_data_config.scale_contract_version == CI_SCALE_CONTRACT
    assert payload.pt_data_config.measurement_domain == COUNT_INTENSITY


@pytest.mark.parametrize(
    "overrides",
    [
        {"scale_contract_version": LEGACY_SCALE_CONTRACT},
        {"measurement_domain": NORMALIZED_AMPLITUDE},
        {
            "scale_contract_version": LEGACY_SCALE_CONTRACT,
            "measurement_domain": COUNT_INTENSITY,
        },
    ],
)
def test_factory_rejects_partial_or_contradictory_profile_overrides(
    tmp_path,
    overrides,
):
    train_npz = tmp_path / "train.npz"
    _write_npz(train_npz)

    with pytest.raises(ValueError, match="scale_contract_version.*measurement_domain"):
        create_training_payload(
            train_data_file=train_npz,
            output_dir=tmp_path / "output",
            overrides={"n_groups": 4, **overrides},
        )


def test_factory_accepts_explicit_legacy_pair_for_training_and_inference(tmp_path):
    data_npz = tmp_path / "data.npz"
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "wts.h5.zip").write_bytes(b"placeholder")
    _write_npz(data_npz)
    profile = {
        "scale_contract_version": LEGACY_SCALE_CONTRACT,
        "measurement_domain": NORMALIZED_AMPLITUDE,
    }

    training = create_training_payload(
        train_data_file=data_npz,
        output_dir=tmp_path / "train-output",
        overrides={"n_groups": 4, **profile},
    )
    inference = create_inference_payload(
        model_path=model_dir,
        test_data_file=data_npz,
        output_dir=tmp_path / "inference-output",
        overrides={"n_groups": 4, **profile},
    )

    assert training.pt_data_config.scale_contract_version == LEGACY_SCALE_CONTRACT
    assert training.pt_data_config.measurement_domain == NORMALIZED_AMPLITUDE
    assert inference.pt_data_config.scale_contract_version == LEGACY_SCALE_CONTRACT
    assert inference.pt_data_config.measurement_domain == NORMALIZED_AMPLITUDE


def test_metadata_free_known_legacy_checkpoint_requires_both_overrides(tmp_path):
    checkpoint_path = _save_checkpoint_run(
        tmp_path / "legacy-run",
        legacy=True,
        metadata_free_legacy=True,
    )

    with pytest.raises(
        ValueError,
        match="--scale-contract-version legacy_v1.*--measurement-domain normalized_amplitude",
    ):
        load_checkpoint_with_configs(
            str(checkpoint_path),
            PtychoPINN_Lightning,
            device="cpu",
        )


def test_metadata_free_known_legacy_checkpoint_loads_strictly_with_pair(tmp_path):
    checkpoint_path = _save_checkpoint_run(
        tmp_path / "legacy-run",
        legacy=True,
        metadata_free_legacy=True,
    )

    model, configs = load_checkpoint_with_configs(
        str(checkpoint_path),
        PtychoPINN_Lightning,
        device="cpu",
        scale_contract_version=LEGACY_SCALE_CONTRACT,
        measurement_domain=NORMALIZED_AMPLITUDE,
    )

    assert model.data_config.scale_contract_version == LEGACY_SCALE_CONTRACT
    assert model.data_config.measurement_domain == NORMALIZED_AMPLITUDE
    assert configs[0].scale_contract_version == LEGACY_SCALE_CONTRACT
    assert configs[0].measurement_domain == NORMALIZED_AMPLITUDE


def test_ci_checkpoint_recovers_frozen_training_statistics(tmp_path):
    expected = {
        "rms_input_scale": torch.tensor([0.125]),
        "mean_measured_intensity": torch.tensor([17.0]),
    }
    checkpoint_path = _save_checkpoint_run(
        tmp_path / "ci-run",
        ci_statistics=expected,
    )

    model, _ = load_checkpoint_with_configs(
        str(checkpoint_path),
        PtychoPINN_Lightning,
        device="cpu",
    )

    actual = model.get_ci_statistics()
    assert actual is not None
    for name, value in expected.items():
        torch.testing.assert_close(actual[name], value)


def test_ci_checkpoint_rejects_missing_frozen_training_statistics(tmp_path):
    checkpoint_path = _save_checkpoint_run(tmp_path / "ci-run")

    with pytest.raises(ValueError, match="CI checkpoint.*ci_statistics"):
        load_checkpoint_with_configs(
            str(checkpoint_path),
            PtychoPINN_Lightning,
            device="cpu",
        )


def test_ci_bundle_recovers_profile_configs_and_frozen_statistics(tmp_path):
    data_config, model_config, training_config, inference_config, _ = _tiny_configs()
    model = PtychoPINN_Lightning(
        model_config,
        data_config,
        training_config,
        inference_config,
    )
    expected = {
        "rms_input_scale": torch.tensor([0.375]),
        "mean_measured_intensity": torch.tensor([9.0]),
    }
    model.register_ci_statistics(expected)
    base_path = tmp_path / "bundle" / "wts.h5"
    canonical_config = CanonicalTrainingConfig(
        model=CanonicalModelConfig(N=64, gridsize=1),
        output_dir=tmp_path / "bundle",
    )
    save_torch_bundle(
        {
            "autoencoder": model,
            "diffraction_to_obj": model,
        },
        str(base_path),
        canonical_config,
    )
    components._persist_bundle_scaling_metadata(
        base_path.with_suffix(".h5.zip"),
        model,
    )

    with zipfile.ZipFile(base_path.with_suffix(".h5.zip"), "r") as archive:
        manifest = dill.loads(archive.read("manifest.dill"))
        persisted_identity = torch.load(
            io.BytesIO(archive.read("torch_scaling_metadata.pt")),
            map_location="cpu",
            weights_only=False,
        )
    assert manifest["backend"] == "pytorch"
    assert manifest["artifact_schema_version"] == "torch-artifact-portable-v1"
    assert persisted_identity["schema_version"] == "torch-artifact-portable-v1"
    assert persisted_identity["model_spec"]["schema_version"] == (
        "torch-model-spec-portable-v1"
    )

    models, params = components.load_inference_bundle_torch(tmp_path / "bundle")

    restored = models["diffraction_to_obj"]
    restored_autoencoder = models["autoencoder"]
    assert restored.model_config.physics_forward_mode == "rectangular_scaled"
    assert restored_autoencoder.model_config.physics_forward_mode == "rectangular_scaled"
    assert restored.data_config.scale_contract_version == CI_SCALE_CONTRACT
    assert params["scale_contract_version"] == CI_SCALE_CONTRACT
    assert legacy_params.cfg["scale_contract_version"] == CI_SCALE_CONTRACT
    for name, value in expected.items():
        torch.testing.assert_close(restored.get_ci_statistics()[name], value)
        torch.testing.assert_close(restored_autoencoder.get_ci_statistics()[name], value)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored_autoencoder.state_dict()[name], value)


def test_checkpoint_loader_rejects_architecture_era_state_mismatch(tmp_path):
    checkpoint_path = _save_checkpoint_run(
        tmp_path / "legacy-run",
        legacy=True,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    removed_key = next(iter(checkpoint["state_dict"]))
    checkpoint["state_dict"].pop(removed_key)
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(RuntimeError, match="architecture-era.*regenerate"):
        load_checkpoint_with_configs(
            str(checkpoint_path),
            PtychoPINN_Lightning,
            device="cpu",
        )


@pytest.mark.parametrize("skip_anchor", [False, True])
def test_flux_evaluator_subprocess_load_smoke_reaches_ci_fields(
    tmp_path,
    skip_anchor,
):
    statistics = {
        "rms_input_scale": torch.tensor([0.25]),
        "mean_measured_intensity": torch.tensor([4.0]),
    }
    checkpoint_path = _save_checkpoint_run(
        tmp_path / "ci-run",
        ci_statistics=statistics,
    )
    test_npz = tmp_path / "smoke.npz"
    _write_npz(test_npz, n_images=2)
    command = [
        sys.executable,
        "scripts/studies/flux_sweep_eval.py",
        "--checkpoint",
        str(checkpoint_path),
        "--anchor-checkpoint",
        str(checkpoint_path),
        "--checkpoint-smoke",
        "--smoke-data",
        str(test_npz),
    ]
    if skip_anchor:
        command.append("--skip-anchor")

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "CI_FIELD_SMOKE_OK primary" in completed.stdout
    if skip_anchor:
        assert "CI_FIELD_SMOKE_OK anchor" not in completed.stdout
    else:
        assert "CI_FIELD_SMOKE_OK anchor" in completed.stdout


def test_flux_evaluator_missing_artifact_prints_deterministic_generation_path(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/studies/flux_sweep_eval.py",
            "--out",
            str(tmp_path / "missing-ci-root"),
            "--skip-anchor",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=60,
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert "scripts/studies/make_flux_sweep.py" in output
    assert "fluxsweep_N64_train.npz" in output
