from dataclasses import asdict

import pytest
import torch

from ptycho_torch.config_bridge import to_model_config
from ptycho_torch.config_params import (
    DataConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
)
from ptycho_torch.model_spec import derive_model_spec
from ptycho_torch.scaling_contract import (
    LEGACY_SCALE_CONTRACT,
    NORMALIZED_AMPLITUDE,
)


def _identity_parts(*, tensor_mask=False):
    data = DataConfig(N=64, C=1, grid_size=(1, 1), probe_scale=4.0)
    mask = torch.arange(16, dtype=torch.float32).reshape(4, 4) if tensor_mask else None
    model = ModelConfig(
        C_model=1,
        C_forward=1,
        object_big=False,
        probe_big=False,
        probe_mask=bool(tensor_mask),
        probe_mask_tensor=mask,
    )
    training = TrainingConfig(device="cpu", torch_loss_mode="poisson")
    inference = InferenceConfig()
    canonical = to_model_config(data, model)
    spec = derive_model_spec(canonical, model, data)
    return spec, data, training, inference


def test_current_artifact_roundtrip_preserves_model_spec_and_tensor_values():
    from ptycho_torch.artifact_schema import (
        CURRENT_ARTIFACT_SCHEMA_VERSION,
        decode_artifact_identity,
        encode_artifact_identity,
        from_json_payload,
        to_json_payload,
    )

    spec, data, training, inference = _identity_parts(tensor_mask=True)
    payload = encode_artifact_identity(
        spec,
        data,
        training,
        inference,
        ci_statistics={"rms_input_scale": torch.tensor([0.5])},
    )
    decoded = decode_artifact_identity(from_json_payload(to_json_payload(payload)))

    assert payload["backend"] == "pytorch"
    assert payload["schema_version"] == CURRENT_ARTIFACT_SCHEMA_VERSION
    torch.testing.assert_close(
        decoded.model_spec.to_model_config().probe_mask_tensor,
        spec.to_model_config().probe_mask_tensor,
    )
    assert decoded.data_config == data
    assert decoded.training_config == training
    assert decoded.inference_config == inference
    assert decoded.ci_statistics == {"rms_input_scale": [0.5]}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("backend", "tensorflow", r"backend.*tensorflow"),
        ("schema_version", "torch-artifact-v999", r"schema.*v999"),
        ("schema_version", "torch-artifact-v1", r"schema.*torch-artifact-v1"),
    ],
)
def test_current_artifact_rejects_unknown_backend_or_schema(field, value, message):
    from ptycho_torch.artifact_schema import (
        decode_artifact_identity,
        encode_artifact_identity,
    )

    payload = encode_artifact_identity(*_identity_parts())
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        decode_artifact_identity(payload)


def test_unversioned_current_sections_require_exact_field_sets():
    from ptycho_torch.artifact_schema import upgrade_unversioned_sections

    spec, data, training, inference = _identity_parts()
    model_payload = spec.to_payload()["model_config"]
    model_payload.pop("fno_width")

    with pytest.raises(ValueError, match=r"unversioned.*model_config.*missing.*fno_width"):
        upgrade_unversioned_sections(
            data_config=asdict(data),
            model_config=model_payload,
            training_config=asdict(training),
            inference_config=asdict(inference),
        )


def test_known_metadata_free_legacy_upgrade_adds_only_explicit_profile():
    from ptycho_torch.artifact_schema import upgrade_unversioned_sections

    spec, data, training, inference = _identity_parts()
    data_payload = asdict(data)
    data_payload.pop("scale_contract_version")
    data_payload.pop("measurement_domain")

    decoded = upgrade_unversioned_sections(
        data_config=data_payload,
        model_config=spec.to_payload()["model_config"],
        training_config=asdict(training),
        inference_config=asdict(inference),
        explicit_profile=(LEGACY_SCALE_CONTRACT, NORMALIZED_AMPLITUDE),
        metadata_free_legacy=True,
    )

    assert decoded.data_config.scale_contract_version == LEGACY_SCALE_CONTRACT
    assert decoded.data_config.measurement_domain == NORMALIZED_AMPLITUDE


def test_bundle_manifest_is_checked_before_construction():
    from ptycho_torch.artifact_schema import validate_torch_bundle_manifest

    valid = {
        "version": "2.0-pytorch",
        "models": ["autoencoder", "diffraction_to_obj"],
    }
    assert validate_torch_bundle_manifest(valid) == "metadata-free-legacy"

    current = {
        **valid,
        "backend": "pytorch",
        "artifact_schema_version": "torch-artifact-portable-v1",
    }
    assert validate_torch_bundle_manifest(current) == "torch-artifact-portable-v1"

    with pytest.raises(ValueError, match=r"backend.*tensorflow"):
        validate_torch_bundle_manifest({**current, "backend": "tensorflow"})
    with pytest.raises(ValueError, match=r"version.*9.0-pytorch"):
        validate_torch_bundle_manifest({**valid, "version": "9.0-pytorch"})
    with pytest.raises(ValueError, match=r"roles.*autoencoder.*diffraction_to_obj"):
        validate_torch_bundle_manifest({**valid, "models": ["autoencoder"]})


def test_current_application_checkpoint_dual_writes_identity_and_reloads(tmp_path):
    from lightning.pytorch import Trainer

    from ptycho_torch.application_factory import build_ptychopinn_application
    from ptycho_torch.artifact_schema import CURRENT_ARTIFACT_SCHEMA_VERSION
    from ptycho_torch.model import PtychoPINN_Lightning

    spec, data, training, inference = _identity_parts()
    model = build_ptychopinn_application(spec, data, training, inference)
    trainer = Trainer(
        max_epochs=0,
        enable_checkpointing=True,
        logger=False,
        enable_progress_bar=False,
        accelerator="cpu",
        default_root_dir=tmp_path,
    )
    trainer.strategy._lightning_module = model
    checkpoint_path = tmp_path / "current.ckpt"
    trainer.save_checkpoint(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["ptychopinn_artifact"] == {
        "backend": "pytorch",
        "schema_version": CURRENT_ARTIFACT_SCHEMA_VERSION,
    }
    assert checkpoint["hyper_parameters"]["model_spec"]["schema_version"] == (
        "torch-model-spec-portable-v1"
    )

    loaded = PtychoPINN_Lightning.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )
    assert loaded.model_config == model.model_config


def test_checkpoint_model_spec_unknown_schema_fails_before_state_load(tmp_path):
    from lightning.pytorch import Trainer

    from ptycho_torch.application_factory import build_ptychopinn_application
    from ptycho_torch.model import PtychoPINN_Lightning

    spec, data, training, inference = _identity_parts()
    model = build_ptychopinn_application(spec, data, training, inference)
    trainer = Trainer(
        max_epochs=0,
        enable_checkpointing=True,
        logger=False,
        enable_progress_bar=False,
        accelerator="cpu",
        default_root_dir=tmp_path,
    )
    trainer.strategy._lightning_module = model
    checkpoint_path = tmp_path / "unsupported.ckpt"
    trainer.save_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["hyper_parameters"]["model_spec"]["schema_version"] = (
        "torch-model-spec-v999"
    )
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match=r"ModelSpec schema.*v999"):
        PtychoPINN_Lightning.load_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
        )


def test_current_checkpoint_rejects_missing_dual_written_config_field(tmp_path):
    from lightning.pytorch import Trainer

    from ptycho_torch.application_factory import build_ptychopinn_application
    from ptycho_torch.model import PtychoPINN_Lightning

    spec, data, training, inference = _identity_parts()
    model = build_ptychopinn_application(spec, data, training, inference)
    trainer = Trainer(
        max_epochs=0,
        enable_checkpointing=True,
        logger=False,
        enable_progress_bar=False,
        accelerator="cpu",
        default_root_dir=tmp_path,
    )
    trainer.strategy._lightning_module = model
    checkpoint_path = tmp_path / "missing-field.ckpt"
    trainer.save_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["hyper_parameters"]["data_config"].pop("N")
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match=r"data_config.*missing.*N"):
        PtychoPINN_Lightning.load_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
        )


def test_config_logger_dual_writes_current_sidecar_identity(tmp_path):
    import json
    from types import SimpleNamespace

    from ptycho_torch.application_factory import build_ptychopinn_application
    from ptycho_torch.artifact_schema import decode_artifact_identity, from_json_payload
    from ptycho_torch.config_params import DatagenConfig
    from ptycho_torch.lightning_utils import ConfigLogger

    spec, data, training, inference = _identity_parts(tensor_mask=True)
    model = build_ptychopinn_application(spec, data, training, inference)
    callback = ConfigLogger(data, model.model_config, training, inference, DatagenConfig())

    callback.on_train_start(SimpleNamespace(log_dir=str(tmp_path)), model)

    path = tmp_path / "configs" / "artifact_identity.json"
    decoded = decode_artifact_identity(from_json_payload(json.loads(path.read_text())))
    torch.testing.assert_close(
        decoded.model_spec.to_model_config().probe_mask_tensor,
        spec.to_model_config().probe_mask_tensor,
    )


def test_sidecar_loader_uses_versioned_identity_without_lossy_model_json(tmp_path):
    from types import SimpleNamespace

    from lightning.pytorch import Trainer

    from ptycho_torch.application_factory import build_ptychopinn_application
    from ptycho_torch.config_params import DatagenConfig
    from ptycho_torch.lightning_utils import ConfigLogger, load_configs_from_checkpoint

    spec, data, training, inference = _identity_parts(tensor_mask=True)
    model = build_ptychopinn_application(spec, data, training, inference)
    run_dir = tmp_path / "run"
    callback = ConfigLogger(data, model.model_config, training, inference, DatagenConfig())
    callback.on_train_start(SimpleNamespace(log_dir=str(run_dir)), model)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "current.ckpt"
    trainer = Trainer(
        max_epochs=0,
        enable_checkpointing=True,
        logger=False,
        enable_progress_bar=False,
        accelerator="cpu",
        default_root_dir=run_dir,
    )
    trainer.strategy._lightning_module = model
    trainer.save_checkpoint(checkpoint_path)

    loaded_data, loaded_model, loaded_training, loaded_inference, _ = (
        load_configs_from_checkpoint(checkpoint_path)
    )

    assert loaded_data == data
    assert loaded_training == training
    assert loaded_inference == inference
    torch.testing.assert_close(
        loaded_model.probe_mask_tensor,
        model.model_config.probe_mask_tensor,
    )


def test_transitional_ci_entrypoints_bundle_upgrades_and_strict_loads(tmp_path):
    import io
    import zipfile

    from ptycho.config.config import ModelConfig as CanonicalModelConfig
    from ptycho.config.config import TrainingConfig as CanonicalTrainingConfig
    from ptycho_torch.application_factory import build_ptychopinn_application
    from ptycho_torch.model_manager import save_torch_bundle
    from ptycho_torch.workflows.components import load_inference_bundle_torch

    spec, data, training, inference = _identity_parts()
    model = build_ptychopinn_application(spec, data, training, inference)
    bundle_dir = tmp_path / "bundle"
    base_path = bundle_dir / "wts.h5"
    save_torch_bundle(
        {"autoencoder": model, "diffraction_to_obj": model},
        str(base_path),
        CanonicalTrainingConfig(
            model=CanonicalModelConfig(N=64, gridsize=1),
            output_dir=bundle_dir,
        ),
    )
    transitional = {
        "schema_version": "ci-entrypoints-v1",
        "data_config": asdict(data),
        "model_config": asdict(model.model_config),
        "training_config": asdict(training),
        "inference_config": asdict(inference),
        "ci_statistics": None,
    }
    buffer = io.BytesIO()
    torch.save(transitional, buffer)
    with zipfile.ZipFile(
        base_path.with_suffix(".h5.zip"), "a", zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr("torch_scaling_metadata.pt", buffer.getvalue())

    loaded, _ = load_inference_bundle_torch(bundle_dir)

    for key, value in model.state_dict().items():
        torch.testing.assert_close(
            loaded["diffraction_to_obj"].state_dict()[key],
            value,
        )
