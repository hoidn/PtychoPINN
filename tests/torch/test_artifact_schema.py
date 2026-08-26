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
    data = DataConfig(N=64, gridsize=1, probe_scale=4.0)
    mask = torch.arange(16, dtype=torch.float32).reshape(4, 4) if tensor_mask else None
    model = ModelConfig(
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


def _legacy_data_payload(data) -> dict:
    """Project a live (v4) DataConfig onto the historical unversioned wire shape."""
    payload = asdict(data)
    gridsize = payload.pop("gridsize")
    payload["C"] = gridsize * gridsize
    payload["grid_size"] = (gridsize, gridsize)
    payload["n_subsample"] = payload.pop("n_raw_frames_selected")
    payload["K"] = payload.pop("neighbor_count")
    return payload


def _legacy_artifact_payload(*, era, c_model, c_forward, data_c, grid_size):
    """Build a legacy artifact payload with chosen model twins and data grid.

    The model section stores ``C_model``/``C_forward`` and the data section
    stores ``C``/``grid_size``; callers may make the two sides disagree to pin
    the cross-section channel-faithfulness rejection.
    """
    from ptycho_torch.artifact_schema import encode_artifact_identity

    spec, data, training, inference = _identity_parts()
    payload = encode_artifact_identity(spec, data, training, inference)

    spec_version = (
        "torch-model-spec-portable-v1"
        if era == "torch-artifact-portable-v1"
        else "torch-model-spec-portable-v2"
    )
    model_fields = dict(payload["model_spec"]["model_config"])
    model_fields["C_model"] = c_model
    model_fields["C_forward"] = c_forward
    if era == "torch-artifact-portable-v1":
        grouped = model_fields.pop("object_layout") == "grouped_patches"
        model_fields.pop("training_canvas")
        model_fields["object_big"] = grouped
    payload["model_spec"] = {
        **payload["model_spec"],
        "schema_version": spec_version,
        "model_config": model_fields,
    }
    payload["schema_version"] = era

    data_section = _legacy_data_payload(data)
    data_section["C"] = data_c
    data_section["grid_size"] = grid_size
    payload["data_config"] = data_section
    return payload


@pytest.mark.parametrize(
    "era",
    ["torch-artifact-portable-v1", "torch-artifact-portable-v2"],
)
def test_legacy_artifact_rejects_model_data_channel_disagreement(era):
    from ptycho_torch.artifact_schema import decode_artifact_identity

    payload = _legacy_artifact_payload(
        era=era,
        c_model=1,
        c_forward=1,
        data_c=4,
        grid_size=(2, 2),
    )
    with pytest.raises(
        ValueError,
        match=r"C_model=1 conflicts with data section grid product 4",
    ):
        decode_artifact_identity(payload)


def test_unversioned_upgrade_rejects_model_data_channel_disagreement():
    from ptycho_torch.artifact_schema import upgrade_unversioned_sections

    spec, data, training, inference = _identity_parts()
    data_payload = _legacy_data_payload(data)
    data_payload["C"] = 4
    data_payload["grid_size"] = (2, 2)
    model_payload = dict(spec.to_payload()["model_config"], C_model=1, C_forward=1)

    with pytest.raises(
        ValueError,
        match=r"C_model=1 conflicts with data section grid product 4",
    ):
        upgrade_unversioned_sections(
            data_config=data_payload,
            model_config=model_payload,
            training_config=asdict(training),
            inference_config=asdict(inference),
        )


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


def test_current_artifact_rejects_historical_data_initialization_identity():
    from ptycho_torch.artifact_schema import (
        decode_artifact_identity,
        encode_artifact_identity,
    )

    payload = encode_artifact_identity(*_identity_parts())
    payload["model_spec"]["model_config"]["rect_s1s2_init"] = "data"

    with pytest.raises(
        ValueError,
        match=r"data.*unsupported.*ones.*dose_closure.*historical code or retraining",
    ):
        decode_artifact_identity(payload)


def test_unversioned_current_sections_require_exact_field_sets():
    from ptycho_torch.artifact_schema import upgrade_unversioned_sections

    spec, data, training, inference = _identity_parts()
    model_payload = dict(spec.to_payload()["model_config"], object_big=False)
    model_payload.pop("fno_width")

    with pytest.raises(ValueError, match=r"unversioned.*model_config.*missing.*fno_width"):
        upgrade_unversioned_sections(
            data_config=_legacy_data_payload(data),
            model_config=model_payload,
            training_config=asdict(training),
            inference_config=asdict(inference),
        )


def test_known_metadata_free_legacy_upgrade_adds_only_explicit_profile():
    from ptycho_torch.artifact_schema import upgrade_unversioned_sections

    spec, data, training, inference = _identity_parts()
    data_payload = _legacy_data_payload(data)
    data_payload.pop("scale_contract_version")
    data_payload.pop("measurement_domain")

    decoded = upgrade_unversioned_sections(
        data_config=data_payload,
        model_config=dict(spec.to_payload()["model_config"], object_big=False),
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
        "artifact_schema_version": "torch-artifact-portable-v2",
    }
    assert validate_torch_bundle_manifest(current) == "torch-artifact-portable-v2"

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
        "torch-model-spec-portable-v3"
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


def test_checkpoint_historical_data_identity_fails_before_state_restoration(
    tmp_path,
    monkeypatch,
):
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
    checkpoint_path = tmp_path / "historical-data.ckpt"
    trainer.save_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["hyper_parameters"]["model_spec"]["model_config"][
        "rect_s1s2_init"
    ] = "data"
    torch.save(checkpoint, checkpoint_path)

    state_restoration_started = False

    def fail_if_state_restoration_starts(self, *args, **kwargs):
        nonlocal state_restoration_started
        state_restoration_started = True
        raise AssertionError("state restoration started before identity validation")

    monkeypatch.setattr(
        PtychoPINN_Lightning,
        "load_state_dict",
        fail_if_state_restoration_starts,
    )

    with pytest.raises(
        ValueError,
        match=r"data.*unsupported.*ones.*dose_closure.*historical code or retraining",
    ):
        PtychoPINN_Lightning.load_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
        )
    assert state_restoration_started is False


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

    from ptycho_torch.checkpoint_decode import decode_checkpoint_hparams

    with pytest.raises(ValueError, match=r"data_config.*missing.*N"):
        decode_checkpoint_hparams(checkpoint["hyper_parameters"])


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
        "data_config": _legacy_data_payload(data),
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
