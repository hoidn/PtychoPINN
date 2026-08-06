"""Focused closed-form gauge initialization tests for rectangular CI physics."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _tiny_rect_scaled_module():
    """Smallest real PtychoPINN_Lightning under the CI contract (2026-07-14 RCA
    arm: N=64, gridsize=1, architecture='cnn', cnn_output_mode='real_imag',
    physics_forward_mode='rectangular_scaled', count_intensity/ci_intensity_v2,
    amplitude_physics_gain=1.0), plus one real training batch built through the
    same factory (create_training_payload), CI dict adapter, and workflow
    dataloader path that _train_with_lightning / run_torch_training use."""
    import tempfile
    from pathlib import Path

    import numpy as np

    from ptycho_torch import helper as hh
    from ptycho_torch.config_factory import create_training_payload
    from ptycho_torch.config_params import InferenceConfig
    from ptycho_torch.model import PtychoPINN_Lightning
    from ptycho_torch.workflows.components import (
        NormalizedAmplitudeCIDictAdapter,
        _build_lightning_dataloaders,
    )

    torch.manual_seed(20260714)
    rng = np.random.default_rng(20260714)
    B, N = 4, 64
    amplitudes = rng.uniform(0.1, 1.0, size=(B, N, N, 1)).astype(np.float32)
    probe = np.ones((N, N), dtype=np.complex64)

    tmpdir = Path(tempfile.mkdtemp(prefix="rect_s1s2_calib_"))
    npz_path = tmpdir / "tiny_train.npz"
    np.savez(
        npz_path,
        diffraction=amplitudes[..., 0],
        xcoords=np.linspace(0.0, 3.0, B),
        ycoords=np.linspace(0.0, 3.0, B),
        probeGuess=probe,
        objectGuess=np.ones((2 * N, 2 * N), dtype=np.complex64),
    )
    payload = create_training_payload(
        train_data_file=npz_path,
        output_dir=tmpdir / "out",
        overrides={
            "n_groups": B,
            "gridsize": 1,
            "architecture": "cnn",
            "model_type": "Unsupervised",
            "cnn_output_mode": "real_imag",
            "physics_forward_mode": "rectangular_scaled",
            "torch_loss_mode": "poisson",
            "scale_contract_version": "ci_intensity_v2",
            "measurement_domain": "count_intensity",
            "amplitude_physics_gain": 1.0,
            "object_big": False,
            "batch_size": B,
        },
    )
    model = PtychoPINN_Lightning(
        payload.pt_model_config,
        payload.pt_data_config,
        payload.pt_training_config,
        InferenceConfig(),
    )

    # Same CI count-domain container preparation as run_torch_training's CI arm
    # (high per-pixel counts reproduce the RCA's init-scale mismatch).
    container = {"observed_images": amplitudes, "probe": probe}
    count_amplitude_scale = hh.derive_intensity_scale_from_amplitudes(
        torch.as_tensor(amplitudes), 1e9
    )
    NormalizedAmplitudeCIDictAdapter(
        count_amplitude_scale=count_amplitude_scale,
        N=N,
    ).adapt(container)
    container["X"] = container["measured_intensity"]

    train_loader, _ = _build_lightning_dataloaders(
        train_container=container,
        test_container=None,
        config=None,
        payload=payload,
    )
    batch = next(iter(train_loader))
    return model, batch


def _record_type():
    try:
        from ptycho_torch.rect_s1s2_initialization import (
            RectS1S2InitializationRecord,
        )
    except (ImportError, ModuleNotFoundError) as error:
        pytest.fail(f"typed rect_s1s2 initialization record is missing: {error}")
    return RectS1S2InitializationRecord


def _initialization_payload(mode="dose_closure", *, gauge=3.25):
    if mode == "ones":
        return {
            "schema_version": "rect-s1s2-initialization-v1",
            "mode": "ones",
            "solved_gauge": 1.0,
            "method": "unit_default_no_solve",
            "sampled_patterns": 0,
        }
    return {
        "schema_version": "rect-s1s2-initialization-v1",
        "mode": "dose_closure",
        "solved_gauge": gauge,
        "method": "dose_closure_unit_object",
        "sampled_patterns": 256,
    }


def test_initialization_record_is_frozen_versioned_and_round_trips_strictly():
    from dataclasses import FrozenInstanceError

    record_type = _record_type()
    payload = _initialization_payload()

    record = record_type.from_mapping(payload)

    assert record.to_jsonable() == payload
    with pytest.raises(FrozenInstanceError):
        record.mode = "ones"


def test_initialization_record_constructor_cannot_bypass_validation():
    record_type = _record_type()

    with pytest.raises(ValueError, match="sampled_patterns"):
        record_type(
            mode="dose_closure",
            solved_gauge=3.25,
            method="dose_closure_unit_object",
            sampled_patterns=1,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(schema_version="obsolete-v0"), "schema_version"),
        (lambda value: value.pop("method"), "fields"),
        (lambda value: value.update(extra=True), "fields"),
        (
            lambda value: value.update(mode="data"),
            "data.*unsupported.*ones.*dose_closure.*historical code or retraining",
        ),
        (lambda value: value.update(solved_gauge=float("nan")), "solved_gauge"),
        (lambda value: value.update(solved_gauge=0.0), "solved_gauge"),
        (lambda value: value.update(sampled_patterns=True), "sampled_patterns"),
        (lambda value: value.update(sampled_patterns=255), "sampled_patterns"),
        (lambda value: value.update(method="unit_default_no_solve"), "method"),
    ],
)
def test_initialization_record_rejects_malformed_schema_keys_and_values(
    mutation,
    message,
):
    record_type = _record_type()
    payload = _initialization_payload()
    mutation(payload)

    with pytest.raises((TypeError, ValueError), match=message):
        record_type.from_mapping(payload)


class _ReadyBatchDataset(torch.utils.data.Dataset):
    def __init__(self, fields, outer_probe, outer_scale):
        self.fields = fields
        self.outer_probe = outer_probe
        self.outer_scale = outer_scale

    def __len__(self):
        return self.fields["images"].shape[0]

    def __getitem__(self, index):
        return (
            {name: value[index] for name, value in self.fields.items()},
            self.outer_probe[index],
            self.outer_scale[index],
        )


class _RealRectBoundary(torch.nn.Module):
    """Small module boundary owning the repository's real ForwardModel."""

    def __init__(
        self,
        *,
        detector_size=8,
        num_datasets=2,
        probe_mask=True,
        channels=1,
    ):
        from ptycho_torch.config_params import DataConfig, ModelConfig
        from ptycho_torch.model import ForwardModel

        super().__init__()
        model_config = ModelConfig(
            physics_forward_mode="rectangular_scaled",
            cnn_output_mode="real_imag",
            object_big=False,
            object_layout="single_patch",
            training_canvas="independent",
            probe_big=False,
            C_model=channels,
            C_forward=channels,
            num_datasets=num_datasets,
            probe_mask=probe_mask,
            probe_mask_diameter=(
                float(detector_size - 2) if probe_mask else None
            ),
            probe_mask_sigma=0.0,
        )
        data_config = DataConfig(
            N=detector_size,
            C=channels,
            grid_size=(1, 1) if channels == 1 else (3, 3),
        )
        self.model = torch.nn.Module()
        self.model.forward_model = ForwardModel(model_config, data_config)


def _known_gauge_loader(
    *, patterns=300, batch_size=73, gauge=3.25, channels=1
):
    model = _RealRectBoundary(channels=channels)
    forward_model = model.model.forward_model
    scaler = forward_model.rect_scaler
    detector_size = forward_model.N

    images = torch.ones(patterns, channels, detector_size, detector_size)
    positions = torch.zeros(patterns, 1, channels, 2)
    experiment_ids = torch.arange(patterns, dtype=torch.long) % 2
    probe = torch.ones(
        patterns,
        1,
        1,
        detector_size,
        detector_size,
        dtype=torch.complex64,
    )
    probe_normalization = torch.linspace(0.75, 1.25, patterns).view(
        patterns, 1, 1, 1, 1
    )
    output_scale = probe_normalization.reshape(patterns, 1, 1, 1).reciprocal()
    unit_object = torch.ones_like(images, dtype=torch.complex64)

    with torch.no_grad():
        scaler.s1.data.fill_(gauge)
        scaler.s2.data.fill_(gauge)
        measured = forward_model(
            unit_object,
            torch.zeros_like(images),
            positions,
            probe,
            output_scale,
            experiment_ids,
        ).detach()
        scaler.s1.data.fill_(1.0)
        scaler.s2.data.fill_(1.0)

    # Detector-pattern slots beyond the representative prefix deliberately use
    # another dose. A row-counted, shuffled, or overlong solve cannot pass.
    measured_patterns = measured.reshape(
        patterns * channels,
        detector_size,
        detector_size,
    )
    if measured_patterns.shape[0] > 256:
        measured_patterns[256:] *= 9.0

    fields = {
        "images": images,
        "measured_intensity": measured,
        "observed_images": measured.clone(),
        "coords_relative": positions,
        "rms_input_scale": torch.ones(patterns, 1, 1, 1),
        "mean_measured_intensity": measured.mean(dim=(1, 2, 3), keepdim=True),
        "experiment_id": experiment_ids,
        "probe_training": probe,
        "probe_physical": probe.clone(),
        "probe_normalization": probe_normalization,
    }
    dataset = _ReadyBatchDataset(
        fields,
        outer_probe=probe,
        outer_scale=probe_normalization.reshape(patterns, 1, 1, 1),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(999),
    )
    return model, loader, dataset


def _known_unit_intensity(model, *, gauge, probe):
    forward_model = model.model.forward_model
    detector_size = forward_model.N
    fields = torch.ones(1, 1, detector_size, detector_size)
    positions = torch.zeros(1, 1, 1, 2)
    experiment_ids = torch.zeros(1, dtype=torch.long)
    probe_batch = probe.view(1, 1, 1, detector_size, detector_size)
    with torch.no_grad():
        forward_model.rect_scaler.s1.data.fill_(gauge)
        forward_model.rect_scaler.s2.data.fill_(gauge)
        measured = forward_model(
            torch.ones_like(fields, dtype=torch.complex64),
            torch.zeros_like(fields),
            positions,
            probe_batch,
            torch.ones(1, 1, 1, 1),
            experiment_ids,
        ).detach()
        forward_model.rect_scaler.s1.data.fill_(1.0)
        forward_model.rect_scaler.s2.data.fill_(1.0)
    return measured[0, 0]


def _known_gauge_dict_loader(*, patterns=300, batch_size=73, gauge=2.75):
    from ptycho_torch.config_params import TrainingConfig
    from ptycho_torch.workflows import components

    model = _RealRectBoundary(
        detector_size=8,
        num_datasets=1,
        probe_mask=False,
    )
    forward_model = model.model.forward_model
    probe = torch.ones(8, 8, dtype=torch.complex64)
    measured = _known_unit_intensity(model, gauge=gauge, probe=probe)
    amplitude = measured.sqrt().view(1, 8, 8, 1).expand(
        patterns, -1, -1, -1
    ).clone()
    container = {
        "X": amplitude.clone(),
        "observed_images": amplitude,
        "coords_relative": torch.zeros(patterns, 1, 2, 1),
        "probe": probe,
    }
    components.NormalizedAmplitudeCIDictAdapter(
        count_amplitude_scale=1.0,
        N=8,
        probe_scale=1.0,
        probe_mask=False,
    ).adapt(container)
    payload = SimpleNamespace(
        pt_data_config=forward_model.data_config,
        pt_model_config=forward_model.model_config,
        pt_training_config=TrainingConfig(
            batch_size=batch_size,
            torch_loss_mode="poisson",
        ),
        execution_config=None,
    )
    loader, _ = components._build_lightning_dataloaders(
        train_container=container,
        test_container=None,
        config=None,
        payload=payload,
    )
    return model, loader


def _known_legacy_scale_loader(*, patterns=256, batch_size=61, gauge=2.5):
    model = _RealRectBoundary()
    forward_model = model.model.forward_model
    scaler = forward_model.rect_scaler
    detector_size = forward_model.N
    images = torch.ones(patterns, 1, detector_size, detector_size)
    positions = torch.zeros(patterns, 1, 1, 2)
    experiment_ids = torch.arange(patterns, dtype=torch.long) % 2
    probe = torch.ones(
        patterns, 1, 1, detector_size, detector_size, dtype=torch.complex64
    )
    outer_scale = torch.linspace(0.8, 1.2, patterns).view(patterns, 1, 1, 1)
    physics_scale = torch.linspace(2.0, 5.0, patterns).view(patterns, 1, 1, 1)
    output_scale = torch.sqrt(
        1.0 / (outer_scale.square() * physics_scale + 1e-9)
    )
    with torch.no_grad():
        scaler.s1.data.fill_(gauge)
        scaler.s2.data.fill_(gauge)
        measured = forward_model(
            torch.ones_like(images, dtype=torch.complex64),
            torch.zeros_like(images),
            positions,
            probe,
            output_scale,
            experiment_ids,
        ).detach()
        scaler.s1.data.fill_(1.0)
        scaler.s2.data.fill_(1.0)

    fields = {
        "images": images,
        "observed_images": measured,
        "coords_relative": positions,
        "rms_scaling_constant": torch.ones(patterns, 1, 1, 1),
        "physics_scaling_constant": physics_scale,
        "experiment_id": experiment_ids,
    }
    dataset = _ReadyBatchDataset(fields, probe, outer_scale)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(321),
    )
    return model, loader


def test_dose_closure_recovers_known_gauge_with_real_multibatch_forward():
    from ptycho_torch.workflows import components

    model, loader, _ = _known_gauge_loader(gauge=3.25)
    record = components._initialize_rect_s1s2(
        model,
        mode="dose_closure",
        training_loader=loader,
    )

    scaler = model.model.forward_model.rect_scaler
    assert record["schema_version"] == "rect-s1s2-initialization-v1"
    assert record["mode"] == "dose_closure"
    assert record["solved_gauge"] == pytest.approx(3.25)
    assert record["method"] == "dose_closure_unit_object"
    assert record["sampled_patterns"] == 256
    assert torch.equal(scaler.s1.detach(), torch.full_like(scaler.s1, 3.25))
    assert torch.equal(scaler.s2.detach(), torch.full_like(scaler.s2, 3.25))


def test_dose_closure_counts_exact_detector_patterns_across_group_channels():
    """The 256-pattern prefix is flattened across B/C, not 256 group rows."""
    from ptycho_torch.workflows import components

    model, loader, _ = _known_gauge_loader(
        patterns=30,
        channels=9,
        batch_size=7,
        gauge=2.25,
    )

    record = components._initialize_rect_s1s2(
        model,
        mode="dose_closure",
        training_loader=loader,
    )

    assert record["solved_gauge"] == pytest.approx(2.25, rel=2e-6)
    assert record["sampled_patterns"] == 256


def test_dose_closure_uses_real_grid_dict_adapter_and_loader_path():
    from ptycho_torch.workflows import components

    model, loader = _known_gauge_dict_loader(gauge=2.75)

    record = components._initialize_rect_s1s2(
        model,
        mode="dose_closure",
        training_loader=loader,
    )

    assert type(loader.dataset).__name__ == "PtychoLightningDataset"
    assert record["solved_gauge"] == pytest.approx(2.75, rel=2e-6)
    assert record["sampled_patterns"] == 256


def test_dose_closure_preserves_tensordict_loader_batch_indexing_contract():
    from ptycho_torch.dataloader import TensorDictDataLoader
    from ptycho_torch.workflows import components

    model, _, dataset = _known_gauge_loader(gauge=3.25)
    loader = TensorDictDataLoader(
        dataset,
        batch_size=73,
        shuffle=True,
        collate_fn=lambda batch: batch,
        generator=torch.Generator().manual_seed(999),
    )

    record = components._initialize_rect_s1s2(
        model,
        mode="dose_closure",
        training_loader=loader,
    )

    assert record["solved_gauge"] == pytest.approx(3.25, rel=2e-6)
    assert record["sampled_patterns"] == 256


def test_dose_closure_uses_real_prebuilt_mmap_data_module_and_loader(
    tmp_path,
):
    from ptycho_torch.config_params import TrainingConfig
    from ptycho_torch.dataloader import PtychoDataset, TensorDictDataLoader
    from ptycho_torch.train_utils import PrebuiltPtychoDataModule
    from ptycho_torch.workflows import components

    patterns = 300
    gauge = 2.25
    model = _RealRectBoundary(
        detector_size=8,
        num_datasets=1,
        probe_mask=False,
    )
    forward_model = model.model.forward_model
    data_config = forward_model.data_config
    data_config.x_bounds = (0.0, 1.0)
    data_config.y_bounds = (0.0, 1.0)
    data_config.probe_scale = 1.0
    probe = torch.ones(8, 8, dtype=torch.complex64)
    measured = _known_unit_intensity(model, gauge=gauge, probe=probe)
    source_dir = tmp_path / "npz"
    source_dir.mkdir()
    np.savez(
        source_dir / "known-gauge.npz",
        diff3d=measured.numpy()[None].repeat(patterns, axis=0),
        xcoords=np.linspace(0.0, 1.0, patterns, dtype=np.float64),
        ycoords=np.linspace(0.0, 1.0, patterns, dtype=np.float64),
        probeGuess=probe.numpy(),
        objectGuess=np.ones((8, 8), dtype=np.complex64),
    )
    training_config = TrainingConfig(
        batch_size=73,
        torch_loss_mode="poisson",
        orchestrator="Mlflow",
        num_workers=0,
    )
    mmap_path = tmp_path / "memmap"
    source_dataset = PtychoDataset(
        ptycho_dir=str(source_dir),
        model_config=forward_model.model_config,
        data_config=data_config,
        training_config=training_config,
        data_dir=str(mmap_path),
        remake_map=True,
    )
    assert len(source_dataset) == patterns
    data_module = PrebuiltPtychoDataModule(
        str(mmap_path),
        forward_model.model_config,
        data_config,
        training_config,
    )
    loader = components._rect_s1s2_training_loader(
        data_module,
        train_loader=None,
        mode="dose_closure",
    )

    assert isinstance(loader, TensorDictDataLoader)
    assert isinstance(loader.dataset.dataset, PtychoDataset)
    record = components._initialize_rect_s1s2(
        model,
        mode="dose_closure",
        training_loader=loader,
    )

    assert record["solved_gauge"] == pytest.approx(gauge, rel=2e-6)
    assert record["sampled_patterns"] == 256


def test_dose_closure_rejects_legacy_normalized_amplitude_loader():
    from ptycho_torch.workflows import components

    model, loader = _known_legacy_scale_loader()
    with pytest.raises(ValueError, match="requires CI count-intensity"):
        components._initialize_rect_s1s2(
            model,
            mode="dose_closure",
            training_loader=loader,
        )


def test_ones_resets_exact_units_without_consuming_training_data():
    from ptycho_torch.workflows import components

    model = _RealRectBoundary()
    scaler = model.model.forward_model.rect_scaler
    scaler.s1.data.fill_(7.0)
    scaler.s2.data.fill_(9.0)

    class _ExplodingLoader:
        def __iter__(self):
            raise AssertionError("ones mode must not consume a batch")

    record = components._initialize_rect_s1s2(
        model,
        mode="ones",
        training_loader=_ExplodingLoader(),
    )

    assert record == _initialization_payload("ones")
    assert torch.equal(scaler.s1.detach(), torch.ones_like(scaler.s1))
    assert torch.equal(scaler.s2.detach(), torch.ones_like(scaler.s2))


def test_dose_closure_fails_clearly_when_fewer_than_256_patterns_exist():
    from ptycho_torch.workflows import components

    model, loader, _ = _known_gauge_loader(patterns=255)

    with pytest.raises(
        ValueError,
        match=r"sampled 255.*required 256.*--rect-s1s2-init ones",
    ):
        components._initialize_rect_s1s2(
            model,
            mode="dose_closure",
            training_loader=loader,
        )


def test_dose_closure_reports_detector_pattern_count_for_short_grouped_loader():
    from ptycho_torch.workflows import components

    model, loader, _ = _known_gauge_loader(
        patterns=28,
        channels=9,
        batch_size=7,
    )

    with pytest.raises(
        ValueError,
        match=r"sampled 252.*required 256.*--rect-s1s2-init ones",
    ):
        components._initialize_rect_s1s2(
            model,
            mode="dose_closure",
            training_loader=loader,
        )


def test_dose_closure_rejects_nonfinite_observed_counts():
    from ptycho_torch.workflows import components

    model, loader, dataset = _known_gauge_loader()
    dataset.fields["measured_intensity"][0, 0, 0, 0] = torch.nan

    with pytest.raises(ValueError, match="observed count sum.*finite"):
        components._initialize_rect_s1s2(
            model,
            mode="dose_closure",
            training_loader=loader,
        )


def test_dose_closure_rejects_zero_observed_count_sum():
    from ptycho_torch.workflows import components

    model, loader, dataset = _known_gauge_loader()
    dataset.fields["measured_intensity"].zero_()

    with pytest.raises(ValueError, match="observed count sum.*positive"):
        components._initialize_rect_s1s2(
            model,
            mode="dose_closure",
            training_loader=loader,
        )


def test_dose_closure_rejects_negative_observed_count_with_positive_sum():
    from ptycho_torch.workflows import components

    model, loader, dataset = _known_gauge_loader()
    dataset.fields["measured_intensity"][0, 0, 0, 0] = -1.0

    with pytest.raises(ValueError, match="observed counts.*nonnegative"):
        components._initialize_rect_s1s2(
            model,
            mode="dose_closure",
            training_loader=loader,
        )


def test_dose_closure_requires_canonical_batch_channel_detector_axes():
    from ptycho_torch.workflows import components

    model, loader, dataset = _known_gauge_loader(
        patterns=30,
        channels=9,
    )
    dataset.fields["measured_intensity"] = dataset.fields[
        "measured_intensity"
    ][:, 0]

    with pytest.raises(ValueError, match=r"\(B, C, H, W\)"):
        components._initialize_rect_s1s2(
            model,
            mode="dose_closure",
            training_loader=loader,
        )


def test_dose_closure_rejects_zero_predicted_intensity_denominator():
    from ptycho_torch.workflows import components

    model, loader, dataset = _known_gauge_loader()
    dataset.fields["probe_training"].zero_()

    with pytest.raises(ValueError, match="predicted intensity sum.*positive"):
        components._initialize_rect_s1s2(
            model,
            mode="dose_closure",
            training_loader=loader,
        )


def test_dose_closure_rejects_nonfinite_predicted_intensity():
    from ptycho_torch.workflows import components

    model, loader, dataset = _known_gauge_loader()
    dataset.fields["probe_training"][0, 0, 0, 0, 0] = torch.complex(
        torch.tensor(float("nan")), torch.tensor(0.0)
    )

    with pytest.raises(ValueError, match="predicted intensity sum.*finite"):
        components._initialize_rect_s1s2(
            model,
            mode="dose_closure",
            training_loader=loader,
        )


def test_dose_closure_preserves_nested_train_eval_state():
    from ptycho_torch.workflows import components

    model, loader, _ = _known_gauge_loader()
    model.train()
    model.model.forward_model.rect_scaler.eval()
    dry_forward_training_states = []
    handle = model.model.forward_model.register_forward_pre_hook(
        lambda module, _inputs: dry_forward_training_states.append(module.training)
    )

    try:
        components._initialize_rect_s1s2(
            model,
            mode="dose_closure",
            training_loader=loader,
        )
    finally:
        handle.remove()

    assert dry_forward_training_states and not any(dry_forward_training_states)
    assert model.training is True
    assert model.model.forward_model.rect_scaler.training is False


_OMIT_INIT_OVERRIDE = object()


def _resolved_training_case(tmp_path, *, mode=_OMIT_INIT_OVERRIDE):
    from ptycho_torch.config_factory import create_training_payload
    from ptycho_torch.execution_request import ExecutionRequest

    detector_size = 64
    patterns = 16
    train_path = tmp_path / "train.npz"
    np.savez(
        train_path,
        diffraction=np.ones(
            (patterns, detector_size, detector_size), dtype=np.float32
        ),
        probeGuess=np.ones(
            (detector_size, detector_size), dtype=np.complex64
        ),
        objectGuess=np.ones(
            (2 * detector_size, 2 * detector_size), dtype=np.complex64
        ),
        xcoords=np.arange(patterns, dtype=np.float32),
        ycoords=np.arange(patterns, dtype=np.float32),
    )
    overrides = {
        "n_groups": patterns,
        "batch_size": 4,
        "gridsize": 1,
        "object_big": False,
        "architecture": "cnn",
    }
    if mode is not _OMIT_INIT_OVERRIDE:
        overrides["rect_s1s2_init"] = mode
    payload = create_training_payload(
        train_data_file=train_path,
        output_dir=tmp_path / "output",
        overrides=overrides,
        execution_config=ExecutionRequest(
            values={"accelerator": "cpu"},
            explicit_fields=frozenset({"accelerator"}),
        ),
        profile="ci",
    )
    return payload.tf_training_config, payload


def test_training_entry_initializes_before_fit_and_persists_same_summary_record(
    tmp_path, monkeypatch, caplog
):
    import lightning.pytorch as L
    from ptycho_torch.workflows import components

    config, payload = _resolved_training_case(tmp_path)
    _, loader, _ = _known_gauge_loader()
    monkeypatch.setattr(
        components,
        "_build_lightning_dataloaders",
        lambda *args, **kwargs: (loader, None),
    )
    record = _initialization_payload()
    events = []

    def fake_initialize(model, *, mode, training_loader, **kwargs):
        events.append(("initialize", mode, training_loader))
        return dict(record)

    def fake_fit(self, model, **kwargs):
        summary_callback = next(
            callback
            for callback in self.callbacks
            if type(callback).__name__ == "_TrainingSummaryCallback"
        )
        summary_callback.on_fit_start(self, model)
        events.append(("fit", model, kwargs))

    monkeypatch.setattr(components, "_initialize_rect_s1s2", fake_initialize)
    monkeypatch.setattr(L.Trainer, "fit", fake_fit)
    caplog.set_level("INFO", logger=components.__name__)

    results = components._train_with_lightning(
        train_container={
            "rms_input_scale": torch.tensor(1.0),
            "mean_measured_intensity": torch.tensor(1.0),
        },
        test_container=None,
        config=config,
        resolved_payload=payload,
    )

    assert events[0] == ("initialize", "dose_closure", loader)
    assert events[1][0] == "fit"
    assert results["rect_s1s2_initialization"] == record
    summary_path = tmp_path / "output" / "training_summary.json"
    assert results["training_summary_path"] == summary_path
    assert json.loads(summary_path.read_text(encoding="utf-8")) == record
    assert str(record) in caplog.text


def test_explicit_ones_training_resolution_does_not_forward_a_loader(tmp_path):
    from ptycho_torch.workflows import components

    _, payload = _resolved_training_case(tmp_path, mode="ones")

    class _ExplodingLoader:
        def __iter__(self):
            raise AssertionError("explicit ones must not consume the training loader")

    assert payload.pt_model_config.rect_s1s2_init == "ones"
    assert (
        components._rect_s1s2_training_loader(
            object(),
            _ExplodingLoader(),
            payload.pt_model_config.rect_s1s2_init,
        )
        is None
    )


def test_supervised_training_publishes_ones_record_without_rectangular_scaler(
    tmp_path,
    monkeypatch,
):
    import lightning.pytorch as L

    from ptycho_torch.config_factory import create_training_payload
    from ptycho_torch.execution_request import ExecutionRequest
    from ptycho_torch.workflows import components

    train_path = tmp_path / "train.npz"
    np.savez(
        train_path,
        diffraction=np.ones((1, 64, 64), dtype=np.float32),
        probeGuess=np.ones((64, 64), dtype=np.complex64),
        xcoords=np.zeros(1, dtype=np.float32),
        ycoords=np.zeros(1, dtype=np.float32),
    )
    payload = create_training_payload(
        train_data_file=train_path,
        output_dir=tmp_path / "output",
        overrides={
            "n_groups": 1,
            "batch_size": 1,
            "gridsize": 1,
            "architecture": "ffno",
            "model_type": "Supervised",
            "torch_loss_mode": "mae",
            "object_big": False,
        },
        execution_config=ExecutionRequest(
            values={"accelerator": "cpu"},
            explicit_fields=frozenset({"accelerator"}),
        ),
    )
    train_loader = [
        (
            {
                "label_amp": torch.ones((1, 1, 64, 64)),
                "label_phase": torch.zeros((1, 1, 64, 64)),
            },
        )
    ]
    monkeypatch.setattr(
        components,
        "_build_lightning_dataloaders",
        lambda *_args, **_kwargs: (train_loader, None),
    )

    def fake_fit(self, model, **_kwargs):
        summary_callback = next(
            callback
            for callback in self.callbacks
            if type(callback).__name__ == "_TrainingSummaryCallback"
        )
        summary_callback.on_fit_start(self, model)

    monkeypatch.setattr(L.Trainer, "fit", fake_fit)

    results = components._train_with_lightning(
        train_container={},
        test_container=None,
        config=payload.tf_training_config,
        resolved_payload=payload,
    )

    assert results["rect_s1s2_initialization"] == _initialization_payload("ones")
    summary_path = tmp_path / "output" / "training_summary.json"
    assert json.loads(summary_path.read_text(encoding="utf-8")) == (
        _initialization_payload("ones")
    )


def test_training_summary_publication_is_rank_zero_atomic_and_all_rank_barrier(
    tmp_path,
    monkeypatch,
):
    from ptycho_torch.workflows import components

    record = _initialization_payload("ones")
    summary_path = tmp_path / "training_summary.json"
    events = []
    atomic_write = components._write_training_summary_atomic

    def observed_write(path, value):
        events.append("write")
        atomic_write(path, value)

    class FakeStrategy:
        def __init__(self, rank):
            self.rank = rank

        def barrier(self, name):
            events.append(("barrier", self.rank, name, summary_path.exists()))

    monkeypatch.setattr(
        components,
        "_write_training_summary_atomic",
        observed_write,
    )

    components._publish_training_summary_and_barrier(
        SimpleNamespace(is_global_zero=True, strategy=FakeStrategy(0)),
        summary_path,
        record,
    )
    components._publish_training_summary_and_barrier(
        SimpleNamespace(is_global_zero=False, strategy=FakeStrategy(1)),
        summary_path,
        record,
    )

    assert json.loads(summary_path.read_text(encoding="utf-8")) == record
    assert events == [
        "write",
        ("barrier", 0, "rect_s1s2_training_summary", True),
        ("barrier", 1, "rect_s1s2_training_summary", True),
    ]
    assert list(tmp_path.glob(".training_summary.json.*.tmp")) == []
