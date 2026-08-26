"""Focused closed-form gauge initialization tests for rectangular CI physics."""

import inspect
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
    same factory (create_training_payload), canonical CI container, and workflow
    dataloader path that _train_with_lightning / run_torch_training use."""
    import tempfile
    from pathlib import Path

    import numpy as np

    from ptycho_torch import helper as hh
    from ptycho_torch.config_factory import create_training_payload
    from ptycho_torch.config_params import InferenceConfig
    from ptycho_torch.model import PtychoPINN_Lightning
    from ptycho_torch.workflows.components import (
        _build_lightning_dataloaders,
        attach_container_ci_fields,
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
            "training_groups": B,
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

    # High per-pixel counts reproduce the RCA's init-scale mismatch through the
    # canonical count-domain container path.
    count_amplitude_scale = hh.derive_intensity_scale_from_amplitudes(
        torch.as_tensor(amplitudes), 1e9
    )
    container = SimpleNamespace(
        X=torch.as_tensor(amplitudes),
        raw_grouped_diffraction=(
            count_amplitude_scale * torch.as_tensor(amplitudes)
        ).square(),
        probe=count_amplitude_scale * torch.as_tensor(probe),
        coords_relative=torch.zeros(B, 1, 2, 1),
    )
    attach_container_ci_fields(
        container,
        N=N,
    )

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


class _StringEqualityImpostor:
    def __eq__(self, other):
        return other == "ones"


@pytest.mark.parametrize(
    "mode",
    [
        _StringEqualityImpostor(),
        np.array(["ones"]),
        np.str_("ones"),
    ],
    ids=("equality-impostor", "one-element-array", "numpy-string"),
)
def test_initialization_mode_requires_exact_builtin_strings(mode):
    from ptycho_torch.rect_s1s2_initialization import (
        validate_rect_s1s2_initialization_mode,
    )

    with pytest.raises(
        ValueError,
        match=r"must be 'ones' or 'dose_closure'",
    ):
        validate_rect_s1s2_initialization_mode(mode)


def _initialization_payload(
    mode="dose_closure",
    *,
    gauge=3.25,
    schema_version="rect-s1s2-initialization-v2",
    sampled_patterns=None,
):
    if mode == "ones":
        return {
            "schema_version": schema_version,
            "mode": "ones",
            "solved_gauge": 1.0,
            "method": "unit_default_no_solve",
            "sampled_patterns": 0,
        }
    method = (
        "dose_closure_seeded_uniform_unit_object"
        if schema_version == "rect-s1s2-initialization-v2"
        else "dose_closure_unit_object"
    )
    return {
        "schema_version": schema_version,
        "mode": "dose_closure",
        "solved_gauge": gauge,
        "method": method,
        "sampled_patterns": 256 if sampled_patterns is None else sampled_patterns,
    }


def test_initialization_record_is_frozen_versioned_and_round_trips_strictly():
    from dataclasses import FrozenInstanceError

    record_type = _record_type()
    payload = _initialization_payload()

    record = record_type.from_mapping(payload)

    assert record.to_jsonable() == payload
    with pytest.raises(FrozenInstanceError):
        record.mode = "ones"


@pytest.mark.parametrize(
    "payload",
    [
        _initialization_payload(
            schema_version="rect-s1s2-initialization-v1",
            sampled_patterns=256,
        ),
        _initialization_payload(
            schema_version="rect-s1s2-initialization-v1",
            sampled_patterns=512,
        ),
        _initialization_payload(
            "ones",
            schema_version="rect-s1s2-initialization-v1",
        ),
        _initialization_payload(
            schema_version="rect-s1s2-initialization-v2",
        ),
        _initialization_payload(
            "ones",
            schema_version="rect-s1s2-initialization-v2",
        ),
    ],
    ids=("v1-dose-256", "v1-dose-512", "v1-ones", "v2-dose", "v2-ones"),
)
def test_initialization_record_round_trips_both_strict_schema_versions(payload):
    record_type = _record_type()

    record = record_type.from_mapping(payload)

    assert record.to_jsonable() == payload
    assert record_type.from_mapping(record).to_jsonable() == payload


def test_fresh_initialization_factories_emit_exact_v2_runtime_identity():
    from ptycho_torch.rect_s1s2_initialization import (
        RECT_S1S2_INITIALIZATION_SCHEMA,
        RECT_S1S2_INITIALIZATION_SCHEMA_V1,
        RECT_S1S2_INITIALIZATION_SCHEMA_V2,
    )

    record_type = _record_type()

    assert RECT_S1S2_INITIALIZATION_SCHEMA == RECT_S1S2_INITIALIZATION_SCHEMA_V2
    assert RECT_S1S2_INITIALIZATION_SCHEMA_V1 == "rect-s1s2-initialization-v1"
    assert RECT_S1S2_INITIALIZATION_SCHEMA_V2 == "rect-s1s2-initialization-v2"
    assert record_type.ones().to_jsonable() == _initialization_payload("ones")
    assert record_type.dose_closure(3.25).to_jsonable() == (
        _initialization_payload()
    )
    assert tuple(inspect.signature(record_type.dose_closure).parameters) == (
        "solved_gauge",
    )


def test_initialization_record_constructor_cannot_bypass_validation():
    record_type = _record_type()

    with pytest.raises(ValueError, match="sampled_patterns"):
        record_type(
            mode="dose_closure",
            solved_gauge=3.25,
            method="dose_closure_seeded_uniform_unit_object",
            sampled_patterns=1,
        )


@pytest.mark.parametrize("sampled_patterns", [0, 255, 257, 512])
def test_v2_dose_closure_requires_exactly_256_patterns(sampled_patterns):
    record_type = _record_type()
    payload = _initialization_payload(
        schema_version="rect-s1s2-initialization-v2",
        sampled_patterns=sampled_patterns,
    )

    with pytest.raises(ValueError, match="sampled_patterns"):
        record_type.from_mapping(payload)


@pytest.mark.parametrize(
    ("schema_version", "method"),
    [
        (
            "rect-s1s2-initialization-v1",
            "dose_closure_seeded_uniform_unit_object",
        ),
        ("rect-s1s2-initialization-v2", "dose_closure_unit_object"),
    ],
    ids=("v1-with-v2-method", "v2-with-v1-method"),
)
def test_initialization_record_rejects_cross_version_dose_methods(
    schema_version,
    method,
):
    record_type = _record_type()
    payload = _initialization_payload(schema_version=schema_version)
    payload["method"] = method

    with pytest.raises(ValueError, match="method"):
        record_type.from_mapping(payload)


@pytest.mark.parametrize(
    ("schema_version", "field", "value"),
    [
        ("rect-s1s2-initialization-v1", "solved_gauge", 2.0),
        ("rect-s1s2-initialization-v2", "solved_gauge", 2.0),
        ("rect-s1s2-initialization-v1", "sampled_patterns", 1),
        ("rect-s1s2-initialization-v2", "sampled_patterns", 1),
        (
            "rect-s1s2-initialization-v1",
            "method",
            "dose_closure_unit_object",
        ),
        (
            "rect-s1s2-initialization-v2",
            "method",
            "dose_closure_seeded_uniform_unit_object",
        ),
    ],
)
def test_ones_invariants_are_identical_and_strict_across_versions(
    schema_version,
    field,
    value,
):
    record_type = _record_type()
    payload = _initialization_payload("ones", schema_version=schema_version)
    payload[field] = value

    with pytest.raises((TypeError, ValueError), match=field):
        record_type.from_mapping(payload)


@pytest.mark.parametrize(
    "gauge",
    [float("nan"), float("inf"), float("-inf"), 0.0, -1.0],
)
def test_v2_record_rejects_nonfinite_or_nonpositive_gauges(gauge):
    record_type = _record_type()
    payload = _initialization_payload(
        gauge=gauge,
        schema_version="rect-s1s2-initialization-v2",
    )

    with pytest.raises(ValueError, match="solved_gauge"):
        record_type.from_mapping(payload)


@pytest.mark.parametrize("field", ["schema_version", "method"])
def test_record_schema_and_method_require_exact_builtin_strings(field):
    record_type = _record_type()
    payload = _initialization_payload(
        schema_version="rect-s1s2-initialization-v2",
    )
    payload[field] = np.str_(payload[field])

    with pytest.raises((TypeError, ValueError), match=field):
        record_type.from_mapping(payload)


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
        self.read_indices = []

    def __len__(self):
        return self.fields["images"].shape[0]

    def __getitem__(self, index):
        if isinstance(index, (list, tuple)):
            self.read_indices.extend(int(value) for value in index)
        elif isinstance(index, torch.Tensor) and index.ndim > 0:
            self.read_indices.extend(int(value) for value in index.tolist())
        else:
            self.read_indices.append(int(index))
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
            num_datasets=num_datasets,
            probe_mask=probe_mask,
            probe_mask_diameter=(
                float(detector_size - 2) if probe_mask else None
            ),
            probe_mask_sigma=0.0,
        )
        data_config = DataConfig(
            N=detector_size,
            gridsize=1 if channels == 1 else 3,
        )
        self.model = torch.nn.Module()
        self.model.forward_model = ForwardModel(model_config, data_config)


def _known_gauge_loader(
    *,
    patterns=300,
    batch_size=73,
    gauge=3.25,
    channels=1,
    varying_probe_normalization=True,
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
    if varying_probe_normalization:
        probe_normalization = torch.linspace(0.75, 1.25, patterns).view(
            patterns, 1, 1, 1, 1
        )
    else:
        probe_normalization = torch.ones(patterns, 1, 1, 1, 1)
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


def _known_gauge_named_loader(*, patterns=300, batch_size=73, gauge=2.75):
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
    count_intensity = amplitude.square()
    container = SimpleNamespace(
        X=count_intensity,
        raw_grouped_diffraction=count_intensity,
        coords_relative=torch.zeros(patterns, 1, 2, 1),
        probe=probe,
    )
    components.attach_container_ci_fields(
        container,
        N=8,
        probe_scale=1.0,
        probe_mask=False,
    )
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
    assert record["schema_version"] == "rect-s1s2-initialization-v2"
    assert record["mode"] == "dose_closure"
    assert record["solved_gauge"] == pytest.approx(3.25)
    assert record["method"] == "dose_closure_seeded_uniform_unit_object"
    assert record["sampled_patterns"] == 256
    assert torch.equal(scaler.s1.detach(), torch.full_like(scaler.s1, 3.25))
    assert torch.equal(scaler.s2.detach(), torch.full_like(scaler.s2, 3.25))


def test_representative_sample_defeats_blocked_prefix_ordering_bias():
    from ptycho_torch.rect_s1s2_sampling import build_dose_closure_sample_plan
    from ptycho_torch.workflows import components

    model, loader, dataset = _known_gauge_loader(
        patterns=1024,
        batch_size=37,
        gauge=1.0,
        varying_probe_normalization=False,
    )
    dataset.fields["measured_intensity"][256:] *= 16.0
    plan = build_dose_closure_sample_plan(dataset, channels=1)
    early = sum(flat_slot < 256 for flat_slot in plan.flat_slots)
    late = len(plan.flat_slots) - early

    record = components._initialize_rect_s1s2(
        model,
        mode="dose_closure",
        training_loader=loader,
    )

    prefix_gauge = 1.0
    full_population_gauge = 3.5
    assert (early, late) == (63, 193)
    assert record["solved_gauge"] == pytest.approx(
        3.508360550171547,
        rel=2e-6,
    )
    assert abs(record["solved_gauge"] - full_population_gauge) < abs(
        prefix_gauge - full_population_gauge
    )


@pytest.mark.parametrize(
    ("batch_size", "shuffle", "num_workers", "seed", "rank"),
    [
        (1, False, 0, 11, 0),
        (73, True, 0, 29, 3),
        (257, True, 2, 47, 7),
    ],
)
def test_dose_closure_is_independent_of_original_loader_and_rng_settings(
    batch_size,
    shuffle,
    num_workers,
    seed,
    rank,
):
    import random

    from ptycho_torch.workflows import components

    class TrackingDistributedSampler(torch.utils.data.DistributedSampler):
        def __iter__(self):
            self.iterations = getattr(self, "iterations", 0) + 1
            return super().__iter__()

    model, _, dataset = _known_gauge_loader(patterns=400, gauge=3.25)
    dataset.current_rank = rank
    generator = torch.Generator().manual_seed(seed)
    sampler = TrackingDistributedSampler(
        dataset,
        num_replicas=8,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        generator=generator,
    )
    generator_before = generator.get_state().clone()
    sampler_generator = getattr(loader.sampler, "generator", None)
    sampler_generator_before = (
        sampler_generator.get_state().clone()
        if sampler_generator is not None
        else None
    )
    random.seed(seed * 101)
    np.random.seed(seed * 103)
    torch.manual_seed(seed * 107)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()

    record = components._initialize_rect_s1s2(
        model,
        mode="dose_closure",
        training_loader=loader,
    )

    assert record["solved_gauge"] == pytest.approx(3.25, rel=2e-6)
    assert getattr(sampler, "iterations", 0) == 0
    assert torch.equal(generator.get_state(), generator_before)
    if sampler_generator_before is not None:
        assert torch.equal(sampler_generator.get_state(), sampler_generator_before)
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_before)


def test_dose_closure_bounds_reads_to_inspection_plus_selected_logical_rows():
    from ptycho_torch.rect_s1s2_sampling import build_dose_closure_sample_plan
    from ptycho_torch.workflows import components

    model, loader, dataset = _known_gauge_loader(patterns=1024, gauge=2.5)
    plan = build_dose_closure_sample_plan(dataset, channels=1)
    dataset.read_indices.clear()

    record = components._initialize_rect_s1s2(
        model,
        mode="dose_closure",
        training_loader=loader,
    )

    assert record["solved_gauge"] == pytest.approx(2.5, rel=2e-6)
    assert len(dataset.read_indices) <= 1 + len(plan.access_rows)
    assert sorted(dataset.read_indices[1:]) == sorted(
        row.logical_row for row in plan.access_rows
    )


def test_dose_closure_rejects_collation_that_reorders_selected_identities():
    from ptycho_torch.workflows import components

    model, _, dataset = _known_gauge_loader(patterns=400, gauge=2.5)

    def reversing_collation(samples):
        batch_size = len(samples)
        batch = torch.utils.data.default_collate(samples)

        def reverse(value):
            if isinstance(value, dict):
                return {name: reverse(item) for name, item in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(reverse(item) for item in value)
            if (
                isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] == batch_size
            ):
                return value.flip(0)
            return value

        return reverse(batch)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=23,
        shuffle=False,
        collate_fn=reversing_collation,
    )

    with pytest.raises(ValueError, match=r"reordered identity coverage"):
        components._initialize_rect_s1s2(
            model,
            mode="dose_closure",
            training_loader=loader,
        )


def test_dose_closure_counts_exact_detector_patterns_across_group_channels():
    """Exactly 256 immutable flat-slot masks contribute across grouped rows."""
    from ptycho_torch.rect_s1s2_sampling import build_dose_closure_sample_plan
    from ptycho_torch.workflows import components

    model, loader, dataset = _known_gauge_loader(
        patterns=30,
        channels=9,
        batch_size=7,
        gauge=2.25,
    )
    plan = build_dose_closure_sample_plan(dataset, channels=9)
    selected = set(plan.flat_slots)
    measured = dataset.fields["measured_intensity"].reshape(30 * 9, 8, 8)
    for flat_slot in range(30 * 9):
        if flat_slot not in selected:
            measured[flat_slot] *= 100.0

    record = components._initialize_rect_s1s2(
        model,
        mode="dose_closure",
        training_loader=loader,
    )

    assert record["solved_gauge"] == pytest.approx(2.25, rel=2e-6)
    assert record["sampled_patterns"] == 256


def test_dose_closure_uses_real_named_container_and_loader_path():
    from ptycho_torch.workflows import components

    model, loader = _known_gauge_named_loader(gauge=2.75)

    record = components._initialize_rect_s1s2(
        model,
        mode="dose_closure",
        training_loader=loader,
    )

    assert loader.dataset._ptycho_vectorized_batch is True
    assert record["solved_gauge"] == pytest.approx(2.75, rel=2e-6)
    assert record["sampled_patterns"] == 256


class _LoggingSubset(torch.utils.data.Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.requests = []

    def __getitem__(self, index):
        self.requests.append(index)
        return super().__getitem__(index)


def test_dose_closure_preserves_nested_subset_membership_and_multiplicity(
):
    from ptycho_torch.rect_s1s2_sampling import build_dose_closure_sample_plan
    from ptycho_torch.workflows import components

    model, _, base = _known_gauge_loader(patterns=400, gauge=3.25)
    inner = _LoggingSubset(base, list(range(400)))
    outer_indices = list(range(300))
    preliminary = build_dose_closure_sample_plan(
        torch.utils.data.Subset(inner, outer_indices),
        channels=1,
    )
    duplicate_logical_rows = tuple(
        row.logical_row
        for row in preliminary.access_rows
        if row.logical_row != 0
    )[:2]
    for logical_row in duplicate_logical_rows:
        outer_indices[logical_row] = 391
    nested = _LoggingSubset(inner, outer_indices)
    plan = build_dose_closure_sample_plan(nested, channels=1)
    loader = torch.utils.data.DataLoader(
        nested,
        batch_size=41,
        shuffle=True,
        generator=torch.Generator().manual_seed(314159),
    )
    base.read_indices.clear()
    nested.requests.clear()
    inner.requests.clear()

    record = components._initialize_rect_s1s2(
        model,
        mode="dose_closure",
        training_loader=loader,
    )

    assert record["solved_gauge"] == pytest.approx(3.25, rel=2e-6)
    duplicate_rows = [row for row in plan.access_rows if row.base_row == 391]
    assert {row.logical_row for row in duplicate_rows} == set(
        duplicate_logical_rows
    )
    assert base.read_indices.count(391) == len(duplicate_rows)


class _InconsistentChannelDataset(torch.utils.data.Dataset):
    def __init__(self, base, inconsistent_row):
        self.base = base
        self.inconsistent_row = inconsistent_row

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        fields, probe, scale = self.base[index]
        if int(index) != self.inconsistent_row:
            return fields, probe, scale
        fields = dict(fields)
        fields["images"] = fields["images"].expand(2, -1, -1).clone()
        fields["measured_intensity"] = fields["measured_intensity"].expand(
            2, -1, -1
        ).clone()
        return fields, probe, scale


def test_channel_inspection_does_not_scan_an_inconsistent_unselected_row():
    from ptycho_torch.rect_s1s2_sampling import build_dose_closure_sample_plan
    from ptycho_torch.workflows import components

    model, _, base = _known_gauge_loader(patterns=400, gauge=2.5)
    selected_rows = {
        row.logical_row
        for row in build_dose_closure_sample_plan(base, channels=1).access_rows
    }
    unselected = next(row for row in range(1, len(base)) if row not in selected_rows)
    dataset = _InconsistentChannelDataset(base, unselected)
    loader = torch.utils.data.DataLoader(dataset, batch_size=17, shuffle=True)

    record = components._initialize_rect_s1s2(
        model,
        mode="dose_closure",
        training_loader=loader,
    )

    assert record["solved_gauge"] == pytest.approx(2.5, rel=2e-6)
    assert unselected not in base.read_indices


def test_selected_row_channel_count_must_match_inspected_row_zero():
    from ptycho_torch.rect_s1s2_sampling import build_dose_closure_sample_plan
    from ptycho_torch.workflows import components

    model, _, base = _known_gauge_loader(patterns=400, gauge=2.5)
    selected = next(
        row.logical_row
        for row in build_dose_closure_sample_plan(base, channels=1).access_rows
        if row.logical_row != 0
    )
    dataset = _InconsistentChannelDataset(base, selected)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    with pytest.raises(ValueError, match=r"selected row.*channel count.*1"):
        components._initialize_rect_s1s2(
            model,
            mode="dose_closure",
            training_loader=loader,
        )


def test_dose_closure_uses_real_prebuilt_mmap_data_module_and_loader(
    tmp_path,
    monkeypatch,
):
    from ptycho_torch.config_params import TrainingConfig
    from ptycho_torch.dataloader import PtychoDataset, TensorDictDataLoader
    from ptycho_torch.rect_s1s2_sampling import (
        _base_row_for_logical,
        build_dose_closure_sample_plan,
    )
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
    plan = build_dose_closure_sample_plan(loader.dataset, channels=1)
    mmap_dataset = loader.dataset.dataset
    physical_reads = []
    original_getitem = PtychoDataset.__getitem__

    def tracked_getitem(self, index):
        if self is mmap_dataset:
            if isinstance(index, torch.Tensor):
                physical_reads.extend(int(value) for value in index.reshape(-1))
            elif isinstance(index, (list, tuple)):
                physical_reads.extend(int(value) for value in index)
            else:
                physical_reads.append(int(index))
        return original_getitem(self, index)

    monkeypatch.setattr(PtychoDataset, "__getitem__", tracked_getitem)
    record = components._initialize_rect_s1s2(
        model,
        mode="dose_closure",
        training_loader=loader,
    )

    assert record["solved_gauge"] == pytest.approx(gauge, rel=2e-6)
    assert record["sampled_patterns"] == 256
    assert len(physical_reads) <= 1 + len(plan.access_rows)
    assert physical_reads[0] == _base_row_for_logical(loader.dataset, 0)
    assert physical_reads[1:] == [row.base_row for row in plan.access_rows]


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


def test_runtime_entry_points_do_not_accept_sampling_policy_overrides():
    from ptycho_torch.workflows import components

    assert tuple(
        inspect.signature(components._initialize_rect_s1s2_unmanaged).parameters
    ) == ("model", "mode", "training_loader")
    assert tuple(inspect.signature(components._initialize_rect_s1s2).parameters) == (
        "model",
        "mode",
        "training_loader",
    )


def test_dose_closure_rejects_empty_indexable_dataset_clearly():
    from ptycho_torch.workflows import components

    model, _, dataset = _known_gauge_loader(patterns=300)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, []),
        batch_size=17,
    )

    with pytest.raises(ValueError, match=r"empty.*dataset"):
        components._initialize_rect_s1s2(
            model,
            mode="dose_closure",
            training_loader=loader,
        )


def test_dose_closure_requires_positive_inspected_channel_count():
    from ptycho_torch.workflows import components

    model, _, dataset = _known_gauge_loader(patterns=300)
    dataset.fields["images"] = dataset.fields["images"][:, :0]
    dataset.fields["measured_intensity"] = dataset.fields[
        "measured_intensity"
    ][:, :0]
    loader = torch.utils.data.DataLoader(dataset, batch_size=13)

    with pytest.raises(ValueError, match=r"positive.*channel count"):
        components._initialize_rect_s1s2(
            model,
            mode="dose_closure",
            training_loader=loader,
        )


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


def _first_selected_row_and_channel(dataset, channels=1):
    from ptycho_torch.rect_s1s2_sampling import build_dose_closure_sample_plan

    selected = build_dose_closure_sample_plan(dataset, channels=channels).access_rows[0]
    return selected.logical_row, selected.channels[0]


def test_dose_closure_rejects_nonfinite_observed_counts():
    from ptycho_torch.workflows import components

    model, loader, dataset = _known_gauge_loader()
    row, channel = _first_selected_row_and_channel(dataset)
    dataset.fields["measured_intensity"][row, channel, 0, 0] = torch.nan

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
    row, channel = _first_selected_row_and_channel(dataset)
    dataset.fields["measured_intensity"][row, channel, 0, 0] = -1.0

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
    row, _ = _first_selected_row_and_channel(dataset)
    dataset.fields["probe_training"][row, 0, 0, 0, 0] = torch.complex(
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
        "training_groups": patterns,
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
        profile="ci",
    )
    return payload.tf_training_config, payload


def test_training_entry_initializes_before_fit_and_persists_same_summary_record(
    tmp_path, monkeypatch, caplog
):
    from dataclasses import replace

    import lightning.pytorch as L
    from ptycho_torch.workflows import components

    _, payload = _resolved_training_case(tmp_path)
    payload = replace(
        payload,
        execution_config=replace(
            payload.execution_config,
            enable_checkpointing=False,
        ),
    )
    _, loader, _ = _known_gauge_loader()
    monkeypatch.setattr("ptycho_torch.workflows.dataloaders._build_lightning_dataloaders",
        lambda *args, **kwargs: (loader, None),
    )
    record = _initialization_payload()
    events = []

    def fake_initialize(model, *, mode, training_loader, **kwargs):
        events.append(("initialize", mode, training_loader))
        return dict(record)

    def fake_fit(self, model, **kwargs):
        callbacks = {
            type(callback).__name__: callback for callback in self.callbacks
        }
        callbacks["_TrainingSummaryCallback"].on_fit_start(self, model)
        selection_callback = callbacks["_FinalModelSelectionCallback"]
        selection_callback.on_fit_start(self, model)
        events.append(("fit", model, kwargs))
        selection_callback.on_train_end(self, model)

    monkeypatch.setattr("ptycho_torch.workflows.rect_s1s2._initialize_rect_s1s2", fake_initialize)
    monkeypatch.setattr(L.Trainer, "fit", fake_fit)
    caplog.set_level("INFO", logger=components.__name__)

    results = components._train_with_lightning(
        payload,
        train_container={
            "rms_input_scale": torch.tensor(1.0),
            "mean_measured_intensity": torch.tensor(1.0),
        },
        test_container=None,
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

    monkeypatch.setattr("ptycho_torch.workflows.rect_s1s2._write_training_summary_atomic",
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
