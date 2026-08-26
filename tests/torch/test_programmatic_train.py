"""Focused contracts for direct Torch training settings resolution."""

from __future__ import annotations

import hashlib
import importlib
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ptycho_torch.config_factory import resolve_training_payload
from ptycho_torch.config_resolution import TRAINING_INPUT_RULES


@pytest.fixture
def training_npz(tmp_path: Path) -> Path:
    path = tmp_path / "train.npz"
    np.savez(path, probeGuess=np.ones((64, 64), dtype=np.complex64))
    return path


def _resolve(training_npz: Path, tmp_path: Path, **settings):
    return resolve_training_payload(
        train_data_file=training_npz,
        output_dir=tmp_path / "out",
        overrides={"training_groups": 8, **settings},
        profile="ci",
    )


def test_candidate_pool_omission_uses_complete_acquisition(
    training_npz: Path,
    tmp_path: Path,
) -> None:
    payload = _resolve(training_npz, tmp_path, gridsize=1)

    assert payload.pt_data_config.n_raw_frames_selected is None


def test_neighbor_default_covers_gridsize_squared_minus_center(
    training_npz: Path,
    tmp_path: Path,
) -> None:
    payload = _resolve(training_npz, tmp_path, gridsize=3)

    assert payload.pt_data_config.neighbor_count >= 8
    assert payload.pt_data_config.n_raw_frames_selected is None


def test_candidate_pool_explicit_raw_cap_is_preserved(
    training_npz: Path,
    tmp_path: Path,
) -> None:
    payload = _resolve(
        training_npz,
        tmp_path,
        gridsize=1,
        n_raw_frames_selected=5,
    )

    assert payload.pt_data_config.n_raw_frames_selected == 5
    assert payload.tf_training_config.train_raw_selection == 5


@pytest.mark.parametrize("raw_cap", [0, -1, 1.5, True])
def test_candidate_pool_rejects_nonpositive_or_nonintegral_raw_cap(
    training_npz: Path,
    tmp_path: Path,
    raw_cap: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="n_raw_frames_selected must be an exact positive integer",
    ):
        _resolve(
            training_npz,
            tmp_path,
            gridsize=1,
            n_raw_frames_selected=raw_cap,
        )


@pytest.mark.parametrize("neighbor_count", [0, 1.5, True])
def test_neighbor_count_requires_an_exact_positive_integer(
    training_npz: Path,
    tmp_path: Path,
    neighbor_count: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="neighbor_count must be an exact positive integer",
    ):
        _resolve(
            training_npz,
            tmp_path,
            gridsize=1,
            neighbor_count=neighbor_count,
        )


def test_neighbor_count_below_group_requirement_fails_instead_of_overwriting(
    training_npz: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"neighbor_count=7.*at least 8.*gridsize=3",
    ):
        _resolve(training_npz, tmp_path, gridsize=3, neighbor_count=7)


def test_neighbor_count_stays_positive_for_gridsize_one_bridge(
    training_npz: Path,
    tmp_path: Path,
) -> None:
    payload = _resolve(training_npz, tmp_path, gridsize=1)

    assert payload.pt_data_config.neighbor_count > 0
    assert payload.tf_training_config.neighbor_count > 0


def test_configuration_table_matches_training_input_rules() -> None:
    guide = Path("docs/CONFIGURATION.md").read_text()
    start = "<!-- programmatic-torch-settings:start -->"
    end = "<!-- programmatic-torch-settings:end -->"
    assert guide.count(start) == guide.count(end) == 1
    section = guide.split(start, 1)[1].split(end, 1)[0]
    documented = dict(
        re.findall(r"^\| `([^`]+)` \| `([^`]+)` \|$", section, re.MULTILINE)
    )
    expected = {rule.canonical: rule.owner for rule in TRAINING_INPUT_RULES}

    assert documented == expected
    assert "Compatibility aliases (non-preferred)" in section
    assert "`model_type` → `mode`" in section
    assert "`max_epochs` → `epochs`" in section


def _public_train():
    return getattr(importlib.import_module("ptycho_torch.train"), "train")


def _raw_data(
    *,
    scale_contract_version=None,
    measurement_domain=None,
    marker=1.0,
):
    rows = 5
    return SimpleNamespace(
        xcoords=np.arange(rows, dtype=np.float64),
        ycoords=np.arange(rows, dtype=np.float64),
        xcoords_start=np.arange(rows, dtype=np.float64),
        ycoords_start=np.arange(rows, dtype=np.float64),
        diff3d=np.full((rows, 2, 2), marker, dtype=np.float32),
        probeGuess=np.full((2, 2), marker, dtype=np.complex64),
        scan_index=np.arange(rows, dtype=np.int64),
        object_index=np.zeros(rows, dtype=np.int64),
        objectGuess=None,
        Y=np.full((rows, 2, 2), marker, dtype=np.complex64),
        label=np.arange(rows, dtype=np.int64),
        probe_simulated=None,
        scale_contract_version=scale_contract_version,
        measurement_domain=measurement_domain,
        sample_indices=None,
        subsample_seed=None,
    )


def _payload(
    dataset: Path,
    output_dir: Path,
    *,
    target=("ci_intensity_v2", "count_intensity"),
    nphotons_source="declared_default",
    raw_cap=None,
    test_data_file=None,
):
    tf_config = SimpleNamespace(
        train_data_file=dataset,
        test_data_file=test_data_file,
        output_dir=output_dir,
    )
    return SimpleNamespace(
        tf_training_config=tf_config,
        pt_data_config=SimpleNamespace(
            scale_contract_version=target[0],
            measurement_domain=target[1],
            nphotons=100.0,
            n_raw_frames_selected=raw_cap,
            subsample_seed=7,
        ),
        pt_training_config=SimpleNamespace(test_data_file=test_data_file),
        overrides_applied={"nphotons_source": nphotons_source},
    )


def _patch_public_dependencies(
    monkeypatch,
    *,
    payload,
    raw_by_path,
    component=None,
):
    factory_calls = []

    def fake_factory(**kwargs):
        factory_calls.append(kwargs)
        return payload

    monkeypatch.setattr(
        "ptycho_torch.config_factory.create_training_payload",
        fake_factory,
    )
    monkeypatch.setattr(
        "ptycho.raw_data.RawData.from_file",
        staticmethod(lambda path: raw_by_path[Path(path)]),
    )
    if component is not None:
        monkeypatch.setattr(
            "ptycho_torch.workflows.legacy.train_cdi_model_torch",
            component,
        )
    return factory_calls


def _write_bundle(output_dir: Path, contents=b"bundle") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = output_dir / "wts.h5.zip"
    bundle.write_bytes(contents)
    return bundle


def test_programmatic_train_public_contract_reuses_factory_and_component(
    training_npz,
    tmp_path,
    monkeypatch,
) -> None:
    from ptycho_torch.execution_request import ExecutionRequest

    output_dir = tmp_path / "any-name"
    settings = {
        "architecture": "cnn",
        "training_groups": 4,
        "rect_s1s2_init": "ones",
        "epochs": 1,
    }
    payload = _payload(training_npz, output_dir)
    raw = _raw_data()
    component_calls = []

    def fake_component(*args, **kwargs):
        component_calls.append((args, kwargs))
        return {"bundle_path": _write_bundle(output_dir)}

    factory_calls = _patch_public_dependencies(
        monkeypatch,
        payload=payload,
        raw_by_path={training_npz: raw},
        component=fake_component,
    )
    execution = ExecutionRequest(
        values={"accelerator": "cpu"},
        explicit_fields=frozenset({"accelerator"}),
    )

    model = _public_train()(
        str(training_npz),
        str(output_dir),
        settings,
        execution_config=execution,
    )

    assert model == output_dir / "wts.h5.zip"
    assert model.stat().st_size > 0
    assert settings == {
        "architecture": "cnn",
        "training_groups": 4,
        "rect_s1s2_init": "ones",
        "epochs": 1,
    }
    assert factory_calls == [
        {
            "train_data_file": training_npz,
            "output_dir": output_dir,
            "overrides": settings,
            "profile": "ci",
            "execution_config": execution,
        }
    ]
    assert factory_calls[0]["overrides"] is not settings
    assert len(component_calls) == 1
    args, kwargs = component_calls[0]
    assert args == (raw, None, payload.tf_training_config)
    assert kwargs == {"resolved_payload": payload, "persist_bundle": True}


def test_programmatic_train_help_names_common_settings_and_guide() -> None:
    documentation = _public_train().__doc__

    assert documentation is not None
    for name in (
        "architecture",
        "training_groups",
        "gridsize",
        "nphotons",
        "epochs",
        "docs/CONFIGURATION.md",
    ):
        assert name in documentation


def test_programmatic_train_import_is_legacy_and_tensorflow_free() -> None:
    script = """
import ptycho_torch.train
import sys
assert "ptycho.params" not in sys.modules
assert "ptycho.config.legacy_state" not in sys.modules
assert not any(name == "tensorflow" or name.startswith("tensorflow.") for name in sys.modules)
"""

    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize("bad_settings", [None, [], "training_groups=4"])
def test_programmatic_train_rejects_nonmapping_settings_before_output(
    training_npz,
    tmp_path,
    bad_settings,
) -> None:
    output_dir = tmp_path / "not-created"

    with pytest.raises(TypeError, match="settings must be a mapping"):
        _public_train()(training_npz, output_dir, bad_settings)

    assert not output_dir.exists()


def test_programmatic_train_resolver_failure_precedes_output_creation(
    training_npz,
    tmp_path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "not-created"

    def reject(**_kwargs):
        raise ValueError("unknown training setting 'archtecture'")

    monkeypatch.setattr(
        "ptycho_torch.config_factory.create_training_payload",
        reject,
    )

    with pytest.raises(ValueError, match="archtecture"):
        _public_train()(
            training_npz,
            output_dir,
            {"training_groups": 4, "archtecture": "cnn"},
        )

    assert not output_dir.exists()


@pytest.mark.parametrize("contents", [None, b""])
def test_programmatic_train_requires_returned_nonempty_bundle(
    training_npz,
    tmp_path,
    monkeypatch,
    contents,
) -> None:
    output_dir = tmp_path / "out"
    payload = _payload(training_npz, output_dir)

    def fake_component(*_args, **_kwargs):
        bundle = output_dir / "wts.h5.zip"
        if contents is not None:
            _write_bundle(output_dir, contents)
        return {"bundle_path": bundle}

    _patch_public_dependencies(
        monkeypatch,
        payload=payload,
        raw_by_path={training_npz: _raw_data()},
        component=fake_component,
    )

    with pytest.raises(RuntimeError, match="nonempty training bundle"):
        _public_train()(training_npz, output_dir, {"training_groups": 4})


def test_programmatic_train_orders_full_decode_scale_cap_group_and_lightning(
    training_npz,
    tmp_path,
    monkeypatch,
) -> None:
    from ptycho_torch.workflows import legacy, lightning_service

    output_dir = tmp_path / "out"
    payload = _payload(
        training_npz,
        output_dir,
        nphotons_source="explicit",
        raw_cap=3,
    )
    raw = _raw_data(marker=2.0)
    events = []
    monkeypatch.setattr(
        "ptycho.raw_data.RawData.from_file",
        staticmethod(lambda _path: events.append("decode") or raw),
    )
    monkeypatch.setattr(
        "ptycho_torch.config_factory.create_training_payload",
        lambda **_kwargs: payload,
    )

    def fake_scale(amplitude, probe, nphotons, probe_simulated=None):
        events.append(("scale", len(amplitude), nphotons))
        return amplitude + 10, probe + 10, probe_simulated

    monkeypatch.setattr(
        "ptycho_torch.scaling_contract.rescale_amplitude_to_nphotons",
        fake_scale,
    )

    def fake_select(source, count, *, seed=None, rng=None):
        del rng
        events.append(("cap", len(source.xcoords), count, seed))
        return SimpleNamespace(
            source_indices=np.array([0, 2, 4]),
            seed=seed,
        )

    monkeypatch.setattr("ptycho.acquisition.select_acquisition", fake_select)
    container = object()

    def fake_container(data, config):
        events.append(("group", len(data.xcoords), config))
        return container

    monkeypatch.setattr(legacy.containers, "create_torch_data_container", fake_container)

    def fake_lightning(resolved, train_container, test_container, **kwargs):
        events.append(("lightning", resolved, train_container, test_container, kwargs))
        return {"bundle_path": _write_bundle(output_dir)}

    monkeypatch.setattr(lightning_service, "_train_with_lightning", fake_lightning)

    model = _public_train()(
        training_npz,
        output_dir,
        {"training_groups": 4, "nphotons": 100.0},
    )

    assert model == output_dir / "wts.h5.zip"
    assert [event[0] if isinstance(event, tuple) else event for event in events] == [
        "decode",
        "scale",
        "cap",
        "group",
        "lightning",
    ]
    assert events[1] == ("scale", 5, 100.0)
    assert events[2] == ("cap", 5, 3, 7)
    assert raw.sample_indices.tolist() == [0, 2, 4]
    assert raw.subsample_seed == 7
    assert raw.xcoords.tolist() == [0.0, 2.0, 4.0]
    assert np.all(raw.diff3d == 12)


@pytest.mark.parametrize(
    (
        "source",
        "target",
        "nphotons_source",
        "scales",
        "digest",
        "error",
    ),
    [
        (("ci_intensity_v2", "count_intensity"), ("ci_intensity_v2", "count_intensity"), "explicit", False, False, None),
        (("legacy_v1", "normalized_amplitude"), ("ci_intensity_v2", "count_intensity"), "declared_default", True, False, None),
        ((None, None), ("ci_intensity_v2", "count_intensity"), "explicit", True, True, None),
        ((None, None), ("ci_intensity_v2", "count_intensity"), "declared_default", False, False, None),
        (("ci_intensity_v2", "count_intensity"), ("legacy_v1", "normalized_amplitude"), "declared_default", False, False, "count-intensity source"),
        (("legacy_v1", "normalized_amplitude"), ("legacy_v1", "normalized_amplitude"), "explicit", False, False, None),
        ((None, None), ("legacy_v1", "normalized_amplitude"), "explicit", False, False, None),
    ],
)
def test_programmatic_train_applies_source_rule(
    training_npz,
    tmp_path,
    monkeypatch,
    source,
    target,
    nphotons_source,
    scales,
    digest,
    error,
) -> None:
    output_dir = tmp_path / "out"
    payload = _payload(
        training_npz,
        output_dir,
        target=target,
        nphotons_source=nphotons_source,
    )
    raw = _raw_data(
        scale_contract_version=source[0],
        measurement_domain=source[1],
    )
    scaling_calls = []
    component_calls = []

    def fake_scale(amplitude, probe, nphotons, probe_simulated=None):
        scaling_calls.append((amplitude, probe, nphotons, probe_simulated))
        return amplitude + 1, probe + 1, probe_simulated

    monkeypatch.setattr(
        "ptycho_torch.scaling_contract.rescale_amplitude_to_nphotons",
        fake_scale,
    )

    def fake_component(*args, **kwargs):
        component_calls.append((args, kwargs))
        return {"bundle_path": _write_bundle(output_dir)}

    _patch_public_dependencies(
        monkeypatch,
        payload=payload,
        raw_by_path={training_npz: raw},
        component=fake_component,
    )

    if error:
        with pytest.raises(ValueError, match=error):
            _public_train()(training_npz, output_dir, {"training_groups": 4})
        assert not component_calls
        assert not output_dir.exists()
        return

    _public_train()(training_npz, output_dir, {"training_groups": 4})

    assert bool(scaling_calls) is scales
    assert len(component_calls) == 1
    kwargs = component_calls[0][1]
    if digest:
        assert kwargs["rescaled_source_sha256"] == hashlib.sha256(
            training_npz.read_bytes()
        ).hexdigest()
    else:
        assert "rescaled_source_sha256" not in kwargs
    if scales:
        assert (raw.scale_contract_version, raw.measurement_domain) == target
    else:
        assert (raw.scale_contract_version, raw.measurement_domain) == source


@pytest.mark.parametrize(
    ("validation_source", "nphotons_source", "expected_scales"),
    [
        (("legacy_v1", "normalized_amplitude"), "declared_default", 1),
        ((None, None), "explicit", 1),
        ((None, None), "declared_default", 0),
        (("ci_intensity_v2", "count_intensity"), "explicit", 0),
    ],
)
def test_programmatic_train_applies_same_source_rule_to_validation(
    training_npz,
    tmp_path,
    monkeypatch,
    validation_source,
    nphotons_source,
    expected_scales,
) -> None:
    validation_npz = tmp_path / "validation.npz"
    validation_npz.touch()
    output_dir = tmp_path / "out"
    payload = _payload(
        training_npz,
        output_dir,
        nphotons_source=nphotons_source,
        test_data_file=validation_npz,
    )
    train_raw = _raw_data(
        scale_contract_version="ci_intensity_v2",
        measurement_domain="count_intensity",
    )
    validation_raw = _raw_data(
        scale_contract_version=validation_source[0],
        measurement_domain=validation_source[1],
        marker=2.0,
    )
    scaled = []

    def fake_scale(amplitude, probe, nphotons, probe_simulated=None):
        scaled.append(amplitude)
        return amplitude + 1, probe + 1, probe_simulated

    monkeypatch.setattr(
        "ptycho_torch.scaling_contract.rescale_amplitude_to_nphotons",
        fake_scale,
    )
    component_calls = []

    def fake_component(*args, **kwargs):
        component_calls.append((args, kwargs))
        return {"bundle_path": _write_bundle(output_dir)}

    _patch_public_dependencies(
        monkeypatch,
        payload=payload,
        raw_by_path={training_npz: train_raw, validation_npz: validation_raw},
        component=fake_component,
    )

    _public_train()(training_npz, output_dir, {"training_groups": 4})

    assert len(scaled) == expected_scales
    assert component_calls[0][0][1] is validation_raw
    assert "rescaled_source_sha256" not in component_calls[0][1]


def test_programmatic_train_keeps_training_digest_when_validation_also_converts(
    training_npz,
    tmp_path,
    monkeypatch,
) -> None:
    validation_npz = tmp_path / "validation.npz"
    validation_npz.touch()
    output_dir = tmp_path / "out"
    payload = _payload(
        training_npz,
        output_dir,
        nphotons_source="explicit",
        test_data_file=validation_npz,
    )
    train_raw = _raw_data()
    validation_raw = _raw_data(
        scale_contract_version="legacy_v1",
        measurement_domain="normalized_amplitude",
    )
    monkeypatch.setattr(
        "ptycho_torch.scaling_contract.rescale_amplitude_to_nphotons",
        lambda amplitude, probe, _nphotons, probe_simulated=None: (
            amplitude,
            probe,
            probe_simulated,
        ),
    )
    component_calls = []

    def fake_component(*args, **kwargs):
        component_calls.append((args, kwargs))
        return {"bundle_path": _write_bundle(output_dir)}

    _patch_public_dependencies(
        monkeypatch,
        payload=payload,
        raw_by_path={training_npz: train_raw, validation_npz: validation_raw},
        component=fake_component,
    )

    _public_train()(training_npz, output_dir, {"training_groups": 4})

    assert component_calls[0][1]["rescaled_source_sha256"] == hashlib.sha256(
        training_npz.read_bytes()
    ).hexdigest()


@pytest.mark.integration
@pytest.mark.slow
def test_run1084_train_reconstruct_smoke(tmp_path: Path) -> None:
    from ptycho_torch.inference import reconstruct
    from ptycho_torch.train import train

    dataset = Path("datasets/Run1084_recon3_postPC_shrunk_3.npz")
    model = train(
        dataset,
        tmp_path / "run1084_cnn",
        {
            "architecture": "cnn",
            "training_groups": 256,
            "nphotons": 1e9,
            "epochs": 1,
        },
    )
    result = reconstruct(model, dataset)

    assert model.is_file() and model.stat().st_size > 0
    assert result.amplitude.size and np.isfinite(result.amplitude).all()
    assert result.phase.size and np.isfinite(result.phase).all()
