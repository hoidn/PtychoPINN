"""Contract tests for count-calibrated synthetic study datasets."""

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIES_DIR = REPO_ROOT / "scripts" / "studies"
sys.path.insert(0, str(STUDIES_DIR))

import make_synthetic_truth_datasets as synthetic  # noqa: E402
import flux_sweep_eval as flux_eval  # noqa: E402
import make_dose_ladder_datasets as dose_ladder  # noqa: E402
import make_lines_datasets as lines  # noqa: E402


BUILDER_PATHS = (
    STUDIES_DIR / "make_synthetic_truth_datasets.py",
    STUDIES_DIR / "make_lines_datasets.py",
    STUDIES_DIR / "make_flux_sweep.py",
    STUDIES_DIR / "make_dose_ladder_datasets.py",
    STUDIES_DIR / "make_gridgeom_dataset.py",
    STUDIES_DIR / "make_weakphase_test.py",
)

PORTABLE_STUDY_PATHS = BUILDER_PATHS + (
    STUDIES_DIR / "flux_sweep_eval.py",
    STUDIES_DIR / "diagnose_placement.py",
)


def _known_inputs(n_patterns: int = 128, n: int = 8):
    yy, xx = np.mgrid[:n, :n]
    base = (0.7 + 0.2 * np.sin(xx / 2.0)) * np.exp(
        1j * (0.15 * np.cos(yy / 3.0))
    )
    phases = np.exp(1j * np.linspace(-0.2, 0.2, n_patterns))
    patches = (phases[:, None, None] * base[None]).astype(np.complex64)
    probe = (
        np.exp(-((xx - 3.5) ** 2 + (yy - 3.5) ** 2) / 7.0)
        * np.exp(1j * 0.03 * xx)
    ).astype(np.complex64)
    return patches, probe


def test_ci_measurement_forward_matches_calibrated_probe_and_requested_dose():
    patches, raw_probe = _known_inputs()
    target_mean_count = 37.0

    generated = synthetic.generate_ci_count_measurement(
        patches,
        raw_probe,
        target_mean_count=target_mean_count,
        rng=np.random.default_rng(19),
    )

    reproduced = synthetic.noiseless_detector_intensity(
        patches,
        generated.probe_physical,
    )
    np.testing.assert_allclose(
        generated.expected_count_intensity,
        reproduced,
        rtol=2e-6,
        atol=1e-7,
    )
    assert generated.expected_count_intensity.mean() == pytest.approx(
        target_mean_count,
        rel=2e-6,
    )
    assert generated.counts.mean() == pytest.approx(target_mean_count, rel=0.05)
    assert generated.dose_amplitude_scale == pytest.approx(
        np.sqrt(generated.dose_intensity_scale)
    )
    assert not np.array_equal(generated.probe_physical, raw_probe)


def test_ci_dataset_payload_reproduces_mean_from_stored_object_and_probe():
    n = 8
    yy, xx = np.mgrid[:32, :32]
    obj = (
        (0.75 + 0.2 * np.sin(xx / 4.0) * np.cos(yy / 5.0))
        * np.exp(1j * 0.1 * np.sin(yy / 6.0))
    ).astype(np.complex64)
    _, raw_probe = _known_inputs(n_patterns=1, n=n)
    xcoords = np.resize(np.array([8.0, 12.0, 16.0, 20.0]), 128)
    ycoords = np.resize(np.array([9.0, 13.0, 17.0, 19.0]), 128)

    generated = synthetic.generate_ci_count_dataset(
        obj,
        raw_probe,
        xcoords,
        ycoords,
        N=n,
        target_mean_count=23.0,
        poisson_seed=71,
    )
    payload = generated.payload
    stored_patches = synthetic.extract_object_patches(
        payload["objectGuess"],
        payload["xcoords"],
        payload["ycoords"],
        n,
    )
    reproduced = synthetic.noiseless_detector_intensity(
        stored_patches,
        payload["probeGuess"],
    )

    np.testing.assert_allclose(
        reproduced,
        generated.expected_count_intensity,
        rtol=2e-6,
        atol=1e-7,
    )
    assert reproduced.mean() == pytest.approx(23.0, rel=2e-6)
    assert payload["diff3d"].dtype == np.uint16
    np.testing.assert_array_equal(
        payload["ground_truth_patches"],
        stored_patches[..., None],
    )

    metadata = json.loads(payload["_metadata"].item())
    assert metadata["scale_contract_version"] == "ci_intensity_v2"
    assert metadata["measurement_domain"] == "count_intensity"
    assert metadata["probe_gauge"] == "physical_calibrated"
    assert metadata["probe_calibration"]["status"] == "calibrated"
    assert metadata["probe_calibration"]["dose_amplitude_scale"] == pytest.approx(
        generated.dose_amplitude_scale
    )
    assert metadata["object_units"] == "absolute"


def test_ci_poisson_generation_is_seed_reproducible():
    patches, raw_probe = _known_inputs(n_patterns=16)

    first = synthetic.generate_ci_count_measurement(
        patches, raw_probe, target_mean_count=11.0, rng=np.random.default_rng(5)
    )
    second = synthetic.generate_ci_count_measurement(
        patches, raw_probe, target_mean_count=11.0, rng=np.random.default_rng(5)
    )
    different = synthetic.generate_ci_count_measurement(
        patches, raw_probe, target_mean_count=11.0, rng=np.random.default_rng(6)
    )

    np.testing.assert_array_equal(first.counts, second.counts)
    np.testing.assert_array_equal(
        first.expected_count_intensity,
        second.expected_count_intensity,
    )
    assert not np.array_equal(first.counts, different.counts)


def test_full_dead_leaves_dataset_generation_is_seed_reproducible():
    n = 8
    obj_res = 24
    _, raw_probe = _known_inputs(n_patterns=1, n=n)
    xcoords = np.resize(np.array([8.0, 12.0, 16.0]), 18)
    ycoords = np.resize(np.array([9.0, 13.0, 15.0]), 18)
    dead_leaves_arg = {
        "max_iters": 30,
        "r_min_frac": 0.08,
        "r_max_frac": 0.25,
        "r_sigma": 2,
    }

    def build(seed: int):
        return synthetic.generate_seeded_dead_leaves_count_dataset(
            raw_probe,
            xcoords,
            ycoords,
            N=n,
            obj_res=obj_res,
            target_mean_count=13.0,
            seed=seed,
            dead_leaves_arg=dead_leaves_arg,
        )

    first = build(123)
    second = build(123)
    different = build(124)

    for key in ("objectGuess", "probeGuess", "diff3d"):
        np.testing.assert_array_equal(first.payload[key], second.payload[key])
    np.testing.assert_array_equal(
        first.expected_count_intensity,
        second.expected_count_intensity,
    )
    assert not np.array_equal(
        first.payload["objectGuess"], different.payload["objectGuess"]
    )
    assert not np.array_equal(first.payload["diff3d"], different.payload["diff3d"])


@pytest.mark.parametrize(
    ("builder", "probe_transform", "expected_shape"),
    [
        pytest.param("dead_leaves", lambda probe: probe, (8, 8), id="two-dimensional"),
        pytest.param(
            "lines",
            lambda probe: np.stack([probe, 0.35j * probe]),
            (2, 8, 8),
            id="multimode",
        ),
        pytest.param(
            "dose_ladder",
            lambda probe: probe[..., None],
            (1, 8, 8),
            id="legacy-trailing-singleton",
        ),
    ],
)
def test_count_builder_executes_with_supported_probe_layouts(
    builder, probe_transform, expected_shape, tmp_path, monkeypatch
):
    n = 8
    yy, xx = np.mgrid[:24, :24]
    obj = (
        (0.65 + 0.3 * np.sin(xx / 2.5) * np.cos(yy / 3.5))
        * np.exp(1j * 0.2 * np.sin((xx + yy) / 4.0))
    ).astype(np.complex64)
    _, raw_probe = _known_inputs(n_patterns=1, n=n)
    probe = probe_transform(raw_probe)
    spec = {
        "obj_res": 24,
        "train": {"n": 18, "seed": 7, "jitter": 0.2},
    }
    xcoords, ycoords = synthetic.scan_positions(24, n, 18, 7, 0.2)
    monkeypatch.setattr(synthetic, "DS_DIR", tmp_path)

    if builder == "dead_leaves":
        info = synthetic.build_one(n, "train", spec, obj, probe)
    elif builder == "lines":
        info = lines.build_one(n, "train", spec, obj, probe)
    else:
        source = tmp_path / "source.npz"
        np.savez(source, xcoords=xcoords, ycoords=ycoords)
        monkeypatch.setattr(dose_ladder, "N", n)
        info = dose_ladder.build_split("train", source, 17.0, obj, probe)

    with np.load(info["path"], allow_pickle=True) as payload:
        assert payload["probeGuess"].shape == expected_shape
        assert payload["diff3d"].shape == (18, n, n)


@pytest.mark.parametrize(
    "shape",
    [
        (8,),
        (8, 7),
        (8, 8, 2),
        (0, 8, 8),
        (2, 8, 7),
        (1, 1, 8, 8),
    ],
)
def test_count_builder_rejects_ambiguous_invalid_probe_layouts(shape):
    probe = np.ones(shape, dtype=np.complex64)

    with pytest.raises(ValueError, match=r"\(N,N\).*\(P,N,N\)"):
        synthetic.canonicalize_probe_modes(probe, N=8)


def test_flux_evaluator_selects_ci_named_physical_fields():
    fields = {
        "measured_intensity": object(),
        "probe_physical": object(),
        "rms_input_scale": object(),
        "probe_normalization": torch.tensor([[[[[2.0]]]], [[[[4.0]]]]]),
        "images": object(),
        "rms_scaling_constant": object(),
    }

    measured, probe, rms_scale = flux_eval.ci_forward_fields(fields)

    assert measured is fields["measured_intensity"]
    assert probe is fields["probe_physical"]
    assert rms_scale is fields["rms_input_scale"]
    torch.testing.assert_close(
        flux_eval.ci_output_scale(fields),
        torch.tensor([0.5, 0.25]).reshape(2, 1, 1, 1),
    )


def test_flux_evaluator_expects_dose_invariant_scale_with_calibrated_probe():
    patches, raw_probe = _known_inputs(n_patterns=24)
    low = synthetic.generate_ci_count_measurement(
        patches,
        raw_probe,
        target_mean_count=4.0,
        rng=np.random.default_rng(100),
    )
    high = synthetic.generate_ci_count_measurement(
        patches,
        raw_probe,
        target_mean_count=100.0,
        rng=np.random.default_rng(101),
    )
    intensity_ratio = (
        high.expected_count_intensity.mean() / low.expected_count_intensity.mean()
    )
    probe_amplitude_ratio = (
        np.linalg.norm(high.probe_physical) / np.linalg.norm(low.probe_physical)
    )

    calibrated = flux_eval.expected_object_scale_ratio(
        intensity_ratio,
        probe_amplitude_ratio=probe_amplitude_ratio,
    )
    legacy_unchanged_probe = flux_eval.expected_object_scale_ratio(
        intensity_ratio,
        probe_amplitude_ratio=1.0,
    )

    assert calibrated == pytest.approx(1.0, rel=2e-6)
    assert legacy_unchanged_probe == pytest.approx(5.0, rel=2e-6)


def test_legacy_amplitude_rescaler_preserves_historical_arithmetic():
    amplitude = np.array([[[0.5, 1.0], [1.5, 2.0]]], dtype=np.float32)
    target = 41.0
    intensity = amplitude.astype(np.float64) ** 2
    expected = np.round(intensity * (target / intensity.mean())).astype(np.uint16)

    actual = synthetic.legacy_rescale_normalized_amplitude_to_counts(
        amplitude,
        target_mean_count=target,
    )

    np.testing.assert_array_equal(actual, expected)


def _called_function_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def _top_level_import_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def test_object_module_keeps_generator_dependencies_out_of_module_scope():
    objects_path = REPO_ROOT / "ptycho_torch" / "datagen" / "objects.py"

    assert _top_level_import_names(objects_path).isdisjoint(
        {"noise", "perlin_noise", "cv2"}
    )


def test_cpu_ci_declares_dead_leaves_rasterization_dependency():
    requirements = (REPO_ROOT / "requirements-ci.txt").read_text().splitlines()

    assert "opencv-python-headless" in requirements


@pytest.mark.parametrize("builder_path", BUILDER_PATHS, ids=lambda path: path.stem)
def test_all_count_builders_use_fresh_ci_poisson_generation(builder_path: Path):
    called = _called_function_names(builder_path)
    assert "generate_ci_count_dataset" in called
    assert "to_counts" not in called
    assert "simulate" not in called


@pytest.mark.parametrize(
    "builder_path",
    PORTABLE_STUDY_PATHS,
    ids=lambda path: path.stem,
)
def test_count_builders_derive_repo_root_from_their_own_location(
    builder_path: Path,
):
    source = builder_path.read_text()

    assert "/home/ollie/Documents/PtychoPINN" not in source
    assert "Path(__file__).resolve().parents[2]" in source
