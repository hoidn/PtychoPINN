"""Tests for D0 parity logger CLI and helper functions.

Test reference: plans/active/seed/reports/2026-01-22T042640Z/d0_parity_logger_plan.md
CLI module: scripts/tools/d0_parity_logger.py
"""
import json
from pathlib import Path

import dill
import numpy as np
import pytest

from scripts.tools.d0_parity_logger import (
    convert_for_json,
    main,
    sha256_file,
    summarize_array,
    summarize_grouped,
    summarize_probe,
)


class TestSummarizeArray:
    """Unit tests for summarize_array helper."""

    def test_basic_stats(self):
        """Verify basic statistics computation."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = summarize_array(arr)

        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["mean"] == 3.0
        assert stats["median"] == 3.0
        assert stats["count"] == 5

    def test_percentiles(self):
        """Verify percentile computation."""
        arr = np.arange(100, dtype=np.float64)
        stats = summarize_array(arr)

        assert stats["p01"] == pytest.approx(0.99, rel=0.1)
        assert stats["p10"] == pytest.approx(9.9, rel=0.1)
        assert stats["p90"] == pytest.approx(89.1, rel=0.1)
        assert stats["p99"] == pytest.approx(98.01, rel=0.1)

    def test_nonzero_fraction(self):
        """Verify nonzero fraction calculation."""
        arr = np.array([0.0, 1.0, 2.0, 0.0, 3.0])
        stats = summarize_array(arr)

        assert stats["nonzero_fraction"] == 0.6

    def test_complex_array(self):
        """Verify complex arrays are handled via magnitude."""
        arr = np.array([1 + 1j, 2 + 2j, 3 + 3j])
        stats = summarize_array(arr)

        # Magnitude of 1+1j is sqrt(2) ~ 1.414
        assert stats["min"] == pytest.approx(np.sqrt(2), rel=1e-6)
        assert stats["max"] == pytest.approx(3 * np.sqrt(2), rel=1e-6)

    def test_empty_array(self):
        """Verify empty arrays return zero stats."""
        arr = np.array([])
        stats = summarize_array(arr)

        assert stats["min"] == 0.0
        assert stats["max"] == 0.0
        assert stats["count"] == 0


class TestSummarizeGrouped:
    """Unit tests for summarize_grouped helper."""

    def test_basic_grouping(self):
        """Verify grouping by scan_index works correctly."""
        # 4 patterns, 2x2 pixels, grouped into 2 scans
        diff3d = np.array([
            [[1, 1], [1, 1]],  # pattern 0, scan 0, mean=1
            [[2, 2], [2, 2]],  # pattern 1, scan 0, mean=2
            [[3, 3], [3, 3]],  # pattern 2, scan 1, mean=3
            [[4, 4], [4, 4]],  # pattern 3, scan 1, mean=4
        ], dtype=np.float64)
        scan_index = np.array([0, 0, 1, 1])

        stats = summarize_grouped(diff3d, scan_index)

        # Scan 0: mean of [1,2] = 1.5
        # Scan 1: mean of [3,4] = 3.5
        assert stats["n_unique_scans"] == 2
        assert stats["n_patterns"] == 4
        assert stats["min"] == pytest.approx(1.5, rel=1e-6)
        assert stats["max"] == pytest.approx(3.5, rel=1e-6)
        assert stats["mean"] == pytest.approx(2.5, rel=1e-6)

    def test_single_scan(self):
        """Verify single scan group works."""
        diff3d = np.array([
            [[1, 2], [3, 4]],  # mean = 2.5
            [[5, 6], [7, 8]],  # mean = 6.5
        ], dtype=np.float64)
        scan_index = np.array([0, 0])

        stats = summarize_grouped(diff3d, scan_index)

        assert stats["n_unique_scans"] == 1
        # Per-scan mean = (2.5 + 6.5) / 2 = 4.5
        assert stats["mean"] == pytest.approx(4.5, rel=1e-6)


class TestSummarizeProbe:
    """Unit tests for summarize_probe helper."""

    def test_probe_stats(self):
        """Verify amplitude and phase stats are computed."""
        # Simple complex probe
        probe = np.array([[1 + 0j, 0 + 1j], [1 + 1j, 2 + 0j]])
        stats = summarize_probe(probe)

        assert "amplitude" in stats
        assert "phase" in stats
        assert stats["l2_norm"] > 0
        assert stats["shape"] == [2, 2]
        assert "complex" in stats["dtype"]

        # Check amplitude stats
        # Amplitudes: [1, 1, sqrt(2), 2]
        assert stats["amplitude"]["min"] == pytest.approx(1.0, rel=1e-6)
        assert stats["amplitude"]["max"] == pytest.approx(2.0, rel=1e-6)


class TestConvertForJson:
    """Unit tests for JSON serialization helper."""

    def test_numpy_scalar(self):
        """Verify numpy scalars are converted to Python floats."""
        val = np.float64(3.14)
        result = convert_for_json(val)
        assert isinstance(result, float)
        assert result == pytest.approx(3.14)

    def test_numpy_bool(self):
        """Verify numpy booleans are converted to Python bools."""
        val = np.bool_(True)
        result = convert_for_json(val)
        assert isinstance(result, bool)
        assert result is True

    def test_tuple(self):
        """Verify tuples are converted to lists."""
        val = (1, 2, np.float32(3.0))
        result = convert_for_json(val)
        assert isinstance(result, list)
        assert result == [1, 2, 3.0]

    def test_nested_dict(self):
        """Verify nested dicts are recursively converted."""
        val = {"a": np.int64(1), "b": {"c": np.float32(2.5)}}
        result = convert_for_json(val)
        assert result == {"a": 1.0, "b": {"c": 2.5}}


def _create_npz_dataset(dataset_dir: Path, filename: str, scale: float = 1.0):
    """Create a synthetic NPZ dataset file with specified intensity scale.

    Args:
        dataset_dir: Directory to create the file in.
        filename: Name of the NPZ file (e.g., "data_p1e5.npz").
        scale: Multiplier for diffraction intensities.

    Returns:
        Path to created NPZ file.
    """
    diff3d = (np.random.rand(2, 2, 2) * scale).astype(np.float32)
    probe = (np.random.rand(2, 2) + 1j * np.random.rand(2, 2)).astype(np.complex64)
    scan_index = np.array([0, 1])
    xcoords = np.array([0.0, 1.0])
    ycoords = np.array([0.0, 1.0])

    npz_path = dataset_dir / filename
    np.savez(
        npz_path,
        diff3d=diff3d,
        probeGuess=probe,
        scan_index=scan_index,
        xcoords=xcoords,
        ycoords=ycoords,
    )
    return npz_path


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Create a minimal synthetic dataset for CLI testing."""
    # Create dataset directory
    dataset_dir = tmp_path / "test_dataset"
    dataset_dir.mkdir()

    # Create small NPZ file with required keys
    diff3d = np.random.rand(2, 2, 2).astype(np.float32)  # 2 patterns, 2x2 pixels
    probe = (np.random.rand(2, 2) + 1j * np.random.rand(2, 2)).astype(np.complex64)
    scan_index = np.array([0, 1])  # Each pattern in its own scan
    xcoords = np.array([0.0, 1.0])
    ycoords = np.array([0.0, 1.0])

    npz_path = dataset_dir / "data_p1e5.npz"
    np.savez(
        npz_path,
        diff3d=diff3d,
        probeGuess=probe,
        scan_index=scan_index,
        xcoords=xcoords,
        ycoords=ycoords,
    )

    # Create minimal params.dill
    params = {
        "N": 64,
        "gridsize": 1,
        "nimgs_train": 100,
        "nimgs_test": 50,
        "batch_size": 32,
        "nepochs": 10,
        "mae_weight": 1.0,
        "nll_weight": 0.0,
        "probe.trainable": False,
        "probe.mask": True,
        "intensity_scale.trainable": False,
        "intensity_scale": 1.0,
        "label": "test_baseline",
        "timestamp": "2025-08-26",
        "ms_ssim": (0.92, 0.91),
        "psnr": (71.0, 70.5),
        "mae": (0.01, 0.012),
        "mse": (0.001, 0.0012),
        "frc50": (0.5, 0.48),
    }

    params_path = dataset_dir / "params.dill"
    with open(params_path, "wb") as f:
        dill.dump(params, f)

    return {
        "dataset_dir": dataset_dir,
        "npz_path": npz_path,
        "params_path": params_path,
        "diff3d": diff3d,
        "probe": probe,
    }


@pytest.fixture
def multi_dataset(tmp_path):
    """Create multiple synthetic datasets for multi-dataset testing."""
    dataset_dir = tmp_path / "multi_dataset"
    dataset_dir.mkdir()

    # Create two NPZ files with different photon doses
    npz_1e5 = _create_npz_dataset(dataset_dir, "data_p1e5.npz", scale=1.0)
    npz_1e6 = _create_npz_dataset(dataset_dir, "data_p1e6.npz", scale=10.0)

    # Create minimal params.dill
    params = {
        "N": 64,
        "gridsize": 1,
        "nimgs_train": 100,
        "nimgs_test": 50,
        "ms_ssim": (0.92, 0.91),
    }

    params_path = dataset_dir / "params.dill"
    with open(params_path, "wb") as f:
        dill.dump(params, f)

    return {
        "dataset_dir": dataset_dir,
        "npz_1e5": npz_1e5,
        "npz_1e6": npz_1e6,
        "params_path": params_path,
    }


def test_cli_emits_outputs(multi_dataset, tmp_path):
    """Verify CLI produces JSON, Markdown, and CSV outputs with multi-dataset coverage.

    This is the primary acceptance test for the D0 parity logger.
    Tests: scripts/tools/d0_parity_logger.py::main

    Validates:
    - JSON output includes all datasets
    - Markdown lists both dataset sections with all three stage tables (raw/normalized/grouped)
    - CSV probe stats cover all datasets
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Run CLI with multi-dataset fixture (two NPZ files: data_p1e5.npz, data_p1e6.npz)
    exit_code = main([
        "--dataset-root", str(multi_dataset["dataset_dir"]),
        "--baseline-params", str(multi_dataset["params_path"]),
        "--scenario-id", "TEST-MULTI-DATASET",
        "--output", str(output_dir),
    ])

    assert exit_code == 0, "CLI should return 0 on success"

    # Check JSON output exists and is valid
    json_path = output_dir / "dose_parity_log.json"
    assert json_path.exists(), "dose_parity_log.json should be created"

    with open(json_path) as f:
        summary = json.load(f)

    assert summary["metadata"]["scenario_id"] == "TEST-MULTI-DATASET"
    assert summary["total_datasets"] == 2
    assert len(summary["datasets"]) == 2

    # Verify both datasets are present
    filenames = [ds["filename"] for ds in summary["datasets"]]
    assert "data_p1e5.npz" in filenames
    assert "data_p1e6.npz" in filenames

    # Verify stats are present for all datasets
    for ds in summary["datasets"]:
        assert "raw" in ds["stats"]
        assert "normalized" in ds["stats"]
        assert "grouped" in ds["stats"]
        assert ds["stats"]["raw"]["count"] == 8  # 2x2x2 = 8 elements
        assert ds["stats"]["grouped"]["n_unique_scans"] == 2
        assert ds["stats"]["grouped"]["n_patterns"] == 2

    # Check Markdown output has stage-level stats for EVERY dataset
    md_path = output_dir / "dose_parity_log.md"
    assert md_path.exists(), "dose_parity_log.md should be created"

    md_content = md_path.read_text()
    assert "D0 Parity Log" in md_content
    assert "TEST-MULTI-DATASET" in md_content

    # Verify multi-dataset section heading
    assert "Stage-Level Stats by Dataset" in md_content

    # Verify both datasets have their own sections with all three stage tables
    assert "data_p1e5.npz" in md_content
    assert "data_p1e6.npz" in md_content
    assert "(photon dose: 1e5)" in md_content
    assert "(photon dose: 1e6)" in md_content

    # Verify stage table headings appear (should appear twice - once per dataset)
    assert md_content.count("#### Raw Diffraction") == 2
    assert md_content.count("#### Normalized Diffraction") == 2
    assert md_content.count("#### Grouped Intensity") == 2

    # Verify grouped stats show n_unique_scans and n_patterns
    assert "n_unique_scans" in md_content
    assert "n_patterns" in md_content

    # Check CSV output exists (probe_stats.csv)
    csv_path = output_dir / "probe_stats.csv"
    assert csv_path.exists(), "probe_stats.csv should be created"

    csv_content = csv_path.read_text()
    assert "filename" in csv_content
    assert "amp_mean" in csv_content
    assert "phase_mean" in csv_content
    # CSV should have both datasets
    assert "data_p1e5.npz" in csv_content
    assert "data_p1e6.npz" in csv_content


def test_cli_handles_missing_params(synthetic_dataset, tmp_path):
    """Verify CLI works without baseline params."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    exit_code = main([
        "--dataset-root", str(synthetic_dataset["dataset_dir"]),
        "--scenario-id", "TEST-NO-PARAMS",
        "--output", str(output_dir),
    ])

    assert exit_code == 0

    json_path = output_dir / "dose_parity_log.json"
    assert json_path.exists()

    with open(json_path) as f:
        summary = json.load(f)

    assert summary["baseline_params"] == {}
    assert summary["total_datasets"] == 1


def test_cli_returns_error_on_missing_dataset(tmp_path):
    """Verify CLI returns error code when no datasets found."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    exit_code = main([
        "--dataset-root", str(empty_dir),
        "--output", str(tmp_path / "output"),
    ])

    assert exit_code == 1, "CLI should return 1 when no datasets found"


def test_cli_limit_datasets_filters_inputs(multi_dataset, tmp_path):
    """Verify --limit-datasets flag filters to only requested datasets.

    Tests: scripts/tools/d0_parity_logger.py::main with --limit-datasets
    Validates:
    - Only the specified dataset appears in JSON output
    - Only the specified dataset appears in Markdown output
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Run CLI with --limit-datasets to only process data_p1e6.npz
    exit_code = main([
        "--dataset-root", str(multi_dataset["dataset_dir"]),
        "--baseline-params", str(multi_dataset["params_path"]),
        "--scenario-id", "TEST-LIMIT-FILTER",
        "--output", str(output_dir),
        "--limit-datasets", "data_p1e6.npz",
    ])

    assert exit_code == 0, "CLI should return 0 on success"

    # Check JSON only mentions the requested dataset
    json_path = output_dir / "dose_parity_log.json"
    assert json_path.exists()

    with open(json_path) as f:
        summary = json.load(f)

    assert summary["total_datasets"] == 1
    assert len(summary["datasets"]) == 1
    assert summary["datasets"][0]["filename"] == "data_p1e6.npz"

    # Check Markdown only mentions the requested dataset
    md_path = output_dir / "dose_parity_log.md"
    md_content = md_path.read_text()

    assert "data_p1e6.npz" in md_content
    assert "data_p1e5.npz" not in md_content  # Filtered out
    assert "(photon dose: 1e6)" in md_content
    assert "(photon dose: 1e5)" not in md_content  # Filtered out


def test_sha256_file(tmp_path):
    """Verify SHA256 computation is correct."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world")

    sha = sha256_file(str(test_file))

    # Known SHA256 of "hello world"
    expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    assert sha == expected
