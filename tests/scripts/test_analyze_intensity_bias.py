"""Tests for plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py.

Per Phase D5: Verifies that the analyzer correctly parses and displays split_intensity_stats
from run_metadata.json, including train/test scale parity analysis.

Tests cover:
1. Parsing split_intensity_stats from run_metadata.json
2. Displaying train/test scale comparison in Markdown output
3. Graceful handling of missing/partial metadata

See: specs/spec-ptycho-core.md §Normalization Invariants
See: docs/fix_plan.md Phase D5
"""
import json
import numpy as np
import pytest
import tempfile
import os
import sys
from pathlib import Path


# Add the bin directory to path for imports
@pytest.fixture
def add_bin_to_path():
    bin_dir = Path(__file__).parent.parent.parent / "plans/active/DEBUG-SIM-LINES-DOSE-001/bin"
    str_path = str(bin_dir.resolve())
    if str_path not in sys.path:
        sys.path.insert(0, str_path)
    yield
    if str_path in sys.path:
        sys.path.remove(str_path)


class TestDatasetStats:
    """Test suite for split_intensity_stats parsing and display."""

    def test_reports_train_test(self, tmp_path, add_bin_to_path):
        """Verify analyzer surfaces train/test stats and tolerates missing metadata gracefully.

        This test:
        1. Creates a minimal fixture with stub run_metadata.json containing split_intensity_stats
        2. Creates required analyzer input files (intensity_stats.json, comparison_metrics.json, etc.)
        3. Verifies the analyzer parses and displays the new train/test stats
        4. Confirms tolerance for scenarios without split_intensity_stats (backward compat)

        Per Phase D5: The analyzer must display train vs test dataset scales alongside
        the bundle value and flag >5% deviation per the spec.
        """
        from analyze_intensity_bias import (
            gather_scenario_data,
            render_markdown,
            build_scenario_input,
            analyze_scenarios,
        )

        # Create scenario directory structure
        scenario_dir = tmp_path / "test_scenario"
        scenario_dir.mkdir()
        inference_dir = scenario_dir / "inference_outputs"
        inference_dir.mkdir()
        train_dir = scenario_dir / "train_outputs"
        train_dir.mkdir()

        # Create minimal test arrays (4x4 for speed)
        amp_truth = np.ones((4, 4), dtype=np.float32) * 0.5
        amp_pred = np.ones((4, 4), dtype=np.float32) * 0.45
        np.save(scenario_dir / "ground_truth_amp.npy", amp_truth)
        np.save(inference_dir / "amplitude.npy", amp_pred)

        # Create run_metadata.json with split_intensity_stats
        split_stats = {
            "train": {
                "batch_mean_sum_intensity": 3200.0,
                "n_samples": 128,
                "dataset_scale": 558.0,
            },
            "test": {
                "batch_mean_sum_intensity": 3100.0,
                "n_samples": 64,
                "dataset_scale": 567.0,
            },
            "nphotons": 1e9,
            "train_vs_test_scale_ratio": 0.984,
            "train_vs_test_scale_delta": -9.0,
            "train_vs_test_deviation_pct": 1.6,
            "deviation_exceeds_5pct": False,
            "spec_reference": "specs/spec-ptycho-core.md §Normalization Invariants",
        }

        run_metadata = {
            "scenario": "test_scenario",
            "split_intensity_stats": split_stats,
            "nphotons": 1e9,
            "intensity_stats": {
                "bundle_intensity_scale": 558.0,
                "legacy_params_intensity_scale": 988.21,
            },
        }
        (scenario_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2))

        # Create intensity_stats.json
        intensity_stats = {
            "bundle_intensity_scale": 558.0,
            "legacy_params_intensity_scale": 988.21,
            "scale_delta": -430.21,
            "stages": [],
        }
        (scenario_dir / "intensity_stats.json").write_text(json.dumps(intensity_stats, indent=2))

        # Create inference_outputs/stats.json
        inference_stats = {
            "amplitude": {"min": 0.0, "max": 1.0, "mean": 0.45},
            "phase": {"min": -3.14, "max": 3.14, "mean": 0.0},
            "fits_canvas": True,
            "padded_size": 128,
        }
        (inference_dir / "stats.json").write_text(json.dumps(inference_stats, indent=2))

        # Create comparison_metrics.json
        comparison_metrics = {
            "amplitude": {
                "mae": 0.05,
                "rmse": 0.06,
                "max_abs": 0.1,
                "pearson_r": 0.95,
                "count": 16,
                "bias_summary": {"mean": -0.05, "median": -0.05, "p05": -0.06, "p95": -0.04},
                "pred_stats": {"min": 0.4, "max": 0.5, "mean": 0.45, "std": 0.02},
                "truth_stats": {"min": 0.45, "max": 0.55, "mean": 0.5, "std": 0.02},
            },
            "phase": {
                "mae": 0.01,
                "rmse": 0.02,
                "max_abs": 0.05,
                "pearson_r": 0.99,
                "count": 16,
                "bias_summary": {"mean": 0.0, "median": 0.0, "p05": -0.01, "p95": 0.01},
                "pred_stats": {"min": -0.1, "max": 0.1, "mean": 0.0, "std": 0.05},
                "truth_stats": {"min": -0.1, "max": 0.1, "mean": 0.0, "std": 0.05},
            },
        }
        (scenario_dir / "comparison_metrics.json").write_text(json.dumps(comparison_metrics, indent=2))

        # Create train_outputs/history_summary.json
        training_summary = {
            "metrics": {
                "loss": {"last": 0.1, "min": 0.05, "max": 0.5, "has_nan": False, "count": 5}
            },
            "nan_overview": {"has_nan": False, "metrics": {}},
        }
        (train_dir / "history_summary.json").write_text(json.dumps(training_summary, indent=2))

        # Build scenario input and gather data
        scenario_input = build_scenario_input("test_scenario", scenario_dir)
        scenario_data = gather_scenario_data(scenario_input)

        # Verify split_intensity_stats was parsed
        assert "split_intensity_stats" in scenario_data, \
            "split_intensity_stats should be in gathered scenario data"
        assert scenario_data["split_intensity_stats"] is not None, \
            "split_intensity_stats should not be None"

        parsed_stats = scenario_data["split_intensity_stats"]
        assert "train" in parsed_stats
        assert "test" in parsed_stats
        assert parsed_stats["train"]["dataset_scale"] == 558.0
        assert parsed_stats["test"]["dataset_scale"] == 567.0
        assert parsed_stats["train_vs_test_deviation_pct"] == 1.6
        assert parsed_stats["deviation_exceeds_5pct"] is False

        # Test full summary rendering
        summary = analyze_scenarios([scenario_input])
        md_output = render_markdown(summary)

        # Verify train/test section appears in markdown
        assert "Train/Test Intensity Scale Parity" in md_output, \
            "Markdown should contain Train/Test Intensity Scale Parity section"
        assert "558" in md_output, "Train scale should appear in output"
        assert "567" in md_output, "Test scale should appear in output"
        assert "Train/Test scale ratio" in md_output
        assert "within 5% tolerance" in md_output, \
            "Should indicate parity is within tolerance for 1.6% deviation"

    def test_tolerates_missing_split_stats(self, tmp_path, add_bin_to_path):
        """Verify analyzer handles scenarios without split_intensity_stats gracefully.

        Per Phase D5: The analyzer must tolerate scenarios created before the
        split_intensity_stats instrumentation was added (backward compatibility).
        """
        from analyze_intensity_bias import (
            gather_scenario_data,
            render_markdown,
            build_scenario_input,
            analyze_scenarios,
        )

        # Create scenario directory structure
        scenario_dir = tmp_path / "legacy_scenario"
        scenario_dir.mkdir()
        inference_dir = scenario_dir / "inference_outputs"
        inference_dir.mkdir()
        train_dir = scenario_dir / "train_outputs"
        train_dir.mkdir()

        # Create minimal test arrays
        amp_truth = np.ones((4, 4), dtype=np.float32) * 0.5
        amp_pred = np.ones((4, 4), dtype=np.float32) * 0.45
        np.save(scenario_dir / "ground_truth_amp.npy", amp_truth)
        np.save(inference_dir / "amplitude.npy", amp_pred)

        # Create run_metadata.json WITHOUT split_intensity_stats (legacy format)
        run_metadata = {
            "scenario": "legacy_scenario",
            "nphotons": 1e9,
            # No split_intensity_stats key
        }
        (scenario_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2))

        # Create intensity_stats.json
        intensity_stats = {
            "bundle_intensity_scale": 558.0,
            "legacy_params_intensity_scale": 988.21,
            "stages": [],
        }
        (scenario_dir / "intensity_stats.json").write_text(json.dumps(intensity_stats, indent=2))

        # Create inference_outputs/stats.json
        inference_stats = {
            "amplitude": {"min": 0.0, "max": 1.0, "mean": 0.45},
            "fits_canvas": True,
        }
        (inference_dir / "stats.json").write_text(json.dumps(inference_stats, indent=2))

        # Create comparison_metrics.json
        comparison_metrics = {
            "amplitude": {
                "mae": 0.05,
                "bias_summary": {"mean": -0.05},
                "pred_stats": {"mean": 0.45},
                "truth_stats": {"mean": 0.5},
            },
            "phase": {
                "mae": 0.01,
                "bias_summary": {"mean": 0.0},
                "pred_stats": {"mean": 0.0},
                "truth_stats": {"mean": 0.0},
            },
        }
        (scenario_dir / "comparison_metrics.json").write_text(json.dumps(comparison_metrics, indent=2))

        # Create train_outputs/history_summary.json
        training_summary = {
            "metrics": {},
            "nan_overview": {"has_nan": False},
        }
        (train_dir / "history_summary.json").write_text(json.dumps(training_summary, indent=2))

        # Build scenario input and gather data - should not raise
        scenario_input = build_scenario_input("legacy_scenario", scenario_dir)
        scenario_data = gather_scenario_data(scenario_input)

        # split_intensity_stats should be None for legacy scenarios
        assert scenario_data.get("split_intensity_stats") is None, \
            "split_intensity_stats should be None for legacy scenarios"

        # Rendering should succeed without the train/test section
        summary = analyze_scenarios([scenario_input])
        md_output = render_markdown(summary)

        # Should NOT contain the train/test section
        assert "Train/Test Intensity Scale Parity" not in md_output, \
            "Legacy scenarios should not show Train/Test section"

    def test_flags_deviation_exceeding_5pct(self, tmp_path, add_bin_to_path):
        """Verify analyzer flags >5% train/test scale deviation per spec."""
        from analyze_intensity_bias import render_markdown, analyze_scenarios, build_scenario_input

        # Create scenario with >5% deviation
        scenario_dir = tmp_path / "high_deviation"
        scenario_dir.mkdir()
        inference_dir = scenario_dir / "inference_outputs"
        inference_dir.mkdir()
        train_dir = scenario_dir / "train_outputs"
        train_dir.mkdir()

        amp = np.ones((4, 4), dtype=np.float32) * 0.5
        np.save(scenario_dir / "ground_truth_amp.npy", amp)
        np.save(inference_dir / "amplitude.npy", amp)

        # Split stats with >5% deviation
        split_stats = {
            "train": {"batch_mean_sum_intensity": 3200.0, "n_samples": 128, "dataset_scale": 558.0},
            "test": {"batch_mean_sum_intensity": 4500.0, "n_samples": 64, "dataset_scale": 471.0},
            "nphotons": 1e9,
            "train_vs_test_scale_ratio": 1.185,
            "train_vs_test_scale_delta": 87.0,
            "train_vs_test_deviation_pct": 18.5,
            "deviation_exceeds_5pct": True,
            "spec_reference": "specs/spec-ptycho-core.md §Normalization Invariants",
        }

        run_metadata = {"scenario": "high_deviation", "split_intensity_stats": split_stats}
        (scenario_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2))

        intensity_stats = {"bundle_intensity_scale": 558.0, "legacy_params_intensity_scale": 988.21, "stages": []}
        (scenario_dir / "intensity_stats.json").write_text(json.dumps(intensity_stats, indent=2))

        inference_stats = {"amplitude": {"mean": 0.5}, "fits_canvas": True}
        (inference_dir / "stats.json").write_text(json.dumps(inference_stats, indent=2))

        comparison_metrics = {
            "amplitude": {"mae": 0.1, "bias_summary": {"mean": -0.1}, "pred_stats": {"mean": 0.4}, "truth_stats": {"mean": 0.5}},
            "phase": {"mae": 0.01, "bias_summary": {"mean": 0.0}, "pred_stats": {"mean": 0.0}, "truth_stats": {"mean": 0.0}},
        }
        (scenario_dir / "comparison_metrics.json").write_text(json.dumps(comparison_metrics, indent=2))

        training_summary = {"metrics": {}, "nan_overview": {"has_nan": False}}
        (train_dir / "history_summary.json").write_text(json.dumps(training_summary, indent=2))

        scenario_input = build_scenario_input("high_deviation", scenario_dir)
        summary = analyze_scenarios([scenario_input])
        md_output = render_markdown(summary)

        # Should contain warning about exceeding 5%
        assert "exceeds 5%" in md_output.lower(), \
            "Should flag when train/test deviation exceeds 5%"
        assert "18.5" in md_output or "18.50" in md_output, \
            "Should display the deviation percentage"
