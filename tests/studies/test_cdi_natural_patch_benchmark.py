"""Tests for the natural-patch expanded CDI benchmark harness."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts.studies.cdi_natural_patch_benchmark import (
    NATURAL_PATCH_ROW_ROSTER,
    _patchwise_metrics,
    _save_patchwise_prediction_artifacts,
    _save_fixed_sample_visuals,
    prepare_natural_patch_inputs,
    run_natural_patch_benchmark,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_split_npz(path: Path, *, count: int, patch_size: int, source_prefix: str) -> None:
    yy, xx = np.indices((patch_size, patch_size))
    objects = []
    diffraction = []
    crop_coords = []
    source_ids = []
    patch_ids = []
    for idx in range(count):
        amp = 0.5 + 0.05 * idx + 0.001 * yy
        phase = -0.25 + 0.02 * idx + 0.001 * xx
        obj = (amp * np.exp(1j * phase)).astype(np.complex64)
        diff = np.abs(np.fft.fftshift(np.fft.fft2(obj)) / np.sqrt(obj.size)).astype(np.float32)
        objects.append(obj)
        diffraction.append(diff)
        crop_coords.append([idx, idx + patch_size, idx, idx + patch_size])
        source_ids.append(f"{source_prefix}_src_{idx}")
        patch_ids.append(f"{source_prefix}_patch_{idx}")
    np.savez(
        path,
        objects=np.stack(objects, axis=0).astype(np.complex64),
        diffraction=np.stack(diffraction, axis=0).astype(np.float32),
        crop_coords=np.asarray(crop_coords, dtype=np.int32),
        source_ids=np.asarray(source_ids),
        patch_ids=np.asarray(patch_ids),
    )


def _build_dataset_root(tmp_path: Path, *, patch_size: int = 8) -> Path:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    probe = np.ones((patch_size, patch_size), dtype=np.complex64)
    _write_json(
        dataset_root / "dataset_manifest.json",
        {
            "dataset_id": "natural_patches128_fixedprobe_v1",
            "patch_size": patch_size,
            "total_patches": 4,
            "source_corpus": "scikit-image",
            "object_count_cap": 10000,
        },
    )
    _write_json(
        dataset_root / "source_manifest.json",
        {
            "source_names": ["alpha", "beta", "gamma", "delta"],
            "package_name": "scikit-image",
            "package_version": "0.0-test",
        },
    )
    _write_json(
        dataset_root / "split_manifest.json",
        {
            "split_counts": {"train": 2, "val": 1, "test": 1},
            "no_source_overlap": True,
        },
    )
    _write_json(
        dataset_root / "probe_manifest.json",
        {
            "canonical_pipeline": "pad_extrapolate:128|smooth:0.5",
            "target_N": patch_size,
        },
    )
    _write_json(
        dataset_root / "simulation_manifest.json",
        {
            "object_encoding": {
                "amplitude_min": 0.5,
                "amplitude_max": 1.0,
                "phase_min_rad": -float(np.pi) / 2.0,
                "phase_max_rad": float(np.pi) / 2.0,
            }
        },
    )
    _write_json(
        dataset_root / "adapter_contract.json",
        {
            "dataset_id": "natural_patches128_fixedprobe_v1",
            "consumer_rule": "one_scan_group_per_object_patch",
        },
    )
    (dataset_root / "verification").mkdir(parents=True, exist_ok=True)
    _write_json(
        dataset_root / "verification" / "post_audit.json",
        {
            "manifests_present": True,
            "no_source_overlap": True,
            "total_objects": 4,
        },
    )
    _write_split_npz(dataset_root / "train.npz", count=2, patch_size=patch_size, source_prefix="train")
    _write_split_npz(dataset_root / "val.npz", count=1, patch_size=patch_size, source_prefix="val")
    _write_split_npz(dataset_root / "test.npz", count=1, patch_size=patch_size, source_prefix="test")
    np.savez(dataset_root / "probe.npz", probeGuess=probe)
    return dataset_root


def test_prepare_natural_patch_inputs_creates_grouped_splits_without_mutating_dataset_root(tmp_path: Path):
    dataset_root = _build_dataset_root(tmp_path)
    item_root = tmp_path / "item"
    before_files = sorted(path.relative_to(dataset_root) for path in dataset_root.rglob("*"))

    prepared = prepare_natural_patch_inputs(dataset_root=dataset_root, item_root=item_root)

    assert set(prepared["grouped_paths"].keys()) == {"train", "val", "test"}
    for split_name, grouped_path in prepared["grouped_paths"].items():
        grouped_npz = Path(grouped_path)
        assert grouped_npz.exists()
        with np.load(grouped_npz, allow_pickle=True) as data:
            assert data["diffraction"].ndim == 4
            assert data["diffraction"].shape[-1] == 1
            assert np.allclose(data["coords_nominal"], 0.0)
            assert np.allclose(data["coords_offsets"], 0.0)
            assert np.allclose(data["coords_true"], 0.0)
            assert data["Y"].shape[-1] == 1
            assert np.allclose(data["Y_I"][..., 0], np.abs(data["Y"][..., 0]))
            assert np.allclose(data["Y_phi"][..., 0], np.angle(data["Y"][..., 0]))
            assert data["metadata_split"][0] == split_name

    after_files = sorted(path.relative_to(dataset_root) for path in dataset_root.rglob("*"))
    assert after_files == before_files


def test_prepare_natural_patch_inputs_reuses_existing_prepared_inputs_when_valid(tmp_path: Path, monkeypatch):
    dataset_root = _build_dataset_root(tmp_path)
    item_root = tmp_path / "item"

    first = prepare_natural_patch_inputs(dataset_root=dataset_root, item_root=item_root)

    def fail_write_grouped_split(**kwargs):
        raise AssertionError(f"unexpected grouped rewrite for {kwargs['split_name']}")

    monkeypatch.setattr(
        "scripts.studies.cdi_natural_patch_benchmark._write_grouped_split",
        fail_write_grouped_split,
    )

    second = prepare_natural_patch_inputs(dataset_root=dataset_root, item_root=item_root)

    assert second["grouped_paths"] == first["grouped_paths"]
    assert second["prepared_input_manifest"] == first["prepared_input_manifest"]
    assert second["grouped_input_identity_audit"] == first["grouped_input_identity_audit"]


def test_prepare_natural_patch_inputs_accepts_pipeline_spec_probe_manifest(tmp_path: Path):
    dataset_root = _build_dataset_root(tmp_path)
    item_root = tmp_path / "item"
    probe_manifest_path = dataset_root / "probe_manifest.json"
    probe_manifest = json.loads(probe_manifest_path.read_text(encoding="utf-8"))
    probe_manifest.pop("canonical_pipeline", None)
    probe_manifest["pipeline_spec"] = "pad_extrapolate:128|smooth:0.5"
    probe_manifest_path.write_text(json.dumps(probe_manifest, indent=2), encoding="utf-8")

    prepared = prepare_natural_patch_inputs(dataset_root=dataset_root, item_root=item_root)

    prepared_manifest = json.loads(Path(prepared["prepared_input_manifest"]).read_text(encoding="utf-8"))
    assert prepared_manifest["probe_lineage"] == "pad_extrapolate:128|smooth:0.5"


def test_prepare_natural_patch_inputs_repairs_stale_probe_lineage_on_reuse(tmp_path: Path, monkeypatch):
    dataset_root = _build_dataset_root(tmp_path)
    item_root = tmp_path / "item"

    first = prepare_natural_patch_inputs(dataset_root=dataset_root, item_root=item_root)
    prepared_manifest_path = Path(first["prepared_input_manifest"])
    prepared_manifest = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
    prepared_manifest["probe_lineage"] = None
    prepared_manifest_path.write_text(json.dumps(prepared_manifest, indent=2), encoding="utf-8")

    def fail_write_grouped_split(**kwargs):
        raise AssertionError(f"unexpected grouped rewrite for {kwargs['split_name']}")

    monkeypatch.setattr(
        "scripts.studies.cdi_natural_patch_benchmark._write_grouped_split",
        fail_write_grouped_split,
    )

    second = prepare_natural_patch_inputs(dataset_root=dataset_root, item_root=item_root)

    repaired_manifest = json.loads(Path(second["prepared_input_manifest"]).read_text(encoding="utf-8"))
    assert repaired_manifest["probe_lineage"] == "pad_extrapolate:128|smooth:0.5"


def test_patchwise_metrics_skips_curve_like_metric_pairs(monkeypatch):
    pred = np.ones((2, 8, 8), dtype=np.complex64)
    gt = np.ones((2, 8, 8), dtype=np.complex64)

    def fake_eval_reconstruction(*args, **kwargs):
        return {
            "mae": (0.1, 0.2),
            "frc50": (5.0, 6.0),
            "frc": (np.array([0.9, 0.8], dtype=np.float32), np.array([0.7, 0.6], dtype=np.float32)),
        }

    monkeypatch.setattr(
        "scripts.studies.cdi_natural_patch_benchmark.evaluation.eval_reconstruction",
        fake_eval_reconstruction,
    )

    metrics = _patchwise_metrics(pred, gt, label="baseline")

    assert metrics["mae"] == (0.1, 0.2)
    assert metrics["frc50"] == (5.0, 6.0)
    assert "frc" not in metrics


def test_save_fixed_sample_visuals_accepts_singleton_channel_first_patches(tmp_path: Path):
    predictions = np.ones((2, 1, 8, 8), dtype=np.complex64)
    ground_truth = np.ones((2, 1, 8, 8), dtype=np.complex64)
    scales = {
        "amplitude": {"vmin": 0.0, "vmax": 1.0},
        "phase": {"vmin": -float(np.pi), "vmax": float(np.pi)},
        "error_amplitude": {"vmin": 0.0, "vmax": 1.0},
        "error_phase": {"vmin": 0.0, "vmax": 1.0},
    }

    visuals = _save_fixed_sample_visuals(
        run_root=tmp_path,
        model_id="pinn_ffno",
        predictions=predictions,
        ground_truth=ground_truth,
        fixed_sample_ids=[0],
        scales=scales,
    )

    assert (tmp_path / visuals["amp_phase_png"]).exists()
    assert (tmp_path / visuals["amp_phase_error_png"]).exists()


def test_save_patchwise_prediction_artifacts_preserves_fixed_sample_batch(tmp_path: Path):
    yy, xx = np.indices((8, 8))
    ground_truth = []
    predictions = []
    for idx in range(3):
        amp = 0.5 + 0.1 * idx + 0.01 * yy
        phase = -0.2 + 0.05 * idx + 0.01 * xx
        gt = (amp * np.exp(1j * phase)).astype(np.complex64)
        ground_truth.append(gt)
        predictions.append((gt * (1.0 + 0.01 * idx)).astype(np.complex64))
    scales = {
        "amplitude": {"vmin": 0.0, "vmax": 1.0},
        "phase": {"vmin": -float(np.pi), "vmax": float(np.pi)},
        "error_amplitude": {"vmin": 0.0, "vmax": 1.0},
        "error_phase": {"vmin": 0.0, "vmax": 1.0},
    }

    artifacts = _save_patchwise_prediction_artifacts(
        run_root=tmp_path,
        model_id="pinn_ffno",
        predictions=np.stack(predictions),
        ground_truth=np.stack(ground_truth),
        fixed_sample_ids=[0, 2],
        scales=scales,
    )

    npz_path = tmp_path / artifacts["patchwise_npz"]
    png_path = tmp_path / artifacts["gt_pred_error_png"]
    assert npz_path.exists()
    assert png_path.exists()
    with np.load(npz_path) as data:
        assert data["sample_ids"].tolist() == [0, 2]
        assert data["YY_pred"].shape == (2, 8, 8)
        assert data["YY_ground_truth"].shape == (2, 8, 8)
        assert np.allclose(data["YY_pred"][0], predictions[0])
        assert np.allclose(data["YY_pred"][1], predictions[2])


def test_run_natural_patch_benchmark_dry_run_writes_contract_and_locked_row_roster(tmp_path: Path):
    dataset_root = _build_dataset_root(tmp_path)
    item_root = tmp_path / "item"

    result = run_natural_patch_benchmark(
        dataset_root=dataset_root,
        item_root=item_root,
        mode="dry-run",
        rows=NATURAL_PATCH_ROW_ROSTER,
        seed=3,
    )

    assert result["mode"] == "dry-run"
    contract_path = item_root / "contract" / "natural_patch_benchmark_contract.json"
    fixed_sample_path = item_root / "contract" / "fixed_sample_manifest.json"
    visual_scales_path = item_root / "contract" / "shared_visual_scales.json"
    assert contract_path.exists()
    assert fixed_sample_path.exists()
    assert visual_scales_path.exists()

    contract_payload = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract_payload["dataset_id"] == "natural_patches128_fixedprobe_v1"
    assert contract_payload["row_roster"] == list(NATURAL_PATCH_ROW_ROSTER)
    assert contract_payload["single_seed"] == 3
    assert contract_payload["three_way_split"]["train"] == "prepared_inputs/train_grouped.npz"
    assert contract_payload["three_way_split"]["val"] == "prepared_inputs/val_grouped.npz"
    assert contract_payload["three_way_split"]["test"] == "prepared_inputs/test_grouped.npz"


def test_run_natural_patch_benchmark_records_explicit_row_statuses(tmp_path: Path):
    dataset_root = _build_dataset_root(tmp_path)
    item_root = tmp_path / "item"

    def fake_execute_rows(*_, **__):
        metrics = {
            "mae": (0.1, 0.2),
            "mse": (0.01, 0.02),
            "psnr": (10.0, 11.0),
            "ssim": (0.7, 0.8),
            "ms_ssim": (0.6, 0.7),
            "frc50": (5.0, 6.0),
        }
        return {
            "baseline": {
                "status": "completed",
                "row_payload": {
                    "model_label": "CDI CNN + supervised",
                    "architecture_id": "cnn",
                    "training_procedure": "supervised",
                    "N": 8,
                    "parameter_count": 10,
                    "epoch_budget": 40,
                    "final_completed_epoch": 40,
                    "final_train_loss": 0.1,
                    "validation_loss": {"status": "emitted", "value": 0.2},
                    "runtime_summary": {"train_wall_time_sec": 1.0, "inference_time_sec": 0.5},
                    "hardware_summary": {"backend": "tensorflow", "accelerator": "cpu"},
                    "row_status": "paper_grade",
                    "caveats": [],
                    "metrics": metrics,
                },
            },
            "pinn": {
                "status": "blocked",
                "reason": "synthetic test blocker",
            },
        }

    result = run_natural_patch_benchmark(
        dataset_root=dataset_root,
        item_root=item_root,
        mode="benchmark",
        rows=("baseline", "pinn"),
        seed=3,
        run_id="unit-test-run",
        execute_rows_fn=fake_execute_rows,
    )

    manifest_path = item_root / "runs" / "unit-test-run" / "paper_benchmark_manifest.json"
    metrics_path = item_root / "runs" / "unit-test-run" / "metrics.json"
    assert manifest_path.exists()
    assert metrics_path.exists()

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["row_statuses"]["baseline"]["status"] == "completed"
    assert manifest_payload["row_statuses"]["pinn"]["status"] == "blocked"
    assert manifest_payload["row_statuses"]["pinn"]["reason"] == "synthetic test blocker"
    assert "pinn" not in result["row_payloads"]


def test_run_natural_patch_benchmark_downgrades_when_visuals_or_provenance_missing(tmp_path: Path):
    dataset_root = _build_dataset_root(tmp_path)
    item_root = tmp_path / "item"

    def fake_execute_rows(*_, **__):
        metrics = {
            "mae": (0.1, 0.2),
            "mse": (0.01, 0.02),
            "psnr": (10.0, 11.0),
            "ssim": (0.7, 0.8),
            "ms_ssim": (0.6, 0.7),
            "frc50": (5.0, 6.0),
        }
        return {
            "baseline": {
                "status": "completed",
                "row_payload": {
                    "model_label": "CDI CNN + supervised",
                    "architecture_id": "cnn",
                    "training_procedure": "supervised",
                    "N": 8,
                    "parameter_count": 10,
                    "epoch_budget": 40,
                    "final_completed_epoch": 40,
                    "final_train_loss": 0.1,
                    "validation_loss": {"status": "emitted", "value": 0.2},
                    "runtime_summary": {"train_wall_time_sec": 1.0, "inference_time_sec": 0.5},
                    "hardware_summary": {"backend": "tensorflow", "accelerator": "cpu"},
                    "row_status": "paper_grade",
                    "caveats": [],
                    "metrics": metrics,
                },
            }
        }

    result = run_natural_patch_benchmark(
        dataset_root=dataset_root,
        item_root=item_root,
        mode="benchmark",
        rows=("baseline",),
        seed=3,
        run_id="downgrade-test-run",
        execute_rows_fn=fake_execute_rows,
    )

    metrics_path = item_root / "runs" / "downgrade-test-run" / "metrics.json"
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics_payload["benchmark_status"] == "benchmark_incomplete"
    missing = set(metrics_payload["missing_fields_by_row"]["baseline"])
    # Required provenance fields must be flagged when not provided.
    assert {"invocation", "config", "git", "environment", "dataset", "splits", "randomness", "outputs", "visuals"} <= missing
    # Result manifest must persist the row execution status alongside the bundle.
    manifest_path = item_root / "runs" / "downgrade-test-run" / "paper_benchmark_manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["row_statuses"]["baseline"]["status"] == "completed"
    # Bundle row_statuses must reflect that the launcher proof was not "supported_for_harness".
    assert metrics_payload["row_statuses"]["baseline"]["status"] == "completed"
    assert "paper_complete" != metrics_payload["benchmark_status"]
    assert result["row_statuses"]["baseline"]["status"] == "completed"


def test_run_natural_patch_benchmark_downgrades_blocked_or_failed_rows(tmp_path: Path):
    dataset_root = _build_dataset_root(tmp_path)
    item_root = tmp_path / "item"

    def fake_execute_rows(*_, **__):
        return {
            "baseline": {"status": "blocked", "reason": "synthetic launcher failure"},
            "pinn": {"status": "blocked", "reason": "synthetic launcher failure"},
        }

    run_natural_patch_benchmark(
        dataset_root=dataset_root,
        item_root=item_root,
        mode="benchmark",
        rows=("baseline", "pinn"),
        seed=3,
        run_id="failed-launch-run",
        execute_rows_fn=fake_execute_rows,
    )

    metrics_path = item_root / "runs" / "failed-launch-run" / "metrics.json"
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics_payload["benchmark_status"] == "benchmark_incomplete"
    # Both required rows must be recorded as missing because no row_payload was produced.
    for row in ("baseline", "pinn"):
        assert metrics_payload["missing_fields_by_row"][row], f"{row} must record missing fields"


def test_run_natural_patch_benchmark_uses_default_executor_in_benchmark_mode(tmp_path: Path, monkeypatch):
    dataset_root = _build_dataset_root(tmp_path)
    item_root = tmp_path / "item"
    calls: list[dict[str, object]] = []

    def fake_default_executor(**kwargs):
        calls.append(kwargs)
        return {
            "baseline": {
                "status": "completed",
                "row_payload": {
                    "model_label": "CDI CNN + supervised",
                    "architecture_id": "cnn",
                    "training_procedure": "supervised",
                    "N": 8,
                    "parameter_count": 10,
                    "epoch_budget": 40,
                    "final_completed_epoch": 40,
                    "final_train_loss": 0.1,
                    "validation_loss": {"status": "emitted", "value": 0.2},
                    "runtime_summary": {"train_wall_time_sec": 1.0, "inference_time_sec": 0.5},
                    "hardware_summary": {"backend": "tensorflow", "accelerator": "cpu"},
                    "row_status": "paper_grade",
                    "caveats": [],
                    "metrics": {
                        "mae": (0.1, 0.2),
                        "mse": (0.01, 0.02),
                        "psnr": (10.0, 11.0),
                        "ssim": (0.7, 0.8),
                        "ms_ssim": (0.6, 0.7),
                        "frc50": (5.0, 6.0),
                    },
                },
            }
        }

    monkeypatch.setattr(
        "scripts.studies.cdi_natural_patch_benchmark._execute_rows",
        fake_default_executor,
    )

    result = run_natural_patch_benchmark(
        dataset_root=dataset_root,
        item_root=item_root,
        mode="benchmark",
        rows=("baseline",),
        seed=3,
        run_id="unit-test-run",
    )

    assert calls, "expected benchmark mode to dispatch through _execute_rows by default"
    assert result["row_statuses"]["baseline"]["status"] == "completed"


def test_cli_entrypoint_runs_direct_script_mode(tmp_path: Path):
    dataset_root = _build_dataset_root(tmp_path)
    item_root = tmp_path / "item"
    script_path = Path("scripts/studies/run_cdi_natural_patch_benchmark.py")
    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--dataset-root",
            str(dataset_root),
            "--item-root",
            str(item_root),
            "--mode",
            "dry-run",
            "--rows",
            "baseline,pinn",
            "--seed",
            "3",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
