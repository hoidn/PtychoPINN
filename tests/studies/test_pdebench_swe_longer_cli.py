import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


def _write_swe_h5(path: Path, *, shape=(4, 3, 8, 8, 1)) -> None:
    values = np.linspace(0.0, 1.0, num=int(np.prod(shape)), dtype=np.float32).reshape(shape)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=values)


def test_parse_args_accepts_longer_plan_flags(tmp_path):
    from scripts.studies.pdebench_swe.longer import parse_args

    args = parse_args(
        [
            "--data-file",
            str(tmp_path / "2D_rdb_NA_NA.h5"),
            "--output-root",
            str(tmp_path / "out"),
            "--profiles",
            "hybrid_resnet_base,fno_base,unet_base",
            "--run-ablations-if-viable",
            "--normalization-max-samples",
            "8000",
            "--eval-splits",
            "val,test",
            "--run-budget-file",
            str(tmp_path / "run_budget.json"),
            "--training-seed",
            "20260420",
            "--run-id",
            "longer-cli-test",
        ]
    )

    assert args.profiles == "hybrid_resnet_base,fno_base,unet_base"
    assert args.run_ablations_if_viable is True
    assert args.normalization_max_samples == 8000
    assert args.eval_splits == "val,test"
    assert args.training_seed == 20260420
    assert args.run_id == "longer-cli-test"


def test_longer_entrypoint_help_mentions_required_flags():
    result = subprocess.run(
        [sys.executable, "scripts/studies/run_pdebench_swe_longer.py", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "PDEBench SWE" in result.stdout
    assert "--profiles" in result.stdout
    assert "--normalization-max-samples" in result.stdout
    assert "--eval-splits" in result.stdout
    assert "--run-id" in result.stdout


def test_duplicate_output_root_rejected_for_longer_run(tmp_path, capsys):
    from scripts.studies.pdebench_swe.longer import main

    data_file = tmp_path / "2D_rdb_NA_NA.h5"
    _write_swe_h5(data_file)
    output_root = tmp_path / "out"
    output_root.mkdir()
    (output_root / "old.json").write_text("{}", encoding="utf-8")

    rc = main(
        [
            "--data-file",
            str(data_file),
            "--output-root",
            str(output_root),
            "--inspect-only",
        ]
    )

    assert rc == 2
    assert "refusing to write into non-empty output root" in capsys.readouterr().err


def test_prelaunch_tmux_markers_do_not_trip_duplicate_root_guard(tmp_path):
    from scripts.studies.pdebench_swe.longer import _guard_output_root

    output_root = tmp_path / "out"
    logs = output_root / "logs"
    logs.mkdir(parents=True)
    (logs / "longer.run_id").write_text("run-a\n", encoding="utf-8")
    (logs / "longer.started_at_ns").write_text("123\n", encoding="utf-8")

    _guard_output_root(output_root, allow_existing=False)

    (logs / "longer.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    try:
        _guard_output_root(output_root, allow_existing=False)
    except FileExistsError as exc:
        assert "live output root" in str(exc)
    else:  # pragma: no cover - failure path
        raise AssertionError("live PID marker without exit code should be rejected")

    try:
        _guard_output_root(output_root, allow_existing=True)
    except FileExistsError as exc:
        assert "live output root" in str(exc)
    else:  # pragma: no cover - failure path
        raise AssertionError("allow_existing must not bypass a live PID marker")

    stale_output_root = tmp_path / "stale-out"
    stale_logs = stale_output_root / "logs"
    stale_logs.mkdir(parents=True)
    (stale_logs / "longer.pid").write_text("999999999\n", encoding="utf-8")
    try:
        _guard_output_root(stale_output_root, allow_existing=True)
    except FileExistsError as exc:
        assert "missing exit code" in str(exc)
    else:  # pragma: no cover - failure path
        raise AssertionError("allow_existing must not bypass any PID marker without exit code")

    (logs / "longer.pid").write_text("999999999\n", encoding="utf-8")
    (logs / "longer.exit_code").write_text("0\n", encoding="utf-8")
    (output_root / "unexpected.json").write_text("{}", encoding="utf-8")
    try:
        _guard_output_root(output_root, allow_existing=False)
    except FileExistsError as exc:
        assert "refusing to write into non-empty output root" in str(exc)
    else:  # pragma: no cover - failure path
        raise AssertionError("unexpected non-marker file should still be rejected")


def test_guard_rejects_live_pid_even_with_stale_exit_code(tmp_path):
    from scripts.studies.pdebench_swe.longer import _guard_output_root

    output_root = tmp_path / "out"
    logs = output_root / "logs"
    logs.mkdir(parents=True)
    (logs / "longer.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    (logs / "longer.exit_code").write_text("0\n", encoding="utf-8")

    try:
        _guard_output_root(output_root, allow_existing=True)
    except FileExistsError as exc:
        assert "live output root" in str(exc)
    else:  # pragma: no cover - failure path
        raise AssertionError("allow_existing must not bypass a live PID marker")


def test_write_start_markers_invalidates_stale_exit_code(tmp_path):
    from scripts.studies.pdebench_swe.longer import _write_start_markers

    output_root = tmp_path / "out"
    logs = output_root / "logs"
    logs.mkdir(parents=True)
    (logs / "longer.exit_code").write_text("0\n", encoding="utf-8")

    _write_start_markers(output_root, "fresh-run")

    assert not (logs / "longer.exit_code").exists()
    assert (logs / "longer.run_id").read_text(encoding="utf-8").strip() == "fresh-run"
    assert (logs / "longer.pid").read_text(encoding="utf-8").strip() == str(os.getpid())


def test_cpu_synthetic_longer_run_writes_metrics_provenance_and_comparison(tmp_path):
    from scripts.studies.pdebench_swe.longer import main

    data_file = tmp_path / "2D_rdb_NA_NA.h5"
    _write_swe_h5(data_file, shape=(5, 4, 8, 8, 1))
    output_root = tmp_path / "out"

    rc = main(
        [
            "--data-file",
            str(data_file),
            "--output-root",
            str(output_root),
            "--dataset-source",
            "PDEBench",
            "--dataset-source-url",
            "https://github.com/pdebench/PDEBench",
            "--dataset-darus-id",
            "133021",
            "--profiles",
            "unet_base",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--learning-rate",
            "0.001",
            "--max-train-trajectories",
            "2",
            "--max-val-trajectories",
            "1",
            "--max-test-trajectories",
            "1",
            "--max-pairs-per-trajectory",
            "1",
            "--normalization-max-samples",
            "2",
            "--eval-splits",
            "val,test",
            "--device",
            "cpu",
            "--num-workers",
            "0",
            "--run-id",
            "longer-synthetic",
        ]
    )

    assert rc == 0
    assert (output_root / "dataset_manifest.json").exists()
    assert (output_root / "split_manifest_full.json").exists()
    assert (output_root / "split_manifest_run.json").exists()
    assert (output_root / "normalization_stats.json").exists()
    metrics = json.loads((output_root / "runs" / "unet_base" / "metrics.json").read_text())
    provenance = json.loads((output_root / "runs" / "unet_base" / "provenance.json").read_text())
    comparison = json.loads((output_root / "comparison_summary.json").read_text())
    invocation = json.loads((output_root / "invocation.json").read_text())

    assert metrics["run_id"] == "longer-synthetic"
    assert metrics["profile_id"] == "unet_base"
    assert metrics["eval"]["test"]["horizon"] == "one_step"
    assert provenance["run_id"] == "longer-synthetic"
    assert provenance["pid"] == metrics["pid"] == invocation["pid"]
    assert metrics["training_seed"] == 20260420
    assert provenance["training_seed"] == 20260420
    assert invocation["parsed_args"]["training_seed"] == 20260420
    assert comparison["run_id"] == "longer-synthetic"
    assert comparison["profiles"]["unet_base"]["status"] == "metrics"
    assert (output_root / "logs" / "longer.pid").read_text().strip() == str(invocation["pid"])


def test_budget_profile_override_cannot_omit_required_primary_profiles(tmp_path):
    from scripts.studies.pdebench_swe.longer import _apply_budget_defaults, parse_args

    budget_path = tmp_path / "run_budget.json"
    budget_path.write_text(
        json.dumps(
            {
                "schema_version": "pdebench_swe_run_budget_v1",
                "budget_id": "target",
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 1e-3,
                "max_train_trajectories": 2,
                "max_val_trajectories": 1,
                "max_test_trajectories": 1,
                "max_pairs_per_trajectory": 1,
                "normalization_max_samples": 2,
                "eval_splits": ["val", "test"],
                "num_workers": 0,
                "device": "cpu",
                "training_seed": 20260420,
                "primary_profiles": ["hybrid_resnet_base", "fno_base", "unet_base"],
                "ablation_profiles": ["hybrid_resnet_spectral_reduced", "hybrid_resnet_local_reduced"],
            }
        ),
        encoding="utf-8",
    )
    args = parse_args(
        [
            "--data-file",
            str(tmp_path / "2D_rdb_NA_NA.h5"),
            "--output-root",
            str(tmp_path / "out"),
            "--run-budget-file",
            str(budget_path),
            "--profiles",
            "hybrid_resnet_base,unet_base",
        ]
    )

    try:
        _apply_budget_defaults(args)
    except ValueError as exc:
        assert "required primary profiles" in str(exc)
        assert "fno_base" in str(exc)
    else:  # pragma: no cover - failure path
        raise AssertionError("budget-backed profile override omitted a required baseline")


def test_validate_fresh_artifacts_rejects_stale_wrong_run_id_and_pid(tmp_path):
    from scripts.studies.pdebench_swe.longer import validate_fresh_artifacts

    root = tmp_path / "out"
    model_root = root / "runs" / "unet_base"
    model_root.mkdir(parents=True)
    for name in [
        "dataset_manifest.json",
        "hdf5_metadata.json",
        "split_manifest_full.json",
        "split_manifest_run.json",
        "normalization_stats.json",
        "comparison_summary.json",
    ]:
        (root / name).write_text('{"run_id": "fresh"}', encoding="utf-8")
    (root / "comparison_summary.csv").write_text("profile_id,status\n", encoding="utf-8")
    (root / "invocation.json").write_text('{"run_id": "fresh", "pid": 123, "parsed_args": {"run_id": "fresh"}}', encoding="utf-8")
    (model_root / "provenance.json").write_text('{"run_id": "other", "pid": 999}', encoding="utf-8")
    (model_root / "metrics.json").write_text('{"run_id": "fresh", "pid": 999}', encoding="utf-8")

    old_ns = 1_000_000_000
    os.utime(root / "comparison_summary.csv", ns=(old_ns, old_ns))

    errors = validate_fresh_artifacts(
        output_root=root,
        run_id="fresh",
        tracked_pid="123",
        start_ns=old_ns + 1,
        profiles=["unet_base"],
    )

    assert any("stale" in error for error in errors)
    assert any("provenance" in error and "run_id" in error for error in errors)
    assert any("PID" in error for error in errors)


def test_validate_fresh_artifacts_requires_run_markers_and_successful_exit(tmp_path):
    from scripts.studies.pdebench_swe.longer import validate_fresh_artifacts

    root = tmp_path / "out"
    model_root = root / "runs" / "unet_base"
    model_root.mkdir(parents=True)
    for name in [
        "dataset_manifest.json",
        "hdf5_metadata.json",
        "split_manifest_full.json",
        "split_manifest_run.json",
        "normalization_stats.json",
        "comparison_summary.json",
    ]:
        (root / name).write_text('{"run_id": "fresh"}', encoding="utf-8")
    (root / "comparison_summary.csv").write_text("profile_id,status\n", encoding="utf-8")
    (root / "invocation.sh").write_text("python scripts/studies/run_pdebench_swe_longer.py\n", encoding="utf-8")
    (root / "invocation.json").write_text(
        '{"run_id": "fresh", "pid": 123, "parsed_args": {"run_id": "fresh"}}',
        encoding="utf-8",
    )
    (model_root / "provenance.json").write_text('{"run_id": "fresh", "pid": 123}', encoding="utf-8")
    (model_root / "metrics.json").write_text('{"run_id": "fresh", "pid": 123}', encoding="utf-8")

    errors = validate_fresh_artifacts(
        output_root=root,
        run_id="fresh",
        tracked_pid="123",
        start_ns=1,
        profiles=["unet_base"],
    )

    assert any("longer.exit_code" in error for error in errors)

    logs = root / "logs"
    logs.mkdir()
    (logs / "longer.run_id").write_text("fresh\n", encoding="utf-8")
    (logs / "longer.started_at_ns").write_text("2\n", encoding="utf-8")
    (logs / "longer.pid").write_text("123\n", encoding="utf-8")
    (logs / "longer.exit_code").write_text("1\n", encoding="utf-8")

    errors = validate_fresh_artifacts(
        output_root=root,
        run_id="fresh",
        tracked_pid="123",
        start_ns=1,
        profiles=["unet_base"],
    )

    assert any("longer.exit_code" in error and "0" in error for error in errors)


def test_validate_fresh_artifacts_rejects_invocation_output_root_mismatch(tmp_path):
    from scripts.studies.pdebench_swe.longer import validate_fresh_artifacts

    root = tmp_path / "out"
    logs = root / "logs"
    model_root = root / "runs" / "unet_base"
    logs.mkdir(parents=True)
    model_root.mkdir(parents=True)
    for name in [
        "dataset_manifest.json",
        "hdf5_metadata.json",
        "split_manifest_full.json",
        "split_manifest_run.json",
        "normalization_stats.json",
        "comparison_summary.json",
    ]:
        (root / name).write_text('{"run_id": "fresh"}', encoding="utf-8")
    (root / "comparison_summary.csv").write_text("profile_id,status\n", encoding="utf-8")
    (root / "invocation.sh").write_text("python scripts/studies/run_pdebench_swe_longer.py\n", encoding="utf-8")
    (root / "invocation.json").write_text(
        json.dumps(
            {
                "run_id": "fresh",
                "pid": 123,
                "parsed_args": {
                    "run_id": "fresh",
                    "output_root": str(tmp_path / "different-root"),
                },
            }
        ),
        encoding="utf-8",
    )
    (logs / "longer.run_id").write_text("fresh\n", encoding="utf-8")
    (logs / "longer.started_at_ns").write_text("1\n", encoding="utf-8")
    (logs / "longer.pid").write_text("123\n", encoding="utf-8")
    (logs / "longer.exit_code").write_text("0\n", encoding="utf-8")
    (model_root / "provenance.json").write_text('{"run_id": "fresh", "pid": 123}', encoding="utf-8")
    (model_root / "metrics.json").write_text('{"run_id": "fresh", "pid": 123}', encoding="utf-8")

    errors = validate_fresh_artifacts(
        output_root=root,
        run_id="fresh",
        tracked_pid="123",
        start_ns=1,
        profiles=["unet_base"],
    )

    assert any("output_root" in error and "does not match" in error for error in errors)
