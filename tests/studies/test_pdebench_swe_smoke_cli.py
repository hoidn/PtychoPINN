import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


def _write_swe_h5(path: Path, *, shape=(4, 3, 8, 8, 1)) -> None:
    values = np.linspace(0.0, 1.0, num=int(np.prod(shape)), dtype=np.float32).reshape(shape)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=values)


def test_parse_args_accepts_plan_cli_flags(tmp_path):
    from scripts.studies.pdebench_swe.smoke import parse_args

    args = parse_args(
        [
            "--data-file",
            str(tmp_path / "2D_rdb_NA_NA.h5"),
            "--output-root",
            str(tmp_path / "out"),
            "--models",
            "hybrid_resnet,fno,unet",
            "--epochs",
            "1",
            "--run-id",
            "cli-test",
        ]
    )

    assert args.models == "hybrid_resnet,fno,unet"
    assert args.epochs == 1
    assert args.run_id == "cli-test"


def test_entrypoint_runs_by_repo_relative_script_path():
    result = subprocess.run(
        [sys.executable, "scripts/studies/run_pdebench_swe_smoke.py", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "PDEBench SWE one-step smoke gate" in result.stdout


def test_duplicate_output_root_rejects_non_empty_directory(tmp_path, capsys):
    from scripts.studies.pdebench_swe.smoke import main

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


def test_inspect_only_writes_manifests_and_invocation_artifacts(tmp_path):
    from scripts.studies.pdebench_swe.smoke import main

    data_file = tmp_path / "2D_rdb_NA_NA.h5"
    _write_swe_h5(data_file)
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
            "--license-note",
            "synthetic",
            "--inspect-only",
            "--run-id",
            "inspect-test",
        ]
    )

    assert rc == 0
    assert (output_root / "dataset_manifest.json").exists()
    assert (output_root / "hdf5_metadata.json").exists()
    invocation = json.loads((output_root / "invocation.json").read_text())
    assert invocation["extra"]["run_id"] == "inspect-test"
    assert "runtime_provenance" in invocation["extra"]


def test_cpu_unet_smoke_writes_metrics_and_provenance(tmp_path):
    from scripts.studies.pdebench_swe.smoke import main

    data_file = tmp_path / "2D_rdb_NA_NA.h5"
    _write_swe_h5(data_file)
    output_root = tmp_path / "out"

    rc = main(
        [
            "--data-file",
            str(data_file),
            "--output-root",
            str(output_root),
            "--models",
            "unet",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--max-train-trajectories",
            "2",
            "--max-val-trajectories",
            "1",
            "--max-test-trajectories",
            "1",
            "--max-pairs-per-trajectory",
            "1",
            "--max-train-batches",
            "1",
            "--max-eval-batches",
            "1",
            "--device",
            "cpu",
            "--run-id",
            "smoke-test",
        ]
    )

    assert rc == 0
    metrics = json.loads((output_root / "runs" / "unet" / "metrics.json").read_text())
    provenance = json.loads((output_root / "runs" / "unet" / "provenance.json").read_text())
    assert metrics["run_id"] == "smoke-test"
    assert metrics["model"] == "unet"
    assert metrics["horizon"] == "one_step"
    assert "err_nRMSE" in metrics
    assert provenance["run_id"] == "smoke-test"
    assert provenance["pid"] == metrics["pid"]


def test_validate_fresh_artifacts_rejects_run_id_mismatch(tmp_path):
    from scripts.studies.pdebench_swe.smoke import validate_fresh_artifacts

    root = tmp_path / "out"
    model_root = root / "runs" / "unet"
    model_root.mkdir(parents=True)
    (root / "dataset_manifest.json").write_text('{"run_id": "other"}', encoding="utf-8")
    (root / "hdf5_metadata.json").write_text('{"run_id": "fresh"}', encoding="utf-8")
    (root / "split_manifest.json").write_text('{"run_id": "fresh"}', encoding="utf-8")
    (root / "normalization_stats.json").write_text('{"run_id": "fresh"}', encoding="utf-8")
    (model_root / "provenance.json").write_text('{"run_id": "fresh", "pid": 123}', encoding="utf-8")
    (model_root / "metrics.json").write_text('{"run_id": "fresh"}', encoding="utf-8")

    errors = validate_fresh_artifacts(
        output_root=root,
        run_id="fresh",
        tracked_pid="123",
        start_ns=0,
        models=["unet"],
    )

    assert any("dataset_manifest.json" in error for error in errors)
