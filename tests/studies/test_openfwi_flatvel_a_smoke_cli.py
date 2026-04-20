import json
import os
import subprocess
import sys

import numpy as np


def _write_openfwi_fixture(root, *, samples=4):
    rng = np.random.default_rng(20260420)
    for prefix, offset in [("1", 0.0), ("49", 1.0)]:
        data = rng.normal(loc=offset, size=(samples, 5, 1000, 70)).astype(np.float32)
        model = rng.normal(loc=1500.0 + offset, scale=10.0, size=(samples, 1, 70, 70)).astype(np.float32)
        np.save(root / f"data{prefix}.npy", data)
        np.save(root / f"model{prefix}.npy", model)


def test_help_mentions_required_flags():
    result = subprocess.run(
        [sys.executable, "scripts/studies/run_openfwi_flatvel_a_smoke.py", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "OpenFWI FlatVel-A" in result.stdout
    assert "--data-root" in result.stdout
    assert "--output-root" in result.stdout
    assert "--profiles" in result.stdout
    assert "--run-id" in result.stdout
    assert "--official-openfwi-repo" in result.stdout


def test_duplicate_output_root_rejects_non_empty_directory(tmp_path, capsys):
    from scripts.studies.openfwi_flatvel_a.smoke import main

    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_openfwi_fixture(data_root)
    output_root = tmp_path / "out"
    output_root.mkdir()
    (output_root / "old.json").write_text("{}", encoding="utf-8")

    rc = main(["--data-root", str(data_root), "--output-root", str(output_root), "--inspect-only"])

    assert rc == 2
    assert "refusing to write into non-empty output root" in capsys.readouterr().err


def test_live_pid_marker_is_rejected_even_with_exit_code(tmp_path, capsys):
    from scripts.studies.openfwi_flatvel_a.smoke import main

    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_openfwi_fixture(data_root)
    output_root = tmp_path / "out"
    log_root = output_root / "runs/stale/logs"
    log_root.mkdir(parents=True)
    (log_root / "smoke.pid").write_text(str(os.getpid()), encoding="utf-8")
    (log_root / "smoke.exit_code").write_text("0", encoding="utf-8")

    rc = main(
        [
            "--data-root",
            str(data_root),
            "--output-root",
            str(output_root),
            "--allow-existing-output-root",
            "--inspect-only",
        ]
    )

    assert rc == 2
    assert "live OpenFWI smoke output root exists" in capsys.readouterr().err


def test_synthetic_cpu_smoke_writes_metrics_provenance_and_comparison(tmp_path):
    from scripts.studies.openfwi_flatvel_a.smoke import main

    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_openfwi_fixture(data_root)
    output_root = tmp_path / "out"

    rc = main(
        [
            "--data-root",
            str(data_root),
            "--output-root",
            str(output_root),
            "--source-url",
            "https://openfwi-lanl.github.io/docs/data.html",
            "--source-access-note",
            "synthetic local shards",
            "--license-note",
            "CC BY-NC-SA 4.0",
            "--profiles",
            "hybrid_resnet_smoke,unet_smoke",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--train-samples",
            "2",
            "--val-samples",
            "1",
            "--test-samples",
            "1",
            "--device",
            "cpu",
            "--run-id",
            "smoke-test",
        ]
    )

    assert rc == 0
    assert (output_root / "invocation.json").exists()
    assert (output_root / "data_manifest.json").exists()
    assert (output_root / "shard_shapes.json").exists()
    assert (output_root / "split_manifest.json").exists()
    assert (output_root / "normalization_stats.json").exists()
    assert (output_root / "official_inversionnet_blocker.json").exists()
    assert (output_root / "runs/hybrid_resnet_smoke/metrics.json").exists()
    assert (output_root / "runs/unet_smoke/metrics.json").exists()
    assert (output_root / "comparison_summary.json").exists()
    comparison = json.loads((output_root / "comparison_summary.json").read_text())
    assert comparison["run_id"] == "smoke-test"
    assert comparison["local_baseline_complete"] is True


def test_validate_fresh_artifacts_rejects_missing_exit_code(tmp_path):
    from scripts.studies.openfwi_flatvel_a.smoke import validate_fresh_artifacts

    root = tmp_path / "out"
    (root / "logs").mkdir(parents=True)
    (root / "data_manifest.json").write_text('{"run_id": "r1"}', encoding="utf-8")

    errors = validate_fresh_artifacts(output_root=root, run_id="r1", start_ns=0)

    assert any("smoke.exit_code" in error for error in errors)
