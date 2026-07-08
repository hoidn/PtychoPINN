from pathlib import Path


def test_build_factorial_matrix_returns_12_runs_with_deterministic_ids():
    from scripts.studies.runbooks.run_nersc_scan807_cameraman_study_n128_factorial import (
        build_factorial_matrix,
    )

    runs = build_factorial_matrix(soft_mask_sigma=1.25, probe_mask_diameter=0.8)

    assert len(runs) == 12
    run_ids = [run["run_id"] for run in runs]
    assert len(set(run_ids)) == 12
    assert run_ids[0] == "pm-off__maenorm-off__ds-bincrop"
    assert run_ids[-1] == "pm-hard__maenorm-on__ds-cropbin"


def test_build_factorial_matrix_probe_mode_mapping():
    from scripts.studies.runbooks.run_nersc_scan807_cameraman_study_n128_factorial import (
        build_factorial_matrix,
    )

    runs = build_factorial_matrix(soft_mask_sigma=1.5, probe_mask_diameter=0.7)
    by_id = {run["run_id"]: run for run in runs}

    off = by_id["pm-off__maenorm-off__ds-bincrop"]
    assert off["probe_mode"] == "off"
    assert off["probe_mask"] is False
    assert off["probe_mask_sigma"] == 1.5
    assert off["probe_mask_diameter"] is None

    soft = by_id["pm-soft__maenorm-off__ds-bincrop"]
    assert soft["probe_mode"] == "on_soft"
    assert soft["probe_mask"] is True
    assert soft["probe_mask_sigma"] == 1.5
    assert soft["probe_mask_diameter"] == 0.7

    hard = by_id["pm-hard__maenorm-off__ds-bincrop"]
    assert hard["probe_mode"] == "on_hard"
    assert hard["probe_mask"] is True
    assert hard["probe_mask_sigma"] == 0.0
    assert hard["probe_mask_diameter"] == 0.7


def test_runbook_main_runs_12_combinations_and_writes_manifest(monkeypatch, tmp_path):
    from scripts.studies.runbooks import run_nersc_scan807_cameraman_study_n128_factorial as runbook

    calls = []

    def fake_subprocess_run(cmd, check, capture_output, text):
        assert check is False
        assert capture_output is True
        assert text is True
        calls.append(cmd)
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "manifest.json").write_text("{}")
        from subprocess import CompletedProcess
        return CompletedProcess(cmd, returncode=0, stdout='{"ok": true}', stderr="")

    monkeypatch.setattr(runbook.subprocess, "run", fake_subprocess_run)

    output_root = tmp_path / "factorial_out"
    argv = [
        "--scan807-dp",
        str(tmp_path / "scan_dp.hdf5"),
        "--scan807-para",
        str(tmp_path / "scan_para.hdf5"),
        "--cameraman-dp",
        str(tmp_path / "cam_dp.hdf5"),
        "--cameraman-para",
        str(tmp_path / "cam_para.hdf5"),
        "--ptychovit-checkpoint",
        str(tmp_path / "best_model.pth"),
        "--epochs",
        "20",
        "--soft-mask-sigma",
        "0.9",
        "--probe-mask-diameter",
        "0.8",
        "--output-root",
        str(output_root),
    ]
    runbook.main(argv)

    assert len(calls) == 12
    assert all("scripts/studies/runbooks/run_nersc_scan807_cameraman_study.py" in call for call in calls)
    assert all(call[call.index("--target-n") + 1] == "128" for call in calls)
    assert all(call[call.index("--epochs") + 1] == "20" for call in calls)
    assert all(call[call.index("--position-reassembly-backend") + 1] == "shift_sum" for call in calls)

    run_ids = [Path(call[call.index("--output-dir") + 1]).name for call in calls]
    assert "pm-off__maenorm-off__ds-bincrop" in run_ids
    assert "pm-hard__maenorm-on__ds-cropbin" in run_ids

    manifest_path = output_root / "factorial_manifest.json"
    assert manifest_path.exists()
    assert (output_root / "invocation.json").exists()
    assert (output_root / "invocation.sh").exists()
