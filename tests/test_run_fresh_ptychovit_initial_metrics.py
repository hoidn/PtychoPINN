from pathlib import Path
import subprocess


def _write_stub_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"stub-checkpoint")


def test_driver_refuses_existing_output_dir_without_force(tmp_path: Path):
    from scripts.studies.run_fresh_ptychovit_initial_metrics import main

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stale.txt").write_text("stale")
    checkpoint = tmp_path / "best_model.pth"
    _write_stub_checkpoint(checkpoint)

    try:
        main([
            "--checkpoint", str(checkpoint),
            "--output-dir", str(out_dir),
        ])
    except FileExistsError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("Expected FileExistsError for non-empty output dir without --force-clean")


def test_driver_copies_checkpoint_to_expected_bridge_location(monkeypatch, tmp_path: Path):
    from scripts.studies.run_fresh_ptychovit_initial_metrics import main

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    checkpoint = tmp_path / "seed" / "best_model.pth"
    _write_stub_checkpoint(checkpoint)
    out_dir = tmp_path / "fresh"

    rc = main([
        "--checkpoint", str(checkpoint),
        "--output-dir", str(out_dir),
        "--ptychovit-repo", str(tmp_path / "ptycho-vit"),
    ])
    assert rc == 0
    copied = out_dir / "runs" / "pinn_ptychovit" / "best_model.pth"
    assert copied.exists()
    assert copied.read_bytes() == checkpoint.read_bytes()


def test_driver_invokes_wrapper_without_reuse_flag(monkeypatch, tmp_path: Path):
    from scripts.studies.run_fresh_ptychovit_initial_metrics import main

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    checkpoint = tmp_path / "best_model.pth"
    _write_stub_checkpoint(checkpoint)
    out_dir = tmp_path / "fresh"

    rc = main([
        "--checkpoint", str(checkpoint),
        "--output-dir", str(out_dir),
    ])
    assert rc == 0
    assert "--reuse-existing-recons" not in captured["cmd"]
    assert "--models" in captured["cmd"]
    assert "pinn_ptychovit" in captured["cmd"]
    assert "--model-n" in captured["cmd"]


def test_driver_allows_nontrivial_scan_counts(monkeypatch, tmp_path: Path):
    from scripts.studies.run_fresh_ptychovit_initial_metrics import main

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    checkpoint = tmp_path / "best_model.pth"
    _write_stub_checkpoint(checkpoint)
    out_dir = tmp_path / "fresh"

    rc = main([
        "--checkpoint", str(checkpoint),
        "--output-dir", str(out_dir),
        "--nimgs-train", "8",
        "--nimgs-test", "8",
    ])
    assert rc == 0
    assert "--nimgs-train" in captured["cmd"]
    assert "--nimgs-test" in captured["cmd"]
