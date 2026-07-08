from pathlib import Path
import subprocess


def test_runner_invokes_subprocess_with_resolved_paths(monkeypatch, tmp_path: Path):
    from scripts.studies.grid_lines_ptychovit_runner import PtychoViTRunnerConfig, run_grid_lines_ptychovit

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    recon_path = tmp_path / "recons" / "pinn_ptychovit" / "recon.npz"
    recon_path.parent.mkdir(parents=True, exist_ok=True)
    recon_path.write_bytes(b"stub")

    cfg = PtychoViTRunnerConfig(
        ptychovit_repo=tmp_path / "ptycho-vit",
        output_dir=tmp_path,
        train_dp=tmp_path / "train_dp.hdf5",
        test_dp=tmp_path / "test_dp.hdf5",
        model_n=256,
        mode="inference",
    )
    result = run_grid_lines_ptychovit(cfg)
    assert captured["cmd"][0] == "python"
    assert result["status"] == "ok"


def test_runner_returns_recon_npz_for_metrics_handoff(monkeypatch, tmp_path: Path):
    from scripts.studies.grid_lines_ptychovit_runner import PtychoViTRunnerConfig, run_grid_lines_ptychovit

    recon_path = tmp_path / "recons" / "pinn_ptychovit" / "recon.npz"
    recon_path.parent.mkdir(parents=True, exist_ok=True)
    recon_path.write_bytes(b"stub")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout="", stderr=""),
    )

    cfg = PtychoViTRunnerConfig(
        ptychovit_repo=tmp_path / "ptycho-vit",
        output_dir=tmp_path,
        train_dp=tmp_path / "train_dp.hdf5",
        test_dp=tmp_path / "test_dp.hdf5",
        model_n=256,
        mode="inference",
    )
    result = run_grid_lines_ptychovit(cfg)
    assert result["recon_npz"] == str(recon_path)
