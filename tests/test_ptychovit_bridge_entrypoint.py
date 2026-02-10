from pathlib import Path

import h5py
import numpy as np
import subprocess
import yaml


def _write_pair(dp_path: Path, para_path: Path) -> None:
    dp_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(dp_path, "w") as dp_file:
        dp_file.create_dataset("dp", data=np.ones((4, 32, 32), dtype=np.float32))
    with h5py.File(para_path, "w") as para_file:
        obj = para_file.create_dataset(
            "object",
            data=(np.random.default_rng(0).normal(size=(1, 64, 64)) + 1j * np.random.default_rng(1).normal(size=(1, 64, 64))).astype(np.complex64),
        )
        obj.attrs["pixel_height_m"] = 1.0e-9
        obj.attrs["pixel_width_m"] = 1.0e-9
        probe = para_file.create_dataset("probe", data=np.ones((1, 1, 32, 32), dtype=np.complex64))
        probe.attrs["pixel_height_m"] = 1.0e-9
        probe.attrs["pixel_width_m"] = 1.0e-9
        para_file.create_dataset("probe_position_x_m", data=np.arange(4, dtype=np.float64))
        para_file.create_dataset("probe_position_y_m", data=np.arange(4, dtype=np.float64))


def _write_fake_repo(repo_dir: Path) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "main.py").write_text("print('stub trainer')\n")
    cfg = {
        "data": {
            "scale": 10000.0,
            "cache_object": False,
            "max_probe_modes": 8,
            "test_normalization": None,
        },
        "model": {"image_size": 32},
        "training": {"epochs": 1, "batch_size": 1, "learning_rate": 1e-4},
        "paths": {"model_save_path": "."},
        "trainer": {"run_num": 1},
        "wandb": {"enabled": False},
    }
    (repo_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def _fake_infer(args, checkpoint_path):
    _ = checkpoint_path
    args.recon_npz.parent.mkdir(parents=True, exist_ok=True)
    recon = (np.ones((64, 64)) + 1j * np.ones((64, 64))).astype(np.complex64)
    np.savez(args.recon_npz, YY_pred=recon, amp=np.abs(recon), phase=np.angle(recon))
    return args.recon_npz


def _fake_training_run(cmd, cwd=None, env=None, capture_output=True, text=True, check=False):
    _ = (cmd, env, capture_output, text, check)
    cfg_path = Path(cwd) / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    run_dir = Path(cfg["paths"]["model_save_path"]) / f"run{cfg['trainer']['run_num']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "best_model.pth").write_bytes(b"\x80\x04N.")
    (run_dir / "checkpoint_model.pth").write_bytes(b"\x80\x04N.")
    (run_dir / "checkpoint.state").write_text("{}")
    (run_dir / "config.yaml").write_text(cfg_path.read_text())
    return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")


def test_bridge_inference_defaults_recon_path_and_accepts_checkpoint(monkeypatch, tmp_path: Path):
    from scripts.studies.ptychovit_bridge_entrypoint import main

    repo = tmp_path / "ptycho-vit"
    _write_fake_repo(repo)
    dp = tmp_path / "interop" / "test_dp.hdf5"
    para = tmp_path / "interop" / "test_para.hdf5"
    _write_pair(dp, para)

    out_dir = tmp_path / "infer"
    ckpt = tmp_path / "best_model.pth"
    ckpt.write_bytes(b"\x80\x04N.")
    monkeypatch.setattr("scripts.studies.ptychovit_bridge_entrypoint._run_model_inference", _fake_infer)

    rc = main(
        [
            "--ptychovit-repo",
            str(repo),
            "--mode",
            "inference",
            "--train-dp",
            str(dp),
            "--test-dp",
            str(dp),
            "--test-para",
            str(para),
            "--output-dir",
            str(out_dir),
            "--checkpoint",
            str(ckpt),
        ]
    )
    assert rc == 0
    assert (out_dir / "recons" / "pinn_ptychovit" / "recon.npz").exists()
    assert (out_dir / "manifest.json").exists()


def test_bridge_inference_bootstraps_training_when_checkpoint_missing(monkeypatch, tmp_path: Path):
    from scripts.studies.ptychovit_bridge_entrypoint import main

    repo = tmp_path / "ptycho-vit"
    _write_fake_repo(repo)
    dp = tmp_path / "interop" / "test_dp.hdf5"
    para = tmp_path / "interop" / "test_para.hdf5"
    _write_pair(dp, para)
    out_dir = tmp_path / "infer_bootstrap"
    monkeypatch.setattr("scripts.studies.ptychovit_bridge_entrypoint.subprocess.run", _fake_training_run)
    monkeypatch.setattr("scripts.studies.ptychovit_bridge_entrypoint._run_model_inference", _fake_infer)

    rc = main(
        [
            "--ptychovit-repo",
            str(repo),
            "--mode",
            "inference",
            "--train-dp",
            str(dp),
            "--test-dp",
            str(dp),
            "--test-para",
            str(para),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    assert (out_dir / "recons" / "pinn_ptychovit" / "recon.npz").exists()
    assert (out_dir / "best_model.pth").exists()
    assert (out_dir / "manifest.json").exists()


def test_bridge_finetune_writes_checkpoint_artifacts(monkeypatch, tmp_path: Path):
    from scripts.studies.ptychovit_bridge_entrypoint import main

    repo = tmp_path / "ptycho-vit"
    _write_fake_repo(repo)
    dp = tmp_path / "interop" / "train_dp.hdf5"
    para = tmp_path / "interop" / "train_para.hdf5"
    _write_pair(dp, para)

    out_dir = tmp_path / "runs" / "pinn_ptychovit"
    monkeypatch.setattr("scripts.studies.ptychovit_bridge_entrypoint.subprocess.run", _fake_training_run)
    monkeypatch.setattr("scripts.studies.ptychovit_bridge_entrypoint._run_model_inference", _fake_infer)
    rc = main(
        [
            "--ptychovit-repo",
            str(repo),
            "--mode",
            "finetune",
            "--train-dp",
            str(dp),
            "--test-dp",
            str(dp),
            "--train-para",
            str(para),
            "--test-para",
            str(para),
            "--output-dir",
            str(out_dir),
            "--resume-from-checkpoint",
            "false",
        ]
    )
    assert rc == 0
    assert (out_dir / "best_model.pth").exists()
    assert (out_dir / "checkpoint_model.pth").exists()
    assert (out_dir / "checkpoint.state").exists()
