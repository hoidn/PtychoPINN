from pathlib import Path

import h5py
import numpy as np
import pickle
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
    manifest = yaml.safe_load((out_dir / "manifest.json").read_text())
    assert manifest["mode"] == "inference"
    assert Path(manifest["checkpoint"]).resolve() == ckpt.resolve()
    assert manifest["training_returncode"] is None


def test_bridge_inference_writes_runtime_normalization_config(monkeypatch, tmp_path: Path):
    from scripts.studies.ptychovit_bridge_entrypoint import main

    repo = tmp_path / "ptycho-vit"
    _write_fake_repo(repo)
    dp = tmp_path / "interop" / "test_dp.hdf5"
    para = tmp_path / "interop" / "test_para.hdf5"
    _write_pair(dp, para)

    out_dir = tmp_path / "infer_cfg"
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

    cfg_path = out_dir / "config.yaml"
    assert cfg_path.exists()
    cfg = yaml.safe_load(cfg_path.read_text())
    norm_path = Path(cfg["data"]["normalization_dict_path"])
    test_norm_path = Path(cfg["data"]["test_normalization"])
    assert norm_path.exists()
    assert test_norm_path.exists()
    assert norm_path == test_norm_path


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
            "true",
        ]
    )
    assert rc == 0
    assert (out_dir / "best_model.pth").exists()
    assert (out_dir / "checkpoint_model.pth").exists()
    assert (out_dir / "checkpoint.state").exists()


def test_bridge_finetune_requires_resume_from_checkpoint_true(monkeypatch, tmp_path: Path):
    from scripts.studies.ptychovit_bridge_entrypoint import main

    repo = tmp_path / "ptycho-vit"
    _write_fake_repo(repo)
    dp = tmp_path / "interop" / "train_dp.hdf5"
    para = tmp_path / "interop" / "train_para.hdf5"
    _write_pair(dp, para)

    out_dir = tmp_path / "runs" / "pinn_ptychovit"
    monkeypatch.setattr("scripts.studies.ptychovit_bridge_entrypoint.subprocess.run", _fake_training_run)
    monkeypatch.setattr("scripts.studies.ptychovit_bridge_entrypoint._run_model_inference", _fake_infer)

    try:
        main(
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
    except ValueError as exc:
        assert "resume-from-checkpoint=true" in str(exc)
    else:
        raise AssertionError("Expected ValueError when finetune runs without resume checkpoint")


def test_prepare_runtime_training_config_writes_absolute_paths(monkeypatch, tmp_path: Path):
    from scripts.studies.ptychovit_bridge_entrypoint import _prepare_runtime_training_config, parse_args

    repo = tmp_path / "ptycho-vit"
    _write_fake_repo(repo)
    dp = tmp_path / "interop" / "train_dp.hdf5"
    para = tmp_path / "interop" / "train_para.hdf5"
    _write_pair(dp, para)

    monkeypatch.chdir(tmp_path)
    args = parse_args(
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
            "relative_out/runs/pinn_ptychovit",
            "--resume-from-checkpoint",
            "false",
        ]
    )

    runtime_cfg_path, cfg = _prepare_runtime_training_config(args)
    data_path = Path(cfg["data"]["data_path"])
    test_path = Path(cfg["data"]["test_path"])
    model_save_path = Path(cfg["paths"]["model_save_path"])

    assert runtime_cfg_path.exists()
    assert data_path.is_absolute()
    assert test_path.is_absolute()
    assert model_save_path.is_absolute()
    assert data_path.exists()


def test_run_training_subprocess_forces_cpu(monkeypatch, tmp_path: Path):
    from scripts.studies.ptychovit_bridge_entrypoint import _run_training_subprocess, parse_args

    repo = tmp_path / "ptycho-vit"
    _write_fake_repo(repo)
    dp = tmp_path / "interop" / "train_dp.hdf5"
    para = tmp_path / "interop" / "train_para.hdf5"
    _write_pair(dp, para)

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_cfg_path = out_dir / "config.yaml"
    runtime_cfg_path.write_text("data: {}\n")

    captured = {}

    def _fake_run(cmd, cwd=None, env=None, capture_output=True, text=True, check=False):
        _ = (cwd, capture_output, text, check)
        captured["cmd"] = cmd
        captured["env"] = env
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr("scripts.studies.ptychovit_bridge_entrypoint.subprocess.run", _fake_run)

    args = parse_args(
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
        ]
    )

    rc, out, err = _run_training_subprocess(args, runtime_cfg_path)
    assert rc == 0
    assert out == "ok"
    assert err == ""
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == ""


def test_prepare_runtime_training_config_writes_normalization_dict(monkeypatch, tmp_path: Path):
    from scripts.studies.ptychovit_bridge_entrypoint import _prepare_runtime_training_config, parse_args

    repo = tmp_path / "ptycho-vit"
    _write_fake_repo(repo)
    dp = tmp_path / "interop" / "train_dp.hdf5"
    para = tmp_path / "interop" / "train_para.hdf5"
    _write_pair(dp, para)

    monkeypatch.chdir(tmp_path)
    args = parse_args(
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
            "relative_out/runs/pinn_ptychovit",
            "--resume-from-checkpoint",
            "false",
        ]
    )

    _runtime_cfg_path, cfg = _prepare_runtime_training_config(args)
    norm_path = Path(cfg["data"]["normalization_dict_path"])
    test_norm_path = Path(cfg["data"]["test_normalization"])

    assert norm_path.exists()
    assert test_norm_path.exists()
    assert norm_path == test_norm_path


def test_generated_normalization_dict_contains_train_and_test_object_names(monkeypatch, tmp_path: Path):
    from scripts.studies.ptychovit_bridge_entrypoint import _prepare_runtime_training_config, parse_args

    repo = tmp_path / "ptycho-vit"
    _write_fake_repo(repo)
    dp = tmp_path / "interop" / "train_dp.hdf5"
    para = tmp_path / "interop" / "train_para.hdf5"
    _write_pair(dp, para)

    monkeypatch.chdir(tmp_path)
    args = parse_args(
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
            "relative_out/runs/pinn_ptychovit",
            "--resume-from-checkpoint",
            "false",
        ]
    )

    _runtime_cfg_path, cfg = _prepare_runtime_training_config(args)
    norm_path = Path(cfg["data"]["normalization_dict_path"])
    with norm_path.open("rb") as handle:
        payload = pickle.load(handle)

    assert isinstance(payload, dict)
    assert "train" in payload
    assert "test" in payload


def test_bridge_inference_stitches_patches_using_positions():
    from scripts.studies.ptychovit_bridge_entrypoint import _stitch_complex_predictions

    patches = np.stack(
        [
            np.ones((3, 3), dtype=np.complex64),
            np.full((3, 3), 3.0 + 0.0j, dtype=np.complex64),
        ],
        axis=0,
    )
    positions = np.array([[2.0, 2.0], [2.0, 3.0]], dtype=np.float32)

    recon = _stitch_complex_predictions(
        patches=patches,
        positions_px=positions,
        object_shape=(6, 6),
    )

    assert np.isclose(recon[2, 1], 1.0 + 0.0j)
    assert np.isclose(recon[2, 2], 2.0 + 0.0j)
    assert np.isclose(recon[2, 4], 3.0 + 0.0j)
    assert np.unique(np.real(recon)).size > 2


def test_bridge_inference_outputs_object_space_shape_not_patch_mean_only():
    from scripts.studies.ptychovit_bridge_entrypoint import _stitch_complex_predictions

    patches = np.stack(
        [
            np.full((2, 2), 2.0 + 0.0j, dtype=np.complex64),
            np.full((2, 2), 4.0 + 0.0j, dtype=np.complex64),
        ],
        axis=0,
    )
    positions = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32)

    recon = _stitch_complex_predictions(
        patches=patches,
        positions_px=positions,
        object_shape=(5, 5),
    )

    assert recon.shape == (5, 5)
    np.testing.assert_allclose(recon[1:3, 1:3], np.full((2, 2), 3.0 + 0.0j, dtype=np.complex64))
    assert np.isclose(recon[0, 0], 0.0 + 0.0j)
