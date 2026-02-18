from pathlib import Path
import subprocess

import numpy as np


def _touch_pair(dp_path: Path, para_path: Path) -> None:
    dp_path.parent.mkdir(parents=True, exist_ok=True)
    para_path.parent.mkdir(parents=True, exist_ok=True)
    dp_path.write_bytes(b"dp")
    para_path.write_bytes(b"para")


def test_ptychovit_inference_stage_invokes_bridge_entrypoint_with_checkpoint(
    monkeypatch, tmp_path
):
    from scripts.studies.nersc_orchestration import run_ptychovit_inference_stage

    checkpoint = tmp_path / "best_model.pth"
    checkpoint.write_bytes(b"ckpt")
    repo = tmp_path / "ptychovit_repo"
    repo.mkdir()

    scan_dp = tmp_path / "scan807_dp.hdf5"
    scan_para = tmp_path / "scan807_para.hdf5"
    cam_dp = tmp_path / "cameraman256_dp.hdf5"
    cam_para = tmp_path / "cameraman256_para.hdf5"
    _touch_pair(scan_dp, scan_para)
    _touch_pair(cam_dp, cam_para)

    calls: list[list[str]] = []

    def fake_run(cmd, check, capture_output, text):
        calls.append([str(x) for x in cmd])
        recon_path = Path(cmd[cmd.index("--recon-npz") + 1])
        recon_path.parent.mkdir(parents=True, exist_ok=True)
        recon_path.write_bytes(b"npz")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    run_ptychovit_inference_stage(
        dataset_pairs={
            "scan807": (scan_dp, scan_para),
            "cameraman256": (cam_dp, cam_para),
        },
        ptychovit_repo=repo,
        checkpoint=checkpoint,
        output_dir=tmp_path / "outputs",
    )

    assert len(calls) == 2
    for cmd in calls:
        assert "scripts/studies/ptychovit_bridge_entrypoint.py" in cmd
        assert "--mode" in cmd and cmd[cmd.index("--mode") + 1] == "inference"
        assert "--checkpoint" in cmd and cmd[cmd.index("--checkpoint") + 1] == str(checkpoint)


def test_ptychovit_stage_writes_recon_artifact_paths_for_scan807_and_cameraman(
    monkeypatch, tmp_path
):
    from scripts.studies.nersc_orchestration import run_ptychovit_inference_stage

    checkpoint = tmp_path / "best_model.pth"
    checkpoint.write_bytes(b"ckpt")
    repo = tmp_path / "ptychovit_repo"
    repo.mkdir()

    scan_dp = tmp_path / "scan807_dp.hdf5"
    scan_para = tmp_path / "scan807_para.hdf5"
    cam_dp = tmp_path / "cameraman256_dp.hdf5"
    cam_para = tmp_path / "cameraman256_para.hdf5"
    _touch_pair(scan_dp, scan_para)
    _touch_pair(cam_dp, cam_para)

    def fake_run(cmd, check, capture_output, text):
        recon_path = Path(cmd[cmd.index("--recon-npz") + 1])
        recon_path.parent.mkdir(parents=True, exist_ok=True)
        recon_path.write_bytes(b"npz")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    result = run_ptychovit_inference_stage(
        dataset_pairs={
            "scan807": (scan_dp, scan_para),
            "cameraman256": (cam_dp, cam_para),
        },
        ptychovit_repo=repo,
        checkpoint=checkpoint,
        output_dir=tmp_path / "outputs",
    )

    assert "scan807" in result
    assert "cameraman256" in result
    assert Path(result["scan807"]["recon_npz"]).exists()
    assert Path(result["cameraman256"]["recon_npz"]).exists()


def test_full_orchestration_uses_cached_test_npz_for_cross_dataset_inference(
    monkeypatch, tmp_path
):
    from scripts.studies import nersc_orchestration as orch

    checkpoint = tmp_path / "best_model.pth"
    checkpoint.write_bytes(b"ckpt")
    scan_dp = tmp_path / "scan807_dp.hdf5"
    scan_para = tmp_path / "scan807_para.hdf5"
    cam_dp = tmp_path / "cameraman_dp.hdf5"
    cam_para = tmp_path / "cameraman_para.hdf5"
    _touch_pair(scan_dp, scan_para)
    _touch_pair(cam_dp, cam_para)

    def fake_materialize(dp, para, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        return dp, para

    def fake_ptychovit_stage(**kwargs):
        out_dir = Path(kwargs["output_dir"])
        out = {}
        for name in ("scan807", "cameraman256"):
            recon = out_dir / name / "recons" / "pinn_ptychovit" / "recon.npz"
            recon.parent.mkdir(parents=True, exist_ok=True)
            recon.write_bytes(b"npz")
            out[name] = {"recon_npz": str(recon)}
        return out

    def fake_prepare(**kwargs):
        out_dir = Path(kwargs["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        train = out_dir / "train_raw.npz"
        test = out_dir / "test_raw.npz"
        down = out_dir / "down_raw.npz"
        for p in (train, test, down):
            p.write_bytes(b"npz")
        return {"train_npz": str(train), "test_npz": str(test), "downsampled_npz": str(down)}

    def fake_convert(**kwargs):
        out = Path(kwargs["out_npz"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"npz")
        return out

    def fake_build_datasets(**kwargs):
        cfg = kwargs["cfg"]
        base = Path(cfg.output_dir) / "datasets" / "N128" / "gs1"
        base.mkdir(parents=True, exist_ok=True)
        train = base / "train.npz"
        test = base / "test.npz"
        train.write_bytes(b"npz")
        test.write_bytes(b"npz")
        return {128: {"train_npz": str(train), "test_npz": str(test), "gt_recon": str(base / "gt.npz"), "tag": "N128"}}

    def fake_run_grid(cfg):
        run_dir = Path(cfg.output_dir) / "runs" / "pinn_hybrid_resnet"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "model.pt").write_bytes(b"weights")
        return {"run_dir": str(run_dir)}

    captured = {}

    def fake_cross(**kwargs):
        captured["dataset_npzs"] = kwargs["dataset_npzs"]
        captured["allow_oom_fallback"] = kwargs.get("allow_oom_fallback")
        out = {}
        for name in kwargs["dataset_npzs"]:
            recon = Path(kwargs["output_dir"]) / name / "recons" / "pinn_hybrid_resnet" / "recon.npz"
            recon.parent.mkdir(parents=True, exist_ok=True)
            recon.write_bytes(b"npz")
            out[name] = {"recon_npz": str(recon)}
        return out

    def fake_gt(dataset_output_dir, external_npz):
        _ = external_npz
        gt = Path(dataset_output_dir) / "recons" / "gt" / "recon.npz"
        gt.parent.mkdir(parents=True, exist_ok=True)
        gt.write_bytes(b"npz")
        return gt

    monkeypatch.setattr(orch, "materialize_pair_working_copy", fake_materialize)
    monkeypatch.setattr(orch, "run_ptychovit_inference_stage", fake_ptychovit_stage)
    monkeypatch.setattr(orch, "prepare_hybrid_dataset", fake_prepare)
    monkeypatch.setattr(orch, "_convert_pair_to_downsampled_external_npz", fake_convert)
    monkeypatch.setattr(orch, "build_datasets", fake_build_datasets)
    monkeypatch.setattr(orch, "run_grid_lines_torch", fake_run_grid)
    monkeypatch.setattr(orch, "run_cross_dataset_hybrid_inference", fake_cross)
    monkeypatch.setattr(orch, "_write_gt_recon_from_external_npz", fake_gt)
    monkeypatch.setattr(orch, "aggregate_metrics_visuals_stage", lambda **kwargs: {"ok": True})
    monkeypatch.setattr(
        orch,
        "_build_cached_external_bundle",
        lambda **kwargs: {"test_npz": str(Path(kwargs["output_dir"]) / "datasets" / "N128" / "gs1" / "test.npz")},
    )

    orch.run_nersc_scan807_cameraman_study(
        scan807_dp=scan_dp,
        scan807_para=scan_para,
        cameraman_dp=cam_dp,
        cameraman_para=cam_para,
        ptychovit_checkpoint=checkpoint,
        output_dir=tmp_path / "study_out",
        half="top",
        seed=3,
        ptychovit_repo=tmp_path / "repo",
    )

    assert str(captured["dataset_npzs"]["scan807"]).endswith("/datasets/N128/gs1/test.npz")
    assert str(captured["dataset_npzs"]["cameraman256"]).endswith("/datasets/N128/gs1/test.npz")
    assert captured["allow_oom_fallback"] is False


def test_full_orchestration_manifest_serializes_numpy_scalars(monkeypatch, tmp_path):
    from scripts.studies import nersc_orchestration as orch

    checkpoint = tmp_path / "best_model.pth"
    checkpoint.write_bytes(b"ckpt")
    scan_dp = tmp_path / "scan807_dp.hdf5"
    scan_para = tmp_path / "scan807_para.hdf5"
    cam_dp = tmp_path / "cameraman_dp.hdf5"
    cam_para = tmp_path / "cameraman_para.hdf5"
    _touch_pair(scan_dp, scan_para)
    _touch_pair(cam_dp, cam_para)

    monkeypatch.setattr(orch, "materialize_pair_working_copy", lambda dp, para, out_dir: (dp, para))
    monkeypatch.setattr(
        orch,
        "run_ptychovit_inference_stage",
        lambda **kwargs: {
            name: {
                "recon_npz": str(
                    Path(kwargs["output_dir"]) / name / "recons" / "pinn_ptychovit" / "recon.npz"
                )
            }
            for name in ("scan807", "cameraman256")
        },
    )
    monkeypatch.setattr(
        orch,
        "prepare_hybrid_dataset",
        lambda **kwargs: {
            "train_npz": str(tmp_path / "train_raw.npz"),
            "test_npz": str(tmp_path / "test_raw.npz"),
            "downsampled_npz": str(tmp_path / "down_raw.npz"),
        },
    )
    monkeypatch.setattr(
        orch,
        "_convert_pair_to_downsampled_external_npz",
        lambda **kwargs: tmp_path / "scan_test_raw.npz",
    )

    def fake_build_datasets(**kwargs):
        base = Path(kwargs["cfg"].output_dir) / "datasets" / "N128" / "gs1"
        base.mkdir(parents=True, exist_ok=True)
        train = base / "train.npz"
        test = base / "test.npz"
        train.write_bytes(b"npz")
        test.write_bytes(b"npz")
        return {128: {"train_npz": str(train), "test_npz": str(test), "gt_recon": str(base / "gt.npz")}}

    monkeypatch.setattr(orch, "build_datasets", fake_build_datasets)
    monkeypatch.setattr(
        orch,
        "run_grid_lines_torch",
        lambda cfg: {
            "run_dir": str((Path(cfg.output_dir) / "runs" / "pinn_hybrid_resnet").resolve())
        },
    )
    monkeypatch.setattr(
        orch,
        "run_cross_dataset_hybrid_inference",
        lambda **kwargs: {
            "scan807": {
                "recon_npz": str(Path(kwargs["output_dir"]) / "scan807" / "recons" / "pinn_hybrid_resnet" / "recon.npz")
            },
            "cameraman256": {
                "recon_npz": str(
                    Path(kwargs["output_dir"]) / "cameraman256" / "recons" / "pinn_hybrid_resnet" / "recon.npz"
                )
            },
        },
    )
    monkeypatch.setattr(
        orch,
        "_write_gt_recon_from_external_npz",
        lambda dataset_output_dir, external_npz: Path(dataset_output_dir) / "recons" / "gt" / "recon.npz",
    )
    monkeypatch.setattr(
        orch,
        "aggregate_metrics_visuals_stage",
        lambda **kwargs: {"legacy_metrics": {"amp_mae": np.float32(0.1)}},
    )
    monkeypatch.setattr(
        orch,
        "_build_cached_external_bundle",
        lambda **kwargs: {"test_npz": str(Path(kwargs["output_dir"]) / "datasets" / "N128" / "gs1" / "test.npz")},
    )

    out_dir = tmp_path / "study_out"
    run_dir = out_dir / "hybrid_training" / "runs" / "pinn_hybrid_resnet"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model.pt").write_bytes(b"weights")
    (tmp_path / "train_raw.npz").write_bytes(b"npz")
    (tmp_path / "test_raw.npz").write_bytes(b"npz")
    (tmp_path / "down_raw.npz").write_bytes(b"npz")
    (tmp_path / "scan_test_raw.npz").write_bytes(b"npz")

    manifest = orch.run_nersc_scan807_cameraman_study(
        scan807_dp=scan_dp,
        scan807_para=scan_para,
        cameraman_dp=cam_dp,
        cameraman_para=cam_para,
        ptychovit_checkpoint=checkpoint,
        output_dir=out_dir,
        half="top",
        seed=3,
        ptychovit_repo=tmp_path / "repo",
    )
    assert (out_dir / "manifest.json").exists()
    assert manifest["metrics_outputs"]["scan807"]["legacy_metrics"]["amp_mae"] == np.float32(0.1)


def test_convert_pair_to_downsampled_external_npz_applies_flipped_policy(monkeypatch, tmp_path):
    from scripts.studies import nersc_orchestration as orch

    dp_h5 = tmp_path / "scan_dp.hdf5"
    para_h5 = tmp_path / "scan_para.hdf5"
    _touch_pair(dp_h5, para_h5)

    canonical = tmp_path / "canonical.npz"
    diffraction = np.arange(1, 37, dtype=np.float32).reshape(1, 6, 6)
    object_guess = np.arange(1, 122, dtype=np.float32).reshape(11, 11).astype(np.complex64)
    probe_guess = np.arange(1, 37, dtype=np.float32).reshape(6, 6).astype(np.complex64)
    np.savez_compressed(
        canonical,
        diffraction=diffraction,
        objectGuess=object_guess,
        probeGuess=probe_guess,
        xcoords=np.array([0.0], dtype=np.float64),
        ycoords=np.array([2.0], dtype=np.float64),
        xcoords_start=np.array([0.0], dtype=np.float64),
        ycoords_start=np.array([2.0], dtype=np.float64),
    )

    monkeypatch.setattr(orch, "materialize_pair_working_copy", lambda dp, para, out_dir: (dp, para))
    monkeypatch.setattr(orch, "pair_to_external_npz", lambda dp, para, out_npz: canonical)

    out_npz = orch._convert_pair_to_downsampled_external_npz(
        dp_h5=dp_h5,
        para_h5=para_h5,
        out_npz=tmp_path / "scan_downsampled.npz",
        work_dir=tmp_path / "work",
        target_n=3,
    )

    with np.load(out_npz, allow_pickle=True) as converted:
        expected_diff = np.sqrt((diffraction.reshape(1, 3, 2, 3, 2) ** 2).sum(axis=(2, 4))).astype(
            np.float32
        )
        assert np.allclose(converted["diffraction"], expected_diff)
        assert np.array_equal(converted["objectGuess"], object_guess[3:8, 3:8])
        assert np.array_equal(converted["probeGuess"], probe_guess[1:4, 1:4])
        assert np.array_equal(converted["xcoords"], np.array([0.0], dtype=np.float64))
        assert np.array_equal(converted["ycoords"], np.array([2.0], dtype=np.float64))


def test_full_orchestration_threads_downsample_policy_to_prep_and_scan_convert(monkeypatch, tmp_path):
    from scripts.studies import nersc_orchestration as orch

    checkpoint = tmp_path / "best_model.pth"
    checkpoint.write_bytes(b"ckpt")
    scan_dp = tmp_path / "scan807_dp.hdf5"
    scan_para = tmp_path / "scan807_para.hdf5"
    cam_dp = tmp_path / "cameraman_dp.hdf5"
    cam_para = tmp_path / "cameraman_para.hdf5"
    _touch_pair(scan_dp, scan_para)
    _touch_pair(cam_dp, cam_para)

    captured = {"prepare_policy": None, "scan_policy": None}

    monkeypatch.setattr(orch, "materialize_pair_working_copy", lambda dp, para, out_dir: (dp, para))
    monkeypatch.setattr(
        orch,
        "run_ptychovit_inference_stage",
        lambda **kwargs: {
            "scan807": {"recon_npz": str(tmp_path / "scan_recon.npz")},
            "cameraman256": {"recon_npz": str(tmp_path / "cam_recon.npz")},
        },
    )

    def fake_prepare(**kwargs):
        captured["prepare_policy"] = kwargs.get("downsample_policy")
        train = tmp_path / "train_raw.npz"
        test = tmp_path / "test_raw.npz"
        down = tmp_path / "down_raw.npz"
        for path in (train, test, down):
            path.write_bytes(b"npz")
        return {"train_npz": str(train), "test_npz": str(test), "downsampled_npz": str(down)}

    def fake_convert(**kwargs):
        captured["scan_policy"] = kwargs.get("downsample_policy")
        out = tmp_path / "scan_test_raw.npz"
        out.write_bytes(b"npz")
        return out

    monkeypatch.setattr(orch, "prepare_hybrid_dataset", fake_prepare)
    monkeypatch.setattr(orch, "_convert_pair_to_downsampled_external_npz", fake_convert)
    monkeypatch.setattr(
        orch,
        "build_datasets",
        lambda **kwargs: {
            128: {
                "train_npz": str(tmp_path / "cached_train.npz"),
                "test_npz": str(tmp_path / "cached_test.npz"),
                "gt_recon": str(tmp_path / "gt.npz"),
            }
        },
    )
    monkeypatch.setattr(
        orch,
        "run_grid_lines_torch",
        lambda cfg: {"run_dir": str((Path(cfg.output_dir) / "runs" / "pinn_hybrid_resnet").resolve())},
    )
    monkeypatch.setattr(
        orch,
        "run_cross_dataset_hybrid_inference",
        lambda **kwargs: {
            "scan807": {"recon_npz": str(tmp_path / "scan_hybrid_recon.npz")},
            "cameraman256": {"recon_npz": str(tmp_path / "cam_hybrid_recon.npz")},
        },
    )
    monkeypatch.setattr(
        orch,
        "_build_cached_external_bundle",
        lambda **kwargs: {"test_npz": str(tmp_path / "cached_test.npz")},
    )
    monkeypatch.setattr(
        orch,
        "_write_gt_recon_from_external_npz",
        lambda dataset_output_dir, external_npz: tmp_path / "gt_recon.npz",
    )
    monkeypatch.setattr(orch, "aggregate_metrics_visuals_stage", lambda **kwargs: {"ok": True})

    run_dir = tmp_path / "out" / "hybrid_training" / "runs" / "pinn_hybrid_resnet"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model.pt").write_bytes(b"weights")

    orch.run_nersc_scan807_cameraman_study(
        scan807_dp=scan_dp,
        scan807_para=scan_para,
        cameraman_dp=cam_dp,
        cameraman_para=cam_para,
        ptychovit_checkpoint=checkpoint,
        output_dir=tmp_path / "out",
        half="top",
        seed=3,
        ptychovit_repo=tmp_path / "repo",
        downsample_policy="crop-bin",
    )
    assert captured["prepare_policy"] == "crop-bin"
    assert captured["scan_policy"] == "crop-bin"
