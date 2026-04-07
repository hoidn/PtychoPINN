from pathlib import Path


def test_bridge_entrypoint_writes_invocation_artifacts(monkeypatch, tmp_path):
    from scripts.studies import ptychovit_bridge_entrypoint as bridge

    repo = tmp_path / "ptychovit_repo"
    repo.mkdir()
    (repo / "config.yaml").write_text("data: {}\ntraining: {}\n")
    dp = tmp_path / "test_dp.hdf5"
    para = tmp_path / "test_para.hdf5"
    dp.write_bytes(b"dp")
    para.write_bytes(b"para")
    output_dir = tmp_path / "bridge_out"
    checkpoint = tmp_path / "best_model.pth"
    checkpoint.write_bytes(b"ckpt")

    runtime_cfg_path = tmp_path / "runtime_config.yaml"
    runtime_cfg_path.write_text("data: {}\n")
    runtime_data_dir = tmp_path / "runtime_data"
    runtime_data_dir.mkdir()
    (runtime_data_dir / "train_dp.hdf5").write_bytes(b"train")
    (runtime_data_dir / "test_dp.hdf5").write_bytes(b"test")

    def fake_prepare(args):
        _ = args
        return runtime_cfg_path, {
            "data": {
                "data_path": str(runtime_data_dir),
                "test_path": str(runtime_data_dir / "test_dp.hdf5"),
            }
        }

    def fake_infer(args, checkpoint_path):
        _ = checkpoint_path
        args.recon_npz.parent.mkdir(parents=True, exist_ok=True)
        args.recon_npz.write_bytes(b"npz")
        return args.recon_npz

    monkeypatch.setattr(bridge, "_prepare_runtime_training_config", fake_prepare)
    monkeypatch.setattr(bridge, "_load_checkpoint_path", lambda args: checkpoint)
    monkeypatch.setattr(bridge, "_run_model_inference", fake_infer)

    rc = bridge.main(
        [
            "--ptychovit-repo",
            str(repo),
            "--train-dp",
            str(dp),
            "--test-dp",
            str(dp),
            "--train-para",
            str(para),
            "--test-para",
            str(para),
            "--mode",
            "inference",
            "--output-dir",
            str(output_dir),
            "--checkpoint",
            str(checkpoint),
        ]
    )
    assert rc == 0
    assert (output_dir / "invocation.json").exists()
    assert (output_dir / "invocation.sh").exists()
