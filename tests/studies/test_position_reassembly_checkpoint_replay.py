from pathlib import Path
import numpy as np


def _write_recon(path: Path, yy_pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    yy = np.asarray(yy_pred, dtype=np.complex64)
    np.savez(path, YY_pred=yy, amp=np.abs(yy).astype(np.float32), phase=np.angle(yy).astype(np.float32))


def test_summarize_backend_parity_passes_for_matching_recons(tmp_path):
    from scripts.studies.position_reassembly_checkpoint_replay import summarize_backend_parity

    shift = np.ones((16, 16), dtype=np.complex64)
    batched = shift * (1.0 + 1e-6)

    shift_path = tmp_path / "shift_sum" / "recon.npz"
    batched_path = tmp_path / "batched" / "recon.npz"
    _write_recon(shift_path, shift)
    _write_recon(batched_path, batched)

    summary = summarize_backend_parity(
        shift_sum_recon=shift_path,
        batched_recon=batched_path,
        out_json=tmp_path / "summary.json",
        atol=1e-4,
        rtol=1e-3,
    )

    assert summary["parity_pass"] is True
    assert summary["diff_metrics"]["complex_allclose"] is True
    assert (tmp_path / "summary.json").exists()


def test_summarize_backend_parity_flags_large_scale_divergence(tmp_path):
    from scripts.studies.position_reassembly_checkpoint_replay import summarize_backend_parity

    shift = np.ones((16, 16), dtype=np.complex64)
    batched = np.full((16, 16), 8.0 + 0j, dtype=np.complex64)

    shift_path = tmp_path / "shift_sum" / "recon.npz"
    batched_path = tmp_path / "batched" / "recon.npz"
    _write_recon(shift_path, shift)
    _write_recon(batched_path, batched)

    summary = summarize_backend_parity(
        shift_sum_recon=shift_path,
        batched_recon=batched_path,
        out_json=tmp_path / "summary.json",
        atol=1e-6,
        rtol=1e-3,
    )

    assert summary["parity_pass"] is False
    assert summary["diff_metrics"]["amp_max_rel_error"] > 0.5


def test_checkpoint_replay_runs_both_backends_and_writes_summary(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig
    from scripts.studies.position_reassembly_checkpoint_replay import run_checkpoint_replay_parity

    model_pt = tmp_path / "model.pt"
    model_pt.write_bytes(b"weights")
    train_npz = tmp_path / "train.npz"
    test_npz = tmp_path / "test.npz"
    np.savez(train_npz, dummy=np.array([1]))
    np.savez(test_npz, dummy=np.array([1]))

    called_backends: list[str] = []

    def fake_cross_dataset(model_pt, dataset_npzs, output_dir, base_cfg):
        _ = (model_pt, dataset_npzs)
        called_backends.append(base_cfg.position_reassembly_backend)
        recon_path = Path(output_dir) / "replay" / "recons" / "pinn_hybrid_resnet" / "recon.npz"
        value = 1.0 if base_cfg.position_reassembly_backend == "shift_sum" else 8.0
        _write_recon(recon_path, np.full((16, 16), value + 0j, dtype=np.complex64))
        return {"replay": {"recon_npz": str(recon_path), "test_npz": str(test_npz)}}

    monkeypatch.setattr(
        "scripts.studies.position_reassembly_checkpoint_replay.run_cross_dataset_hybrid_inference",
        fake_cross_dataset,
    )

    base_cfg = TorchRunnerConfig(
        train_npz=train_npz,
        test_npz=test_npz,
        output_dir=tmp_path / "run",
        architecture="hybrid_resnet",
        N=128,
        gridsize=1,
        reassembly_mode="position",
        position_reassembly_backend="shift_sum",
    )
    manifest = run_checkpoint_replay_parity(
        model_pt=model_pt,
        dataset_name="replay",
        train_npz=train_npz,
        test_npz=test_npz,
        output_dir=tmp_path / "parity",
        base_cfg=base_cfg,
        atol=1e-6,
        rtol=1e-3,
    )

    assert called_backends == ["shift_sum", "batched"]
    assert (tmp_path / "parity" / "summary.json").exists()
    assert Path(manifest["summary_json"]).exists()
