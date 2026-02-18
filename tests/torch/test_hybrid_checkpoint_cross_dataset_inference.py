from pathlib import Path
import json

import numpy as np


def _write_min_npz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 2
    N = 128
    np.savez(
        path,
        diffraction=np.ones((n, N, N), dtype=np.float32),
        coords_nominal=np.zeros((n, 1, 2, 1), dtype=np.float32),
        coords_offsets=np.zeros((n, 1, 2, 1), dtype=np.float32),
        probeGuess=np.ones((N, N), dtype=np.complex64),
        YY_ground_truth=np.ones((N, N), dtype=np.complex64),
    )


def test_cross_dataset_inference_loads_single_model_pt_and_runs_two_test_npzs(
    monkeypatch, tmp_path
):
    from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig
    from scripts.studies.hybrid_checkpoint_inference import run_cross_dataset_hybrid_inference

    model_pt = tmp_path / "model.pt"
    model_pt.write_bytes(b"weights")

    scan_npz = tmp_path / "scan807_test.npz"
    cam_npz = tmp_path / "cameraman_test.npz"
    _write_min_npz(scan_npz)
    _write_min_npz(cam_npz)

    cfg = TorchRunnerConfig(
        train_npz=scan_npz,
        test_npz=scan_npz,
        output_dir=tmp_path / "run",
        architecture="hybrid_resnet",
        N=128,
        gridsize=1,
        reassembly_mode="position",
        position_reassembly_backend="shift_sum",
    )

    load_calls = {"count": 0}
    infer_calls: list[str] = []

    def fake_load_model(config, checkpoint_path):
        _ = (config, checkpoint_path)
        load_calls["count"] += 1
        return object()

    def fake_run_single_dataset(model, config, dataset_name, test_npz, recon_npz, allow_oom_fallback):
        _ = (model, config, test_npz, allow_oom_fallback)
        infer_calls.append(dataset_name)
        recon_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(recon_npz, YY_pred=np.ones((16, 16), dtype=np.complex64))
        return recon_npz

    monkeypatch.setattr(
        "scripts.studies.hybrid_checkpoint_inference._load_model_from_checkpoint",
        fake_load_model,
    )
    monkeypatch.setattr(
        "scripts.studies.hybrid_checkpoint_inference._run_single_dataset_inference",
        fake_run_single_dataset,
    )

    run_cross_dataset_hybrid_inference(
        model_pt=model_pt,
        dataset_npzs={"scan807": scan_npz, "cameraman256": cam_npz},
        output_dir=tmp_path / "outputs",
        base_cfg=cfg,
    )

    assert load_calls["count"] == 1
    assert set(infer_calls) == {"scan807", "cameraman256"}


def test_cross_dataset_inference_writes_recon_npz_for_each_dataset(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig
    from scripts.studies.hybrid_checkpoint_inference import run_cross_dataset_hybrid_inference

    model_pt = tmp_path / "model.pt"
    model_pt.write_bytes(b"weights")

    scan_npz = tmp_path / "scan807_test.npz"
    cam_npz = tmp_path / "cameraman_test.npz"
    _write_min_npz(scan_npz)
    _write_min_npz(cam_npz)

    cfg = TorchRunnerConfig(
        train_npz=scan_npz,
        test_npz=scan_npz,
        output_dir=tmp_path / "run",
        architecture="hybrid_resnet",
        N=128,
        gridsize=1,
        reassembly_mode="position",
        position_reassembly_backend="shift_sum",
    )

    monkeypatch.setattr(
        "scripts.studies.hybrid_checkpoint_inference._load_model_from_checkpoint",
        lambda config, checkpoint_path: object(),
    )

    def fake_run_single_dataset(model, config, dataset_name, test_npz, recon_npz, allow_oom_fallback):
        _ = (model, config, dataset_name, test_npz, allow_oom_fallback)
        recon_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(recon_npz, YY_pred=np.ones((16, 16), dtype=np.complex64))
        return recon_npz

    monkeypatch.setattr(
        "scripts.studies.hybrid_checkpoint_inference._run_single_dataset_inference",
        fake_run_single_dataset,
    )

    result = run_cross_dataset_hybrid_inference(
        model_pt=model_pt,
        dataset_npzs={"scan807": scan_npz, "cameraman256": cam_npz},
        output_dir=tmp_path / "outputs",
        base_cfg=cfg,
    )

    assert Path(result["scan807"]["recon_npz"]).exists()
    assert Path(result["cameraman256"]["recon_npz"]).exists()


def test_cross_dataset_inference_allows_auto_backend(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig
    from scripts.studies.hybrid_checkpoint_inference import run_cross_dataset_hybrid_inference

    model_pt = tmp_path / "model.pt"
    model_pt.write_bytes(b"weights")
    scan_npz = tmp_path / "scan807_test.npz"
    _write_min_npz(scan_npz)

    cfg = TorchRunnerConfig(
        train_npz=scan_npz,
        test_npz=scan_npz,
        output_dir=tmp_path / "run",
        architecture="hybrid_resnet",
        N=128,
        gridsize=1,
        reassembly_mode="position",
        position_reassembly_backend="auto",
    )

    monkeypatch.setattr(
        "scripts.studies.hybrid_checkpoint_inference._load_model_from_checkpoint",
        lambda config, checkpoint_path: object(),
    )

    def fake_run_single_dataset(model, config, dataset_name, test_npz, recon_npz, allow_oom_fallback):
        _ = (model, config, dataset_name, test_npz, allow_oom_fallback)
        recon_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(recon_npz, YY_pred=np.ones((16, 16), dtype=np.complex64))
        return recon_npz

    monkeypatch.setattr(
        "scripts.studies.hybrid_checkpoint_inference._run_single_dataset_inference",
        fake_run_single_dataset,
    )

    result = run_cross_dataset_hybrid_inference(
        model_pt=model_pt,
        dataset_npzs={"scan807": scan_npz},
        output_dir=tmp_path / "outputs",
        base_cfg=cfg,
    )
    assert Path(result["scan807"]["recon_npz"]).exists()
    manifest = json.loads((tmp_path / "outputs" / "hybrid_cross_dataset_manifest.json").read_text())
    assert manifest["position_reassembly_backend"] == "auto"
    assert manifest["allow_oom_fallback"] is True


def test_cross_dataset_inference_propagates_allow_oom_fallback(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig
    from scripts.studies.hybrid_checkpoint_inference import run_cross_dataset_hybrid_inference

    model_pt = tmp_path / "model.pt"
    model_pt.write_bytes(b"weights")
    scan_npz = tmp_path / "scan807_test.npz"
    _write_min_npz(scan_npz)

    cfg = TorchRunnerConfig(
        train_npz=scan_npz,
        test_npz=scan_npz,
        output_dir=tmp_path / "run",
        architecture="hybrid_resnet",
        N=128,
        gridsize=1,
        reassembly_mode="position",
        position_reassembly_backend="shift_sum",
    )

    monkeypatch.setattr(
        "scripts.studies.hybrid_checkpoint_inference._load_model_from_checkpoint",
        lambda config, checkpoint_path: object(),
    )

    calls: list[bool] = []

    def fake_run_single_dataset(
        model,
        config,
        dataset_name,
        test_npz,
        recon_npz,
        allow_oom_fallback,
    ):
        _ = (model, config, dataset_name, test_npz)
        calls.append(bool(allow_oom_fallback))
        recon_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(recon_npz, YY_pred=np.ones((16, 16), dtype=np.complex64))
        return recon_npz

    monkeypatch.setattr(
        "scripts.studies.hybrid_checkpoint_inference._run_single_dataset_inference",
        fake_run_single_dataset,
    )

    _ = run_cross_dataset_hybrid_inference(
        model_pt=model_pt,
        dataset_npzs={"scan807": scan_npz},
        output_dir=tmp_path / "outputs",
        base_cfg=cfg,
        allow_oom_fallback=False,
    )

    assert calls == [False]


def test_cross_dataset_inference_rejects_unknown_backend(tmp_path):
    import pytest
    from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig
    from scripts.studies.hybrid_checkpoint_inference import run_cross_dataset_hybrid_inference

    model_pt = tmp_path / "model.pt"
    model_pt.write_bytes(b"weights")
    scan_npz = tmp_path / "scan807_test.npz"
    _write_min_npz(scan_npz)

    cfg = TorchRunnerConfig(
        train_npz=scan_npz,
        test_npz=scan_npz,
        output_dir=tmp_path / "run",
        architecture="hybrid_resnet",
        N=128,
        gridsize=1,
        reassembly_mode="position",
        position_reassembly_backend="not_a_backend",
    )

    with pytest.raises(ValueError, match="Unsupported position reassembly backend"):
        run_cross_dataset_hybrid_inference(
            model_pt=model_pt,
            dataset_npzs={"scan807": scan_npz},
            output_dir=tmp_path / "outputs",
            base_cfg=cfg,
        )


def test_build_model_calls_update_legacy_before_generator_build(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig
    from scripts.studies.hybrid_checkpoint_inference import _build_model_for_config

    scan_npz = tmp_path / "scan807_test.npz"
    _write_min_npz(scan_npz)
    events: list[str] = []

    def fake_update_legacy_dict(cfg_dict, config):
        _ = (cfg_dict, config)
        events.append("update_legacy_dict")

    class FakeGenerator:
        def build_model(self, pt_configs):
            _ = pt_configs
            assert "update_legacy_dict" in events
            events.append("build_model")

            class FakeModel:
                def load_state_dict(self, state_dict, strict=False):
                    _ = (state_dict, strict)

                def eval(self):
                    return self

            return FakeModel()

    monkeypatch.setattr("ptycho.config.config.update_legacy_dict", fake_update_legacy_dict)
    monkeypatch.setattr(
        "scripts.studies.hybrid_checkpoint_inference.resolve_generator",
        lambda config: FakeGenerator(),
    )

    cfg = TorchRunnerConfig(
        train_npz=scan_npz,
        test_npz=scan_npz,
        output_dir=tmp_path / "run",
        architecture="hybrid_resnet",
        N=128,
        gridsize=1,
        reassembly_mode="position",
        position_reassembly_backend="shift_sum",
    )

    _ = _build_model_for_config(cfg)
    assert events[0] == "update_legacy_dict"
    assert "build_model" in events
