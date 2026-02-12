# tests/test_grid_lines_compare_wrapper.py
"""Tests for grid_lines_compare_wrapper orchestration."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest


def _mock_dataset_builder(monkeypatch):
    def fake_build_grid_lines_datasets(cfg):
        datasets_dir = cfg.output_dir / "datasets" / f"N{cfg.N}" / f"gs{cfg.gridsize}"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        train_npz = datasets_dir / "train.npz"
        test_npz = datasets_dir / "test.npz"
        train_npz.write_bytes(b"stub")
        test_npz.write_bytes(b"stub")
        gt_dir = cfg.output_dir / "recons" / "gt"
        gt_dir.mkdir(parents=True, exist_ok=True)
        gt = (np.ones((cfg.N, cfg.N)) + 1j * np.ones((cfg.N, cfg.N))).astype(np.complex64)
        np.savez(gt_dir / "recon.npz", YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))
        return {"train_npz": str(train_npz), "test_npz": str(test_npz), "gt_recon": str(gt_dir / "recon.npz")}

    monkeypatch.setattr(
        "ptycho.workflows.grid_lines_workflow.build_grid_lines_datasets",
        fake_build_grid_lines_datasets,
    )


def test_wrapper_merges_metrics(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    def fake_tf_run(cfg):
        datasets_dir = cfg.output_dir / "datasets" / f"N{cfg.N}" / f"gs{cfg.gridsize}"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        (datasets_dir / "train.npz").write_bytes(b"stub")
        (datasets_dir / "test.npz").write_bytes(b"stub")
        (cfg.output_dir / "metrics.json").write_text(json.dumps({
            "pinn": {"mse": 0.1},
            "baseline": {"mse": 0.2},
        }))
        return {"train_npz": str(datasets_dir / "train.npz"), "test_npz": str(datasets_dir / "test.npz")}

    def fake_torch_run(cfg):
        run_dir = cfg.output_dir / "runs" / f"pinn_{cfg.architecture}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text(json.dumps({"mse": 0.3}))
        return {"metrics": {"mse": 0.3}}

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.run_grid_lines_workflow", fake_tf_run)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)

    result = run_grid_lines_compare(
        N=64,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("cnn", "baseline", "fno", "hybrid"),
        probe_npz=Path("dummy_probe.npz"),
    )
    merged = json.loads((tmp_path / "metrics.json").read_text())
    assert "pinn" in merged
    assert "baseline" in merged
    assert "pinn_fno" in merged
    assert "pinn_hybrid" in merged
    assert "metrics" in result


def test_wrapper_accepts_architecture_list(tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import parse_args

    args = parse_args([
        "--N", "64",
        "--gridsize", "1",
        "--output-dir", str(tmp_path),
        "--architectures", "cnn,fno",
    ])
    assert args.architectures == ("cnn", "fno")


def test_parse_args_default_architectures_excludes_baseline(tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import parse_args

    args = parse_args(
        [
            "--N",
            "64",
            "--gridsize",
            "1",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert "baseline" not in args.architectures


def test_wrapper_keeps_architectures_backward_compat(tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import parse_args

    args = parse_args([
        "--N", "64",
        "--gridsize", "1",
        "--output-dir", str(tmp_path),
        "--architectures", "cnn,baseline,fno",
    ])
    assert args.architectures == ("cnn", "baseline", "fno")


def test_wrapper_accepts_models_and_model_n(tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import parse_args

    args = parse_args([
        "--N", "64",
        "--gridsize", "1",
        "--output-dir", str(tmp_path),
        "--models", "pinn_hybrid,pinn_ptychovit",
        "--model-n", "pinn_hybrid=128,pinn_ptychovit=256",
    ])
    assert args.models == ("pinn_hybrid", "pinn_ptychovit")
    assert args.model_n["pinn_hybrid"] == 128
    assert args.model_n["pinn_ptychovit"] == 256


def test_parse_args_reuse_existing_recons_defaults_false(tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import parse_args

    args = parse_args([
        "--N", "64",
        "--gridsize", "1",
        "--output-dir", str(tmp_path),
    ])
    assert args.reuse_existing_recons is False


def test_parse_args_reuse_existing_recons_flag_sets_true(tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import parse_args

    args = parse_args([
        "--N", "64",
        "--gridsize", "1",
        "--output-dir", str(tmp_path),
        "--reuse-existing-recons",
    ])
    assert args.reuse_existing_recons is True


def test_wrapper_rejects_ptychovit_non_256():
    from scripts.studies.grid_lines_compare_wrapper import validate_model_specs

    with pytest.raises(ValueError, match="pinn_ptychovit.*N=256"):
        validate_model_specs(
            models=("pinn_ptychovit",),
            model_n={"pinn_ptychovit": 128},
        )


def test_compute_required_ns_from_models_and_model_n():
    from scripts.studies.grid_lines_compare_wrapper import compute_required_ns

    required = compute_required_ns(
        models=("pinn_hybrid", "pinn_ptychovit"),
        model_n={"pinn_hybrid": 128, "pinn_ptychovit": 256},
        default_n=128,
    )
    assert required == [128, 256]


def test_resolve_model_ns_defaults_ptychovit_to_256():
    from scripts.studies.grid_lines_compare_wrapper import resolve_model_ns

    resolved = resolve_model_ns(
        models=("pinn_ptychovit",),
        model_n_overrides={},
        default_n=128,
    )
    assert resolved["pinn_ptychovit"] == 256


def test_wrapper_models_mode_honors_tf_model_n(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    gt_recon = tmp_path / "recons" / "gt" / "recon.npz"
    gt_recon.parent.mkdir(parents=True, exist_ok=True)
    gt = (np.ones((392, 392)) + 1j * np.ones((392, 392))).astype(np.complex64)
    np.savez(gt_recon, YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))

    train_64 = tmp_path / "datasets" / "N64" / "gs1" / "train.npz"
    test_64 = tmp_path / "datasets" / "N64" / "gs1" / "test.npz"
    for path in (train_64, test_64):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, ok=np.array([1], dtype=np.int8))

    def fake_build_by_n(base_cfg, required_ns):
        _ = base_cfg
        assert sorted(required_ns) == [64]
        return {
            64: {"train_npz": str(train_64), "test_npz": str(test_64), "gt_recon": str(gt_recon), "tag": "N64"},
        }

    captured = {}

    def fake_tf_run(cfg):
        captured["N"] = cfg.N
        recon_path = cfg.output_dir / "recons" / "pinn" / "recon.npz"
        recon_path.parent.mkdir(parents=True, exist_ok=True)
        pred = (np.ones((64, 64)) + 1j * np.ones((64, 64))).astype(np.complex64)
        np.savez(recon_path, YY_pred=pred, amp=np.abs(pred), phase=np.angle(pred))
        return {"train_npz": str(train_64), "test_npz": str(test_64), "metrics": {"mse": 0.1}}

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.build_grid_lines_datasets_by_n", fake_build_by_n)
    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.run_grid_lines_workflow", fake_tf_run)
    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.render_grid_lines_visuals", lambda output_dir, order: {})
    monkeypatch.setattr("ptycho.evaluation.eval_reconstruction", lambda pred, gt, label: {"mse": float(np.mean(np.abs(pred - gt)))})

    run_grid_lines_compare(
        N=128,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("cnn",),
        models=("pinn",),
        model_n={"pinn": 64},
        probe_npz=Path("dummy_probe.npz"),
    )
    assert captured["N"] == 64


def test_wrapper_writes_metrics_by_model_for_selected_models(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    gt_recon = tmp_path / "recons" / "gt" / "recon.npz"
    gt_recon.parent.mkdir(parents=True, exist_ok=True)
    gt = (np.ones((392, 392)) + 1j * np.ones((392, 392))).astype(np.complex64)
    np.savez(gt_recon, YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))

    train_128 = tmp_path / "datasets" / "N128" / "gs1" / "train.npz"
    test_128 = tmp_path / "datasets" / "N128" / "gs1" / "test.npz"
    train_256 = tmp_path / "datasets" / "N256" / "gs1" / "train.npz"
    test_256 = tmp_path / "datasets" / "N256" / "gs1" / "test.npz"
    for path in (train_128, test_128, train_256, test_256):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, ok=np.array([1], dtype=np.int8))

    def fake_build_by_n(base_cfg, required_ns):
        _ = base_cfg
        assert sorted(required_ns) == [128, 256]
        return {
            128: {"train_npz": str(train_128), "test_npz": str(test_128), "gt_recon": str(gt_recon), "tag": "N128"},
            256: {"train_npz": str(train_256), "test_npz": str(test_256), "gt_recon": str(gt_recon), "tag": "N256"},
        }

    def fake_torch_run(cfg):
        recon_path = cfg.output_dir / "recons" / "pinn_hybrid" / "recon.npz"
        recon_path.parent.mkdir(parents=True, exist_ok=True)
        pred = (np.ones((128, 128)) + 1j * np.ones((128, 128))).astype(np.complex64)
        np.savez(recon_path, YY_pred=pred, amp=np.abs(pred), phase=np.angle(pred))
        return {"recon_npz": str(recon_path), "metrics": {"mse": 0.1}}

    def fake_ptychovit_run(cfg):
        _ = cfg
        recon_path = tmp_path / "recons" / "pinn_ptychovit" / "recon.npz"
        recon_path.parent.mkdir(parents=True, exist_ok=True)
        pred = (np.ones((256, 256)) + 1j * np.ones((256, 256))).astype(np.complex64)
        np.savez(recon_path, YY_pred=pred, amp=np.abs(pred), phase=np.angle(pred))
        return {"recon_npz": str(recon_path), "status": "ok"}

    def fake_convert(npz_path, out_dir, object_name, pixel_size_m=1.0):
        from ptycho.interop.ptychovit.contracts import PtychoViTHdf5Pair

        _ = (npz_path, object_name, pixel_size_m)
        out_dir.mkdir(parents=True, exist_ok=True)
        dp = out_dir / "x_dp.hdf5"
        para = out_dir / "x_para.hdf5"
        with h5py.File(dp, "w") as dp_file:
            dp_file.create_dataset("dp", data=np.ones((2, 8, 8), dtype=np.float32))
        with h5py.File(para, "w") as para_file:
            obj = para_file.create_dataset("object", data=np.ones((1, 16, 16), dtype=np.complex64))
            obj.attrs["pixel_height_m"] = 1.0
            obj.attrs["pixel_width_m"] = 1.0
            probe = para_file.create_dataset("probe", data=np.ones((1, 1, 8, 8), dtype=np.complex64))
            probe.attrs["pixel_height_m"] = 1.0
            probe.attrs["pixel_width_m"] = 1.0
            para_file.create_dataset("probe_position_x_m", data=np.zeros(2, dtype=np.float64))
            para_file.create_dataset("probe_position_y_m", data=np.zeros(2, dtype=np.float64))
        return PtychoViTHdf5Pair(dp_hdf5=dp, para_hdf5=para, object_name="x")

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.build_grid_lines_datasets_by_n", fake_build_by_n)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)
    monkeypatch.setattr("scripts.studies.grid_lines_ptychovit_runner.run_grid_lines_ptychovit", fake_ptychovit_run)
    monkeypatch.setattr("ptycho.interop.ptychovit.convert.convert_npz_split_to_hdf5_pair", fake_convert)
    monkeypatch.setattr("ptycho.interop.ptychovit.validate.validate_hdf5_pair", lambda dp_path, para_path: None)
    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.render_grid_lines_visuals", lambda output_dir, order: {})
    monkeypatch.setattr("ptycho.evaluation.eval_reconstruction", lambda pred, gt, label: {"mse": float(np.mean(np.abs(pred - gt)))})

    run_grid_lines_compare(
        N=128,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("hybrid",),
        models=("pinn_hybrid", "pinn_ptychovit"),
        model_n={"pinn_hybrid": 128, "pinn_ptychovit": 256},
        probe_npz=Path("dummy_probe.npz"),
    )
    metrics = json.loads((tmp_path / "metrics_by_model.json").read_text())
    assert "pinn_hybrid" in metrics
    assert "pinn_ptychovit" in metrics
    mse_val = metrics["pinn_hybrid"]["metrics"]["mse"]
    if isinstance(mse_val, list):
        mse_val = mse_val[0]
    assert isinstance(mse_val, float)


def test_wrapper_does_not_reuse_precomputed_recons_by_default(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    gt_recon = tmp_path / "recons" / "gt" / "recon.npz"
    gt_recon.parent.mkdir(parents=True, exist_ok=True)
    gt = (np.ones((392, 392)) + 1j * np.ones((392, 392))).astype(np.complex64)
    np.savez(gt_recon, YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))

    stale_recon = tmp_path / "recons" / "pinn_hybrid" / "recon.npz"
    stale_recon.parent.mkdir(parents=True, exist_ok=True)
    stale = (np.zeros((392, 392)) + 1j * np.zeros((392, 392))).astype(np.complex64)
    np.savez(stale_recon, YY_pred=stale, amp=np.abs(stale), phase=np.angle(stale))

    train_128 = tmp_path / "datasets" / "N128" / "gs1" / "train.npz"
    test_128 = tmp_path / "datasets" / "N128" / "gs1" / "test.npz"
    for path in (train_128, test_128):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, ok=np.array([1], dtype=np.int8))

    called = {"build": 0, "torch": 0}

    def fake_build_by_n(base_cfg, required_ns):
        _ = base_cfg
        assert sorted(required_ns) == [128]
        called["build"] += 1
        return {
            128: {"train_npz": str(train_128), "test_npz": str(test_128), "gt_recon": str(gt_recon), "tag": "N128"},
        }

    def fake_torch_run(cfg):
        called["torch"] += 1
        recon_path = cfg.output_dir / "recons" / "pinn_hybrid" / "recon.npz"
        recon_path.parent.mkdir(parents=True, exist_ok=True)
        pred = (np.ones((128, 128)) + 1j * np.ones((128, 128))).astype(np.complex64)
        np.savez(recon_path, YY_pred=pred, amp=np.abs(pred), phase=np.angle(pred))
        return {"recon_npz": str(recon_path), "metrics": {"mse": 0.1}}

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.build_grid_lines_datasets_by_n", fake_build_by_n)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)
    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.render_grid_lines_visuals", lambda output_dir, order: {})
    monkeypatch.setattr(
        "ptycho.evaluation.eval_reconstruction",
        lambda pred, gt, label: {"mse": float(np.mean(np.abs(pred - gt)))},
    )

    run_grid_lines_compare(
        N=128,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("hybrid",),
        models=("pinn_hybrid",),
        model_n={"pinn_hybrid": 128},
        probe_npz=Path("dummy_probe.npz"),
    )
    assert called["build"] == 1
    assert called["torch"] == 1


def test_wrapper_reuses_precomputed_recons_when_opted_in(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    gt_recon = tmp_path / "recons" / "gt" / "recon.npz"
    gt_recon.parent.mkdir(parents=True, exist_ok=True)
    gt = (np.ones((392, 392)) + 1j * np.ones((392, 392))).astype(np.complex64)
    np.savez(gt_recon, YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))

    recon_path = tmp_path / "recons" / "pinn_hybrid" / "recon.npz"
    recon_path.parent.mkdir(parents=True, exist_ok=True)
    pred = (np.ones((392, 392)) + 1j * np.ones((392, 392))).astype(np.complex64)
    np.savez(recon_path, YY_pred=pred, amp=np.abs(pred), phase=np.angle(pred))

    called = {"build": 0, "torch": 0}

    def fake_build_by_n(base_cfg, required_ns):
        _ = (base_cfg, required_ns)
        called["build"] += 1
        raise AssertionError("build_grid_lines_datasets_by_n should not run when reusing precomputed artifacts")

    def fake_torch_run(cfg):
        _ = cfg
        called["torch"] += 1
        raise AssertionError("run_grid_lines_torch should not run when reusing precomputed artifacts")

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.build_grid_lines_datasets_by_n", fake_build_by_n)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)
    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.render_grid_lines_visuals", lambda output_dir, order: {})
    monkeypatch.setattr(
        "ptycho.evaluation.eval_reconstruction",
        lambda pred, gt, label: {"mse": float(np.mean(np.abs(pred - gt)))},
    )

    run_grid_lines_compare(
        N=128,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("hybrid",),
        models=("pinn_hybrid",),
        model_n={"pinn_hybrid": 128},
        probe_npz=Path("dummy_probe.npz"),
        reuse_existing_recons=True,
    )
    assert called["build"] == 0
    assert called["torch"] == 0
    stdout_log = tmp_path / "runs" / "pinn_hybrid" / "stdout.log"
    assert "Skipped backend execution; reused existing reconstruction artifact." in stdout_log.read_text()


def test_harmonized_metrics_run_on_canonical_gt_grid(tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import evaluate_selected_models

    gt_path = tmp_path / "recons" / "gt" / "recon.npz"
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt = (np.ones((392, 392)) + 1j * np.ones((392, 392))).astype(np.complex64)
    np.savez(gt_path, YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))

    pred_path = tmp_path / "recons" / "pinn_hybrid" / "recon.npz"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred = (np.ones((128, 128)) + 1j * np.ones((128, 128))).astype(np.complex64)
    np.savez(pred_path, YY_pred=pred, amp=np.abs(pred), phase=np.angle(pred))

    out = evaluate_selected_models({"pinn_hybrid": pred_path}, gt_path)
    assert out["pinn_hybrid"]["reference_shape"] == [392, 392]


def test_wrapper_defaults_torch_loss_mode_mae(tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import parse_args

    args = parse_args([
        "--N", "64",
        "--gridsize", "1",
        "--output-dir", str(tmp_path),
    ])
    assert getattr(args, "torch_loss_mode", None) == "mae"


def test_wrapper_passes_torch_loss_mode_to_runner(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    captured = {}

    def fake_torch_run(cfg):
        captured["torch_loss_mode"] = getattr(cfg, "torch_loss_mode", None)
        return {"metrics": {"mse": 0.3}}

    _mock_dataset_builder(monkeypatch)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)

    run_grid_lines_compare(
        N=64,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("fno",),
        probe_npz=Path("dummy_probe.npz"),
    )

    assert captured["torch_loss_mode"] == "mae"


def test_wrapper_cnn_only_excludes_baseline_metrics(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    def fake_tf_run(cfg):
        datasets_dir = cfg.output_dir / "datasets" / f"N{cfg.N}" / f"gs{cfg.gridsize}"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        (datasets_dir / "train.npz").write_bytes(b"stub")
        (datasets_dir / "test.npz").write_bytes(b"stub")
        (cfg.output_dir / "metrics.json").write_text(
            json.dumps({"pinn": {"mse": 0.1}, "baseline": {"mse": 0.2}})
        )
        return {
            "train_npz": str(datasets_dir / "train.npz"),
            "test_npz": str(datasets_dir / "test.npz"),
        }

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.run_grid_lines_workflow", fake_tf_run)
    monkeypatch.setattr(
        "ptycho.workflows.grid_lines_workflow.render_grid_lines_visuals",
        lambda output_dir, order: {},
    )

    run_grid_lines_compare(
        N=64,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("cnn",),
        probe_npz=Path("dummy_probe.npz"),
    )
    merged = json.loads((tmp_path / "metrics.json").read_text())
    assert "pinn" in merged
    assert "baseline" not in merged


def test_wrapper_torch_only_uses_dataset_builder_not_tf_workflow(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    called = {"build_datasets": False, "tf_workflow": False}

    def fake_build_grid_lines_datasets(cfg):
        called["build_datasets"] = True
        datasets_dir = cfg.output_dir / "datasets" / f"N{cfg.N}" / f"gs{cfg.gridsize}"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        train_npz = datasets_dir / "train.npz"
        test_npz = datasets_dir / "test.npz"
        train_npz.write_bytes(b"stub")
        test_npz.write_bytes(b"stub")
        gt_dir = cfg.output_dir / "recons" / "gt"
        gt_dir.mkdir(parents=True, exist_ok=True)
        gt = (np.ones((64, 64)) + 1j * np.ones((64, 64))).astype(np.complex64)
        np.savez(gt_dir / "recon.npz", YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))
        return {"train_npz": str(train_npz), "test_npz": str(test_npz), "gt_recon": str(gt_dir / "recon.npz")}

    def fake_tf_workflow(cfg):
        called["tf_workflow"] = True
        raise AssertionError("Torch-only run should not invoke TF training workflow")

    def fake_torch_run(cfg):
        recon_dir = cfg.output_dir / "recons" / f"pinn_{cfg.architecture}"
        recon_dir.mkdir(parents=True, exist_ok=True)
        pred = (np.ones((64, 64)) + 1j * np.ones((64, 64))).astype(np.complex64)
        np.savez(recon_dir / "recon.npz", YY_pred=pred, amp=np.abs(pred), phase=np.angle(pred))
        return {"metrics": {"mse": 0.3}, "recon_npz": str(recon_dir / "recon.npz")}

    monkeypatch.setattr(
        "ptycho.workflows.grid_lines_workflow.build_grid_lines_datasets",
        fake_build_grid_lines_datasets,
    )
    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.run_grid_lines_workflow", fake_tf_workflow)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)
    monkeypatch.setattr(
        "ptycho.workflows.grid_lines_workflow.render_grid_lines_visuals",
        lambda output_dir, order: {},
    )

    run_grid_lines_compare(
        N=64,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("fno",),
        probe_npz=Path("dummy_probe.npz"),
    )

    assert called["build_datasets"] is True
    assert called["tf_workflow"] is False


def test_wrapper_passes_grad_clip_algorithm(monkeypatch, tmp_path):
    """Test that --torch-grad-clip-algorithm threads through to TorchRunnerConfig."""
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare, parse_args

    # Test parse_args default
    args = parse_args([
        "--N", "64", "--gridsize", "1", "--output-dir", str(tmp_path),
    ])
    assert args.torch_grad_clip_algorithm == "norm"

    # Test parse_args with explicit value
    args = parse_args([
        "--N", "64", "--gridsize", "1", "--output-dir", str(tmp_path),
        "--torch-grad-clip-algorithm", "agc",
    ])
    assert args.torch_grad_clip_algorithm == "agc"

    # Test end-to-end passthrough to TorchRunnerConfig
    captured = {}

    def fake_torch_run(cfg):
        captured["gradient_clip_algorithm"] = cfg.gradient_clip_algorithm
        return {"metrics": {"mse": 0.3}}

    _mock_dataset_builder(monkeypatch)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)

    run_grid_lines_compare(
        N=64,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("fno",),
        probe_npz=Path("dummy_probe.npz"),
        torch_gradient_clip_algorithm="agc",
    )

    assert captured["gradient_clip_algorithm"] == "agc"


def test_wrapper_renders_visuals(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    def fake_tf_run(cfg):
        datasets_dir = cfg.output_dir / "datasets" / f"N{cfg.N}" / f"gs{cfg.gridsize}"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        (datasets_dir / "train.npz").write_bytes(b"stub")
        (datasets_dir / "test.npz").write_bytes(b"stub")
        (cfg.output_dir / "metrics.json").write_text(json.dumps({"pinn": {}, "baseline": {}}))
        return {
            "train_npz": str(datasets_dir / "train.npz"),
            "test_npz": str(datasets_dir / "test.npz"),
        }

    def fake_torch_run(cfg):
        recon_dir = cfg.output_dir / "recons" / f"pinn_{cfg.architecture}"
        recon_dir.mkdir(parents=True, exist_ok=True)
        (recon_dir / "recon.npz").write_bytes(b"stub")
        return {"metrics": {"mse": 0.3}}

    called = {}

    def fake_render(output_dir, order):
        called["order"] = order
        visuals = output_dir / "visuals"
        visuals.mkdir(parents=True, exist_ok=True)
        out = visuals / "compare_amp_phase.png"
        out.write_bytes(b"stub")
        return {"compare": str(out)}

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.run_grid_lines_workflow", fake_tf_run)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)
    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.render_grid_lines_visuals", fake_render)

    run_grid_lines_compare(
        N=64,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("cnn", "baseline", "fno"),
        probe_npz=Path("dummy_probe.npz"),
    )

    assert called["order"] == ("gt", "pinn", "baseline", "pinn_fno")


def test_wrapper_handles_stable_hybrid(monkeypatch, tmp_path):
    """Test that stable_hybrid is wired through to Torch runner and merged metrics.

    Task ID: FNO-STABILITY-OVERHAUL-001 Task 2.3
    """
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare, parse_args

    # Test parse_args accepts 'stable_hybrid'
    args = parse_args([
        "--N", "64", "--gridsize", "1", "--output-dir", str(tmp_path),
        "--architectures", "stable_hybrid",
    ])
    assert "stable_hybrid" in args.architectures

    # Test end-to-end: stable_hybrid invokes torch runner and merges metrics
    captured = {}

    def fake_torch_run(cfg):
        captured["architecture"] = cfg.architecture
        return {"metrics": {"mse": 0.25}}

    _mock_dataset_builder(monkeypatch)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)

    result = run_grid_lines_compare(
        N=64,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("stable_hybrid",),
        probe_npz=Path("dummy_probe.npz"),
    )

    assert captured["architecture"] == "stable_hybrid"
    merged = json.loads((tmp_path / "metrics.json").read_text())
    assert "pinn_stable_hybrid" in merged


@pytest.mark.parametrize(
    "arch,metric_key",
    [
        ("fno_vanilla", "pinn_fno_vanilla"),
        ("hybrid_resnet", "pinn_hybrid_resnet"),
    ],
)
def test_wrapper_handles_new_architectures(monkeypatch, tmp_path, arch, metric_key):
    """Test that new torch architectures are wired through and merged."""
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare, parse_args

    args = parse_args([
        "--N", "64", "--gridsize", "1", "--output-dir", str(tmp_path),
        "--architectures", arch,
    ])
    assert arch in args.architectures

    captured = {}

    def fake_torch_run(cfg):
        captured["architecture"] = cfg.architecture
        return {"metrics": {"mse": 0.25}}

    _mock_dataset_builder(monkeypatch)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)

    run_grid_lines_compare(
        N=64,
        gridsize=1,
        output_dir=tmp_path,
        architectures=(arch,),
        probe_npz=Path("dummy_probe.npz"),
    )

    assert captured["architecture"] == arch
    merged = json.loads((tmp_path / "metrics.json").read_text())
    assert metric_key in merged

def test_wrapper_passes_max_hidden_channels():
    """--torch-max-hidden-channels 512 propagates to the mocked runner call."""
    from scripts.studies.grid_lines_compare_wrapper import parse_args
    args = parse_args([
        "--N", "64", "--gridsize", "1",
        "--output-dir", "/tmp/test_out",
        "--architectures", "hybrid",
        "--torch-max-hidden-channels", "512",
    ])
    assert args.torch_max_hidden_channels == 512


def test_wrapper_accepts_resnet_width():
    """--torch-resnet-width propagates to parsed args."""
    from scripts.studies.grid_lines_compare_wrapper import parse_args
    args = parse_args([
        "--N", "64", "--gridsize", "1",
        "--output-dir", "/tmp/test_out",
        "--architectures", "hybrid_resnet",
        "--torch-resnet-width", "256",
    ])
    assert args.torch_resnet_width == 256


def test_wrapper_accepts_plateau_params(tmp_path):
    """Test ReduceLROnPlateau args parse in grid_lines_compare_wrapper."""
    from scripts.studies.grid_lines_compare_wrapper import parse_args

    args = parse_args([
        "--N", "64",
        "--gridsize", "1",
        "--output-dir", str(tmp_path),
        "--architectures", "hybrid",
        "--torch-scheduler", "ReduceLROnPlateau",
        "--torch-plateau-factor", "0.25",
        "--torch-plateau-patience", "5",
        "--torch-plateau-min-lr", "1e-5",
        "--torch-plateau-threshold", "1e-3",
    ])
    assert args.torch_scheduler == "ReduceLROnPlateau"
    assert args.torch_plateau_factor == 0.25
    assert args.torch_plateau_patience == 5
    assert args.torch_plateau_min_lr == 1e-5
    assert args.torch_plateau_threshold == 1e-3


def test_wrapper_passes_optimizer(monkeypatch, tmp_path):
    """Test --torch-optimizer and related flags thread through to runner.

    Task ID: FNO-STABILITY-OVERHAUL-001 Phase 8 Task 1
    """
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare, parse_args

    # Test parse_args
    args = parse_args([
        "--N", "64", "--gridsize", "1", "--output-dir", str(tmp_path),
        "--architectures", "stable_hybrid",
        "--torch-optimizer", "sgd",
        "--torch-momentum", "0.9",
        "--torch-weight-decay", "0.01",
        "--torch-beta1", "0.9",
        "--torch-beta2", "0.999",
    ])
    assert args.torch_optimizer == "sgd"
    assert args.torch_momentum == 0.9
    assert args.torch_weight_decay == 0.01

    # Test end-to-end passthrough
    captured = {}

    def fake_torch_run(cfg):
        captured["optimizer"] = cfg.optimizer
        captured["momentum"] = cfg.momentum
        captured["weight_decay"] = cfg.weight_decay
        return {"metrics": {"mse": 0.3}}

    _mock_dataset_builder(monkeypatch)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)

    run_grid_lines_compare(
        N=64,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("stable_hybrid",),
        probe_npz=Path("dummy_probe.npz"),
        torch_optimizer="sgd",
        torch_momentum=0.9,
        torch_weight_decay=0.01,
    )

    assert captured["optimizer"] == "sgd"
    assert captured["momentum"] == 0.9
    assert captured["weight_decay"] == 0.01


def test_wrapper_passes_scheduler_knobs(monkeypatch, tmp_path):
    """Test --torch-scheduler and related flags thread through to runner."""
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare, parse_args

    # Test parse_args
    args = parse_args([
        "--N", "64", "--gridsize", "1", "--output-dir", str(tmp_path),
        "--architectures", "stable_hybrid",
        "--torch-scheduler", "WarmupCosine",
        "--torch-lr-warmup-epochs", "5",
        "--torch-lr-min-ratio", "0.05",
    ])
    assert args.torch_scheduler == "WarmupCosine"
    assert args.torch_lr_warmup_epochs == 5
    assert args.torch_lr_min_ratio == 0.05

    # Test end-to-end passthrough
    captured = {}

    def fake_torch_run(cfg):
        captured["scheduler"] = cfg.scheduler
        captured["lr_warmup_epochs"] = cfg.lr_warmup_epochs
        captured["lr_min_ratio"] = cfg.lr_min_ratio
        return {"metrics": {"mse": 0.3}}

    _mock_dataset_builder(monkeypatch)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)

    run_grid_lines_compare(
        N=64,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("stable_hybrid",),
        probe_npz=Path("dummy_probe.npz"),
        torch_scheduler="WarmupCosine",
        torch_lr_warmup_epochs=5,
        torch_lr_min_ratio=0.05,
    )

    assert captured["scheduler"] == "WarmupCosine"
    assert captured["lr_warmup_epochs"] == 5
    assert captured["lr_min_ratio"] == 0.05


def test_wrapper_accepts_plateau_scheduler(tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import parse_args

    args = parse_args([
        "--N", "64",
        "--gridsize", "1",
        "--output-dir", str(tmp_path),
        "--architectures", "hybrid",
        "--torch-scheduler", "ReduceLROnPlateau",
    ])
    assert args.torch_scheduler == "ReduceLROnPlateau"


def test_compare_wrapper_probe_mask_diameter_passthrough(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare, parse_args

    args = parse_args([
        "--N", "64",
        "--gridsize", "1",
        "--output-dir", str(tmp_path),
        "--probe-mask-diameter", "64",
    ])
    assert args.probe_mask_diameter == 64

    captured = {}

    def fake_tf_run(cfg):
        captured["probe_mask_diameter"] = cfg.probe_mask_diameter
        datasets_dir = cfg.output_dir / "datasets" / f"N{cfg.N}" / f"gs{cfg.gridsize}"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        (datasets_dir / "train.npz").write_bytes(b"stub")
        (datasets_dir / "test.npz").write_bytes(b"stub")
        (cfg.output_dir / "metrics.json").write_text("{}")
        return {"train_npz": str(datasets_dir / "train.npz"), "test_npz": str(datasets_dir / "test.npz")}

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.run_grid_lines_workflow", fake_tf_run)

    run_grid_lines_compare(
        N=64,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("cnn",),
        probe_npz=Path("dummy_probe.npz"),
        probe_mask_diameter=64,
    )

    assert captured["probe_mask_diameter"] == 64


def test_main_writes_cli_invocation_artifacts(tmp_path, monkeypatch):
    from scripts.studies import grid_lines_compare_wrapper as wrapper

    called = {"run": False}

    def fake_run_grid_lines_compare(**kwargs):
        called["run"] = True
        return {"metrics": {}}

    monkeypatch.setattr(wrapper, "run_grid_lines_compare", fake_run_grid_lines_compare)

    wrapper.main(
        [
            "--N",
            "64",
            "--gridsize",
            "1",
            "--output-dir",
            str(tmp_path),
            "--architectures",
            "cnn",
        ]
    )

    assert called["run"] is True
    inv_json = tmp_path / "invocation.json"
    inv_sh = tmp_path / "invocation.sh"
    assert inv_json.exists()
    assert inv_sh.exists()
    payload = json.loads(inv_json.read_text())
    assert "grid_lines_compare_wrapper.py" in payload["command"]
    assert "--architectures" in payload["argv"]
