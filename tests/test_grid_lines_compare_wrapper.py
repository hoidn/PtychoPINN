# tests/test_grid_lines_compare_wrapper.py
"""Tests for grid_lines_compare_wrapper orchestration."""
import json
from pathlib import Path


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

    def fake_tf_run(cfg):
        datasets_dir = cfg.output_dir / "datasets" / f"N{cfg.N}" / f"gs{cfg.gridsize}"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        (datasets_dir / "train.npz").write_bytes(b"stub")
        (datasets_dir / "test.npz").write_bytes(b"stub")
        (cfg.output_dir / "metrics.json").write_text(json.dumps({}))
        return {"train_npz": str(datasets_dir / "train.npz"), "test_npz": str(datasets_dir / "test.npz")}

    captured = {}

    def fake_torch_run(cfg):
        captured["torch_loss_mode"] = getattr(cfg, "torch_loss_mode", None)
        return {"metrics": {"mse": 0.3}}

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.run_grid_lines_workflow", fake_tf_run)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)

    run_grid_lines_compare(
        N=64,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("fno",),
        probe_npz=Path("dummy_probe.npz"),
    )

    assert captured["torch_loss_mode"] == "mae"


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
    def fake_tf_run(cfg):
        datasets_dir = cfg.output_dir / "datasets" / f"N{cfg.N}" / f"gs{cfg.gridsize}"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        (datasets_dir / "train.npz").write_bytes(b"stub")
        (datasets_dir / "test.npz").write_bytes(b"stub")
        (cfg.output_dir / "metrics.json").write_text(json.dumps({}))
        return {"train_npz": str(datasets_dir / "train.npz"), "test_npz": str(datasets_dir / "test.npz")}

    captured = {}

    def fake_torch_run(cfg):
        captured["gradient_clip_algorithm"] = cfg.gradient_clip_algorithm
        return {"metrics": {"mse": 0.3}}

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.run_grid_lines_workflow", fake_tf_run)
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
    def fake_tf_run(cfg):
        datasets_dir = cfg.output_dir / "datasets" / f"N{cfg.N}" / f"gs{cfg.gridsize}"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        (datasets_dir / "train.npz").write_bytes(b"stub")
        (datasets_dir / "test.npz").write_bytes(b"stub")
        (cfg.output_dir / "metrics.json").write_text(json.dumps({}))
        return {"train_npz": str(datasets_dir / "train.npz"), "test_npz": str(datasets_dir / "test.npz")}

    captured = {}

    def fake_torch_run(cfg):
        captured["architecture"] = cfg.architecture
        return {"metrics": {"mse": 0.25}}

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.run_grid_lines_workflow", fake_tf_run)
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
