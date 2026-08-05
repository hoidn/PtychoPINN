"""Focused public-runner contracts for rectangular gauge initialization."""

import json
from pathlib import Path

import numpy as np
import pytest


def _write_synthetic_npz_pair(tmp_path):
    from ptycho.config.config import ModelConfig, SimulationConfig, TrainingConfig
    from ptycho.metadata import MetadataManager

    N = 64
    n_samples = 4
    outer_offset_test = 20
    border_size = (N - outer_offset_test / 2) / 2
    tile_size = N - (int(np.ceil(border_size)) + int(np.floor(border_size)))
    data = {
        "diffraction": np.ones((n_samples, N, N, 1), dtype=np.float32),
        "Y_I": np.ones((n_samples, N, N, 1), dtype=np.float32),
        "Y_phi": np.zeros((n_samples, N, N, 1), dtype=np.float32),
        "coords_nominal": np.zeros((n_samples, 2), dtype=np.float32),
        "coords_true": np.zeros((n_samples, 2), dtype=np.float32),
        "YY_full": np.ones((1, N * 2, N * 2), dtype=np.complex64),
        "YY_ground_truth": np.ones((tile_size, tile_size, 1), dtype=np.complex64),
        "norm_Y_I": np.array(1.0, dtype=np.float32),
    }
    metadata = MetadataManager.create_metadata(
        TrainingConfig(model=ModelConfig(N=N, gridsize=1)),
        script_name="test_fixture",
        nimgs_test=n_samples,
        outer_offset_test=outer_offset_test,
        offset=SimulationConfig().scan.offset,
    )
    train_path = tmp_path / "train.npz"
    test_path = tmp_path / "test.npz"
    MetadataManager.save_with_metadata(str(train_path), data, metadata)
    MetadataManager.save_with_metadata(str(test_path), data, metadata)
    return train_path, test_path


def test_main_forwards_dose_closure_rect_s1s2_init(tmp_path, monkeypatch):
    from scripts.studies import grid_lines_torch_runner as runner

    captured = {}

    def fake_run_grid_lines_torch(cfg, **_kwargs):
        captured["cfg"] = cfg
        run_dir = cfg.output_dir / "runs" / f"pinn_{cfg.architecture}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return {"run_dir": str(run_dir), "metrics": {}}

    monkeypatch.setattr(runner, "run_grid_lines_torch", fake_run_grid_lines_torch)
    train_npz = tmp_path / "train.npz"
    test_npz = tmp_path / "test.npz"
    train_npz.write_bytes(b"stub")
    test_npz.write_bytes(b"stub")

    runner.main(
        [
            "--train-npz",
            str(train_npz),
            "--test-npz",
            str(test_npz),
            "--output-dir",
            str(tmp_path / "output"),
            "--architecture",
            "ffno",
            "--physics-forward-mode",
            "rectangular_scaled",
            "--torch-loss-mode",
            "poisson",
            "--rect-s1s2-init",
            "dose_closure",
        ]
    )

    assert captured["cfg"].rect_s1s2_init == "dose_closure"


def test_grid_training_rejects_dose_closure_on_legacy_contract(tmp_path):
    from scripts.studies.grid_lines_torch_runner import (
        TorchRunnerConfig,
        run_torch_training,
    )

    cfg = TorchRunnerConfig(
        train_npz=tmp_path / "train.npz",
        test_npz=tmp_path / "test.npz",
        output_dir=tmp_path / "output",
        architecture="ffno",
        physics_forward_mode="rectangular_scaled",
        rect_s1s2_init="dose_closure",
        scale_contract_version="legacy_v1",
        measurement_domain="normalized_amplitude",
        torch_loss_mode="mae",
    )

    with pytest.raises(ValueError, match="dose_closure.*ci_intensity_v2"):
        run_torch_training(cfg, {}, {})


def test_rect_s1s2_help_separates_ci_and_legacy_count_scaling(capsys):
    from scripts.studies import grid_lines_torch_runner as runner

    with pytest.raises(SystemExit) as error:
        runner.main(["--help"])

    assert error.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "CI dictionary adapter derives its count-" in help_text
    assert "amplitude scale independently" in help_text
    assert "--count-scale-mode" in help_text
    assert "legacy/non-CI" in help_text
    assert "ignored by the CI path" in help_text


def test_runner_exposes_and_persists_training_initialization(
    tmp_path,
    monkeypatch,
):
    import torch

    from scripts.studies import grid_lines_torch_runner as runner

    train_path, test_path = _write_synthetic_npz_pair(tmp_path)
    output_dir = tmp_path / "output"
    summary_path = output_dir / "training_summary.json"
    summary_path.parent.mkdir(parents=True)
    initialization = {
        "schema_version": "rect-s1s2-initialization-v1",
        "mode": "dose_closure",
        "solved_gauge": 3.25,
        "method": "dose_closure_unit_object",
        "sampled_patterns": 256,
    }
    summary_path.write_text(json.dumps(initialization), encoding="utf-8")
    cfg = runner.TorchRunnerConfig(
        train_npz=train_path,
        test_npz=test_path,
        output_dir=output_dir,
        architecture="ffno",
        epochs=1,
    )
    monkeypatch.setattr(
        runner,
        "run_torch_training",
        lambda *_args, **_kwargs: {
            "model": None,
            "history": {},
            "generator": "ffno",
            "scaffold": True,
            "rect_s1s2_initialization": initialization,
            "training_summary_path": summary_path,
        },
    )
    monkeypatch.setattr(
        runner,
        "run_torch_inference",
        lambda *_args, **_kwargs: np.ones((64, 64), dtype=np.complex64),
    )
    monkeypatch.setattr(
        runner,
        "compute_metrics",
        lambda *_args, **_kwargs: {"mse": 0.1},
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    result = runner.run_grid_lines_torch(cfg)

    assert result["rect_s1s2_initialization"] == initialization
    assert Path(result["training_summary_path"]) == summary_path
    assert json.loads(summary_path.read_text(encoding="utf-8")) == initialization
    config = json.loads(
        (Path(result["run_dir"]) / "config.json").read_text(encoding="utf-8")
    )
    assert config["rect_s1s2_initialization"] == initialization
    assert config["training_summary_path"] == result["training_summary_path"]
    assert Path(config["training_summary_path"]) == summary_path
