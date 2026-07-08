from __future__ import annotations

import json
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import pytest
import torch


def _write_tiny_cfd_cns(path: Path, *, n: int = 4, t: int = 7, h: int = 6, w: int = 6) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["eta"] = 0.01
        handle.attrs["zeta"] = 0.01
        base = np.arange(n * t * h * w, dtype=np.float32).reshape(n, t, h, w)
        for index, field in enumerate(("density", "Vx", "Vy", "pressure")):
            handle.create_dataset(field, data=base + np.float32(index * 1000))
        handle.create_dataset("x-coordinate", data=np.linspace(0.0, 1.0, w, dtype=np.float32))
        handle.create_dataset("y-coordinate", data=np.linspace(0.0, 1.0, h, dtype=np.float32))
        handle.create_dataset("t-coordinate", data=np.linspace(0.0, 0.6, t, dtype=np.float32))
    return path


def _state_stats() -> dict:
    return {
        "schema_version": "pdebench_image128_multifield_dynamic_normalization_stats_v1",
        "mean": [0.0, 0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0, 1.0],
        "field_order": ["density", "Vx", "Vy", "pressure"],
        "axis_order": "NTHW",
    }


def _write_run_root(run_root: Path, *, data_file: Path, row_id: str = "unet_tiny_smoke", history_len: int = 2) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "normalization_stats_state.json").write_text(json.dumps(_state_stats()) + "\n", encoding="utf-8")
    (run_root / "split_manifest.json").write_text(
        json.dumps({"splits": {"train": [0, 1], "val": [2], "test": [3]}}) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "task_id": "2d_cfd_cns",
        "data_file": str(data_file),
        "field_order": ["density", "Vx", "Vy", "pressure"],
        "field_axis_order": "NTHW",
        "state_shape": [4, 7, 6, 6],
        "dimensions": {"num_trajectories": 4, "time_steps": 7, "height": 6, "width": 6},
        "trajectory_count": 4,
        "time_steps": 7,
        "history_len": history_len,
        "input_channels": history_len * 4,
        "target_channels": 4,
        "dx": 0.2,
        "dy": 0.2,
        "dt": 0.1,
        "eta": 0.01,
        "zeta": 0.01,
        "boundary_condition": "periodic",
        "dynamic_history_contract": f"concat u[t-{history_len}:t] -> u[t]",
    }
    (run_root / "hdf5_metadata.json").write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    (run_root / "dataset_manifest.json").write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    (run_root / f"model_profile_{row_id}.json").write_text(
        json.dumps(
            {
                "schema_version": "pdebench_image128_model_profile_v1",
                "profile_id": row_id,
                "profile_config": {
                    "profile_id": row_id,
                    "base_model": "unet_tiny",
                    "hidden_channels": 16,
                    "evidence_scope": "readiness-only",
                    "strong_baseline": False,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return run_root


def test_trajectory_window_uses_split_sample_and_normalizes_frames(tmp_path):
    from scripts.studies.pdebench_image128.cns_rollout_data import load_cns_trajectory_window

    data_file = _write_tiny_cfd_cns(tmp_path / "data" / "cns.h5")
    run_root = _write_run_root(tmp_path / "run", data_file=data_file)

    window = load_cns_trajectory_window(
        run_root=run_root,
        split="test",
        sample_id=0,
        start_time=3,
        steps=2,
    )

    assert window.trajectory_id == 3
    assert window.sample_id == 0
    assert window.start_time == 3
    assert window.history_len == 2
    assert window.field_order == ("density", "Vx", "Vy", "pressure")
    assert tuple(window.initial_history_phys.shape) == (2, 4, 6, 6)
    assert tuple(window.true_future_phys.shape) == (2, 4, 6, 6)
    torch.testing.assert_close(window.initial_history_norm, window.initial_history_phys)
    torch.testing.assert_close(window.true_future_norm, window.true_future_phys)


def test_missing_checkpoint_error_explains_one_step_npz_is_insufficient(tmp_path):
    from scripts.studies.pdebench_image128.cns_rollout_models import MissingCnsCheckpointError, load_cns_predictor

    data_file = _write_tiny_cfd_cns(tmp_path / "data" / "cns.h5")
    run_root = _write_run_root(tmp_path / "run", data_file=data_file)
    np.savez_compressed(run_root / "comparison_unet_tiny_smoke_sample0.npz", prediction=np.zeros((4, 6, 6)))

    with pytest.raises(MissingCnsCheckpointError, match="one-step comparison NPZ files cannot produce"):
        load_cns_predictor(run_root=run_root, row_id="unet_tiny_smoke", device="cpu")


def test_predictor_loads_state_dict_and_accepts_history_tensor(tmp_path):
    from scripts.studies.pdebench_image128.cns_rollout_models import load_cns_predictor
    from scripts.studies.pdebench_image128.models import build_model_from_profile
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    data_file = _write_tiny_cfd_cns(tmp_path / "data" / "cns.h5")
    run_root = _write_run_root(tmp_path / "run", data_file=data_file)
    profile = get_model_profile("unet_tiny_smoke")
    model = build_model_from_profile(profile, in_channels=8, out_channels=4, spatial_shape=(6, 6), task_metadata={})
    torch.save(model.state_dict(), run_root / "model_state_unet_tiny_smoke.pt")

    predictor = load_cns_predictor(run_root=run_root, row_id="unet_tiny_smoke", device="cpu")
    result = predictor(torch.zeros(2, 4, 6, 6))

    assert tuple(result.shape) == (4, 6, 6)
    assert predictor.history_len == 2
    assert predictor.field_order == ("density", "Vx", "Vy", "pressure")


def test_autoregressive_rollout_uses_previous_prediction_not_true_future():
    from scripts.studies.pdebench_image128.cns_rollout import autoregressive_rollout
    from scripts.studies.pdebench_image128.cns_rollout_data import CnsTrajectoryWindow

    class IncrementPredictor:
        row_id = "increment"
        history_len = 2
        field_order = ("density", "Vx", "Vy", "pressure")

        def __call__(self, history_norm: torch.Tensor) -> torch.Tensor:
            return history_norm[-1] + 1.0

    window = CnsTrajectoryWindow(
        trajectory_id=0,
        sample_id=0,
        start_time=2,
        history_len=2,
        field_order=("density", "Vx", "Vy", "pressure"),
        initial_history_norm=torch.stack([torch.zeros(4, 2, 2), torch.ones(4, 2, 2)]),
        initial_history_phys=torch.stack([torch.zeros(4, 2, 2), torch.ones(4, 2, 2)]),
        true_future_norm=torch.zeros(3, 4, 2, 2),
        true_future_phys=torch.full((3, 4, 2, 2), 99.0),
        dt=0.1,
    )

    result = autoregressive_rollout(window=window, predictor=IncrementPredictor(), state_stats=_state_stats())

    assert result.pred_phys.shape == (3, 4, 2, 2)
    np.testing.assert_allclose(result.pred_phys[:, 0, 0, 0], np.array([2.0, 3.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(result.true_phys[:, 0, 0, 0], np.array([99.0, 99.0, 99.0], dtype=np.float32))
    assert result.frame_time_indices == (2, 3, 4)


def test_render_field_rollout_gif_writes_frames_and_manifest(tmp_path):
    from scripts.studies.pdebench_image128.cns_rollout import CnsRolloutResult
    from scripts.studies.pdebench_image128.cns_rollout_render import render_field_rollout_gif

    result = CnsRolloutResult(
        initial_state_phys=np.zeros((4, 3, 3), dtype=np.float32),
        true_phys=np.ones((2, 4, 3, 3), dtype=np.float32),
        pred_phys=np.full((2, 4, 3, 3), 2.0, dtype=np.float32),
        abs_error_phys=np.ones((2, 4, 3, 3), dtype=np.float32),
        field_order=("density", "Vx", "Vy", "pressure"),
        frame_time_indices=(2, 3),
    )

    gif_path = render_field_rollout_gif(
        result=result,
        field="density",
        output_path=tmp_path / "density.gif",
        fps=2.0,
        include_error=True,
    )

    assert gif_path.exists()
    assert imageio.mimread(gif_path)
    manifest = json.loads((tmp_path / "density.json").read_text(encoding="utf-8"))
    assert manifest["field"] == "density"
    assert manifest["frame_count"] == 2
    assert manifest["panel_layout"] == ["initial", "true", "prediction", "absolute_error"]
    assert manifest["value_scale"]["vmin"] == 0.0
    assert manifest["value_scale"]["vmax"] == 2.0


def test_cli_dry_run_resolves_manifest_and_missing_checkpoint_fails(tmp_path):
    from scripts.studies.pdebench_image128.render_cns_rollout_video import main

    data_file = _write_tiny_cfd_cns(tmp_path / "data" / "cns.h5")
    run_root = _write_run_root(tmp_path / "run", data_file=data_file)
    output_root = tmp_path / "videos"

    assert (
        main(
            [
                "--run-root",
                str(run_root),
                "--row-id",
                "unet_tiny_smoke",
                "--sample-id",
                "0",
                "--start-time",
                "2",
                "--steps",
                "2",
                "--field",
                "density",
                "--output-root",
                str(output_root),
                "--dry-run",
            ]
        )
        == 0
    )
    manifest = json.loads((output_root / "unet_tiny_smoke_sample000_rollout_manifest.json").read_text(encoding="utf-8"))
    assert manifest["row_id"] == "unet_tiny_smoke"
    assert manifest["sample_id"] == 0
    assert manifest["trajectory_id"] == 3
    assert manifest["requires_checkpoint"] is True

    assert (
        main(
            [
                "--run-root",
                str(run_root),
                "--row-id",
                "unet_tiny_smoke",
                "--sample-id",
                "0",
                "--start-time",
                "2",
                "--steps",
                "2",
                "--field",
                "density",
                "--output-root",
                str(output_root),
            ]
        )
        == 2
    )


def test_cli_exports_rollout_gif_with_checkpoint(tmp_path):
    from scripts.studies.pdebench_image128.render_cns_rollout_video import main
    from scripts.studies.pdebench_image128.models import build_model_from_profile
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    data_file = _write_tiny_cfd_cns(tmp_path / "data" / "cns.h5")
    run_root = _write_run_root(tmp_path / "run", data_file=data_file)
    profile = get_model_profile("unet_tiny_smoke")
    model = build_model_from_profile(profile, in_channels=8, out_channels=4, spatial_shape=(6, 6), task_metadata={})
    torch.save(model.state_dict(), run_root / "model_state_unet_tiny_smoke.pt")
    output_root = tmp_path / "videos"

    assert (
        main(
            [
                "--run-root",
                str(run_root),
                "--row-id",
                "unet_tiny_smoke",
                "--sample-id",
                "0",
                "--start-time",
                "2",
                "--steps",
                "2",
                "--field",
                "density",
                "--output-root",
                str(output_root),
                "--include-error",
            ]
        )
        == 0
    )

    gif_path = output_root / "unet_tiny_smoke_sample000_density_rollout.gif"
    assert gif_path.exists()
    assert imageio.mimread(gif_path)
    manifest = json.loads((output_root / "unet_tiny_smoke_sample000_rollout_manifest.json").read_text(encoding="utf-8"))
    assert manifest["outputs"] == [str(gif_path)]
