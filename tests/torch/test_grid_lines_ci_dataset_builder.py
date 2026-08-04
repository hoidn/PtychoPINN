"""Fast determinism contract tests for the isolated FFNO CI dataset builder."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from ptycho.config.config import (
    simulation_config_sha256,
    simulation_config_to_dict,
)
from ptycho.workflows.grid_lines_workflow import dataset_out_dir


BUILDER_PATH = Path(__file__).with_name("_grid_lines_ci_dataset_builder.py")


def _load_builder(monkeypatch):
    module_name = "_grid_lines_ci_dataset_builder_under_test"
    spec = importlib.util.spec_from_file_location(module_name, BUILDER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_builder_seeds_before_simulation_and_binds_seed_to_dataset_identity(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "unit-test-visible")
    monkeypatch.setenv("PTYCHO_MEMOIZE_KEY_MODE", "dataset")
    builder = _load_builder(monkeypatch)
    events = []
    captured = {}
    probe = np.ones((8, 8), dtype=np.complex64)
    real_grid_lines_config = builder.GridLinesConfig

    def capture_grid_lines_config(**kwargs):
        cfg = real_grid_lines_config(**kwargs)
        captured["cfg"] = cfg
        return cfg

    def fake_simulate(cfg, realized_probe):
        assert realized_probe is probe
        events.append(("simulate", cfg.seed))
        return {
            "train": {"probe_simulated": probe},
            "test": {"probe_simulated": probe},
        }

    def fake_save(cfg, split, data, config):
        assert data["probeGuess"] is probe
        assert config is captured["legacy_config"]
        events.append(("save", split, cfg.simulation.seed))
        path = dataset_out_dir(cfg) / f"{split}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    monkeypatch.setattr(builder, "GridLinesConfig", capture_grid_lines_config)
    monkeypatch.setattr(builder, "load_probe_guess", lambda _path: probe)
    monkeypatch.setattr(builder, "scale_probe", lambda *args: probe)
    monkeypatch.setattr(builder, "apply_probe_mask", lambda *args: probe)
    monkeypatch.setattr(
        builder,
        "_apply_execution_seed",
        lambda seed: events.append(("seed", seed)),
        raising=False,
    )
    monkeypatch.setattr(builder, "simulate_grid_data", fake_simulate)
    captured["legacy_config"] = object()
    monkeypatch.setattr(
        builder,
        "configure_legacy_params",
        lambda _cfg, _probe: captured["legacy_config"],
    )
    monkeypatch.setattr(builder, "save_split_npz", fake_save)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(BUILDER_PATH),
            "--output-dir",
            str(tmp_path),
            "--probe-npz",
            str(tmp_path / "probe.npz"),
        ],
    )

    builder.main()

    assert [
        event for event in events if event[0] in {"seed", "simulate"}
    ] == [
        ("seed", 3),
        ("simulate", 3),
    ]
    cfg = captured["cfg"]
    assert cfg.seed == cfg.simulation.seed == 3
    assert simulation_config_to_dict(cfg.simulation)["seed"] == 3
    assert simulation_config_sha256(cfg.simulation) != simulation_config_sha256(
        replace(cfg.simulation, seed=None)
    )
    expected_dataset_dir = dataset_out_dir(cfg)
    manifest = json.loads(
        (tmp_path / "dataset_paths.json").read_text(encoding="utf-8")
    )
    assert manifest == {
        "train_npz": str((expected_dataset_dir / "train.npz").relative_to(tmp_path)),
        "test_npz": str((expected_dataset_dir / "test.npz").relative_to(tmp_path)),
    }
    assert [event for event in events if event[0] == "save"] == [
        ("save", "train", 3),
        ("save", "test", 3),
    ]
