"""Amplitude-physics-gain calibration harness tests (plan Task 26, design §4).

Covers the calibration seams that must be trustworthy BEFORE any GPU sweep
runs: the runner-level ``amplitude_physics_gain`` plumbing (dictionary flow
and the mmap mirror), rung0 baseline-recipe resolution from the ladder spec,
the fixed-batch replication of the inline-dataset emission conventions, the
Rule B init-scale-match gain math, and the predeclared Rule A/Rule B
selection criterion (design §4: Rule B preferred iff it lands within the
sweep's quality plateau; amp SSIM >= 0.85 or halt).

Contract: docs/superpowers/specs/2026-07-12-probe-rank-physics-contract-fix-design.md §4;
docs/specs/spec-ptycho-torch-probe-layout.md §3.3.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.studies.ablation import gain_calibration as gc
from scripts.studies.ablation.runtime_errors import RuntimeExecutionError
from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
LADDER_SPEC = REPO_ROOT / "scripts/studies/specs/grid_lines_bridge_ladder.toml"


# ---------------------------------------------------------------------------
# Runner plumbing
# ---------------------------------------------------------------------------


def _runner_cfg(tmp_path: Path, **overrides) -> TorchRunnerConfig:
    kwargs = dict(
        train_npz=tmp_path / "train.npz",
        test_npz=tmp_path / "test.npz",
        output_dir=tmp_path / "out",
        architecture="cnn",
        N=64,
        gridsize=1,
        epochs=1,
        torch_loss_mode="mae",
        scale_contract_version="legacy_v1",
        measurement_domain="normalized_amplitude",
    )
    kwargs.update(overrides)
    return TorchRunnerConfig(**kwargs)


def test_torch_runner_config_defaults_to_unit_gain(tmp_path):
    assert _runner_cfg(tmp_path).amplitude_physics_gain == 1.0


@pytest.mark.torch
def test_run_torch_training_threads_gain_into_factory_overrides(tmp_path, monkeypatch):
    """The dictionary flow must forward the runner gain to
    ``create_training_payload`` via ``_train_with_lightning``'s overrides
    channel (the same channel every torch-only ModelConfig knob uses)."""
    from ptycho_torch.workflows import components
    from scripts.studies import grid_lines_torch_runner as runner_mod

    captured: dict = {}

    def fake_train_with_lightning(
        train_container, test_container, config, execution_config=None, overrides=None
    ):
        captured["overrides"] = dict(overrides or {})
        return {
            "history": {"train_loss": [], "val_loss": []},
            "train_container": train_container,
            "test_container": test_container,
            "models": {"diffraction_to_obj": object()},
        }

    monkeypatch.setattr(components, "_train_with_lightning", fake_train_with_lightning)

    n = 8
    data = {
        "diffraction": np.random.default_rng(0)
        .random((n, 64, 64, 1))
        .astype(np.float32),
        "Y_I": np.zeros((n, 64, 64, 1), dtype=np.float32),
        "Y_phi": np.zeros((n, 64, 64, 1), dtype=np.float32),
        "coords_nominal": np.zeros((n, 1, 2, 1), dtype=np.float32),
        "probeGuess": np.ones((64, 64), dtype=np.complex64),
    }
    cfg = _runner_cfg(tmp_path, amplitude_physics_gain=7.0)
    results = runner_mod.run_torch_training(cfg, data, dict(data))
    assert results["models"]["diffraction_to_obj"] is not None
    assert captured["overrides"]["amplitude_physics_gain"] == 7.0


def test_mmap_wiring_mirrors_runner_gain_override(tmp_path):
    """train_via_generic_loader mirrors run_torch_training's torch-only
    override keys verbatim; the gain must ride along (Task 28 depends on
    the two flows staying in lockstep)."""
    from scripts.studies.ablation.runtime_ladder_mmap import runner_torch_overrides

    cfg = _runner_cfg(tmp_path, amplitude_physics_gain=16.0)
    overrides = runner_torch_overrides(
        cfg,
        {
            "mmap_scale_convention": "loader",
            "probe_normalize": False,
        },
    )
    assert overrides["amplitude_physics_gain"] == 16.0
    assert overrides["normalize"] == "Batch"
    # The pre-existing mirror keys must survive the refactor.
    for key in (
        "training_patch_weighting",
        "physics_forward_mode",
        "cnn_output_mode",
        "rect_s1s2_trainable",
        "scale_contract_version",
        "measurement_domain",
        "normalize",
        "probe_normalize",
        "n_subsample",
        "strategy",
    ):
        assert key in overrides, key


# ---------------------------------------------------------------------------
# Baseline (rung0) recipe resolution
# ---------------------------------------------------------------------------


def test_baseline_point_resolves_rung0_recipe(tmp_path):
    point = gc.resolve_baseline_point(
        LADDER_SPEC,
        train_npz=tmp_path / "train.npz",
        test_npz=tmp_path / "test.npz",
        work_dir=tmp_path / "work",
        gain=16.0,
        base_dir=REPO_ROOT,
    )
    assert point.config["loader"] == "dictionary"
    assert point.dataset.expression == "dictionary"
    cfg = point.runner_cfg
    assert cfg.architecture == "hybrid_resnet"
    assert cfg.seed == 3
    assert cfg.epochs == 5
    assert cfg.N == 128
    assert cfg.batch_size == 16
    assert cfg.amplitude_physics_gain == 16.0


def test_baseline_point_epoch_override(tmp_path):
    point = gc.resolve_baseline_point(
        LADDER_SPEC,
        train_npz=tmp_path / "train.npz",
        test_npz=tmp_path / "test.npz",
        work_dir=tmp_path / "work",
        gain=1.0,
        epochs=0,
        base_dir=REPO_ROOT,
    )
    assert point.runner_cfg.epochs == 0


# ---------------------------------------------------------------------------
# Fixed dictionary batch (inline-dataset emission conventions)
# ---------------------------------------------------------------------------


def _tiny_container(n=4, size=16):
    rng = np.random.default_rng(3)
    return {
        "X": rng.random((n, size, size, 1)).astype(np.float32),
        "observed_images": rng.random((n, size, size, 1)).astype(np.float32),
        "coords_relative": rng.random((n, 1, 2, 1)).astype(np.float32),
        "probe": (rng.random((size, size)) + 1j * rng.random((size, size))).astype(
            np.complex64
        ),
    }


def test_fixed_dictionary_batch_layout():
    container = _tiny_container()
    fields, probe, scale = gc.fixed_dictionary_batch(container, range(4))
    # Channel-first images, exactly the inline PtychoLightningDataset permute.
    assert fields["images"].shape == (4, 1, 16, 16)
    np.testing.assert_array_equal(
        fields["images"].numpy(), container["X"].transpose(0, 3, 1, 2)
    )
    np.testing.assert_array_equal(
        fields["observed_images"].numpy(),
        container["observed_images"].transpose(0, 3, 1, 2),
    )
    # coords (B, 1, 2, C) -> (B, C, 1, 2).
    assert fields["coords_relative"].shape == (4, 1, 1, 2)
    np.testing.assert_array_equal(
        fields["coords_relative"].numpy(),
        container["coords_relative"].transpose(0, 3, 1, 2),
    )
    # Documented probe layout (PROBE-RANK-001): (B, C, P, H, W).
    assert probe.shape == (4, 1, 1, 16, 16)
    assert probe.dtype == torch.complex64
    np.testing.assert_array_equal(probe[2, 0, 0].numpy(), container["probe"])
    # Unit legacy scaling constants (dictionary-parity convention).
    assert torch.equal(
        fields["rms_scaling_constant"], torch.ones(4, 1, 1, 1)
    )
    assert torch.equal(
        fields["physics_scaling_constant"], torch.ones(4, 1, 1, 1)
    )
    assert fields["experiment_id"].dtype == torch.long
    assert torch.equal(scale, torch.ones(4, 1, 1, 1))


# ---------------------------------------------------------------------------
# Rule B gain math + tensor stats
# ---------------------------------------------------------------------------


def test_rule_b_gain_is_rms_ratio():
    obs = torch.full((4, 1, 8, 8), 2.0)
    pred = torch.full((4, 1, 8, 8), 0.5)
    assert gc.init_scale_match_gain(
        obs_sq_sum=float((obs**2).sum()),
        pred_sq_sum=float((pred**2).sum()),
    ) == pytest.approx(4.0)


def test_rule_b_gain_rejects_degenerate_prediction():
    with pytest.raises(RuntimeExecutionError):
        gc.init_scale_match_gain(obs_sq_sum=1.0, pred_sq_sum=0.0)


def test_tensor_stats_structure():
    t = torch.linspace(0.0, 1.0, 101)
    stats = gc.tensor_stats(t)
    assert stats["mean"] == pytest.approx(0.5)
    assert stats["rms"] == pytest.approx(float(torch.sqrt((t**2).mean())))
    assert stats["max"] == pytest.approx(1.0)
    assert stats["p99"] == pytest.approx(0.99, abs=0.01)
    assert stats["frac_gt_0p9"] == pytest.approx(0.1, abs=0.02)


# ---------------------------------------------------------------------------
# Predeclared Rule A / Rule B selection criterion (design §4)
# ---------------------------------------------------------------------------


def _points(amp_by_gain):
    return [
        {"gain": g, "amp_ssim": a, "phase_ssim": 0.9}
        for g, a in sorted(amp_by_gain.items())
    ]


def _dataset_provenance(**overrides):
    dataset = {
        "id": "lines128_reference",
        "recipe_fingerprint_sha256": "recipe-sha",
        "train_sha256": "train-sha",
        "test_sha256": "test-sha",
        "probe_sha256": "probe-sha",
        "n_train": 8993,
        "n_test": 735,
    }
    dataset.update(overrides)
    return dataset


def _write_gain_evidence(
    root: Path,
    *,
    tag: str,
    gain: float,
    amp_ssim: float,
    seed: int = 3,
    epochs: int = 5,
    recipe: str = "rung0_reference (ladder spec baseline.config, dictionary flow)",
    git_commit: str = "commit-sha",
    dataset=None,
    selected_checkpoint: str | None = None,
    checkpoint_sha256: str | None = None,
):
    point_dir = root / tag
    point_dir.mkdir(parents=True, exist_ok=True)
    if selected_checkpoint is None:
        checkpoint = root / "_checkpoints" / f"{tag}.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint:{tag}".encode("ascii"))
        selected_checkpoint = str(checkpoint)
        checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    payload = {
        "schema_version": "gain_calibration_point_v1",
        "mode": "sweep_point",
        "tag": tag,
        "gain": gain,
        "seed": seed,
        "epochs": epochs,
        "recipe": recipe,
        "git_commit": git_commit,
        "dataset": dataset or _dataset_provenance(),
        "selected_checkpoint": selected_checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "history": {"train_loss": [1.0], "val_loss": [0.5]},
        "metrics": {
            "amp_ssim": amp_ssim,
            "phase_ssim": 0.9,
            "amp_mae": 0.1,
            "phase_mae": 0.2,
        },
    }
    (point_dir / "gain_point.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _write_rule_b_evidence(
    root: Path,
    *,
    gain: float = 16.0,
    seed: int = 3,
    recipe: str = "rung0_reference (ladder spec baseline.config, dictionary flow)",
    git_commit: str = "commit-sha",
    dataset=None,
):
    point_dir = root / "init_stats"
    point_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "gain_calibration_point_v1",
        "mode": "init_stats",
        "tag": "init_stats",
        "gain": 1.0,
        "seed": seed,
        "epochs": 0,
        "recipe": recipe,
        "git_commit": git_commit,
        "dataset": dataset or _dataset_provenance(),
        "rule_b": {"rule": "init_scale_match", "seed": seed, "gain": gain},
    }
    (point_dir / "rule_b_derivation.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _write_complete_sweep(root: Path, amp_by_gain=None):
    amp_by_gain = amp_by_gain or {1: 0.48, 4: 0.80, 16: 0.896, 64: 0.890}
    for gain, amp_ssim in amp_by_gain.items():
        _write_gain_evidence(
            root,
            tag=f"gain_{gain}",
            gain=gain,
            amp_ssim=amp_ssim,
        )


def _write_seed4_evidence(root: Path):
    for gain, amp_ssim in {1: 0.48, 4: 0.80, 16: 0.896, 64: 0.890}.items():
        _write_gain_evidence(
            root,
            tag=f"gain_{gain}",
            gain=gain,
            amp_ssim=amp_ssim,
            seed=4,
        )
    _write_rule_b_evidence(root, seed=4)


def test_selection_prefers_rule_b_inside_plateau():
    decision = gc.select_gain_rule(
        _points({1: 0.48, 4: 0.80, 16: 0.896, 64: 0.890}), rule_b_gain=20.0
    )
    assert decision["plateau_gains"] == [16.0, 64.0]
    assert decision["rule_b_in_plateau"] is True
    assert decision["preferred_rule"] == "rule_b_init_scale_match"
    assert decision["requires_confirmation_run"] is True


def test_selection_falls_back_to_swept_constant_outside_plateau():
    decision = gc.select_gain_rule(
        _points({1: 0.48, 4: 0.80, 16: 0.896, 64: 0.890}), rule_b_gain=5.0
    )
    assert decision["rule_b_in_plateau"] is False
    assert decision["preferred_rule"] == "rule_a_fixed_constant"
    assert decision["selected_gain"] == 16.0


def test_selection_peak_not_plateau_selects_the_peak():
    decision = gc.select_gain_rule(
        _points({1: 0.48, 4: 0.70, 16: 0.90, 64: 0.86}), rule_b_gain=32.0
    )
    assert decision["plateau_gains"] == [16.0]
    assert decision["rule_b_in_plateau"] is False
    assert decision["selected_gain"] == 16.0


def test_selection_halts_when_no_gain_reaches_floor():
    decision = gc.select_gain_rule(
        _points({1: 0.48, 4: 0.60, 16: 0.70, 64: 0.65}), rule_b_gain=16.0
    )
    assert decision["halt"] is True
    assert decision["preferred_rule"] is None


def test_selection_rule_b_at_swept_value_needs_no_confirmation():
    decision = gc.select_gain_rule(
        _points({1: 0.48, 4: 0.80, 16: 0.896, 64: 0.890}), rule_b_gain=16.0
    )
    assert decision["preferred_rule"] == "rule_b_init_scale_match"
    assert decision["requires_confirmation_run"] is False


def test_selection_rule_b_near_swept_value_requires_confirmation():
    decision = gc.select_gain_rule(
        _points({1: 0.48, 4: 0.80, 16: 0.896, 64: 0.890}),
        rule_b_gain=16.00000001,
    )
    assert decision["preferred_rule"] == "rule_b_init_scale_match"
    assert decision["selected_gain"] == 16.00000001
    assert decision["requires_confirmation_run"] is True


def test_summarize_sweep_rejects_one_point_false_success(tmp_path):
    _write_gain_evidence(
        tmp_path, tag="gain_16", gain=16.0, amp_ssim=0.90
    )
    _write_rule_b_evidence(tmp_path, gain=16.0)

    with pytest.raises(RuntimeExecutionError, match="predeclared.*gain set"):
        gc.summarize_sweep(tmp_path)


def test_summarize_sweep_rejects_duplicate_gain(tmp_path):
    _write_complete_sweep(tmp_path)
    _write_gain_evidence(
        tmp_path, tag="gain_16_duplicate", gain=16.0, amp_ssim=0.90
    )
    _write_rule_b_evidence(tmp_path)

    with pytest.raises(RuntimeExecutionError, match="predeclared.*gain set"):
        gc.summarize_sweep(tmp_path)


def test_summarize_sweep_rejects_unexpected_gain(tmp_path):
    _write_complete_sweep(tmp_path)
    _write_gain_evidence(tmp_path, tag="gain_8", gain=8.0, amp_ssim=0.90)
    _write_rule_b_evidence(tmp_path)

    with pytest.raises(RuntimeExecutionError, match="predeclared.*gain set"):
        gc.summarize_sweep(tmp_path)


def test_summarize_sweep_rejects_nonfinite_gain(tmp_path):
    _write_complete_sweep(tmp_path, {1: 0.48, 4: 0.80, 16: 0.896})
    _write_gain_evidence(
        tmp_path, tag="gain_nonfinite", gain=float("nan"), amp_ssim=0.90
    )
    _write_rule_b_evidence(tmp_path)

    with pytest.raises(RuntimeExecutionError, match="finite"):
        gc.summarize_sweep(tmp_path)


def test_summarize_sweep_rejects_nonfinite_point_metric(tmp_path):
    _write_complete_sweep(
        tmp_path, {1: 0.48, 4: 0.80, 16: float("nan"), 64: 0.890}
    )
    _write_rule_b_evidence(tmp_path)

    with pytest.raises(RuntimeExecutionError, match="finite"):
        gc.summarize_sweep(tmp_path)


@pytest.mark.parametrize(
    "missing_field", ["selected_checkpoint", "checkpoint_sha256"]
)
def test_summarize_sweep_requires_checkpoint_identity(tmp_path, missing_field):
    _write_complete_sweep(tmp_path)
    _write_rule_b_evidence(tmp_path)
    point_path = tmp_path / "gain_64" / "gain_point.json"
    payload = json.loads(point_path.read_text(encoding="utf-8"))
    del payload[missing_field]
    point_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeExecutionError, match="checkpoint"):
        gc.summarize_sweep(tmp_path)


def test_summarize_sweep_rejects_malformed_checkpoint_sha256(tmp_path):
    _write_complete_sweep(tmp_path)
    _write_rule_b_evidence(tmp_path)
    point_path = tmp_path / "gain_64" / "gain_point.json"
    payload = json.loads(point_path.read_text(encoding="utf-8"))
    payload["checkpoint_sha256"] = "not-a-sha256"
    point_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeExecutionError, match="checkpoint.*SHA-256"):
        gc.summarize_sweep(tmp_path)


def test_summarize_sweep_verifies_existing_local_checkpoint_hash(tmp_path):
    _write_complete_sweep(tmp_path)
    _write_rule_b_evidence(tmp_path)
    checkpoint = tmp_path / "selected.ckpt"
    checkpoint.write_bytes(b"actual checkpoint bytes")
    point_path = tmp_path / "gain_64" / "gain_point.json"
    payload = json.loads(point_path.read_text(encoding="utf-8"))
    payload["selected_checkpoint"] = str(checkpoint)
    payload["checkpoint_sha256"] = "0" * 64
    point_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeExecutionError, match="checkpoint.*does not match"):
        gc.summarize_sweep(tmp_path)


def test_summarize_sweep_rejects_nonexistent_checkpoint_path(tmp_path):
    _write_complete_sweep(tmp_path)
    _write_rule_b_evidence(tmp_path)
    point_path = tmp_path / "gain_64" / "gain_point.json"
    payload = json.loads(point_path.read_text(encoding="utf-8"))
    payload["selected_checkpoint"] = str(tmp_path / "missing.ckpt")
    payload["checkpoint_sha256"] = "a" * 64
    point_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeExecutionError, match="checkpoint.*does not exist"):
        gc.summarize_sweep(tmp_path)


@pytest.mark.parametrize(
    "override",
    [
        {"seed": 4},
        {"epochs": 4},
        {"recipe": "different recipe"},
        {"git_commit": "different-commit"},
        {"dataset": _dataset_provenance(train_sha256="different-train-sha")},
    ],
    ids=["seed", "epochs", "recipe", "git_commit", "dataset"],
)
def test_summarize_sweep_rejects_mixed_point_provenance(tmp_path, override):
    _write_complete_sweep(tmp_path)
    _write_gain_evidence(
        tmp_path,
        tag="gain_64",
        gain=64.0,
        amp_ssim=0.890,
        **override,
    )
    _write_rule_b_evidence(tmp_path)

    with pytest.raises(RuntimeExecutionError, match="provenance"):
        gc.summarize_sweep(tmp_path)


def test_summarize_sweep_rejects_homogeneous_nonreference_seed(tmp_path):
    _write_seed4_evidence(tmp_path)

    with pytest.raises(RuntimeExecutionError, match="reference seed.*3"):
        gc.summarize_sweep(tmp_path)


def test_summarize_cli_rejects_homogeneous_nonreference_seed(tmp_path):
    _write_seed4_evidence(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.studies.ablation.gain_calibration",
            "summarize",
            "--output-root",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "reference seed" in completed.stderr


def test_summarize_sweep_rejects_nonintegral_reference_seed(tmp_path):
    for gain, amp_ssim in {1: 0.48, 4: 0.80, 16: 0.896, 64: 0.890}.items():
        _write_gain_evidence(
            tmp_path,
            tag=f"gain_{gain}",
            gain=gain,
            amp_ssim=amp_ssim,
            seed=3.5,
        )
    _write_rule_b_evidence(tmp_path, seed=3.5)

    with pytest.raises(RuntimeExecutionError, match="reference seed.*3"):
        gc.summarize_sweep(tmp_path)


@pytest.mark.parametrize(
    "override",
    [
        {"seed": 4},
        {"recipe": "different recipe"},
        {"git_commit": "different-commit"},
        {"dataset": _dataset_provenance(test_sha256="different-test-sha")},
    ],
    ids=["seed", "recipe", "git_commit", "dataset"],
)
def test_summarize_sweep_rejects_mismatched_rule_b_provenance(
    tmp_path, override
):
    _write_complete_sweep(tmp_path)
    _write_rule_b_evidence(tmp_path, **override)

    with pytest.raises(RuntimeExecutionError, match="Rule B.*provenance"):
        gc.summarize_sweep(tmp_path)


@pytest.mark.parametrize(
    "provenance_override",
    [
        {"git_commit": None},
        {"dataset": _dataset_provenance(train_sha256="")},
        {"dataset": _dataset_provenance(n_train=0)},
    ],
    ids=["null_commit", "empty_dataset_hash", "empty_dataset"],
)
def test_summarize_sweep_rejects_invalid_common_provenance(
    tmp_path, provenance_override
):
    for gain, amp_ssim in {1: 0.48, 4: 0.80, 16: 0.896, 64: 0.890}.items():
        _write_gain_evidence(
            tmp_path,
            tag=f"gain_{gain}",
            gain=gain,
            amp_ssim=amp_ssim,
            **provenance_override,
        )
    _write_rule_b_evidence(tmp_path, **provenance_override)

    with pytest.raises(RuntimeExecutionError, match="provenance"):
        gc.summarize_sweep(tmp_path)


def test_summarize_sweep_preserves_common_provenance(tmp_path):
    _write_complete_sweep(tmp_path)
    _write_rule_b_evidence(tmp_path)

    summary = gc.summarize_sweep(tmp_path)

    assert summary["provenance"] == {
        "seed": 3,
        "epochs": 5,
        "recipe": "rung0_reference (ladder spec baseline.config, dictionary flow)",
        "git_commit": "commit-sha",
        "dataset": _dataset_provenance(),
    }


def test_summarize_sweep_keeps_required_confirmation_incomplete(tmp_path):
    _write_complete_sweep(tmp_path)
    _write_rule_b_evidence(tmp_path, gain=20.0)

    summary = gc.summarize_sweep(tmp_path)

    assert summary["decision"]["requires_confirmation_run"] is True
    assert summary["decision"]["confirmation_complete"] is False
    assert summary["decision"]["complete"] is False
    assert gc.main(["summarize", "--output-root", str(tmp_path)]) == 4


def test_summarize_sweep_accepts_one_matching_rule_b_confirmation(tmp_path):
    _write_complete_sweep(tmp_path)
    _write_rule_b_evidence(tmp_path, gain=20.0)
    _write_gain_evidence(
        tmp_path,
        tag="ruleb_confirm_20",
        gain=20.0,
        amp_ssim=0.86,
    )

    summary = gc.summarize_sweep(tmp_path)

    decision = summary["decision"]
    assert decision["halt"] is False
    assert decision["complete"] is True
    assert decision["requires_confirmation_run"] is False
    assert decision["confirmation_complete"] is True
    assert decision["confirmation_result"]["gain"] == 20.0
    assert summary["confirmation_runs"] == [decision["confirmation_result"]]
    assert gc.main(["summarize", "--output-root", str(tmp_path)]) == 0


def test_summarize_sweep_rejects_duplicate_rule_b_confirmation(tmp_path):
    _write_complete_sweep(tmp_path)
    _write_rule_b_evidence(tmp_path, gain=20.0)
    _write_gain_evidence(
        tmp_path, tag="ruleb_confirm_20_a", gain=20.0, amp_ssim=0.86
    )
    _write_gain_evidence(
        tmp_path, tag="ruleb_confirm_20_b", gain=20.0, amp_ssim=0.87
    )

    with pytest.raises(RuntimeExecutionError, match="exactly one.*confirmation"):
        gc.summarize_sweep(tmp_path)


def test_summarize_sweep_rejects_wrong_gain_rule_b_confirmation(tmp_path):
    _write_complete_sweep(tmp_path)
    _write_rule_b_evidence(tmp_path, gain=20.0)
    _write_gain_evidence(
        tmp_path, tag="ruleb_confirm_wrong_gain", gain=21.0, amp_ssim=0.86
    )

    with pytest.raises(RuntimeExecutionError, match="selected gain"):
        gc.summarize_sweep(tmp_path)


def test_summarize_sweep_rejects_near_but_not_exact_confirmation_gain(tmp_path):
    _write_complete_sweep(tmp_path)
    _write_rule_b_evidence(tmp_path, gain=20.0)
    _write_gain_evidence(
        tmp_path,
        tag="ruleb_confirm_near_gain",
        gain=20.00000001,
        amp_ssim=0.86,
    )

    with pytest.raises(RuntimeExecutionError, match="selected gain"):
        gc.summarize_sweep(tmp_path)


def test_summarize_sweep_rejects_mismatched_confirmation_provenance(tmp_path):
    _write_complete_sweep(tmp_path)
    _write_rule_b_evidence(tmp_path, gain=20.0)
    _write_gain_evidence(
        tmp_path,
        tag="ruleb_confirm_wrong_provenance",
        gain=20.0,
        amp_ssim=0.86,
        git_commit="different-commit",
    )

    with pytest.raises(RuntimeExecutionError, match="confirmation.*provenance"):
        gc.summarize_sweep(tmp_path)


def test_summarize_sweep_rejects_subfloor_rule_b_confirmation(tmp_path):
    _write_complete_sweep(tmp_path)
    _write_rule_b_evidence(tmp_path, gain=20.0)
    _write_gain_evidence(
        tmp_path,
        tag="ruleb_confirm_subfloor",
        gain=20.0,
        amp_ssim=0.84,
    )

    with pytest.raises(RuntimeExecutionError, match="amp SSIM floor"):
        gc.summarize_sweep(tmp_path)


def test_summarize_sweep_rejects_nonfinite_rule_b_gain(tmp_path):
    _write_complete_sweep(tmp_path)
    _write_rule_b_evidence(tmp_path, gain=float("nan"))

    with pytest.raises(RuntimeExecutionError, match="Rule B gain.*finite"):
        gc.summarize_sweep(tmp_path)


def test_summarize_sweep_publishes_summary_atomically(tmp_path, monkeypatch):
    _write_complete_sweep(tmp_path)
    _write_rule_b_evidence(tmp_path)
    summary_path = tmp_path / "sweep_summary.json"
    direct_writes = []
    original_write_text = Path.write_text

    def record_write(path, *args, **kwargs):
        if path == summary_path:
            direct_writes.append(path)
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", record_write)

    gc.summarize_sweep(tmp_path)

    assert direct_writes == []
    assert json.loads(summary_path.read_text(encoding="utf-8"))["decision"][
        "complete"
    ] is True


def test_summarize_sweep_removes_stale_success_before_validation(tmp_path):
    _write_complete_sweep(tmp_path)
    _write_rule_b_evidence(tmp_path)
    summary_path = tmp_path / "sweep_summary.json"
    assert gc.summarize_sweep(tmp_path)["decision"]["complete"] is True
    assert summary_path.exists()

    point_path = tmp_path / "gain_64" / "gain_point.json"
    payload = json.loads(point_path.read_text(encoding="utf-8"))
    payload["checkpoint_sha256"] = "corrupt"
    point_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeExecutionError, match="checkpoint"):
        gc.summarize_sweep(tmp_path)

    assert not summary_path.exists()


# ---------------------------------------------------------------------------
# Evidence hygiene
# ---------------------------------------------------------------------------


def test_run_gain_point_uses_selected_checkpoint_for_derived_evidence(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from ptycho_torch import lightning_utils
    from ptycho_torch.model import PtychoPINN_Lightning
    from scripts.studies import grid_lines_torch_runner as runner_mod

    final_epoch_model = object()

    class SelectedCheckpointModel:
        def eval(self):
            return self

    selected_checkpoint_model = SelectedCheckpointModel()
    selected_checkpoint = tmp_path / "selected.ckpt"
    selected_checkpoint.write_bytes(b"selected checkpoint state")
    observed_models = []

    point = SimpleNamespace(
        config={"seed": 3, "epochs": 5},
        runner_cfg=object(),
        dataset=SimpleNamespace(id="lines128_reference"),
    )
    materialized = SimpleNamespace(
        recipe_fingerprint_sha256="recipe-sha",
        train_sha256="train-sha",
        test_sha256="test-sha",
        probe_sha256="probe-sha",
        n_train=8993,
        n_test=735,
    )
    results = {
        "history": {"train_loss": [1.0], "val_loss": [0.5]},
        "train_container": {"X": np.zeros((1, 1, 1, 1))},
        "test_container": {"X": np.zeros((1, 1, 1, 1))},
    }

    monkeypatch.setattr(gc, "resolve_baseline_point", lambda *a, **k: point)
    monkeypatch.setattr(gc, "validate_ladder_npz_pair", lambda *a, **k: materialized)
    monkeypatch.setattr(
        runner_mod,
        "load_cached_dataset_with_metadata",
        lambda *a, **k: ({"test": True}, {"metadata": True}),
    )
    monkeypatch.setattr(
        gc,
        "_train_dictionary",
        lambda *a, **k: (final_epoch_model, tmp_path / "work", results),
    )
    monkeypatch.setattr(
        lightning_utils, "find_best_checkpoint", lambda *a, **k: selected_checkpoint
    )
    monkeypatch.setattr(
        PtychoPINN_Lightning,
        "load_from_checkpoint",
        classmethod(lambda cls, *a, **k: selected_checkpoint_model),
    )

    def fake_evaluate(model, *args, **kwargs):
        observed_models.append(model)
        return {
            "amp_ssim": 0.90,
            "phase_ssim": 0.91,
            "amp_mae": 0.10,
            "phase_mae": 0.11,
        }

    def fake_decoder_stats(model, *args, **kwargs):
        observed_models.append(model)
        return {"label": kwargs["label"]}

    monkeypatch.setattr(gc, "_evaluate_point", fake_evaluate)
    monkeypatch.setattr(gc, "decoder_output_statistics", fake_decoder_stats)
    monkeypatch.setattr(gc, "_copy_metrics_csv", lambda *a, **k: None)
    monkeypatch.setattr(gc, "_git_commit", lambda: "commit-sha")

    evidence = gc.run_gain_point(
        gc.GainPointRequest(
            spec=LADDER_SPEC,
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_root=tmp_path / "evidence",
            gain=16.0,
            tag="gain_16",
        )
    )

    assert observed_models == [selected_checkpoint_model] * 3
    assert evidence["selected_checkpoint"] == str(selected_checkpoint)
    assert "best_checkpoint" not in evidence


@pytest.mark.parametrize("mode", ["point", "init_stats"])
@pytest.mark.parametrize("publication_case", ["failure", "race"])
def test_run_evidence_publication_is_atomic_and_no_clobber(
    tmp_path, monkeypatch, mode, publication_case
):
    from types import SimpleNamespace

    from ptycho_torch import lightning_utils
    from ptycho_torch.model import PtychoPINN_Lightning
    from scripts.studies import grid_lines_torch_runner as runner_mod

    class StubModel:
        def eval(self):
            return self

    model = StubModel()
    runner_cfg = _runner_cfg(tmp_path, epochs=0 if mode == "init_stats" else 5)
    point = SimpleNamespace(
        config={"seed": 3, "epochs": 0 if mode == "init_stats" else 5},
        runner_cfg=runner_cfg,
        dataset=SimpleNamespace(id="lines128_reference"),
    )
    materialized = SimpleNamespace(
        recipe_fingerprint_sha256="recipe-sha",
        train_sha256="train-sha",
        test_sha256="test-sha",
        probe_sha256="probe-sha",
        n_train=8993,
        n_test=735,
    )
    results = {
        "history": {"train_loss": [1.0], "val_loss": [0.5]},
        "train_container": {"X": np.zeros((1, 1, 1, 1))},
        "test_container": {"X": np.zeros((1, 1, 1, 1))},
    }
    checkpoint = tmp_path / "selected.ckpt"
    checkpoint.write_bytes(b"selected checkpoint state")

    monkeypatch.setattr(gc, "resolve_baseline_point", lambda *a, **k: point)
    monkeypatch.setattr(gc, "validate_ladder_npz_pair", lambda *a, **k: materialized)
    monkeypatch.setattr(
        runner_mod,
        "load_cached_dataset_with_metadata",
        lambda *a, **k: ({"test": True}, {"metadata": True}),
    )
    monkeypatch.setattr(
        gc, "_train_dictionary", lambda *a, **k: (model, tmp_path / "work", results)
    )
    monkeypatch.setattr(
        lightning_utils, "find_best_checkpoint", lambda *a, **k: checkpoint
    )
    monkeypatch.setattr(
        PtychoPINN_Lightning,
        "load_from_checkpoint",
        classmethod(lambda cls, *a, **k: model),
    )
    monkeypatch.setattr(
        gc,
        "_evaluate_point",
        lambda *a, **k: {
            "amp_ssim": 0.90,
            "phase_ssim": 0.91,
            "amp_mae": 0.10,
            "phase_mae": 0.11,
        },
    )
    monkeypatch.setattr(gc, "decoder_output_statistics", lambda *a, **k: {})
    monkeypatch.setattr(
        gc,
        "rule_b_scan",
        lambda *a, **k: {"rule": "init_scale_match", "seed": 3, "gain": 16.0},
    )
    monkeypatch.setattr(gc, "_copy_metrics_csv", lambda *a, **k: None)
    monkeypatch.setattr(gc, "_git_commit", lambda: "commit-sha")
    tag = "gain_16" if mode == "point" else "init_stats"
    request = gc.GainPointRequest(
        spec=LADDER_SPEC,
        train_npz=tmp_path / "train.npz",
        test_npz=tmp_path / "test.npz",
        output_root=tmp_path / "evidence",
        gain=16.0 if mode == "point" else 1.0,
        tag=tag,
    )
    destination = request.output_root / tag / (
        "gain_point.json" if mode == "point" else "rule_b_derivation.json"
    )
    competitor_bytes = b"competitor sealed evidence"
    original_replace = gc.os.replace
    original_link = gc.os.link

    if publication_case == "failure":
        def fail_publication(*args, **kwargs):
            raise OSError("injected")

        monkeypatch.setattr(gc.os, "replace", fail_publication)
        monkeypatch.setattr(gc.os, "link", fail_publication)
    else:
        def compete_then_replace(source, target):
            Path(target).write_bytes(competitor_bytes)
            return original_replace(source, target)

        def compete_then_link(source, target):
            Path(target).write_bytes(competitor_bytes)
            return original_link(source, target)

        monkeypatch.setattr(gc.os, "replace", compete_then_replace)
        monkeypatch.setattr(gc.os, "link", compete_then_link)

    with pytest.raises((OSError, RuntimeExecutionError)):
        (gc.run_gain_point if mode == "point" else gc.run_init_stats)(request)

    if publication_case == "race":
        assert destination.read_bytes() == competitor_bytes
    else:
        assert not destination.exists()
    assert list(destination.parent.glob(f".{destination.name}.*.tmp")) == []


def test_run_gain_point_refuses_existing_evidence(tmp_path):
    point_dir = tmp_path / "gain_16"
    point_dir.mkdir(parents=True)
    (point_dir / "gain_point.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeExecutionError, match="refusing"):
        gc.run_gain_point(
            gc.GainPointRequest(
                spec=LADDER_SPEC,
                train_npz=tmp_path / "train.npz",
                test_npz=tmp_path / "test.npz",
                output_root=tmp_path,
                gain=16.0,
                tag="gain_16",
            )
        )


def test_run_init_stats_refuses_existing_evidence(tmp_path):
    point_dir = tmp_path / "init_stats"
    point_dir.mkdir(parents=True)
    evidence_path = point_dir / "rule_b_derivation.json"
    evidence_path.write_text('{"sealed": true}', encoding="utf-8")

    with pytest.raises(RuntimeExecutionError, match="refusing"):
        gc.run_init_stats(
            gc.GainPointRequest(
                spec=LADDER_SPEC,
                train_npz=tmp_path / "train.npz",
                test_npz=tmp_path / "test.npz",
                output_root=tmp_path,
                gain=1.0,
                tag="init_stats",
            )
        )

    assert evidence_path.read_text(encoding="utf-8") == '{"sealed": true}'
