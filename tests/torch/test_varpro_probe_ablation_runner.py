"""Fast, no-training unit tests for scripts/studies/varpro_probe_ablation_runner.py.

Covers only the pure helpers per the Task 1.5 brief:
  (a) phase-alignment metric correctness
  (b) arm table resolves to the exact knob/variant sets
  (c) canvas/metrics writer round-trip on tiny synthetic data

No training, no Lightning, no subprocess -- these must stay fast.
"""
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "studies"))
import varpro_probe_ablation_runner as runner  # noqa: E402
import ablation_diagnostics  # noqa: E402
import ablation_figures  # noqa: E402


# ---------------------------------------------------------------------------
# (a) Phase-alignment metric correctness
# ---------------------------------------------------------------------------

def test_align_global_phase_removes_uniform_offset():
    rng = np.random.default_rng(0)
    truth = (rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))).astype(np.complex64)
    recon = truth * np.exp(1j * 0.7)

    aligned, truth_crop = runner.align_global_phase(recon, truth)

    np.testing.assert_allclose(aligned, truth_crop, atol=1e-4)


def test_compute_metrics_complex_mae_near_zero_after_alignment():
    rng = np.random.default_rng(1)
    truth = (rng.normal(size=(20, 20)) + 1j * rng.normal(size=(20, 20))).astype(np.complex64)
    recon = truth * np.exp(1j * 0.7)

    metrics = runner.compute_metrics(recon, truth)

    assert metrics["complex_mae"] < 1e-3
    assert metrics["amp_mae"] < 1e-3
    assert metrics["phase_mae"] < 1e-3


def test_compute_metrics_nonzero_for_genuinely_different_arrays():
    rng = np.random.default_rng(2)
    truth = (rng.normal(size=(20, 20)) + 1j * rng.normal(size=(20, 20))).astype(np.complex64)
    recon = (rng.normal(size=(20, 20)) + 1j * rng.normal(size=(20, 20))).astype(np.complex64)

    metrics = runner.compute_metrics(recon, truth)

    assert metrics["complex_mae"] > 0.1


def test_align_global_phase_crops_to_overlap_when_shapes_differ():
    rng = np.random.default_rng(3)
    truth = (rng.normal(size=(24, 24)) + 1j * rng.normal(size=(24, 24))).astype(np.complex64)
    # Recon is a smaller, *centered* canvas (simulating a differently-sized but
    # co-registered stitched region -- center-cropping only recovers a common
    # overlap when both arrays share the same physical center).
    recon = truth[4:20, 4:20] * np.exp(1j * 1.3)

    aligned, truth_crop = runner.align_global_phase(recon, truth)

    assert aligned.shape == (16, 16)
    assert truth_crop.shape == (16, 16)
    np.testing.assert_allclose(aligned, truth_crop, atol=1e-4)


# ---------------------------------------------------------------------------
# (b) Arm table resolution
# ---------------------------------------------------------------------------

def test_arm_names_lists_all_arms():
    assert set(runner.arm_names()) == {
        "gs1_frozen", "gs1_trainable",
        "gs2_neither", "gs2_probe_frozen", "gs2_probe_trainable",
        "gs2_neither_n128", "gs2_probe_trainable_n128",
        "repr_ampphase", "repr_realimag",
    }


@pytest.mark.parametrize(
    "arm,expected",
    [
        ("gs1_frozen", {
            "N": 64, "gridsize": 1, "training_patch_weighting": "probe",
            "rect_s1s2_trainable": False, "nphotons": 1e9,
            "physics_forward_mode": "rectangular_scaled",
            "cnn_output_mode": "real_imag",
            "variants": ["uniform_novarpro", "uniform_varpro", "probe_novarpro", "probe_varpro"],
        }),
        ("gs1_trainable", {
            "N": 64, "gridsize": 1, "training_patch_weighting": "probe",
            "rect_s1s2_trainable": True, "nphotons": 1e9,
            "physics_forward_mode": "rectangular_scaled",
            "cnn_output_mode": "real_imag",
            "variants": ["uniform_novarpro", "uniform_varpro", "probe_novarpro", "probe_varpro"],
        }),
        ("gs2_neither", {
            "N": 64, "gridsize": 2, "training_patch_weighting": "uniform",
            "rect_s1s2_trainable": False, "nphotons": 1e9,
            "physics_forward_mode": "rectangular_scaled",
            "cnn_output_mode": "real_imag",
            "variants": ["probe_varpro", "uniform_novarpro"],
        }),
        ("gs2_probe_frozen", {
            "N": 64, "gridsize": 2, "training_patch_weighting": "probe",
            "rect_s1s2_trainable": False, "nphotons": 1e9,
            "physics_forward_mode": "rectangular_scaled",
            "cnn_output_mode": "real_imag",
            "variants": ["probe_varpro", "uniform_novarpro"],
        }),
        ("gs2_probe_trainable", {
            "N": 64, "gridsize": 2, "training_patch_weighting": "probe",
            "rect_s1s2_trainable": True, "nphotons": 1e9,
            "physics_forward_mode": "rectangular_scaled",
            "cnn_output_mode": "real_imag",
            "variants": ["probe_varpro", "uniform_novarpro"],
        }),
        ("gs2_neither_n128", {
            "N": 128, "gridsize": 2, "training_patch_weighting": "uniform",
            "rect_s1s2_trainable": False, "nphotons": 1e9,
            "physics_forward_mode": "rectangular_scaled",
            "cnn_output_mode": "real_imag",
            "variants": ["probe_varpro", "uniform_novarpro"],
        }),
        ("gs2_probe_trainable_n128", {
            "N": 128, "gridsize": 2, "training_patch_weighting": "probe",
            "rect_s1s2_trainable": True, "nphotons": 1e9,
            "physics_forward_mode": "rectangular_scaled",
            "cnn_output_mode": "real_imag",
            "variants": ["probe_varpro", "uniform_novarpro"],
        }),
        ("repr_ampphase", {
            "N": 64, "gridsize": 1, "training_patch_weighting": "probe",
            "rect_s1s2_trainable": False, "nphotons": 1e9,
            "architecture": "cnn", "cnn_output_mode": "amp_phase",
            "variants": ["probe_varpro"],
        }),
        ("repr_realimag", {
            "N": 64, "gridsize": 1, "training_patch_weighting": "probe",
            "rect_s1s2_trainable": False, "nphotons": 1e9,
            "architecture": "cnn", "cnn_output_mode": "real_imag",
            "variants": ["probe_varpro"],
        }),
    ],
)
def test_resolve_arm_matches_exact_knob_and_variant_dicts(arm, expected):
    assert runner.resolve_arm(arm) == expected


def test_resolve_variants_expands_to_patch_weighting_and_varpro_dicts():
    variants = runner.resolve_variants("gs1_frozen")
    assert variants == {
        "uniform_novarpro": {"patch_weighting": "uniform", "varpro_scaling": False},
        "uniform_varpro":   {"patch_weighting": "uniform", "varpro_scaling": True},
        "probe_novarpro":   {"patch_weighting": "probe",   "varpro_scaling": False},
        "probe_varpro":     {"patch_weighting": "probe",   "varpro_scaling": True},
    }

    gs2_variants = runner.resolve_variants("gs2_neither")
    assert gs2_variants == {
        "probe_varpro":   {"patch_weighting": "probe",   "varpro_scaling": True},
        "uniform_novarpro": {"patch_weighting": "uniform", "varpro_scaling": False},
    }


def test_resolve_arm_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown arm"):
        runner.resolve_arm("not_a_real_arm")


# ---------------------------------------------------------------------------
# (c) Canvas / metrics writer round-trip
# ---------------------------------------------------------------------------

def test_canvas_npz_round_trip(tmp_path):
    rng = np.random.default_rng(4)
    canvas = (rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))).astype(np.complex64)
    path = tmp_path / "canvas.npz"

    runner.write_canvas_npz(path, canvas)
    loaded = runner.read_canvas_npz(path)

    assert loaded.dtype == np.complex64
    np.testing.assert_allclose(loaded, canvas, atol=1e-6)


def test_metrics_json_round_trip(tmp_path):
    metrics = {"complex_mae": 0.01, "amp_mae": 0.02, "phase_mae": 0.03, "s1": 1.0, "s2": 1.0}
    path = tmp_path / "metrics.json"

    runner.write_metrics_json(path, metrics)
    loaded = runner.read_metrics_json(path)

    assert loaded == metrics
    # Also confirm it's genuinely on disk as valid JSON (not just round-trippable
    # through our own reader).
    assert json.loads(path.read_text()) == metrics


# ---------------------------------------------------------------------------
# (d) F1 -- object_big derived from gridsize in build_configs()
# ---------------------------------------------------------------------------

def test_build_configs_derives_object_big_false_for_gridsize1_arm():
    arm_cfg = runner.resolve_arm("gs1_frozen")

    _, model_config, _, _, _ = runner.build_configs(arm_cfg, batch_size=2, epochs=1)

    assert model_config.object_big is False


def test_build_configs_derives_object_big_true_for_gridsize2_arm():
    arm_cfg = runner.resolve_arm("gs2_neither")

    _, model_config, _, _, _ = runner.build_configs(arm_cfg, batch_size=2, epochs=1)

    assert model_config.object_big is True


# ---------------------------------------------------------------------------
# (e) F3 -- validate_arm_outputs (ablation_diagnostics.py)
# ---------------------------------------------------------------------------

def _write_variant(variant_dir: Path, complex_mae: float = 0.1, canvas: Optional[np.ndarray] = None) -> None:
    variant_dir.mkdir(parents=True, exist_ok=True)
    if canvas is None:
        canvas = np.ones((4, 4), dtype=np.complex64)
    runner.write_canvas_npz(variant_dir / "canvas.npz", canvas)
    metrics = {"complex_mae": complex_mae, "amp_mae": 0.05, "phase_mae": 0.02, "s1": 1.0, "s2": 1.0}
    runner.write_metrics_json(variant_dir / "metrics.json", metrics)
    (variant_dir / "recon_panel.png").write_bytes(b"fake-png")
    (variant_dir / "error.png").write_bytes(b"fake-png")


def test_validate_arm_outputs_clean_when_complete_and_valid(tmp_path):
    arm_dir = tmp_path / "arm"
    _write_variant(arm_dir / "uniform_novarpro", canvas=np.ones((4, 4), dtype=np.complex64))
    _write_variant(arm_dir / "probe_novarpro", canvas=np.ones((4, 4), dtype=np.complex64) * 2)

    problems = ablation_diagnostics.validate_arm_outputs(arm_dir, ["uniform_novarpro", "probe_novarpro"])

    assert problems == []


def test_validate_arm_outputs_reports_missing_file(tmp_path):
    arm_dir = tmp_path / "arm"
    _write_variant(arm_dir / "uniform_novarpro")
    (arm_dir / "uniform_novarpro" / "error.png").unlink()

    problems = ablation_diagnostics.validate_arm_outputs(arm_dir, ["uniform_novarpro"])

    assert any("missing error.png" in p for p in problems)


def test_validate_arm_outputs_reports_non_finite_metric(tmp_path):
    arm_dir = tmp_path / "arm"
    _write_variant(arm_dir / "uniform_novarpro", complex_mae=float("nan"))

    problems = ablation_diagnostics.validate_arm_outputs(arm_dir, ["uniform_novarpro"])

    assert any("non-finite" in p for p in problems)


def test_validate_arm_outputs_reports_identical_probe_uniform_canvases(tmp_path):
    arm_dir = tmp_path / "arm"
    same_canvas = np.ones((4, 4), dtype=np.complex64)
    _write_variant(arm_dir / "uniform_novarpro", canvas=same_canvas)
    _write_variant(arm_dir / "probe_novarpro", canvas=same_canvas)

    problems = ablation_diagnostics.validate_arm_outputs(arm_dir, ["uniform_novarpro", "probe_novarpro"])

    assert any("identical" in p for p in problems)


# ---------------------------------------------------------------------------
# (f) F4 -- canvas_rail_diagnostics (ablation_diagnostics.py)
# ---------------------------------------------------------------------------

def test_canvas_rail_diagnostics_constant_rail_canvas_reads_near_one():
    canvas = np.full((8, 8), 1.2 - 1.2j, dtype=np.complex64)

    diag = ablation_diagnostics.canvas_rail_diagnostics(canvas)

    assert diag["rail_fraction_real"] > 0.99
    assert diag["rail_fraction_imag"] > 0.99
    assert diag["canvas_phase_std"] < 1e-6


def test_canvas_rail_diagnostics_random_canvas_reads_near_zero():
    rng = np.random.default_rng(5)
    canvas = (rng.uniform(-2, 2, size=(64, 64)) + 1j * rng.uniform(-2, 2, size=(64, 64))).astype(np.complex64)

    diag = ablation_diagnostics.canvas_rail_diagnostics(canvas)

    assert diag["rail_fraction_real"] < 0.01
    assert diag["rail_fraction_imag"] < 0.01


# ---------------------------------------------------------------------------
# (g) F2 -- combined per-arm comparison grids (ablation_figures.py)
# ---------------------------------------------------------------------------

def test_save_reconstruction_and_error_grid_write_nonempty_files(tmp_path):
    rng = np.random.default_rng(6)
    truth_crop = (rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))).astype(np.complex64)
    aligned_variants = {
        "uniform_novarpro": truth_crop + 0.01,
        "probe_novarpro": truth_crop + 0.02,
    }
    recon_path = tmp_path / "reconstruction_grid.png"
    error_path = tmp_path / "error_grid.png"

    ablation_figures.save_reconstruction_grid(recon_path, truth_crop, aligned_variants)
    ablation_figures.save_error_grid(error_path, truth_crop, aligned_variants)

    assert recon_path.exists() and recon_path.stat().st_size > 0
    assert error_path.exists() and error_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# (h) Fix wave 3 -- build_variant_metrics computes degeneracy diagnostics on
# the PRE-varpro canvas, not the final (possibly rescaled) one.
# ---------------------------------------------------------------------------

def test_build_variant_metrics_computes_rail_diagnostics_on_prescale_canvas():
    truth = np.ones((8, 8), dtype=np.complex64)
    # Final (post-VarPro) canvas: nowhere near the tanh rails.
    recon = np.full((8, 8), 0.05 + 0.05j, dtype=np.complex64)
    # Pre-scale canvas: saturated on both rails everywhere -- if the
    # diagnostics were (incorrectly) computed on `recon` instead, these rail
    # fractions would read ~0, hiding a degenerate model (see Fix wave 2's
    # gs1_frozen smoke finding: 0.66 rail fraction for *_novarpro variants but
    # 0.0 for *_varpro variants of the identical checkpoint).
    prescale_canvas = np.full((8, 8), 1.2 - 1.2j, dtype=np.complex64)

    metrics = runner.build_variant_metrics(recon, truth, prescale_canvas, s1=0.05, s2=0.05)

    assert metrics["rail_fraction_real"] > 0.99
    assert metrics["rail_fraction_imag"] > 0.99
    assert metrics["diagnostics_basis"] == "prescale_canvas"
    assert metrics["s1"] == 0.05
    assert metrics["s2"] == 0.05


def test_build_variant_metrics_matches_compute_metrics_on_final_canvas():
    rng = np.random.default_rng(7)
    truth = (rng.normal(size=(12, 12)) + 1j * rng.normal(size=(12, 12))).astype(np.complex64)
    recon = truth * np.exp(1j * 0.4)
    prescale_canvas = np.zeros((12, 12), dtype=np.complex64)

    metrics = runner.build_variant_metrics(recon, truth, prescale_canvas, s1=1.0, s2=1.0)
    expected = runner.compute_metrics(recon, truth)

    assert metrics["complex_mae"] == pytest.approx(expected["complex_mae"])
    assert metrics["amp_mae"] == pytest.approx(expected["amp_mae"])
    assert metrics["phase_mae"] == pytest.approx(expected["phase_mae"])


# ---------------------------------------------------------------------------
# (i) Task E2 -- comparison arms: build_configs routing for the new
# training-time knobs (architecture, cnn_output_mode, physics_forward_mode),
# a Phase-1 defaults-preserved regression, and the --architecture /
# --cnn-output-mode CLI overrides.
# ---------------------------------------------------------------------------

def test_repr_ampphase_routes_cnn_amp_phase_and_probe_varpro_variant():
    arm_cfg = runner.resolve_arm("repr_ampphase")

    _, model_config, _, _, _ = runner.build_configs(arm_cfg, batch_size=2, epochs=1)

    assert model_config.architecture == "cnn"
    assert model_config.cnn_output_mode == "amp_phase"
    assert model_config.object_big is False
    assert runner.resolve_variants("repr_ampphase") == {
        "probe_varpro": {"patch_weighting": "probe", "varpro_scaling": True},
    }


def test_repr_realimag_routes_cnn_real_imag():
    arm_cfg = runner.resolve_arm("repr_realimag")

    _, model_config, _, _, _ = runner.build_configs(arm_cfg, batch_size=2, epochs=1)

    assert model_config.architecture == "cnn"
    assert model_config.cnn_output_mode == "real_imag"


def test_phase1_arm_model_config_matches_dataclass_defaults_for_new_knobs():
    """architecture stays at its ModelConfig default for a Phase-1 base arm
    (unaffected by Task B1). physics_forward_mode and cnn_output_mode are the
    deliberate exceptions: Task B1 pins them to 'rectangular_scaled' /
    'real_imag' for main parity (PORT-ABLATION-FWD-001) -- main's PT model has
    no amplitude forward and no amp_phase head -- so they intentionally no
    longer match the ModelConfig defaults ('amplitude' / 'amp_phase'). See
    test_base_arms_pin_rectangular_scaled_forward and
    test_base_arms_pin_real_imag_output_mode for those contracts.
    """
    from ptycho_torch.config_params import ModelConfig

    arm_cfg = runner.resolve_arm("gs1_frozen")

    _, model_config, _, _, _ = runner.build_configs(arm_cfg, batch_size=2, epochs=1)

    defaults = ModelConfig()
    assert model_config.architecture == defaults.architecture
    assert model_config.cnn_output_mode == "real_imag"
    assert model_config.physics_forward_mode == "rectangular_scaled"


def test_cli_architecture_and_cnn_output_mode_overrides_reach_model_config():
    parser = runner.build_arg_parser()
    args = parser.parse_args([
        "--arm", "gs1_frozen",
        "--train-npz", "dummy_train.npz",
        "--test-npz", "dummy_test.npz",
        "--output-root", "dummy_out",
        "--architecture", "fno",
        "--cnn-output-mode", "real_imag",
    ])

    arm_cfg = runner.resolve_arm_with_overrides(
        args.arm, architecture=args.architecture, cnn_output_mode=args.cnn_output_mode,
    )
    _, model_config, _, _, _ = runner.build_configs(arm_cfg, batch_size=2, epochs=1)

    assert model_config.architecture == "fno"
    assert model_config.cnn_output_mode == "real_imag"


# ---------------------------------------------------------------------------
# (j) Task B1 -- main parity: base arms must pin BOTH the rectangular_scaled
# forward AND the real_imag head. On origin/main there is no
# physics_forward_mode knob (the rectangular-scaled forward is the only
# training forward) and no amp_phase head (main's PT model unconditionally
# builds real/imag decoder branches, recombined via CombineComplexRectangular).
# The ported ARM_TABLE base arms must pin both explicitly so they don't
# silently fall back to fno-stable's ModelConfig defaults ('amplitude' /
# 'amp_phase') -- finding PORT-ABLATION-FWD-001. The two pins are also coupled
# on fno-stable: the model fail-fasts on rectangular_scaled unless the
# resolved head output is real_imag (ptycho_torch/model.py:1863-1870).
# ---------------------------------------------------------------------------

_BASE_ARMS = [
    "gs1_frozen", "gs1_trainable",
    "gs2_neither", "gs2_probe_frozen", "gs2_probe_trainable",
    "gs2_neither_n128", "gs2_probe_trainable_n128",
]


@pytest.mark.parametrize("arm", _BASE_ARMS)
def test_base_arms_pin_rectangular_scaled_forward(arm):
    arm_cfg = runner.resolve_arm(arm)

    assert arm_cfg["physics_forward_mode"] == "rectangular_scaled"


@pytest.mark.parametrize("arm", _BASE_ARMS)
def test_base_arms_pin_real_imag_output_mode(arm):
    arm_cfg = runner.resolve_arm(arm)

    assert arm_cfg["cnn_output_mode"] == "real_imag"


def test_gs2_probe_frozen_build_configs_carries_rectangular_scaled_forward():
    arm_cfg = runner.resolve_arm("gs2_probe_frozen")

    _, model_config, _, _, _ = runner.build_configs(arm_cfg, batch_size=2, epochs=1)

    assert model_config.physics_forward_mode == "rectangular_scaled"


def test_gs2_probe_frozen_build_configs_carries_real_imag_output_mode():
    arm_cfg = runner.resolve_arm("gs2_probe_frozen")

    _, model_config, _, _, _ = runner.build_configs(arm_cfg, batch_size=2, epochs=1)

    assert model_config.cnn_output_mode == "real_imag"


def test_gs1_frozen_build_configs_carries_rectangular_scaled_forward():
    arm_cfg = runner.resolve_arm("gs1_frozen")

    _, model_config, _, _, _ = runner.build_configs(arm_cfg, batch_size=2, epochs=1)

    assert model_config.physics_forward_mode == "rectangular_scaled"


def test_gs1_frozen_build_configs_carries_real_imag_output_mode():
    arm_cfg = runner.resolve_arm("gs1_frozen")

    _, model_config, _, _, _ = runner.build_configs(arm_cfg, batch_size=2, epochs=1)

    assert model_config.cnn_output_mode == "real_imag"


def test_cli_overrides_default_to_none_and_leave_arm_cfg_unchanged():
    parser = runner.build_arg_parser()
    args = parser.parse_args([
        "--arm", "gs1_frozen",
        "--train-npz", "dummy_train.npz",
        "--test-npz", "dummy_test.npz",
        "--output-root", "dummy_out",
    ])

    arm_cfg = runner.resolve_arm_with_overrides(
        args.arm, architecture=args.architecture, cnn_output_mode=args.cnn_output_mode,
    )

    assert arm_cfg == runner.resolve_arm("gs1_frozen")


# ---------------------------------------------------------------------------
# (l) N=128 support -- --N CLI override plumbed through resolve_arm_with_overrides
# into build_configs' DataConfig, plus a startup fail-fast when the effective N
# mismatches the train npz's diff3d frame size (pins the 2026-07-07 crash --
# a RuntimeError deep in mmap probe allocation when ARM_TABLE's N=64 was
# silently run against a 128-frame npz -- into a one-line ValueError).
# ---------------------------------------------------------------------------

def test_cli_n_override_reaches_data_config():
    parser = runner.build_arg_parser()
    args = parser.parse_args([
        "--arm", "gs1_frozen",
        "--train-npz", "dummy_train.npz",
        "--test-npz", "dummy_test.npz",
        "--output-root", "dummy_out",
        "--N", "128",
    ])

    arm_cfg = runner.resolve_arm_with_overrides(
        args.arm, architecture=args.architecture, cnn_output_mode=args.cnn_output_mode,
        N=args.N,
    )
    data_config, _, _, _, _ = runner.build_configs(arm_cfg, batch_size=2, epochs=1)

    assert arm_cfg["N"] == 128
    assert data_config.N == 128


def test_validate_n_matches_train_npz_raises_on_mismatch(tmp_path):
    train_npz = tmp_path / "lines_N128_train.npz"
    diff3d = np.zeros((2, 128, 128), dtype=np.float32)
    np.savez(train_npz, diff3d=diff3d)

    with pytest.raises(ValueError, match=r"--N") as excinfo:
        runner.validate_n_matches_train_npz(64, train_npz)

    assert "64" in str(excinfo.value)
    assert "128" in str(excinfo.value)


def test_validate_n_matches_train_npz_passes_when_n_matches(tmp_path):
    train_npz = tmp_path / "lines_N64_train.npz"
    diff3d = np.zeros((2, 64, 64), dtype=np.float32)
    np.savez(train_npz, diff3d=diff3d)

    # Must not raise.
    runner.validate_n_matches_train_npz(64, train_npz)


def test_cli_n_override_defaults_to_none_and_leaves_arm_cfg_unchanged():
    parser = runner.build_arg_parser()
    args = parser.parse_args([
        "--arm", "gs1_frozen",
        "--train-npz", "dummy_train.npz",
        "--test-npz", "dummy_test.npz",
        "--output-root", "dummy_out",
    ])

    arm_cfg = runner.resolve_arm_with_overrides(
        args.arm, architecture=args.architecture, cnn_output_mode=args.cnn_output_mode,
        N=args.N,
    )

    assert arm_cfg == runner.resolve_arm("gs1_frozen")


# ---------------------------------------------------------------------------
# (m) Task E2 -- --physics-forward-mode CLI override, plumbed through
# resolve_arm_with_overrides exactly like --architecture/--cnn-output-mode/--N.
# ---------------------------------------------------------------------------

def test_cli_physics_forward_mode_override_reaches_model_config():
    parser = runner.build_arg_parser()
    args = parser.parse_args([
        "--arm", "gs1_trainable",
        "--train-npz", "dummy_train.npz",
        "--test-npz", "dummy_test.npz",
        "--output-root", "dummy_out",
        "--physics-forward-mode", "amplitude",
    ])

    arm_cfg = runner.resolve_arm_with_overrides(
        args.arm, architecture=args.architecture, cnn_output_mode=args.cnn_output_mode,
        N=args.N, physics_forward_mode=args.physics_forward_mode,
    )
    _, model_config, _, _, _ = runner.build_configs(arm_cfg, batch_size=2, epochs=1)

    assert arm_cfg["physics_forward_mode"] == "amplitude"
    assert model_config.physics_forward_mode == "amplitude"


def test_cli_physics_forward_mode_override_defaults_to_none_and_leaves_arm_cfg_unchanged():
    parser = runner.build_arg_parser()
    args = parser.parse_args([
        "--arm", "gs1_trainable",
        "--train-npz", "dummy_train.npz",
        "--test-npz", "dummy_test.npz",
        "--output-root", "dummy_out",
    ])

    arm_cfg = runner.resolve_arm_with_overrides(
        args.arm, architecture=args.architecture, cnn_output_mode=args.cnn_output_mode,
        N=args.N, physics_forward_mode=args.physics_forward_mode,
    )

    assert arm_cfg == runner.resolve_arm("gs1_trainable")
    assert arm_cfg["physics_forward_mode"] == "rectangular_scaled"


@pytest.mark.parametrize("mode", ["amplitude", "rectangular_scaled"])
def test_cli_physics_forward_mode_accepts_valid_choices(mode):
    parser = runner.build_arg_parser()
    args = parser.parse_args([
        "--arm", "gs1_trainable",
        "--train-npz", "dummy_train.npz",
        "--test-npz", "dummy_test.npz",
        "--output-root", "dummy_out",
        "--physics-forward-mode", mode,
    ])

    assert args.physics_forward_mode == mode


def test_cli_physics_forward_mode_rejects_invalid_choice():
    parser = runner.build_arg_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([
            "--arm", "gs1_trainable",
            "--train-npz", "dummy_train.npz",
            "--test-npz", "dummy_test.npz",
            "--output-root", "dummy_out",
            "--physics-forward-mode", "garbage",
        ])


# ---------------------------------------------------------------------------
# (k) Task B4a -- object-frame direct-placement metric (pure numpy, no torch;
# synthetic patches built in-test per the brief -- .artifacts datasets are
# frozen and must not be read from tests).
# ---------------------------------------------------------------------------

def _make_objframe_oracle(obj_size=64, patch_size=16, stride=8, seed=11):
    """Build a synthetic dead-leaves-like complex object plus ground-truth
    patches cut at integer coords_global, mirroring the B4 report's oracle
    construction (stitching ground_truth_patches at their npz coords)."""
    rng = np.random.default_rng(seed)
    truth = (rng.normal(size=(obj_size, obj_size)) + 1j * rng.normal(size=(obj_size, obj_size))).astype(np.complex128)
    half = patch_size // 2
    coords = []
    patches = []
    for cy in range(half, obj_size - half, stride):
        for cx in range(half, obj_size - half, stride):
            coords.append((cx, cy))  # col0=x, col1=y
            patches.append(truth[cy - half:cy + half, cx - half:cx + half].copy())
    return truth, np.stack(patches), np.array(coords, dtype=np.float64)


def test_place_patches_objframe_reproduces_truth_on_integer_coords():
    truth, patches, coords = _make_objframe_oracle()

    canvas, coverage_mask, n_used, n_total = runner.place_patches_objframe(
        patches, coords, truth.shape, patch_size=16,
    )

    assert n_used == n_total
    np.testing.assert_allclose(canvas[coverage_mask], truth[coverage_mask], atol=1e-8)


def test_place_patches_objframe_skips_out_of_bounds_without_dropping_in_bounds():
    truth, patches, coords = _make_objframe_oracle()
    # Append one deliberately out-of-bounds coordinate (off the object edge).
    patches = np.concatenate([patches, patches[:1]], axis=0)
    coords = np.concatenate([coords, [[-100.0, -100.0]]], axis=0)

    canvas, coverage_mask, n_used, n_total = runner.place_patches_objframe(
        patches, coords, truth.shape, patch_size=16,
    )

    assert n_total == patches.shape[0]
    assert n_used == patches.shape[0] - 1
    np.testing.assert_allclose(canvas[coverage_mask], truth[coverage_mask], atol=1e-8)


def test_compute_objframe_metrics_oracle_exceeds_report_thresholds():
    """Oracle test (TDD): stitching ground-truth-like synthetic patches at
    known integer coords through the object-frame metric path reproduces the
    truth region with corr > 0.99 and gauged MAE < 0.02, matching the B4
    report's oracle numbers (corr 0.982, MAE 0.0117) for direct object-frame
    placement -- built in tmp_path/in-test synthetic data only, per the brief.
    """
    truth, patches, coords = _make_objframe_oracle()

    metrics = runner.compute_objframe_metrics(patches, coords, truth, patch_size=16)

    assert metrics["amp_pearson_objframe"] > 0.99
    assert metrics["amp_mae_objframe_gauged"] < 0.02
    assert metrics["patch_amp_pearson_mean"] > 0.99
    assert metrics["patch_lsq_scalar_median"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["n_patches_used"] == metrics["n_patches_total"] == patches.shape[0]
    assert 0.0 < metrics["coverage_fraction"] <= 1.0


def test_compute_objframe_metrics_scale_and_phase_invariant_gauged_mae():
    """A uniform complex scale+phase applied to every patch must be absorbed
    by the single complex LSQ gauge -- gauged MAE stays near zero even though
    the raw (ungauged) reconstruction differs from truth by a large factor."""
    truth, patches, coords = _make_objframe_oracle()
    scaled_patches = patches * (3.7 * np.exp(1j * 0.9))

    metrics = runner.compute_objframe_metrics(scaled_patches, coords, truth, patch_size=16)

    assert metrics["amp_mae_objframe_gauged"] < 0.02
    assert metrics["amp_pearson_objframe"] > 0.99


def test_compute_objframe_metrics_low_for_uncorrelated_patches():
    truth, patches, coords = _make_objframe_oracle(seed=11)
    rng = np.random.default_rng(99)
    noise_patches = (rng.normal(size=patches.shape) + 1j * rng.normal(size=patches.shape))

    metrics = runner.compute_objframe_metrics(noise_patches, coords, truth, patch_size=16)

    assert metrics["amp_pearson_objframe"] < 0.5
    assert metrics["patch_amp_pearson_mean"] < 0.5


def test_compute_objframe_metrics_raises_when_all_patches_out_of_bounds():
    truth, patches, coords = _make_objframe_oracle()
    coords = np.full_like(coords, -1000.0)

    with pytest.raises(ValueError, match="0/"):
        runner.compute_objframe_metrics(patches, coords, truth, patch_size=16)


# ---------------------------------------------------------------------------
# (l) collect_objframe_patches -- chunked forward must match the single-batch
# reference byte-for-byte (OOM fix; synthetic dataset + deterministic model
# stub built in-test, no checkpoint loaded).
# ---------------------------------------------------------------------------

class _StubObjframeDataset:
    """Minimal dataset stub mirroring PtychoDataset.__getitem__'s contract:
    indexing by a list of ints returns (tensor_dict, probes_indexed,
    probe_scaling), where tensor_dict is dict-like with the keys
    collect_objframe_patches reads. Shapes follow dataloader.py's mmap
    convention: images (n, C, N, N), coords_relative/coords_global
    (n, C, 1, 2), rms_scaling_constant (n, 1, 1, 1)."""

    def __init__(self, n, image_size=8, n_channels=1, seed=5):
        rng = np.random.default_rng(seed)
        self.n = n
        self.image_size = image_size
        self.n_channels = n_channels
        self.images = torch.tensor(
            rng.normal(size=(n, n_channels, image_size, image_size)), dtype=torch.float32,
        )
        self.coords_relative = torch.zeros((n, n_channels, 1, 2), dtype=torch.float32)
        coords_global = torch.zeros((n, n_channels, 1, 2), dtype=torch.float32)
        # Index-dependent coordinates so any row reordering is detectable.
        coords_global[:, 0, 0, 0] = torch.arange(n, dtype=torch.float32)
        coords_global[:, 0, 0, 1] = torch.arange(n, dtype=torch.float32) * 2.0
        self.coords_global = coords_global
        self.rms_scaling_constant = torch.ones((n, 1, 1, 1), dtype=torch.float32)

    def __len__(self):
        return self.n

    def __getitem__(self, idx_list):
        idx = torch.as_tensor(list(idx_list), dtype=torch.long)
        tensor_dict = {
            "images": self.images[idx],
            "coords_relative": self.coords_relative[idx],
            "coords_global": self.coords_global[idx],
            "rms_scaling_constant": self.rms_scaling_constant[idx],
        }
        probe = torch.ones(
            (len(idx), self.n_channels, 1, self.image_size, self.image_size), dtype=torch.float32,
        )
        probe_scaling = torch.ones(len(idx), dtype=torch.float32)
        return tensor_dict, probe, probe_scaling


class _StubObjframeModel:
    """Deterministic, batch-size-independent stand-in for forward_predict:
    a pure elementwise function of ``images`` only, so per-chunk results are
    identical to the single-batch reference regardless of chunk boundaries."""

    def forward_predict(self, images, coords_relative, probe, rms_scaling_constant):
        return torch.complex(images, images * 2.0 + 1.0)


def test_collect_objframe_patches_chunked_matches_full_batch_reference(monkeypatch):
    from ptycho_torch.dataloader import Collate_Lightning

    n = 10
    monkeypatch.setattr(runner, "_OBJFRAME_FORWARD_CHUNK", 4)  # forces chunks 4+4+2
    dataset = _StubObjframeDataset(n)
    model = _StubObjframeModel()
    middle_trim = 4

    # Reference: the old single-batch path, computed independently in-test.
    batch = Collate_Lightning(False)(dataset[list(range(n))])
    tensor_dict, probe = batch[0], batch[1]
    with torch.no_grad():
        raw = model.forward_predict(
            tensor_dict["images"], tensor_dict["coords_relative"], probe,
            tensor_dict["rms_scaling_constant"],
        )
    expected_patches = runner._center_crop(
        raw.numpy(), (middle_trim, middle_trim),
    ).reshape(-1, middle_trim, middle_trim)
    expected_coords = tensor_dict["coords_global"].squeeze(2).numpy().reshape(-1, 2)

    patches, coords_global = runner.collect_objframe_patches(model, dataset, middle_trim)

    assert patches.shape == expected_patches.shape
    assert patches.dtype == expected_patches.dtype
    np.testing.assert_array_equal(patches, expected_patches)
    assert coords_global.shape == expected_coords.shape
    np.testing.assert_allclose(coords_global, expected_coords, atol=0)


def test_build_variant_metrics_merges_objframe_metrics_and_relabels_basis():
    truth = np.ones((8, 8), dtype=np.complex64)
    recon = np.full((8, 8), 0.05 + 0.05j, dtype=np.complex64)
    prescale_canvas = np.full((8, 8), 1.2 - 1.2j, dtype=np.complex64)
    objframe_metrics = {
        "amp_mae_objframe_gauged": 0.01,
        "amp_pearson_objframe": 0.98,
        "patch_amp_pearson_mean": 0.97,
        "patch_lsq_scalar_median": 1.02,
        "coverage_fraction": 0.9,
        "n_patches_used": 57,
        "n_patches_total": 59,
    }

    metrics = runner.build_variant_metrics(
        recon, truth, prescale_canvas, s1=0.05, s2=0.05, objframe_metrics=objframe_metrics,
    )

    # Existing (center-crop-basis) keys stay intact for cross-run comparability.
    assert "amp_mae" in metrics and "complex_mae" in metrics
    assert metrics["rail_fraction_real"] > 0.99
    # New objframe keys are merged in verbatim.
    for key, value in objframe_metrics.items():
        assert metrics[key] == value
    # diagnostics_basis names both bases so downstream readers can disambiguate.
    assert metrics["diagnostics_basis"] == "prescale_canvas+objframe_direct"


def test_build_variant_metrics_without_objframe_metrics_keeps_legacy_basis_label():
    """Backward compatibility: omitting objframe_metrics (the default) leaves
    diagnostics_basis and the key set unchanged from before Task B4a."""
    truth = np.ones((8, 8), dtype=np.complex64)
    recon = np.full((8, 8), 0.05 + 0.05j, dtype=np.complex64)
    prescale_canvas = np.full((8, 8), 1.2 - 1.2j, dtype=np.complex64)

    metrics = runner.build_variant_metrics(recon, truth, prescale_canvas, s1=0.05, s2=0.05)

    assert metrics["diagnostics_basis"] == "prescale_canvas"
    assert "amp_mae_objframe_gauged" not in metrics


# ---------------------------------------------------------------------------
# (k) Seed plumbing -- --seed CLI flag and invocation.json effective-seed record
# ---------------------------------------------------------------------------

def test_cli_seed_defaults_to_none():
    parser = runner.build_arg_parser()
    args = parser.parse_args([
        "--arm", "gs1_frozen",
        "--train-npz", "dummy_train.npz",
        "--test-npz", "dummy_test.npz",
        "--output-root", "dummy_out",
    ])

    assert args.seed is None


def test_cli_seed_parses_int():
    parser = runner.build_arg_parser()
    args = parser.parse_args([
        "--arm", "gs1_frozen",
        "--train-npz", "dummy_train.npz",
        "--test-npz", "dummy_test.npz",
        "--output-root", "dummy_out",
        "--seed", "11",
    ])

    assert args.seed == 11


def test_write_invocation_record_no_seed_arg_records_default_42(tmp_path, monkeypatch):
    monkeypatch.delenv("PTYCHO_TORCH_SEED", raising=False)

    runner._write_invocation_record(
        tmp_path, "gs1_frozen", Path("train.npz"), Path("test.npz"), smoke=False, seed=None,
    )

    record = json.loads((tmp_path / "invocation.json").read_text())
    assert record["runs"]["gs1_frozen"]["seed"] == 42


def test_write_invocation_record_explicit_seed_overrides_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PTYCHO_TORCH_SEED", "99")

    runner._write_invocation_record(
        tmp_path, "gs1_frozen", Path("train.npz"), Path("test.npz"), smoke=False, seed=11,
    )

    record = json.loads((tmp_path / "invocation.json").read_text())
    assert record["runs"]["gs1_frozen"]["seed"] == 11


def test_write_invocation_record_env_seed_used_when_no_explicit_seed(tmp_path, monkeypatch):
    monkeypatch.setenv("PTYCHO_TORCH_SEED", "7")

    runner._write_invocation_record(
        tmp_path, "gs1_frozen", Path("train.npz"), Path("test.npz"), smoke=False, seed=None,
    )

    record = json.loads((tmp_path / "invocation.json").read_text())
    assert record["runs"]["gs1_frozen"]["seed"] == 7


# ---------------------------------------------------------------------------
# (l) Task 1 -- --parity-scale-mode/--parity-fixed-delta/--parity-init-scheme
# CLI flags, mirroring the --physics-forward-mode override pattern. Unlike
# --physics-forward-mode these do NOT touch resolve_arm_with_overrides/
# build_configs -- they are threaded straight to PtychoPINN_Lightning's
# constructor kwargs (see run_training/run_arm), so there is no arm_cfg/
# ModelConfig assertion here, only argument parsing + invocation.json record.
# ---------------------------------------------------------------------------

def test_cli_parity_scale_mode_defaults_to_off():
    parser = runner.build_arg_parser()
    args = parser.parse_args([
        "--arm", "gs1_trainable",
        "--train-npz", "dummy_train.npz",
        "--test-npz", "dummy_test.npz",
        "--output-root", "dummy_out",
    ])

    assert args.parity_scale_mode == "off"
    assert args.parity_fixed_delta == 0.0
    assert args.parity_init_scheme == "default"


@pytest.mark.parametrize("mode", ["off", "tied", "input", "output", "fixed"])
def test_cli_parity_scale_mode_accepts_valid_choices(mode):
    parser = runner.build_arg_parser()
    args = parser.parse_args([
        "--arm", "gs1_trainable",
        "--train-npz", "dummy_train.npz",
        "--test-npz", "dummy_test.npz",
        "--output-root", "dummy_out",
        "--parity-scale-mode", mode,
    ])

    assert args.parity_scale_mode == mode


def test_cli_parity_scale_mode_rejects_invalid_choice():
    parser = runner.build_arg_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([
            "--arm", "gs1_trainable",
            "--train-npz", "dummy_train.npz",
            "--test-npz", "dummy_test.npz",
            "--output-root", "dummy_out",
            "--parity-scale-mode", "garbage",
        ])


@pytest.mark.parametrize("scheme", ["default", "tf_glorot"])
def test_cli_parity_init_scheme_accepts_valid_choices(scheme):
    parser = runner.build_arg_parser()
    args = parser.parse_args([
        "--arm", "gs1_trainable",
        "--train-npz", "dummy_train.npz",
        "--test-npz", "dummy_test.npz",
        "--output-root", "dummy_out",
        "--parity-init-scheme", scheme,
    ])

    assert args.parity_init_scheme == scheme


def test_cli_parity_init_scheme_rejects_invalid_choice():
    parser = runner.build_arg_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([
            "--arm", "gs1_trainable",
            "--train-npz", "dummy_train.npz",
            "--test-npz", "dummy_test.npz",
            "--output-root", "dummy_out",
            "--parity-init-scheme", "garbage",
        ])


def test_cli_parity_fixed_delta_parses_float():
    parser = runner.build_arg_parser()
    args = parser.parse_args([
        "--arm", "gs1_trainable",
        "--train-npz", "dummy_train.npz",
        "--test-npz", "dummy_test.npz",
        "--output-root", "dummy_out",
        "--parity-fixed-delta", "0.735",
    ])

    assert args.parity_fixed_delta == pytest.approx(0.735)


def test_write_invocation_record_records_parity_defaults(tmp_path):
    runner._write_invocation_record(
        tmp_path, "gs1_frozen", Path("train.npz"), Path("test.npz"), smoke=False,
    )

    record = json.loads((tmp_path / "invocation.json").read_text())["runs"]["gs1_frozen"]
    assert record["parity_scale_mode"] == "off"
    assert record["parity_fixed_delta"] == 0.0
    assert record["parity_init_scheme"] == "default"


def test_write_invocation_record_records_explicit_parity_values(tmp_path):
    runner._write_invocation_record(
        tmp_path, "gs1_frozen", Path("train.npz"), Path("test.npz"), smoke=False,
        parity_scale_mode="tied", parity_fixed_delta=0.42, parity_init_scheme="tf_glorot",
    )

    record = json.loads((tmp_path / "invocation.json").read_text())["runs"]["gs1_frozen"]
    assert record["parity_scale_mode"] == "tied"
    assert record["parity_fixed_delta"] == 0.42
    assert record["parity_init_scheme"] == "tf_glorot"
