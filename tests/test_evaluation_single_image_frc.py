import numpy as np
import json
import inspect
from pathlib import Path

from ptycho import params


def _make_complex_object(seed: int = 0, size: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.meshgrid(
        np.linspace(-1.0, 1.0, size, dtype=np.float32),
        np.linspace(-1.0, 1.0, size, dtype=np.float32),
        indexing="ij",
    )
    amp = 0.25 + np.exp(-3.0 * (x * x + y * y)) + 0.05 * rng.standard_normal((size, size))
    amp = np.clip(amp, 1e-4, None).astype(np.float32)
    phase = (0.8 * np.sin(6.0 * x) + 0.6 * np.cos(4.0 * y) + 0.05 * rng.standard_normal((size, size))).astype(
        np.float32
    )
    return (amp * np.exp(1j * phase)).astype(np.complex64)


def _make_low_amp_background_object(size: int = 128) -> np.ndarray:
    y, x = np.meshgrid(
        np.linspace(-1.0, 1.0, size, dtype=np.float32),
        np.linspace(-1.0, 1.0, size, dtype=np.float32),
        indexing="ij",
    )
    amp = np.full((size, size), 1e-4, dtype=np.float32)
    amp[(x * x + y * y) <= 0.3] = 1.0
    phase = (0.7 * np.sin(8.0 * x) + 0.5 * np.cos(5.0 * y)).astype(np.float32)
    return (amp * np.exp(1j * phase)).astype(np.complex64)


def _apply_phase_plane(obj: np.ndarray, ax: float, by: float) -> np.ndarray:
    h, w = obj.shape[:2]
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    ramp = np.exp(1j * (ax * xx + by * yy)).astype(np.complex64)
    return (obj * ramp).astype(np.complex64)


def _as_hw1(obj: np.ndarray) -> np.ndarray:
    return np.asarray(obj, dtype=np.complex64)[..., None]


def _write_calibration_json(tmp_path) -> str:
    path = tmp_path / "spatial_calibration.json"
    payload = {
        "profiles": {
            "test_constant": {
                "kind": "ratio_constant",
                "factor": 2.0,
            }
        }
    }
    path.write_text(json.dumps(payload))
    return str(path)


def test_single_image_frc_returns_both_threshold_pairs_for_amp_and_phase():
    from ptycho.evaluation import single_image_frc_metrics

    obj = _make_complex_object(seed=3, size=128)
    metrics = single_image_frc_metrics(_as_hw1(obj), phase_align_method="plane")
    assert "single_frc50" in metrics
    assert "single_frc1over7" in metrics
    assert len(metrics["single_frc50"]) == 2
    assert len(metrics["single_frc1over7"]) == 2


def test_single_image_frc_uses_support_weighted_phase_phasor():
    from ptycho.evaluation import single_image_frc_metrics

    obj = _make_low_amp_background_object(size=128)
    out = single_image_frc_metrics(_as_hw1(obj), support_amp_floor_ratio=0.05)
    assert np.isfinite(out["single_frc50"][1])


def test_single_image_frc_phase_is_stable_to_added_plane_ramp():
    from ptycho.evaluation import single_image_frc_metrics

    obj = _make_complex_object(seed=5, size=128)
    ramped = _apply_phase_plane(obj, ax=0.02, by=-0.03)
    a = single_image_frc_metrics(_as_hw1(obj), phase_align_method="plane")
    b = single_image_frc_metrics(_as_hw1(ramped), phase_align_method="plane")
    assert abs(a["single_frc1over7"][1] - b["single_frc1over7"][1]) <= 1


def test_single_image_frc_handles_odd_image_sizes_by_center_crop():
    from ptycho.evaluation import single_image_frc_metrics

    obj = _make_complex_object(seed=7, size=129)
    out = single_image_frc_metrics(_as_hw1(obj))
    assert "single_frc50" in out


def test_single_image_frc_supports_spatial_dual_and_legacy_modes():
    from ptycho.evaluation import single_image_frc_metrics

    obj = _make_complex_object(seed=11, size=128)
    spatial_dual = single_image_frc_metrics(_as_hw1(obj), split_mode="spatial_dual")
    spatial_alias = single_image_frc_metrics(_as_hw1(obj), split_mode="spatial")
    spatial_legacy = single_image_frc_metrics(_as_hw1(obj), split_mode="spatial_legacy")
    binom = single_image_frc_metrics(_as_hw1(obj), split_mode="binomial", rng_seed=11)
    assert "single_frc50" in spatial_dual
    assert "single_frc50" in spatial_alias
    assert "single_frc50" in spatial_legacy
    assert "single_frc50" in binom


def test_spatial_dual_split_uses_direct_strided_samples_without_averaging():
    from frc.single_image_frc import split_diagonal_strided_anti, split_diagonal_strided_main

    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    main_a, main_b = split_diagonal_strided_main(arr)
    anti_a, anti_b = split_diagonal_strided_anti(arr)

    np.testing.assert_array_equal(main_a, np.asarray([[0, 2], [8, 10]], dtype=np.float32))
    np.testing.assert_array_equal(main_b, np.asarray([[5, 7], [13, 15]], dtype=np.float32))
    np.testing.assert_array_equal(anti_a, np.asarray([[1, 3], [9, 11]], dtype=np.float32))
    np.testing.assert_array_equal(anti_b, np.asarray([[4, 6], [12, 14]], dtype=np.float32))


def test_spatial_dual_default_antialias_matches_explicit_default_sigma():
    from frc.single_image_frc import single_image_frc_curve

    img = np.random.default_rng(0).random((96, 96), dtype=np.float32)
    curve_default = single_image_frc_curve(img, split_mode="spatial_dual")
    curve_explicit = single_image_frc_curve(img, split_mode="spatial_dual", spatial_antialias_sigma=0.8)
    np.testing.assert_allclose(curve_default, curve_explicit, atol=0.0, rtol=0.0)


def test_spatial_dual_antialias_changes_curve_for_high_frequency_input():
    from frc.single_image_frc import single_image_frc_curve

    yy, xx = np.indices((96, 96))
    checker = ((xx + yy) % 2).astype(np.float32)
    curve_no_aa = single_image_frc_curve(checker, split_mode="spatial_dual", spatial_antialias_sigma=0.0)
    curve_aa = single_image_frc_curve(checker, split_mode="spatial_dual", spatial_antialias_sigma=1.0)
    assert float(np.mean(np.abs(curve_no_aa - curve_aa))) > 0.1


def test_single_image_frc_binomial_is_seed_deterministic():
    from ptycho.evaluation import single_image_frc_metrics

    obj = _make_complex_object(seed=13, size=128)
    out_a = single_image_frc_metrics(_as_hw1(obj), split_mode="binomial", rng_seed=123)
    out_b = single_image_frc_metrics(_as_hw1(obj), split_mode="binomial", rng_seed=123)
    assert out_a["single_frc50"] == out_b["single_frc50"]
    assert out_a["single_frc1over7"] == out_b["single_frc1over7"]


def test_single_image_frc_binomial_is_stable_to_global_amp_scaling_by_default():
    from ptycho.evaluation import single_image_frc_metrics

    obj = _make_complex_object(seed=15, size=128)
    scaled = (obj * np.float32(10.0)).astype(np.complex64)
    out_ref = single_image_frc_metrics(_as_hw1(obj), split_mode="binomial", rng_seed=321)
    out_scaled = single_image_frc_metrics(_as_hw1(scaled), split_mode="binomial", rng_seed=321)
    assert out_ref["single_frc50"] == out_scaled["single_frc50"]
    assert out_ref["single_frc1over7"] == out_scaled["single_frc1over7"]


def test_split_binomial_thinned_mean_lambda_controls_split_noise_strength():
    from ptycho.evaluation import single_image_frc_metrics
    obj = _make_complex_object(seed=17, size=128)
    low = single_image_frc_metrics(
        _as_hw1(obj),
        split_mode="binomial",
        rng_seed=111,
        binomial_mean_lambda=8.0,
    )
    high = single_image_frc_metrics(
        _as_hw1(obj),
        split_mode="binomial",
        rng_seed=111,
        binomial_mean_lambda=512.0,
    )
    # Higher lambda should reduce split noise and generally increase the cutoff.
    assert float(high["single_frc50"][0]) >= float(low["single_frc50"][0])


def test_split_binomial_thinned_normalize_intensity_makes_noise_scale_stable():
    from ptycho.evaluation import single_image_frc_metrics
    obj = _make_complex_object(seed=19, size=128)
    scaled = (obj * np.float32(10.0)).astype(np.complex64)

    # With normalization enabled, global amplitude rescaling should not change
    # SI-FRC cutoff at fixed mean_lambda.
    n1 = single_image_frc_metrics(
        _as_hw1(obj),
        split_mode="binomial",
        rng_seed=7,
        binomial_mean_lambda=64.0,
        binomial_normalize_intensity=True,
    )
    n2 = single_image_frc_metrics(
        _as_hw1(scaled),
        split_mode="binomial",
        rng_seed=7,
        binomial_mean_lambda=64.0,
        binomial_normalize_intensity=True,
    )
    assert n1["single_frc50"] == n2["single_frc50"]
    assert n1["single_frc1over7"] == n2["single_frc1over7"]

    # Without normalization, larger global scale increases effective lambda and
    # should alter the cutoff.
    u1 = single_image_frc_metrics(
        _as_hw1(obj),
        split_mode="binomial",
        rng_seed=7,
        binomial_mean_lambda=64.0,
        binomial_normalize_intensity=False,
    )
    u2 = single_image_frc_metrics(
        _as_hw1(scaled),
        split_mode="binomial",
        rng_seed=7,
        binomial_mean_lambda=64.0,
        binomial_normalize_intensity=False,
    )
    assert u1["single_frc50"] != u2["single_frc50"]


def test_eval_reconstruction_emits_single_image_frc_when_enabled():
    from ptycho.evaluation import eval_reconstruction

    params.set("offset", 4)
    gt = _as_hw1(_make_complex_object(seed=11, size=128))
    pred = _as_hw1(_make_complex_object(seed=12, size=128))
    out = eval_reconstruction(pred, gt, label="pinn", single_image_frc=True)
    assert "single_frc50" in out
    assert "single_frc1over7" in out


def test_external_frc_package_exposes_full_single_image_metrics_api():
    from frc.single_image_frc import single_image_frc_metrics as external_single_image_frc_metrics

    obj = _make_complex_object(seed=77, size=128)
    out = external_single_image_frc_metrics(_as_hw1(obj), offset=4, split_mode="binomial", rng_seed=11)
    assert "single_frc50" in out
    assert "single_frc1over7" in out
    assert len(out["single_frc50"]) == 2
    assert len(out["single_frc1over7"]) == 2


def test_single_image_frc_import_resolves_to_in_repo_module():
    from frc.single_image_frc import single_image_frc_metrics as external_single_image_frc_metrics

    repo_root = Path(__file__).resolve().parents[1]
    module_path = Path(inspect.getsourcefile(external_single_image_frc_metrics)).resolve()
    assert module_path == (repo_root / "frc" / "single_image_frc.py").resolve()


def test_eval_reconstruction_default_does_not_break_legacy_keys():
    from ptycho.evaluation import eval_reconstruction

    params.set("offset", 4)
    gt = _as_hw1(_make_complex_object(seed=21, size=128))
    pred = _as_hw1(_make_complex_object(seed=22, size=128))
    out = eval_reconstruction(pred, gt, label="pinn")
    for key in ("mae", "mse", "psnr", "ssim", "ms_ssim", "frc50", "frc1over7", "frc"):
        assert key in out


def test_eval_reconstruction_reports_gt_frc1over7_pair():
    from ptycho.evaluation import eval_reconstruction

    params.set("offset", 4)
    gt = _as_hw1(_make_complex_object(seed=41, size=128))
    pred = _as_hw1(_make_complex_object(seed=42, size=128))
    out = eval_reconstruction(pred, gt, label="pinn")
    assert "frc1over7" in out
    assert len(out["frc1over7"]) == 2


def test_eval_reconstruction_accepts_binomial_single_image_frc_mode():
    from ptycho.evaluation import eval_reconstruction

    params.set("offset", 4)
    gt = _as_hw1(_make_complex_object(seed=31, size=128))
    pred = _as_hw1(_make_complex_object(seed=32, size=128))
    out = eval_reconstruction(
        pred,
        gt,
        label="pinn",
        single_image_frc=True,
        single_image_frc_split_mode="binomial",
        single_image_frc_rng_seed=99,
    )
    assert "single_frc50" in out


def test_eval_reconstruction_accepts_spatial_dual_single_image_mode():
    from ptycho.evaluation import eval_reconstruction

    params.set("offset", 4)
    gt = _as_hw1(_make_complex_object(seed=201, size=128))
    pred = _as_hw1(_make_complex_object(seed=202, size=128))
    out = eval_reconstruction(
        pred,
        gt,
        label="pinn",
        single_image_frc=True,
        single_image_frc_split_mode="spatial_dual",
    )
    assert "single_frc50" in out


def test_eval_reconstruction_accepts_spatial_antialias_override():
    from ptycho.evaluation import eval_reconstruction

    params.set("offset", 4)
    gt = _as_hw1(_make_complex_object(seed=211, size=128))
    pred = _as_hw1(_make_complex_object(seed=212, size=128))
    out = eval_reconstruction(
        pred,
        gt,
        label="pinn",
        single_image_frc=True,
        single_image_frc_split_mode="spatial_dual",
        single_image_frc_spatial_antialias_sigma=0.0,
    )
    assert "single_frc50" in out


def test_spatial_dual_calibrated_mode_applies_json_coefficients(tmp_path):
    from ptycho.evaluation import single_image_frc_metrics

    obj = _make_complex_object(seed=313, size=128)
    cal_path = _write_calibration_json(tmp_path)
    raw = single_image_frc_metrics(_as_hw1(obj), split_mode="spatial_dual")
    cal = single_image_frc_metrics(
        _as_hw1(obj),
        split_mode="spatial_dual_calibrated",
        spatial_calibration_json=cal_path,
        spatial_calibration_profile="test_constant",
    )
    assert np.isclose(float(cal["single_frc50"][0]), 0.5 * float(raw["single_frc50"][0]))
    assert np.isclose(float(cal["single_frc1over7"][0]), 0.5 * float(raw["single_frc1over7"][0]))


def test_eval_reconstruction_accepts_spatial_calibration_json(tmp_path):
    from ptycho.evaluation import eval_reconstruction

    params.set("offset", 4)
    gt = _as_hw1(_make_complex_object(seed=411, size=128))
    pred = _as_hw1(_make_complex_object(seed=412, size=128))
    cal_path = _write_calibration_json(tmp_path)
    out = eval_reconstruction(
        pred,
        gt,
        label="pinn",
        single_image_frc=True,
        single_image_frc_split_mode="spatial_dual_calibrated",
        single_image_frc_spatial_calibration_json=cal_path,
        single_image_frc_spatial_calibration_profile="test_constant",
    )
    assert "single_frc50" in out


def test_frc50_uses_subbin_interpolation_for_threshold_crossing(monkeypatch):
    from ptycho import evaluation
    from ptycho.FRC import fourier_ring_corr

    curve = np.asarray([0.95, 0.72, 0.55, 0.45, 0.20], dtype=np.float64)

    def _fake_fsc(_a, _b):
        return curve.copy()

    monkeypatch.setattr(fourier_ring_corr, "FSC", _fake_fsc)
    _, cutoff = evaluation.frc50(np.ones((16, 16), dtype=np.float32), np.ones((16, 16), dtype=np.float32))
    assert np.isclose(float(cutoff), 2.5, atol=1e-9)


def test_frc50_returns_full_length_when_curve_never_crosses_threshold(monkeypatch):
    from ptycho import evaluation
    from ptycho.FRC import fourier_ring_corr

    curve = np.asarray([0.95, 0.72, 0.66, 0.51], dtype=np.float64)

    def _fake_fsc(_a, _b):
        return curve.copy()

    monkeypatch.setattr(fourier_ring_corr, "FSC", _fake_fsc)
    _, cutoff = evaluation.frc50(np.ones((16, 16), dtype=np.float32), np.ones((16, 16), dtype=np.float32))
    assert np.isclose(float(cutoff), float(len(curve)), atol=1e-9)


def test_frc_cutoffs_reports_interpolated_frc1over7(monkeypatch):
    from ptycho import evaluation
    from ptycho.FRC import fourier_ring_corr

    curve = np.asarray([0.80, 0.40, 0.20, 0.10], dtype=np.float64)

    def _fake_fsc(_a, _b):
        return curve.copy()

    monkeypatch.setattr(fourier_ring_corr, "FSC", _fake_fsc)
    _, cutoff_50, cutoff_1o7 = evaluation.frc_cutoffs(
        np.ones((16, 16), dtype=np.float32),
        np.ones((16, 16), dtype=np.float32),
    )
    assert np.isclose(float(cutoff_50), 0.75, atol=1e-9)
    assert np.isclose(float(cutoff_1o7), 2.5714285714285716, atol=1e-9)
