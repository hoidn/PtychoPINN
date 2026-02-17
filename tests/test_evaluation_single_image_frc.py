import numpy as np

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


def test_single_image_frc_supports_spatial_and_binomial_modes():
    from ptycho.evaluation import single_image_frc_metrics

    obj = _make_complex_object(seed=11, size=128)
    spatial = single_image_frc_metrics(_as_hw1(obj), split_mode="spatial")
    binom = single_image_frc_metrics(_as_hw1(obj), split_mode="binomial", rng_seed=11)
    assert "single_frc50" in spatial
    assert "single_frc50" in binom


def test_single_image_frc_binomial_is_seed_deterministic():
    from ptycho.evaluation import single_image_frc_metrics

    obj = _make_complex_object(seed=13, size=128)
    out_a = single_image_frc_metrics(_as_hw1(obj), split_mode="binomial", rng_seed=123)
    out_b = single_image_frc_metrics(_as_hw1(obj), split_mode="binomial", rng_seed=123)
    assert out_a["single_frc50"] == out_b["single_frc50"]
    assert out_a["single_frc1over7"] == out_b["single_frc1over7"]


def test_eval_reconstruction_emits_single_image_frc_when_enabled():
    from ptycho.evaluation import eval_reconstruction

    params.set("offset", 4)
    gt = _as_hw1(_make_complex_object(seed=11, size=128))
    pred = _as_hw1(_make_complex_object(seed=12, size=128))
    out = eval_reconstruction(pred, gt, label="pinn", single_image_frc=True)
    assert "single_frc50" in out
    assert "single_frc1over7" in out


def test_eval_reconstruction_default_does_not_break_legacy_keys():
    from ptycho.evaluation import eval_reconstruction

    params.set("offset", 4)
    gt = _as_hw1(_make_complex_object(seed=21, size=128))
    pred = _as_hw1(_make_complex_object(seed=22, size=128))
    out = eval_reconstruction(pred, gt, label="pinn")
    for key in ("mae", "mse", "psnr", "ssim", "ms_ssim", "frc50", "frc"):
        assert key in out


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
