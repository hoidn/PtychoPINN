import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

from ptycho import params
from ptycho.evaluation import eval_reconstruction


def _make_ground_truth(seed: int = 0, size: int = 96) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.meshgrid(
        np.linspace(-1.0, 1.0, size, dtype=np.float32),
        np.linspace(-1.0, 1.0, size, dtype=np.float32),
        indexing="ij",
    )
    amp = 0.2 + np.exp(-3.0 * (x * x + y * y)) + 0.03 * rng.standard_normal((size, size))
    amp = np.clip(amp, 1e-3, None).astype(np.float32)
    phase = (0.9 * np.sin(5.0 * x) + 0.7 * np.cos(4.0 * y)).astype(np.float32)
    return (amp * np.exp(1j * phase)).astype(np.complex64)


def _degrade_object(
    obj: np.ndarray,
    *,
    blur_sigma: float,
    phase_noise_sigma: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    amp = np.abs(obj).astype(np.float32)
    phi = np.angle(obj).astype(np.float32)
    amp_blur = gaussian_filter(amp, sigma=blur_sigma, mode="reflect")
    phi_noisy = phi + rng.normal(0.0, phase_noise_sigma, size=phi.shape).astype(np.float32)
    return (amp_blur * np.exp(1j * phi_noisy)).astype(np.complex64)


def _is_non_increasing(values: np.ndarray, tol: float = 0.0) -> bool:
    arr = np.asarray(values, dtype=np.float64)
    return bool(np.all(arr[1:] <= arr[:-1] + float(tol)))


def _collect_frc_series(split_mode: str) -> tuple[np.ndarray, np.ndarray]:
    params.set("offset", 4)
    gt = _make_ground_truth(seed=23, size=96)
    levels = [0.0, 0.5, 1.0, 1.5, 2.0]
    gt_frc50_amp = []
    single_frc50_amp = []

    for idx, level in enumerate(levels):
        pred = _degrade_object(
            gt,
            blur_sigma=level,
            phase_noise_sigma=0.03 * level,
            seed=1000 + idx,
        )
        eval_kwargs = {}
        if split_mode == "binomial":
            # Keep sanity checks in a stable-SNR binomial regime independent of
            # runtime defaults.
            eval_kwargs["single_image_frc_binomial_mean_lambda"] = 100.0

        metrics = eval_reconstruction(
            pred[..., None],
            gt[..., None],
            label=f"{split_mode}_{idx}",
            phase_align_method="plane",
            single_image_frc=True,
            single_image_frc_split_mode=split_mode,
            single_image_frc_rng_seed=123 + idx,
            **eval_kwargs,
        )
        gt_frc50_amp.append(float(metrics["frc50"][0]))
        single_frc50_amp.append(float(metrics["single_frc50"][0]))

    return np.asarray(gt_frc50_amp), np.asarray(single_frc50_amp)


def test_single_image_frc_amp_tracks_distinct_monotonic_trends_by_split_mode():
    gt_spatial_dual, single_spatial_dual = _collect_frc_series("spatial_dual")
    gt_spatial_legacy, single_spatial_legacy = _collect_frc_series("spatial_legacy")
    gt_binom, single_binom = _collect_frc_series("binomial")

    assert _is_non_increasing(gt_spatial_dual, tol=1.0)
    assert _is_non_increasing(gt_spatial_legacy, tol=1.0)
    assert _is_non_increasing(gt_binom, tol=1.0)
    # Spatial subsampling modes increase with blur on this sweep.
    assert _is_non_increasing(-single_spatial_dual, tol=2.0)
    # Legacy interpolated split is retained for reproducibility only.
    assert _is_non_increasing(-single_spatial_legacy, tol=2.0)
    # Binomial split tracks GT direction under the same perturbation.
    assert _is_non_increasing(single_binom, tol=3.0)


def test_single_image_frc_amp_rank_correlation_shows_mode_behavior_vs_gt():
    gt_spatial_dual, single_spatial_dual = _collect_frc_series("spatial_dual")
    gt_spatial_legacy, single_spatial_legacy = _collect_frc_series("spatial_legacy")
    gt_binom, single_binom = _collect_frc_series("binomial")

    rho_spatial_dual, _ = spearmanr(gt_spatial_dual, single_spatial_dual)
    rho_spatial_legacy, _ = spearmanr(gt_spatial_legacy, single_spatial_legacy)
    rho_binom, _ = spearmanr(gt_binom, single_binom)

    assert np.isfinite(rho_spatial_dual)
    assert np.isfinite(rho_spatial_legacy)
    assert np.isfinite(rho_binom)
    assert float(rho_spatial_dual) < -0.5
    assert float(rho_spatial_legacy) < -0.5
    assert float(rho_binom) > 0.5


def test_binomial_and_spatial_dual_single_image_frc_have_consistent_ordering():
    _, single_spatial = _collect_frc_series("spatial_dual")
    _, single_binom = _collect_frc_series("binomial")
    rho_modes, _ = spearmanr(single_spatial, single_binom)
    assert np.isfinite(rho_modes)
    assert float(rho_modes) < -0.5


def test_binomial_single_image_frc1over7_high_blur_not_ceiling_saturated():
    params.set("offset", 4)
    seed_idx = 0
    level_idx = 12
    blur_sigma = 2.0
    gt = _make_ground_truth(seed=1000 + seed_idx, size=96)
    pred = _degrade_object(
        gt,
        blur_sigma=blur_sigma,
        phase_noise_sigma=0.03 * blur_sigma,
        seed=50000 + seed_idx * 100 + level_idx,
    )
    metrics = eval_reconstruction(
        pred[..., None],
        gt[..., None],
        label="binomial_high_blur",
        phase_align_method="plane",
        single_image_frc=True,
        single_image_frc_split_mode="binomial",
        single_image_frc_rng_seed=70000 + seed_idx * 100 + level_idx,
    )
    curve_len = float(len(np.asarray(metrics["frc"][0], dtype=np.float64)))
    assert float(metrics["single_frc1over7"][0]) < curve_len
