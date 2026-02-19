"""Single-image FRC utilities and metrics."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter as gf


DEFAULT_SPATIAL_ANTIALIAS_SIGMA = 0.8
DEFAULT_BINOMIAL_MEAN_LAMBDA = 10.0
DEFAULT_BINOMIAL_NORMALIZE_INTENSITY = True


def center_crop_even_square(arr: np.ndarray) -> np.ndarray:
    """Center-crop to an even square canvas for split FRC."""
    img = np.asarray(arr)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {img.shape}")
    h, w = img.shape
    side = min(h, w)
    if side % 2 == 1:
        side -= 1
    if side < 2:
        raise ValueError(f"Image too small for single-image FRC: {img.shape}")
    h0 = (h - side) // 2
    w0 = (w - side) // 2
    return img[h0:h0 + side, w0:w0 + side]


def split_diagonal_interleaved(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split one image into two diagonal interleaved half-images."""
    img = np.asarray(arr)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {img.shape}")
    if img.shape[0] != img.shape[1]:
        raise ValueError(f"Expected square image, got shape {img.shape}")
    if img.shape[0] % 2 != 0:
        raise ValueError(f"Expected even side length, got shape {img.shape}")
    half_a = 0.5 * (img[0::2, 0::2] + img[1::2, 1::2])
    half_b = 0.5 * (img[0::2, 1::2] + img[1::2, 0::2])
    return half_a, half_b


def split_diagonal_strided_main(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Main-diagonal strided split: (0,0) vs (1,1) samples."""
    img = np.asarray(arr)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {img.shape}")
    if img.shape[0] != img.shape[1]:
        raise ValueError(f"Expected square image, got shape {img.shape}")
    if img.shape[0] % 2 != 0:
        raise ValueError(f"Expected even side length, got shape {img.shape}")
    half_a = img[0::2, 0::2]
    half_b = img[1::2, 1::2]
    return half_a, half_b


def split_diagonal_strided_anti(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Anti-diagonal strided split: (0,1) vs (1,0) samples."""
    img = np.asarray(arr)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {img.shape}")
    if img.shape[0] != img.shape[1]:
        raise ValueError(f"Expected square image, got shape {img.shape}")
    if img.shape[0] % 2 != 0:
        raise ValueError(f"Expected even side length, got shape {img.shape}")
    half_a = img[0::2, 1::2]
    half_b = img[1::2, 0::2]
    return half_a, half_b


def split_binomial_thinned(
    arr: np.ndarray,
    *,
    rng_seed: int | None = None,
    mean_lambda: float = DEFAULT_BINOMIAL_MEAN_LAMBDA,
    normalize_intensity: bool = DEFAULT_BINOMIAL_NORMALIZE_INTENSITY,
    count_scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split an image into two statistically independent Poisson half-images."""
    img = np.asarray(arr)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {img.shape}")
    if img.shape[0] != img.shape[1]:
        raise ValueError(f"Expected square image, got shape {img.shape}")

    rng = np.random.default_rng(0 if rng_seed is None else int(rng_seed))
    mag = np.abs(np.asarray(img, dtype=np.complex128))
    intensity = np.nan_to_num(mag * mag, nan=0.0, posinf=0.0, neginf=0.0)
    eps = np.finfo(np.float64).eps
    if count_scale is not None:
        # Legacy override path preserved for explicit reproducibility.
        legacy_scale = float(count_scale)
        if legacy_scale <= 0:
            raise ValueError(f"count_scale must be > 0, got {count_scale}")
        lam = np.clip(intensity * legacy_scale, 0.0, None)
        intensity_gain = 1.0
        recon_scale = legacy_scale
    else:
        lam_mean = float(mean_lambda)
        if lam_mean <= 0:
            raise ValueError(f"mean_lambda must be > 0, got {mean_lambda}")
        if normalize_intensity:
            mean_intensity = float(np.mean(intensity))
            if np.isfinite(mean_intensity) and mean_intensity > eps:
                intensity_ref = intensity / mean_intensity
                intensity_gain = float(np.sqrt(mean_intensity))
            else:
                intensity_ref = np.zeros_like(intensity, dtype=np.float64)
                intensity_gain = 1.0
        else:
            intensity_ref = intensity
            intensity_gain = 1.0
        lam = np.clip(intensity_ref * lam_mean, 0.0, None)
        recon_scale = lam_mean

    half_counts_a = rng.poisson(0.5 * lam)
    half_counts_b = rng.poisson(0.5 * lam)
    mag_a = np.sqrt(half_counts_a / recon_scale) * intensity_gain
    mag_b = np.sqrt(half_counts_b / recon_scale) * intensity_gain

    if np.iscomplexobj(img):
        phase = np.angle(img)
        half_a = (mag_a * np.exp(1j * phase)).astype(np.complex64)
        half_b = (mag_b * np.exp(1j * phase)).astype(np.complex64)
    else:
        sign = np.sign(np.asarray(img, dtype=np.float64))
        sign[sign == 0] = 1.0
        half_a = (mag_a * sign).astype(np.float32)
        half_b = (mag_b * sign).astype(np.float32)
    return half_a, half_b


def first_below_threshold(curve: np.ndarray, threshold: float) -> float:
    """Return first index where curve falls below threshold."""
    vals = np.asarray(curve, dtype=np.float64)
    if vals.size == 0:
        return np.nan
    finite = np.isfinite(vals)
    if not finite.any():
        return np.nan
    finite_vals = vals[finite]
    idx = np.where(finite_vals < threshold)[0]
    if len(idx) == 0:
        return float(len(finite_vals))
    return float(idx[0])


def _normalize_split_mode(split_mode: str) -> tuple[str, bool]:
    """Resolve split mode aliases and calibrated variants."""
    if split_mode == "spatial":
        return "spatial_dual", False
    if split_mode == "spatial_calibrated":
        return "spatial_dual", True
    if split_mode in ("spatial_dual", "spatial_legacy", "binomial"):
        return split_mode, False
    if split_mode in ("spatial_dual_calibrated", "spatial_legacy_calibrated"):
        return split_mode.replace("_calibrated", ""), True
    raise ValueError(
        f"Unknown split_mode={split_mode!r}; expected one of "
        f"('spatial', 'spatial_calibrated', 'spatial_dual', 'spatial_dual_calibrated', "
        f"'spatial_legacy', 'spatial_legacy_calibrated', 'binomial')."
    )


@lru_cache(maxsize=16)
def _load_spatial_calibration_json(calibration_json: str) -> dict:
    """Load and cache spatial calibration payload from JSON."""
    path = Path(calibration_json).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid calibration JSON in {path}: expected top-level object")
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict) or len(profiles) == 0:
        raise ValueError(f"Invalid calibration JSON in {path}: expected non-empty 'profiles' object")
    return payload


def _resolve_spatial_calibration_profile(
    *,
    calibration_json: str,
    calibration_profile: str | None,
) -> dict:
    payload = _load_spatial_calibration_json(calibration_json)
    profiles = payload["profiles"]
    if calibration_profile is None:
        if "default" in profiles:
            profile_name = "default"
        elif len(profiles) == 1:
            profile_name = next(iter(profiles.keys()))
        else:
            raise ValueError(
                "calibration_profile is required when calibration JSON defines multiple profiles "
                "and no 'default' profile is present"
            )
    else:
        profile_name = str(calibration_profile)

    if profile_name not in profiles:
        raise ValueError(f"Unknown calibration_profile={profile_name!r} in {calibration_json}")
    profile = profiles[profile_name]
    if not isinstance(profile, dict):
        raise ValueError(f"Invalid profile {profile_name!r} in {calibration_json}: expected object")
    return profile


def _apply_spatial_calibration_to_cutoff(cutoff: float, profile: dict) -> float:
    """Apply scalar calibration transform to a spatial single-image FRC cutoff."""
    val = float(cutoff)
    if not np.isfinite(val):
        return val
    if val <= 0:
        return val

    kind = str(profile.get("kind", "ratio_log"))
    eps = float(profile.get("eps", 1e-12))

    if kind == "ratio_constant":
        factor = float(profile["factor"])
        if factor <= eps:
            raise ValueError(f"Invalid ratio_constant calibration factor={factor}")
        return val / factor

    if kind == "ratio_log":
        a = float(profile["a"])
        b = float(profile["b"])
        c = float(profile["c"])
        corr = a + b * np.log(max(c * val, eps))
        if corr <= eps:
            return np.nan
        return val / corr

    raise ValueError(f"Unsupported calibration kind={kind!r}; expected 'ratio_constant' or 'ratio_log'")


def fit_and_remove_plane(phase_img: np.ndarray, reference_phase: np.ndarray | None = None) -> np.ndarray:
    """Fit and remove a plane from a 2D phase image."""
    h, w = phase_img.shape
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    phase_flat = phase_img.flatten()
    a = np.column_stack([x_flat, y_flat, np.ones(len(x_flat))])
    coeffs, _, _, _ = np.linalg.lstsq(a, phase_flat, rcond=None)
    fitted_plane = coeffs[0] * x_coords + coeffs[1] * y_coords + coeffs[2]
    phase_aligned = phase_img - fitted_plane
    if reference_phase is not None:
        ref_coeffs, _, _, _ = np.linalg.lstsq(a, reference_phase.flatten(), rcond=None)
        ref_plane = ref_coeffs[0] * x_coords + ref_coeffs[1] * y_coords + ref_coeffs[2]
        phase_aligned = phase_img - fitted_plane + ref_plane
    return phase_aligned


def trim_image(arr2d: np.ndarray, offset: int) -> np.ndarray:
    """Trim an image by offset/2 on each border."""
    arr = np.asarray(arr2d)
    off = int(offset)
    if off < 0 or (off % 2):
        raise ValueError(f"offset must be a non-negative even integer, got {offset}")
    if off == 0:
        return arr
    half = off // 2
    if arr.shape[0] <= off or arr.shape[1] <= off:
        raise ValueError(f"offset={off} too large for shape {arr.shape}")
    return arr[half:-half, half:-half]


def _phase_align(phase: np.ndarray, phase_align_method: str) -> np.ndarray:
    phase_unwrapped = np.unwrap(np.unwrap(np.asarray(phase, dtype=np.float32), axis=0), axis=1)
    if phase_align_method == "plane":
        return fit_and_remove_plane(phase_unwrapped)
    if phase_align_method == "mean":
        return phase_unwrapped - np.mean(phase_unwrapped)
    raise ValueError(f"Unknown phase_align_method: {phase_align_method}. Use 'plane' or 'mean'.")


def _support_weighted_phase_phasor(
    phase_aligned: np.ndarray,
    amp: np.ndarray,
    support_amp_floor_ratio: float,
) -> np.ndarray:
    amp_np = np.asarray(amp, dtype=np.float32)
    phase_np = np.asarray(phase_aligned, dtype=np.float32)
    if amp_np.shape != phase_np.shape:
        raise ValueError(f"Amplitude/phase shape mismatch: {amp_np.shape} vs {phase_np.shape}")
    max_amp = float(np.max(amp_np)) if amp_np.size else 0.0
    if max_amp <= 0:
        return np.zeros_like(amp_np, dtype=np.complex64)
    floor = max_amp * float(support_amp_floor_ratio)
    support = amp_np >= floor
    phasor = np.zeros_like(amp_np, dtype=np.complex64)
    if np.any(support):
        phasor[support] = np.exp(1j * phase_np[support]).astype(np.complex64)
    return phasor


def _spin_average_2d(x: np.ndarray) -> np.ndarray:
    """Radial ring average mirroring legacy spin_average behavior."""
    nr, nc = np.shape(x)
    nrdc = np.floor(nr / 2) + 1
    ncdc = np.floor(nc / 2) + 1
    r = np.arange(nr) - nrdc + 1
    c = np.arange(nc) - ncdc + 1
    rr, cc = np.meshgrid(r, c)
    index = np.round(np.sqrt(rr**2 + cc**2)) + 1
    maxindex = int(np.max(index))
    output = np.zeros(maxindex, dtype=np.complex128)
    for i in range(maxindex):
        idx = np.where(index == (i + 1))
        if len(idx[0]) == 0:
            output[i] = 0.0
        else:
            output[i] = np.sum(x[idx]) / len(idx[0])
    return output


def _spin_sum_and_count_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ring-wise sums and sample counts using the legacy radial indexing."""
    nr, nc = np.shape(x)
    nrdc = np.floor(nr / 2) + 1
    ncdc = np.floor(nc / 2) + 1
    r = np.arange(nr) - nrdc + 1
    c = np.arange(nc) - ncdc + 1
    rr, cc = np.meshgrid(r, c, indexing="ij")
    ring = (np.round(np.sqrt(rr**2 + cc**2)) + 1).astype(np.int64)

    labels = ring.ravel()
    flat = np.asarray(x, dtype=np.complex128).ravel()
    max_label = int(np.max(labels))

    sums_real = np.bincount(labels, weights=np.real(flat), minlength=max_label + 1)
    sums_imag = np.bincount(labels, weights=np.imag(flat), minlength=max_label + 1)
    counts = np.bincount(labels, minlength=max_label + 1).astype(np.float64)

    sums = (sums_real + 1j * sums_imag)[1 : max_label + 1]
    counts = counts[1 : max_label + 1]
    return sums, counts


def _curve_from_half_images(half_a: np.ndarray, half_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute signed FRC curve and ring weights from two half-images."""
    i1 = np.fft.fftshift(np.fft.fft2(half_a))
    i2 = np.fft.fftshift(np.fft.fft2(half_b))

    c_sum, c_count = _spin_sum_and_count_2d(np.multiply(i1, np.conj(i2)))
    c1_sum, c1_count = _spin_sum_and_count_2d(np.multiply(i1, np.conj(i1)))
    c2_sum, c2_count = _spin_sum_and_count_2d(np.multiply(i2, np.conj(i2)))

    eps = np.finfo(np.float64).eps
    c = c_sum / np.maximum(c_count, 1.0)
    c1 = c1_sum / np.maximum(c1_count, 1.0)
    c2 = c2_sum / np.maximum(c2_count, 1.0)
    denom = np.sqrt(np.abs(np.multiply(c1, c2)))
    curve = np.asarray(np.real(c) / np.maximum(denom, eps), dtype=np.float64)
    curve = np.clip(curve, -1.0, 1.0)
    return curve, c_count


def _single_image_frc_curve_spatial_dual(canvas: np.ndarray, *, frc_sigma: float = 0.0) -> np.ndarray:
    """Compute dual-diagonal spatial FRC curve with ring-count weighting."""
    main_a, main_b = split_diagonal_strided_main(canvas)
    anti_a, anti_b = split_diagonal_strided_anti(canvas)
    curve_main, w_main = _curve_from_half_images(main_a, main_b)
    curve_anti, w_anti = _curve_from_half_images(anti_a, anti_b)

    n = max(len(curve_main), len(curve_anti))
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        num = 0.0
        den = 0.0
        if i < len(curve_main) and np.isfinite(curve_main[i]) and i < len(w_main) and w_main[i] > 0:
            num += float(w_main[i]) * float(curve_main[i])
            den += float(w_main[i])
        if i < len(curve_anti) and np.isfinite(curve_anti[i]) and i < len(w_anti) and w_anti[i] > 0:
            num += float(w_anti[i]) * float(curve_anti[i])
            den += float(w_anti[i])
        if den > 0:
            out[i] = num / den

    out = np.clip(out, -1.0, 1.0)
    if frc_sigma > 0:
        out = np.asarray(gf(out, frc_sigma), dtype=np.float64)
    return out


def _apply_spatial_antialias(
    canvas: np.ndarray,
    *,
    spatial_antialias_sigma: float | None,
) -> np.ndarray:
    """Prefilter spatial split input to reduce decimation aliasing."""
    if spatial_antialias_sigma is None:
        sigma = float(DEFAULT_SPATIAL_ANTIALIAS_SIGMA)
    else:
        sigma = float(spatial_antialias_sigma)
    if sigma < 0:
        raise ValueError(f"spatial_antialias_sigma must be >= 0, got {sigma}")
    if sigma == 0:
        return canvas
    if np.iscomplexobj(canvas):
        real_f = gf(np.real(canvas).astype(np.float64), sigma=sigma, mode="reflect")
        imag_f = gf(np.imag(canvas).astype(np.float64), sigma=sigma, mode="reflect")
        return np.asarray(real_f + 1j * imag_f, dtype=np.complex128)
    return np.asarray(gf(np.asarray(canvas, dtype=np.float64), sigma=sigma, mode="reflect"), dtype=np.float64)


def single_image_frc_curve(
    image_2d: np.ndarray,
    *,
    frc_sigma: float = 0.0,
    split_mode: str = "spatial",
    rng_seed: int | None = None,
    spatial_antialias_sigma: float | None = None,
    binomial_mean_lambda: float = DEFAULT_BINOMIAL_MEAN_LAMBDA,
    binomial_normalize_intensity: bool = DEFAULT_BINOMIAL_NORMALIZE_INTENSITY,
    binomial_count_scale: float | None = None,
) -> np.ndarray:
    """Compute single-image FRC curve for one real/complex 2D image."""
    canvas = center_crop_even_square(np.asarray(image_2d))
    normalized_mode, _ = _normalize_split_mode(split_mode)
    if normalized_mode == "spatial_dual":
        canvas = _apply_spatial_antialias(canvas, spatial_antialias_sigma=spatial_antialias_sigma)
        return _single_image_frc_curve_spatial_dual(canvas, frc_sigma=frc_sigma)
    if normalized_mode == "spatial_legacy":
        canvas = _apply_spatial_antialias(canvas, spatial_antialias_sigma=spatial_antialias_sigma)
        half_a, half_b = split_diagonal_interleaved(canvas)
    elif normalized_mode == "binomial":
        half_a, half_b = split_binomial_thinned(
            canvas,
            rng_seed=rng_seed,
            mean_lambda=binomial_mean_lambda,
            normalize_intensity=binomial_normalize_intensity,
            count_scale=binomial_count_scale,
        )
    else:
        raise ValueError(f"Unhandled normalized split mode: {normalized_mode!r}")

    curve, _ = _curve_from_half_images(half_a, half_b)
    if frc_sigma > 0:
        curve = np.asarray(gf(curve, frc_sigma), dtype=np.float64)
    return curve


def _extract_prediction_hw(stitched_obj: np.ndarray) -> np.ndarray:
    pred = np.asarray(stitched_obj)
    if pred.ndim == 4:
        pred = pred[0]
    pred = np.squeeze(pred)
    if pred.ndim != 2:
        raise ValueError(f"Expected a 2D prediction after squeeze, got {pred.shape}")
    return pred


def single_image_frc_metrics(
    stitched_obj: np.ndarray,
    *,
    offset: int = 0,
    split_mode: str = "spatial",
    rng_seed: int | None = None,
    phase_align_method: str = "plane",
    support_amp_floor_ratio: float = 0.05,
    frc_sigma: float = 0.0,
    spatial_antialias_sigma: float | None = None,
    spatial_calibration_json: str | None = None,
    spatial_calibration_profile: str | None = None,
    binomial_mean_lambda: float = DEFAULT_BINOMIAL_MEAN_LAMBDA,
    binomial_normalize_intensity: bool = DEFAULT_BINOMIAL_NORMALIZE_INTENSITY,
    binomial_count_scale: float | None = None,
) -> dict[str, tuple[float, float]]:
    """Compute no-GT single-image FRC cutoff metrics for amplitude and phase."""
    pred_hw = _extract_prediction_hw(stitched_obj)
    amp = trim_image(np.abs(pred_hw), offset)
    phi = trim_image(np.angle(pred_hw), offset)
    phi_aligned = _phase_align(phi, phase_align_method)

    amp_seed = 0 if rng_seed is None else int(rng_seed)
    phase_seed = amp_seed + 1
    normalized_mode, calibrated_mode = _normalize_split_mode(split_mode)

    amp_curve = single_image_frc_curve(
        amp,
        frc_sigma=frc_sigma,
        split_mode=normalized_mode,
        rng_seed=amp_seed,
        spatial_antialias_sigma=spatial_antialias_sigma,
        binomial_mean_lambda=binomial_mean_lambda,
        binomial_normalize_intensity=binomial_normalize_intensity,
        binomial_count_scale=binomial_count_scale,
    )
    amp_cut_50 = first_below_threshold(amp_curve, 0.5)
    amp_cut_1o7 = first_below_threshold(amp_curve, 1.0 / 7.0)

    phase_phasor = _support_weighted_phase_phasor(
        phi_aligned,
        amp,
        support_amp_floor_ratio=support_amp_floor_ratio,
    )
    if np.count_nonzero(np.abs(phase_phasor) > 0) < 4:
        phi_cut_50 = np.nan
        phi_cut_1o7 = np.nan
    else:
        phi_curve = single_image_frc_curve(
            phase_phasor,
            frc_sigma=frc_sigma,
            split_mode=normalized_mode,
            rng_seed=phase_seed,
            spatial_antialias_sigma=spatial_antialias_sigma,
            binomial_mean_lambda=binomial_mean_lambda,
            binomial_normalize_intensity=binomial_normalize_intensity,
            binomial_count_scale=binomial_count_scale,
        )
        phi_cut_50 = first_below_threshold(phi_curve, 0.5)
        phi_cut_1o7 = first_below_threshold(phi_curve, 1.0 / 7.0)

    if calibrated_mode:
        if spatial_calibration_json is None:
            raise ValueError(
                f"split_mode={split_mode!r} requires spatial_calibration_json to apply calibration"
            )
        profile = _resolve_spatial_calibration_profile(
            calibration_json=str(spatial_calibration_json),
            calibration_profile=spatial_calibration_profile,
        )
        amp_cut_50 = _apply_spatial_calibration_to_cutoff(amp_cut_50, profile)
        amp_cut_1o7 = _apply_spatial_calibration_to_cutoff(amp_cut_1o7, profile)
        phi_cut_50 = _apply_spatial_calibration_to_cutoff(phi_cut_50, profile)
        phi_cut_1o7 = _apply_spatial_calibration_to_cutoff(phi_cut_1o7, profile)

    return {
        "single_frc50": (amp_cut_50, phi_cut_50),
        "single_frc1over7": (amp_cut_1o7, phi_cut_1o7),
    }
