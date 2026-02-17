#!/usr/bin/env python3
"""Visual sanity analysis for single-image FRC alignment vs GT metrics.

Generates:
- raw_metrics.csv
- summary.json
- plots/*.png
- README.md
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

from ptycho import params
from ptycho.evaluation import eval_reconstruction


def make_ground_truth(seed: int, size: int) -> np.ndarray:
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


def degrade_object(
    obj: np.ndarray,
    *,
    blur_sigma: float,
    phase_noise_sigma: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    amp = np.abs(obj).astype(np.float32)
    phase = np.angle(obj).astype(np.float32)
    amp_blur = gaussian_filter(amp, sigma=float(blur_sigma), mode="reflect")
    phase_noisy = phase + rng.normal(0.0, float(phase_noise_sigma), size=phase.shape).astype(np.float32)
    return (amp_blur * np.exp(1j * phase_noisy)).astype(np.complex64)


def safe_spearman(x: Iterable[float], y: Iterable[float]) -> float:
    stat = spearmanr(np.asarray(list(x), dtype=float), np.asarray(list(y), dtype=float)).statistic
    if stat is None or not np.isfinite(stat):
        return float("nan")
    return float(stat)


def bootstrap_ci_mean(values: List[float], *, n_boot: int = 3000, alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "ci95_lo": float("nan"), "ci95_hi": float("nan")}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    n = arr.size
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[i] = float(np.mean(arr[idx]))
    return {
        "mean": float(np.mean(arr)),
        "ci95_lo": float(np.quantile(draws, alpha / 2.0)),
        "ci95_hi": float(np.quantile(draws, 1.0 - alpha / 2.0)),
    }


def monotonic_direction(values: Iterable[float], atol: float = 1e-8) -> str:
    arr = np.asarray(list(values), dtype=float)
    if arr.size < 2:
        return "flat"
    d = np.diff(arr)
    inc = bool(np.all(d >= -atol))
    dec = bool(np.all(d <= atol))
    if inc and dec:
        return "flat"
    if inc:
        return "increasing"
    if dec:
        return "decreasing"
    return "mixed"


def to_amp_phase_pair(values: tuple[float, float]) -> tuple[float, float]:
    return float(values[0]), float(values[1])


def run_sweep(args: argparse.Namespace) -> List[Dict[str, float]]:
    params.set("offset", int(args.offset))
    levels = np.linspace(float(args.level_min), float(args.level_max), int(args.n_levels), dtype=float)
    modes = ("spatial_dual", "spatial_legacy", "binomial")
    rows: List[Dict[str, float]] = []

    for mode in modes:
        for seed_idx in range(int(args.n_seeds)):
            gt = make_ground_truth(seed=1000 + seed_idx, size=int(args.size))
            for level_idx, blur_sigma in enumerate(levels):
                pred = degrade_object(
                    gt,
                    blur_sigma=float(blur_sigma),
                    phase_noise_sigma=float(args.phase_noise_scale) * float(blur_sigma),
                    seed=50000 + seed_idx * 100 + level_idx,
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    m = eval_reconstruction(
                        pred[..., None],
                        gt[..., None],
                        label=f"{mode}_s{seed_idx}_l{level_idx}",
                        phase_align_method="plane",
                        single_image_frc=True,
                        single_image_frc_split_mode=mode,
                        single_image_frc_rng_seed=70000 + seed_idx * 100 + level_idx,
                    )

                single50_amp, single50_phase = to_amp_phase_pair(m["single_frc50"])
                single17_amp, single17_phase = to_amp_phase_pair(m["single_frc1over7"])
                mae_amp, mae_phase = to_amp_phase_pair(m["mae"])
                mse_amp, mse_phase = to_amp_phase_pair(m["mse"])
                psnr_amp, psnr_phase = to_amp_phase_pair(m["psnr"])
                ssim_amp, ssim_phase = to_amp_phase_pair(m["ssim"])
                msssim_amp, msssim_phase = to_amp_phase_pair(m["ms_ssim"])
                frc50_amp, frc50_phase = to_amp_phase_pair(m["frc50"])
                frc17_amp, frc17_phase = to_amp_phase_pair(m["frc1over7"])
                gt_frc_len = float(len(np.asarray(m["frc"][0], dtype=np.float64)))

                rows.append(
                    {
                        "mode": mode,
                        "seed": float(seed_idx),
                        "level_idx": float(level_idx),
                        "blur_sigma": float(blur_sigma),
                        "single_frc50_amp": single50_amp,
                        "single_frc50_phase": single50_phase,
                        "single_frc1over7_amp": single17_amp,
                        "single_frc1over7_phase": single17_phase,
                        "mae_amp": mae_amp,
                        "mae_phase": mae_phase,
                        "mse_amp": mse_amp,
                        "mse_phase": mse_phase,
                        "psnr_amp": psnr_amp,
                        "psnr_phase": psnr_phase,
                        "ssim_amp": ssim_amp,
                        "ssim_phase": ssim_phase,
                        "ms_ssim_amp": msssim_amp,
                        "ms_ssim_phase": msssim_phase,
                        "frc50_amp": frc50_amp,
                        "frc50_phase": frc50_phase,
                        "frc1over7_amp": frc17_amp,
                        "frc1over7_phase": frc17_phase,
                        "frc_amp_curve_len": gt_frc_len,
                    }
                )
    return rows


def write_csv(rows: List[Dict[str, float]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rows_for(rows: List[Dict[str, float]], *, mode: str, seed: int) -> List[Dict[str, float]]:
    out = [r for r in rows if r["mode"] == mode and int(r["seed"]) == int(seed)]
    out.sort(key=lambda r: int(r["level_idx"]))
    return out


def summarize(rows: List[Dict[str, float]], *, n_seeds: int, n_levels: int) -> Dict[str, object]:
    modes = ("spatial_dual", "spatial_legacy", "binomial")
    amp_pairs = ("ssim_amp", "frc50_amp")
    phase_pairs = ("ssim_phase", "frc50_phase")

    out: Dict[str, object] = {
        "n_seeds": int(n_seeds),
        "n_levels": int(n_levels),
        "modes": {},
    }

    for mode in modes:
        mode_out: Dict[str, object] = {
            "amplitude": {},
            "phase": {},
            "monotonicity": {"single_frc50_amp": {}, "single_frc50_phase": {}},
        }
        amp_rhos: Dict[str, List[float]] = {k: [] for k in amp_pairs}
        phase_rhos: Dict[str, List[float]] = {k: [] for k in phase_pairs}
        amp_dirs: List[str] = []
        phase_dirs: List[str] = []

        for seed in range(int(n_seeds)):
            seed_rows = rows_for(rows, mode=mode, seed=seed)
            x_amp = [r["single_frc50_amp"] for r in seed_rows]
            x_phase = [r["single_frc50_phase"] for r in seed_rows]
            amp_dirs.append(monotonic_direction(x_amp))
            phase_dirs.append(monotonic_direction(x_phase))
            for metric in amp_pairs:
                y = [r[metric] for r in seed_rows]
                amp_rhos[metric].append(safe_spearman(x_amp, y))
            for metric in phase_pairs:
                y = [r[metric] for r in seed_rows]
                phase_rhos[metric].append(safe_spearman(x_phase, y))

        for metric, values in amp_rhos.items():
            stats = bootstrap_ci_mean(values, seed=42 + abs(hash((mode, metric))) % 1000)
            mode_out["amplitude"][metric] = {
                **stats,
                "per_seed_min": float(np.nanmin(values)),
                "per_seed_max": float(np.nanmax(values)),
            }

        for metric, values in phase_rhos.items():
            stats = bootstrap_ci_mean(values, seed=99 + abs(hash((mode, metric))) % 1000)
            mode_out["phase"][metric] = {
                **stats,
                "per_seed_min": float(np.nanmin(values)),
                "per_seed_max": float(np.nanmax(values)),
            }

        mode_out["monotonicity"]["single_frc50_amp"] = {
            direction: int(amp_dirs.count(direction)) for direction in sorted(set(amp_dirs))
        }
        mode_out["monotonicity"]["single_frc50_phase"] = {
            direction: int(phase_dirs.count(direction)) for direction in sorted(set(phase_dirs))
        }
        out["modes"][mode] = mode_out

    return out


def _series_by_level(rows: List[Dict[str, float]], mode: str, key: str, n_levels: int) -> tuple[np.ndarray, np.ndarray]:
    mean = np.zeros(n_levels, dtype=float)
    std = np.zeros(n_levels, dtype=float)
    for li in range(n_levels):
        vals = [r[key] for r in rows if r["mode"] == mode and int(r["level_idx"]) == li]
        mean[li] = float(np.mean(vals))
        std[li] = float(np.std(vals))
    return mean, std


def make_plots(rows: List[Dict[str, float]], summary: Dict[str, object], out_dir: Path, levels: np.ndarray) -> None:
    _ = summary
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    mode = "binomial"
    color = "#d62728"

    # 1) trend_single_frc50_amp_vs_blur.png
    plt.figure(figsize=(8, 5))
    mean, std = _series_by_level(rows, mode, "single_frc50_amp", len(levels))
    plt.plot(levels, mean, label=f"{mode} single_frc50_amp", color=color)
    plt.fill_between(levels, mean - std, mean + std, color=color, alpha=0.2)
    plt.xlabel("Blur Sigma")
    plt.ylabel("single_frc50_amp")
    plt.title("Single-Image FRC50 (Amplitude) vs Blur")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "trend_single_frc50_amp_vs_blur.png", dpi=160)
    plt.close()

    # 3) scatter_single_frc50_amp_vs_ssim_amp.png
    plt.figure(figsize=(8, 5))
    vals = [r for r in rows if r["mode"] == mode]
    x = np.asarray([r["single_frc50_amp"] for r in vals], dtype=float)
    y = np.asarray([r["ssim_amp"] for r in vals], dtype=float)
    rho = safe_spearman(x, y)
    plt.scatter(x, y, s=16, alpha=0.55, label=f"rho={rho:+.3f}", color=color)
    if np.unique(x).size >= 2:
        coef = np.polyfit(x, y, deg=1)
        xx = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        yy = coef[0] * xx + coef[1]
        plt.plot(xx, yy, color=color, linewidth=2)
    plt.xlabel("single_frc50_amp")
    plt.ylabel("ssim_amp")
    plt.title("Scatter: single_frc50_amp vs ssim_amp")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "scatter_single_frc50_amp_vs_ssim_amp.png", dpi=160)
    plt.close()

    # 3b) scatter by degradation level (color)
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.3))
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=float(np.min(levels)), vmax=float(np.max(levels)))
    vals = [r for r in rows if r["mode"] == mode]
    x = np.asarray([r["single_frc50_amp"] for r in vals], dtype=float)
    y = np.asarray([r["ssim_amp"] for r in vals], dtype=float)
    z = np.asarray([r["blur_sigma"] for r in vals], dtype=float)
    sc = ax.scatter(x, y, c=z, cmap=cmap, norm=norm, s=22, alpha=0.7)
    rho = safe_spearman(x, y)
    ax.text(
        0.03,
        0.93,
        f"Spearman rho={rho:+.3f}",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    ax.set_title("single_frc50_amp vs ssim_amp by degradation level")
    ax.set_xlabel("single_frc50_amp")
    ax.set_ylabel("ssim_amp")
    ax.grid(alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.92)
    cbar.set_label("Degradation level (blur sigma)")
    fig.tight_layout()
    fig.savefig(plot_dir / "scatter_single_frc50_amp_vs_ssim_amp_by_level.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 4) single_frc1over7_amp vs GT frc1over7_amp by degradation level
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.3))
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=float(np.min(levels)), vmax=float(np.max(levels)))
    vals = [r for r in rows if r["mode"] == mode]
    x_all = np.asarray([r["single_frc1over7_amp"] for r in vals], dtype=float)
    y_all = np.asarray([r["frc1over7_amp"] for r in vals], dtype=float)
    z_all = np.asarray([r["blur_sigma"] for r in vals], dtype=float)
    gt_len = np.asarray([r["frc_amp_curve_len"] for r in vals], dtype=float)
    # Exclude only GT-censored points for this plot.
    keep = y_all < (gt_len - 1e-9)
    if int(np.count_nonzero(keep)) >= 2:
        x = x_all[keep]
        y = y_all[keep]
        z = z_all[keep]
    else:
        x = x_all
        y = y_all
        z = z_all
    sc = ax.scatter(x, y, c=z, cmap=cmap, norm=norm, s=20, alpha=0.7)
    rho = safe_spearman(x, y)
    ax.text(
        0.03,
        0.93,
        f"Spearman rho={rho:+.3f}",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    ax.set_title("single_frc1over7_amp vs GT frc1over7_amp by degradation level")
    ax.set_xlabel("single_frc1over7_amp")
    ax.set_ylabel("GT frc1over7_amp")
    ax.grid(alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.92)
    cbar.set_label("Degradation level (blur sigma)")
    fig.tight_layout()
    fig.savefig(plot_dir / "scatter_single_frc1over7_amp_vs_gt_frc1over7_amp_by_level_gt_nosat.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Remove stale files from previous report variants.
    stale = [
        "trend_ssim_amp_vs_blur.png",
        "rho_ci_bar_amp.png",
        "phase_stability_overview.png",
        "scatter_single_frc50_amp_vs_gt_frc50_amp_by_level.png",
        "scatter_single_frc1over7_amp_vs_gt_frc1over7_amp_by_level.png",
    ]
    for name in stale:
        p = plot_dir / name
        if p.exists():
            p.unlink()


def write_readme(args: argparse.Namespace, out_dir: Path, summary: Dict[str, object]) -> None:
    _ = args, out_dir, summary

    readme = out_dir / "README.md"

    lines = [
        "# frc",
        "",
        "Utilities for single-image Fourier Ring Correlation (FRC) split construction.",
        "",
        "## Installation",
        "```bash",
        "pip install -e .",
        "```",
        "",
        "## API",
        "- `center_crop_even_square(arr)`: center-crop to an even square canvas.",
        "- `split_diagonal_interleaved(arr)`: diagonal interleaved split into two half-images.",
        "- `split_diagonal_strided_main(arr)`: strided (00 vs 11) spatial split.",
        "- `split_diagonal_strided_anti(arr)`: strided (01 vs 10) spatial split.",
        "- `split_binomial_thinned(arr, rng_seed=None, count_scale=4096.0)`: independent Poisson half-split.",
        "- `single_image_frc_curve(image_2d, split_mode=..., spatial_antialias_sigma=...)`: supports `binomial`, `spatial_dual`, `spatial_legacy` plus calibrated aliases.",
        "- `single_image_frc_metrics(..., spatial_calibration_json=..., spatial_calibration_profile=...)`: applies JSON coefficient calibration for spatial calibrated modes.",
        "- `first_below_threshold(curve, threshold)`: first index where a curve falls below threshold.",
        "",
        "Spatial modes apply anti-alias prefiltering by default (`spatial_antialias_sigma=0.8`).",
        "",
        "## Example",
        "```python",
        "import numpy as np",
        "",
        "from frc.single_image_frc import (",
        "    center_crop_even_square,",
        "    split_binomial_thinned,",
        ")",
        "",
        "img = np.random.rand(96, 96).astype(np.float32)",
        "img = center_crop_even_square(img)",
        "",
        "half_a, half_b = split_binomial_thinned(img, rng_seed=123, count_scale=4096.0)",
        "```",
        "",
        "## Method (Math)",
        "Let the complex object be",
        "",
        "`O(x) = A(x) * exp(i * phi(x))`.",
        "",
        "For single-image FRC on amplitude, define an intensity proxy",
        "",
        "`lambda(x) = count_scale * |O(x)|^2`.",
        "",
        "Then form two statistically independent half-images using Poisson splitting:",
        "",
        "`n1(x) ~ Poisson(lambda(x)/2),  n2(x) ~ Poisson(lambda(x)/2)`.",
        "",
        "`h1(x) = sqrt(n1(x)/count_scale) * exp(i * angle(O(x)))`,",
        "`h2(x) = sqrt(n2(x)/count_scale) * exp(i * angle(O(x)))`.",
        "",
        "Compute FFTs `H1(k), H2(k)` and ring-average by radius `r`. The single-image FRC curve is the signed shell correlation",
        "",
        "`FRC(r) = Re(<H1(k) * conj(H2(k))>_r) / sqrt(<|H1(k)|^2>_r * <|H2(k)|^2>_r)`.",
        "",
        "Cutoff metrics are first-below-threshold crossings:",
        "",
        "`r_t = min { r : FRC(r) < t }`, with `t in {0.5, 1/7}`.",
        "",
        "If no crossing occurs, `r_t` is set to the curve length (right-censored at Nyquist).",
        "",
        "GT-based FRC uses the same cutoff rule but correlates prediction vs ground-truth images.",
        "",
        "## Plot Artifacts",
        "The following plots are generated by the external alignment sweep and saved under `plots/`.",
        "",
        "### single_frc50_amp trend",
        "![single_frc50_amp trend](./plots/trend_single_frc50_amp_vs_blur.png)",
        "",
        "### Scatter (single_frc50_amp vs ssim_amp)",
        "![scatter](./plots/scatter_single_frc50_amp_vs_ssim_amp.png)",
        "",
        "### Scatter by degradation level",
        "![scatter by level](./plots/scatter_single_frc50_amp_vs_ssim_amp_by_level.png)",
        "",
        "### single_frc1over7_amp vs GT frc1over7_amp (by degradation level)",
        "This compares no-GT single-image FRC@1/7 (x-axis) against GT-based FRC@1/7 (y-axis).",
        "GT-censored points (`GT FRC@1/7 == curve length`) are excluded in this plot only.",
        "![scatter frc1over7](./plots/scatter_single_frc1over7_amp_vs_gt_frc1over7_amp_by_level_gt_nosat.png)",
        "",
        "## Data Artifacts",
        "- `raw_metrics.csv`",
        "- `summary.json`",
    ]
    readme.write_text("\n".join(lines) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze single-image FRC alignment and generate visual report.")
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--n-levels", type=int, default=13)
    parser.add_argument("--level-min", type=float, default=0.0)
    parser.add_argument("--level-max", type=float, default=2.0)
    parser.add_argument("--phase-noise-scale", type=float, default=0.03)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--offset", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../frc"),
    )
    parser.add_argument(
        "--write-readme",
        action="store_true",
        help="If set, overwrite output-dir README.md with generated report content.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = run_sweep(args)
    write_csv(rows, out_dir / "raw_metrics.csv")
    summary = summarize(rows, n_seeds=int(args.n_seeds), n_levels=int(args.n_levels))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    levels = np.linspace(float(args.level_min), float(args.level_max), int(args.n_levels), dtype=float)
    make_plots(rows, summary, out_dir, levels)
    if args.write_readme:
        write_readme(args, out_dir, summary)

    print(json.dumps({"output_dir": str(out_dir), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
