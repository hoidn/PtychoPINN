#!/usr/bin/env python3
"""Visual sanity analysis for single-image FRC alignment vs GT metrics.

Generates:
- raw_metrics.csv
- summary.json
- plots/*.png
- FRC_README.md
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
    levels = np.linspace(0.0, float(args.level_max), int(args.n_levels), dtype=float)
    modes = ("spatial", "binomial")
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
    modes = ("spatial", "binomial")
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
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    modes = ("spatial", "binomial")
    colors = {"spatial": "#1f77b4", "binomial": "#d62728"}

    # 1) trend_single_frc50_amp_vs_blur.png
    plt.figure(figsize=(8, 5))
    for mode in modes:
        mean, std = _series_by_level(rows, mode, "single_frc50_amp", len(levels))
        plt.plot(levels, mean, label=f"{mode} single_frc50_amp", color=colors[mode])
        plt.fill_between(levels, mean - std, mean + std, color=colors[mode], alpha=0.2)
    plt.xlabel("Blur Sigma")
    plt.ylabel("single_frc50_amp")
    plt.title("Single-Image FRC50 (Amplitude) vs Blur")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "trend_single_frc50_amp_vs_blur.png", dpi=160)
    plt.close()

    # 2) trend_ssim_amp_vs_blur.png
    plt.figure(figsize=(8, 5))
    for mode in modes:
        mean, std = _series_by_level(rows, mode, "ssim_amp", len(levels))
        plt.plot(levels, mean, label=f"{mode} ssim_amp", color=colors[mode])
        plt.fill_between(levels, mean - std, mean + std, color=colors[mode], alpha=0.2)
    plt.xlabel("Blur Sigma")
    plt.ylabel("ssim_amp")
    plt.title("SSIM (Amplitude) vs Blur")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "trend_ssim_amp_vs_blur.png", dpi=160)
    plt.close()

    # 3) scatter_single_frc50_amp_vs_ssim_amp.png
    plt.figure(figsize=(8, 5))
    for mode in modes:
        vals = [r for r in rows if r["mode"] == mode]
        x = np.asarray([r["single_frc50_amp"] for r in vals], dtype=float)
        y = np.asarray([r["ssim_amp"] for r in vals], dtype=float)
        rho = safe_spearman(x, y)
        plt.scatter(x, y, s=14, alpha=0.45, label=f"{mode} (rho={rho:+.3f})", color=colors[mode])
        if np.unique(x).size >= 2:
            coef = np.polyfit(x, y, deg=1)
            xx = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            yy = coef[0] * xx + coef[1]
            plt.plot(xx, yy, color=colors[mode], linewidth=2)
    plt.xlabel("single_frc50_amp")
    plt.ylabel("ssim_amp")
    plt.title("Scatter: single_frc50_amp vs ssim_amp")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "scatter_single_frc50_amp_vs_ssim_amp.png", dpi=160)
    plt.close()

    # 3b) scatter by degradation level (color) per mode
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=float(np.min(levels)), vmax=float(np.max(levels)))
    for i, mode in enumerate(modes):
        vals = [r for r in rows if r["mode"] == mode]
        x = np.asarray([r["single_frc50_amp"] for r in vals], dtype=float)
        y = np.asarray([r["ssim_amp"] for r in vals], dtype=float)
        z = np.asarray([r["blur_sigma"] for r in vals], dtype=float)
        sc = axs[i].scatter(x, y, c=z, cmap=cmap, norm=norm, s=18, alpha=0.7)
        axs[i].set_title(f"{mode}: single_frc50_amp vs ssim_amp")
        axs[i].set_xlabel("single_frc50_amp")
        axs[i].grid(alpha=0.3)
        rho = safe_spearman(x, y)
        axs[i].text(
            0.03,
            0.05,
            f"Spearman rho={rho:+.3f}",
            transform=axs[i].transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    axs[0].set_ylabel("ssim_amp")
    cbar = fig.colorbar(sc, ax=axs, shrink=0.92)
    cbar.set_label("Degradation level (blur sigma)")
    fig.suptitle("Scatter by degradation level", y=1.02)
    fig.tight_layout()
    fig.savefig(plot_dir / "scatter_single_frc50_amp_vs_ssim_amp_by_level.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 4) rho_ci_bar_amp.png
    labels = ["ssim_amp", "frc50_amp"]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(8, 5))
    for i, mode in enumerate(modes):
        means = []
        yerr_low = []
        yerr_high = []
        for label in labels:
            stats = summary["modes"][mode]["amplitude"][label]
            means.append(float(stats["mean"]))
            yerr_low.append(float(stats["mean"]) - float(stats["ci95_lo"]))
            yerr_high.append(float(stats["ci95_hi"]) - float(stats["mean"]))
        offs = (i - 0.5) * width
        plt.bar(x + offs, means, width=width, label=mode, color=colors[mode], alpha=0.85)
        plt.errorbar(x + offs, means, yerr=[yerr_low, yerr_high], fmt="none", ecolor="black", capsize=4)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xticks(x, labels)
    plt.ylabel("Spearman rho (mean over seeds)")
    plt.title("Amplitude Correlation: single_frc50 vs SSIM / GT FRC50 (95% CI)")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(plot_dir / "rho_ci_bar_amp.png", dpi=160)
    plt.close()

    # 5) phase_stability_overview.png
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for mode in modes:
        mean, std = _series_by_level(rows, mode, "single_frc50_phase", len(levels))
        axs[0].plot(levels, mean, label=f"{mode}", color=colors[mode])
        axs[0].fill_between(levels, mean - std, mean + std, color=colors[mode], alpha=0.2)
    axs[0].set_title("single_frc50_phase vs Blur")
    axs[0].set_xlabel("Blur Sigma")
    axs[0].set_ylabel("single_frc50_phase")
    axs[0].grid(alpha=0.3)
    axs[0].legend()

    labels_phase = ["ssim_phase", "frc50_phase"]
    xp = np.arange(len(labels_phase))
    for i, mode in enumerate(modes):
        means = []
        err_lo = []
        err_hi = []
        for lab in labels_phase:
            stats = summary["modes"][mode]["phase"][lab]
            means.append(float(stats["mean"]))
            err_lo.append(float(stats["mean"]) - float(stats["ci95_lo"]))
            err_hi.append(float(stats["ci95_hi"]) - float(stats["mean"]))
        offs = (i - 0.5) * width
        axs[1].bar(xp + offs, means, width=width, color=colors[mode], alpha=0.85, label=mode)
        axs[1].errorbar(xp + offs, means, yerr=[err_lo, err_hi], fmt="none", ecolor="black", capsize=4)
    axs[1].axhline(0.0, color="black", linewidth=1)
    axs[1].set_xticks(xp)
    axs[1].set_xticklabels(labels_phase)
    axs[1].set_title("Phase Correlation (95% CI)")
    axs[1].set_ylabel("Spearman rho")
    axs[1].grid(alpha=0.3, axis="y")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(plot_dir / "phase_stability_overview.png", dpi=160)
    plt.close(fig)


def _fmt_ci(stats: Dict[str, float]) -> str:
    return f"{stats['mean']:+.3f} [{stats['ci95_lo']:+.3f}, {stats['ci95_hi']:+.3f}]"


def write_readme(args: argparse.Namespace, out_dir: Path, summary: Dict[str, object]) -> None:
    plot_dir = out_dir / "plots"
    spatial_amp_ssim = summary["modes"]["spatial"]["amplitude"]["ssim_amp"]
    spatial_amp_frc = summary["modes"]["spatial"]["amplitude"]["frc50_amp"]
    binom_amp_ssim = summary["modes"]["binomial"]["amplitude"]["ssim_amp"]
    binom_amp_frc = summary["modes"]["binomial"]["amplitude"]["frc50_amp"]

    binom_ssim_lower = float(binom_amp_ssim["ci95_lo"])
    acceptance_binomial = "PASS" if binom_ssim_lower > 0 else "FAIL"

    readme = out_dir / "FRC_README.md"
    cmd = (
        "python scripts/studies/analyze_single_image_frc_alignment.py "
        f"--n-seeds {args.n_seeds} --n-levels {args.n_levels} --level-max {args.level_max} "
        f"--output-dir {args.output_dir}"
    )

    lines = [
        "# Single-Image FRC Visual Validation",
        "",
        "## Command",
        "```bash",
        cmd,
        "```",
        "",
        "## Sweep Settings",
        f"- `n_seeds`: {int(args.n_seeds)}",
        f"- `n_levels`: {int(args.n_levels)}",
        f"- `level_max`: {float(args.level_max)}",
        f"- `phase_noise_scale`: {float(args.phase_noise_scale)}",
        f"- `size`: {int(args.size)}",
        f"- `offset`: {int(args.offset)}",
        "",
        "## Key Correlation Summary (Amplitude)",
        "",
        "| Mode | rho(single_frc50_amp, ssim_amp) | rho(single_frc50_amp, frc50_amp) |",
        "|---|---:|---:|",
        f"| spatial | {_fmt_ci(spatial_amp_ssim)} | {_fmt_ci(spatial_amp_frc)} |",
        f"| binomial | {_fmt_ci(binom_amp_ssim)} | {_fmt_ci(binom_amp_frc)} |",
        "",
        "## Acceptance Checks",
        f"- Binomial vs SSIM CI lower bound > 0: **{acceptance_binomial}**",
        "- Spatial mode behavior on blur sweeps is documented; anti-alignment is expected for checkerboard split under pure blur perturbation.",
        "- Interpretation is relative-trend only; no absolute physical resolution claim is made.",
        "",
        "## Plots",
        "",
        "### single_frc50_amp trend",
        "![single_frc50_amp trend](plots/trend_single_frc50_amp_vs_blur.png)",
        "",
        "### ssim_amp trend",
        "![ssim_amp trend](plots/trend_ssim_amp_vs_blur.png)",
        "",
        "### Scatter (single_frc50_amp vs ssim_amp)",
        "![scatter](plots/scatter_single_frc50_amp_vs_ssim_amp.png)",
        "",
        "### Scatter by degradation level",
        "![scatter by level](plots/scatter_single_frc50_amp_vs_ssim_amp_by_level.png)",
        "",
        "### rho + 95% CI bars",
        "![rho ci amp](plots/rho_ci_bar_amp.png)",
        "",
        "### Phase stability overview",
        "![phase stability](plots/phase_stability_overview.png)",
        "",
        "## Artifacts",
        f"- Raw metrics: `{(out_dir / 'raw_metrics.csv').as_posix()}`",
        f"- Summary JSON: `{(out_dir / 'summary.json').as_posix()}`",
    ]
    readme.write_text("\n".join(lines) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze single-image FRC alignment and generate visual report.")
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--n-levels", type=int, default=13)
    parser.add_argument("--level-max", type=float, default=2.0)
    parser.add_argument("--phase-noise-scale", type=float, default=0.03)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--offset", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("frc"),
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
    levels = np.linspace(0.0, float(args.level_max), int(args.n_levels), dtype=float)
    make_plots(rows, summary, out_dir, levels)
    write_readme(args, out_dir, summary)

    print(json.dumps({"output_dir": str(out_dir), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
