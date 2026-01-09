#!/usr/bin/env python3
"""
scaling reproducer for CDI/Ptycho panels with consistent color mapping.

This script:
  1) Loads four PNGs (idealized/hybrid x CDI/Ptycho)
  2) Inverts Matplotlib's 'jet' colormap (256 levels) to approximate original scalar fields
  3) Computes a single global vmin (low percentile across all panels)
  4) Sets hybrid panels' vmax to the maximum value across the two hybrids
  5) Adjusts idealized panels' vmax so their fraction of "red" pixels matches hybrid_ptycho
     at a chosen red threshold (default t_red = 0.98 in normalized space)
  6) Renders scaled panels using the shared vmin and panel-specific vmax, generates small
     versions, a 2x2 mosaic, a manifest JSON, and (optionally) a universal LaTeX figure.
     
Assumptions:
  - Input PNGs were originally rendered with Matplotlib's 'jet' colormap at 256 steps.
  - Inversion uses nearest palette entry; this recovers discrete scalar indices in [0, 1].
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm


def load_rgb_uint8(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def invert_jet_unique(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Invert 256-level 'jet' by mapping each unique RGB to the nearest entry in the LUT.
    Returns a float32 array in [0,1].
    """
    lut = (cm.get_cmap("jet", 256)(np.linspace(0, 1, 256))[:, :3] * 255.0).astype(np.uint8)
    h, w, _ = rgb_uint8.shape
    flat = rgb_uint8.reshape(-1, 3)
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)  # uniq: (K,3)
    diffs = uniq[:, None, :].astype(np.int16) - lut[None, :, :].astype(np.int16)  # (K,256,3)
    dist2 = np.sum(diffs.astype(np.int32) ** 2, axis=2)  # (K,256)
    best_idx = np.argmin(dist2, axis=1).astype(np.uint8)  # (K,)
    scal_idx = best_idx[inv].reshape(h, w)
    return scal_idx.astype(np.float32) / 255.0


def render_with_jet(scalar: np.ndarray, vmin: float, vmax: float) -> Image.Image:
    normed = np.clip((scalar - vmin) / (vmax - vmin + 1e-12), 0, 1)
    rgba = cm.get_cmap("jet")(normed)  # (H,W,4)
    rgb_uint8 = (rgba[:, :, :3] * 255.0).astype(np.uint8)
    return Image.fromarray(rgb_uint8, mode="RGB")


def vmax_for_fraction_match(arr: np.ndarray, vmin: float, t_red: float, f_target: float) -> float:
    """
    Choose vmax so that P( (X - vmin)/(vmax - vmin) >= t_red ) equals f_target.
    Solve via quantile at 1 - f_target: Q = vmin + t_red * (vmax - vmin) => vmax = vmin + (Q - vmin)/t_red.
    """
    if f_target <= 0.0 + 1e-9:
        return float(max(np.max(arr), vmin + 1e-4))
    Q = float(np.quantile(arr, 1.0 - f_target))
    vmax = vmin + (Q - vmin) / max(t_red, 1e-6)
    vmax = max(vmax, vmin + 1e-4)
    return float(vmax)


def red_fraction(arr: np.ndarray, vmin: float, vmax: float, t_red: float) -> float:
    normed = np.clip((arr - vmin) / (vmax - vmin + 1e-12), 0, 1)
    return float(np.mean(normed >= t_red))


def save_resized(img: Image.Image, out_png: Path, out_jpg: Path, max_w: int) -> None:
    w, h = img.size
    if w > max_w:
        new_h = int(h * (max_w / w))
        img = img.resize((max_w, new_h), Image.Resampling.LANCZOS)
    img.save(out_png, "PNG", optimize=True)
    img.save(out_jpg, "JPEG", quality=85, optimize=True, progressive=True)


def write_universal_tex(outdir: Path, files_small: dict) -> Path:
    """
    Write a self-contained 2x2 LaTeX snippet that needs only graphicx.
    Files are expected to be the *_small.png variants.
    """
    tex = f"""
% Requires only: \usepackage{{graphicx}}
\begin{{figure}}[t]
  \centering
  % Row 1: Idealized
  \begin{{minipage}}[t]{{0.48\linewidth}}
    \centering
    \includegraphics[width=\linewidth]{{{files_small['idealized_cdi']}}}\\
    \vspace{{0.25em}}\scriptsize (a) Idealized — CDI
  \end{{minipage}}\hfill
  \begin{{minipage}}[t]{{0.48\linewidth}}
    \centering
    \includegraphics[width=\linewidth]{{{files_small['idealized_ptycho']}}}\\
    \vspace{{0.25em}}\scriptsize (b) Idealized — Ptycho
  \end{{minipage}}

  \vspace{{0.8em}}

  % Row 2: Hybrid
  \begin{{minipage}}[t]{{0.48\linewidth}}
    \centering
    \includegraphics[width=\linewidth]{{{files_small['hybrid_cdi']}}}\\
    \vspace{{0.25em}}\scriptsize (c) Hybrid — CDI
  \end{{minipage}}\hfill
  \begin{{minipage}}[t]{{0.48\linewidth}}
    \centering
    \includegraphics[width=\linewidth]{{{files_small['hybrid_ptycho']}}}\\
    \vspace{{0.25em}}\scriptsize (d) Hybrid — Ptycho
  \end{{minipage}}

  \caption{{Reconstruction comparison with shared vmin and matched red-fraction scaling. Rows: Idealized vs Hybrid. Columns: CDI vs Ptycho.}}
  \label{{fig:recon_2x2}}
\end{{figure}}
"""
    tex_path = outdir / "figure_2x2_universal.tex"
    with open(tex_path, "w") as f:
        f.write(tex)
    return tex_path


def main():
    p = argparse.ArgumentParser(description="Reproduce v5 scaling: match idealized red-fraction to hybrid_ptycho.")
    p.add_argument("--idealized-cdi", type=Path, required=True)
    p.add_argument("--idealized-ptycho", type=Path, required=True)
    p.add_argument("--hybrid-cdi", type=Path, required=True)
    p.add_argument("--hybrid-ptycho", type=Path, required=True)
    p.add_argument("--outdir", type=Path, default=Path("out_scaling_v5"))
    p.add_argument("--low-percentile", type=float, default=1.0, help="Global vmin percentile across all panels (default: 1.0)")
    p.add_argument("--t-red", type=float, default=0.98, help="Red threshold in normalized space (default: 0.98)")
    p.add_argument("--small-width", type=int, default=800, help="Max width for small PNG/JPG (default: 800)")
    p.add_argument("--write-tex", action="store_true", help="Also emit a universal LaTeX figure snippet")
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Load and invert
    panels = {
        "idealized_cdi": args.idealized_cdi,
        "idealized_ptycho": args.idealized_ptycho,
        "hybrid_cdi": args.hybrid_cdi,
        "hybrid_ptycho": args.hybrid_ptycho,
    }
    scalars = {}
    for k, path in panels.items():
        rgb = load_rgb_uint8(path)
        s = invert_jet_unique(rgb)
        scalars[k] = s
        np.save(args.outdir / f"{k}_scalar.npy", s)

    # Global vmin
    all_vals = np.concatenate([v.ravel() for v in scalars.values()])
    global_vmin = float(np.percentile(all_vals, args.low_percentile))

    # Fix hybrids: vmax = max across both hybrids
    hybrid_max = max(float(np.max(scalars["hybrid_cdi"])), float(np.max(scalars["hybrid_ptycho"])))
    vmax_map = {
        "hybrid_cdi": hybrid_max,
        "hybrid_ptycho": hybrid_max,
    }

    # Target red fraction from hybrid_ptycho
    f_target = red_fraction(scalars["hybrid_ptycho"], global_vmin, hybrid_max, args.t_red)

    # Idealized vmax to match f_target
    vmax_map["idealized_cdi"] = vmax_for_fraction_match(scalars["idealized_cdi"], global_vmin, args.t_red, f_target)
    vmax_map["idealized_ptycho"] = vmax_for_fraction_match(scalars["idealized_ptycho"], global_vmin, args.t_red, f_target)

    # Render scaled panels
    out_paths = {}
    for k in panels.keys():
        img = render_with_jet(scalars[k], global_vmin, vmax_map[k])
        png = args.outdir / f"{k}_scaled.png"
        img.save(png, "PNG", optimize=True)
        out_paths[k] = png

    # Small variants + JPGs
    small_paths = {}
    for k, png in out_paths.items():
        img = Image.open(png).convert("RGB")
        out_png = args.outdir / f"{k}_scaled_small.png"
        out_jpg = args.outdir / f"{k}_scaled_small.jpg"
        save_resized(img, out_png, out_jpg, args.small_width)
        small_paths[k] = out_png.name  # basename for LaTeX

    # 2x2 mosaic (rows: idealized/hybrid; cols: cdi/ptycho)
    def load_u8(png_path: Path) -> np.ndarray:
        return np.asarray(Image.open(png_path).convert("RGB"), dtype=np.uint8)
    A = load_u8(args.outdir / f"idealized_cdi_scaled_small.png")
    B = load_u8(args.outdir / f"idealized_ptycho_scaled_small.png")
    C = load_u8(args.outdir / f"hybrid_cdi_scaled_small.png")
    D = load_u8(args.outdir / f"hybrid_ptycho_scaled_small.png")
    h = min(A.shape[0], B.shape[0], C.shape[0], D.shape[0])
    w = min(A.shape[1], B.shape[1], C.shape[1], D.shape[1])
    A, B, C, D = A[:h, :w], B[:h, :w], C[:h, :w], D[:h, :w]
    top = np.concatenate([A, B], axis=1)
    bot = np.concatenate([C, D], axis=1)
    mosaic = np.concatenate([top, bot], axis=0)

    plt.figure(figsize=(8, 8))
    plt.imshow(mosaic)
    plt.axis("off")
    plt.tight_layout()
    mosaic_path = args.outdir / "recon_mosaic_small.png"
    plt.savefig(mosaic_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()

    # Manifest
    fractions = {k: red_fraction(scalars[k], global_vmin, vmax_map[k], args.t_red) for k in panels.keys()}
    manifest = {
        "low_percentile": args.low_percentile,
        "t_red": args.t_red,
        "global_vmin": global_vmin,
        "hybrid_max": hybrid_max,
        "panel_vmax": vmax_map,
        "achieved_red_fractions": fractions,
        "files": {k: str(out_paths[k]) for k in panels.keys()},
        "mosaic": str(mosaic_path),
    }
    with open(args.outdir / "scaling_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Optional: LaTeX snippet
    if args.write_tex:
        files_small = {k: small_paths[k] for k in panels.keys()}
        tex_path = write_universal_tex(args.outdir, files_small)
        print(f"Wrote LaTeX snippet to: {tex_path}")

    # Console summary
    print("=== V5 Scaling Reproduction ===")
    print(f"Output dir: {args.outdir}")
    print(f"Global vmin (p{args.low_percentile}): {global_vmin:.6f}")
    print(f"Hybrid vmax (max hybrid): {hybrid_max:.6f}")
    print(f"Target red fraction (hybrid_ptycho @ t_red={args.t_red}): {fractions['hybrid_ptycho']:.6f}")
    print("Per-panel vmax:")
    for k in ["idealized_cdi","idealized_ptycho","hybrid_cdi","hybrid_ptycho"]:
        print(f"  {k}: {vmax_map[k]:.6f}")
    print("Achieved red fractions:")
    for k in ["idealized_cdi","idealized_ptycho","hybrid_cdi","hybrid_ptycho"]:
        print(f"  {k}: {fractions[k]:.6f}")
    print("Rendered files:")
    for k in ["idealized_cdi","idealized_ptycho","hybrid_cdi","hybrid_ptycho"]:
        print(f"  {k}: {args.outdir / (k + '_scaled.png')}")
    print(f"Mosaic: {mosaic_path}")
    print(f"Manifest: {args.outdir / 'scaling_manifest.json'}")


if __name__ == "__main__":
    main()

#python repro_scaling_v5.py \
#  --idealized-cdi /path/to/idealized_cdi.png \
#  --idealized-ptycho /path/to/idealized_ptycho.png \
#  --hybrid-cdi /path/to/hybrid_cdi.png \
#  --hybrid-ptycho /path/to/hybrid_ptycho.png \
#  --outdir out_scaling_v5 \
#  --low-percentile 1.0 \
#  --t-red 0.98 \
#  --small-width 800 \
#  --write-tex
