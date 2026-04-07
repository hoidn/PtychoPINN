#!/usr/bin/env python3
"""Generate Hybrid ResNet schematic artifacts (manifest + TikZ + DOT)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ptycho_torch.generators.schematic_manifest import build_hybrid_resnet_manifest
from ptycho_torch.generators.schematic_render import (
    render_high_level_tikz,
    render_module_flow_dot,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render schematic artifacts for ptycho_torch hybrid_resnet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".artifacts/hybrid_resnet_schematics/latest"),
        help="Output directory for manifest and schematic source files.",
    )
    parser.add_argument("--N", type=int, default=128, help="Patch size.")
    parser.add_argument("--gridsize", type=int, default=2, help="Grid size (C=gridsize^2).")
    parser.add_argument("--fno-width", type=int, default=32, help="FNO width.")
    parser.add_argument("--fno-blocks", type=int, default=4, help="FNO block count.")
    parser.add_argument("--fno-modes", type=int, default=12, help="FNO spectral modes.")
    parser.add_argument(
        "--resnet-width",
        type=int,
        default=None,
        help="Optional fixed bottleneck width (must be divisible by 4).",
    )
    parser.add_argument(
        "--max-hidden-channels",
        type=int,
        default=None,
        help="Optional cap on hidden channels during encoder growth.",
    )
    parser.add_argument(
        "--output-mode",
        choices=["real_imag", "amp_phase"],
        default="real_imag",
        help="Generator output mode.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_hybrid_resnet_manifest(
        N=int(args.N),
        gridsize=int(args.gridsize),
        fno_width=int(args.fno_width),
        fno_blocks=int(args.fno_blocks),
        fno_modes=int(args.fno_modes),
        resnet_width=args.resnet_width,
        output_mode=str(args.output_mode),
        max_hidden_channels=args.max_hidden_channels,
    )
    tikz = render_high_level_tikz(manifest)
    dot = render_module_flow_dot(manifest)

    manifest_path = args.output_dir / "hybrid_resnet_manifest.json"
    tikz_path = args.output_dir / "hybrid_resnet_high_level.tex"
    dot_path = args.output_dir / "hybrid_resnet_module_flow.dot"

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tikz_path.write_text(tikz, encoding="utf-8")
    dot_path.write_text(dot, encoding="utf-8")

    print(f"[schematics] wrote {manifest_path}")
    print(f"[schematics] wrote {tikz_path}")
    print(f"[schematics] wrote {dot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
