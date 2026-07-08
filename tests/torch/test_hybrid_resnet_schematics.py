"""Tests for hybrid_resnet schematic manifest/render/CLI generation."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ptycho_torch.generators.schematic_manifest import build_hybrid_resnet_manifest
from ptycho_torch.generators.schematic_render import (
    render_high_level_tikz,
    render_module_flow_dot,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_manifest_contains_expected_hybrid_resnet_stages():
    """Manifest should include core Hybrid ResNet stages and output contract."""
    manifest = build_hybrid_resnet_manifest(
        N=64,
        gridsize=2,
        fno_width=32,
        fno_blocks=4,
        fno_modes=12,
        resnet_width=None,
        output_mode="real_imag",
    )
    names = [node["name"] for node in manifest["nodes"]]
    assert "lifter" in names
    assert "encoder_block_0" in names
    assert "downsample_0" in names
    assert "resnet" in names
    assert "up1" in names
    assert "up2" in names
    assert manifest["tensor_contract"]["output"] == [1, 64, 64, 4, 2]


def test_tikz_contains_key_resnet_style_blocks():
    """TikZ renderer should include high-level stage labels and output contract."""
    manifest = build_hybrid_resnet_manifest(
        N=64,
        gridsize=2,
        fno_width=32,
        fno_blocks=4,
        fno_modes=12,
        resnet_width=None,
        output_mode="real_imag",
    )
    tikz = render_high_level_tikz(manifest)
    assert "SpatialLifter" in tikz
    assert "FNO Encoder" in tikz
    assert "N\\rightarrow N/2\\rightarrow N/4" in tikz
    assert "ResNet-6" in tikz
    assert "Upsampling Decoder x2" in tikz
    assert "ConvTranspose3x3(s=2) + InstanceNorm + ReLU" in tikz
    assert "B,H,W,C,2" in tikz
    assert "Output (complex object patches)" in tikz
    assert "O = \\mathrm{Re} + i\\,\\mathrm{Im}" in tikz
    assert "PtychoBlock internals" in tikz
    assert "Inverse-map generator $G:X\\rightarrow Y$" in tikz
    assert "forward model $F:Y\\rightarrow X$" in tikz
    assert "F\\circ G" in tikz
    assert "CycleGAN" not in tikz
    assert "DO NOT CIRCULATE" not in tikz
    assert "No U-Net skip path" not in tikz
    assert "spectral features" not in tikz


def test_dot_contains_module_edges():
    """DOT renderer should include directed graph structure and key nodes."""
    manifest = build_hybrid_resnet_manifest(
        N=64,
        gridsize=2,
        fno_width=32,
        fno_blocks=4,
        fno_modes=12,
        resnet_width=None,
        output_mode="real_imag",
    )
    dot = render_module_flow_dot(manifest)
    assert "digraph" in dot
    assert "lifter" in dot
    assert "resnet" in dot
    assert "up1" in dot
    assert "up2" in dot


def test_cli_writes_manifest_and_non_mermaid_files(tmp_path: Path):
    """CLI should emit manifest plus .tex and .dot schematic sources."""
    out_dir = tmp_path / "schematics"
    cmd = [
        "python",
        str(REPO_ROOT / "scripts/studies/render_hybrid_resnet_schematics.py"),
        "--output-dir",
        str(out_dir),
        "--N",
        "64",
        "--gridsize",
        "2",
        "--fno-width",
        "32",
        "--fno-blocks",
        "4",
        "--fno-modes",
        "12",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stderr
    assert (out_dir / "hybrid_resnet_manifest.json").exists()
    assert (out_dir / "hybrid_resnet_high_level.tex").exists()
    assert (out_dir / "hybrid_resnet_module_flow.dot").exists()


def test_docs_reference_hybrid_resnet_schematic_command():
    """Architecture docs should reference the schematic generation command."""
    arch_doc = Path("docs/architecture_torch.md").read_text(encoding="utf-8")
    assert "render_hybrid_resnet_schematics.py" in arch_doc
