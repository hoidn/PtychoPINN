"""Renderers for Hybrid ResNet schematic artifacts."""

from __future__ import annotations

from typing import Any


def _fmt_shape(shape: Any) -> str:
    if isinstance(shape, list) and shape and isinstance(shape[0], int):
        return "x".join(str(v) for v in shape)
    if isinstance(shape, dict):
        return ", ".join(f"{k}:{_fmt_shape(v)}" for k, v in shape.items())
    return "n/a"


def _escape_dot(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\"", "\\\"")


def render_high_level_tikz(manifest: dict[str, Any]) -> str:
    """Render a publication-style high-level TikZ architecture diagram."""
    stage_shapes = manifest.get("stage_shapes", {})
    tensor_contract = manifest.get("tensor_contract", {})
    config = manifest.get("config", {})
    nodes = manifest.get("nodes", [])
    output_shape = tensor_contract.get("output")
    output_mode = str(config.get("output_mode", "real_imag"))
    if output_mode == "amp_phase":
        output_contract_line = "contract: amp/phase tensors"
        output_complex_line = "complex decode: $O = A\\exp(i\\phi)$"
    else:
        output_contract_line = "contract: (B,H,W,C,2)"
        output_complex_line = "complex decode: $O = \\mathrm{Re} + i\\,\\mathrm{Im}$"

    n_val = int(config.get("N", 0)) if config.get("N") is not None else 0
    n_half = n_val // 2 if n_val else "N/2"
    n_quarter = n_val // 4 if n_val else "N/4"
    fno_blocks = int(config.get("fno_blocks", 0)) if config.get("fno_blocks") is not None else 0
    fno_modes = int(config.get("fno_modes", 0)) if config.get("fno_modes") is not None else 0
    fno_width = int(config.get("fno_width", 0)) if config.get("fno_width") is not None else 0
    ptychoblock_count = sum(
        1
        for node in nodes
        if node.get("module_class") in {"PtychoBlock", "HybridResnetEncoderBlock"}
    )
    if ptychoblock_count == 0 and fno_blocks:
        ptychoblock_count = fno_blocks
    resolution_math = (
        f"$N\\rightarrow N/2\\rightarrow N/4\\;({n_val}\\rightarrow {n_half}\\rightarrow {n_quarter})$"
    )

    input_shape = _fmt_shape(tensor_contract.get("input"))
    lifter_shape = _fmt_shape(stage_shapes.get("lifter"))
    encoder_shape = _fmt_shape(stage_shapes.get("downsample_1") or stage_shapes.get("encoder_block_2"))
    resnet_shape = _fmt_shape(stage_shapes.get("resnet"))
    up_shape = _fmt_shape(stage_shapes.get("up2"))
    out_shape = _fmt_shape(output_shape)

    return f"""\\documentclass[tikz,border=12pt]{{standalone}}
\\usepackage[scaled=0.95]{{helvet}}
\\renewcommand\\familydefault{{\\sfdefault}}
\\usepackage{{tikz}}
\\usetikzlibrary{{positioning,arrows.meta,calc}}
\\begin{{document}}
\\begin{{tikzpicture}}[
  >=Latex,
  stage/.style={{draw, rounded corners=2.5pt, line width=0.9pt, minimum height=24mm, text width=52mm, align=center, inner sep=4pt}},
  ann/.style={{draw, rounded corners=2.5pt, line width=0.8pt, fill=yellow!12, text width=92mm, align=center, font=\\small, inner sep=4pt}},
  flow/.style={{-Latex, line width=1.0pt}}
]
\\node[stage, fill=blue!10, text width=34mm] (input) at (0,0) {{\\textbf{{Input}}\\\\\\small {input_shape}}};
\\node[stage, fill=cyan!12, text width=54mm, right=10mm of input] (lifter) {{\\textbf{{InputTransform + SpatialLifter}}\\\\\\small width={fno_width}\\\\\\small out={lifter_shape}}};
\\node[stage, fill=teal!14, text width=66mm, right=10mm of lifter] (encoder) {{\\textbf{{FNO Encoder}}\\\\\\small PtychoBlock x{fno_blocks}, modes={fno_modes}\\\\\\small resolution: {resolution_math}\\\\\\small out={encoder_shape}}};
\\node[stage, fill=orange!12, text width=56mm, below=28mm of encoder] (resnet) {{\\textbf{{ResNet-6 bottleneck @ N/4}}\\\\\\small out={resnet_shape}}};
\\node[stage, fill=green!14, text width=66mm, right=10mm of resnet] (up) {{\\textbf{{Upsampling Decoder x2}}\\\\\\small per stage: ConvTranspose3x3(s=2) + InstanceNorm + ReLU\\\\\\small out={up_shape}}};
\\node[stage, fill=purple!10, text width=66mm, right=10mm of up] (out) {{\\textbf{{Output (complex object patches)}}\\\\\\small {output_contract_line}\\\\\\small {output_complex_line}\\\\\\small shape={out_shape}}};
\\node[font=\\bfseries\\Large, above=12mm of encoder] (title) {{Hybrid ResNet Generator}};
\\node[
  font=\\small,
  align=center,
  text width=186mm,
  below=1mm of title
] (subtitle) {{Inverse-map generator $G:X\\rightarrow Y$ (diffraction $\\rightarrow$ real-space complex object patches), paired with differentiable forward model $F:Y\\rightarrow X$ in training ($F\\circ G$).}};

\\node[ann, text width=98mm, below=12mm of encoder] (pbnote) {{\\textbf{{PtychoBlock internals (x{ptychoblock_count})}}\\\\$y = x + \\mathrm{{GELU}}\\left(\\mathrm{{SpectralConv}}(x) + \\mathrm{{Conv3x3}}(x)\\right)$}};

\\draw[flow] (input) -- (lifter);
\\draw[flow] (lifter) -- (encoder);
\\draw[flow] (encoder) -- (resnet);
\\draw[flow] (resnet) -- (up);
\\draw[flow] (up) -- (out);
\\end{{tikzpicture}}
\\end{{document}}
"""


def render_module_flow_dot(manifest: dict[str, Any]) -> str:
    """Render Graphviz DOT for module-level execution flow."""
    lines: list[str] = [
        "digraph hybrid_resnet {",
        '  rankdir=LR;',
        '  node [shape=box, style="rounded,filled", color="#333344", fillcolor="#eeeeff", fontname="Helvetica"];',
        '  edge [color="#444455", fontname="Helvetica"];',
    ]

    for node in manifest.get("nodes", []):
        name = str(node.get("name", "unknown"))
        shape = _fmt_shape(node.get("output_shape"))
        label = _escape_dot(f"{name}\\n{shape}")
        lines.append(f'  "{_escape_dot(name)}" [label="{label}"];')

    for edge in manifest.get("edges", []):
        src = _escape_dot(str(edge.get("src", "")))
        dst = _escape_dot(str(edge.get("dst", "")))
        if src and dst:
            lines.append(f'  "{src}" -> "{dst}";')

    lines.append("}")
    lines.append("")
    return "\n".join(lines)
