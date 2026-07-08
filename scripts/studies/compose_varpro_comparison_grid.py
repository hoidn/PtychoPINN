"""Composite comparison figures + merged metrics table (Task E4).

Consumes the per-arm harness outputs produced by Task E3 (the representation
matrix at `.artifacts/varpro_ablation/ext_matrix/`, the Phase-1 CNN dyads at
`.artifacts/varpro_ablation/matrix_lines/`, and the flux-sweep eval logs at
`.artifacts/varpro_ablation/ext_fluxsweep/`) and renders three axis-comparison
figures plus one merged metrics table:

  - Axis A (representation): amp/phase decoder head `repr_ampphase` vs
    `repr_realimag`, `probe_varpro` variant, plus a real-imag scatter column.
  - Axis B (dynamic scaling across flux): the `cnn_ri` flux-sweep SCALE
    table parsed from its eval log.
  - Axis C (gs1 neither vs both): the Phase-1 CNN `gs1_frozen` /
    `gs1_trainable` dyads.

All amplitude panels are gauge-quotiented via `diagnose_placement.gauge`
before display/NCC: on this pipeline varpro-ON output is in count-amplitude
units, not the manuscript's `~1` normalized convention (`amendment #14`,
`docs/plans/2026-07-01-varpro-ablation-phase1-findings.md`), so a raw
amplitude comparison against truth is not meaningful without first dividing
out the global complex gauge `c_A = |s|`. See `.superpowers/sdd/ext/
task-E4-brief.md` for the full spec and `.superpowers/sdd/ext/
task-E3-report.md` for the source numbers.

Every compose_* function takes explicit input paths and an explicit output
path (no implicit `.artifacts/` lookups), so tests can drive them against
synthetic tmp dirs.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_placement import gauge  # noqa: E402
from varpro_probe_ablation_runner import _overlap_crop  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_variant(variant_dir: Path) -> Tuple[np.ndarray, Dict]:
    """Load `(canvas, metrics)` from one harness variant directory
    (`<root>/<arm>/<variant>/{canvas.npz,metrics.json}`)."""
    canvas = np.load(variant_dir / "canvas.npz")["canvas"]
    metrics = json.loads((variant_dir / "metrics.json").read_text())
    return canvas, metrics


def load_truth(truth_path: Path) -> np.ndarray:
    """Load the truth object (`objectGuess`, complex (R,R)) from a dataset npz."""
    return np.load(truth_path, allow_pickle=True)["objectGuess"]


def gauged_recon(canvas: np.ndarray, truth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Crop `canvas` to its overlap with `truth` and apply the global complex
    gauge that divides out the count-amplitude scale (and the residual global
    phase offset) before any truth comparison."""
    recon_crop, truth_crop = _overlap_crop(canvas, truth)
    return gauge(recon_crop, truth_crop), truth_crop


def prepare_row(variant_dir: Path, truth: np.ndarray) -> Dict:
    """Load one arm's canvas + metrics and gauge-quotient it against `truth`
    for DISPLAY (figure amplitude/phase panels only).

    Does NOT compute a fidelity-vs-truth NCC here: center-cropping the
    assembled canvas against the padded `objectGuess` is the framing artifact
    `extension-plan.md:30` bans (lines dyads scored 0.28-0.32 vs the
    gate-validated 0.9722 for the same arm) -- see C1 in
    `.superpowers/sdd/ext/audit-e4.md`. Any gate-path fidelity value for
    these rows comes from `build_combined_metrics_table`'s `gate_fidelity`
    argument instead (supplied externally; None means no gate run yet)."""
    canvas, metrics = load_variant(variant_dir)
    gauged, truth_crop = gauged_recon(canvas, truth)
    return {
        "canvas": canvas,
        "gauged": gauged,
        "truth_crop": truth_crop,
        "covered_mask": np.abs(canvas) > 0,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Axis A / Axis C: shared amp|phase[|scatter] grid renderer
# ---------------------------------------------------------------------------

def _shared_vmax(arrays: Sequence[np.ndarray]) -> float:
    return max(float(np.max(a)) for a in arrays if a.size)


def _render_amp_phase_grid(
    row_names: Sequence[str], rows: Sequence[Dict], truth: np.ndarray,
    out_path: Path, include_scatter: bool,
) -> None:
    ncols = 3 if include_scatter else 2
    nrows = 1 + len(rows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

    # Shared per-column color scales: amplitude across truth + every
    # gauge-quotiented recon; phase is naturally bounded to (-pi, pi].
    amp_vmax = _shared_vmax([np.abs(truth)] + [np.abs(r["gauged"]) for r in rows])

    axes[0, 0].imshow(np.abs(truth), cmap="gray", vmin=0, vmax=amp_vmax)
    axes[0, 0].set_title("truth\n|amp|")
    axes[0, 1].imshow(np.angle(truth), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[0, 1].set_title("truth\nphase")
    if include_scatter:
        axes[0, 2].axis("off")
    for ax in axes[0, :2]:
        ax.axis("off")

    for i, (name, row) in enumerate(zip(row_names, rows), start=1):
        axes[i, 0].imshow(np.abs(row["gauged"]), cmap="gray", vmin=0, vmax=amp_vmax)
        axes[i, 0].set_title(f"{name}\n|amp| (gauge-quotiented)")
        axes[i, 1].imshow(np.angle(row["gauged"]), cmap="twilight", vmin=-np.pi, vmax=np.pi)
        axes[i, 1].set_title(f"{name}\nphase")
        for ax in axes[i, :2]:
            ax.axis("off")
        if include_scatter:
            covered = row["canvas"][row["covered_mask"]]
            ax = axes[i, 2]
            if covered.size:
                ax.scatter(covered.real, covered.imag, s=2, alpha=0.3)
            theta = np.linspace(0, 2 * np.pi, 200)
            ax.plot(np.cos(theta), np.sin(theta), "r--", linewidth=1)
            ax.axhline(0, color="0.8", linewidth=0.5)
            ax.axvline(0, color="0.8", linewidth=0.5)
            ax.set_aspect("equal")
            ax.set_title(f"{name}\nreal-imag (raw canvas)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def compose_axis_a_figure(
    repr_ampphase_dir: Path, repr_realimag_dir: Path, truth_path: Path, out_path: Path,
) -> Dict[str, Dict]:
    """Axis A: representation (`repr_ampphase` vs `repr_realimag`, `probe_varpro`).

    Columns: amplitude | phase | real-imag scatter. Rows: truth then each arm.
    """
    truth = load_truth(truth_path)
    row_spec = [("repr_ampphase", repr_ampphase_dir), ("repr_realimag", repr_realimag_dir)]
    rows = {name: prepare_row(d, truth) for name, d in row_spec}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _render_amp_phase_grid(
        [n for n, _ in row_spec], [rows[n] for n, _ in row_spec], truth, out_path, include_scatter=True,
    )
    return rows


def compose_axis_c_figure(
    rows_spec: Sequence[Tuple[str, Path]], truth_path: Path, out_path: Path,
) -> Dict[str, Dict]:
    """Axis C: the Phase-1 CNN `gs1_frozen`/`gs1_trainable` dyads.
    Columns: amplitude | phase (no scatter column).
    `rows_spec` is `[(row_label, variant_dir), ...]` in display order.
    """
    truth = load_truth(truth_path)
    rows = {name: prepare_row(d, truth) for name, d in rows_spec}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _render_amp_phase_grid(
        [n for n, _ in rows_spec], [rows[n] for n, _ in rows_spec], truth, out_path, include_scatter=False,
    )
    return rows


# ---------------------------------------------------------------------------
# Axis B: flux-sweep SCALE/FIDELITY table parsing + figure
# ---------------------------------------------------------------------------

_SCALE_ROW_RE = re.compile(
    r"^\s*(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s*$"
)
_SCALE_KEYS = ("mean_ct", "sqrt_ct", "s1", "s2", "c_A", "c_phi_deg", "O_on", "O_off")

_FIDELITY_ROW_RE = re.compile(
    r"^\s*(\d+)\s*\|\s*(\d+\.\d+)\s+(\d+\.\d+)\s*\|\s*(\d+\.\d+)\s+(\d+\.\d+)\s*\|\s*(\d+)\s*$"
)
_FIDELITY_KEYS = ("mean_ct", "ampNCC_on", "ampNCC_off", "phaseMAE_on", "phaseMAE_off", "n_cov")

# Fallback SCALE rows transcribed verbatim from task-E3-report.md, used only
# when fewer than 3 rows are recovered from the eval .txt's own SCALE table
# (the .txt interleaves TensorFlow/Lightning log noise around the table).
_FALLBACK_SCALE = {
    "cnn_ri": [
        (1, 1.0, 0.0025, 0.0084, 0.0087, 73.07, 0.0234, 8.369),
        (100, 10.0, 0.0277, 0.0476, 0.0550, 59.79, 0.264, 9.372),
        (10000, 100.0, 0.277, 0.477, 0.551, 59.84, 2.638, 9.372),
    ],
}


def _extract_section(text: str, header: str) -> List[str]:
    """Return the lines strictly between a `== header` line and the next `==` line."""
    lines: List[str] = []
    in_section = False
    for line in text.splitlines():
        if header in line:
            in_section = True
            continue
        if in_section and line.strip().startswith("=="):
            break
        if in_section:
            lines.append(line)
    return lines


def parse_scale_table(eval_txt_path: Path, generator: str) -> List[Dict[str, float]]:
    """Parse the `== SCALE (varpro solve) ==` rows of a flux-sweep eval log.

    Falls back to the numbers transcribed in `task-E3-report.md` (keyed by
    `generator`, currently only `"cnn_ri"`) if fewer than 3 rows are recovered.
    """
    rows: List[Dict[str, float]] = []
    for line in _extract_section(eval_txt_path.read_text(), "== SCALE"):
        m = _SCALE_ROW_RE.match(line)
        if m:
            rows.append(dict(zip(_SCALE_KEYS, (float(g) for g in m.groups()))))
    if len(rows) < 3:
        rows = [dict(zip(_SCALE_KEYS, vals)) for vals in _FALLBACK_SCALE[generator]]
    return rows


def parse_fidelity_table(eval_txt_path: Path) -> List[Dict[str, float]]:
    """Parse the `== FIDELITY ... gauge-quotiented ==` rows of a flux-sweep
    eval log. Returns an empty list (no fallback) if none are recovered --
    the merged table leaves the fidelity column blank for those rows -- but
    warns loudly first, since a zero-row parse of an existing input is
    otherwise indistinguishable from Axis A/C's intentional "no gate value"
    absence (M2, `.superpowers/sdd/ext/audit-e4.md`)."""
    rows: List[Dict[str, float]] = []
    for line in _extract_section(eval_txt_path.read_text(), "== FIDELITY"):
        m = _FIDELITY_ROW_RE.match(line)
        if m:
            rows.append(dict(zip(_FIDELITY_KEYS, (float(g) for g in m.groups()))))
    if not rows:
        print(
            f"WARNING: parse_fidelity_table parsed 0 rows from {eval_txt_path} "
            "-- the '== FIDELITY ==' table may be missing or its format may "
            "have drifted from the expected regex. Returning an empty list.",
            file=sys.stderr,
        )
    return rows


def compose_axis_b_figure(
    cnn_ri_eval_txt: Path, out_path: Path,
) -> Tuple[Dict[str, List[Dict[str, float]]], Dict[str, List[Dict[str, float]]]]:
    """Axis B: dynamic scaling across flux for the `cnn_ri` generator.

    Panels: log-log c_A vs mean-count (with a sqrt(mean-count) reference
    line), c_phi (deg) vs mean-count, and |O|_on vs |O|_off vs mean-count.
    No Fourier/measurement-error panel: the eval logs do not report one.
    """
    scale_tables = {
        "cnn_ri": parse_scale_table(cnn_ri_eval_txt, "cnn_ri"),
    }
    fidelity_tables = {
        "cnn_ri": parse_fidelity_table(cnn_ri_eval_txt),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {"cnn_ri": "tab:blue"}

    mean_ct_grid = np.array([r["mean_ct"] for r in scale_tables["cnn_ri"]])
    mid = len(mean_ct_grid) // 2
    anchor = float(np.mean([scale_tables[g][mid]["c_A"] for g in scale_tables]))
    ref = anchor * np.sqrt(mean_ct_grid / mean_ct_grid[mid])
    axes[0].loglog(mean_ct_grid, ref, "k--", label="sqrt(mean count) ref")

    for gen, rows in scale_tables.items():
        mean_ct = np.array([r["mean_ct"] for r in rows])
        axes[0].loglog(mean_ct, [r["c_A"] for r in rows], "o-", color=colors[gen], label=gen)
        axes[1].semilogx(mean_ct, [r["c_phi_deg"] for r in rows], "o-", color=colors[gen], label=gen)
        axes[2].loglog(mean_ct, [r["O_on"] for r in rows], "o-", color=colors[gen], label=f"{gen} |O|_on")
        axes[2].loglog(mean_ct, [r["O_off"] for r in rows], "s--", color=colors[gen], label=f"{gen} |O|_off")

    axes[0].set_xlabel("mean count"); axes[0].set_ylabel("c_A = |s|"); axes[0].set_title("c_A vs flux (log-log)")
    axes[1].set_xlabel("mean count"); axes[1].set_ylabel("c_phi (deg)"); axes[1].set_title("c_phi vs flux (flux-invariant)")
    axes[2].set_xlabel("mean count"); axes[2].set_ylabel("|O|"); axes[2].set_title("|O|_on vs |O|_off vs flux")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return scale_tables, fidelity_tables


# ---------------------------------------------------------------------------
# Merged metrics table
# ---------------------------------------------------------------------------

_METHODOLOGY_NOTE = (
    "**Methodology note (C1 fix, see `.superpowers/sdd/ext/audit-e4.md`):** "
    "`amp_fidelity_ncc_gate` is the validated gate-path amplitude NCC vs. "
    "truth ONLY -- direct per-patch placement into the truth object's own "
    "pixel frame, gauge-quotiented (the `recon_quality_gate.py` methodology). "
    "Axis B: parsed from the eval log's `== FIDELITY ... direct-placement, "
    "gauge-quotiented ==` table. Axis A/C: supplied externally via the "
    "`gate_fidelity` argument (CLI: `--gate-fidelity-json`); `n/a` here means "
    "no gate run has been performed yet for that arm, NOT a computed value -- "
    "this column never contains the center-crop-canvas-vs-padded-`objectGuess` "
    "NCC that `extension-plan.md:30` bans as an untrustworthy framing artifact "
    "(lines dyads previously scored 0.28-0.32 there vs. gate-validated 0.9722)."
)


def build_combined_metrics_table(
    axis_a_rows: Dict[str, Dict],
    axis_c_rows: Dict[str, Dict],
    axis_b_scale_tables: Dict[str, List[Dict[str, float]]],
    axis_b_fidelity_tables: Dict[str, List[Dict[str, float]]],
    out_json_path: Path,
    out_md_path: Path,
    gate_fidelity: Dict[str, float] | None = None,
) -> List[Dict]:
    """One row per arm across all three axes: phase MAE, s1, s2, c_A, c_phi,
    and a SINGLE-methodology `amp_fidelity_ncc_gate` column (C1 fix). That
    column is populated only from the validated gate placement: Axis B rows
    from the eval log's gate-validated FIDELITY table; Axis A/C rows from the
    optional `gate_fidelity` mapping (`{arm_name: gate_ncc}`, e.g. a future
    `recon_quality_gate.py` run -- an R3 input) and are `None`/absent-valued
    when no such value is supplied. The banned center-crop-vs-truth NCC is
    never computed here. Writes both `out_json_path` (JSON) and `out_md_path`
    (markdown, with a methodology footnote)."""
    gate_fidelity = gate_fidelity or {}
    table: List[Dict] = []

    for axis_name, rows in (("A", axis_a_rows), ("C", axis_c_rows)):
        for arm_name, row in rows.items():
            m = row["metrics"]
            table.append({
                "axis": axis_name,
                "arm": arm_name,
                "amp_fidelity_ncc_gate": gate_fidelity.get(arm_name),
                "phase_mae": m.get("phase_mae"),
                "s1": m.get("s1"),
                "s2": m.get("s2"),
                "c_A": None,
                "c_phi_deg": None,
            })

    for gen, rows in axis_b_scale_tables.items():
        fidelity_by_ct = {r["mean_ct"]: r for r in axis_b_fidelity_tables.get(gen, [])}
        for r in rows:
            fid = fidelity_by_ct.get(r["mean_ct"])
            table.append({
                "axis": "B",
                "arm": f"{gen}_mean{int(r['mean_ct'])}",
                "amp_fidelity_ncc_gate": fid["ampNCC_on"] if fid else None,
                "phase_mae": fid["phaseMAE_on"] if fid else None,
                "s1": r["s1"],
                "s2": r["s2"],
                "c_A": r["c_A"],
                "c_phi_deg": r["c_phi_deg"],
            })

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(table, indent=2))

    def fmt(v) -> str:
        return f"{v:.4g}" if isinstance(v, (int, float)) else "n/a"

    columns = ["axis", "arm", "amp_fidelity_ncc_gate", "phase_mae", "s1", "s2", "c_A", "c_phi_deg"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in table:
        cells = [str(row["axis"]), row["arm"]] + [fmt(row[c]) for c in columns[2:]]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(_METHODOLOGY_NOTE)
    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    out_md_path.write_text("\n".join(lines) + "\n")

    return table


# ---------------------------------------------------------------------------
# CLI entrypoint (real E3 outputs)
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    artifacts = REPO / ".artifacts" / "varpro_ablation"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ext-matrix-root", type=Path, default=artifacts / "ext_matrix")
    parser.add_argument("--matrix-lines-root", type=Path, default=artifacts / "matrix_lines")
    parser.add_argument("--fluxsweep-root", type=Path, default=artifacts / "ext_fluxsweep")
    parser.add_argument("--truth", type=Path, default=artifacts / "datasets" / "lines_N64_test.npz")
    parser.add_argument("--out-dir", type=Path, default=artifacts / "composite")
    parser.add_argument(
        "--gate-fidelity-json", type=Path, default=None,
        help="Optional JSON file mapping Axis-A/C arm name -> validated "
             "gate-path amplitude NCC (e.g. from a recon_quality_gate.py run, "
             "an R3 input). Omitted arms read null in amp_fidelity_ncc_gate "
             "(C1 fix -- never the banned center-crop NCC).",
    )
    args = parser.parse_args(argv)

    axis_a_rows = compose_axis_a_figure(
        args.ext_matrix_root / "repr_ampphase" / "probe_varpro",
        args.ext_matrix_root / "repr_realimag" / "probe_varpro",
        args.truth,
        args.out_dir / "axis_a_representation.png",
    )

    axis_c_rows_spec = [
        ("gs1_frozen_neither", args.matrix_lines_root / "gs1_frozen" / "uniform_novarpro"),
        ("gs1_frozen_both", args.matrix_lines_root / "gs1_frozen" / "probe_varpro"),
        ("gs1_trainable_neither", args.matrix_lines_root / "gs1_trainable" / "uniform_novarpro"),
        ("gs1_trainable_both", args.matrix_lines_root / "gs1_trainable" / "probe_varpro"),
    ]
    axis_c_rows = compose_axis_c_figure(
        axis_c_rows_spec, args.truth, args.out_dir / "axis_c_gs1_dyads.png",
    )

    scale_tables, fidelity_tables = compose_axis_b_figure(
        args.fluxsweep_root / "cnn_ri_eval.txt",
        args.out_dir / "axis_b_flux_scaling.png",
    )

    gate_fidelity = (
        json.loads(args.gate_fidelity_json.read_text()) if args.gate_fidelity_json else None
    )
    build_combined_metrics_table(
        axis_a_rows, axis_c_rows, scale_tables, fidelity_tables,
        args.out_dir / "combined_metrics.json", args.out_dir / "combined_metrics.md",
        gate_fidelity=gate_fidelity,
    )
    print(f"Wrote figures + table to {args.out_dir}")


if __name__ == "__main__":
    main()
