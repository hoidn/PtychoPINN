"""Fast, no-training unit tests for scripts/studies/compose_varpro_comparison_grid.py.

Builds tiny SYNTHETIC arm directories (no dependency on real `.artifacts/`
outputs) and exercises the three axis-figure entrypoints plus the merged
metrics table writer end to end. Task E4 (`.superpowers/sdd/ext/task-E4-brief.md`),
cross-checked against `.superpowers/sdd/ext/task-E3-report.md` for the real
input-data layout these synthetic dirs mimic.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "studies"))
import compose_varpro_comparison_grid as compose  # noqa: E402


def _write_variant(variant_dir: Path, canvas: np.ndarray, s1: float = 1.0, s2: float = 1.0) -> Path:
    variant_dir.mkdir(parents=True, exist_ok=True)
    np.savez(variant_dir / "canvas.npz", canvas=canvas.astype(np.complex64))
    metrics = {
        "amp_mae": 0.1,
        "phase_mae": 0.05,
        "complex_mae": 0.12,
        "s1": s1,
        "s2": s2,
        "canvas_amp_std": float(np.std(np.abs(canvas))),
        "canvas_phase_std": float(np.std(np.angle(canvas))),
        "rail_fraction_real": 0.0,
        "rail_fraction_imag": 0.0,
        "diagnostics_basis": "prescale_canvas",
    }
    (variant_dir / "metrics.json").write_text(json.dumps(metrics))
    return variant_dir


def _synthetic_truth(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    obj = (0.5 + 0.5 * rng.uniform(size=(20, 20))) * np.exp(1j * 0.3 * rng.uniform(size=(20, 20)))
    truth_path = tmp_path / "truth.npz"
    np.savez(truth_path, objectGuess=obj.astype(np.complex64))
    return truth_path


def _synthetic_canvas(truth: np.ndarray, scale: complex, noise: float, rng: np.random.Generator) -> np.ndarray:
    """A (12,12) canvas correlated with the center crop of `truth`, at an
    arbitrary complex gauge (`scale`) plus a border of exact zeros (uncovered
    pixels) -- mirrors the real harness's zero-padded canvas.npz."""
    crop = truth[4:10, 4:10]
    core = scale * crop + noise * (rng.normal(size=crop.shape) + 1j * rng.normal(size=crop.shape))
    canvas = np.zeros((12, 12), dtype=np.complex64)
    canvas[3:9, 3:9] = core
    return canvas


SCALE_TXT_TEMPLATE = """label: {label}
some unrelated log noise here
tensor(0.1234, device='cuda:0')

== SCALE (varpro solve) ==
 mean_ct  sqrt(ct)        s1        s2   c_A=|s|  cphi_deg    |O|_on   |O|_off
------------------------------------------------------------------------------
Memory map length: 59
       1     1.000    0.0025    0.0084    0.0087    73.066    0.0234    8.3687
Memory map length: 59
     100    10.000    0.0277    0.0476    0.0550    59.790    0.2637    9.3716
Memory map length: 59
   10000   100.000    0.2769    0.4766    0.5512    59.838    2.6381    9.3715

== RAW MAE vs truth (phase-aligned barycentric canvas only; NOT gauge-quotiented) ==
 mean_ct | amp_mae ON      OFF |  phase_mae ON      OFF
-------------------------------------------------------
       1 |     0.7040   6.5819 |        0.0959   0.1064

== FIDELITY vs truth (label={label}, direct-placement, gauge-quotiented; anchor |amp| NCC=skipped) ==
 mean_ct |  ampNCC_gauged ON      OFF |  phaseMAE_gauged ON      OFF |  n_cov
-----------------------------------------------------------------------------
       1 |            0.7739   0.7752 |              0.0828   0.1007 |  25335
     100 |            0.8653   0.8651 |              0.0881   0.1000 |  25335
   10000 |            0.8653   0.8651 |              0.0881   0.1000 |  25335
"""


@pytest.fixture
def synthetic_layout(tmp_path):
    truth_path = _synthetic_truth(tmp_path)
    truth = np.load(truth_path)["objectGuess"]
    rng = np.random.default_rng(1)

    ext_matrix = tmp_path / "ext_matrix"
    ampphase_dir = _write_variant(
        ext_matrix / "repr_ampphase" / "probe_varpro",
        _synthetic_canvas(truth, 2.0 + 0.5j, 0.02, rng), s1=0.03, s2=-0.01,
    )
    realimag_dir = _write_variant(
        ext_matrix / "repr_realimag" / "probe_varpro",
        _synthetic_canvas(truth, 1.5 - 0.3j, 0.02, rng), s1=0.03, s2=0.05,
    )

    neither_dir = _write_variant(
        ext_matrix / "hybres_gs1_neither" / "uniform_novarpro",
        _synthetic_canvas(truth, 1.0 + 0.0j, 0.05, rng), s1=1.0, s2=1.0,
    )
    both_dir = _write_variant(
        ext_matrix / "hybres_gs1_both" / "probe_varpro",
        _synthetic_canvas(truth, 3.0 + 0.1j, 0.02, rng), s1=131.65, s2=0.548,
    )

    matrix_lines = tmp_path / "matrix_lines"
    dyad_neither = _write_variant(
        matrix_lines / "gs1_frozen" / "uniform_novarpro",
        _synthetic_canvas(truth, 1.0 + 0.0j, 0.02, rng), s1=1.0, s2=1.0,
    )
    dyad_both = _write_variant(
        matrix_lines / "gs1_frozen" / "probe_varpro",
        _synthetic_canvas(truth, 4.5 + 0.2j, 0.02, rng), s1=4.46, s2=8.69,
    )

    fluxsweep = tmp_path / "ext_fluxsweep"
    fluxsweep.mkdir()
    cnn_ri_txt = fluxsweep / "cnn_ri_eval.txt"
    cnn_ri_txt.write_text(SCALE_TXT_TEMPLATE.format(label="cnn_ri"))
    hybres_txt = fluxsweep / "hybres_eval.txt"
    hybres_txt.write_text(SCALE_TXT_TEMPLATE.format(label="hybres"))

    return {
        "truth_path": truth_path,
        "ampphase_dir": ampphase_dir,
        "realimag_dir": realimag_dir,
        "axis_c_rows": [
            ("hybres_gs1_neither", neither_dir),
            ("hybres_gs1_both", both_dir),
            ("gs1_frozen_neither", dyad_neither),
            ("gs1_frozen_both", dyad_both),
        ],
        "cnn_ri_txt": cnn_ri_txt,
        "hybres_txt": hybres_txt,
    }


# ---------------------------------------------------------------------------
# Parsing helpers (pure, fast)
# ---------------------------------------------------------------------------

def test_parse_scale_table_recovers_three_rows_from_noisy_log(synthetic_layout):
    rows = compose.parse_scale_table(synthetic_layout["cnn_ri_txt"], "cnn_ri")

    assert len(rows) == 3
    assert rows[0]["mean_ct"] == 1
    assert rows[1]["mean_ct"] == 100
    assert rows[2]["mean_ct"] == 10000
    assert rows[1]["c_A"] == pytest.approx(0.0550)
    assert rows[1]["c_phi_deg"] == pytest.approx(59.790)


def test_parse_scale_table_falls_back_when_rows_missing(tmp_path):
    bad_txt = tmp_path / "bad_eval.txt"
    bad_txt.write_text("== SCALE (varpro solve) ==\nnot a data row\n== RAW MAE ==\n")

    rows = compose.parse_scale_table(bad_txt, "hybres")

    assert len(rows) == 3
    assert rows[0]["mean_ct"] == 1


def test_parse_fidelity_table_recovers_rows(synthetic_layout):
    rows = compose.parse_fidelity_table(synthetic_layout["cnn_ri_txt"])

    assert len(rows) == 3
    assert rows[0]["ampNCC_on"] == pytest.approx(0.7739)


# ---------------------------------------------------------------------------
# Figure entrypoints
# ---------------------------------------------------------------------------

def test_compose_axis_a_figure_writes_nonempty_file(tmp_path, synthetic_layout):
    out_path = tmp_path / "out" / "axis_a.png"

    rows = compose.compose_axis_a_figure(
        synthetic_layout["ampphase_dir"], synthetic_layout["realimag_dir"],
        synthetic_layout["truth_path"], out_path,
    )

    assert out_path.exists() and out_path.stat().st_size > 0
    assert set(rows.keys()) == {"repr_ampphase", "repr_realimag"}


def test_compose_axis_b_figure_writes_nonempty_file(tmp_path, synthetic_layout):
    out_path = tmp_path / "out" / "axis_b.png"

    scale_tables, fidelity_tables = compose.compose_axis_b_figure(
        synthetic_layout["cnn_ri_txt"], synthetic_layout["hybres_txt"], out_path,
    )

    assert out_path.exists() and out_path.stat().st_size > 0
    assert set(scale_tables.keys()) == {"cnn_ri", "hybres"}
    assert len(scale_tables["cnn_ri"]) == 3


def test_compose_axis_c_figure_writes_nonempty_file(tmp_path, synthetic_layout):
    out_path = tmp_path / "out" / "axis_c.png"

    rows = compose.compose_axis_c_figure(
        synthetic_layout["axis_c_rows"], synthetic_layout["truth_path"], out_path,
    )

    assert out_path.exists() and out_path.stat().st_size > 0
    assert len(rows) == 4


# ---------------------------------------------------------------------------
# Merged metrics table
# ---------------------------------------------------------------------------

def test_combined_metrics_table_has_one_row_per_arm(tmp_path, synthetic_layout):
    axis_a_rows = compose.compose_axis_a_figure(
        synthetic_layout["ampphase_dir"], synthetic_layout["realimag_dir"],
        synthetic_layout["truth_path"], tmp_path / "axis_a.png",
    )
    axis_c_rows = compose.compose_axis_c_figure(
        synthetic_layout["axis_c_rows"], synthetic_layout["truth_path"], tmp_path / "axis_c.png",
    )
    scale_tables, fidelity_tables = compose.compose_axis_b_figure(
        synthetic_layout["cnn_ri_txt"], synthetic_layout["hybres_txt"], tmp_path / "axis_b.png",
    )

    json_path = tmp_path / "combined_metrics.json"
    md_path = tmp_path / "combined_metrics.md"
    table = compose.build_combined_metrics_table(
        axis_a_rows, axis_c_rows, scale_tables, fidelity_tables, json_path, md_path,
    )

    expected_rows = len(axis_a_rows) + len(axis_c_rows) + sum(len(r) for r in scale_tables.values())
    assert len(table) == expected_rows

    assert json_path.exists() and json_path.stat().st_size > 0
    assert md_path.exists() and md_path.stat().st_size > 0
    loaded = json.loads(json_path.read_text())
    assert len(loaded) == expected_rows
    md_text = md_path.read_text()
    assert "repr_ampphase" in md_text
    assert "hybres_gs1_both" in md_text
    assert "cnn_ri_mean100" in md_text


# ---------------------------------------------------------------------------
# C1 -- single-methodology fidelity column (audit-summary.md C1 / audit-e4.md).
#
# Axis-B rows get amp_fidelity_ncc_gate from the eval log's gate-validated
# FIDELITY table; Axis-A/C rows have no gate value this session (that needs a
# GPU recon_quality_gate.py run, deferred to R3) and must read null/absent --
# NEVER the banned center-crop-canvas-vs-padded-objectGuess NCC that used to
# populate this column (extension-plan.md:30; lines dyads scored 0.28-0.32 vs
# gate-validated 0.9722).
# ---------------------------------------------------------------------------

def _build_table(tmp_path, synthetic_layout, **kwargs):
    axis_a_rows = compose.compose_axis_a_figure(
        synthetic_layout["ampphase_dir"], synthetic_layout["realimag_dir"],
        synthetic_layout["truth_path"], tmp_path / "axis_a.png",
    )
    axis_c_rows = compose.compose_axis_c_figure(
        synthetic_layout["axis_c_rows"], synthetic_layout["truth_path"], tmp_path / "axis_c.png",
    )
    scale_tables, fidelity_tables = compose.compose_axis_b_figure(
        synthetic_layout["cnn_ri_txt"], synthetic_layout["hybres_txt"], tmp_path / "axis_b.png",
    )
    return compose.build_combined_metrics_table(
        axis_a_rows, axis_c_rows, scale_tables, fidelity_tables,
        tmp_path / "combined_metrics.json", tmp_path / "combined_metrics.md",
        **kwargs,
    )


def test_axis_a_gate_column_is_null_without_a_supplied_gate_value(tmp_path, synthetic_layout):
    table = _build_table(tmp_path, synthetic_layout)

    axis_a_row = next(r for r in table if r["arm"] == "repr_ampphase")
    assert axis_a_row["amp_fidelity_ncc_gate"] is None
    # The banned center-crop methodology must not exist under any key on this row.
    assert "amp_fidelity_ncc" not in axis_a_row


def test_axis_c_gate_column_is_null_without_a_supplied_gate_value(tmp_path, synthetic_layout):
    table = _build_table(tmp_path, synthetic_layout)

    axis_c_row = next(r for r in table if r["arm"] == "gs1_frozen_neither")
    assert axis_c_row["amp_fidelity_ncc_gate"] is None
    assert "amp_fidelity_ncc" not in axis_c_row


def test_axis_b_gate_column_still_sourced_from_eval_log_fidelity_table(tmp_path, synthetic_layout):
    table = _build_table(tmp_path, synthetic_layout)

    b_row = next(r for r in table if r["arm"] == "cnn_ri_mean100")
    assert b_row["amp_fidelity_ncc_gate"] == pytest.approx(0.8653)


def test_gate_fidelity_override_populates_only_the_named_axis_a_c_arm(tmp_path, synthetic_layout):
    table = _build_table(tmp_path, synthetic_layout, gate_fidelity={"repr_ampphase": 0.9722})

    axis_a_row = next(r for r in table if r["arm"] == "repr_ampphase")
    assert axis_a_row["amp_fidelity_ncc_gate"] == pytest.approx(0.9722)

    other_axis_a_row = next(r for r in table if r["arm"] == "repr_realimag")
    assert other_axis_a_row["amp_fidelity_ncc_gate"] is None

    axis_c_row = next(r for r in table if r["arm"] == "hybres_gs1_both")
    assert axis_c_row["amp_fidelity_ncc_gate"] is None


def test_combined_metrics_md_states_gate_column_methodology(tmp_path, synthetic_layout):
    md_path = tmp_path / "combined_metrics.md"
    _build_table(tmp_path, synthetic_layout)

    md_text = md_path.read_text()
    assert "amp_fidelity_ncc_gate" in md_text
    assert "gate" in md_text.lower()


def test_parse_fidelity_table_warns_loudly_on_zero_parsed_rows(tmp_path, capsys):
    bad_txt = tmp_path / "bad_eval.txt"
    bad_txt.write_text("== FIDELITY (drifted format) ==\nnot a data row\n== DONE ==\n")

    rows = compose.parse_fidelity_table(bad_txt)

    assert rows == []
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "parse_fidelity_table" in captured.err
