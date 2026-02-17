"""Tests for study metrics table rendering."""

from scripts.studies.metrics_tables import METRICS, _build_main_table


def test_main_table_keeps_one_cell_per_metric_when_values_missing():
    metrics = {
        "pinn": {
            "mae": (0.1, 0.2),
            # intentionally omit the rest
        }
    }
    table = _build_main_table(metrics, model_ns={"pinn": 64})
    model_line = next(line for line in table.splitlines() if "PtychoPINN (CNN)" in line)
    cells = [cell.strip() for cell in model_line.rstrip("\\").split("&")]
    assert len(cells) == 2 + len(METRICS)
    assert cells[0] == "64"
    assert cells[1] == "PtychoPINN (CNN)"
    # all non-MAE metric cells should be a single '-' placeholder
    assert all(cell == "-" for cell in cells[3:])


def test_main_table_header_contains_binomial_single_image_frc_columns():
    metrics = {"pinn": {"mae": (0.1, 0.2)}}
    table = _build_main_table(metrics, model_ns={"pinn": 64})
    header = next(line for line in table.splitlines() if line.startswith("N & Model"))
    assert "1FRC50 Bin (A/P)" in header
    assert "1FRC1/7 Bin (A/P)" in header


def test_main_table_escapes_underscore_in_fallback_model_labels():
    metrics = {"pinn_custom_model": {"mae": (0.1, 0.2)}}
    table = _build_main_table(metrics, model_ns={"pinn_custom_model": 64})
    assert "pinn\\_custom\\_model" in table
