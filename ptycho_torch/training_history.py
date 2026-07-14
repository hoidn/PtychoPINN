"""Capture per-run training histories from the Lightning CSV logger.

The CI model-compatibility ablation design ("Training And Inference Flow"
step 4 in docs/superpowers/specs/2026-07-09-ci-model-compatibility-ablation-design.md)
requires the runtime to record losses, gradient norms, and output statistics
alongside the checkpoint identity. Lightning already routes every
``LightningModule.log(...)`` metric to the CSV logger's ``metrics.csv``
(training/validation losses, ``grad_norm_preclip``/``grad_norm_postclip``
when ``training_config.log_grad_norm`` is enabled, and per-arm output
statistics). This module parses that existing log into a JSON-able history
instead of adding a parallel recording mechanism.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

TRAINING_HISTORY_SCHEMA = "training_history_v1"
_INDEX_COLUMNS = ("step", "epoch")


def _metrics_csv_path(run_dir: Path, csv_logger: Any = None) -> Path:
    experiment = getattr(csv_logger, "experiment", None)
    recorded = getattr(experiment, "metrics_file_path", None)
    if recorded:
        return Path(recorded)
    return Path(run_dir) / "metrics.csv"


def _parse_index(raw: Optional[str]) -> Optional[int]:
    if raw is None or raw == "":
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def read_metrics_series(metrics_csv: Path) -> Dict[str, Dict[str, List[Any]]]:
    """Parse a Lightning ``metrics.csv`` into per-metric step/epoch/value series."""
    series: Dict[str, Dict[str, List[Any]]] = {}
    with Path(metrics_csv).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            step = _parse_index(row.get("step"))
            epoch = _parse_index(row.get("epoch"))
            for name, raw in row.items():
                if name in _INDEX_COLUMNS or raw is None or raw == "":
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    continue
                entry = series.setdefault(name, {"step": [], "epoch": [], "value": []})
                entry["step"].append(step)
                entry["epoch"].append(epoch)
                entry["value"].append(value)
    return series


def build_training_history(
    run_dir: Path,
    *,
    csv_logger: Any = None,
    model: Any = None,
    training_config: Any = None,
) -> Optional[Dict[str, Any]]:
    """Build the JSON-able training history for ``TrainingRunResult``.

    Returns ``None`` when no ``metrics.csv`` exists (e.g. stubbed trainers in
    tests); callers must treat that as an explicit, typed absence rather than
    fabricate a history.
    """
    metrics_csv = _metrics_csv_path(run_dir, csv_logger)
    if not metrics_csv.is_file():
        return None
    return {
        "schema_version": TRAINING_HISTORY_SCHEMA,
        "source": "lightning_csv_logger",
        "metrics_csv": str(metrics_csv),
        "train_loss_name": getattr(model, "loss_name", None),
        "val_loss_name": getattr(model, "val_loss_name", None),
        "gradient_clip_val": getattr(training_config, "gradient_clip_val", None),
        "gradient_clip_algorithm": getattr(
            training_config, "gradient_clip_algorithm", None
        ),
        "series": read_metrics_series(metrics_csv),
    }
