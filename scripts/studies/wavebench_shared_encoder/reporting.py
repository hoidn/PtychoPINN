from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def read_json_if_exists(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row",
        "latent_channels",
        "status",
        "MAE",
        "RMSE",
        "RelL2",
        "SSIM",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metrics = row.get("metrics", {})
            writer.writerow(
                {
                    "row": row["row"],
                    "latent_channels": row["latent_channels"],
                    "status": row["status"],
                    "MAE": metrics.get("MAE"),
                    "RMSE": metrics.get("RMSE"),
                    "RelL2": metrics.get("RelL2"),
                    "SSIM": metrics.get("SSIM"),
                }
            )

