from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


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
        "mode",
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
                    "mode": row.get("mode"),
                    "status": row["status"],
                    "MAE": metrics.get("MAE"),
                    "RMSE": metrics.get("RMSE"),
                    "RelL2": metrics.get("RelL2"),
                    "SSIM": metrics.get("SSIM"),
                }
            )


def write_row_figures(
    *,
    output_root: Path,
    row: str,
    latent_channels: int,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    sample_indices: Iterable[int],
) -> dict[str, Any]:
    sample_indices = tuple(int(index) for index in sample_indices)
    figures_dir = output_root / "figures" / f"c{latent_channels}" / row
    arrays_dir = output_root / "figures" / "source_arrays" / row / f"c{latent_channels}"
    figures_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir.mkdir(parents=True, exist_ok=True)

    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    target_min = float(target_np.min())
    target_max = float(target_np.max())
    error = pred_np - target_np
    error_abs = float(np.max(np.abs(error)))

    sample_records = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for sample_idx in sample_indices:
            if sample_idx >= pred_np.shape[0]:
                continue
            pred_2d = pred_np[sample_idx, 0]
            target_2d = target_np[sample_idx, 0]
            error_2d = pred_2d - target_2d

            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            axes[0].imshow(target_2d, vmin=target_min, vmax=target_max, cmap="viridis")
            axes[0].set_title("target")
            axes[0].axis("off")
            axes[1].imshow(pred_2d, vmin=target_min, vmax=target_max, cmap="viridis")
            axes[1].set_title("prediction")
            axes[1].axis("off")
            axes[2].imshow(error_2d, vmin=-error_abs, vmax=error_abs, cmap="coolwarm")
            axes[2].set_title("error")
            axes[2].axis("off")
            fig.suptitle(f"{row} C={latent_channels} sample {sample_idx}")
            fig.tight_layout()
            figure_path = figures_dir / f"sample_{sample_idx:03d}.png"
            fig.savefig(figure_path, dpi=120, bbox_inches="tight")
            plt.close(fig)

            np.savez_compressed(
                arrays_dir / f"sample_{sample_idx:03d}.npz",
                target=target_2d.astype(np.float32),
                prediction=pred_2d.astype(np.float32),
                error=error_2d.astype(np.float32),
            )
            sample_records.append(
                {
                    "sample_index": sample_idx,
                    "figure_path": str(figure_path.relative_to(output_root)),
                    "source_array_path": str(
                        (arrays_dir / f"sample_{sample_idx:03d}.npz").relative_to(output_root)
                    ),
                }
            )
    except ImportError:
        for sample_idx in sample_indices:
            if sample_idx >= pred_np.shape[0]:
                continue
            np.savez_compressed(
                arrays_dir / f"sample_{sample_idx:03d}.npz",
                target=target_np[sample_idx, 0].astype(np.float32),
                prediction=pred_np[sample_idx, 0].astype(np.float32),
                error=(pred_np[sample_idx, 0] - target_np[sample_idx, 0]).astype(np.float32),
            )
            sample_records.append(
                {
                    "sample_index": sample_idx,
                    "figure_path": None,
                    "source_array_path": str(
                        (arrays_dir / f"sample_{sample_idx:03d}.npz").relative_to(output_root)
                    ),
                }
            )

    manifest = {
        "row": row,
        "latent_channels": latent_channels,
        "shared_color_scale": {
            "target_min": target_min,
            "target_max": target_max,
            "error_abs": error_abs,
        },
        "samples": sample_records,
    }
    write_json(output_root / "figures" / f"c{latent_channels}" / row / "figure_manifest.json", manifest)
    return manifest

