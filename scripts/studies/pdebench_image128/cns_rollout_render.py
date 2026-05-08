"""Rendering helpers for CNS rollout GIF/MP4 artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from scripts.studies.pdebench_image128.cns_rollout import CnsRolloutResult
from scripts.studies.pdebench_image128.visualization import cfd_cns_shared_scale_bundle


def _frame_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    return np.ascontiguousarray(rgba[..., :3])


def _field_index(result: CnsRolloutResult, field: str) -> int:
    try:
        return list(result.field_order).index(field)
    except ValueError as exc:
        raise ValueError(f"unknown CNS field {field!r}; expected one of {list(result.field_order)}") from exc


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def render_field_rollout_gif(
    *,
    result: CnsRolloutResult,
    field: str,
    output_path: Path,
    fps: float = 4.0,
    include_error: bool = False,
) -> Path:
    """Render one field rollout as a side-by-side animated GIF."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    channel = _field_index(result, field)
    initial = np.asarray(result.initial_state_phys[channel], dtype=np.float32)
    true_frames = np.asarray(result.true_phys[:, channel], dtype=np.float32)
    pred_frames = np.asarray(result.pred_phys[:, channel], dtype=np.float32)
    error_frames = np.asarray(result.abs_error_phys[:, channel], dtype=np.float32)
    scale_bundle = cfd_cns_shared_scale_bundle(
        field,
        value_arrays=[initial, *list(true_frames), *list(pred_frames)],
        error_arrays=list(error_frames),
    )
    value_scale = scale_bundle["value_scale"]
    error_scale = scale_bundle["error_scale"]
    panel_layout = ["initial", "true", "prediction"]
    if include_error:
        panel_layout.append("absolute_error")
    frames = []
    for index, time_index in enumerate(result.frame_time_indices):
        fig, axes = plt.subplots(1, len(panel_layout), figsize=(4.0 * len(panel_layout), 3.8), squeeze=False)
        panels = [
            (f"Initial {field}", initial, value_scale),
            (f"True {field} t={time_index}", true_frames[index], value_scale),
            (f"Pred {field} t={time_index}", pred_frames[index], value_scale),
        ]
        if include_error:
            panels.append((f"Abs error {field}", error_frames[index], error_scale))
        for axis, (title, image, spec) in zip(axes[0], panels, strict=True):
            handle = axis.imshow(image, cmap=spec["cmap"], vmin=spec["vmin"], vmax=spec["vmax"])
            axis.set_title(title)
            axis.set_xticks([])
            axis.set_yticks([])
            fig.colorbar(handle, ax=axis, fraction=0.046, pad=0.04)
        fig.tight_layout()
        frames.append(_frame_to_rgb(fig))
        plt.close(fig)
    imageio.mimsave(output_path, frames, duration=1.0 / max(float(fps), 1e-6))
    _write_json(
        output_path.with_suffix(".json"),
        {
            "schema_version": "pdebench_cns_rollout_gif_manifest_v1",
            "field": field,
            "frame_count": int(len(frames)),
            "fps": float(fps),
            "panel_layout": panel_layout,
            "value_scale": {key: (float(value) if isinstance(value, (int, float, np.floating)) else value) for key, value in value_scale.items()},
            "error_scale": {key: (float(value) if isinstance(value, (int, float, np.floating)) else value) for key, value in error_scale.items()},
            "output_path": str(output_path),
        },
    )
    return output_path


def render_all_field_rollouts(
    *,
    result: CnsRolloutResult,
    output_root: Path,
    stem: str,
    fps: float = 4.0,
    include_error: bool = False,
) -> list[Path]:
    return [
        render_field_rollout_gif(
            result=result,
            field=field,
            output_path=Path(output_root) / f"{stem}_{field}_rollout.gif",
            fps=fps,
            include_error=include_error,
        )
        for field in result.field_order
    ]
