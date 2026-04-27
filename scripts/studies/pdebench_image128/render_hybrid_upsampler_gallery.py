"""Render side-by-side Hybrid upsampler galleries from saved comparison NPZs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from scripts.studies.pdebench_image128.visualization import cfd_cns_field_visual_spec


def _parse_profile_ids(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _load_comparison(run_root: Path, profile_id: str, sample_index: int) -> dict[str, Any]:
    path = run_root / f"comparison_{profile_id}_sample{sample_index}.npz"
    with np.load(path, allow_pickle=False) as data:
        field_order = [str(item) for item in data["field_order"].tolist()]
        return {
            "profile_id": profile_id,
            "path": path,
            "prediction": np.asarray(data["prediction"]),
            "target": np.asarray(data["target"]),
            "abs_error": np.asarray(data["abs_error"]),
            "field_order": field_order,
        }


def _assert_targets_match(records: list[dict[str, Any]]) -> None:
    first = records[0]
    for record in records[1:]:
        if record["prediction"].shape != first["prediction"].shape:
            raise ValueError(f"shape mismatch between {first['profile_id']} and {record['profile_id']}")
        if record["field_order"] != first["field_order"]:
            raise ValueError(f"field_order mismatch between {first['profile_id']} and {record['profile_id']}")
        if not np.allclose(record["target"], first["target"], atol=1e-6, rtol=1e-6):
            raise ValueError(f"target mismatch between {first['profile_id']} and {record['profile_id']}")


def _render_gallery(
    *,
    target: np.ndarray,
    panels: list[tuple[str, np.ndarray]],
    field_order: list[str],
    output_path: Path,
    is_error: bool,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    channels = int(target.shape[0])
    fig, axes = plt.subplots(
        channels,
        len(panels),
        figsize=(4.2 * len(panels), 3.0 * channels),
        squeeze=False,
        constrained_layout=True,
    )
    for channel in range(channels):
        label = field_order[channel] if channel < len(field_order) else f"c{channel}"
        if is_error:
            spec = cfd_cns_field_visual_spec(label, [image[channel] for _, image in panels], is_error=True)
        else:
            spec = cfd_cns_field_visual_spec(label, [target[channel], *(image[channel] for _, image in panels)])
        for col, (title, image) in enumerate(panels):
            axis = axes[channel][col]
            handle = axis.imshow(image[channel], cmap=spec["cmap"], vmin=spec["vmin"], vmax=spec["vmax"])
            if channel == 0:
                axis.set_title(title)
            if col == 0:
                axis.set_ylabel(label)
            axis.set_xticks([])
            axis.set_yticks([])
            fig.colorbar(handle, ax=axis, fraction=0.046, pad=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def render_galleries(
    *,
    run_root: Path,
    sample_index: int,
    baseline_profile: str,
    variant_profiles: list[str],
    output_png: Path,
    output_error_png: Path,
) -> dict[str, str]:
    profile_ids = [baseline_profile, *variant_profiles]
    ordered_profile_ids: list[str] = []
    for profile_id in profile_ids:
        if profile_id not in ordered_profile_ids:
            ordered_profile_ids.append(profile_id)
    records = [_load_comparison(run_root, profile_id, sample_index) for profile_id in ordered_profile_ids]
    _assert_targets_match(records)
    field_order = records[0]["field_order"]
    target = records[0]["target"]
    prediction_panels = [("Ground truth", target)] + [(record["profile_id"], record["prediction"]) for record in records]
    error_panels = [(record["profile_id"], record["abs_error"]) for record in records]
    prediction_path = _render_gallery(
        target=target,
        panels=prediction_panels,
        field_order=field_order,
        output_path=output_png,
        is_error=False,
    )
    error_path = _render_gallery(
        target=target,
        panels=error_panels,
        field_order=field_order,
        output_path=output_error_png,
        is_error=True,
    )
    return {"prediction_gallery": str(prediction_path), "error_gallery": str(error_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--baseline-profile", required=True)
    parser.add_argument("--variant-profiles", default="")
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-error-png", type=Path, required=True)
    args = parser.parse_args()

    result = render_galleries(
        run_root=args.run_root,
        sample_index=int(args.sample_index),
        baseline_profile=str(args.baseline_profile),
        variant_profiles=_parse_profile_ids(args.variant_profiles),
        output_png=args.output_png,
        output_error_png=args.output_error_png,
    )
    print(f"PREDICTION_GALLERY={result['prediction_gallery']}")
    print(f"ERROR_GALLERY={result['error_gallery']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
