"""Figure rendering for ablation reports.

Split out of :mod:`scripts.studies.ablation.reporting` to keep both modules
within the project size budget. This module stays free of training/runtime
imports; it consumes plot-ready ``ReportInput`` evidence and writes PNGs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import textwrap
from typing import TYPE_CHECKING

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
from matplotlib.patches import ConnectionPatch
from matplotlib.transforms import ScaledTranslation

from .reporting_scatter_layout import (
    REPORT_RENDERER_LAYOUT_SCHEMA_VERSION,
    ArmPresentation,
    ArmVisualRole,
    PanelBounds,
    ScatterLayoutPoint,
    ScatterLayoutRecord,
    build_zero_annotation_records,
    layout_scatter_points,
    resolve_arm_visual_roles,
)

if TYPE_CHECKING:
    from .reporting import MilestoneEvidence, ReportInput, ReportRow, RunIdentity


_DISPLAY_ACRONYMS = {
    "ci": "CI",
    "cnn": "CNN",
    "mae": "MAE",
    "nll": "NLL",
    "resnet": "ResNet",
    "xl": "XL",
}


def _prettify_grid_token(value: str) -> str:
    words = re.sub(r"[_-]+", " ", value).split()
    return " ".join(_DISPLAY_ACRONYMS.get(word.lower(), word.title()) for word in words)


def _compact_grid_label(identity: RunIdentity) -> str:
    parts = [part for part in identity.arm_id.split("--") if part]
    components = parts[-2:] if len(parts) >= 2 else parts
    labels = [
        textwrap.shorten(_prettify_grid_token(component), width=24, placeholder="...")
        for component in components
    ]
    return " | ".join((*labels, f"seed {identity.seed}"))


def _compact_arm_label(identity: RunIdentity) -> str:
    parts = [part for part in identity.arm_id.split("--") if part]
    components = parts[-2:] if len(parts) >= 2 else parts
    return " | ".join(_prettify_grid_token(component) for component in components)


def _typed_figure_styles(
    study: ReportInput,
    registry: dict[str, ArmVisualRole] | None = None,
) -> tuple[dict[str, object], dict[int, str]]:
    registry = _typed_visual_role_registry(study) if registry is None else registry
    style_ids = sorted({role.visual_style_id for role in registry.values()})
    palette = plt.get_cmap("tab10").colors
    style_colors = {
        style_id: palette[index % len(palette)]
        for index, style_id in enumerate(style_ids)
    }
    markers = ("o", "s", "^", "D", "P", "X", "v", "<", ">")
    seeds = sorted({identity.seed for identity in study.requested_runs})
    seed_markers = {
        seed: markers[index % len(markers)] for index, seed in enumerate(seeds)
    }
    return style_colors, seed_markers


def _typed_visual_role_registry(study: ReportInput) -> dict[str, ArmVisualRole]:
    return resolve_arm_visual_roles(
        ArmPresentation(
            identity.arm_id,
            str(identity.object_family),
            _compact_arm_label(identity),
        )
        for identity in study.requested_runs
    )


def _ordered_arm_ids(
    study: ReportInput, object_family: str, run_ids: object
) -> list[str]:
    selected = set(run_ids)
    return list(
        dict.fromkeys(
            identity.arm_id
            for identity in study.requested_runs
            if identity.object_family == object_family and identity.run_id in selected
        )
    )


def _scatter_typed_point(
    axis: object,
    identity: RunIdentity,
    x: float,
    y: float,
    style_colors: dict[str, object],
    seed_markers: dict[int, str],
    visual_role: ArmVisualRole,
    *,
    display_offset_points: tuple[float, float] = (0.0, 0.0),
    artist_id: str | None = None,
) -> object:
    transform = None
    if display_offset_points != (0.0, 0.0):
        transform = axis.transData + ScaledTranslation(
            display_offset_points[0] / 72.0,
            display_offset_points[1] / 72.0,
            axis.figure.dpi_scale_trans,
        )
    scatter_options = {} if transform is None else {"transform": transform}
    collection = axis.scatter(
        [x],
        [y],
        color=style_colors[visual_role.visual_style_id],
        marker=seed_markers[identity.seed],
        s=42,
        **scatter_options,
    )
    collection.set_gid(artist_id)
    return collection


def _arm_legend_handles(
    arm_ids: list[str],
    style_colors: dict[str, object],
    registry: dict[str, ArmVisualRole],
) -> list[Line2D]:
    roles = list(dict.fromkeys(registry[arm_id] for arm_id in arm_ids))
    return [
        Line2D(
            [],
            [],
            color=style_colors[role.visual_style_id],
            linewidth=3,
            label=role.display_label,
        )
        for role in roles
    ]


def _seed_legend_handles(seed_markers: dict[int, str]) -> list[Line2D]:
    return [
        Line2D(
            [],
            [],
            color="black",
            marker=marker,
            linestyle="None",
            markersize=6,
            label=f"seed {seed}",
        )
        for seed, marker in seed_markers.items()
    ]


def _add_legend(
    axis: object,
    handles: list[Line2D],
    *,
    title: str,
    anchor: tuple[float, float],
    location: str = "upper left",
) -> None:
    legend = Legend(
        axis,
        handles,
        [handle.get_label() for handle in handles],
        title=title,
        loc=location,
        bbox_to_anchor=anchor,
        bbox_transform=axis.transAxes,
        borderaxespad=0.0,
        frameon=False,
        fontsize=8,
    )
    legend.get_title().set_fontsize(9)
    axis.add_artist(legend)


@dataclass(frozen=True)
class _ScatterPanelPoint:
    identity: RunIdentity
    exact_anchor: tuple[float, float]


@dataclass(frozen=True)
class _ScatterPanelSpec:
    axis: object
    object_family: str
    panel: str
    points: tuple[_ScatterPanelPoint, ...]


def _ordered_panel_points(
    study: ReportInput,
    object_family: str,
    series: dict[str, dict[str, float]],
    x_key: str,
    y_key: str,
) -> tuple[_ScatterPanelPoint, ...]:
    arm_order = {
        arm_id: index
        for index, arm_id in enumerate(
            dict.fromkeys(
                identity.arm_id
                for identity in study.requested_runs
                if identity.object_family == object_family
            )
        )
    }
    identities = sorted(
        (
            identity
            for identity in study.requested_runs
            if identity.object_family == object_family and identity.run_id in series
        ),
        key=lambda identity: (
            arm_order[identity.arm_id],
            identity.seed,
            identity.run_id,
        ),
    )
    return tuple(
        _ScatterPanelPoint(
            identity,
            (series[identity.run_id][x_key], series[identity.run_id][y_key]),
        )
        for identity in identities
    )


def _set_exact_panel_limits(spec: _ScatterPanelSpec) -> None:
    if not spec.points:
        return
    spec.axis.update_datalim(
        np.asarray([point.exact_anchor for point in spec.points], dtype=np.float64)
    )
    spec.axis.autoscale_view()
    spec.axis.set_autoscale_on(False)


def _render_scatter_panels(
    figure: object,
    specs: list[_ScatterPanelSpec],
    style_colors: dict[str, object],
    seed_markers: dict[int, str],
    registry: dict[str, ArmVisualRole],
) -> tuple[ScatterLayoutRecord, ...]:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    point_scale = 72.0 / figure.dpi
    output: list[ScatterLayoutRecord] = []
    for spec in specs:
        bounds = spec.axis.get_window_extent(renderer)
        panel_bounds = PanelBounds(
            bounds.x0 * point_scale,
            bounds.y0 * point_scale,
            bounds.x1 * point_scale,
            bounds.y1 * point_scale,
        )
        layout_points = tuple(
            ScatterLayoutPoint(
                run_id=point.identity.run_id,
                arm_id=point.identity.arm_id,
                visual_role_id=registry[point.identity.arm_id].visual_role_id,
                visual_style_id=registry[point.identity.arm_id].visual_style_id,
                arm_display_label=registry[point.identity.arm_id].display_label,
                object_family=spec.object_family,
                panel=spec.panel,
                seed=point.identity.seed,
                exact_anchor=point.exact_anchor,
                display_anchor_points=tuple(
                    float(value) * point_scale
                    for value in spec.axis.transData.transform(point.exact_anchor)
                ),
                marker_artist_id=(
                    f"scatter-marker:{spec.panel}:{point.identity.run_id}"
                ),
            )
            for point in spec.points
        )
        records = layout_scatter_points(layout_points, panel_bounds)
        by_run = {point.identity.run_id: point for point in spec.points}
        for record in records:
            point = by_run[record.run_id]
            marker = _scatter_typed_point(
                spec.axis,
                point.identity,
                record.exact_anchor[0],
                record.exact_anchor[1],
                style_colors,
                seed_markers,
                registry[point.identity.arm_id],
                display_offset_points=record.marker_display_offset_points,
                artist_id=record.marker_artist_id,
            )
            marker.set_in_layout(False)
            if record.connector_id is not None:
                connector = ConnectionPatch(
                    xyA=record.exact_anchor,
                    coordsA=spec.axis.transData,
                    xyB=record.exact_anchor,
                    coordsB=marker.get_offset_transform(),
                    axesA=spec.axis,
                    axesB=spec.axis,
                    arrowstyle="-",
                    color=style_colors[record.visual_style_id],
                    linewidth=0.75,
                    shrinkA=1,
                    shrinkB=5,
                )
                connector.set_gid(record.connector_id)
                connector.set_in_layout(False)
                spec.axis.add_patch(connector)
        output.extend(records)
    return tuple(output)


def _wrapped_grid_label(identity: RunIdentity) -> str:
    return "\n".join(
        textwrap.wrap(
            _compact_grid_label(identity),
            width=40,
            break_long_words=True,
            break_on_hyphens=False,
        )
    )


def _family_compact_grid_label(identity: RunIdentity) -> str:
    assert identity.object_family is not None
    for separator in ("--", "-"):
        prefix = f"{identity.object_family}{separator}"
        if identity.arm_id.startswith(prefix):
            arm = _prettify_grid_token(identity.arm_id[len(prefix) :])
            compact = textwrap.shorten(arm, width=32, placeholder="...")
            return f"{compact} | seed {identity.seed}"
    return _compact_grid_label(identity)


def _plot_rows(study: ReportInput) -> list[tuple[RunIdentity, ReportRow | None]]:
    actual = {row.attempt.run_id: row for row in study.rows}
    return [
        (identity, actual.get(identity.run_id))
        for identity in sorted(
            study.requested_runs, key=lambda item: (item.arm_id, item.seed, item.run_id)
        )
    ]


def _limits(*arrays: np.ndarray) -> tuple[float, float]:
    values = np.concatenate(
        [np.ravel(array.astype(np.float64, copy=False)) for array in arrays]
    )
    lower, upper = float(np.min(values)), float(np.max(values))
    if lower == upper:
        return lower - 0.5, upper + 0.5
    return lower, upper


def render_milestone_grid(
    milestones: tuple[MilestoneEvidence, ...],
    path: Path,
    *,
    title: str,
) -> None:
    """Render canonical milestone reconstruction arrays without transformation."""
    if not milestones:
        raise ValueError("milestone grid requires at least one reconstruction")
    reconstructions = tuple(
        np.asarray(item.arrays["reconstruction"]) for item in milestones
    )
    if any(array.ndim != 2 or not np.isfinite(array).all() for array in reconstructions):
        raise ValueError("milestone reconstructions must be finite 2D arrays")
    lower, upper = _limits(*reconstructions)
    figure, axes_value = plt.subplots(
        1,
        len(milestones),
        figsize=(3.0 * len(milestones), 3.2),
        squeeze=False,
        facecolor="white",
    )
    axes = tuple(axes_value[0])
    for axis, milestone, reconstruction in zip(
        axes, milestones, reconstructions
    ):
        axis.imshow(
            reconstruction,
            cmap="gray",
            interpolation="nearest",
            vmin=lower,
            vmax=upper,
        )
        axis.set_title(
            f"Epoch {milestone.epoch}",
            color="black",
            bbox={
                "boxstyle": "square,pad=0.2",
                "edgecolor": "none",
                "facecolor": "white",
            },
        )
        axis.axis("off")
    display_title = textwrap.fill(
        _prettify_grid_token(title),
        width=48,
        max_lines=2,
        placeholder="...",
        break_long_words=False,
        break_on_hyphens=False,
    )
    figure.suptitle(display_title, fontsize=10, y=0.98)
    figure.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.04,
        top=0.76,
        wspace=0.08,
    )
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _metric_values(row: ReportRow) -> dict[str, float]:
    return {
        record.path: float(record.value)
        for record in row.metric_records
        if isinstance(record.value, float)
    }


def _visible_not_applicable(axis: object, reason: str) -> None:
    axis.axis("off")
    axis.text(
        0.5, 0.58, "Not applicable", ha="center", va="center", transform=axis.transAxes
    )
    axis.text(
        0.5,
        0.36,
        reason.replace("_", " "),
        ha="center",
        va="center",
        fontsize=8,
        wrap=True,
        transform=axis.transAxes,
    )


def _eligible_ci_run_ids(study: ReportInput) -> set[str]:
    successful = {
        row.attempt.run_id for row in study.rows if row.attempt.terminal_success
    }
    return {
        identity.run_id
        for identity in study.requested_runs
        if (
            identity.run_id in successful
            and identity.contract_declared
            and identity.ci_scaling_active
        )
    }


def _object_family_groups(study: ReportInput, run_ids: object) -> dict[str, list[str]]:
    selected = set(run_ids)
    groups: dict[str, list[str]] = {}
    for identity in sorted(study.requested_runs, key=lambda item: item.run_id):
        if identity.run_id not in selected:
            continue
        assert identity.object_family is not None
        groups.setdefault(identity.object_family, []).append(identity.run_id)
    return groups


def _requested_object_families(study: ReportInput) -> list[str]:
    return sorted(
        {
            identity.object_family
            for identity in study.requested_runs
            if identity.object_family is not None
        }
    ) or ["study"]


def _typed_not_applicable_reason(
    study: ReportInput, object_family: str | None = None
) -> str:
    identities = [
        identity
        for identity in study.requested_runs
        if object_family is None or identity.object_family == object_family
    ]
    declared_ci = any(
        identity.contract_declared and identity.ci_scaling_active
        for identity in identities
    )
    eligible = _eligible_ci_run_ids(study)
    successful_ci = any(identity.run_id in eligible for identity in identities)
    if declared_ci and not successful_ci:
        return "no_successful_ci_evidence"
    if not declared_ci:
        return "legacy_normalized_amplitude"
    return "typed_metrics_missing"


def _typed_varpro_series(study: ReportInput) -> dict[str, dict[str, float]]:
    series: dict[str, dict[str, float]] = {}
    for row in study.rows:
        if not row.attempt.terminal_success:
            continue
        values = _metric_values(row)
        if (
            not {
                "measurement_consistency.varpro.s1",
                "measurement_consistency.varpro.s2",
            }
            <= values.keys()
        ):
            continue
        s1 = values["measurement_consistency.varpro.s1"]
        s2 = values["measurement_consistency.varpro.s2"]
        series[row.attempt.run_id] = {
            "s1": s1,
            "s2": s2,
            "c_A": float(np.hypot(s1, s2)),
            "c_phi": float(np.arctan2(s2, s1)),
        }
    return series


def _dashboard_absolute_series(study: ReportInput) -> dict[str, dict[str, float]]:
    required = {
        "truth_quality.amp_mean_ratio",
        "truth_quality.absolute_amp_nrmse",
    }
    series: dict[str, dict[str, float]] = {}
    for row in study.rows:
        if not row.attempt.terminal_success:
            continue
        values = _metric_values(row)
        if required <= values.keys():
            series[row.attempt.run_id] = {
                "amp_mean_ratio": values["truth_quality.amp_mean_ratio"],
                "absolute_amp_nrmse": values["truth_quality.absolute_amp_nrmse"],
            }
    return series


def _dashboard_physics_series(study: ReportInput) -> dict[str, dict[str, float]]:
    required = {
        "measurement_consistency.varpro.unit_objective",
        "measurement_consistency.varpro.fitted_objective",
        "stability.reload_max_abs_error",
    }
    series: dict[str, dict[str, float]] = {}
    for row in study.rows:
        if not row.attempt.terminal_success:
            continue
        values = _metric_values(row)
        if not required <= values.keys():
            continue
        unit = values["measurement_consistency.varpro.unit_objective"]
        if unit <= 0.0:
            continue
        series[row.attempt.run_id] = {
            "varpro_objective_ratio": (
                values["measurement_consistency.varpro.fitted_objective"] / unit
            ),
            "reload_max_abs_error": values["stability.reload_max_abs_error"],
        }
    return series


def _dashboard_series(study: ReportInput) -> dict[str, dict[str, float]]:
    absolute = _dashboard_absolute_series(study)
    physics = _dashboard_physics_series(study)
    return {
        run_id: {**absolute.get(run_id, {}), **physics.get(run_id, {})}
        for run_id in sorted(set(absolute) | set(physics))
    }


def _render_grid_entry(
    identity: RunIdentity,
    row: ReportRow | None,
    row_axes: object,
    panels: list[dict[str, object]],
    *,
    gauge_normalized: bool = False,
) -> None:
    if row is None or not row.attempt.terminal_success:
        status = "missing" if row is None else row.attempt.status.value
        detail = (
            "requested run has no row"
            if row is None
            else f"{row.failure_stage or 'failure'}: {row.failure_error or ''}"
        )
        for axis, label in zip(
            row_axes,
            ("Reconstruction", "Target/reference", "Absolute error"),
            strict=True,
        ):
            axis.axis("off")
            axis.text(
                0.5,
                0.55,
                status.upper(),
                ha="center",
                va="center",
                color="darkred",
                transform=axis.transAxes,
            )
            axis.text(
                0.5,
                0.28,
                detail,
                ha="center",
                va="center",
                fontsize=8,
                wrap=True,
                transform=axis.transAxes,
            )
            axis.set_title(label)
            panels.append(
                {
                    "run_id": identity.run_id,
                    "object_family": identity.object_family,
                    "panel": label.lower().replace("/", "_"),
                    "quantity_kind": "missing_or_failed",
                    "vmin": None,
                    "vmax": None,
                }
            )
        return

    reconstruction, target, error = _grid_display_arrays(
        row, gauge_normalized=gauge_normalized
    )
    shared = (
        _limits(reconstruction, target)
        if target is not None
        else _limits(reconstruction)
    )
    error_limits = _limits(error) if error is not None else (0.0, 1.0)
    images = (
        ("reconstruction", reconstruction, shared, "Reconstruction"),
        (
            "target",
            target,
            shared,
            "Truth" if row.truth_role == "object_truth" else "Reference",
        ),
        ("error", error, error_limits, "Absolute error"),
    )
    for axis, (panel, array, limits, label) in zip(row_axes, images, strict=True):
        axis.set_title(label)
        if array is None:
            axis.axis("off")
            axis.text(
                0.5,
                0.5,
                "Not applicable",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        else:
            axis.imshow(
                array,
                cmap="viridis" if panel != "error" else "magma",
                vmin=limits[0],
                vmax=limits[1],
            )
            axis.set_xticks([])
            axis.set_yticks([])
        panels.append(
            {
                "run_id": identity.run_id,
                "object_family": identity.object_family,
                "panel": panel,
                "quantity_kind": "absolute_quantity"
                if not gauge_normalized and (panel != "target" or row.truth_role == "object_truth")
                else (
                    "gauge_normalized_structure"
                    if gauge_normalized
                    else "reference_agreement_not_truth"
                ),
                "vmin": limits[0],
                "vmax": limits[1],
            }
        )


def _grid_display_arrays(
    row: ReportRow, *, gauge_normalized: bool
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Return consistently masked/cropped arrays for one report grid row."""
    assert row.reconstruction is not None
    reconstruction = np.abs(row.reconstruction)
    target = None if row.target is None else np.abs(row.target)
    mask = row.common_valid_mask
    if gauge_normalized and target is not None and mask is not None:
        rows, cols = np.where(mask)
        row_slice = slice(int(rows.min()), int(rows.max()) + 1)
        col_slice = slice(int(cols.min()), int(cols.max()) + 1)
        reconstruction_mean = float(np.mean(reconstruction[mask]))
        target_mean = float(np.mean(target[mask]))
        if reconstruction_mean > 0.0 and target_mean > 0.0:
            reconstruction = reconstruction * (target_mean / reconstruction_mean)
        reconstruction = np.where(mask, reconstruction, 0.0)[row_slice, col_slice]
        target = np.where(mask, target, 0.0)[row_slice, col_slice]
    elif gauge_normalized and target is not None:
        reconstruction_mean = float(np.mean(reconstruction))
        target_mean = float(np.mean(target))
        if reconstruction_mean > 0.0 and target_mean > 0.0:
            reconstruction = reconstruction * (target_mean / reconstruction_mean)
    error = (
        None if target is None else np.abs(reconstruction - target)
    ) if gauge_normalized else (
        np.abs(row.error)
        if row.error is not None
        else (None if target is None else np.abs(reconstruction - target))
    )
    return reconstruction, target, error


def render_grid(
    study: ReportInput, path: Path, *, gauge_normalized: bool = False
) -> tuple[list[dict[str, object]], list[str]]:
    entries = _plot_rows(study)
    grouped: dict[str, list[tuple[RunIdentity, ReportRow | None]]] = {}
    for identity, row in entries:
        assert identity.object_family is not None
        grouped.setdefault(identity.object_family, []).append((identity, row))
    panels: list[dict[str, object]] = []
    run_ids = [identity.run_id for identity, _ in entries]

    if len(grouped) <= 1:
        figure, axes = plt.subplots(
            max(1, len(entries)),
            3,
            figsize=(9, max(2.5, 2.8 * len(entries))),
            squeeze=False,
        )
        figure.subplots_adjust(left=0.31, right=0.98, hspace=0.55)
        if not entries:
            for axis in axes[0]:
                _visible_not_applicable(axis, "no_requested_runs")
        for row_axes, (identity, row) in zip(axes, entries, strict=False):
            _render_grid_entry(
                identity,
                row,
                row_axes,
                panels,
                gauge_normalized=gauge_normalized,
            )
            position = row_axes[0].get_position()
            figure.text(
                0.015,
                (position.y0 + position.y1) / 2,
                _wrapped_grid_label(identity),
                ha="left",
                va="center",
                fontsize=8,
                rotation=0,
            )
    else:
        slots_per_band = 3
        family_bands = {
            family: (len(family_entries) + slots_per_band - 1) // slots_per_band
            for family, family_entries in grouped.items()
        }
        total_bands = sum(family_bands.values())
        figure, axes = plt.subplots(
            total_bands,
            slots_per_band * 3,
            figsize=(18, max(4.0, 2.2 * total_bands)),
            squeeze=False,
        )
        figure.subplots_adjust(
            left=0.065, right=0.99, top=0.94, bottom=0.025, hspace=0.9, wspace=0.28
        )
        band_offset = 0
        for family, family_entries in grouped.items():
            for index, (identity, row) in enumerate(family_entries):
                band = band_offset + index // slots_per_band
                slot = index % slots_per_band
                row_axes = axes[band, slot * 3 : slot * 3 + 3]
                _render_grid_entry(
                    identity,
                    row,
                    row_axes,
                    panels,
                    gauge_normalized=gauge_normalized,
                )
                left = row_axes[0].get_position().x0
                right = row_axes[-1].get_position().x1
                top = max(axis.get_position().y1 for axis in row_axes)
                figure.text(
                    (left + right) / 2,
                    top + 0.025,
                    _family_compact_grid_label(identity),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
            used_slots = len(family_entries) % slots_per_band
            if used_slots:
                last_band = band_offset + family_bands[family] - 1
                for axis in axes[last_band, used_slots * 3 :]:
                    axis.axis("off")
            first_axis = axes[band_offset, 0]
            last_axis = axes[band_offset + family_bands[family] - 1, 0]
            figure.text(
                0.018,
                (first_axis.get_position().y1 + last_axis.get_position().y0) / 2,
                _prettify_grid_token(family),
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                rotation=90,
            )
            band_offset += family_bands[family]

    figure.savefig(path, dpi=120)
    plt.close(figure)
    return panels, run_ids


def _curve_panel_run_ids(study: ReportInput) -> dict[str, dict[str, list[str]]]:
    family_by_run = {
        identity.run_id: identity.object_family for identity in study.requested_runs
    }
    row_families = {
        family_by_run.get(row.attempt.run_id, row.attempt.dataset_id)
        for row in study.rows
    }
    requested_families = (
        _requested_object_families(study) if study.requested_runs else []
    )
    families = sorted(set(requested_families) | row_families) or ["study"]
    contributors = {
        family: {"training_loss": [], "gradient_norm": []} for family in families
    }
    for row in sorted(study.rows, key=lambda item: item.attempt.run_id):
        family = family_by_run.get(row.attempt.run_id, row.attempt.dataset_id)
        if row.training_loss:
            contributors[family]["training_loss"].append(row.attempt.run_id)
        if row.gradient_norm:
            contributors[family]["gradient_norm"].append(row.attempt.run_id)
    return contributors


def render_curves(study: ReportInput, path: Path) -> list[str]:
    family_by_run = {
        identity.run_id: identity.object_family for identity in study.requested_runs
    }
    identity_by_run = {identity.run_id: identity for identity in study.requested_runs}
    contributors = _curve_panel_run_ids(study)
    figure, axes = plt.subplots(
        len(contributors),
        2,
        figsize=(10, 4.5 * len(contributors)),
        squeeze=False,
    )
    sorted_rows = sorted(study.rows, key=lambda item: item.attempt.run_id)
    for family, row_axes in zip(contributors, axes, strict=True):
        for row in sorted_rows:
            if family_by_run.get(row.attempt.run_id, row.attempt.dataset_id) != family:
                continue
            identity = identity_by_run.get(row.attempt.run_id)
            label = (
                row.attempt.run_id
                if identity is None
                else _family_compact_grid_label(identity)
            )
            if row.training_loss:
                row_axes[0].plot(
                    row.training_loss,
                    label=label,
                    marker="o" if len(row.training_loss) == 1 else None,
                )
            if row.gradient_norm:
                row_axes[1].plot(
                    row.gradient_norm,
                    label=label,
                    marker="o" if len(row.gradient_norm) == 1 else None,
                )
        family_label = _prettify_grid_token(family)
        row_axes[0].set_title(f"{family_label}: Training loss")
        row_axes[0].set_xlabel("Epoch")
        row_axes[0].set_ylabel("Normalized optimization loss")
        row_axes[1].set_title(f"{family_label}: Gradient norm")
        row_axes[1].set_xlabel("Epoch")
        row_axes[1].set_ylabel("Absolute gradient norm")
        for axis in row_axes:
            if axis.lines:
                line_count = len(axis.lines)
                axis.legend(
                    fontsize=6,
                    ncol=min(3, max(1, (line_count + 4) // 5)),
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.24),
                )
            else:
                _visible_not_applicable(axis, "metric_not_logged")
    figure.tight_layout(h_pad=4.0)
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return sorted(
        {
            run_id
            for panel_run_ids in contributors.values()
            for run_ids in panel_run_ids.values()
            for run_id in run_ids
        }
    )


def render_seed_distribution(study: ReportInput, path: Path) -> list[str]:
    figure, axis = plt.subplots(figsize=(5.5, 3.5))
    plotted = [
        (row.attempt.run_id, float(record.value))
        for row in study.rows
        if row.attempt.terminal_success
        for record in row.metric_records
        if record.path.endswith("amp_pearson") and isinstance(record.value, float)
    ]
    values = [value for _, value in plotted]
    if values:
        axis.scatter(range(1, len(values) + 1), values)
    else:
        axis.text(
            0.5,
            0.5,
            "No successful seed metrics",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set_title("Seed distribution")
    axis.set_xlabel("Successful seed order")
    axis.set_ylabel("Mean-normalized/recognizability metric")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return sorted({run_id for run_id, _ in plotted})


def render_varpro(
    study: ReportInput, path: Path
) -> tuple[list[str], list[dict[str, str]], list[dict[str, object]]]:
    series = _typed_varpro_series(study)
    groups = _object_family_groups(study, series)
    families = _requested_object_families(study)
    visual_roles = _typed_visual_role_registry(study)
    style_colors, seed_markers = _typed_figure_styles(study, visual_roles)
    figure, axes = plt.subplots(
        len(families),
        3,
        figsize=(12, 3.7 * len(families)),
        squeeze=False,
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.0, 1.0, 0.62)},
    )
    panels: list[dict[str, str]] = []
    scatter_specs: list[_ScatterPanelSpec] = []
    for row_index, family in enumerate(families):
        reason = _typed_not_applicable_reason(study, family)
        coefficient_axis, derived_axis, legend_axis = axes[row_index]
        legend_axis.axis("off")
        coefficient_spec = _ScatterPanelSpec(
            coefficient_axis,
            family,
            "s1_s2",
            _ordered_panel_points(study, family, series, "s1", "s2"),
        )
        derived_spec = _ScatterPanelSpec(
            derived_axis,
            family,
            "c_A_c_phi",
            _ordered_panel_points(study, family, series, "c_A", "c_phi"),
        )
        scatter_specs.extend((coefficient_spec, derived_spec))
        _set_exact_panel_limits(coefficient_spec)
        _set_exact_panel_limits(derived_spec)
        family_label = _prettify_grid_token(family)
        coefficient_axis.set_title(f"{family_label}: typed VarPro coefficients")
        coefficient_axis.set_xlabel("s1 (physical count basis)")
        coefficient_axis.set_ylabel("s2 (physical count basis)")
        derived_axis.set_title(f"{family_label}: derived manuscript scale")
        derived_axis.set_xlabel("c_A = sqrt(s1^2 + s2^2)")
        derived_axis.set_ylabel("c_phi = atan2(s2, s1) [rad]")
        for spec in (coefficient_spec, derived_spec):
            if not spec.points:
                _visible_not_applicable(spec.axis, reason)
        arm_ids = _ordered_arm_ids(study, family, groups.get(family, []))
        if arm_ids:
            _add_legend(
                legend_axis,
                _arm_legend_handles(arm_ids, style_colors, visual_roles),
                title="CI arms",
                anchor=(0.0, 1.0),
            )
            _add_legend(
                legend_axis,
                _seed_legend_handles(seed_markers),
                title="Seeds",
                anchor=(0.0, 0.0),
                location="lower left",
            )
        family_panels = (
            {"object_family": family, "panel": "s1_s2"},
            {"object_family": family, "panel": "c_A_c_phi"},
        )
        if not groups.get(family):
            for panel in family_panels:
                panel["not_applicable_reason"] = reason
        panels.extend(family_panels)
    scatter_layout = _render_scatter_panels(
        figure,
        scatter_specs,
        style_colors,
        seed_markers,
        visual_roles,
    )
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return sorted(series), panels, [record.to_payload() for record in scatter_layout]


def render_dashboard(
    study: ReportInput, path: Path
) -> tuple[
    list[str],
    list[dict[str, str]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    absolute_series = _dashboard_absolute_series(study)
    physics_series = _dashboard_physics_series(study)
    series = _dashboard_series(study)
    absolute_groups = _object_family_groups(study, absolute_series)
    physics_groups = _object_family_groups(study, physics_series)
    families = _requested_object_families(study)
    visual_roles = _typed_visual_role_registry(study)
    style_colors, seed_markers = _typed_figure_styles(study, visual_roles)
    figure, axes = plt.subplots(
        len(families),
        3,
        figsize=(12, 4.2 * len(families)),
        squeeze=False,
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.0, 1.0, 0.68)},
    )
    panels: list[dict[str, str]] = []
    scatter_specs: list[_ScatterPanelSpec] = []
    physics_axes: dict[str, object] = {}
    for row_index, family in enumerate(families):
        reason = _typed_not_applicable_reason(study, family)
        scale_axis, physics_axis, legend_axis = axes[row_index]
        physics_axes[family] = physics_axis
        legend_axis.axis("off")
        absolute_spec = _ScatterPanelSpec(
            scale_axis,
            family,
            "absolute_scale",
            _ordered_panel_points(
                study,
                family,
                absolute_series,
                "amp_mean_ratio",
                "absolute_amp_nrmse",
            ),
        )
        physics_spec = _ScatterPanelSpec(
            physics_axis,
            family,
            "physics_reload",
            _ordered_panel_points(
                study,
                family,
                physics_series,
                "varpro_objective_ratio",
                "reload_max_abs_error",
            ),
        )
        scatter_specs.extend((absolute_spec, physics_spec))
        scale_axis.axvline(1.0, color="black", linestyle="--", linewidth=1)
        _set_exact_panel_limits(absolute_spec)
        _set_exact_panel_limits(physics_spec)
        family_label = _prettify_grid_token(family)
        scale_axis.set_title(f"{family_label}: absolute amplitude scale")
        scale_axis.set_xlabel("Amplitude mean ratio (target = 1)")
        scale_axis.set_ylabel("Absolute amplitude NRMSE")
        physics_axis.set_title(f"{family_label}: physics and reload stability")
        physics_axis.set_xlabel("VarPro fitted / unit objective")
        physics_axis.set_ylabel("Checkpoint reload max abs error")
        if not absolute_spec.points:
            _visible_not_applicable(scale_axis, "no_object_truth")
        if not physics_spec.points:
            _visible_not_applicable(physics_axis, reason)
        absolute_arm_ids = _ordered_arm_ids(
            study, family, absolute_groups.get(family, [])
        )
        physics_arm_ids = _ordered_arm_ids(
            study, family, physics_groups.get(family, [])
        )
        if absolute_arm_ids:
            _add_legend(
                legend_axis,
                _arm_legend_handles(absolute_arm_ids, style_colors, visual_roles),
                title="Absolute scale arms",
                anchor=(0.0, 1.0),
            )
        if physics_arm_ids:
            _add_legend(
                legend_axis,
                _arm_legend_handles(physics_arm_ids, style_colors, visual_roles),
                title="Physics/reload arms",
                anchor=(0.0, 0.43),
            )
        if absolute_arm_ids or physics_arm_ids:
            _add_legend(
                legend_axis,
                _seed_legend_handles(seed_markers),
                title="Seeds",
                anchor=(0.0, 0.0),
                location="lower left",
            )
        family_panels = (
            {"object_family": family, "panel": "absolute_scale"},
            {"object_family": family, "panel": "physics_reload"},
        )
        if not absolute_groups.get(family):
            family_panels[0]["not_applicable_reason"] = "no_object_truth"
        if not physics_groups.get(family):
            family_panels[1]["not_applicable_reason"] = reason
        panels.extend(family_panels)
    scatter_layout = _render_scatter_panels(
        figure,
        scatter_specs,
        style_colors,
        seed_markers,
        visual_roles,
    )
    zero_annotations = build_zero_annotation_records(scatter_layout)
    for record in zero_annotations:
        axis = physics_axes[record.scatter.object_family]
        marker = next(
            collection
            for collection in axis.collections
            if collection.get_gid() == record.scatter.marker_artist_id
        )
        annotation = axis.annotate(
            "0",
            record.scatter.exact_anchor,
            xycoords=marker.get_offset_transform(),
            xytext=record.annotation_display_offset_points,
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=style_colors[record.scatter.visual_style_id],
            annotation_clip=True,
            arrowprops={
                "arrowstyle": "-",
                "color": style_colors[record.scatter.visual_style_id],
                "linewidth": 0.75,
                "shrinkA": 2,
                "shrinkB": 5,
            },
        )
        annotation.set_gid(record.annotation_artist_id)
        annotation.set_in_layout(False)
        assert annotation.arrow_patch is not None
        annotation.arrow_patch.set_gid(record.annotation_connector_id)
        annotation.arrow_patch.set_in_layout(False)
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return (
        sorted(series),
        panels,
        [record.to_payload() for record in scatter_layout],
        [record.to_payload() for record in zero_annotations],
    )


def render_all_figures(
    study: ReportInput, root: Path
) -> tuple[dict[str, object], dict[str, object]]:
    """Render every report figure; return plot metadata and run-id mappings."""
    metadata: dict[str, object] = {
        "renderer_layout_schema_version": REPORT_RENDERER_LAYOUT_SCHEMA_VERSION
    }
    grid_panels, grid_runs = render_grid(
        study, root / "reconstruction_truth_error_grid.png"
    )
    metadata["reconstruction_truth_error_grid.png"] = {
        "panels": grid_panels,
        "view": "absolute_scale",
    }
    structural_panels, structural_runs = render_grid(
        study,
        root / "structural_quality_grid.png",
        gauge_normalized=True,
    )
    metadata["structural_quality_grid.png"] = {
        "panels": structural_panels,
        "view": "gauge_normalized_structure",
    }
    mappings: dict[str, object] = {
        "renderer_layout_schema_version": REPORT_RENDERER_LAYOUT_SCHEMA_VERSION,
        "reconstruction_truth_error_grid.png": grid_runs,
        "structural_quality_grid.png": structural_runs,
    }
    renderers = (
        ("training_gradient_curves.png", render_curves),
        ("seed_distribution.png", render_seed_distribution),
    )
    for filename, renderer in renderers:
        mapped = renderer(study, root / filename)
        mappings[filename] = mapped
        metadata[filename] = {"panels": [], "run_ids": mapped}
    curve_panel_run_ids = _curve_panel_run_ids(study)
    curve_groups = {
        family: sorted(
            {run_id for run_ids in panel_run_ids.values() for run_id in run_ids}
        )
        for family, panel_run_ids in curve_panel_run_ids.items()
        if any(panel_run_ids.values())
    }
    curve_panels = []
    for family, panel_run_ids in curve_panel_run_ids.items():
        for panel, run_ids in panel_run_ids.items():
            panel_metadata = {
                "object_family": family,
                "panel": panel,
                "run_ids": run_ids,
            }
            if not run_ids:
                panel_metadata["not_applicable_reason"] = "metric_not_logged"
            curve_panels.append(panel_metadata)
    metadata["training_gradient_curves.png"].update(
        {
            "object_family_groups": curve_groups,
            "panels": curve_panels,
        }
    )
    varpro_ids, varpro_panels, varpro_layout = render_varpro(
        study, root / "varpro_scale.png"
    )
    mappings["varpro_scale.png"] = varpro_ids
    metadata["varpro_scale.png"] = {
        "panels": varpro_panels,
        "run_ids": varpro_ids,
        "scatter_layout": varpro_layout,
    }
    dashboard_ids, dashboard_panels, dashboard_layout, zero_annotations = (
        render_dashboard(study, root / "absolute_scale_stability_dashboard.png")
    )
    mappings["absolute_scale_stability_dashboard.png"] = dashboard_ids
    metadata["absolute_scale_stability_dashboard.png"] = {
        "panels": dashboard_panels,
        "run_ids": dashboard_ids,
        "scatter_layout": dashboard_layout,
        "zero_annotations": zero_annotations,
    }
    eligible = _eligible_ci_run_ids(study)
    varpro_series = _typed_varpro_series(study)
    dashboard_series = _dashboard_series(study)
    dashboard_absolute = _dashboard_absolute_series(study)
    dashboard_physics = _dashboard_physics_series(study)
    metadata["varpro_scale.png"]["series"] = varpro_series
    metadata["absolute_scale_stability_dashboard.png"]["series"] = dashboard_series
    metadata["absolute_scale_stability_dashboard.png"]["panel_run_ids"] = {
        "absolute_truth": sorted(dashboard_absolute),
        "physics_reload": sorted(dashboard_physics),
    }
    metadata["varpro_scale.png"]["eligible_run_ids"] = sorted(eligible)
    metadata["absolute_scale_stability_dashboard.png"]["eligible_run_ids"] = sorted(
        eligible
    )
    metadata["varpro_scale.png"]["object_family_groups"] = _object_family_groups(
        study, varpro_series
    )
    metadata["absolute_scale_stability_dashboard.png"]["object_family_groups"] = (
        _object_family_groups(study, dashboard_series)
    )
    family_reasons = {
        family: _typed_not_applicable_reason(study, family)
        for family in _requested_object_families(study)
        if family not in _object_family_groups(study, varpro_series)
    }
    metadata["varpro_scale.png"]["not_applicable_reasons_by_object_family"] = (
        family_reasons
    )
    metadata["absolute_scale_stability_dashboard.png"][
        "not_applicable_reasons_by_object_family"
    ] = {
        family: _typed_not_applicable_reason(study, family)
        for family in _requested_object_families(study)
        if family not in _object_family_groups(study, dashboard_series)
    }
    truth_eligible = {
        identity.run_id
        for identity in study.requested_runs
        if identity.run_id in eligible and identity.truth_role == "object_truth"
    }
    required_series = (
        ("varpro_scale.png", eligible, varpro_series),
        (
            "absolute_scale_stability_dashboard.png",
            truth_eligible,
            dashboard_absolute,
        ),
        (
            "absolute_scale_stability_dashboard.png",
            eligible,
            dashboard_physics,
        ),
    )
    for filename, expected, series in required_series:
        missing = expected - series.keys()
        if missing:
            from .reporting import ReportingError

            raise ReportingError(
                f"eligible run ids have zero plotted marks in {filename}: {sorted(missing)!r}"
            )
    for filename, series in (
        ("varpro_scale.png", varpro_series),
        ("absolute_scale_stability_dashboard.png", dashboard_series),
    ):
        if not series:
            metadata[filename].update(
                {
                    "visible_label": "Not applicable",
                    "not_applicable_reason": (_typed_not_applicable_reason(study)),
                }
            )
    return metadata, mappings
