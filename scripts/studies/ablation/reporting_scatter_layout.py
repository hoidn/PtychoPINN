"""Versioned, deterministic layout contracts for report scatter panels."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import re
from typing import Iterable


REPORT_RENDERER_LAYOUT_SCHEMA_VERSION = "ablation_report_renderer_layout_v2"


class ScatterLayoutError(ValueError):
    """Raised when bounded display-space placement cannot be satisfied."""


@dataclass(frozen=True)
class PanelBounds:
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class ScatterLayoutPoint:
    run_id: str
    arm_id: str
    visual_role_id: str
    visual_style_id: str
    arm_display_label: str
    object_family: str
    panel: str
    seed: int
    exact_anchor: tuple[float, float]
    display_anchor_points: tuple[float, float]
    marker_artist_id: str


@dataclass(frozen=True)
class ScatterLayoutRecord:
    renderer_layout_schema_version: str
    run_id: str
    arm_id: str
    visual_role_id: str
    visual_style_id: str
    arm_display_label: str
    object_family: str
    panel: str
    seed: int
    exact_anchor: tuple[float, float]
    marker_display_offset_points: tuple[float, float]
    display_marker_center_points: tuple[float, float]
    marker_artist_id: str
    connector_id: str | None

    def to_payload(self) -> dict[str, object]:
        return {
            "renderer_layout_schema_version": self.renderer_layout_schema_version,
            "run_id": self.run_id,
            "arm_id": self.arm_id,
            "visual_role_id": self.visual_role_id,
            "visual_style_id": self.visual_style_id,
            "arm_display_label": self.arm_display_label,
            "object_family": self.object_family,
            "panel": self.panel,
            "seed": self.seed,
            "exact_anchor": {"x": self.exact_anchor[0], "y": self.exact_anchor[1]},
            "marker_display_offset_points": {
                "x": self.marker_display_offset_points[0],
                "y": self.marker_display_offset_points[1],
            },
            "marker_artist_id": self.marker_artist_id,
            "connector_id": self.connector_id,
        }


@dataclass(frozen=True)
class ZeroAnnotationRecord:
    scatter: ScatterLayoutRecord
    annotation_slot: int
    annotation_display_offset_points: tuple[float, float]
    annotation_artist_id: str
    annotation_connector_id: str

    def to_payload(self) -> dict[str, object]:
        payload = self.scatter.to_payload()
        payload.update(
            {
                "metric": "reload_max_abs_error",
                "label": "0",
                "annotation_slot": self.annotation_slot,
                "annotation_display_offset_points": {
                    "x": self.annotation_display_offset_points[0],
                    "y": self.annotation_display_offset_points[1],
                },
                "annotation_artist_id": self.annotation_artist_id,
                "connectors": {
                    "anchor_to_marker": self.scatter.connector_id,
                    "marker_to_annotation": self.annotation_connector_id,
                },
            }
        )
        return payload


def build_zero_annotation_records(
    scatter_records: Iterable[ScatterLayoutRecord],
) -> tuple[ZeroAnnotationRecord, ...]:
    """Extend zero-valued physics layouts without recomputing marker identity."""
    horizontal_offsets = (-30.0, -18.0, -8.0, 8.0, 18.0, 30.0)
    by_family: dict[str, list[ScatterLayoutRecord]] = defaultdict(list)
    for record in scatter_records:
        if record.panel == "physics_reload" and record.exact_anchor[1] == 0.0:
            by_family[record.object_family].append(record)
    output: list[ZeroAnnotationRecord] = []
    for records in by_family.values():
        for slot, record in enumerate(records):
            output.append(
                ZeroAnnotationRecord(
                    scatter=record,
                    annotation_slot=slot,
                    annotation_display_offset_points=(
                        horizontal_offsets[slot % len(horizontal_offsets)],
                        12.0 + 14.0 * slot,
                    ),
                    annotation_artist_id=f"dashboard-zero-label:{record.run_id}",
                    annotation_connector_id=(f"dashboard-marker-zero:{record.run_id}"),
                )
            )
    return tuple(output)


def _boxes_overlap(
    first: tuple[float, float],
    second: tuple[float, float],
    half_size: float,
) -> bool:
    return (
        abs(first[0] - second[0]) < 2.0 * half_size
        and abs(first[1] - second[1]) < 2.0 * half_size
    )


def _collision_components(
    points: tuple[ScatterLayoutPoint, ...], half_size: float
) -> list[list[int]]:
    neighbors = {index: set() for index in range(len(points))}
    for first in range(len(points)):
        for second in range(first + 1, len(points)):
            if _boxes_overlap(
                points[first].display_anchor_points,
                points[second].display_anchor_points,
                half_size,
            ):
                neighbors[first].add(second)
                neighbors[second].add(first)
    components: list[list[int]] = []
    unseen = set(range(len(points)))
    while unseen:
        start = min(unseen)
        stack = [start]
        component: list[int] = []
        unseen.remove(start)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(neighbors[current], reverse=True):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def _candidate_offsets(step: float = 12.0) -> Iterable[tuple[float, float]]:
    for radius in range(1, 16):
        cells = [
            (x, y)
            for y in range(-radius, radius + 1)
            for x in range(-radius, radius + 1)
            if max(abs(x), abs(y)) == radius
        ]
        for x, y in sorted(cells, key=lambda item: (item[1], item[0])):
            yield x * step, y * step


def layout_scatter_points(
    points: Iterable[ScatterLayoutPoint],
    panel_bounds: PanelBounds,
    *,
    marker_half_size_points: float = 5.0,
) -> tuple[ScatterLayoutRecord, ...]:
    """Lay out colliding markers in display points while preserving data anchors."""
    ordered = tuple(points)
    if marker_half_size_points <= 0.0:
        raise ScatterLayoutError("marker_half_size_points must be positive")
    canonical_indices = sorted(
        range(len(ordered)),
        key=lambda index: (
            ordered[index].object_family,
            ordered[index].panel,
            ordered[index].visual_role_id,
            ordered[index].arm_id,
            ordered[index].seed,
            ordered[index].run_id,
        ),
    )
    canonical = tuple(ordered[index] for index in canonical_indices)
    components = _collision_components(canonical, marker_half_size_points)
    offsets: dict[int, tuple[float, float]] = {}
    placed_centers: list[tuple[float, float]] = []

    def within_bounds(center: tuple[float, float]) -> bool:
        return (
            center[0] - marker_half_size_points >= panel_bounds.x0
            and center[0] + marker_half_size_points <= panel_bounds.x1
            and center[1] - marker_half_size_points >= panel_bounds.y0
            and center[1] + marker_half_size_points <= panel_bounds.y1
        )

    singleton_indices = [
        component[0] for component in components if len(component) == 1
    ]
    collision_indices = [
        index for component in components if len(component) > 1 for index in component
    ]
    for canonical_index in singleton_indices:
        original_index = canonical_indices[canonical_index]
        center = canonical[canonical_index].display_anchor_points
        if not within_bounds(center):
            raise ScatterLayoutError("noncolliding marker lies outside panel bounds")
        offsets[original_index] = (0.0, 0.0)
        placed_centers.append(center)

    candidates = tuple(_candidate_offsets())
    domains: dict[int, tuple[tuple[tuple[float, float], tuple[float, float]], ...]] = {}
    for canonical_index in collision_indices:
        anchor = canonical[canonical_index].display_anchor_points
        domains[canonical_index] = tuple(
            (offset, (anchor[0] + offset[0], anchor[1] + offset[1]))
            for offset in candidates
            if within_bounds((anchor[0] + offset[0], anchor[1] + offset[1]))
            and not any(
                _boxes_overlap(
                    (anchor[0] + offset[0], anchor[1] + offset[1]),
                    singleton,
                    marker_half_size_points,
                )
                for singleton in placed_centers
            )
        )

    assignments: dict[int, tuple[tuple[float, float], tuple[float, float]]] = {}

    def available(
        canonical_index: int,
    ) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
        occupied = [center for _, center in assignments.values()]
        return tuple(
            candidate
            for candidate in domains[canonical_index]
            if not any(
                _boxes_overlap(candidate[1], center, marker_half_size_points)
                for center in occupied
            )
        )

    def search() -> bool:
        if len(assignments) == len(collision_indices):
            return True
        choices = [
            (available(index), index)
            for index in collision_indices
            if index not in assignments
        ]
        domain, canonical_index = min(choices, key=lambda item: (len(item[0]), item[1]))
        for candidate in domain:
            assignments[canonical_index] = candidate
            if search():
                return True
            del assignments[canonical_index]
        return False

    if not search():
        raise ScatterLayoutError(
            "no collision-free scatter layout exists within panel bounds "
            "and finite candidate offsets"
        )

    for canonical_index, (offset, _) in assignments.items():
        offsets[canonical_indices[canonical_index]] = offset

    records = []
    for index, point in enumerate(ordered):
        offset = offsets[index]
        center = (
            point.display_anchor_points[0] + offset[0],
            point.display_anchor_points[1] + offset[1],
        )
        records.append(
            ScatterLayoutRecord(
                renderer_layout_schema_version=REPORT_RENDERER_LAYOUT_SCHEMA_VERSION,
                run_id=point.run_id,
                arm_id=point.arm_id,
                visual_role_id=point.visual_role_id,
                visual_style_id=point.visual_style_id,
                arm_display_label=point.arm_display_label,
                object_family=point.object_family,
                panel=point.panel,
                seed=point.seed,
                exact_anchor=point.exact_anchor,
                marker_display_offset_points=offset,
                display_marker_center_points=center,
                marker_artist_id=point.marker_artist_id,
                connector_id=(
                    None
                    if offset == (0.0, 0.0)
                    else f"scatter-anchor-marker:{point.panel}:{point.run_id}"
                ),
            )
        )
    return tuple(records)


@dataclass(frozen=True)
class ArmPresentation:
    """Stable arm identity plus its concise, non-unique base display label."""

    arm_id: str
    object_family: str
    base_display_label: str


@dataclass(frozen=True)
class ArmVisualRole:
    """Global visual identity resolved for one exact selected arm."""

    arm_id: str
    visual_role_id: str
    visual_style_id: str
    display_label: str


def _readable_token(value: str) -> str:
    return " ".join(word.title() for word in re.sub(r"[_-]+", " ", value).split())


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode()).hexdigest()[:12]
    return f"{prefix}:{digest}"


def _family_variant_suffixes(
    group: list[ArmPresentation],
) -> dict[str, str]:
    prefixes = {
        arm.arm_id: tuple(part for part in arm.arm_id.split("--")[:-2] if part)
        for arm in group
    }
    common = set.intersection(*(set(parts) for parts in prefixes.values()))
    suffixes: dict[str, str] = {}
    for arm in group:
        candidates = [part for part in prefixes[arm.arm_id] if part not in common]
        if candidates:
            suffixes[arm.arm_id] = _readable_token(candidates[-1])
    if len(suffixes) != len(group) or len(set(suffixes.values())) != len(group):
        return {
            arm.arm_id: hashlib.sha256(arm.arm_id.encode()).hexdigest()[:8]
            for arm in group
        }
    return suffixes


def resolve_arm_visual_roles(
    arms: Iterable[ArmPresentation],
) -> dict[str, ArmVisualRole]:
    """Resolve dataset-independent roles without merging same-family variants."""
    by_arm: dict[str, ArmPresentation] = {}
    for arm in arms:
        existing = by_arm.setdefault(arm.arm_id, arm)
        if existing != arm:
            raise ScatterLayoutError(
                f"arm {arm.arm_id!r} has conflicting visual presentations"
            )
    grouped: dict[str, list[ArmPresentation]] = defaultdict(list)
    for arm in by_arm.values():
        grouped[arm.base_display_label].append(arm)
    registry: dict[str, ArmVisualRole] = {}
    for base_label, base_group in sorted(grouped.items()):
        by_family: dict[str, list[ArmPresentation]] = defaultdict(list)
        for arm in base_group:
            by_family[arm.object_family].append(arm)
        if max(len(group) for group in by_family.values()) == 1:
            role_id = _stable_id("visual-role", base_label)
            style_id = _stable_id("visual-style", role_id)
            for arm in base_group:
                registry[arm.arm_id] = ArmVisualRole(
                    arm.arm_id, role_id, style_id, base_label
                )
            continue
        suffixes: dict[str, str] = {}
        known_readable: set[str] = set()
        for family_group in by_family.values():
            if len(family_group) > 1:
                family_suffixes = _family_variant_suffixes(family_group)
                suffixes.update(family_suffixes)
                known_readable.update(family_suffixes.values())
        for arm in base_group:
            if arm.arm_id in suffixes:
                continue
            tokens = {_readable_token(part) for part in arm.arm_id.split("--")[:-2]}
            matches = sorted(tokens & known_readable)
            suffixes[arm.arm_id] = (
                matches[0]
                if len(matches) == 1
                else hashlib.sha256(arm.arm_id.encode()).hexdigest()[:8]
            )
        for arm in base_group:
            suffix = suffixes[arm.arm_id]
            role_id = _stable_id("visual-role", base_label, suffix)
            registry[arm.arm_id] = ArmVisualRole(
                arm.arm_id,
                role_id,
                _stable_id("visual-style", role_id),
                f"{base_label} [{suffix}]",
            )
    return registry
