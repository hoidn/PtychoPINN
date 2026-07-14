"""Closed schema for the human visual review sidecar (``visual_review.json``).

The driver writes a pending template and never self-approves it; a completed
review is human-authored and validated by :func:`parse_review`. Public names
are re-exported by :mod:`scripts.studies.ablation.verdicts`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType


class ReviewError(ValueError):
    """Raised when a completed visual review violates its closed schema."""


class ReviewDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"


REVIEW_SCHEMA_VERSION = "visual_review_v1"
PENDING_REVIEW_SCHEMA_VERSION = "visual_review_pending_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UTC_RFC3339_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z$"
)
_FAMILY_REVIEW_FIELDS = frozenset(
    {"schema_version", "reviewer", "timestamp", "figure_sha256", "families"}
)
_FAMILY_FIELDS = frozenset(
    {
        "decision",
        "recognizable",
        "flat",
        "checkerboard",
        "mirrored",
        "saturation",
        "collapse",
        "notes",
    }
)
_LEGACY_REVIEW_FIELDS = frozenset(
    {"schema_version", "reviewer", "timestamp", "figure_sha256", *_FAMILY_FIELDS}
)
_OBJECT_FAMILIES = ("deadleaves", "lines")


@dataclass(frozen=True)
class FamilyVisualReview:
    decision: ReviewDecision
    recognizable: bool
    flat: bool
    checkerboard: bool
    mirrored: bool
    saturation: bool
    collapse: bool
    notes: str


@dataclass(frozen=True)
class VisualReview:
    reviewer: str
    timestamp: str
    figure_sha256: str
    families: Mapping[str, FamilyVisualReview]
    legacy: FamilyVisualReview | None = None

    def _records(self) -> tuple[FamilyVisualReview, ...]:
        if self.families:
            return tuple(self.families.values())
        return () if self.legacy is None else (self.legacy,)

    @property
    def decision(self) -> ReviewDecision:
        records = self._records()
        return (
            ReviewDecision.APPROVE
            if records
            and all(review.decision is ReviewDecision.APPROVE for review in records)
            else ReviewDecision.REJECT
        )

    # Compatibility accessors keep report preservation code independent of the
    # family-aware storage shape. Gate evaluation uses the selected family.
    @property
    def recognizable(self) -> bool:
        records = self._records()
        return bool(records) and all(review.recognizable for review in records)

    def _any(self, name: str) -> bool:
        return any(getattr(review, name) for review in self._records())

    flat = property(lambda self: self._any("flat"))
    checkerboard = property(lambda self: self._any("checkerboard"))
    mirrored = property(lambda self: self._any("mirrored"))
    saturation = property(lambda self: self._any("saturation"))
    collapse = property(lambda self: self._any("collapse"))

    @property
    def notes(self) -> str:
        if self.legacy is not None:
            return self.legacy.notes
        return "; ".join(
            f"{family}: {review.notes}"
            for family, review in self.families.items()
            if review.notes
        )


def pending_review_template(
    figure_path: str, figure_sha256: str | None = None
) -> dict[str, object]:
    """Return a deliberately non-completed review record.

    ``figure_sha256`` binds a rendered figure to the pending review when
    available; this document is never accepted by :func:`parse_review`.
    """
    if not isinstance(figure_path, str) or not figure_path:
        raise ReviewError("pending review figure_path must be a nonempty string")
    if figure_sha256 is not None and (
        not isinstance(figure_sha256, str) or not _SHA256_RE.fullmatch(figure_sha256)
    ):
        raise ReviewError("pending review figure_sha256 must be a lowercase SHA-256")
    return {
        "schema_version": PENDING_REVIEW_SCHEMA_VERSION,
        "state": "pending",
        "figure_path": figure_path,
        "figure_sha256": figure_sha256,
        "families": list(_OBJECT_FAMILIES),
        "instructions": "Replace this pending template with a completed visual_review_v1 record after human review.",
    }


def parse_review(payload: object) -> VisualReview:
    """Validate and parse one completed, human-authored visual review."""
    if not isinstance(payload, Mapping):
        raise ReviewError("review must be an object")
    if payload.get("schema_version") == PENDING_REVIEW_SCHEMA_VERSION:
        raise ReviewError("pending review is not a completed review")
    review_fields = (
        _FAMILY_REVIEW_FIELDS if "families" in payload else _LEGACY_REVIEW_FIELDS
    )
    unknown = set(payload) - review_fields
    missing = review_fields - set(payload)
    if unknown:
        raise ReviewError(f"review has unknown fields: {sorted(unknown)!r}")
    if missing:
        raise ReviewError(f"review is missing fields: {sorted(missing)!r}")
    if payload["schema_version"] != REVIEW_SCHEMA_VERSION:
        raise ReviewError("review schema_version is unsupported")
    reviewer = payload["reviewer"]
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ReviewError("reviewer must be nonempty")
    timestamp = payload["timestamp"]
    if not isinstance(timestamp, str) or not _UTC_RFC3339_RE.fullmatch(timestamp):
        raise ReviewError("timestamp must be a UTC RFC3339 timestamp ending in Z")
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp[:-1] + "+00:00")
    except ValueError as error:
        raise ReviewError("timestamp must be a valid UTC RFC3339 timestamp") from error
    if parsed_timestamp.tzinfo is not timezone.utc:
        raise ReviewError("timestamp must use UTC")
    figure_sha256 = payload["figure_sha256"]
    if not isinstance(figure_sha256, str) or not _SHA256_RE.fullmatch(figure_sha256):
        raise ReviewError("figure_sha256 must be a lowercase SHA-256 digest")
    if "families" in payload:
        families = payload["families"]
        if not isinstance(families, Mapping) or set(families) != set(_OBJECT_FAMILIES):
            raise ReviewError("families must contain exactly deadleaves and lines")
        parsed_families = {
            family: _parse_family_review(families[family], family)
            for family in _OBJECT_FAMILIES
        }
    else:
        legacy = _parse_family_review(
            {field: payload[field] for field in _FAMILY_FIELDS}, "legacy"
        )
        parsed_families = {}
    return VisualReview(
        reviewer,
        timestamp,
        figure_sha256,
        MappingProxyType(parsed_families),
        None if "families" in payload else legacy,
    )


def _parse_family_review(payload: object, family: str) -> FamilyVisualReview:
    if not isinstance(payload, Mapping):
        raise ReviewError(f"family {family} review must be an object")
    unknown = set(payload) - _FAMILY_FIELDS
    missing = _FAMILY_FIELDS - set(payload)
    if unknown:
        raise ReviewError(
            f"family {family} review has unknown fields: {sorted(unknown)!r}"
        )
    if missing:
        raise ReviewError(
            f"family {family} review is missing fields: {sorted(missing)!r}"
        )
    try:
        decision = ReviewDecision(payload["decision"])
    except (TypeError, ValueError) as error:
        raise ReviewError("decision must be approve or reject") from error
    component_names = (
        "recognizable",
        "flat",
        "checkerboard",
        "mirrored",
        "saturation",
        "collapse",
    )
    components = {name: payload[name] for name in component_names}
    if any(type(value) is not bool for value in components.values()):
        raise ReviewError("review components must be bool values")
    notes = payload["notes"]
    if not isinstance(notes, str):
        raise ReviewError("notes must be a string")
    has_failure = (not components["recognizable"]) or any(
        components[name]
        for name in ("flat", "checkerboard", "mirrored", "saturation", "collapse")
    )
    if decision is ReviewDecision.APPROVE and has_failure:
        raise ReviewError(
            "approval requires recognizable structure and no failure flags"
        )
    return FamilyVisualReview(decision, notes=notes, **components)
