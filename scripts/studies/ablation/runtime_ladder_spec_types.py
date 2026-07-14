"""Value types for the bridge-ladder checked spec (budget extraction)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runtime_reference_spec import ExpectedDifference

__all__ = ["LadderBaseline", "LadderResidual", "LadderRung"]


@dataclass(frozen=True)
class LadderBaseline:
    """Rung 0 reference binding, current only after evidence is pinned."""

    id: str
    status: str
    reference_spec: Path
    reference_spec_declared: str
    reference_id: str
    evidence: Path
    evidence_declared: str
    evidence_sha256: str | None
    historical_evidence: Path | None
    historical_evidence_declared: str | None
    historical_evidence_sha256: str | None
    dataset: str
    config: Mapping[str, Any]


@dataclass(frozen=True)
class LadderRung:
    """One single-group configuration step, resolved against its predecessor."""

    id: str
    group: str
    dataset: str
    changes: Mapping[str, Any]
    expected_differences: Mapping[str, ExpectedDifference]
    requires_scan_accounting: bool
    requires_normalization_evidence: bool
    requires_count_error_evidence: bool
    #: Execution contract. Historical-only rungs remain parseable but cannot
    #: be selected or launched by the ladder runtime.
    execution_status: str
    #: Diagnostic branch: excluded from inheritance/control/endpoint chains.
    diagnostic: bool
    #: Diagnostic-only: names an earlier rung whose resolved config is the
    #: resolution base AND whose sealed evidence is the gate control.
    control_rung: str | None
    resolved_config: Mapping[str, Any]

    @property
    def runnable(self) -> bool:
        return self.execution_status == "runnable"


@dataclass(frozen=True)
class LadderResidual:
    """A documented reference-to-endpoint difference outside the ladder."""

    id: str
    description: str
