"""Study-request loading, dataset-spec registration, and dry-run planning.

Everything in this module is safe for ``--dry-run``: it never reads NPZ
content, never preflights dataset files, and never allocates an accelerator.
Standalone ``--dataset-spec`` descriptors are schema-parsed only; their
content preflight happens at execution time in :mod:`runtime_study`.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .configuration import ConfigResolutionError, resolve_torch_configs
from .datasets import DatasetError, parse_standalone_dataset_descriptor
from .manifest import (
    Manifest,
    ManifestError,
    ResolvedRun,
    ResolvedStudy,
    _parse_manifest,
    load_manifest,
    resolve_manifest,
    select_runs,
)
from .runtime_errors import StudyRequestError
from .verdicts import (
    IntegrationBridgeEvidence,
    Verdict,
    VerdictInputError,
    evaluate_integration_bridge,
)
from .visual_review import ReviewError


#: Error types the CLI reports as usage/validation failures (exit code 2).
USAGE_ERRORS = (StudyRequestError, ManifestError, DatasetError, ReviewError)


@dataclass(frozen=True)
class StudyRequest:
    """Typed CLI request consumed by the study runtime."""

    spec: Path
    dataset: str | None = None
    dataset_spec: Path | None = None
    dry_run: bool = False
    only: str | None = None
    seeds: tuple[int, ...] | None = None
    epochs: int | None = None
    output_root: Path | None = None
    resume: bool = False
    rerun: bool = False
    fail_fast: bool = False
    visual_review: Path | None = None
    integration_bridge_evidence: Path | None = None


@dataclass(frozen=True)
class LoadedStudy:
    manifest: Manifest
    manifest_sha256: str
    study: ResolvedStudy
    selected: tuple[ResolvedRun, ...]
    standalone_specs: dict[str, Path]
    requested_seeds: tuple[int, ...]
    effective_milestones: tuple[int, ...]
    integration_bridge_evidence: IntegrationBridgeEvidence | None
    integration_bridge_evidence_sha256: str | None


def _load_integration_bridge_evidence(
    manifest: Manifest,
    path: Path | None,
) -> tuple[IntegrationBridgeEvidence | None, str | None]:
    if path is None:
        return None, None
    requirement = manifest.integration_bridge_requirement
    if requirement is None:
        raise StudyRequestError(
            "--integration-bridge-evidence was supplied, but the manifest has "
            "no integration-bridge requirement"
        )
    try:
        artifact = path.read_bytes()
    except OSError as error:
        raise StudyRequestError(
            f"cannot read --integration-bridge-evidence {path}: {error}"
        ) from error
    try:
        evidence = IntegrationBridgeEvidence.from_sealed_artifact_bytes(artifact)
        result = evaluate_integration_bridge(requirement, evidence)
    except VerdictInputError as error:
        raise StudyRequestError(
            f"--integration-bridge-evidence {path}: {error}"
        ) from error
    if result.verdict is not Verdict.PASS:
        reason = result.reason or result.verdict.value
        raise StudyRequestError(
            f"--integration-bridge-evidence {path} does not satisfy the "
            f"manifest requirement: {reason}"
        )
    return evidence, hashlib.sha256(artifact).hexdigest()


def _resolve_effective_milestones(
    manifest: Manifest,
    study: ResolvedStudy,
    *,
    epochs_override: int | None,
) -> tuple[int, ...]:
    declared = manifest.diagnostics.milestones
    if not declared:
        return ()
    if epochs_override is not None:
        return tuple(epoch for epoch in declared if epoch <= epochs_override)

    maximum = declared[-1]
    for arm in study.arms:
        epochs = arm.overrides.get("training.epochs")
        if type(epochs) is not int:
            raise StudyRequestError(
                f"arm {arm.id} diagnostics milestone {maximum} requires explicit "
                "training.epochs; default-resolved epoch budgets are not accepted"
            )
        if epochs < maximum:
            raise StudyRequestError(
                f"arm {arm.id} resolves training.epochs={epochs}, below declared "
                f"diagnostics milestone {maximum}; use an explicit epochs override "
                "to reduce the milestone schedule"
            )
    return declared


def _merge_dataset_spec(manifest: Manifest, path: Path) -> tuple[Manifest, str]:
    """Register one standalone dataset descriptor without editing the spec."""
    try:
        with path.open("rb") as handle:
            raw_spec = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise StudyRequestError(f"cannot read --dataset-spec {path}: {error}") from error
    try:
        descriptor = parse_standalone_dataset_descriptor(
            raw_spec,
            descriptor_path=path,
        )
    except DatasetError as error:
        raise StudyRequestError(f"--dataset-spec {path}: {error}") from error
    table = dict(raw_spec["dataset"])
    dataset_id = descriptor.id
    table.pop("id")
    raw = manifest.to_dict()
    declared = raw.get("datasets", {})
    if dataset_id in declared:
        raise StudyRequestError(
            f"--dataset-spec id {dataset_id!r} duplicates a declared dataset id"
        )
    declared[dataset_id] = table
    raw["datasets"] = declared
    return (
        _parse_manifest(raw, dataset_descriptors={dataset_id: descriptor}),
        dataset_id,
    )


def load_study(request: StudyRequest) -> LoadedStudy:
    """Load, optionally extend, resolve, and select the requested study."""
    manifest = load_manifest(Path(request.spec))
    standalone_specs: dict[str, Path] = {}
    if request.dataset_spec is not None:
        manifest, dataset_id = _merge_dataset_spec(
            manifest, Path(request.dataset_spec)
        )
        standalone_specs[dataset_id] = Path(request.dataset_spec)
    study = resolve_manifest(
        manifest,
        dataset=request.dataset,
        epochs=request.epochs,
        seeds=request.seeds,
        output_root=None if request.output_root is None else str(request.output_root),
    )
    selected = select_runs(study.runs, request.only)
    manifest_sha256 = hashlib.sha256(
        manifest.canonical_json.encode("utf-8")
    ).hexdigest()
    requested_seeds = (
        tuple(request.seeds) if request.seeds is not None else manifest.seeds
    )
    effective_milestones = _resolve_effective_milestones(
        manifest,
        study,
        epochs_override=request.epochs,
    )
    bridge_evidence, bridge_evidence_sha256 = _load_integration_bridge_evidence(
        manifest,
        request.integration_bridge_evidence,
    )
    return LoadedStudy(
        manifest,
        manifest_sha256,
        study,
        selected,
        standalone_specs,
        requested_seeds,
        effective_milestones,
        bridge_evidence,
        bridge_evidence_sha256,
    )


def study_output_root(loaded: LoadedStudy, *, required: bool) -> Path | None:
    roots = {run.output_root for run in loaded.selected}
    root = roots.pop() if len(roots) == 1 else None
    if root is None and required:
        raise StudyRequestError(
            "an output root is required: pass --output-root or set study.output_root"
        )
    return None if root is None else Path(root)


def render_dry_run(loaded: LoadedStudy, request: StudyRequest) -> str:
    """Render the auditable expansion without loading NPZ data or CUDA."""
    output_root = study_output_root(loaded, required=False)
    lines = [
        f"study {loaded.manifest.study_id}",
        f"manifest_sha256 {loaded.manifest_sha256}",
        f"output_root {output_root if output_root is not None else '(not set)'}",
    ]
    if loaded.integration_bridge_evidence_sha256 is not None:
        lines.append(
            "integration_bridge_evidence_sha256 "
            + loaded.integration_bridge_evidence_sha256
        )
    for dataset_id in sorted(loaded.standalone_specs):
        lines.append(f"dataset_spec {dataset_id} schema_parsed_only")
    if loaded.manifest.diagnostics.milestones:
        declared = json.dumps(loaded.manifest.diagnostics.milestones)
        effective = json.dumps(loaded.effective_milestones)
        lines.append(f"milestones declared={declared} effective={effective}")
    lines.append(f"runs {len(loaded.selected)}")
    for run in loaded.selected:
        lines.append(f"run {run.id} dataset={run.dataset_id} seed={run.seed}")
    seen_arms: dict[str, ResolvedRun] = {}
    for run in loaded.selected:
        seen_arms.setdefault(run.arm_id, run)
    for arm_id, run in seen_arms.items():
        overrides = json.dumps(dict(run.overrides), sort_keys=True)
        lines.append(f"arm {arm_id} overrides {overrides}")
        try:
            resolve_torch_configs(
                dict(run.overrides),
                require_all_explicit=loaded.manifest.require_all_explicit,
            )
        except ConfigResolutionError as error:
            raise StudyRequestError(
                f"arm {arm_id} failed config validation: {error}"
            ) from error
    if loaded.manifest.require_all_explicit:
        lines.append(
            "config_validation strict_explicit_ok "
            f"arms={len(seen_arms)} dataset_validation=deferred"
        )
    else:
        lines.append(
            f"config_validation resolved_ok arms={len(seen_arms)} strict=false"
        )
    for gate in loaded.study.gates:
        line = (
            f"gate {gate.id} target={gate.target_arm_id} dataset={gate.dataset_id} "
            f"applicability={gate.applicability.value}"
        )
        if gate.reason:
            line += f" reason={gate.reason}"
        lines.append(line)
    for comparison in loaded.study.comparisons:
        line = (
            f"comparison {comparison.id} left={comparison.left_arm_id} "
            f"right={comparison.right_arm_id} "
            f"applicability={comparison.applicability.value}"
        )
        if comparison.reason:
            line += f" reason={comparison.reason}"
        lines.append(line)
    return "\n".join(lines)
