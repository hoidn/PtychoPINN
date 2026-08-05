"""Typed stage orchestration for the public synthetic PyTorch workflow.

This module owns workflow-level persistence and reuse.  Expensive scientific
work remains behind typed executor ports so the TensorFlow simulation leaf can
stay in a CUDA-hidden child process and reconstruction/evaluation can be
installed only when their public APIs exist.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Protocol

from ptycho.invocation_logging import write_invocation_artifacts
from ptycho.workflows.synthetic_config import (
    ResolvedSyntheticWorkflow,
    resolve_synthetic_workflow,
    synthetic_workflow_to_dict,
)
from ptycho_torch.rect_s1s2_initialization import (
    RectS1S2InitializationRecord,
)


STAGE_ORDER = ("simulate", "train", "reconstruct", "evaluate")
STAGE_MANIFEST_SCHEMA = "synthetic-stage-manifest-v2"
DIAGNOSTICS_SCHEMA = "synthetic-reconstruction-diagnostics-v1"
METRIC_CONTRACT_VERSION = "synthetic-quality-metrics-v1"
RECONSTRUCTION_SCHEMA = "synthetic-barycentric-reconstruction-v1"
_STAGE_LOG_STREAM_LIMIT = 16_384

_STAGE_NAMESPACES = {
    "simulate": ("simulation",),
    "train": ("simulation", "model", "training"),
    "reconstruct": ("simulation", "model", "training", "inference"),
    "evaluate": ("simulation", "model", "training", "inference"),
}
_STAGE_ARTIFACTS = {
    "simulate": (
        "datasets/source.npz",
        "datasets/train.npz",
        "datasets/test.npz",
        "datasets/manifest.json",
    ),
    "train": (
        "training/wts.h5.zip",
        "training/training_summary.json",
    ),
    "reconstruct": (
        "reconstruction/reconstruction.npz",
        "reconstruction/diagnostics.json",
    ),
    "evaluate": (
        "reconstruction/metrics.json",
        "reconstruction/comparison.png",
        "reconstruction/diagnostics.json",
    ),
}
_FRESH_STAGE_ARTIFACTS = {
    **_STAGE_ARTIFACTS,
    "reconstruct": ("reconstruction/reconstruction.npz",),
    "evaluate": (
        "reconstruction/metrics.json",
        "reconstruction/comparison.png",
    ),
}
_PARTIAL_STAGE_ARTIFACTS = {
    **_FRESH_STAGE_ARTIFACTS,
    "reconstruct": _STAGE_ARTIFACTS["reconstruct"],
}
_MANAGED_DIRECTORIES = (
    "datasets",
    "training",
    "training/checkpoints",
    "training/lightning_logs",
    "training/mlruns",
    "reconstruction",
    "stage_logs",
)
_MANAGED_FILES = tuple(
    dict.fromkeys(
        (
            "invocation.json",
            "invocation.sh",
            "resolved_workflow.json",
            "stage_manifest.json",
            "stage_logs/simulate_request.json",
            *(f"stage_logs/{stage}.log" for stage in STAGE_ORDER),
            *(item for paths in _STAGE_ARTIFACTS.values() for item in paths),
        )
    )
)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(frozen=True)
class SyntheticPipelineRequest:
    """Raw canonical inputs needed to resolve and reproduce one pipeline run."""

    profile: str = "synthetic-lines"
    file_values: Mapping[str, Any] | None = None
    cli_values: Mapping[str, Any] | None = None
    raw_argv: tuple[str, ...] = ()
    script_path: str = "ptycho_synthetic"

    def __post_init__(self) -> None:
        if self.file_values is not None and not isinstance(self.file_values, Mapping):
            raise TypeError("file_values must be a mapping or None")
        if self.cli_values is not None and not isinstance(self.cli_values, Mapping):
            raise TypeError("cli_values must be a mapping or None")
        object.__setattr__(
            self,
            "file_values",
            None if self.file_values is None else _freeze(self.file_values),
        )
        object.__setattr__(
            self,
            "cli_values",
            None if self.cli_values is None else _freeze(self.cli_values),
        )
        object.__setattr__(self, "raw_argv", tuple(str(item) for item in self.raw_argv))
        if not isinstance(self.profile, str) or not self.profile:
            raise TypeError("profile must be a nonempty string")
        if not isinstance(self.script_path, str) or not self.script_path:
            raise TypeError("script_path must be a nonempty string")


@dataclass(frozen=True)
class SimulationStageRequest:
    profile: str
    file_values: Mapping[str, Any] | None
    cli_values: Mapping[str, Any] | None
    resolved_workflow: ResolvedSyntheticWorkflow
    output_root: Path


@dataclass(frozen=True)
class TrainingStageRequest:
    resolved_workflow: ResolvedSyntheticWorkflow
    output_root: Path
    train_path: Path
    test_path: Path
    dataset_manifest_path: Path


@dataclass(frozen=True)
class ReconstructionStageRequest:
    resolved_workflow: ResolvedSyntheticWorkflow
    output_root: Path
    test_path: Path
    dataset_manifest_path: Path
    bundle_path: Path


@dataclass(frozen=True)
class EvaluationStageRequest:
    resolved_workflow: ResolvedSyntheticWorkflow
    output_root: Path
    source_path: Path
    dataset_manifest_path: Path
    reconstruction_path: Path
    diagnostics_path: Path


@dataclass(frozen=True)
class SimulationStageResult:
    source_path: Path
    train_path: Path
    test_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class TrainingStageResult:
    bundle_path: Path
    training_summary_path: Path
    rect_s1s2_initialization: RectS1S2InitializationRecord

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rect_s1s2_initialization",
            RectS1S2InitializationRecord.from_mapping(
                self.rect_s1s2_initialization
            ),
        )


@dataclass(frozen=True)
class ReconstructionStageResult:
    reconstruction_path: Path
    reassembly: Mapping[str, Any]


@dataclass(frozen=True)
class EvaluationStageResult:
    metrics_path: Path
    comparison_path: Path
    metric_validity: Mapping[str, Any]
    render: Mapping[str, Any]


@dataclass(frozen=True)
class SyntheticPipelineResult:
    output_root: Path
    resolved_workflow: ResolvedSyntheticWorkflow
    resolved_workflow_path: Path
    stage_manifest_path: Path
    diagnostics_path: Path
    completed_stages: tuple[str, ...]
    reused_stages: tuple[str, ...]


class SimulationExecutor(Protocol):
    def __call__(self, request: SimulationStageRequest) -> SimulationStageResult: ...


class TrainingExecutor(Protocol):
    def __call__(self, request: TrainingStageRequest) -> TrainingStageResult: ...


class ReconstructionExecutor(Protocol):
    def __call__(
        self, request: ReconstructionStageRequest
    ) -> ReconstructionStageResult: ...


class EvaluationExecutor(Protocol):
    def __call__(self, request: EvaluationStageRequest) -> EvaluationStageResult: ...


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_atomic(path: Path, encoded: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            _thaw(payload),
            indent=2,
            sort_keys=False,
            allow_nan=False,
            separators=(",", ": "),
        )
        + "\n"
    ).encode("utf-8")
    _write_bytes_atomic(path, encoded)


def _read_json_object(path: Path, *, artifact: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {artifact} at {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"invalid {artifact} at {path}: expected a JSON object")
    return payload


def _read_training_summary_record(
    path: Path,
    *,
    expected_mode: str | None = None,
) -> RectS1S2InitializationRecord:
    payload = _read_json_object(path, artifact="training summary")
    record = RectS1S2InitializationRecord.from_mapping(payload)
    if expected_mode is not None and record.mode != expected_mode:
        raise ValueError(
            f"training summary mode {record.mode!r} disagrees with resolved "
            f"rect_s1s2_init {expected_mode!r}"
        )
    return record


def _validate_training_stage_result(
    result: TrainingStageResult,
    resolved: ResolvedSyntheticWorkflow,
) -> RectS1S2InitializationRecord:
    if not isinstance(result, TrainingStageResult):
        raise TypeError("training executor must return TrainingStageResult")
    expected_mode = resolved.model.rect_s1s2_init
    backend_record = RectS1S2InitializationRecord.from_mapping(
        result.rect_s1s2_initialization
    )
    persisted_record = _read_training_summary_record(
        Path(result.training_summary_path),
        expected_mode=expected_mode,
    )
    if persisted_record != backend_record:
        raise ValueError(
            "persisted rect_s1s2 initialization record does not match backend "
            "training result"
        )
    if backend_record.mode != expected_mode:
        raise ValueError(
            f"backend rect_s1s2 initialization mode {backend_record.mode!r} "
            f"disagrees with resolved rect_s1s2_init {expected_mode!r}"
        )
    return backend_record


def _compact_log_text(value: Any, *, limit: int = _STAGE_LOG_STREAM_LIMIT) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    retained = (limit - 80) // 2
    omitted = len(text) - (2 * retained)
    return (
        text[:retained]
        + f"\n... truncated {omitted} characters ...\n"
        + text[-retained:]
    )


def _write_stage_failure_log(
    output_root: Path,
    *,
    stage: str,
    started_at: str,
    error: BaseException,
) -> None:
    path = output_root / "stage_logs" / f"{stage}.log"
    summary = (
        f"stage: {stage}\n"
        "status: failed\n"
        f"started_at: {started_at}\n"
        f"failed_at: {_timestamp()}\n"
        f"error_type: {type(error).__name__}\n"
        f"error: {_compact_log_text(error, limit=4096)}\n"
    ).encode("utf-8", errors="replace")
    try:
        if stage == "simulate" and path.is_file():
            previous = path.read_bytes()
            if previous and not previous.endswith(b"\n"):
                previous += b"\n"
            summary = previous + summary
        _write_bytes_atomic(path, summary)
    except OSError:
        # Failure evidence is best-effort and must never hide the stage error.
        pass


def _validate_pending_diagnostics(path: Path) -> dict[str, Any]:
    payload = _read_json_object(path, artifact="reconstruction diagnostics")
    if payload.get("schema_version") != DIAGNOSTICS_SCHEMA:
        raise ValueError(f"diagnostics.schema_version must be {DIAGNOSTICS_SCHEMA!r}")
    expected_fields = {
        "schema_version",
        "reassembly",
        "metric_validity",
        "render",
    }
    if set(payload) != expected_fields:
        raise ValueError(
            "reconstruction diagnostics fields must be "
            f"{sorted(expected_fields)!r}, got {sorted(payload)!r}"
        )
    if not isinstance(payload["reassembly"], dict):
        raise ValueError("diagnostics.reassembly must be an object")
    if payload["metric_validity"] is not None:
        raise ValueError("diagnostics.metric_validity must be null before evaluation")
    if payload["render"] is not None:
        raise ValueError("diagnostics.render must be null before evaluation")
    return payload


def _restore_bytes_if_changed(path: Path, expected: bytes) -> None:
    try:
        current = path.read_bytes()
    except OSError:
        current = None
    if current != expected:
        _write_bytes_atomic(path, expected)


def _relative_artifact_path(root: Path, path: Path) -> str:
    root_absolute = root.absolute()
    path_absolute = path.absolute()
    try:
        relative = path_absolute.relative_to(root_absolute)
    except ValueError as error:
        raise ValueError(
            f"unmanaged artifact path outside output root: {path}"
        ) from error
    if path.exists() and not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"unmanaged artifact path outside output root: {path}")
    return relative.as_posix()


def _validate_managed_preflight(root: Path) -> None:
    """Reject path redirection before any workflow-owned read or write."""

    resolved_root = root.resolve()
    for relative in _MANAGED_DIRECTORIES:
        candidate = root / relative
        if candidate.is_symlink():
            raise ValueError(f"managed path {relative} must not be a symlink")
        if candidate.exists():
            if not candidate.is_dir():
                raise ValueError(f"managed path {relative} must be a directory")
            if not candidate.resolve().is_relative_to(resolved_root):
                raise ValueError(f"managed path {relative} escapes output root")
    for relative in _MANAGED_FILES:
        candidate = root / relative
        if candidate.is_symlink():
            raise ValueError(f"managed path {relative} must not be a symlink")
        if candidate.exists():
            if not candidate.is_file():
                raise ValueError(f"managed path {relative} must be a file")
            if not candidate.resolve().is_relative_to(resolved_root):
                raise ValueError(f"managed path {relative} escapes output root")


def _validate_nonempty_file(path: Path, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"{label} is empty: {path}")


def _validate_exact_artifacts(
    root: Path,
    stage: str,
    observed: tuple[Path, ...],
) -> tuple[str, ...]:
    expected = _STAGE_ARTIFACTS[stage]
    relative = tuple(_relative_artifact_path(root, Path(path)) for path in observed)
    if relative != expected:
        raise ValueError(
            f"{stage} executor artifacts must be {expected!r}, got {relative!r}"
        )
    for item in relative:
        _validate_nonempty_file(root / item, label=f"{stage} artifact {item}")
    return relative


def _validate_fresh_artifacts(
    root: Path,
    stage: str,
    observed: tuple[Path, ...],
) -> tuple[str, ...]:
    expected = _FRESH_STAGE_ARTIFACTS[stage]
    relative = tuple(_relative_artifact_path(root, Path(path)) for path in observed)
    if relative != expected:
        raise ValueError(
            f"{stage} executor artifacts must be {expected!r}, got {relative!r}"
        )
    for item in relative:
        _validate_nonempty_file(root / item, label=f"{stage} artifact {item}")
    return relative


def _validate_manifest_entry(root: Path, stage: str, entry: Any) -> None:
    if not isinstance(entry, dict):
        raise ValueError(f"stage_manifest.stages.{stage} must be an object")
    if entry.get("status") != "complete":
        raise ValueError(f"stage_manifest.stages.{stage}.status must be 'complete'")
    for timestamp_name in ("started_at", "completed_at"):
        timestamp = entry.get(timestamp_name)
        if not isinstance(timestamp, str) or not timestamp:
            raise ValueError(
                f"stage_manifest.stages.{stage}.{timestamp_name} must be nonempty"
            )
    artifacts = entry.get("artifacts")
    if not isinstance(artifacts, list) or any(
        not isinstance(item, str) for item in artifacts
    ):
        raise ValueError(f"stage_manifest.stages.{stage}.artifacts must be strings")
    expected = _STAGE_ARTIFACTS[stage]
    if tuple(artifacts) != expected:
        raise ValueError(
            f"stage_manifest.stages.{stage}.artifacts must be {expected!r}, "
            f"got {tuple(artifacts)!r}"
        )
    for relative in artifacts:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(
                f"stage_manifest.stages.{stage}.artifacts contains unmanaged "
                f"path {relative!r}"
            )
        expected_path = root / candidate
        if not expected_path.absolute().is_relative_to(root.absolute()):
            raise ValueError(f"unmanaged stage artifact path {relative!r}")
        _validate_nonempty_file(
            expected_path,
            label=f"recorded {stage} artifact {relative}",
        )
        if not expected_path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"unmanaged stage artifact path {relative!r}")
    if stage == "train":
        _read_training_summary_record(root / "training" / "training_summary.json")


def _load_stage_manifest(path: Path, root: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": STAGE_MANIFEST_SCHEMA,
            "metric_contract_version": METRIC_CONTRACT_VERSION,
            "stages": {},
        }
    payload = _read_json_object(path, artifact="stage manifest")
    schema_version = payload.get("schema_version")
    if schema_version == "synthetic-stage-manifest-v1":
        raise ValueError(
            "historical synthetic-stage-manifest-v1 outputs do not carry the "
            "versioned training-summary contract; use a new output root or "
            "retrain before stage reuse"
        )
    if schema_version != STAGE_MANIFEST_SCHEMA:
        raise ValueError(
            f"stage_manifest.schema_version must be {STAGE_MANIFEST_SCHEMA!r}"
        )
    stages = payload.get("stages")
    if not isinstance(stages, dict):
        raise ValueError("stage_manifest.stages must be an object")
    unknown = set(stages) - set(STAGE_ORDER)
    if unknown:
        raise ValueError(f"stage_manifest contains unknown stages: {sorted(unknown)!r}")
    ordered: dict[str, Any] = {}
    for stage in STAGE_ORDER:
        if stage in stages:
            _validate_manifest_entry(root, stage, stages[stage])
            ordered[stage] = stages[stage]
    payload["stages"] = ordered
    return payload


def _first_difference(recorded: Any, current: Any, path: str) -> str | None:
    if isinstance(recorded, Mapping) and isinstance(current, Mapping):
        for key in current:
            child_path = f"{path}.{key}" if path else str(key)
            if key not in recorded:
                return child_path
            difference = _first_difference(recorded[key], current[key], child_path)
            if difference is not None:
                return difference
        for key in recorded:
            if key not in current:
                return f"{path}.{key}" if path else str(key)
        return None
    if type(recorded) is not type(current) or recorded != current:
        return path
    return None


def _assert_stage_identity(
    stage: str,
    recorded: Mapping[str, Any],
    current: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    for name in ("schema_version", "profile", "recipe_version"):
        difference = _first_difference(recorded.get(name), current.get(name), name)
        if difference is not None:
            raise ValueError(f"{difference} conflicts with reusable {stage} identity")
    for namespace in _STAGE_NAMESPACES[stage]:
        difference = _first_difference(
            recorded.get(namespace),
            current.get(namespace),
            namespace,
        )
        if difference is not None:
            raise ValueError(f"{difference} conflicts with reusable {stage} identity")
    if stage == "evaluate":
        recorded_version = manifest.get("metric_contract_version")
        if recorded_version != METRIC_CONTRACT_VERSION:
            raise ValueError(
                "metric_contract_version conflicts with reusable evaluate identity: "
                f"recorded {recorded_version!r}, current {METRIC_CONTRACT_VERSION!r}"
            )


def _stage_identity_error(
    stage: str,
    recorded: Mapping[str, Any],
    current: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> ValueError | None:
    try:
        _assert_stage_identity(stage, recorded, current, manifest)
    except ValueError as error:
        return error
    return None


def _reject_partial_artifacts(root: Path, stage: str) -> None:
    found = [
        relative
        for relative in _PARTIAL_STAGE_ARTIFACTS[stage]
        if (root / relative).exists()
    ]
    if found:
        raise FileExistsError(
            f"partial {stage} artifact exists; use a new output root: {found[0]}"
        )


def _required_predecessors(stage: str) -> tuple[str, ...]:
    return STAGE_ORDER[: STAGE_ORDER.index(stage)]


def _require_prerequisite(
    manifest: Mapping[str, Any],
    root: Path,
    prerequisite: str,
    stage: str,
) -> None:
    entry = manifest["stages"].get(prerequisite)
    if entry is None:
        raise FileNotFoundError(
            f"{prerequisite} prerequisite for {stage} is not complete under {root}"
        )
    _validate_manifest_entry(root, prerequisite, entry)


def _simulation_result_paths(result: SimulationStageResult) -> tuple[Path, ...]:
    if not isinstance(result, SimulationStageResult):
        raise TypeError("simulation executor must return SimulationStageResult")
    return (
        Path(result.source_path),
        Path(result.train_path),
        Path(result.test_path),
        Path(result.manifest_path),
    )


def _training_result_paths(result: TrainingStageResult) -> tuple[Path, ...]:
    if not isinstance(result, TrainingStageResult):
        raise TypeError("training executor must return TrainingStageResult")
    return (
        Path(result.bundle_path),
        Path(result.training_summary_path),
    )


def _reconstruction_result_paths(
    result: ReconstructionStageResult,
    diagnostics_path: Path,
) -> tuple[Path, ...]:
    if not isinstance(result, ReconstructionStageResult):
        raise TypeError("reconstruction executor must return ReconstructionStageResult")
    if not isinstance(result.reassembly, Mapping):
        raise TypeError("reconstruction reassembly diagnostics must be a mapping")
    return (Path(result.reconstruction_path), diagnostics_path)


def _evaluation_result_paths(
    result: EvaluationStageResult,
    diagnostics_path: Path,
) -> tuple[Path, ...]:
    if not isinstance(result, EvaluationStageResult):
        raise TypeError("evaluation executor must return EvaluationStageResult")
    if not isinstance(result.metric_validity, Mapping):
        raise TypeError("evaluation metric_validity must be a mapping")
    if not isinstance(result.render, Mapping):
        raise TypeError("evaluation render diagnostics must be a mapping")
    return (
        Path(result.metrics_path),
        Path(result.comparison_path),
        diagnostics_path,
    )


def execute_simulation_stage(request: SimulationStageRequest) -> SimulationStageResult:
    """Run the TensorFlow-reachable simulation only in its CUDA-hidden worker."""

    if not isinstance(request, SimulationStageRequest):
        raise TypeError("request must be a SimulationStageRequest")
    request.output_root.mkdir(parents=True, exist_ok=True)
    _validate_managed_preflight(request.output_root)
    log_root = request.output_root / "stage_logs"
    request_path = log_root / "simulate_request.json"
    log_path = log_root / "simulate.log"
    _write_json_atomic(
        request_path,
        {
            "profile": request.profile,
            "file_values": _thaw(request.file_values),
            "cli_values": _thaw(request.cli_values),
        },
    )
    command = [
        sys.executable,
        "-m",
        "scripts.simulation.synthetic_simulation_worker",
        "--request-json",
        str(request_path),
        "--output-root",
        str(request.output_root),
    ]
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout = _compact_log_text(completed.stdout)
    stderr = _compact_log_text(completed.stderr)
    log_payload = (
        f"command: {shlex.join(command)}\n"
        f"returncode: {completed.returncode}\n"
        f"stdout:\n{stdout}\n"
        f"stderr:\n{stderr}\n"
    )
    _write_bytes_atomic(log_path, log_payload.encode("utf-8"))
    if completed.returncode != 0:
        raise RuntimeError(
            "simulate stage failed: "
            f"command={shlex.join(command)!r}, returncode={completed.returncode}, "
            f"log={log_path}"
        )
    dataset_root = request.output_root / "datasets"
    result = SimulationStageResult(
        source_path=dataset_root / "source.npz",
        train_path=dataset_root / "train.npz",
        test_path=dataset_root / "test.npz",
        manifest_path=dataset_root / "manifest.json",
    )
    for path in _simulation_result_paths(result):
        _validate_nonempty_file(path, label="simulation worker artifact")
    return result


def _run_shared_training_workflow(request: Any) -> Any:
    from ptycho.workflows.training import run_training_workflow

    return run_training_workflow(request)


def execute_training_stage(request: TrainingStageRequest) -> TrainingStageResult:
    """Delegate model construction and training to the shared generic workflow."""

    if not isinstance(request, TrainingStageRequest):
        raise TypeError("request must be a TrainingStageRequest")
    _validate_managed_preflight(request.output_root)
    manifest = _load_matching_dataset_manifest(
        request.dataset_manifest_path,
        request.resolved_workflow,
    )
    _verify_split_artifact(
        request.train_path,
        manifest=manifest,
        manifest_path=request.dataset_manifest_path,
        resolved=request.resolved_workflow,
        split="train",
    )
    _verify_split_artifact(
        request.test_path,
        manifest=manifest,
        manifest_path=request.dataset_manifest_path,
        resolved=request.resolved_workflow,
        split="test",
    )
    from ptycho.workflows.training import TrainingWorkflowRequest

    result = _run_shared_training_workflow(
        TrainingWorkflowRequest(
            resolved_synthetic_workflow=request.resolved_workflow,
            train_data_file=request.train_path,
            test_data_file=request.test_path,
            output_dir=request.output_root / "training",
            do_stitching=False,
        )
    )
    if result.bundle_path is None:
        raise FileNotFoundError("shared training workflow did not return a bundle path")
    if result.training_summary_path is None:
        raise FileNotFoundError(
            "shared training workflow did not return a training summary path"
        )
    if result.rect_s1s2_initialization is None:
        raise ValueError(
            "shared training workflow did not return rect_s1s2 initialization"
        )
    stage_result = TrainingStageResult(
        bundle_path=Path(result.bundle_path),
        training_summary_path=Path(result.training_summary_path),
        rect_s1s2_initialization=result.rect_s1s2_initialization,
    )
    _validate_training_stage_result(stage_result, request.resolved_workflow)
    return stage_result


def _load_matching_dataset_manifest(
    manifest_path: Path,
    resolved: ResolvedSyntheticWorkflow,
) -> dict[str, Any]:
    from ptycho.simulation.flat_acquisition import (
        OBJECT_PRODUCER_SYMBOLS,
        derive_seed_lineage,
    )
    from ptycho.simulation.identity import file_sha256

    manifest = _read_json_object(manifest_path, artifact="dataset manifest")
    if manifest.get("schema_version") != "flat-acquisition-manifest-v1":
        raise ValueError(
            "dataset manifest schema_version must be 'flat-acquisition-manifest-v1'"
        )
    if manifest.get("storage_layout") != "flat_acquisition_v1":
        raise ValueError(
            "dataset manifest storage_layout must be 'flat_acquisition_v1'"
        )
    if manifest.get("profile") != resolved.profile:
        raise ValueError("dataset manifest profile disagrees with resolved workflow")
    if manifest.get("recipe_version") != resolved.recipe_version:
        raise ValueError(
            "dataset manifest recipe_version disagrees with resolved workflow"
        )
    expected_semantic = synthetic_workflow_to_dict(resolved)["simulation"]
    if manifest.get("simulation") != expected_semantic:
        raise ValueError("dataset manifest simulation disagrees with resolved workflow")
    expected_lineage = derive_seed_lineage(resolved.simulation.train.seed)
    if manifest.get("seed_lineage") != expected_lineage:
        raise ValueError(
            "dataset manifest seed_lineage disagrees with resolved workflow"
        )
    expected_measurement = {
        "measurement_domain": resolved.simulation.measurement_domain,
        "scale_contract_version": resolved.simulation.scale_contract_version,
    }
    if manifest.get("measurement_identity") != expected_measurement:
        raise ValueError(
            "dataset manifest measurement_identity disagrees with resolved workflow"
        )
    object_record = manifest.get("object")
    if not isinstance(object_record, Mapping):
        raise ValueError("dataset manifest object must be an object")
    if object_record.get("recipe") != resolved.simulation.object_recipe:
        raise ValueError("dataset manifest object.recipe mismatch")
    if object_record.get("producer_symbols") != list(OBJECT_PRODUCER_SYMBOLS):
        raise ValueError("dataset manifest object.producer_symbols mismatch")
    source_commit = object_record.get("source_commit")
    if not isinstance(source_commit, str) or not source_commit:
        raise ValueError("dataset manifest object.source_commit must be non-empty")
    _require_sha256(
        object_record.get("array_sha256"),
        label="dataset manifest object.array_sha256",
    )
    if object_record.get("seed") != expected_lineage["object"]:
        raise ValueError("dataset manifest object.seed mismatch")
    probe_record = manifest.get("probe")
    if not isinstance(probe_record, Mapping):
        raise ValueError("dataset manifest probe must be an object")
    expected_probe = resolved.simulation.train.probe
    expected_source_path = (
        str(expected_probe.source_path)
        if expected_probe.source_path is not None
        else None
    )
    for name, expected in (
        ("source_kind", expected_probe.source),
        ("source_path", expected_source_path),
        ("normalized_transform_pipeline", expected_probe.transform_pipeline),
    ):
        if probe_record.get(name) != expected:
            raise ValueError(f"dataset manifest probe.{name} mismatch")
    source_digest = probe_record.get("source_file_sha256")
    if expected_probe.source_path is None:
        if source_digest is not None:
            raise ValueError("dataset manifest probe.source_file_sha256 must be null")
    elif not expected_probe.source_path.is_file():
        raise FileNotFoundError(
            f"custom probe source does not exist: {expected_probe.source_path}"
        )
    elif source_digest != file_sha256(expected_probe.source_path):
        raise ValueError("dataset manifest probe.source_file_sha256 mismatch")
    for name in ("raw_probe_sha256", "transformed_probe_sha256"):
        _require_sha256(
            probe_record.get(name),
            label=f"dataset manifest probe.{name}",
        )
    return manifest


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be a SHA-256 digest") from error
    return value


def _manifest_artifact_path(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    *,
    name: str,
    expected_path: Path,
) -> None:
    artifacts = manifest.get("artifacts")
    artifact_name = artifacts.get(name) if isinstance(artifacts, Mapping) else None
    if not isinstance(artifact_name, str) or not artifact_name:
        raise ValueError(f"dataset manifest artifacts.{name} must be a path")
    recorded_path = manifest_path.parent / artifact_name
    if recorded_path.resolve() != expected_path.resolve():
        raise ValueError(
            f"dataset manifest artifacts.{name} does not identify {expected_path}"
        )


def _npz_identity(
    path: Path,
) -> tuple[dict[str, str], dict[str, list[int]], dict[str, str]]:
    import numpy as np

    from ptycho.simulation.identity import array_sha256

    hashes: dict[str, str] = {}
    shapes: dict[str, list[int]] = {}
    dtypes: dict[str, str] = {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            for name in archive.files:
                array = np.asarray(archive[name])
                hashes[name] = array_sha256(array)
                shapes[name] = list(array.shape)
                dtypes[name] = array.dtype.name
    except (OSError, ValueError) as error:
        raise ValueError(
            f"invalid flat acquisition artifact at {path}: {error}"
        ) from error
    return hashes, shapes, dtypes


def _verify_split_artifact(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    resolved: ResolvedSyntheticWorkflow,
    split: str,
) -> None:
    from ptycho.config import simulation_config_sha256, simulation_config_to_dict
    from ptycho.simulation.flat_acquisition import (
        STORAGE_LAYOUT,
        derive_seed_lineage,
    )
    from ptycho.simulation.identity import canonical_sha256, file_sha256

    if split not in {"train", "test"}:
        raise ValueError(f"unsupported flat acquisition split {split!r}")
    _manifest_artifact_path(
        manifest,
        manifest_path,
        name=split,
        expected_path=path,
    )
    splits = manifest.get("splits")
    record = splits.get(split) if isinstance(splits, Mapping) else None
    if not isinstance(record, Mapping):
        raise ValueError(f"dataset manifest is missing splits.{split}")
    artifact_path = manifest_path.parent / str(record.get("artifact_path", ""))
    if artifact_path.resolve() != path.resolve():
        raise ValueError(
            f"dataset manifest splits.{split}.artifact_path does not identify {path}"
        )
    if record.get("storage_layout") != STORAGE_LAYOUT:
        raise ValueError(f"dataset manifest splits.{split}.storage_layout mismatch")
    expected_simulation = getattr(resolved.simulation, split)
    if record.get("simulation_config") != simulation_config_to_dict(
        expected_simulation
    ):
        raise ValueError(
            f"dataset manifest splits.{split}.simulation_config disagrees "
            "with resolved workflow"
        )
    if record.get("simulation_config_sha256") != simulation_config_sha256(
        expected_simulation
    ):
        raise ValueError(
            f"dataset manifest splits.{split}.simulation_config_sha256 mismatch"
        )
    expected_measurement = {
        "measurement_domain": resolved.simulation.measurement_domain,
        "scale_contract_version": resolved.simulation.scale_contract_version,
        "photons_per_pattern": float(expected_simulation.detector.photons_per_pattern),
    }
    if record.get("measurement_identity") != expected_measurement:
        raise ValueError(
            f"dataset manifest splits.{split}.measurement_identity mismatch"
        )
    seed_lineage = derive_seed_lineage(resolved.simulation.train.seed)
    if record.get("seed_lineage") != seed_lineage:
        raise ValueError(f"dataset manifest splits.{split}.seed_lineage mismatch")
    for field_name, lineage_name in (
        ("coordinate_seed", f"{split}_coordinates"),
        ("detector_seed", f"{split}_noise"),
    ):
        if record.get(field_name) != seed_lineage[lineage_name]:
            raise ValueError(f"dataset manifest splits.{split}.{field_name} mismatch")
    object_record = manifest.get("object")
    probe_record = manifest.get("probe")
    if not isinstance(object_record, Mapping) or not isinstance(probe_record, Mapping):
        raise ValueError("dataset manifest object and probe lineage are required")
    object_identity = {
        name: object_record.get(name)
        for name in (
            "recipe",
            "producer_symbols",
            "source_commit",
            "array_sha256",
        )
    }
    expected_recipe_identity = {
        "split": split,
        "storage_layout": STORAGE_LAYOUT,
        "simulation_config_sha256": simulation_config_sha256(expected_simulation),
        "object_identity": object_identity,
        "raw_probe_sha256": probe_record.get("raw_probe_sha256"),
        "transformed_probe_sha256": probe_record.get("transformed_probe_sha256"),
        "coordinate_seed": seed_lineage[f"{split}_coordinates"],
        "detector_seed": seed_lineage[f"{split}_noise"],
        "measurement_identity": expected_measurement,
    }
    if record.get("split_recipe_identity") != expected_recipe_identity:
        raise ValueError(
            f"dataset manifest splits.{split}.split_recipe_identity mismatch"
        )
    split_recipe_sha256 = canonical_sha256(expected_recipe_identity)
    for name in ("split_recipe_sha256", "dataset_recipe_sha256"):
        if record.get(name) != split_recipe_sha256:
            raise ValueError(f"dataset manifest splits.{split}.{name} mismatch")
    if record.get("npz_sha256") != file_sha256(path):
        raise ValueError(f"dataset manifest splits.{split}.npz_sha256 mismatch")
    hashes, shapes, dtypes = _npz_identity(path)
    for name, computed in (
        ("array_sha256", hashes),
        ("shapes", shapes),
        ("dtypes", dtypes),
    ):
        recorded = record.get(name)
        if not isinstance(recorded, Mapping) or dict(recorded) != computed:
            raise ValueError(f"dataset manifest splits.{split}.{name} mismatch")
    if hashes.get("objectGuess") != object_identity["array_sha256"]:
        raise ValueError(
            f"dataset manifest splits.{split}.objectGuess lineage mismatch"
        )
    if hashes.get("probeGuess") != probe_record.get("transformed_probe_sha256"):
        raise ValueError(f"dataset manifest splits.{split}.probeGuess lineage mismatch")
    dataset_identity = {
        "split_recipe_sha256": split_recipe_sha256,
        "array_sha256": hashes,
        "shapes": shapes,
        "dtypes": dtypes,
    }
    if record.get("dataset_identity") != dataset_identity:
        raise ValueError(f"dataset manifest splits.{split}.dataset_identity mismatch")
    if record.get("dataset_sha256") != canonical_sha256(dataset_identity):
        raise ValueError(f"dataset manifest splits.{split}.dataset_sha256 mismatch")


def _load_verified_source_truth(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> Any:
    import numpy as np

    from ptycho.simulation.identity import array_sha256, file_sha256

    _manifest_artifact_path(
        manifest,
        manifest_path,
        name="source",
        expected_path=path,
    )
    if manifest.get("source_npz_sha256") != file_sha256(path):
        raise ValueError("dataset manifest source_npz_sha256 mismatch")
    try:
        with np.load(path, allow_pickle=False) as source:
            if "objectGuess" not in source.files:
                raise ValueError("source artifact is missing objectGuess")
            if "probeGuess" not in source.files:
                raise ValueError("source artifact is missing probeGuess")
            truth = np.array(source["objectGuess"], copy=True)
            probe = np.array(source["probeGuess"], copy=True)
    except OSError as error:
        raise ValueError(f"invalid source artifact at {path}: {error}") from error
    object_record = manifest.get("object")
    if not isinstance(object_record, Mapping):
        raise ValueError("dataset manifest object must be an object")
    if object_record.get("array_sha256") != array_sha256(truth):
        raise ValueError("dataset manifest object.array_sha256 mismatch")
    if object_record.get("shape") != list(truth.shape):
        raise ValueError("dataset manifest object.shape mismatch")
    if object_record.get("dtype") != truth.dtype.name:
        raise ValueError("dataset manifest object.dtype mismatch")
    probe_record = manifest.get("probe")
    if not isinstance(probe_record, Mapping):
        raise ValueError("dataset manifest probe must be an object")
    if probe_record.get("transformed_probe_sha256") != array_sha256(probe):
        raise ValueError("dataset manifest probe.transformed_probe_sha256 mismatch")
    return truth


def _json_scalar(value: Mapping[str, Any], *, name: str) -> str:
    try:
        return json.dumps(
            _thaw(value),
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite JSON object") from error


def _validated_reconstruction_arrays(
    *,
    complex_canvas: Any,
    amplitude: Any,
    phase: Any,
    prescale_canvas: Any,
    canvas_weights: Any,
    canvas_anchor: Any,
    channel_indices: Any,
    expected_channels: int,
) -> dict[str, Any]:
    """Copy and validate the complete portable reconstruction payload."""

    import numpy as np

    arrays = {
        "complex_canvas": np.asarray(complex_canvas),
        "amplitude": np.asarray(amplitude),
        "phase": np.asarray(phase),
        "prescale_canvas": np.asarray(prescale_canvas),
        "canvas_weights": np.asarray(canvas_weights),
        "channel_indices": np.asarray(channel_indices),
    }
    canvas = arrays["complex_canvas"]
    if canvas.ndim != 2 or not np.issubdtype(canvas.dtype, np.complexfloating):
        raise ValueError("complex_canvas must be a rank-2 complex array")
    for name in ("amplitude", "phase", "prescale_canvas", "canvas_weights"):
        array = arrays[name]
        if array.ndim != 2 or array.shape != canvas.shape:
            raise ValueError(f"{name} shape must match complex_canvas")
    if not np.issubdtype(arrays["prescale_canvas"].dtype, np.complexfloating):
        raise ValueError("prescale_canvas must be complex")
    for name in ("amplitude", "phase", "canvas_weights"):
        if not np.issubdtype(arrays[name].dtype, np.number) or np.issubdtype(
            arrays[name].dtype, np.complexfloating
        ):
            raise ValueError(f"{name} must be a real numeric array")
    for name, array in arrays.items():
        if name != "channel_indices" and not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")
    weights = arrays["canvas_weights"]
    if np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("canvas_weights must contain nonempty nonnegative support")
    rows = arrays["channel_indices"]
    if (
        rows.ndim != 2
        or rows.shape[0] == 0
        or rows.shape[1] != expected_channels
        or not np.issubdtype(rows.dtype, np.integer)
        or np.any(rows < 0)
    ):
        raise ValueError(
            "channel_indices must contain nonempty nonnegative integer rows "
            f"with C={expected_channels}"
        )
    if any(len(set(row.tolist())) != expected_channels for row in rows):
        raise ValueError(
            "channel_indices must contain distinct scan ids within every group"
        )
    if not isinstance(canvas_anchor, Mapping):
        raise ValueError("canvas_anchor must be a mapping")
    anchor = dict(canvas_anchor)
    required_anchor = {"scan_com", "canvas_shape", "canvas_origin_offset"}
    if not required_anchor.issubset(anchor):
        raise ValueError(
            "canvas_anchor must contain scan_com, canvas_shape, and "
            "canvas_origin_offset"
        )
    if tuple(anchor["canvas_shape"]) != canvas.shape:
        raise ValueError("canvas_anchor canvas_shape must match complex_canvas")
    anchor_json = _json_scalar(anchor, name="canvas_anchor")
    return {
        "schema_version": np.asarray(RECONSTRUCTION_SCHEMA),
        "complex_canvas": np.array(canvas, copy=True),
        "amplitude": np.array(arrays["amplitude"], copy=True),
        "phase": np.array(arrays["phase"], copy=True),
        "prescale_canvas": np.array(arrays["prescale_canvas"], copy=True),
        "canvas_weights": np.array(weights, copy=True),
        "canvas_anchor_json": np.asarray(anchor_json),
        "channel_indices": np.array(rows, dtype=np.int64, copy=True),
    }


def _write_reconstruction_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    import io

    import numpy as np

    stream = io.BytesIO()
    np.savez(stream, **payload)
    _write_bytes_atomic(path, stream.getvalue())


def _validate_reassembly_channel_handoff(
    reassembly: Mapping[str, Any],
    channel_indices: Any,
) -> None:
    import numpy as np

    rows = np.asarray(channel_indices)
    patch_count = int(rows.size)
    for name in ("accepted_patches", "total_patches"):
        value = reassembly.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"reassembly.{name} must be an integer")
        if value != patch_count:
            raise ValueError(
                f"reassembly.{name} must match channel_indices size "
                f"({value} != {patch_count})"
            )
    used_scan_ids = reassembly.get("used_scan_ids")
    if not isinstance(used_scan_ids, (list, tuple)) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in used_scan_ids
    ):
        raise ValueError("reassembly.used_scan_ids must contain integer ids")
    if set(used_scan_ids) != set(rows.reshape(-1).tolist()):
        raise ValueError("reassembly.used_scan_ids must match channel_indices scan ids")


def _load_reconstruction_artifact(
    path: Path,
    *,
    expected_channels: int,
) -> dict[str, Any]:
    """Strictly reload one portable reconstruction without pickle support."""

    import numpy as np

    expected_fields = {
        "schema_version",
        "complex_canvas",
        "amplitude",
        "phase",
        "prescale_canvas",
        "canvas_weights",
        "canvas_anchor_json",
        "channel_indices",
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected_fields:
                raise ValueError(
                    "reconstruction artifact fields must be "
                    f"{sorted(expected_fields)!r}, got {sorted(archive.files)!r}"
                )
            schema = np.asarray(archive["schema_version"])
            anchor_text = np.asarray(archive["canvas_anchor_json"])
            if schema.shape != () or schema.item() != RECONSTRUCTION_SCHEMA:
                raise ValueError(
                    f"reconstruction schema_version must be {RECONSTRUCTION_SCHEMA!r}"
                )
            if anchor_text.shape != () or not isinstance(anchor_text.item(), str):
                raise ValueError("canvas_anchor_json must be a scalar JSON string")
            try:
                anchor = json.loads(anchor_text.item())
            except json.JSONDecodeError as error:
                raise ValueError("canvas_anchor_json is invalid JSON") from error
            if not isinstance(anchor, dict):
                raise ValueError("canvas_anchor_json must decode to an object")
            payload = _validated_reconstruction_arrays(
                complex_canvas=np.array(archive["complex_canvas"], copy=True),
                amplitude=np.array(archive["amplitude"], copy=True),
                phase=np.array(archive["phase"], copy=True),
                prescale_canvas=np.array(archive["prescale_canvas"], copy=True),
                canvas_weights=np.array(archive["canvas_weights"], copy=True),
                canvas_anchor=anchor,
                channel_indices=np.array(archive["channel_indices"], copy=True),
                expected_channels=expected_channels,
            )
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(
            (
                "reconstruction ",
                "canvas_",
                "complex_",
                "amplitude",
                "phase",
                "prescale",
                "channel_",
            )
        ):
            raise
        raise ValueError(
            f"invalid reconstruction artifact at {path}: {error}"
        ) from error
    payload["canvas_anchor"] = anchor
    return payload


def _execution_request_for_resolved(resolved: ResolvedSyntheticWorkflow) -> Any:
    from ptycho.config.config import PyTorchExecutionConfig
    from ptycho_torch.execution_request import ExecutionRequest

    execution_names = {item.name for item in fields(PyTorchExecutionConfig)}
    values = {
        item.name: getattr(resolved.workflow, item.name)
        for item in fields(resolved.workflow)
        if item.name in execution_names
    }
    return ExecutionRequest(values=values, explicit_fields=frozenset(values))


def execute_reconstruction_stage(
    request: ReconstructionStageRequest,
) -> ReconstructionStageResult:
    """Strictly reload, reconstruct through mmap, and persist raw evidence."""

    if not isinstance(request, ReconstructionStageRequest):
        raise TypeError("request must be a ReconstructionStageRequest")
    _validate_managed_preflight(request.output_root)
    from ptycho_torch.execution_request import resolve_runtime_execution_request
    from ptycho_torch.inference import reconstruct_npz_barycentric

    execution = resolve_runtime_execution_request(
        _execution_request_for_resolved(request.resolved_workflow),
        mode="inference",
    ).config
    if execution.devices != 1:
        raise ValueError(
            "barycentric reconstruction requires exactly one resolved device"
        )
    device = "cuda" if execution.accelerator == "gpu" else execution.accelerator
    result = reconstruct_npz_barycentric(
        request.bundle_path,
        request.test_path,
        run_root=request.output_root,
        groups_per_center=(request.resolved_workflow.inference.groups_per_center),
        expected_workflow=request.resolved_workflow,
        dataset_manifest_path=request.dataset_manifest_path,
        device=device,
        num_workers=execution.num_workers,
        inference_batch_size=request.resolved_workflow.inference.batch_size,
        precision=execution.precision,
        quiet=True,
    )
    payload = _validated_reconstruction_arrays(
        complex_canvas=result.complex_canvas,
        amplitude=result.amplitude,
        phase=result.phase,
        prescale_canvas=result.prescale_canvas,
        canvas_weights=result.canvas_weights,
        canvas_anchor=result.canvas_anchor,
        channel_indices=result.channel_indices,
        expected_channels=int(request.resolved_workflow.data.C),
    )
    reassembly = result.reassembly.to_jsonable()
    if not isinstance(reassembly, Mapping):
        raise TypeError("reassembly.to_jsonable() must return a mapping")
    _json_scalar(dict(reassembly), name="reassembly")
    _validate_reassembly_channel_handoff(
        reassembly,
        payload["channel_indices"],
    )
    reconstruction_path = request.output_root / "reconstruction" / "reconstruction.npz"
    _write_reconstruction_atomic(reconstruction_path, payload)
    _load_reconstruction_artifact(
        reconstruction_path,
        expected_channels=int(request.resolved_workflow.data.C),
    )
    return ReconstructionStageResult(
        reconstruction_path=reconstruction_path,
        reassembly=dict(reassembly),
    )


def execute_evaluation_stage(
    request: EvaluationStageRequest,
) -> EvaluationStageResult:
    """Reload raw reconstruction/truth arrays and publish quality artifacts."""

    if not isinstance(request, EvaluationStageRequest):
        raise TypeError("request must be an EvaluationStageRequest")
    _validate_managed_preflight(request.output_root)
    import numpy as np

    from ptycho_torch.reconstruction_evaluation import (
        evaluate_reconstruction_quality,
    )

    reconstruction = _load_reconstruction_artifact(
        request.reconstruction_path,
        expected_channels=int(request.resolved_workflow.data.C),
    )
    manifest = _load_matching_dataset_manifest(
        request.dataset_manifest_path,
        request.resolved_workflow,
    )
    truth = _load_verified_source_truth(
        request.source_path,
        manifest=manifest,
        manifest_path=request.dataset_manifest_path,
    )
    if (
        truth.ndim != 2
        or not np.issubdtype(truth.dtype, np.complexfloating)
        or not np.isfinite(truth).all()
    ):
        raise ValueError("source objectGuess must be a finite rank-2 complex array")
    diagnostics = _validate_pending_diagnostics(request.diagnostics_path)
    result = evaluate_reconstruction_quality(
        complex_canvas=reconstruction["complex_canvas"],
        prescale_canvas=reconstruction["prescale_canvas"],
        canvas_weights=reconstruction["canvas_weights"],
        canvas_anchor=reconstruction["canvas_anchor"],
        truth=truth,
        reassembly=diagnostics["reassembly"],
        channel_indices=reconstruction["channel_indices"],
        groups_per_center=(request.resolved_workflow.inference.groups_per_center),
        output_dir=request.output_root / "reconstruction",
        expected_channels=int(request.resolved_workflow.data.C),
    )
    return EvaluationStageResult(
        metrics_path=Path(result.metrics_path),
        comparison_path=Path(result.comparison_path),
        metric_validity=dict(result.metric_validity),
        render=dict(result.render),
    )


def _write_invocation(
    request: SyntheticPipelineRequest,
    resolved: ResolvedSyntheticWorkflow,
    output_root: Path,
) -> None:
    write_invocation_artifacts(
        output_root,
        request.script_path,
        request.raw_argv,
        {
            "profile": request.profile,
            "file_values": _thaw(request.file_values),
            "cli_values": _thaw(request.cli_values),
            "stages": list(resolved.workflow.stages),
            "output_root": str(output_root),
        },
    )


def _validate_execution_preflight(
    resolved: ResolvedSyntheticWorkflow,
) -> None:
    """Run the canonical pure Torch execution contract before any stage."""

    from ptycho_torch.execution_request import (
        normalize_execution_input,
        validate_execution_input_phase,
        validate_execution_input_structure,
    )

    normalized = normalize_execution_input(
        _execution_request_for_resolved(resolved),
        mode="training",
    )
    if normalized is None:  # pragma: no cover - a request was supplied above
        raise RuntimeError("synthetic execution request did not normalize")
    validate_execution_input_structure(normalized)
    validate_execution_input_phase(normalized, mode="training")


def _execute_pipeline_stage(
    stage: str,
    *,
    request: SyntheticPipelineRequest,
    resolved: ResolvedSyntheticWorkflow,
    output_root: Path,
    diagnostics_path: Path,
    simulation_executor: SimulationExecutor,
    training_executor: TrainingExecutor,
    reconstruction_executor: ReconstructionExecutor,
    evaluation_executor: EvaluationExecutor,
) -> tuple[str, ...]:
    """Execute and validate one stage without publishing its completion."""

    if stage == "simulate":
        stage_result = simulation_executor(
            SimulationStageRequest(
                profile=request.profile,
                file_values=request.file_values,
                cli_values=request.cli_values,
                resolved_workflow=resolved,
                output_root=output_root,
            )
        )
        return _validate_exact_artifacts(
            output_root,
            stage,
            _simulation_result_paths(stage_result),
        )
    if stage == "train":
        stage_result = training_executor(
            TrainingStageRequest(
                resolved_workflow=resolved,
                output_root=output_root,
                train_path=output_root / "datasets" / "train.npz",
                test_path=output_root / "datasets" / "test.npz",
                dataset_manifest_path=output_root / "datasets" / "manifest.json",
            )
        )
        _validate_training_stage_result(stage_result, resolved)
        return _validate_exact_artifacts(
            output_root,
            stage,
            _training_result_paths(stage_result),
        )
    if stage == "reconstruct":
        if diagnostics_path.exists():
            raise FileExistsError(
                "partial reconstruct artifact exists; use a new output root: "
                "reconstruction/diagnostics.json"
            )
        stage_result = reconstruction_executor(
            ReconstructionStageRequest(
                resolved_workflow=resolved,
                output_root=output_root,
                test_path=output_root / "datasets" / "test.npz",
                dataset_manifest_path=output_root / "datasets" / "manifest.json",
                bundle_path=output_root / "training" / "wts.h5.zip",
            )
        )
        if diagnostics_path.exists():
            raise ValueError(
                "reconstruction executor must not write pipeline-owned "
                "reconstruction/diagnostics.json"
            )
        if not isinstance(stage_result, ReconstructionStageResult):
            raise TypeError(
                "reconstruction executor must return ReconstructionStageResult"
            )
        reconstruction_paths = _reconstruction_result_paths(
            stage_result,
            diagnostics_path,
        )
        _validate_fresh_artifacts(
            output_root,
            stage,
            reconstruction_paths[:-1],
        )
        _write_json_atomic(
            diagnostics_path,
            {
                "schema_version": DIAGNOSTICS_SCHEMA,
                "reassembly": stage_result.reassembly,
                "metric_validity": None,
                "render": None,
            },
        )
        return _validate_exact_artifacts(
            output_root,
            stage,
            reconstruction_paths,
        )

    diagnostics_before = diagnostics_path.read_bytes()
    diagnostics = _validate_pending_diagnostics(diagnostics_path)
    try:
        stage_result = evaluation_executor(
            EvaluationStageRequest(
                resolved_workflow=resolved,
                output_root=output_root,
                source_path=output_root / "datasets" / "source.npz",
                dataset_manifest_path=(output_root / "datasets" / "manifest.json"),
                reconstruction_path=(
                    output_root / "reconstruction" / "reconstruction.npz"
                ),
                diagnostics_path=diagnostics_path,
            )
        )
    except BaseException:
        _restore_bytes_if_changed(diagnostics_path, diagnostics_before)
        raise
    try:
        diagnostics_unchanged = diagnostics_path.read_bytes() == diagnostics_before
    except OSError:
        diagnostics_unchanged = False
    if not diagnostics_unchanged:
        _restore_bytes_if_changed(diagnostics_path, diagnostics_before)
        raise ValueError(
            "evaluation executor must not overwrite pipeline-owned "
            "reconstruction/diagnostics.json"
        )
    if not isinstance(stage_result, EvaluationStageResult):
        raise TypeError("evaluation executor must return EvaluationStageResult")
    evaluation_paths = _evaluation_result_paths(
        stage_result,
        diagnostics_path,
    )
    _validate_fresh_artifacts(
        output_root,
        stage,
        evaluation_paths[:-1],
    )
    diagnostics["metric_validity"] = dict(stage_result.metric_validity)
    diagnostics["render"] = dict(stage_result.render)
    _write_json_atomic(diagnostics_path, diagnostics)
    return _validate_exact_artifacts(
        output_root,
        stage,
        evaluation_paths,
    )


def run_synthetic_pipeline(
    request: SyntheticPipelineRequest,
    *,
    simulation_executor: SimulationExecutor = execute_simulation_stage,
    training_executor: TrainingExecutor = execute_training_stage,
    reconstruction_executor: ReconstructionExecutor = execute_reconstruction_stage,
    evaluation_executor: EvaluationExecutor = execute_evaluation_stage,
) -> SyntheticPipelineResult:
    """Resolve, validate, and execute one strict synthetic stage subsequence."""

    if not isinstance(request, SyntheticPipelineRequest):
        raise TypeError("request must be a SyntheticPipelineRequest")
    resolved = resolve_synthetic_workflow(
        profile=request.profile,
        file_values=request.file_values,
        cli_values=request.cli_values,
    )
    _validate_execution_preflight(resolved)
    stages = tuple(resolved.workflow.stages)
    if "simulate" in stages:
        from ptycho.simulation.flat_acquisition import (
            validate_flat_acquisition_workflow,
        )

        validate_flat_acquisition_workflow(resolved)
    output_root = Path(resolved.workflow.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _validate_managed_preflight(output_root)
    resolved_path = output_root / "resolved_workflow.json"
    manifest_path = output_root / "stage_manifest.json"
    diagnostics_path = output_root / "reconstruction" / "diagnostics.json"
    current_payload = synthetic_workflow_to_dict(resolved)
    recorded_payload = (
        _read_json_object(resolved_path, artifact="resolved workflow")
        if resolved_path.exists()
        else None
    )
    manifest = _load_stage_manifest(manifest_path, output_root)
    completed_before = manifest["stages"]
    if completed_before and recorded_payload is None:
        raise FileNotFoundError(
            f"resolved_workflow.json is required to reuse stages under {output_root}"
        )
    selected = set(stages)
    required_identity = set(stages)
    manifest_pruned = False
    for stage in stages:
        required_identity.update(_required_predecessors(stage))
    if recorded_payload is not None:
        identity_errors = {
            completed_stage: error
            for completed_stage in completed_before
            if (
                error := _stage_identity_error(
                    completed_stage,
                    recorded_payload,
                    current_payload,
                    manifest,
                )
            )
            is not None
        }
        for completed_stage in STAGE_ORDER:
            if (
                completed_stage in required_identity
                and completed_stage in identity_errors
            ):
                raise identity_errors[completed_stage]
        if identity_errors:
            first_incompatible = min(
                STAGE_ORDER.index(stage) for stage in identity_errors
            )
            for downstream in STAGE_ORDER[first_incompatible:]:
                manifest["stages"].pop(downstream, None)
            manifest["metric_contract_version"] = METRIC_CONTRACT_VERSION
            manifest_pruned = True
            completed_before = manifest["stages"]

    if "train" in completed_before:
        _read_training_summary_record(
            output_root / "training" / "training_summary.json",
            expected_mode=resolved.model.rect_s1s2_init,
        )

    for stage in stages:
        for prerequisite in _required_predecessors(stage):
            if prerequisite not in selected:
                _require_prerequisite(
                    manifest,
                    output_root,
                    prerequisite,
                    stage,
                )

    for stage in stages:
        if stage in completed_before:
            if not resolved.workflow.reuse_complete_artifacts:
                raise FileExistsError(
                    f"complete {stage} artifact already exists; use a new output root"
                )
        else:
            _reject_partial_artifacts(output_root, stage)

    if manifest_pruned:
        _write_json_atomic(manifest_path, manifest)
    _write_invocation(request, resolved, output_root)
    _write_json_atomic(resolved_path, current_payload)

    reused: list[str] = []
    for stage in stages:
        if stage in manifest["stages"]:
            reused.append(stage)
            continue
        for prerequisite in _required_predecessors(stage):
            _require_prerequisite(
                manifest,
                output_root,
                prerequisite,
                stage,
            )

        started_at = _timestamp()
        try:
            artifacts = _execute_pipeline_stage(
                stage,
                request=request,
                resolved=resolved,
                output_root=output_root,
                diagnostics_path=diagnostics_path,
                simulation_executor=simulation_executor,
                training_executor=training_executor,
                reconstruction_executor=reconstruction_executor,
                evaluation_executor=evaluation_executor,
            )
        except BaseException as error:
            _write_stage_failure_log(
                output_root,
                stage=stage,
                started_at=started_at,
                error=error,
            )
            raise

        manifest["metric_contract_version"] = METRIC_CONTRACT_VERSION
        manifest["stages"][stage] = {
            "status": "complete",
            "started_at": started_at,
            "completed_at": _timestamp(),
            "artifacts": list(artifacts),
        }
        manifest["stages"] = {
            name: manifest["stages"][name]
            for name in STAGE_ORDER
            if name in manifest["stages"]
        }
        _write_json_atomic(manifest_path, manifest)

    return SyntheticPipelineResult(
        output_root=output_root,
        resolved_workflow=resolved,
        resolved_workflow_path=resolved_path,
        stage_manifest_path=manifest_path,
        diagnostics_path=diagnostics_path,
        completed_stages=tuple(
            stage for stage in STAGE_ORDER if stage in manifest["stages"]
        ),
        reused_stages=tuple(reused),
    )


__all__ = [
    "DIAGNOSTICS_SCHEMA",
    "METRIC_CONTRACT_VERSION",
    "RECONSTRUCTION_SCHEMA",
    "STAGE_MANIFEST_SCHEMA",
    "STAGE_ORDER",
    "EvaluationExecutor",
    "EvaluationStageRequest",
    "EvaluationStageResult",
    "ReconstructionExecutor",
    "ReconstructionStageRequest",
    "ReconstructionStageResult",
    "SimulationExecutor",
    "SimulationStageRequest",
    "SimulationStageResult",
    "SyntheticPipelineRequest",
    "SyntheticPipelineResult",
    "TrainingExecutor",
    "TrainingStageRequest",
    "TrainingStageResult",
    "execute_evaluation_stage",
    "execute_reconstruction_stage",
    "execute_simulation_stage",
    "execute_training_stage",
    "run_synthetic_pipeline",
]
