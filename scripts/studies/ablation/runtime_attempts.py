"""Attempt lifecycle for one logical run: fingerprints, artifacts, reuse.

Bridges canonical-run results to the Task 6 atomic-artifact contract: content
fingerprint inputs, resume/rerun attempt preparation, required-artifact
writing, completion, and reconstruction of typed rows from a validated
completed attempt.
"""

from __future__ import annotations

import csv
import io
import json
import traceback
from pathlib import Path
from typing import Any

from .artifacts import (
    CorruptCompletionError,
    FingerprintPair,
    InferenceFingerprintInput,
    PreparedAttempt,
    PrepareOutcome,
    TrainingFingerprintInput,
    allocate_attempt,
    inference_fingerprint,
    prepare_attempt,
    write_artifact_atomic,
)
from .manifest import ResolvedRun, _thaw
from .reporting import ReportRow
from .runtime_records import (
    TRAINING_HISTORY_ARTIFACT,
    flat_metrics,
    load_npy,
    npy_bytes,
    records_from_payload,
    records_to_payload,
    stored_history_curves,
    training_history_payload,
)
from .verdicts import AttemptRow, AttemptStatus, CompletionState

_RUN_METRICS_SCHEMA = "ablation_run_metrics_v1"
_PROVENANCE_SCHEMA = "ablation_provenance_v1"
_EMPTY_CURVES: tuple[tuple[float, ...], tuple[float, ...]] = ((), ())


def content_sha256s(descriptor: Any) -> dict[str, str]:
    """Collect every dataset/provenance/probe SHA-256 for the fingerprint."""
    content = {
        "dataset.train": descriptor.train_sha256,
        "dataset.test": descriptor.test_sha256,
        "dataset.provenance": descriptor.provenance_sha256,
    }
    if descriptor.reference_sha256 is not None:
        content["dataset.reference"] = descriptor.reference_sha256
    probe = descriptor.probe
    if probe.sha256 is not None:
        content["dataset.probe"] = probe.sha256
    else:
        content["dataset.probe.train"] = probe.train_sha256
        content["dataset.probe.test"] = probe.test_sha256
    return content


def latest_completed_attempt(run_root: Path) -> Path | None:
    if not run_root.is_dir():
        return None
    completed: list[tuple[int, Path]] = []
    for entry in run_root.iterdir():
        name = entry.name
        if entry.is_dir() and name.startswith("attempt-") and name[8:].isdigit():
            if (entry / "completion.json").is_file():
                completed.append((int(name[8:]), entry))
    if not completed:
        return None
    return max(completed)[1]


def _stored_checkpoint_sha256(attempt: Path) -> str:
    try:
        payload = json.loads((attempt / "checkpoint.json").read_text(encoding="utf-8"))
        stored = payload["best_checkpoint_sha256"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise CorruptCompletionError(
            "cannot read the stored checkpoint identity; use --rerun to archive "
            f"the attempt: {error}"
        ) from error
    if not isinstance(stored, str) or not stored:
        raise CorruptCompletionError(
            "stored checkpoint identity is invalid; use --rerun to archive the attempt"
        )
    return stored


def prepare_run_attempt(
    run_root: Path,
    training_input: TrainingFingerprintInput,
    training_fp: str,
    *,
    rerun: bool,
) -> PreparedAttempt:
    """Reuse a validated completion, or allocate/archive per resume/rerun."""
    if rerun:
        # The inference member of the pair is unused by rerun archiving; the
        # training fingerprint is repeated to satisfy the typed pair contract.
        return prepare_attempt(
            run_root, FingerprintPair(training_fp, training_fp), rerun=True
        )
    completed = latest_completed_attempt(run_root)
    if completed is None:
        return PreparedAttempt(PrepareOutcome.ALLOCATED, allocate_attempt(run_root))
    expected = FingerprintPair(
        training_fp,
        inference_fingerprint(
            InferenceFingerprintInput(
                training=training_input,
                selected_checkpoint_sha256=_stored_checkpoint_sha256(completed),
            )
        ),
    )
    return prepare_attempt(run_root, expected)


def write_json_artifact(attempt: Path, relative: str, payload: Any) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    write_artifact_atomic(attempt, relative, encoded.encode("utf-8"))


def provenance_payload(
    training_input: TrainingFingerprintInput,
    *,
    selected_checkpoint_sha256: str,
    training_fp: str,
    inference_fp: str,
) -> dict[str, Any]:
    """Readable per-attempt provenance (design step 2) from the fingerprint input."""
    git = training_input.git
    return {
        "schema_version": _PROVENANCE_SCHEMA,
        "manifest_sha256": training_input.manifest_sha256,
        "logical_run_id": training_input.logical_run_id,
        "seed": training_input.seed,
        "git": {
            "commit": git.commit,
            "clean": git.clean,
            "tracked_patch_sha256": git.tracked_patch_sha256,
            "untracked_sources": [
                {"path": item.path, "sha256": item.sha256}
                for item in git.untracked_sources
            ],
        },
        "environment_digest": training_input.environment_digest,
        "content_sha256s": _thaw(training_input.content_sha256s),
        "resolved_configs": _thaw(training_input.resolved_configs),
        "claim_grade": training_input.claim_grade,
        "selected_checkpoint_sha256": selected_checkpoint_sha256,
        "fingerprints": {"training": training_fp, "inference": inference_fp},
    }


def write_run_artifacts(
    attempt: Path,
    run: ResolvedRun,
    resolved: Any,
    result: Any,
    records: tuple[Any, ...],
    arrays: dict[str, Any],
    training_input: TrainingFingerprintInput,
    training_fp: str,
    inference_fp: str,
    *,
    source_manifest: bytes,
) -> tuple[str, ...]:
    """Write every required artifact and return the completion artifact list."""
    claim_grade = training_input.claim_grade
    write_artifact_atomic(
        attempt, "resolved_config.json", resolved.canonical_json.encode("utf-8")
    )
    write_artifact_atomic(attempt, "source_manifest.toml", source_manifest)
    write_artifact_atomic(
        attempt, "source_config.json", resolved.canonical_json.encode("utf-8")
    )
    write_json_artifact(attempt, "effective_runtime.json", result.effective_runtime)
    write_json_artifact(
        attempt,
        "provenance.json",
        provenance_payload(
            training_input,
            selected_checkpoint_sha256=result.best_checkpoint_sha256,
            training_fp=training_fp,
            inference_fp=inference_fp,
        ),
    )
    write_json_artifact(
        attempt,
        TRAINING_HISTORY_ARTIFACT,
        training_history_payload(result.training_history),
    )
    write_json_artifact(
        attempt,
        "checkpoint.json",
        {
            "best_checkpoint": str(result.best_checkpoint),
            "best_checkpoint_sha256": result.best_checkpoint_sha256,
            "fixed_batch_identity": result.fixed_batch_identity,
            "run_dir": str(result.run_dir),
        },
    )
    write_json_artifact(
        attempt,
        "fingerprints.json",
        {
            "training": training_fp,
            "inference": inference_fp,
            "claim_grade": claim_grade,
        },
    )
    write_json_artifact(
        attempt,
        "diagnostics.json",
        {
            "reference": result.reference_diagnostics.to_jsonable(),
            "reloaded": result.reloaded_diagnostics.to_jsonable(),
            "count_metrics": result.count_metrics.to_jsonable(),
            "reload_max_abs_error": result.reload_max_abs_error,
            "reload_allclose": result.reload_allclose,
        },
    )
    write_json_artifact(
        attempt,
        "metrics.json",
        {"schema_version": _RUN_METRICS_SCHEMA, "records": records_to_payload(records)},
    )
    metrics = flat_metrics(records)
    buffer = io.StringIO(newline="")
    fieldnames = ("run_id", "arm_id", "dataset_id", "seed", "metrics")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(
        {
            "run_id": run.id,
            "arm_id": run.arm_id,
            "dataset_id": run.dataset_id,
            "seed": run.seed,
            "metrics": json.dumps(metrics, sort_keys=True, separators=(",", ":")),
        }
    )
    write_artifact_atomic(attempt, "metrics.csv", buffer.getvalue().encode("utf-8"))
    write_json_artifact(
        attempt,
        "verdict.json",
        {
            "status": "success",
            "run_id": run.id,
            "arm_id": run.arm_id,
            "dataset_id": run.dataset_id,
            "seed": run.seed,
        },
    )
    canvases = {
        "arrays/prereload_texture.npy": result.reference_texture,
        "arrays/prereload_canvas.npy": result.reference_canvas,
        "arrays/reload_texture.npy": result.reloaded_texture,
        "arrays/stitched_canvas.npy": result.reloaded_canvas,
    }
    for relative, value in canvases.items():
        write_artifact_atomic(attempt, relative, npy_bytes(value))
    report_arrays = {"arrays/report_reconstruction.npy": arrays["reconstruction"]}
    if "target" in arrays:
        report_arrays["arrays/report_target.npy"] = arrays["target"]
        report_arrays["arrays/report_error.npy"] = arrays["error"]
        report_arrays["arrays/report_common_valid_mask.npy"] = arrays[
            "common_valid_mask"
        ]
    for relative, value in report_arrays.items():
        write_artifact_atomic(attempt, relative, npy_bytes(value))
    return (
        "resolved_config.json",
        "source_manifest.toml",
        "source_config.json",
        "effective_runtime.json",
        "provenance.json",
        TRAINING_HISTORY_ARTIFACT,
        "checkpoint.json",
        "fingerprints.json",
        "diagnostics.json",
        "metrics.json",
        "metrics.csv",
        "verdict.json",
        *canvases,
        *report_arrays,
    )


def success_rows(
    run: ResolvedRun,
    truth_role: str,
    records: tuple[Any, ...],
    arrays: dict[str, Any],
    history_curves: tuple[tuple[float, ...], tuple[float, ...]] = _EMPTY_CURVES,
    source_fingerprint: str | None = None,
) -> tuple[AttemptRow, ReportRow]:
    attempt_row = AttemptRow(
        run_id=run.id,
        arm_id=run.arm_id,
        dataset_id=run.dataset_id,
        seed=run.seed,
        status=AttemptStatus.SUCCESS,
        completion=CompletionState.TERMINAL,
        metrics=flat_metrics(records),
    )
    report_row = ReportRow(
        attempt=attempt_row,
        truth_role=truth_role,
        reconstruction=arrays["reconstruction"],
        target=arrays.get("target"),
        error=arrays.get("error"),
        common_valid_mask=arrays.get("common_valid_mask"),
        training_loss=history_curves[0],
        gradient_norm=history_curves[1],
        metric_records=records,
        source_fingerprint=source_fingerprint,
    )
    return attempt_row, report_row


def rows_from_completed_attempt(
    run: ResolvedRun, attempt: Path
) -> tuple[AttemptRow, ReportRow]:
    """Rebuild the typed evidence rows from a validated completed attempt."""
    try:
        payload = json.loads((attempt / "metrics.json").read_text(encoding="utf-8"))
        rows = payload["records"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise CorruptCompletionError(
            f"cannot reuse stored run metrics; use --rerun to archive: {error}"
        ) from error
    records = records_from_payload(rows, run.dataset.truth)
    arrays: dict[str, Any] = {
        "reconstruction": load_npy(attempt / "arrays" / "report_reconstruction.npy")
    }
    target_path = attempt / "arrays" / "report_target.npy"
    if target_path.is_file():
        arrays["target"] = load_npy(target_path)
        arrays["error"] = load_npy(attempt / "arrays" / "report_error.npy")
        mask_path = attempt / "arrays" / "report_common_valid_mask.npy"
        if mask_path.is_file():
            arrays["common_valid_mask"] = load_npy(mask_path)
    try:
        fingerprints = json.loads(
            (attempt / "fingerprints.json").read_text(encoding="utf-8")
        )
        source_fingerprint = fingerprints["inference"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise CorruptCompletionError(
            f"cannot reuse stored run fingerprints; use --rerun to archive: {error}"
        ) from error
    return success_rows(
        run,
        run.dataset.truth,
        records,
        arrays,
        stored_history_curves(attempt),
        source_fingerprint,
    )


def record_failure(attempt: Path | None, stage: str, error: Exception) -> None:
    if attempt is None:
        return
    payload = {
        "stage": stage,
        "error": f"{type(error).__name__}: {error}",
        "traceback": traceback.format_exc(),
    }
    try:
        write_json_artifact(attempt, "failure.json", payload)
    except Exception:
        # Recording is auxiliary evidence; the original failure propagates.
        pass
