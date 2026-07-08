"""Immutable BRDT 40-epoch paper-evidence runner."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from scripts.studies.born_rytov_dt import convergence as conv_mod
from scripts.studies.born_rytov_dt import dataset_contract as dc
from scripts.studies.born_rytov_dt import extension_bundle as ext_bundle
from scripts.studies.born_rytov_dt import preflight_metrics as metrics_mod
from scripts.studies.born_rytov_dt import preflight_visuals as visuals_mod
from scripts.studies.born_rytov_dt import run_preflight as preflight_mod
from scripts.studies.born_rytov_dt.data import (
    DatasetAuthority,
    load_dataset_authority,
)
from scripts.studies.born_rytov_dt.run_config import RowConfig
from scripts.studies.invocation_logging import (
    capture_runtime_provenance,
    get_git_commit,
    get_git_dirty,
    write_invocation_artifacts,
)


SCRIPT_PATH = "scripts/studies/born_rytov_dt/run_brdt_40ep_paper_evidence.py"
BACKLOG_ITEM = "2026-05-05-brdt-supervised-born-40ep-paper-evidence"
PRE_GATE_CLAIM_BOUNDARY = "decision_support_convergence_followup"
PASSED_CLAIM_BOUNDARY = "paper_evidence_brdt_additive"
# Backward-compat alias for tests/imports that still reference the bare name.
CLAIM_BOUNDARY = PRE_GATE_CLAIM_BOUNDARY
EXPECTED_EPOCHS = 40
ROW_SUMMARY_NAME = preflight_mod.ROW_SUMMARY_NAME
WRITER_LOCK_NAME = ".writer.lock"
DURABLE_SUMMARY_PATH = (
    "docs/plans/NEURIPS-HYBRID-RESNET-2026/"
    "brdt_supervised_born_40ep_paper_evidence_summary.md"
)
PAPER_EVIDENCE_INDEX_PATH = (
    "docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md"
)
PAPER_EVIDENCE_MANIFEST_PATH = (
    "docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json"
)
PAPER_EVIDENCE_PACKAGE_DESIGN_PATH = (
    "docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md"
)
KNOWN_CLAIM_BOUNDARIES = (
    PASSED_CLAIM_BOUNDARY,
    PRE_GATE_CLAIM_BOUNDARY,
)


class WriterConflictError(RuntimeError):
    """Raised when another writer is or was claiming the same output root."""


def _default_contract() -> preflight_mod.TrainingContract:
    return preflight_mod.TrainingContract(
        epochs=EXPECTED_EPOCHS,
        batch_size=16,
        learning_rate=2e-4,
        optimizer="Adam",
        seed=42,
        scheduler="reduce_on_plateau",
        plateau_factor=0.5,
        plateau_patience=2,
        plateau_threshold=0.0,
        plateau_min_lr=1e-5,
    )


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False
    return True


def _acquire_writer_lock(output_root: Path, *, allow_force: bool) -> Path:
    """Refuse to start when another writer is targeting the same output root."""
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / WRITER_LOCK_NAME
    if lock_path.exists():
        try:
            existing = json.loads(lock_path.read_text())
        except Exception:
            existing = {}
        existing_pid = int(existing.get("pid", 0) or 0)
        if existing_pid and _pid_alive(existing_pid) and existing_pid != os.getpid():
            raise WriterConflictError(
                f"another writer (pid={existing_pid}) holds {lock_path}; refuse to launch duplicate run"
            )
        if not allow_force:
            raise WriterConflictError(
                f"stale writer lock at {lock_path}; rerun with --force-overwrite or remove manually"
            )
    payload = {
        "pid": int(os.getpid()),
        "acquired_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
    }
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return lock_path


def _release_writer_lock(lock_path: Path) -> None:
    try:
        if lock_path.exists():
            existing = json.loads(lock_path.read_text())
            if int(existing.get("pid", 0) or 0) == os.getpid():
                lock_path.unlink()
    except Exception:
        pass


def _refuse_overwrite_when_completed(output_root: Path, *, allow_force: bool) -> None:
    """Refuse to launch a duplicate full training run against a completed root."""
    completed = (output_root / "paper_evidence_gate.json").exists() and (
        output_root / "run_exit_status.json"
    ).exists()
    if completed and not allow_force:
        raise WriterConflictError(
            "output root already contains a completed paper-evidence bundle; "
            "rerun with --force-overwrite or use --rebuild-meta-only to refresh meta"
        )


def _capture_extended_runtime_provenance(
    *,
    log_path: Optional[Path] = None,
    tracked_pid_override: Optional[int] = None,
    launch_timestamp_override: Optional[str] = None,
    meta_rebuild: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Augment the shared runtime provenance payload with the extra fields the
    paper-evidence gate validates (git, host, GPU count, launch time, log link).

    When ``tracked_pid_override`` is provided (e.g. during ``--rebuild-meta-only``)
    the recorded ``tracked_pid``/``pid`` and ``launch_timestamp_utc`` reflect the
    original training run rather than the rebuild process. Rebuild-process
    provenance is recorded separately under ``meta_rebuild``.
    """
    base = capture_runtime_provenance()
    if tracked_pid_override is not None:
        base["pid"] = int(tracked_pid_override)
        base["tracked_pid"] = int(tracked_pid_override)
    else:
        base["pid"] = int(os.getpid())
        base["tracked_pid"] = int(os.getpid())
    base["git_sha"] = get_git_commit()
    base["git_dirty"] = get_git_dirty()
    base["hostname"] = socket.gethostname()
    base["platform"] = platform.platform()
    if launch_timestamp_override is not None:
        base["launch_timestamp_utc"] = str(launch_timestamp_override)
    else:
        base["launch_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    gpu_count = 0
    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = int(torch.cuda.device_count())
    except Exception:
        gpu_count = 0
    base["gpu_count"] = int(gpu_count)
    if log_path is not None:
        base["log_path"] = str(log_path)
    if meta_rebuild is not None:
        base["meta_rebuild"] = dict(meta_rebuild)
    return base


def _make_rows(*, dataset_id: str, operator_version: str) -> List[RowConfig]:
    return [
        RowConfig(
            row_id="hybrid_resnet",
            model="hybrid_resnet",
            training=preflight_mod.DEFAULT_TRAINING_LABEL,
            input_mode="born_init_image",
            dataset_id=dataset_id,
            operator_version=operator_version,
            paper_label="Hybrid ResNet",
        ),
        RowConfig(
            row_id="ffno",
            model="ffno",
            training=preflight_mod.DEFAULT_TRAINING_LABEL,
            input_mode="born_init_image",
            dataset_id=dataset_id,
            operator_version=operator_version,
            paper_label="FFNO",
        ),
    ]


def _validate_ffno_extension_bundle(
    ffno_extension_root: Path,
    *,
    baseline_root: Path,
) -> Dict[str, Any]:
    root = Path(ffno_extension_root).resolve()
    manifest_path = root / "preflight_manifest.json"
    metrics_path = root / "metrics.json"
    combined_path = root / "combined_metrics.json"
    if not manifest_path.exists():
        raise ext_bundle.BaselineContractMismatchError(
            f"ffno extension bundle missing preflight_manifest.json at {manifest_path}"
        )
    if not metrics_path.exists():
        raise ext_bundle.BaselineContractMismatchError(
            f"ffno extension bundle missing metrics.json at {metrics_path}"
        )
    if not combined_path.exists():
        raise ext_bundle.BaselineContractMismatchError(
            f"ffno extension bundle missing combined_metrics.json at {combined_path}"
        )
    manifest = _read_json(manifest_path)
    if manifest.get("backlog_item") != ext_bundle.EXTENSION_BACKLOG_ITEM:
        raise ext_bundle.BaselineContractMismatchError(
            "ffno extension backlog_item mismatch"
        )
    lineage = manifest.get("baseline_lineage") or {}
    declared_baseline = lineage.get("baseline_root")
    if declared_baseline is not None and Path(declared_baseline).resolve() != Path(
        baseline_root
    ).resolve():
        raise ext_bundle.BaselineContractMismatchError(
            "ffno extension baseline_root does not match the requested baseline root"
        )
    return manifest


def _baseline_rows(
    *, baseline_root: Path, ffno_extension_root: Path
) -> Dict[str, Mapping[str, Any]]:
    baseline_metrics = _read_json(Path(baseline_root) / "metrics.json")
    ffno_metrics = _read_json(Path(ffno_extension_root) / "metrics.json")
    indexed = {str(row["row_id"]): row for row in baseline_metrics.get("rows") or []}
    for row in ffno_metrics.get("rows") or []:
        indexed[str(row["row_id"])] = row
    return indexed


def _write_bundle_visuals(
    *,
    output_root: Path,
    sample_id: int,
    fixed_targets: Mapping[int, Mapping[str, np.ndarray]],
    fixed_q_pred_by_row: Mapping[int, Mapping[str, np.ndarray]],
    fixed_sino_pred_by_row: Mapping[int, Mapping[str, np.ndarray]],
    baseline_root: Path,
) -> Dict[str, Any]:
    visuals_dir = output_root / "visuals"
    arrays_dir = output_root / "figures" / "source_arrays"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir.mkdir(parents=True, exist_ok=True)
    targets = fixed_targets.get(int(sample_id))
    preds_by_row = dict(fixed_q_pred_by_row.get(int(sample_id)) or {})
    sino_preds_by_row = dict(fixed_sino_pred_by_row.get(int(sample_id)) or {})
    classical_q_path = (
        Path(baseline_root)
        / "figures"
        / "source_arrays"
        / f"sample_{int(sample_id):04d}_classical_born_backprop_q_pred.npy"
    )
    classical_sino_path = (
        Path(baseline_root)
        / "figures"
        / "source_arrays"
        / f"sample_{int(sample_id):04d}_classical_born_backprop_sino_pred.npy"
    )
    classical_present = False
    if classical_q_path.exists() and classical_sino_path.exists():
        preds_by_row["classical_born_backprop"] = np.load(classical_q_path)
        sino_preds_by_row["classical_born_backprop"] = np.load(classical_sino_path)
        np.save(
            arrays_dir / f"sample_{int(sample_id):04d}_classical_born_backprop_q_pred.npy",
            preds_by_row["classical_born_backprop"],
        )
        np.save(
            arrays_dir
            / f"sample_{int(sample_id):04d}_classical_born_backprop_sino_pred.npy",
            sino_preds_by_row["classical_born_backprop"],
        )
        classical_present = True
    figure_paths: List[str] = []
    if targets is not None and preds_by_row:
        compare_path = visuals_dir / f"sample_{int(sample_id):04d}_compare_q.png"
        error_path = visuals_dir / f"sample_{int(sample_id):04d}_error_q.png"
        sino_path = visuals_dir / f"sample_{int(sample_id):04d}_sinogram_residual.png"
        visuals_mod.render_compare_q(
            preds_by_row=preds_by_row,
            target=targets["q_target"],
            out_path=compare_path,
            sample_id=int(sample_id),
        )
        visuals_mod.render_error_q(
            preds_by_row=preds_by_row,
            target=targets["q_target"],
            out_path=error_path,
            sample_id=int(sample_id),
        )
        visuals_mod.render_sinogram_residual(
            sino_obs=targets["sino_obs"],
            sino_preds_by_row=sino_preds_by_row,
            out_path=sino_path,
            sample_id=int(sample_id),
        )
        figure_paths = [
            str(compare_path.relative_to(output_root)),
            str(error_path.relative_to(output_root)),
            str(sino_path.relative_to(output_root)),
        ]
    visual_manifest = {
        "schema_version": "brdt_paper_evidence_visuals_v1",
        "required_sample_id": int(sample_id),
        "classical_present": bool(classical_present),
        "rows_present": sorted(preds_by_row.keys()),
        "figures": figure_paths,
    }
    _write_json(output_root / "visual_manifest.json", visual_manifest)
    return {
        "classical_present": bool(classical_present),
        "figures": figure_paths,
        "rows_present": sorted(preds_by_row.keys()),
        "required_paper_sample": int(sample_id),
        "visual_manifest_path": str(output_root / "visual_manifest.json"),
    }


REQUIRED_ORIGINAL_RUNTIME_PROVENANCE_FIELDS = (
    "git_sha",
    "git_dirty",
    "hostname",
    "platform",
    "gpu_count",
    "python_executable",
    "python_version",
    "torch",
    "launch_timestamp_utc",
    "tracked_pid",
)

# When the original training-run runtime_provenance.json was lost (because an
# earlier ``--rebuild-meta-only`` invocation under prior code overwrote it
# with the rebuild host's snapshot), invocation.json — written by the original
# training process at startup — is the only preserved authoritative record of
# the original training-run launch identity. The fields below are the subset
# the original training process captured under invocation.json's
# ``extra.runtime_provenance`` block plus the top-level ``pid``/
# ``timestamp_utc``. Fields outside this set were never preserved by
# invocation.json and cannot be honestly reconstructed: the gate's
# git_provenance and host_provenance checks naturally fail on a reconstructed
# payload, demoting the bundle rather than fabricating values.
RECONSTRUCTABLE_RUNTIME_PROVENANCE_FIELDS = (
    "tracked_pid",
    "pid",
    "launch_timestamp_utc",
    "python_executable",
    "python_version",
    "torch",
)
UNRECOVERABLE_RUNTIME_PROVENANCE_FIELDS = (
    "git_sha",
    "git_dirty",
    "hostname",
    "platform",
    "gpu_count",
)


def reconstruct_runtime_provenance_from_invocation(
    *,
    output_root: Path,
) -> Path:
    """Reconstruct ``runtime_provenance.json`` from the on-disk
    ``invocation.json`` after an earlier rebuild-host overwrite.

    invocation.json was written by the original training process at startup
    and preserves the launch timestamp, the tracked PID, and the
    Python/PyTorch identity captured under
    ``extra.runtime_provenance``. It does not preserve git SHA, git-dirty
    state, hostname, platform, or GPU count — those existed only in the
    original training-run ``runtime_provenance.json`` and were lost when an
    earlier ``--rebuild-meta-only`` invocation under prior code overwrote
    the file with the rebuild host's snapshot.

    This helper restores only the fields invocation.json preserved and sets
    the unrecoverable fields to ``None`` with an explicit
    ``provenance_reconstruction`` block listing the source, the recovered
    fields, and the unrecoverable fields. The paper-evidence gate's
    ``git_provenance`` and ``host_provenance`` checks naturally fail on a
    reconstructed payload, so the bundle is honestly demoted rather than
    promoted on top of fabricated values.
    """
    invocation_path = Path(output_root) / "invocation.json"
    if not invocation_path.exists():
        raise FileNotFoundError(
            f"reconstruction requires existing {invocation_path}; cannot "
            "rebuild runtime provenance without the original training-run "
            "invocation record"
        )
    try:
        invocation_payload = json.loads(invocation_path.read_text())
    except Exception as exc:
        raise RuntimeError(
            f"reconstruction cannot parse {invocation_path}: {exc!r}"
        ) from exc
    if not isinstance(invocation_payload, dict):
        raise RuntimeError(
            f"reconstruction expected dict in {invocation_path}, got "
            f"{type(invocation_payload).__name__}"
        )
    pid = invocation_payload.get("pid")
    timestamp = invocation_payload.get("timestamp_utc")
    if pid is None or not timestamp:
        raise RuntimeError(
            f"reconstruction requires non-null pid and timestamp_utc in "
            f"{invocation_path}; got pid={pid!r}, timestamp_utc={timestamp!r}"
        )
    extra = invocation_payload.get("extra") or {}
    if not isinstance(extra, Mapping):
        extra = {}
    extra_provenance = extra.get("runtime_provenance") or {}
    if not isinstance(extra_provenance, Mapping):
        extra_provenance = {}
    payload: Dict[str, Any] = {
        "tracked_pid": int(pid),
        "pid": int(pid),
        "launch_timestamp_utc": str(timestamp),
        "python_executable": extra_provenance.get("python_executable"),
        "python_version": extra_provenance.get("python_version"),
        "torch": dict(extra_provenance.get("torch") or {}),
        "cwd": extra_provenance.get("cwd") or invocation_payload.get("cwd"),
        "pythonpath": extra_provenance.get("pythonpath"),
        "ptycho_torch_file": extra_provenance.get("ptycho_torch_file"),
        "git_sha": None,
        "git_dirty": None,
        "hostname": None,
        "platform": None,
        "gpu_count": None,
        "provenance_reconstruction": {
            "source": "invocation.json",
            "source_path": str(invocation_path),
            "reconstructed_utc": datetime.now(timezone.utc).isoformat(),
            "recovered_fields": list(RECONSTRUCTABLE_RUNTIME_PROVENANCE_FIELDS),
            "unrecoverable_fields": list(
                UNRECOVERABLE_RUNTIME_PROVENANCE_FIELDS
            ),
            "rationale": (
                "An earlier --rebuild-meta-only invocation under prior code "
                "overwrote the original training-run runtime_provenance.json "
                "with the rebuild host's snapshot. invocation.json is the "
                "only preserved authoritative record of the original "
                "training-run launch identity. Fields not preserved by "
                "invocation.json are marked unrecoverable rather than "
                "fabricated; the paper-evidence gate's git_provenance and "
                "host_provenance checks fail on this payload, demoting the "
                "bundle's claim_boundary to decision_support_convergence_"
                "followup."
            ),
        },
    }
    runtime_provenance_path = Path(output_root) / "runtime_provenance.json"
    _write_json(runtime_provenance_path, payload)
    return runtime_provenance_path


def _amend_existing_runtime_provenance_for_rebuild(
    *,
    output_root: Path,
    log_path: Optional[Path],
    meta_rebuild: Mapping[str, Any],
) -> Path:
    """Preserve the original training-run ``runtime_provenance.json`` payload
    and only attach a ``meta_rebuild`` block plus an optional refreshed
    ``log_path``.

    The reviewer flagged that the prior rebuild path called
    :func:`_capture_extended_runtime_provenance` from the rebuild process,
    which re-sampled ``git_sha``/``git_dirty``/``hostname``/``gpu_count``/
    Python/PyTorch/CUDA/host fields from the rebuild host while keeping the
    original ``launch_timestamp_utc`` and ``tracked_pid``. That left the
    top-level provenance surface a hybrid that no longer faithfully described
    the original 40-epoch training run. This helper preserves every recorded
    original field exactly and refuses to fabricate provenance when the
    original file is missing or malformed: a bundle whose original runtime
    provenance has been lost must be retrained, reconstructed via
    :func:`reconstruct_runtime_provenance_from_invocation`, or otherwise
    rebuilt from authentic original evidence — never silently regenerated
    from the rebuild host.

    When the on-disk payload carries a ``provenance_reconstruction`` block
    (written by :func:`reconstruct_runtime_provenance_from_invocation`),
    this helper recognizes the explicitly unrecoverable fields and only
    requires the reconstruction's declared ``recovered_fields`` to be
    non-null. Fields the reconstruction marks unrecoverable may pass through
    as ``None`` so the gate's ``git_provenance``/``host_provenance`` checks
    fail honestly rather than silently passing on fabricated values.
    """
    runtime_provenance_path = output_root / "runtime_provenance.json"
    if not runtime_provenance_path.exists():
        raise FileNotFoundError(
            f"meta-rebuild requires existing {runtime_provenance_path}; refusing "
            "to fabricate runtime provenance from the rebuild host"
        )
    try:
        existing_payload = json.loads(runtime_provenance_path.read_text())
    except Exception as exc:
        raise RuntimeError(
            f"meta-rebuild cannot parse {runtime_provenance_path}: {exc!r}; refusing "
            "to fabricate runtime provenance"
        ) from exc
    if not isinstance(existing_payload, dict):
        raise RuntimeError(
            f"meta-rebuild expected dict in {runtime_provenance_path}, got "
            f"{type(existing_payload).__name__}"
        )
    reconstruction_block = existing_payload.get("provenance_reconstruction")
    if isinstance(reconstruction_block, Mapping):
        recovered = list(reconstruction_block.get("recovered_fields") or [])
        if not recovered:
            raise RuntimeError(
                f"meta-rebuild found provenance_reconstruction block in "
                f"{runtime_provenance_path} with empty recovered_fields; "
                "refusing to bless a payload that recovered nothing"
            )
        missing = [
            key for key in recovered if existing_payload.get(key) is None
        ]
    else:
        missing = [
            key
            for key in REQUIRED_ORIGINAL_RUNTIME_PROVENANCE_FIELDS
            if existing_payload.get(key) is None
        ]
    if missing:
        raise RuntimeError(
            f"meta-rebuild requires {sorted(missing)} in {runtime_provenance_path}; "
            "refusing to fabricate provenance for missing fields"
        )
    payload = dict(existing_payload)
    if log_path is not None:
        payload["log_path"] = str(log_path)
    payload["meta_rebuild"] = dict(meta_rebuild)
    _write_json(runtime_provenance_path, payload)
    return runtime_provenance_path


def _write_dataset_and_split_manifests(
    *,
    output_root: Path,
    authority: DatasetAuthority,
    fixed_sample_ids: List[int],
) -> Dict[str, str]:
    """Write the dataset identity manifest and split manifest. These payloads
    are deterministically derivable from the dataset authority and locked split
    contract, so they can be regenerated on rebuild without losing original
    training-run information."""
    dataset_identity_path = output_root / "dataset_identity_manifest.json"
    split_manifest_path = output_root / "split_manifest.json"
    dataset_identity_payload: Dict[str, Any] = {
        "dataset_id": authority.dataset_id,
        "manifest_path": str(authority.manifest_path),
        "dataset_identity": authority.raw_manifest.get("dataset_identity"),
    }
    manifest_path_obj = Path(authority.manifest_path)
    if manifest_path_obj.exists():
        try:
            stat = manifest_path_obj.stat()
            dataset_identity_payload["manifest_size_bytes"] = int(stat.st_size)
            dataset_identity_payload["manifest_mtime_utc"] = (
                datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            )
        except Exception:
            pass
    if not dataset_identity_payload.get("dataset_identity", {}) or not (
        dataset_identity_payload["dataset_identity"] or {}
    ).get("checksum"):
        dataset_identity_payload["checksum_exception"] = (
            "decision_support dataset manifest does not record a checksum; "
            "size/mtime/source-rationale recorded as reviewed exception"
        )
    _write_json(dataset_identity_path, dataset_identity_payload)
    _write_json(
        split_manifest_path,
        {
            "split_counts": authority.raw_manifest.get("split", {}).get("counts"),
            "fixed_sample_ids": [int(i) for i in fixed_sample_ids],
        },
    )
    return {
        "dataset_identity_manifest_path": str(dataset_identity_path),
        "split_manifest_path": str(split_manifest_path),
    }


def _write_top_level_provenance(
    *,
    output_root: Path,
    authority: DatasetAuthority,
    fixed_sample_ids: List[int],
    log_path: Optional[Path] = None,
    tracked_pid_override: Optional[int] = None,
    launch_timestamp_override: Optional[str] = None,
    meta_rebuild: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    runtime_provenance_path = output_root / "runtime_provenance.json"
    dataset_identity_path = output_root / "dataset_identity_manifest.json"
    split_manifest_path = output_root / "split_manifest.json"
    _write_json(
        runtime_provenance_path,
        _capture_extended_runtime_provenance(
            log_path=log_path,
            tracked_pid_override=tracked_pid_override,
            launch_timestamp_override=launch_timestamp_override,
            meta_rebuild=meta_rebuild,
        ),
    )
    dataset_identity_payload: Dict[str, Any] = {
        "dataset_id": authority.dataset_id,
        "manifest_path": str(authority.manifest_path),
        "dataset_identity": authority.raw_manifest.get("dataset_identity"),
    }
    manifest_path_obj = Path(authority.manifest_path)
    if manifest_path_obj.exists():
        try:
            stat = manifest_path_obj.stat()
            dataset_identity_payload["manifest_size_bytes"] = int(stat.st_size)
            dataset_identity_payload["manifest_mtime_utc"] = (
                datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            )
        except Exception:
            pass
    if not dataset_identity_payload.get("dataset_identity", {}) or not (
        dataset_identity_payload["dataset_identity"] or {}
    ).get("checksum"):
        dataset_identity_payload["checksum_exception"] = (
            "decision_support dataset manifest does not record a checksum; "
            "size/mtime/source-rationale recorded as reviewed exception"
        )
    _write_json(dataset_identity_path, dataset_identity_payload)
    _write_json(
        split_manifest_path,
        {
            "split_counts": authority.raw_manifest.get("split", {}).get("counts"),
            "fixed_sample_ids": [int(i) for i in fixed_sample_ids],
        },
    )
    return {
        "runtime_provenance_path": str(runtime_provenance_path),
        "dataset_identity_manifest_path": str(dataset_identity_path),
        "split_manifest_path": str(split_manifest_path),
    }


def _resolve_log_path(output_root: Path) -> Optional[Path]:
    """Return the most recently modified file under output_root/logs/ if any."""
    logs_dir = output_root / "logs"
    if not logs_dir.exists():
        return None
    candidates = [p for p in logs_dir.iterdir() if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _write_run_exit_status(
    output_root: Path,
    *,
    pid: int,
    exit_code: int,
    status: str,
    log_path: Optional[Path],
) -> Path:
    payload: Dict[str, Any] = {
        "pid": int(pid),
        "tracked_pid": int(pid),
        "exit_code": int(exit_code),
        "status": str(status),
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "log_path": str(log_path) if log_path is not None else None,
    }
    out_path = output_root / "run_exit_status.json"
    _write_json(out_path, payload)
    return out_path


DOCS_INDEX_PATH = "docs/index.md"


CANONICAL_ARTIFACT_ROOT = (
    f".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/{BACKLOG_ITEM}"
)


def _manifest_entry_for_backlog(
    manifest_payload: Mapping[str, Any], backlog_item: str
) -> Optional[Dict[str, Any]]:
    """Walk ``paper_evidence_manifest.json`` looking for the structured entry
    whose ``source_root`` (or equivalent pointer) belongs to ``backlog_item``.
    Returns the first dict containing both ``claim_boundary`` and a
    ``source_root`` referencing the backlog item.
    """
    matches: List[Dict[str, Any]] = []

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            source_root = str(obj.get("source_root") or "")
            if (
                "claim_boundary" in obj
                and backlog_item in source_root
            ):
                matches.append(dict(obj))
            for value in obj.values():
                _walk(value)
        elif isinstance(obj, list):
            for value in obj:
                _walk(value)

    _walk(manifest_payload)
    return matches[0] if matches else None


def _check_evidence_surfaces_consistent(
    *,
    repo_root: Path = Path.cwd(),
    output_root: Optional[Path] = None,
) -> bool:
    """Return True only when every required discoverability surface references
    this backlog item, the canonical artifact root, AND the same single
    claim-boundary label that the structured paper-evidence manifest records
    for this backlog item.

    Required surfaces (matching the plan's ``Task 6`` discoverability contract):

    - durable summary at :data:`DURABLE_SUMMARY_PATH`
    - paper-evidence index at :data:`PAPER_EVIDENCE_INDEX_PATH`
    - paper-evidence manifest at :data:`PAPER_EVIDENCE_MANIFEST_PATH`
    - repo-wide docs index at :data:`DOCS_INDEX_PATH`

    The reviewer flagged that an earlier substring-only check could pass even
    when one surface referenced a stale claim boundary or a wrong artifact
    root. This stricter check uses the structured manifest entry (the only
    machine-readable surface) as the authoritative ``claim_boundary`` and
    ``source_root`` for the bundle, then enforces:

    1. The manifest contains a structured entry for this backlog item with
       both ``claim_boundary`` and ``source_root`` fields.
    2. The manifest's ``source_root`` matches the canonical artifact root for
       this backlog item (or the supplied ``output_root`` resolved to the
       repo-relative form).
    3. Every surface mentions :data:`BACKLOG_ITEM`, the canonical artifact
       root, and the manifest's authoritative claim-boundary string.
    """
    repo_root = Path(repo_root)
    summary_path = repo_root / DURABLE_SUMMARY_PATH
    index_path = repo_root / PAPER_EVIDENCE_INDEX_PATH
    manifest_path = repo_root / PAPER_EVIDENCE_MANIFEST_PATH
    docs_index_path = repo_root / DOCS_INDEX_PATH
    package_design_path = repo_root / PAPER_EVIDENCE_PACKAGE_DESIGN_PATH

    if output_root is not None:
        try:
            relative_root = (
                Path(output_root).resolve().relative_to(repo_root.resolve())
            )
            artifact_root_token = str(relative_root).rstrip("/")
        except Exception:
            artifact_root_token = CANONICAL_ARTIFACT_ROOT
    else:
        artifact_root_token = CANONICAL_ARTIFACT_ROOT

    if not manifest_path.exists():
        return False
    try:
        manifest_payload = json.loads(manifest_path.read_text())
    except Exception:
        return False
    manifest_entry = _manifest_entry_for_backlog(manifest_payload, BACKLOG_ITEM)
    if manifest_entry is None:
        return False
    manifest_boundary = str(manifest_entry.get("claim_boundary") or "")
    manifest_source_root = str(manifest_entry.get("source_root") or "").rstrip("/")
    if manifest_boundary not in KNOWN_CLAIM_BOUNDARIES:
        return False
    if not manifest_source_root.startswith(artifact_root_token):
        return False

    # The paper-evidence package design must carry a checked-in evidence
    # amendment consistent with the gate result whenever the manifest reports
    # the promoted boundary. The plan ties promotion to the presence of that
    # amendment ("the checked-in evidence amendment is prepared and consistent
    # with the gate result"), so a passed manifest with no design-doc reference
    # must fail the gate. When the manifest still records the pre-gate
    # boundary, no amendment is required (the plan only authorizes amending
    # the design doc on a passed gate).
    required_paths = [summary_path, index_path, manifest_path, docs_index_path]
    if manifest_boundary == PASSED_CLAIM_BOUNDARY:
        required_paths.append(package_design_path)

    for path in required_paths:
        if not path.exists():
            return False
        try:
            text = path.read_text()
        except Exception:
            return False
        if BACKLOG_ITEM not in text:
            return False
        if artifact_root_token not in text:
            return False
        if manifest_boundary not in text:
            return False
    return True


_SCHEDULER_KEY_MAP = (
    ("plateau_factor", "factor"),
    ("plateau_patience", "patience"),
    ("plateau_threshold", "threshold"),
    ("plateau_min_lr", "min_lr"),
)


def _values_close(a: Any, b: Any) -> bool:
    try:
        return abs(float(a) - float(b)) <= 1e-9
    except (TypeError, ValueError):
        return False


def _scheduler_matches_contract(
    *,
    row_summary: Mapping[str, Any],
    contract_dict: Mapping[str, Any],
) -> bool:
    """Return True only when the row's recorded scheduler/optimizer state
    matches every plan-bound field of ``contract_dict``.

    The reviewer flagged that the prior implementation only checked the
    scheduler name. The conservative gate must also verify ``plateau_factor``,
    ``plateau_patience``, ``plateau_threshold``, ``plateau_min_lr``, and the
    surrounding optimizer recipe (``epochs``, ``batch_size``, ``learning_rate``)
    because a bundle with drifted plateau settings would otherwise still pass.
    """
    contract_scheduler_name = contract_dict.get("scheduler")
    sched = dict(row_summary.get("scheduler") or {})
    if contract_scheduler_name is None:
        return not sched
    if str(sched.get("name") or "") != str(contract_scheduler_name):
        return False
    for contract_key, sched_key in _SCHEDULER_KEY_MAP:
        contract_val = contract_dict.get(contract_key)
        sched_val = sched.get(sched_key)
        if contract_val is None or sched_val is None:
            return False
        if not _values_close(contract_val, sched_val):
            return False
    runtime = dict(row_summary.get("runtime") or {})
    if int(runtime.get("epochs", -1)) != int(contract_dict.get("epochs", -2)):
        return False
    if int(runtime.get("batch_size", -1)) != int(contract_dict.get("batch_size", -2)):
        return False
    if not _values_close(
        runtime.get("learning_rate"), contract_dict.get("learning_rate")
    ):
        return False
    return True


def _check_sample_visual_source_arrays(
    *, output_root: Path, sample_id: int
) -> bool:
    """Return True only when every required source-array file for the
    paper-facing comparison panel exists on disk.

    The reviewer flagged that ``classical_present`` plus a non-empty figure
    list does not actually prove the panel can be reproduced from source
    arrays. The bundle's per-row ``q_pred``/``sino_pred`` and per-sample
    ``q_target``/``sino_obs`` files are the durable source-of-truth, so this
    check now requires each of them.
    """
    arrays_dir = Path(output_root) / "figures" / "source_arrays"
    if not arrays_dir.exists():
        return False
    sid = int(sample_id)
    required: List[Path] = [
        arrays_dir / f"sample_{sid:04d}_q_target.npy",
        arrays_dir / f"sample_{sid:04d}_sino_obs.npy",
        arrays_dir / f"sample_{sid:04d}_classical_born_backprop_q_pred.npy",
        arrays_dir / f"sample_{sid:04d}_classical_born_backprop_sino_pred.npy",
    ]
    for row_id in ("hybrid_resnet", "ffno"):
        required.extend(
            [
                arrays_dir / f"sample_{sid:04d}_{row_id}_q_pred.npy",
                arrays_dir / f"sample_{sid:04d}_{row_id}_sino_pred.npy",
            ]
        )
    return all(path.exists() for path in required)


_LINEAGE_TRAINING_FIELDS = (
    "batch_size",
    "learning_rate",
    "optimizer",
    "seed",
)


def _normalize_split_counts(payload: Any) -> Dict[str, int]:
    if not isinstance(payload, Mapping):
        return {}
    return {str(k): int(v) for k, v in payload.items()}


def _normalize_geometry(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    return {str(k): payload[k] for k in payload}


def _normalize_loss_weights(payload: Any) -> Dict[str, float]:
    if not isinstance(payload, Mapping):
        return {}
    out: Dict[str, float] = {}
    for k, v in payload.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            return {}
    return out


def _check_same_contract_lineage(
    *,
    output_root: Path,
    baseline_root: Path,
    ffno_extension_root: Path,
) -> bool:
    """Re-verify that the baseline and FFNO-extension lineage roots still
    satisfy the locked contract and that the current bundle's frozen
    invariants (split counts, fixed-sample roster, operator geometry,
    normalization, input mode, training-contract fields) match the lineage.

    The reviewer flagged that the previous implementation only re-checked
    lineage-root pointers and ``dataset_id``; that left split-count drift,
    fixed-sample drift, operator drift, normalization drift, input-mode
    drift, and training-contract drift undetected by the gate.

    Returns ``False`` rather than raising so the gate records a failed check
    instead of crashing the rebuild path.
    """
    try:
        ext_bundle.validate_baseline_bundle(Path(baseline_root))
        _validate_ffno_extension_bundle(
            Path(ffno_extension_root), baseline_root=Path(baseline_root)
        )
    except Exception:
        return False
    current_manifest_path = Path(output_root) / "preflight_manifest.json"
    if not current_manifest_path.exists():
        return False
    try:
        current_manifest = json.loads(current_manifest_path.read_text())
        baseline_manifest = json.loads(
            (Path(baseline_root) / "preflight_manifest.json").read_text()
        )
        ffno_manifest = json.loads(
            (Path(ffno_extension_root) / "preflight_manifest.json").read_text()
        )
    except Exception:
        return False
    lineage = current_manifest.get("baseline_lineage") or {}
    declared_baseline = lineage.get("baseline_root")
    declared_ffno = lineage.get("ffno_extension_root")
    if not declared_baseline or not declared_ffno:
        return False
    if Path(declared_baseline).resolve() != Path(baseline_root).resolve():
        return False
    if Path(declared_ffno).resolve() != Path(ffno_extension_root).resolve():
        return False
    cur_dataset_block = current_manifest.get("dataset") or {}
    base_dataset_block = baseline_manifest.get("dataset") or {}
    ffno_dataset_block = ffno_manifest.get("dataset") or {}
    cur_dataset_id = cur_dataset_block.get("dataset_id")
    if (
        not cur_dataset_id
        or cur_dataset_id != base_dataset_block.get("dataset_id")
        or cur_dataset_id != ffno_dataset_block.get("dataset_id")
    ):
        return False
    cur_splits = _normalize_split_counts(cur_dataset_block.get("split_counts"))
    base_splits = _normalize_split_counts(base_dataset_block.get("split_counts"))
    if not cur_splits or cur_splits != base_splits:
        return False
    cur_norm = cur_dataset_block.get("normalization") or {}
    base_norm = base_dataset_block.get("normalization") or {}
    if not isinstance(cur_norm, Mapping) or not isinstance(base_norm, Mapping):
        return False
    if set(cur_norm.keys()) != set(base_norm.keys()):
        return False
    for key in cur_norm.keys():
        if not _values_close(cur_norm[key], base_norm[key]):
            return False
    cur_geom = _normalize_geometry(
        (current_manifest.get("operator") or {}).get("geometry")
    )
    base_geom = _normalize_geometry(
        (baseline_manifest.get("operator") or {}).get("geometry")
    )
    if not cur_geom or cur_geom != base_geom:
        return False
    cur_fixed = [int(i) for i in current_manifest.get("fixed_sample_ids") or []]
    base_fixed = [int(i) for i in baseline_manifest.get("fixed_sample_ids") or []]
    if not cur_fixed or cur_fixed != base_fixed:
        return False
    base_input_mode = (
        baseline_manifest.get("input_contract") or {}
    ).get("input_mode")
    if base_input_mode is not None:
        rows = current_manifest.get("rows") or []
        if not rows:
            return False
        for row in rows:
            if str(row.get("input_mode")) != str(base_input_mode):
                return False
    cur_training = current_manifest.get("training_contract") or {}
    base_training = baseline_manifest.get("training_contract") or {}
    for key in _LINEAGE_TRAINING_FIELDS:
        cur_val = cur_training.get(key)
        base_val = base_training.get(key)
        if cur_val is None or base_val is None:
            return False
        if isinstance(cur_val, str) or isinstance(base_val, str):
            if str(cur_val) != str(base_val):
                return False
        elif not _values_close(cur_val, base_val):
            return False
    cur_lw = _normalize_loss_weights(cur_training.get("loss_weights"))
    base_lw = _normalize_loss_weights(base_training.get("loss_weights"))
    if not cur_lw or set(cur_lw.keys()) != set(base_lw.keys()):
        return False
    for key in cur_lw.keys():
        if not _values_close(cur_lw[key], base_lw[key]):
            return False
    return True


def _build_provenance_checks(
    *,
    output_root: Path,
    provenance_paths: Mapping[str, str],
    visual_status: Mapping[str, Any],
    rows: Mapping[str, Mapping[str, Any]],
    log_path: Optional[Path],
    baseline_root: Optional[Path] = None,
    ffno_extension_root: Optional[Path] = None,
) -> Dict[str, bool]:
    runtime_provenance_payload: Dict[str, Any] = {}
    runtime_provenance_path = Path(provenance_paths["runtime_provenance_path"])
    if runtime_provenance_path.exists():
        try:
            runtime_provenance_payload = json.loads(
                runtime_provenance_path.read_text()
            )
        except Exception:
            runtime_provenance_payload = {}
    git_provenance_present = bool(
        runtime_provenance_payload.get("git_sha")
    ) and runtime_provenance_payload.get("git_dirty") is not None
    host_provenance_present = bool(runtime_provenance_payload.get("hostname")) and (
        runtime_provenance_payload.get("gpu_count") is not None
    )
    # Plan Task 4 promotion prerequisites: "repo git SHA/dirty state, Python/
    # PyTorch/CUDA/GPU/host provenance". The reviewer flagged that the prior
    # gate accepted bundles whose runtime_provenance.json was missing
    # Python/PyTorch/CUDA fields entirely, so add explicit checks for the
    # interpreter and torch identity fields the plan calls out.
    python_provenance_present = bool(
        runtime_provenance_payload.get("python_executable")
    ) and bool(runtime_provenance_payload.get("python_version"))
    torch_payload = runtime_provenance_payload.get("torch") or {}
    torch_provenance_present = (
        isinstance(torch_payload, Mapping)
        and bool(torch_payload.get("version"))
        and torch_payload.get("cuda_available") is not None
        and bool(torch_payload.get("cuda_version"))
    )
    model_profiles_present = all(
        (output_root / "rows" / row_id / "model_profile.json").exists()
        for row_id in rows
    )
    run_log_present = log_path is not None and Path(log_path).exists()
    if baseline_root is not None and ffno_extension_root is not None:
        same_contract_lineage = _check_same_contract_lineage(
            output_root=output_root,
            baseline_root=Path(baseline_root),
            ffno_extension_root=Path(ffno_extension_root),
        )
    else:
        same_contract_lineage = False
    exit_status_path = output_root / "run_exit_status.json"
    exit_code_proof = False
    if exit_status_path.exists():
        try:
            exit_payload = json.loads(exit_status_path.read_text())
        except Exception:
            exit_payload = {}
        runtime_tracked = _coerce_int(runtime_provenance_payload.get("tracked_pid"))
        exit_tracked = _coerce_int(exit_payload.get("tracked_pid"))
        pids_agree = (
            runtime_tracked is not None
            and exit_tracked is not None
            and runtime_tracked == exit_tracked
        )
        # exit_code_proof requires PID agreement, exit_code==0, and an explicit
        # status=="completed" record so the gate cannot bless a partial run.
        exit_code_proof = bool(
            pids_agree
            and _coerce_int(exit_payload.get("exit_code")) == 0
            and str(exit_payload.get("status") or "") == "completed"
        )
    required_paper_sample = _coerce_int(visual_status.get("required_paper_sample"))
    if required_paper_sample is None:
        try:
            manifest_payload = json.loads(
                (Path(output_root) / "preflight_manifest.json").read_text()
            )
            required_paper_sample = _coerce_int(
                manifest_payload.get("required_paper_sample")
            )
        except Exception:
            required_paper_sample = None
    sample_bundle_ok = bool(
        visual_status.get("classical_present")
    ) and bool(visual_status.get("figures"))
    if required_paper_sample is not None:
        sample_bundle_ok = sample_bundle_ok and _check_sample_visual_source_arrays(
            output_root=Path(output_root), sample_id=int(required_paper_sample)
        )
    else:
        sample_bundle_ok = False
    return {
        "runtime_provenance": runtime_provenance_path.exists(),
        "git_provenance": bool(git_provenance_present),
        "host_provenance": bool(host_provenance_present),
        "python_provenance": bool(python_provenance_present),
        "torch_provenance": bool(torch_provenance_present),
        "dataset_identity": Path(
            provenance_paths["dataset_identity_manifest_path"]
        ).exists(),
        "split_manifest": Path(provenance_paths["split_manifest_path"]).exists(),
        "model_profiles": bool(model_profiles_present),
        "run_log_present": bool(run_log_present),
        "sample_255_visual_bundle": bool(sample_bundle_ok),
        "exit_code_proof": bool(exit_code_proof),
        "evidence_surfaces_prepared": _check_evidence_surfaces_consistent(
            output_root=Path(output_root)
        ),
        "same_contract_lineage": bool(same_contract_lineage),
    }


def _reseed_top_level_manifest_with_gate(
    manifest_path: Path,
    *,
    gate_payload: Mapping[str, Any],
    paper_evidence_gate_path: Path,
) -> None:
    if not manifest_path.exists():
        return
    payload = json.loads(manifest_path.read_text())
    payload["claim_boundary"] = str(gate_payload.get("claim_boundary"))
    payload["promotion_status"] = str(gate_payload.get("promotion_status"))
    payload["paper_evidence_gate_path"] = str(paper_evidence_gate_path)
    _write_json(manifest_path, payload)


def _reseed_metrics_with_gate(
    *, output_root: Path, gate_payload: Mapping[str, Any]
) -> None:
    """Reseed ``metrics.json``, ``combined_metrics.json``, and
    ``metric_schema.json`` so their ``claim_boundary`` matches the bundle's
    final gate verdict.

    The reviewer flagged that the live training path stamps these files with
    the pre-gate label (``decision_support_convergence_followup``) before the
    gate runs, leaving the bundle internally inconsistent when promotion
    succeeds: the gate and top-level manifest say
    ``paper_evidence_brdt_additive`` while the metric tables still advertise
    the pre-gate label. Reseed them post-gate so every machine-consumed
    artifact in the bundle agrees on a single boundary.
    """
    final_boundary = str(gate_payload.get("claim_boundary") or "")
    if not final_boundary:
        return
    for filename in ("metrics.json", "combined_metrics.json", "metric_schema.json"):
        path = output_root / filename
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if str(payload.get("claim_boundary") or "") == final_boundary:
            continue
        payload["claim_boundary"] = final_boundary
        _write_json(path, payload)


def _build_manifest(
    *,
    output_root: Path,
    baseline_root: Path,
    ffno_extension_root: Path,
    authority: DatasetAuthority,
    rows: List[RowConfig],
    fixed_sample_ids: List[int],
    required_paper_sample: int,
    contract: preflight_mod.TrainingContract,
) -> Dict[str, Any]:
    return {
        "schema_version": "brdt_paper_evidence_runner_v1",
        "backlog_item": BACKLOG_ITEM,
        "claim_boundary": PRE_GATE_CLAIM_BOUNDARY,
        "promotion_status": "pending",
        "output_root": str(output_root),
        "baseline_lineage": {
            "baseline_root": str(Path(baseline_root).resolve()),
            "ffno_extension_root": str(Path(ffno_extension_root).resolve()),
        },
        "dataset": {
            "dataset_id": authority.dataset_id,
            "tier": authority.raw_manifest["dataset_identity"].get("tier"),
            "manifest_path": str(authority.manifest_path),
            "split_counts": authority.raw_manifest["split"]["counts"],
            "normalization": authority.normalization.as_dict(),
        },
        "operator": {
            "validation_artifact": authority.raw_manifest.get("operator", {}).get(
                "validation_artifact"
            ),
            "geometry": {
                "grid_size": dc.LOCKED_GRID_SIZE,
                "detector_size": dc.LOCKED_DETECTOR_SIZE,
                "angle_count": dc.LOCKED_ANGLE_COUNT,
                "wavelength_px": dc.LOCKED_WAVELENGTH_PX,
                "medium_ri": dc.LOCKED_MEDIUM_RI,
                "mode": dc.LOCKED_OPERATOR_MODE,
                "normalize": dc.LOCKED_NORMALIZE,
            },
        },
        "training_contract": contract.as_dict(),
        "fixed_sample_ids": [int(i) for i in fixed_sample_ids],
        "required_paper_sample": int(required_paper_sample),
        "rows": [row.to_dict() for row in rows],
    }


def run_paper_evidence(
    *,
    baseline_root: Path,
    ffno_extension_root: Path,
    manifest_path: Path,
    output_root: Path,
    contract: Optional[preflight_mod.TrainingContract] = None,
    device_choice: str = "auto",
    dry_run: bool = False,
    fixed_sample_ids: Optional[List[int]] = None,
    required_paper_sample: int = 255,
    parent_argv: Optional[List[str]] = None,
    force_overwrite: bool = False,
) -> Dict[str, Any]:
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    baseline_root = Path(baseline_root).resolve()
    ffno_extension_root = Path(ffno_extension_root).resolve()
    parent_argv = list(parent_argv) if parent_argv is not None else []
    if not dry_run:
        _refuse_overwrite_when_completed(output_root, allow_force=force_overwrite)
    lock_path = _acquire_writer_lock(output_root, allow_force=force_overwrite)
    try:
        return _run_paper_evidence_inner(
            baseline_root=baseline_root,
            ffno_extension_root=ffno_extension_root,
            manifest_path=manifest_path,
            output_root=output_root,
            contract=contract,
            device_choice=device_choice,
            dry_run=dry_run,
            fixed_sample_ids=fixed_sample_ids,
            required_paper_sample=required_paper_sample,
            parent_argv=parent_argv,
        )
    finally:
        _release_writer_lock(lock_path)


def _run_paper_evidence_inner(
    *,
    baseline_root: Path,
    ffno_extension_root: Path,
    manifest_path: Path,
    output_root: Path,
    contract: Optional[preflight_mod.TrainingContract],
    device_choice: str,
    dry_run: bool,
    fixed_sample_ids: Optional[List[int]],
    required_paper_sample: int,
    parent_argv: List[str],
) -> Dict[str, Any]:
    baseline_manifest = ext_bundle.validate_baseline_bundle(baseline_root)
    _validate_ffno_extension_bundle(ffno_extension_root, baseline_root=baseline_root)
    authority = load_dataset_authority(manifest_path)
    preflight_mod.assert_decision_support_manifest(authority.raw_manifest)
    contract = contract or _default_contract()
    fixed_sample_ids = (
        [int(i) for i in fixed_sample_ids]
        if fixed_sample_ids is not None
        else [int(i) for i in baseline_manifest.get("fixed_sample_ids") or []]
    )
    if int(required_paper_sample) not in set(int(i) for i in fixed_sample_ids):
        raise ValueError(
            "required_paper_sample must be included in fixed_sample_ids for the paper-facing bundle"
        )
    operator_pointer = (
        authority.raw_manifest.get("operator", {}).get("validation_artifact")
        or authority.raw_manifest.get("operator", {}).get("validation_report")
        or "unspecified"
    )
    rows = _make_rows(dataset_id=str(authority.dataset_id), operator_version=str(operator_pointer))
    manifest_payload = _build_manifest(
        output_root=output_root,
        baseline_root=baseline_root,
        ffno_extension_root=ffno_extension_root,
        authority=authority,
        rows=rows,
        fixed_sample_ids=fixed_sample_ids,
        required_paper_sample=int(required_paper_sample),
        contract=contract,
    )
    preflight_manifest_path = output_root / "preflight_manifest.json"
    _write_json(preflight_manifest_path, manifest_payload)
    write_invocation_artifacts(
        output_dir=output_root,
        script_path=SCRIPT_PATH,
        argv=parent_argv,
        parsed_args={
            "baseline_root": str(baseline_root),
            "ffno_extension_root": str(ffno_extension_root),
            "manifest_path": str(manifest_path),
            "output_root": str(output_root),
            "device": str(device_choice),
            "dry_run": bool(dry_run),
            "fixed_sample_ids": [int(i) for i in fixed_sample_ids],
            "required_paper_sample": int(required_paper_sample),
            "training_contract": contract.as_dict(),
        },
        extra={
            "backlog_item": BACKLOG_ITEM,
            "claim_boundary": PRE_GATE_CLAIM_BOUNDARY,
            "runtime_provenance": capture_runtime_provenance(),
        },
    )
    log_path = _resolve_log_path(output_root)
    provenance_paths = _write_top_level_provenance(
        output_root=output_root,
        authority=authority,
        fixed_sample_ids=fixed_sample_ids,
        log_path=log_path,
    )
    if dry_run:
        metrics_mod.write_metric_schema(
            output_root / "metric_schema.json",
            claim_boundary=PRE_GATE_CLAIM_BOUNDARY,
        )
        return {
            "dry_run": True,
            "preflight_manifest_path": str(preflight_manifest_path),
            "metric_schema_path": str(output_root / "metric_schema.json"),
            **provenance_paths,
        }

    device = preflight_mod._select_device(device_choice)
    operator = preflight_mod._build_operator(device)
    input_backend = preflight_mod._born_init_backend()
    source_arrays_dir = output_root / "figures" / "source_arrays"
    fixed_targets: Dict[int, Dict[str, np.ndarray]] = {}
    fixed_q_pred_by_row: Dict[int, Dict[str, np.ndarray]] = {
        int(sid): {} for sid in fixed_sample_ids
    }
    fixed_sino_pred_by_row: Dict[int, Dict[str, np.ndarray]] = {
        int(sid): {} for sid in fixed_sample_ids
    }
    row_metrics: List[metrics_mod.RowMetrics] = []
    current_rows: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        row_dir = output_root / "rows" / row.row_id
        row_dir.mkdir(parents=True, exist_ok=True)
        row_fp = preflight_mod.row_contract_fingerprint(
            row_id=row.row_id,
            model=row.model,
            training=row.training,
            input_mode=row.input_mode,
            dataset_id=str(authority.dataset_id),
            operator_pointer=str(operator_pointer),
            training_contract=contract.as_dict(),
            fixed_sample_ids=fixed_sample_ids,
            in_channels=1,
            classical_backend_name=f"{input_backend.name}:born_init_image",
        )
        preflight_mod._write_row_invocation_artifacts(
            row_dir=row_dir,
            row=row,
            contract=contract,
            fixed_sample_ids=fixed_sample_ids,
            in_channels=1,
            device=device,
            classical_backend=input_backend,
            contract_fingerprint=row_fp,
            parent_argv=parent_argv,
            backlog_item=BACKLOG_ITEM,
        )
        module, runtime_meta, train_seconds = preflight_mod._train_neural_row(
            row=row,
            authority=authority,
            operator=operator,
            backend=input_backend,
            device=device,
            contract=contract,
            in_channels=1,
            output_dir=row_dir,
        )
        if module is None:
            raise ValueError(f"failed to build adapter for row {row.row_id}")
        eval_started = time.perf_counter()
        image_metrics, meas_metrics, sample_arrays, output_dynamic_range = (
            preflight_mod._evaluate_split(
                module=module,
                authority=authority,
                operator=operator,
                backend=input_backend,
                device=device,
                in_channels=1,
                classical_only=False,
                fixed_sample_ids=fixed_sample_ids,
                out_dir=row_dir,
            )
        )
        eval_seconds = time.perf_counter() - eval_started
        preflight_mod._save_fixed_sample_arrays(
            source_arrays_dir=source_arrays_dir,
            sample_arrays=sample_arrays,
            row_id=row.row_id,
        )
        for sid, arrays in sample_arrays.items():
            fixed_q_pred_by_row[int(sid)][row.row_id] = arrays["q_pred"]
            fixed_sino_pred_by_row[int(sid)][row.row_id] = arrays["sino_pred"]
            fixed_targets.setdefault(
                int(sid),
                {
                    "q_target": arrays["q_target"],
                    "sino_obs": arrays["sino_obs"],
                },
            )
        runtime_block = metrics_mod.collect_runtime_metadata(
            device=str(device),
            device_name=preflight_mod._device_name(device),
            epochs=int(contract.epochs),
            batch_size=int(contract.batch_size),
            learning_rate=float(contract.learning_rate),
            parameter_count=int(runtime_meta.get("parameter_count", 0)),
            wall_time_train_s=float(train_seconds),
            wall_time_eval_s=float(eval_seconds),
            row_status="completed",
            extras={
                "train_steps": runtime_meta.get("train_steps"),
                "final_loss_breakdown": runtime_meta.get("final_loss_breakdown"),
                "model_state_path": runtime_meta.get("model_state_path"),
                "history_json_path": runtime_meta.get("history_json_path"),
                "history_csv_path": runtime_meta.get("history_csv_path"),
                "history_length": runtime_meta.get("history_length"),
                "scheduler": runtime_meta.get("scheduler"),
                "output_dynamic_range": output_dynamic_range,
            },
        )
        row_metric = metrics_mod.RowMetrics(
            row_id=row.row_id,
            paper_label=row.visible_label,
            architecture=row.model,
            row_status="completed",
            image={k: v for k, v in image_metrics.items() if k in metrics_mod.IMAGE_METRICS},
            measurement=meas_metrics,
            supporting={
                k: v for k, v in image_metrics.items() if k in metrics_mod.SUPPORTING_METRICS
            },
            runtime=runtime_block,
        )
        row_metrics.append(row_metric)
        model_profile_path = row_dir / "model_profile.json"
        _write_json(
            model_profile_path,
            {
                "row_id": row.row_id,
                "architecture": row.model,
                "parameter_count": int(runtime_meta.get("parameter_count", 0)),
                "arch_kwargs": runtime_meta.get("arch_kwargs") or {},
            },
        )
        row_summary = {
            "row_id": row.row_id,
            "row_status": "completed",
            "paper_label": row.visible_label,
            "architecture": row.model,
            "execution_path": "paper_evidence_train_eval",
            "image_metrics": {
                k: v for k, v in image_metrics.items() if k in metrics_mod.IMAGE_METRICS
            },
            "measurement_metrics": meas_metrics,
            "supporting": {
                k: v for k, v in image_metrics.items() if k in metrics_mod.SUPPORTING_METRICS
            },
            "model_state_path": runtime_meta.get("model_state_path"),
            "model_profile_path": str(model_profile_path),
            "history_json_path": runtime_meta.get("history_json_path"),
            "history_csv_path": runtime_meta.get("history_csv_path"),
            "history_length": runtime_meta.get("history_length"),
            "scheduler": runtime_meta.get("scheduler"),
            "runtime": runtime_block,
            "contract_fingerprint": row_fp,
            "output_dynamic_range": output_dynamic_range,
        }
        _write_json(row_dir / ROW_SUMMARY_NAME, row_summary)
        history_payload = conv_mod.load_history(Path(runtime_meta["history_json_path"]))
        current_rows[row.row_id] = {
            "row_summary": row_summary,
            "history_summary": conv_mod.summarize_history(
                row_id=row.row_id,
                history_payload=history_payload,
            ),
        }

    metrics_mod.write_metric_schema(
        output_root / "metric_schema.json",
        claim_boundary=PRE_GATE_CLAIM_BOUNDARY,
    )
    metrics_mod.write_metrics_json(
        output_root / "metrics.json",
        row_metrics,
        claim_boundary=PRE_GATE_CLAIM_BOUNDARY,
    )
    metrics_mod.write_metrics_csv(output_root / "metrics.csv", row_metrics)
    metrics_json_payload = _read_json(output_root / "metrics.json")
    _write_json(output_root / "combined_metrics.json", metrics_json_payload)
    metrics_mod.write_metrics_csv(output_root / "combined_metrics.csv", row_metrics)

    visual_status = _write_bundle_visuals(
        output_root=output_root,
        sample_id=int(required_paper_sample),
        fixed_targets=fixed_targets,
        fixed_q_pred_by_row=fixed_q_pred_by_row,
        fixed_sino_pred_by_row=fixed_sino_pred_by_row,
        baseline_root=baseline_root,
    )
    baseline_rows = _baseline_rows(
        baseline_root=baseline_root,
        ffno_extension_root=ffno_extension_root,
    )
    audit_payload = conv_mod.build_convergence_audit(
        backlog_item=BACKLOG_ITEM,
        baseline_rows={
            row_id: {
                "image_metrics": row.get("image"),
                "measurement_metrics": row.get("measurement"),
                "supporting": row.get("supporting"),
                "runtime": row.get("runtime"),
            }
            for row_id, row in baseline_rows.items()
            if row_id in current_rows
        },
        current_rows=current_rows,
    )
    conv_mod.write_convergence_audit_json(
        output_root / "convergence_audit.json", audit_payload
    )
    conv_mod.write_convergence_audit_csv(
        output_root / "convergence_audit.csv", audit_payload
    )

    contract_dict = contract.as_dict()
    gate_rows = {
        row_id: {
            "row_status": data["row_summary"]["row_status"],
            "history_records": data["history_summary"]["history_records"],
            "scheduler_matches_contract": _scheduler_matches_contract(
                row_summary=data["row_summary"],
                contract_dict=contract_dict,
            ),
        }
        for row_id, data in current_rows.items()
    }
    log_path = _resolve_log_path(output_root)
    # Write the run_exit_status record only after all training/eval/visuals
    # have produced fresh artifacts. The live runner only blesses this run
    # once it has reached the point of computing the gate; on any subsequent
    # failure the failure path below overwrites the file with "failed".
    paper_evidence_gate_path = output_root / "paper_evidence_gate.json"
    try:
        _write_run_exit_status(
            output_root,
            pid=os.getpid(),
            exit_code=0,
            status="completed",
            log_path=log_path,
        )
        provenance_checks = _build_provenance_checks(
            output_root=output_root,
            provenance_paths=provenance_paths,
            visual_status=visual_status,
            rows=gate_rows,
            log_path=log_path,
            baseline_root=baseline_root,
            ffno_extension_root=ffno_extension_root,
        )
        gate_payload = conv_mod.build_paper_evidence_gate(
            backlog_item=BACKLOG_ITEM,
            expected_epochs=EXPECTED_EPOCHS,
            rows=gate_rows,
            provenance_checks=provenance_checks,
        )
        conv_mod.write_paper_evidence_gate(paper_evidence_gate_path, gate_payload)
        _reseed_top_level_manifest_with_gate(
            preflight_manifest_path,
            gate_payload=gate_payload,
            paper_evidence_gate_path=paper_evidence_gate_path,
        )
        _reseed_metrics_with_gate(
            output_root=output_root, gate_payload=gate_payload
        )
    except BaseException:
        # Replace the optimistic "completed" exit-status record with a failed
        # one so a stale file can never falsely bless a partial run.
        try:
            _write_run_exit_status(
                output_root,
                pid=os.getpid(),
                exit_code=1,
                status="failed",
                log_path=log_path,
            )
        except Exception:
            pass
        raise

    return {
        "preflight_manifest_path": str(preflight_manifest_path),
        "metrics_json_path": str(output_root / "metrics.json"),
        "combined_metrics_json_path": str(output_root / "combined_metrics.json"),
        "convergence_audit_json_path": str(output_root / "convergence_audit.json"),
        "paper_evidence_gate_json_path": str(paper_evidence_gate_path),
        "visual_manifest_path": visual_status["visual_manifest_path"],
        **provenance_paths,
    }


def rebuild_meta_only(
    *,
    baseline_root: Path,
    ffno_extension_root: Path,
    manifest_path: Path,
    output_root: Path,
    contract: Optional[preflight_mod.TrainingContract] = None,
    fixed_sample_ids: Optional[List[int]] = None,
    required_paper_sample: int = 255,
    parent_argv: Optional[List[str]] = None,
    force_overwrite: bool = False,
) -> Dict[str, Any]:
    """Rebuild the top-level manifest, provenance, gate, and audit artifacts
    from existing per-row outputs without retraining.

    The per-row history.json, row_summary.json, model_profile.json, and
    model_state.pt files are preserved as-is. The visual bundle is also
    regenerated from existing per-row source arrays so the run-log linkage and
    same-contract lineage match the on-disk training results.

    A writer lock is acquired against the output root so a meta rebuild cannot
    race with an active training run. Per-row training outputs are preserved
    exactly as they are; only the meta artifacts (manifest, provenance,
    run-exit-status, audit, gate, visuals) are rewritten. The original
    training-run tracked PID and launch timestamp are preserved in
    runtime_provenance.json; the rebuild process records its own provenance in
    a separate ``meta_rebuild`` block.
    """
    output_root = Path(output_root).resolve()
    if not output_root.exists():
        raise FileNotFoundError(f"output_root {output_root} does not exist")
    baseline_root = Path(baseline_root).resolve()
    ffno_extension_root = Path(ffno_extension_root).resolve()
    parent_argv = list(parent_argv) if parent_argv is not None else []
    contract = contract or _default_contract()
    lock_path = _acquire_writer_lock(output_root, allow_force=force_overwrite)
    try:
        return _rebuild_meta_only_inner(
            baseline_root=baseline_root,
            ffno_extension_root=ffno_extension_root,
            manifest_path=manifest_path,
            output_root=output_root,
            contract=contract,
            fixed_sample_ids=fixed_sample_ids,
            required_paper_sample=int(required_paper_sample),
            parent_argv=parent_argv,
        )
    finally:
        _release_writer_lock(lock_path)


def _rebuild_meta_only_inner(
    *,
    baseline_root: Path,
    ffno_extension_root: Path,
    manifest_path: Path,
    output_root: Path,
    contract: preflight_mod.TrainingContract,
    fixed_sample_ids: Optional[List[int]],
    required_paper_sample: int,
    parent_argv: List[str],
) -> Dict[str, Any]:

    baseline_manifest = ext_bundle.validate_baseline_bundle(baseline_root)
    _validate_ffno_extension_bundle(ffno_extension_root, baseline_root=baseline_root)
    authority = load_dataset_authority(manifest_path)
    preflight_mod.assert_decision_support_manifest(authority.raw_manifest)
    fixed_sample_ids = (
        [int(i) for i in fixed_sample_ids]
        if fixed_sample_ids is not None
        else [int(i) for i in baseline_manifest.get("fixed_sample_ids") or []]
    )
    operator_pointer = (
        authority.raw_manifest.get("operator", {}).get("validation_artifact")
        or authority.raw_manifest.get("operator", {}).get("validation_report")
        or "unspecified"
    )
    rows = _make_rows(
        dataset_id=str(authority.dataset_id), operator_version=str(operator_pointer)
    )
    manifest_payload = _build_manifest(
        output_root=output_root,
        baseline_root=baseline_root,
        ffno_extension_root=ffno_extension_root,
        authority=authority,
        rows=rows,
        fixed_sample_ids=fixed_sample_ids,
        required_paper_sample=int(required_paper_sample),
        contract=contract,
    )
    preflight_manifest_path = output_root / "preflight_manifest.json"
    _write_json(preflight_manifest_path, manifest_payload)

    log_path = _resolve_log_path(output_root)
    # The reviewer flagged that the prior code defaulted ``exit_code=0`` and
    # ``status="completed"`` whenever ``run_exit_status.json`` was missing or
    # unreadable, which let the rebuild fabricate completion evidence after
    # the original training proof had been lost. Refuse to proceed unless the
    # original record is present and parseable AND carries a tracked_pid plus
    # a non-empty status. Genuine prior evidence is required for the rebuild
    # to bless the bundle; if the record was lost or corrupted, the bundle
    # must be rerun, not silently regenerated.
    exit_status_path = output_root / "run_exit_status.json"
    if not exit_status_path.exists():
        raise FileNotFoundError(
            "meta-rebuild requires existing run_exit_status.json at "
            f"{exit_status_path}; refusing to fabricate completion evidence"
        )
    try:
        existing_exit_payload = json.loads(exit_status_path.read_text())
    except Exception as exc:
        raise RuntimeError(
            f"meta-rebuild cannot parse {exit_status_path}: {exc!r}; refusing "
            "to fabricate completion evidence"
        ) from exc
    if not isinstance(existing_exit_payload, dict):
        raise RuntimeError(
            f"meta-rebuild expected dict in {exit_status_path}, got "
            f"{type(existing_exit_payload).__name__}"
        )
    raw_tracked_pid = (
        existing_exit_payload.get("tracked_pid")
        if existing_exit_payload.get("tracked_pid") is not None
        else existing_exit_payload.get("pid")
    )
    raw_exit_code = existing_exit_payload.get("exit_code")
    raw_status = existing_exit_payload.get("status")
    if raw_tracked_pid is None or raw_exit_code is None or not raw_status:
        raise RuntimeError(
            f"meta-rebuild requires tracked_pid, exit_code, and status in "
            f"{exit_status_path}; got tracked_pid={raw_tracked_pid!r}, "
            f"exit_code={raw_exit_code!r}, status={raw_status!r}"
        )
    tracked_pid = int(raw_tracked_pid)
    exit_code = int(raw_exit_code)
    status = str(raw_status)
    rebuild_gpu_count = 0
    try:
        import torch

        if torch.cuda.is_available():
            rebuild_gpu_count = int(torch.cuda.device_count())
    except Exception:
        rebuild_gpu_count = 0
    meta_rebuild_block = {
        "rebuild_pid": int(os.getpid()),
        "rebuild_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rebuild_git_sha": get_git_commit(),
        "rebuild_git_dirty": get_git_dirty(),
        "rebuild_hostname": socket.gethostname(),
        "rebuild_platform": platform.platform(),
        "rebuild_gpu_count": int(rebuild_gpu_count),
        "rebuild_argv": list(parent_argv),
    }
    runtime_provenance_path = _amend_existing_runtime_provenance_for_rebuild(
        output_root=output_root,
        log_path=log_path,
        meta_rebuild=meta_rebuild_block,
    )
    dataset_split_paths = _write_dataset_and_split_manifests(
        output_root=output_root,
        authority=authority,
        fixed_sample_ids=fixed_sample_ids,
    )
    provenance_paths = {
        "runtime_provenance_path": str(runtime_provenance_path),
        **dataset_split_paths,
    }
    _write_run_exit_status(
        output_root,
        pid=tracked_pid,
        exit_code=exit_code,
        status=status,
        log_path=log_path,
    )

    current_rows: Dict[str, Dict[str, Any]] = {}
    fixed_targets: Dict[int, Dict[str, np.ndarray]] = {}
    fixed_q_pred_by_row: Dict[int, Dict[str, np.ndarray]] = {
        int(sid): {} for sid in fixed_sample_ids
    }
    fixed_sino_pred_by_row: Dict[int, Dict[str, np.ndarray]] = {
        int(sid): {} for sid in fixed_sample_ids
    }
    source_arrays_dir = output_root / "figures" / "source_arrays"
    for row in rows:
        row_dir = output_root / "rows" / row.row_id
        row_summary_path = row_dir / ROW_SUMMARY_NAME
        if not row_summary_path.exists():
            raise FileNotFoundError(
                f"meta-rebuild requires existing {row_summary_path}"
            )
        row_summary = json.loads(row_summary_path.read_text())
        history_path = Path(
            row_summary.get("history_json_path") or (row_dir / "history.json")
        )
        if not history_path.exists():
            raise FileNotFoundError(
                f"meta-rebuild requires existing history.json at {history_path}"
            )
        history_payload = conv_mod.load_history(history_path)
        current_rows[row.row_id] = {
            "row_summary": row_summary,
            "history_summary": conv_mod.summarize_history(
                row_id=row.row_id,
                history_payload=history_payload,
            ),
        }
        for sid in fixed_sample_ids:
            q_pred_path = (
                source_arrays_dir
                / f"sample_{int(sid):04d}_{row.row_id}_q_pred.npy"
            )
            sino_pred_path = (
                source_arrays_dir
                / f"sample_{int(sid):04d}_{row.row_id}_sino_pred.npy"
            )
            q_target_path = source_arrays_dir / f"sample_{int(sid):04d}_q_target.npy"
            sino_obs_path = source_arrays_dir / f"sample_{int(sid):04d}_sino_obs.npy"
            if q_pred_path.exists() and sino_pred_path.exists():
                fixed_q_pred_by_row[int(sid)][row.row_id] = np.load(q_pred_path)
                fixed_sino_pred_by_row[int(sid)][row.row_id] = np.load(sino_pred_path)
            if q_target_path.exists() and sino_obs_path.exists():
                fixed_targets.setdefault(
                    int(sid),
                    {
                        "q_target": np.load(q_target_path),
                        "sino_obs": np.load(sino_obs_path),
                    },
                )

    visual_status = _write_bundle_visuals(
        output_root=output_root,
        sample_id=int(required_paper_sample),
        fixed_targets=fixed_targets,
        fixed_q_pred_by_row=fixed_q_pred_by_row,
        fixed_sino_pred_by_row=fixed_sino_pred_by_row,
        baseline_root=baseline_root,
    )
    baseline_rows = _baseline_rows(
        baseline_root=baseline_root,
        ffno_extension_root=ffno_extension_root,
    )
    audit_payload = conv_mod.build_convergence_audit(
        backlog_item=BACKLOG_ITEM,
        baseline_rows={
            row_id: {
                "image_metrics": row.get("image"),
                "measurement_metrics": row.get("measurement"),
                "supporting": row.get("supporting"),
                "runtime": row.get("runtime"),
            }
            for row_id, row in baseline_rows.items()
            if row_id in current_rows
        },
        current_rows=current_rows,
    )
    conv_mod.write_convergence_audit_json(
        output_root / "convergence_audit.json", audit_payload
    )
    conv_mod.write_convergence_audit_csv(
        output_root / "convergence_audit.csv", audit_payload
    )

    rebuild_contract_dict = contract.as_dict()
    gate_rows = {
        row_id: {
            "row_status": data["row_summary"]["row_status"],
            "history_records": data["history_summary"]["history_records"],
            "scheduler_matches_contract": _scheduler_matches_contract(
                row_summary=data["row_summary"],
                contract_dict=rebuild_contract_dict,
            ),
        }
        for row_id, data in current_rows.items()
    }
    provenance_checks = _build_provenance_checks(
        output_root=output_root,
        provenance_paths=provenance_paths,
        visual_status=visual_status,
        rows=gate_rows,
        log_path=log_path,
        baseline_root=baseline_root,
        ffno_extension_root=ffno_extension_root,
    )
    gate_payload = conv_mod.build_paper_evidence_gate(
        backlog_item=BACKLOG_ITEM,
        expected_epochs=EXPECTED_EPOCHS,
        rows=gate_rows,
        provenance_checks=provenance_checks,
    )
    paper_evidence_gate_path = output_root / "paper_evidence_gate.json"
    conv_mod.write_paper_evidence_gate(paper_evidence_gate_path, gate_payload)
    _reseed_top_level_manifest_with_gate(
        preflight_manifest_path,
        gate_payload=gate_payload,
        paper_evidence_gate_path=paper_evidence_gate_path,
    )
    _reseed_metrics_with_gate(output_root=output_root, gate_payload=gate_payload)
    return {
        "rebuild_meta_only": True,
        "preflight_manifest_path": str(preflight_manifest_path),
        "convergence_audit_json_path": str(output_root / "convergence_audit.json"),
        "paper_evidence_gate_json_path": str(paper_evidence_gate_path),
        "visual_manifest_path": visual_status["visual_manifest_path"],
        **provenance_paths,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brdt_run_40ep_paper_evidence",
        description="Run the immutable BRDT 40-epoch Hybrid ResNet + FFNO paper-evidence bundle.",
    )
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--ffno-extension-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--rebuild-meta-only",
        action="store_true",
        help=(
            "Skip training and rebuild manifest/provenance/audit/gate from existing "
            "per-row outputs. Use after authoring the durable summary or fixing a "
            "previously incomplete provenance payload."
        ),
    )
    parser.add_argument(
        "--reconstruct-runtime-provenance-from-invocation",
        action="store_true",
        help=(
            "Reconstruct runtime_provenance.json from the on-disk invocation.json "
            "(written by the original training process at startup) and then run "
            "the meta-only rebuild. Use only when the original "
            "runtime_provenance.json was overwritten by an earlier rebuild path "
            "and must be honestly restored. Fields invocation.json did not "
            "preserve (git_sha, git_dirty, hostname, platform, gpu_count) are "
            "explicitly marked unrecoverable; the gate's git_provenance and "
            "host_provenance checks fail on the reconstructed payload, demoting "
            "the bundle to decision_support_convergence_followup rather than "
            "promoting it on top of fabricated values."
        ),
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Force training to proceed against a populated output root.",
    )
    parser.add_argument("--required-paper-sample", type=int, default=255)
    parser.add_argument("--fixed-sample-ids", nargs="+", type=int, default=[145, 83, 255, 126])
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    parent_argv = sys.argv[1:] if argv is None else list(argv)
    if args.reconstruct_runtime_provenance_from_invocation:
        reconstruct_runtime_provenance_from_invocation(
            output_root=args.output_root.resolve()
        )
    if args.rebuild_meta_only or args.reconstruct_runtime_provenance_from_invocation:
        result = rebuild_meta_only(
            baseline_root=args.baseline_root,
            ffno_extension_root=args.ffno_extension_root,
            manifest_path=args.manifest,
            output_root=args.output_root,
            fixed_sample_ids=[int(i) for i in args.fixed_sample_ids],
            required_paper_sample=int(args.required_paper_sample),
            parent_argv=parent_argv,
            force_overwrite=bool(args.force_overwrite),
        )
    else:
        result = run_paper_evidence(
            baseline_root=args.baseline_root,
            ffno_extension_root=args.ffno_extension_root,
            manifest_path=args.manifest,
            output_root=args.output_root,
            device_choice=str(args.device),
            dry_run=bool(args.dry_run),
            fixed_sample_ids=[int(i) for i in args.fixed_sample_ids],
            required_paper_sample=int(args.required_paper_sample),
            parent_argv=parent_argv,
            force_overwrite=bool(args.force_overwrite),
        )
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
