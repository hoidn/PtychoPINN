"""Shared provenance helpers for paper-grade study bundles."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, default=_json_default), encoding="utf-8")
    return path


def load_json_if_exists(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, dict) else {}


def relative_to_output_dir(output_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(output_dir))
    except ValueError:
        return str(path)


def file_identity(path: Path, *, source: str | None = None) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        payload = {
            "path": str(path),
            "filename": path.name,
            "missing": True,
        }
        if source:
            payload["source"] = source
        return payload
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    payload: dict[str, Any] = {
        "path": str(path),
        "filename": path.name,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "sha256": digest.hexdigest(),
    }
    if source:
        payload["source"] = source
    return payload


def git_dirty(repo_root: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--short"],
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return None
    return bool(result.stdout.strip())


def current_runtime_provenance(*, hardware_summary: Mapping[str, object] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version.split()[0],
        "cwd": str(Path.cwd()),
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "host": socket.gethostname(),
    }
    try:
        ptycho_torch = importlib.import_module("ptycho_torch")
    except Exception:
        payload["ptycho_torch_file"] = None
    else:
        payload["ptycho_torch_file"] = str(Path(ptycho_torch.__file__).resolve())

    torch_version = None
    cuda_version = None
    gpu = None
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None:
        torch_version = getattr(torch, "__version__", None)
        cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        try:
            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_name(torch.cuda.current_device())
        except Exception:
            gpu = "cuda"
    if hardware_summary:
        accelerator = hardware_summary.get("accelerator")
        if isinstance(accelerator, str) and accelerator.strip() and accelerator != "unknown":
            gpu = accelerator
    payload["torch_version"] = torch_version
    payload["cuda_version"] = cuda_version
    payload["gpu"] = gpu or "unknown"
    return payload


def merge_runtime_provenance(
    runtime_payload: object,
    *,
    hardware_summary: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    merged = current_runtime_provenance(hardware_summary=hardware_summary)
    if isinstance(runtime_payload, Mapping):
        for key, value in runtime_payload.items():
            if value is not None and value != "":
                merged[str(key)] = value
    return merged


def merge_git_provenance(
    git_payload: object,
    *,
    repo_root: Path,
    commit: str | None = None,
    note_source: str,
) -> dict[str, Any]:
    merged = dict(git_payload) if isinstance(git_payload, Mapping) else {}
    commit_value = merged.get("commit") or commit
    if commit_value is not None:
        merged["commit"] = str(commit_value)
    dirty_note = merged.get("dirty_state_note")
    if not isinstance(dirty_note, Mapping) or "source" not in dirty_note or "dirty" not in dirty_note:
        dirty_value = git_dirty(repo_root)
        merged["dirty_state_note"] = {
            "source": note_source,
            "dirty": dirty_value,
        }
    return merged


def write_dataset_identity_manifest(
    output_dir: Path,
    *,
    train_npz: Path,
    test_npz: Path,
    dataset_source: str,
    probe_npz: Path | None = None,
    probe_source: str | None = None,
    probe_scale_mode: str | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_source": dataset_source,
        "train_npz": file_identity(train_npz, source=dataset_source),
        "test_npz": file_identity(test_npz, source=dataset_source),
    }
    if probe_npz is not None and Path(probe_npz).exists():
        payload["probe_npz"] = file_identity(probe_npz, source=probe_source or "probe")
        payload["probe_source"] = probe_source
        payload["probe_scale_mode"] = probe_scale_mode
    return write_json(Path(output_dir) / "dataset_identity_manifest.json", payload)


def write_split_manifest(
    output_dir: Path,
    *,
    train_npz: Path,
    test_npz: Path,
    seed: int,
    nimgs_train: int,
    nimgs_test: int,
    gridsize: int,
    set_phi: bool,
) -> Path:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_npz": str(Path(train_npz)),
        "test_npz": str(Path(test_npz)),
        "seed": int(seed),
        "nimgs_train": int(nimgs_train),
        "nimgs_test": int(nimgs_test),
        "gridsize": int(gridsize),
        "set_phi": bool(set_phi),
    }
    return write_json(Path(output_dir) / "split_manifest.json", payload)


def write_exit_code_proof(
    output_dir: Path,
    *,
    model_id: str,
    invocation_json: Path | None,
    stdout_log: Path,
    stderr_log: Path,
    proof_source: str,
) -> Path:
    invocation_payload = load_json_if_exists(invocation_json) if invocation_json is not None else {}
    payload = {
        "model_id": model_id,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "proof_source": proof_source,
        "exit_code": 0,
        "invocation_json": relative_to_output_dir(output_dir, invocation_json) if invocation_json is not None else None,
        "invocation_status": invocation_payload.get("status"),
        "stdout_log": relative_to_output_dir(output_dir, stdout_log),
        "stderr_log": relative_to_output_dir(output_dir, stderr_log),
    }
    return write_json(Path(output_dir) / "runs" / model_id / "exit_code_proof.json", payload)
