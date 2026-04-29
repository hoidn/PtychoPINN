#!/usr/bin/env python3
"""Build the bounded CNS paper table and figure bundle from locked rows."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from scripts.studies.pdebench_image128.reporting import (
    _load_run_record,
    build_cns_paper_table_bundle,
    validate_cns_paper_table_bundle,
    write_cns_paper_table_bundle,
)
from scripts.studies.pdebench_image128.visualization import cfd_cns_shared_scale_bundle


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCKED_ROWS_PATH = (
    REPO_ROOT
    / ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/cns_paper_locked_rows.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle"
)
DEFAULT_SEARCH_ROOTS = [REPO_ROOT / ".artifacts/NEURIPS-HYBRID-RESNET-2026"]
REQUIRED_1024_HEADLINE_ROWS = [
    "spectral_resnet_bottleneck_base",
    "fno_base",
    "unet_strong",
    "author_ffno_cns_base",
]
PREFERRED_1024_RUN_ROOTS = {
    "spectral_resnet_bottleneck_base": REPO_ROOT
    / ".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-finalists-1024cap-40ep-20260428T054559Z",
}
CONTRACT_KEYS = [
    "dataset_file",
    "split_counts",
    "max_windows_per_trajectory",
    "history_len",
    "epochs",
    "batch_size",
    "training_loss",
    "metric_family",
]
RERUN_REQUIRED_LOGS = {
    "started_at_ns": "cns_bundle_rerun.started_at_ns",
    "pid": "cns_bundle_rerun.pid",
    "exit_code": "cns_bundle_rerun.exit_code",
}
RERUN_PREFLIGHT_COMMANDS = [
    (
        "pytest_required.log",
        [
            "pytest",
            "-q",
            "tests/studies/test_pdebench_cfd_cns_metrics.py",
            "tests/studies/test_pdebench_image128_runner.py",
        ],
    ),
    (
        "compileall.log",
        [
            "python",
            "-m",
            "compileall",
            "-q",
            "scripts/studies/pdebench_image128",
            "scripts/studies/run_pdebench_image128_suite.py",
        ],
    ),
]
AUTHORITATIVE_EVIDENCE_SCOPE = "capped_decision_support_only"
AUTHORITATIVE_ROW_STATUS = "completed"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _expected_1024_contract(locked_rows_payload: dict[str, Any]) -> dict[str, Any]:
    selected = dict(locked_rows_payload.get("selected_contract", {}))
    split_counts = {"train": 1024, "val": 128, "test": 128}
    selected["split_counts"] = split_counts
    selected["max_windows_per_trajectory"] = int(selected.get("max_windows_per_trajectory", 8))
    selected["history_len"] = int(selected.get("history_len", locked_rows_payload.get("history_len", 2)))
    selected["epochs"] = int(selected.get("epochs", locked_rows_payload.get("epochs", 40)))
    selected["batch_size"] = int(locked_rows_payload.get("batch_size", 4))
    selected["training_loss"] = str(locked_rows_payload.get("training_loss", selected.get("training_loss", "mse")))
    selected["metric_family"] = list(locked_rows_payload.get("metric_family", selected.get("metric_family", [])))
    selected["dataset_file"] = str(locked_rows_payload.get("dataset_file", selected.get("dataset_file", "")))
    return {key: selected.get(key) for key in CONTRACT_KEYS}


def _contract_matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(actual.get(key) == expected.get(key) for key in CONTRACT_KEYS)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    deduped: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _row_is_authoritative(record: dict[str, Any]) -> bool:
    row = dict(record.get("row", {}))
    return (
        str(row.get("status", "")) == AUTHORITATIVE_ROW_STATUS
        and str(row.get("evidence_scope", "")) == AUTHORITATIVE_EVIDENCE_SCOPE
    )


def _discover_compatible_run(
    profile_id: str,
    *,
    expected_contract: dict[str, Any],
    search_roots: list[Path],
    preferred_run_root: Path | None = None,
) -> dict[str, Any] | None:
    candidate_roots: list[Path] = []
    if preferred_run_root is not None and preferred_run_root.exists():
        candidate_roots.append(Path(preferred_run_root))
    for search_root in search_roots:
        if not Path(search_root).exists():
            continue
        for metrics_path in Path(search_root).rglob(f"metrics_{profile_id}.json"):
            candidate_roots.append(metrics_path.parent)
    for run_root in _dedupe_paths(candidate_roots):
        try:
            record = _load_run_record(run_root, profile_id=profile_id, source_document="artifact_scan")
        except (FileNotFoundError, ValueError):
            continue
        if _contract_matches(record["contract"], expected_contract) and _row_is_authoritative(record):
            return {
                "row_id": profile_id,
                "run_root": str(run_root),
                "contract": record["contract"],
                "metrics": record["row"],
            }
    return None


def _rerun_output_root(profile_id: str, output_root: Path) -> Path:
    stem = profile_id.replace("_base", "")
    return output_root / "rerun_candidates" / f"{stem}-1024cap-40ep"


def _tmux_socket_path() -> Path:
    socket_dir = Path(os.environ.get("CLAUDE_TMUX_SOCKET_DIR", Path(tempfile.gettempdir()) / "claude-tmux-sockets"))
    socket_dir.mkdir(parents=True, exist_ok=True)
    return socket_dir / "claude.sock"


def _tmux_session_name(profile_id: str) -> str:
    return f"cns1024-{profile_id.replace('_', '-')[:36]}"


def _rerun_log_paths(run_root: Path) -> dict[str, Path]:
    logs_dir = Path(run_root) / "logs"
    return {name: logs_dir / filename for name, filename in RERUN_REQUIRED_LOGS.items()}


def _rerun_command(profile_id: str, *, expected_contract: dict[str, Any], output_root: Path) -> str:
    dataset_file = Path(str(expected_contract["dataset_file"]))
    data_root = dataset_file.parent.parent
    split_counts = dict(expected_contract["split_counts"])
    return (
        "python scripts/studies/run_pdebench_image128_suite.py "
        "--task 2d_cfd_cns "
        "--mode pilot "
        f"--data-root {data_root} "
        f"--output-root {_rerun_output_root(profile_id, output_root)} "
        f"--profiles {profile_id} "
        f"--history-len {int(expected_contract['history_len'])} "
        f"--epochs {int(expected_contract['epochs'])} "
        f"--batch-size {int(expected_contract['batch_size'])} "
        f"--max-train-trajectories {int(split_counts['train'])} "
        f"--max-val-trajectories {int(split_counts['val'])} "
        f"--max-test-trajectories {int(split_counts['test'])} "
        f"--max-windows-per-trajectory {int(expected_contract['max_windows_per_trajectory'])} "
        "--device cuda "
        "--num-workers 0"
    )


def _rerun_candidate(profile_id: str, *, expected_contract: dict[str, Any], output_root: Path) -> dict[str, Any]:
    run_root = _rerun_output_root(profile_id, output_root)
    socket_path = _tmux_socket_path()
    session_name = _tmux_session_name(profile_id)
    log_paths = _rerun_log_paths(run_root)
    stdout_path = run_root / "logs" / "cns_bundle_rerun.stdout.log"
    stderr_path = run_root / "logs" / "cns_bundle_rerun.stderr.log"
    return {
        "row_id": profile_id,
        "reason": "missing_same_contract_1024_row",
        "output_root": str(run_root),
        "rerun_command": _rerun_command(profile_id, expected_contract=expected_contract, output_root=output_root),
        "tmux_socket_path": str(socket_path),
        "tmux_session_name": session_name,
        "tmux_attach_command": f"tmux -S {shlex.quote(str(socket_path))} attach -t {shlex.quote(session_name)}",
        "tmux_capture_command": (
            f"tmux -S {shlex.quote(str(socket_path))} capture-pane -p -J -t {shlex.quote(session_name)}:0.0 -S -200"
        ),
        "log_paths": {name: str(path) for name, path in log_paths.items()},
        "stdout_log_path": str(stdout_path),
        "stderr_log_path": str(stderr_path),
    }


def _required_rerun_artifact_paths(run_root: Path, profile_id: str) -> list[Path]:
    log_paths = _rerun_log_paths(run_root)
    return [
        run_root / "invocation.json",
        run_root / "dataset_manifest.json",
        run_root / "split_manifest.json",
        run_root / "comparison_summary.json",
        run_root / f"metrics_{profile_id}.json",
        run_root / f"model_profile_{profile_id}.json",
        log_paths["started_at_ns"],
        log_paths["pid"],
        log_paths["exit_code"],
    ]


def _guard_rerun_output_root(run_root: Path) -> None:
    run_root = Path(run_root)
    if not run_root.exists():
        return
    if not any(run_root.iterdir()):
        return
    raise ValueError(f"rerun output root must be empty before launch: {run_root}")


def _launch_tmux_rerun(candidate: dict[str, Any]) -> dict[str, Any]:
    run_root = Path(candidate["output_root"])
    _guard_rerun_output_root(run_root)
    log_paths = _rerun_log_paths(run_root)
    for path in [*log_paths.values(), run_root / "logs" / "cns_bundle_rerun.stdout.log", run_root / "logs" / "cns_bundle_rerun.stderr.log"]:
        path.parent.mkdir(parents=True, exist_ok=True)
    for path in log_paths.values():
        if path.exists():
            path.unlink()

    shell_lines = [
        "set -euo pipefail",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        "source ~/miniconda3/etc/profile.d/conda.sh",
        "conda activate ptycho311",
        f"printf '%s\\n' \"$(date +%s%N)\" > {shlex.quote(str(log_paths['started_at_ns']))}",
        f"{candidate['rerun_command']} > {shlex.quote(str(run_root / 'logs' / 'cns_bundle_rerun.stdout.log'))} "
        f"2> {shlex.quote(str(run_root / 'logs' / 'cns_bundle_rerun.stderr.log'))} &",
        "pid=$!",
        f"printf '%s\\n' \"$pid\" > {shlex.quote(str(log_paths['pid']))}",
        "set +e",
        "wait \"$pid\"",
        "rc=$?",
        "set -e",
        f"printf '%s\\n' \"$rc\" > {shlex.quote(str(log_paths['exit_code']))}",
        "exit \"$rc\"",
    ]
    socket_path = Path(str(candidate["tmux_socket_path"]))
    session_name = str(candidate["tmux_session_name"])
    command = ["tmux", "-S", str(socket_path), "new-session", "-d", "-s", session_name, "bash", "-lc", "\n".join(shell_lines)]
    subprocess.run(command, check=True)
    return {
        "row_id": str(candidate["row_id"]),
        "output_root": str(run_root),
        "tmux_socket_path": str(socket_path),
        "tmux_session_name": session_name,
        "tmux_attach_command": str(candidate["tmux_attach_command"]),
        "tmux_capture_command": str(candidate["tmux_capture_command"]),
        "log_paths": dict(candidate["log_paths"]),
    }


def _wait_for_tmux_session(result: dict[str, Any], *, poll_interval_sec: float = 2.0) -> None:
    socket_path = str(result["tmux_socket_path"])
    session_name = str(result["tmux_session_name"])
    while True:
        has_session = subprocess.run(
            ["tmux", "-S", socket_path, "has-session", "-t", session_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if has_session.returncode != 0:
            return
        time.sleep(poll_interval_sec)


def _verify_rerun_completion(candidate: dict[str, Any], *, expected_contract: dict[str, Any]) -> dict[str, Any]:
    run_root = Path(candidate["output_root"])
    row_id = str(candidate["row_id"])
    log_paths = _rerun_log_paths(run_root)
    for path in _required_rerun_artifact_paths(run_root, row_id):
        if not path.exists():
            raise FileNotFoundError(f"missing rerun artifact for {row_id}: {path}")

    started_at_ns = int(log_paths["started_at_ns"].read_text(encoding="utf-8").strip())
    pid = log_paths["pid"].read_text(encoding="utf-8").strip()
    if not pid.isdigit():
        raise ValueError(f"tracked rerun pid is not numeric for {row_id}: {pid!r}")
    exit_code = log_paths["exit_code"].read_text(encoding="utf-8").strip()
    if exit_code != "0":
        raise ValueError(f"rerun for {row_id} failed with exit code {exit_code}")

    invocation = _load_json(run_root / "invocation.json")
    if str(invocation.get("pid")) != pid:
        raise ValueError(f"invocation pid mismatch for {row_id}: {invocation.get('pid')!r} != {pid!r}")

    record = _load_run_record(run_root, profile_id=row_id, source_document="rerun_execution")
    if not _contract_matches(record["contract"], expected_contract):
        raise ValueError(f"rerun contract mismatch for {row_id}")
    if not _row_is_authoritative(record):
        raise ValueError(f"rerun output is not authoritative for {row_id}")

    for artifact_path in _required_rerun_artifact_paths(run_root, row_id):
        if artifact_path.stat().st_mtime_ns < started_at_ns:
            raise ValueError(f"stale rerun artifact for {row_id}: {artifact_path}")

    return {
        "row_id": row_id,
        "run_root": str(run_root),
        "contract": record["contract"],
        "metrics": record["row"],
        "verified_pid": pid,
        "verified_exit_code": exit_code,
    }


def _default_rerun_executor(
    *,
    rerun_candidates: list[dict[str, Any]],
    expected_contract: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    output_root = Path(output_root)
    launched: list[dict[str, Any]] = []
    for candidate in rerun_candidates:
        launch_result = _launch_tmux_rerun(candidate)
        launched.append(launch_result)
    for launch_result in launched:
        _wait_for_tmux_session(launch_result)
    payload = {
        "executor": "tmux",
        "launched_row_ids": [str(item["row_id"]) for item in rerun_candidates],
        "launches": launched,
        "expected_contract": expected_contract,
    }
    _write_json(output_root / "1024_rerun_execution.json", payload)
    return payload


def _default_rerun_preflight_verifier(*, output_root: Path) -> dict[str, Any]:
    verification_root = Path(output_root) / "verification"
    verification_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for log_name, command in RERUN_PREFLIGHT_COMMANDS:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log_path = verification_root / log_name
        log_path.write_text(completed.stdout or "", encoding="utf-8")
        results.append(
            {
                "command": " ".join(command),
                "exit_code": int(completed.returncode),
                "log_path": str(log_path),
            }
        )
    payload = {
        "schema_version": "pdebench_cns_rerun_preflight_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(item["exit_code"] == 0 for item in results) else "failed",
        "results": results,
    }
    _write_json(verification_root / "rerun_preflight_checks.json", payload)
    if payload["status"] != "passed":
        raise RuntimeError("rerun preflight checks failed")
    return payload


def audit_cns_paper_bundle_upgrade(
    *,
    locked_rows_path: Path,
    output_root: Path,
    search_roots: list[Path] | None = None,
    preferred_run_roots: dict[str, Path] | None = None,
    execute_missing_reruns: bool = False,
    rerun_executor: Callable[..., dict[str, Any]] | None = None,
    rerun_preflight_verifier: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    locked_rows_payload = _load_json(Path(locked_rows_path))
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    search_roots = [Path(path) for path in (search_roots or DEFAULT_SEARCH_ROOTS)]
    preferred_run_roots = {**PREFERRED_1024_RUN_ROOTS, **(preferred_run_roots or {})}
    expected_contract = _expected_1024_contract(locked_rows_payload)

    compatible_rows: dict[str, Any] = {}
    missing_rows: list[dict[str, Any]] = []
    for row_id in REQUIRED_1024_HEADLINE_ROWS:
        match = _discover_compatible_run(
            row_id,
            expected_contract=expected_contract,
            search_roots=search_roots,
            preferred_run_root=preferred_run_roots.get(row_id),
        )
        if match is None:
            missing_rows.append(_rerun_candidate(row_id, expected_contract=expected_contract, output_root=output_root))
            continue
        compatible_rows[row_id] = match

    rerun_manifest = {
        "schema_version": "pdebench_cns_paper_bundle_rerun_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_contract": expected_contract,
        "rerun_candidates": missing_rows,
    }
    _write_json(output_root / "1024_rerun_manifest.json", rerun_manifest)

    rerun_execution: dict[str, Any] | None = None
    rerun_preflight: dict[str, Any] | None = None
    if missing_rows and execute_missing_reruns:
        verifier = rerun_preflight_verifier or _default_rerun_preflight_verifier
        rerun_preflight = verifier(output_root=output_root)
        executor = rerun_executor or _default_rerun_executor
        rerun_execution = executor(
            rerun_candidates=missing_rows,
            expected_contract=expected_contract,
            output_root=output_root,
        )
        rerun_verified: dict[str, Any] = {}
        for candidate in missing_rows:
            verified = _verify_rerun_completion(candidate, expected_contract=expected_contract)
            rerun_verified[str(candidate["row_id"])] = verified
        compatible_rows.update(rerun_verified)
        missing_rows = []

    if not missing_rows and rerun_execution is None:
        audit_outcome = "upgrade_ready"
    elif not missing_rows:
        audit_outcome = "upgrade_ready_after_reruns"
    else:
        audit_outcome = "fallback_to_512_required"
    payload = {
        "schema_version": "pdebench_cns_paper_bundle_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "locked_rows_path": str(locked_rows_path),
        "fallback_locked_rows_path": str(locked_rows_path),
        "audit_outcome": audit_outcome,
        "expected_1024_contract": expected_contract,
        "compatible_1024_rows": compatible_rows,
        "missing_or_incompatible_rows": missing_rows,
        "rerun_manifest_path": str(output_root / "1024_rerun_manifest.json"),
        "rerun_preflight": rerun_preflight,
        "rerun_execution": rerun_execution,
        "comparison_standard": "Exact match on dataset file, split counts, max_windows_per_trajectory, history_len, epochs, batch_size, training_loss, and metric_family.",
        "authority_standard": "Rows must be completed and expose evidence_scope=capped_decision_support_only.",
    }
    _write_json(output_root / "1024_same_cap_audit.json", payload)
    _write_audit_markdown(output_root / "1024_same_cap_audit.md", payload)
    return payload


def _write_audit_markdown(path: Path, payload: dict[str, Any]) -> Path:
    lines = [
        "# 1024 Same-Cap Audit",
        "",
        f"- Outcome: `{payload['audit_outcome']}`",
        f"- Locked fallback manifest: `{payload['fallback_locked_rows_path']}`",
        f"- Rerun manifest: `{payload['rerun_manifest_path']}`",
        "",
        "## Compatible 1024 Rows",
    ]
    if payload["compatible_1024_rows"]:
        for row_id, row in payload["compatible_1024_rows"].items():
            lines.append(f"- `{row_id}`: `{row['run_root']}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Missing Or Incompatible Rows"])
    if payload["missing_or_incompatible_rows"]:
        for row in payload["missing_or_incompatible_rows"]:
            lines.append(f"- `{row['row_id']}`: `{row['reason']}`")
            lines.append(f"  rerun: `{row['rerun_command']}`")
    else:
        lines.append("- none")
    if payload.get("rerun_execution"):
        lines.extend(["", "## Rerun Execution"])
        lines.append(f"- executor: `{payload['rerun_execution'].get('executor', '')}`")
        launched = payload["rerun_execution"].get("launched_row_ids", [])
        lines.append(f"- launched rows: `{', '.join(str(item) for item in launched)}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _parse_sample_id(path: Path) -> int:
    stem = path.stem
    marker = "sample"
    if marker not in stem:
        raise ValueError(f"cannot parse sample id from {path}")
    return int(stem.split(marker)[-1])


def _row_npz_candidates(row: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    asset_pointers = row.get("asset_pointers", {}) or {}
    sample_npz = asset_pointers.get("sample_npz")
    if isinstance(sample_npz, str) and sample_npz.strip():
        candidates.append(Path(sample_npz))
    run_root = Path(str(row.get("run_root", "")))
    row_id = str(row.get("row_id"))
    if run_root.exists():
        candidates.extend(sorted(run_root.glob(f"comparison_{row_id}_sample*.npz")))
    return _dedupe_paths(candidates)


def _load_npz_payload(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "prediction": np.asarray(data["prediction"], dtype=np.float32),
            "target": np.asarray(data["target"], dtype=np.float32),
            "abs_error": np.asarray(data["abs_error"], dtype=np.float32),
            "field_order": [str(item) for item in data["field_order"].tolist()],
        }


def _visual_rows_from_locked_payload(locked_rows_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows_by_id = {str(row["row_id"]): dict(row) for row in locked_rows_payload["rows"]}
    headline_ids = [str(row_id) for row_id in locked_rows_payload.get("headline_row_ids", [])]
    continuity_ids = [str(row_id) for row_id in locked_rows_payload.get("continuity_row_ids", [])]
    headline_split = rows_by_id[headline_ids[0]]["split_counts"]
    selected_rows = [rows_by_id[row_id] for row_id in headline_ids]
    for row_id in continuity_ids:
        if rows_by_id[row_id]["split_counts"] == headline_split:
            selected_rows.append(rows_by_id[row_id])
    return selected_rows


def _resolve_shared_sample_ids(rows: list[dict[str, Any]]) -> tuple[list[int], dict[str, dict[int, Path]]]:
    sample_map: dict[str, dict[int, Path]] = {}
    intersection: set[int] | None = None
    for row in rows:
        row_id = str(row["row_id"])
        row_samples = { _parse_sample_id(path): path for path in _row_npz_candidates(row) if path.exists() }
        if not row_samples:
            raise FileNotFoundError(f"no sample npz artifacts available for {row_id}")
        sample_map[row_id] = row_samples
        row_ids = set(row_samples)
        intersection = row_ids if intersection is None else intersection & row_ids
    if not intersection:
        raise ValueError("no compatible sample ids exist across the selected visual rows")
    return sorted(intersection), sample_map


def _copy_source_npz(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _save_panel(path: Path, image: np.ndarray, spec: dict[str, Any]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, image, cmap=spec["cmap"], vmin=spec["vmin"], vmax=spec["vmax"])
    return path


def _build_figure_bundle(locked_rows_payload: dict[str, Any], *, output_root: Path) -> dict[str, Any]:
    rows = _visual_rows_from_locked_payload(locked_rows_payload)
    sample_ids, sample_paths = _resolve_shared_sample_ids(rows)
    field_order: list[str] | None = None
    loaded_by_sample: dict[int, dict[str, dict[str, Any]]] = {}
    entries: list[dict[str, Any]] = []

    for sample_id in sample_ids:
        sample_label = f"sample{sample_id:03d}"
        loaded_rows: dict[str, dict[str, Any]] = {}
        for row in rows:
            row_id = str(row["row_id"])
            source_path = sample_paths[row_id][sample_id]
            copied_path = _copy_source_npz(
                source_path,
                output_root / "figure_sources" / sample_label / f"{row_id}.npz",
            )
            loaded_rows[row_id] = {
                **_load_npz_payload(copied_path),
                "copied_npz_path": str(copied_path),
            }

        first_row_id = str(rows[0]["row_id"])
        reference_target = loaded_rows[first_row_id]["target"]
        reference_field_order = loaded_rows[first_row_id]["field_order"]
        for row_id, payload in loaded_rows.items():
            if payload["field_order"] != reference_field_order:
                raise ValueError(f"field-order mismatch for sample {sample_id}: {row_id}")
            if not np.allclose(payload["target"], reference_target, atol=1e-6, rtol=1e-6):
                raise ValueError(f"target mismatch for sample {sample_id}: {row_id}")
        sample_field_order = list(reference_field_order)
        if field_order is None:
            field_order = sample_field_order
        elif sample_field_order != field_order:
            raise ValueError(f"field-order mismatch across samples for sample {sample_id}")
        loaded_by_sample[sample_id] = loaded_rows

    field_order = field_order or []
    shared_field_scales: dict[str, Any] = {}
    shared_error_scales: dict[str, Any] = {}
    first_row_id = str(rows[0]["row_id"])
    for channel, field_name in enumerate(field_order):
        value_arrays = []
        error_arrays = []
        for sample_id in sample_ids:
            loaded_rows = loaded_by_sample[sample_id]
            value_arrays.append(loaded_rows[first_row_id]["target"][channel])
            value_arrays.extend(payload["prediction"][channel] for payload in loaded_rows.values())
            error_arrays.extend(payload["abs_error"][channel] for payload in loaded_rows.values())
        bundle = cfd_cns_shared_scale_bundle(
            field_name,
            value_arrays=value_arrays,
            error_arrays=error_arrays,
        )
        shared_field_scales[field_name] = bundle["value_scale"]
        shared_error_scales[field_name] = bundle["error_scale"]

    for sample_id in sample_ids:
        sample_label = f"sample{sample_id:03d}"
        loaded_rows = loaded_by_sample[sample_id]
        reference_target = loaded_rows[first_row_id]["target"]
        for channel, field_name in enumerate(field_order):
            value_scale = shared_field_scales[field_name]
            error_scale = shared_error_scales[field_name]

            target_path = _save_panel(
                output_root / "figures" / sample_label / f"{field_name}__target.png",
                reference_target[channel],
                value_scale,
            )
            entries.append(
                {
                    "sample_id": sample_id,
                    "field_name": field_name,
                    "row_id": "ground_truth",
                    "panel_kind": "target",
                    "png_path": str(target_path),
                    "scale_kind": "value",
                }
            )
            for row in rows:
                row_id = str(row["row_id"])
                prediction_path = _save_panel(
                    output_root / "figures" / sample_label / f"{field_name}__{row_id}__prediction.png",
                    loaded_rows[row_id]["prediction"][channel],
                    value_scale,
                )
                error_path = _save_panel(
                    output_root / "figures" / sample_label / f"{field_name}__{row_id}__abs_error.png",
                    loaded_rows[row_id]["abs_error"][channel],
                    error_scale,
                )
                entries.extend(
                    [
                        {
                            "sample_id": sample_id,
                            "field_name": field_name,
                            "row_id": row_id,
                            "panel_kind": "prediction",
                            "png_path": str(prediction_path),
                            "scale_kind": "value",
                            "source_npz_path": loaded_rows[row_id]["copied_npz_path"],
                        },
                        {
                            "sample_id": sample_id,
                            "field_name": field_name,
                            "row_id": row_id,
                            "panel_kind": "abs_error",
                            "png_path": str(error_path),
                            "scale_kind": "error",
                            "source_npz_path": loaded_rows[row_id]["copied_npz_path"],
                        },
                    ]
                )

    field_scale_path = _write_json(output_root / "shared_field_scales.json", shared_field_scales)
    error_scale_path = _write_json(output_root / "shared_error_scales.json", shared_error_scales)
    sample_manifest = {
        "schema_version": "pdebench_cns_fixed_sample_manifest_v1",
        "sample_ids": sample_ids,
        "rows_in_visual_bundle": [str(row["row_id"]) for row in rows],
        "field_order": field_order,
    }
    figure_manifest = {
        "schema_version": "pdebench_cns_figure_manifest_v1",
        "sample_ids": sample_ids,
        "rows_in_visual_bundle": [str(row["row_id"]) for row in rows],
        "field_order": field_order,
        "entries": entries,
    }
    _write_json(output_root / "fixed_sample_manifest.json", sample_manifest)
    _write_json(output_root / "figure_manifest.json", figure_manifest)
    return {
        "sample_manifest_path": str(output_root / "fixed_sample_manifest.json"),
        "figure_manifest_path": str(output_root / "figure_manifest.json"),
        "field_scale_path": str(field_scale_path),
        "error_scale_path": str(error_scale_path),
        "sample_ids": sample_ids,
        "rows_in_visual_bundle": [str(row["row_id"]) for row in rows],
    }


def run_cns_paper_bundle(
    *,
    locked_rows_path: Path = DEFAULT_LOCKED_ROWS_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    search_roots: list[Path] | None = None,
    execute_missing_1024_reruns: bool = False,
    rerun_executor: Callable[..., dict[str, Any]] | None = None,
    rerun_preflight_verifier: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    locked_rows_path = Path(locked_rows_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    audit_payload = audit_cns_paper_bundle_upgrade(
        locked_rows_path=locked_rows_path,
        output_root=output_root,
        search_roots=search_roots,
        execute_missing_reruns=execute_missing_1024_reruns,
        rerun_executor=rerun_executor,
        rerun_preflight_verifier=rerun_preflight_verifier,
    )
    locked_rows_payload = _load_json(locked_rows_path)
    bundle_kind = (
        "same_contract_1024_bundle_complete"
        if int(locked_rows_payload["split_counts"]["train"]) == 1024
        else "fallback_512_bundle_used"
    )
    _write_json(
        output_root / "bundle_input_manifest.json",
        {
            "schema_version": "pdebench_cns_bundle_input_manifest_v1",
            "authoritative_locked_rows_path": str(locked_rows_path),
            "contract_authority": str(locked_rows_payload.get("contract_authority", "")),
            "bundle_kind": bundle_kind,
            "audit_outcome": audit_payload["audit_outcome"],
        },
    )

    table_payload = build_cns_paper_table_bundle(
        locked_rows_payload,
        authoritative_manifest_path=str(locked_rows_path),
    )
    table_json, table_csv, table_tex = write_cns_paper_table_bundle(table_payload, output_root)
    figure_payload = _build_figure_bundle(locked_rows_payload, output_root=output_root)
    sample_manifest = _load_json(Path(figure_payload["sample_manifest_path"]))
    figure_manifest = _load_json(Path(figure_payload["figure_manifest_path"]))

    table_rows = list(table_payload.get("rows", []))
    rows_by_id = {str(row["row_id"]): row for row in table_rows}
    headline_row_ids = [str(row_id) for row_id in table_payload.get("headline_row_ids", [])]
    continuity_row_ids = [str(row_id) for row_id in table_payload.get("continuity_row_ids", [])]
    headline_split_label = str(rows_by_id[headline_row_ids[0]]["split_label"]) if headline_row_ids else ""
    expected_visual_row_ids = list(headline_row_ids)
    expected_visual_row_ids.extend(
        row_id
        for row_id in continuity_row_ids
        if str(rows_by_id[row_id]["split_label"]) == headline_split_label
    )
    sample_manifest_row_ids = [str(row_id) for row_id in sample_manifest.get("rows_in_visual_bundle", [])]
    figure_manifest_row_ids = [str(row_id) for row_id in figure_manifest.get("rows_in_visual_bundle", [])]
    figure_entry_row_ids = sorted(
        {
            str(entry["row_id"])
            for entry in figure_manifest.get("entries", [])
            if str(entry.get("row_id", "")) != "ground_truth"
        }
    )
    figure_entry_sample_ids = sorted({int(entry["sample_id"]) for entry in figure_manifest.get("entries", [])})

    validation_payload = {
        **validate_cns_paper_table_bundle(table_payload),
        "table_json_path": str(table_json),
        "table_csv_path": str(table_csv),
        "table_tex_path": str(table_tex),
        "figure_manifest_path": figure_payload["figure_manifest_path"],
        "sample_manifest_path": figure_payload["sample_manifest_path"],
        "visual_bundle_row_ids": expected_visual_row_ids,
        "sample_manifest_row_ids": sample_manifest_row_ids,
        "figure_manifest_row_ids": figure_manifest_row_ids,
        "figure_entry_row_ids": figure_entry_row_ids,
        "sample_manifest_matches_figure_manifest": (
            sample_manifest.get("sample_ids") == figure_manifest.get("sample_ids")
            and sample_manifest.get("field_order") == figure_manifest.get("field_order")
            and sample_manifest_row_ids == figure_manifest_row_ids
        ),
        "table_and_visual_row_rosters_agree": (
            expected_visual_row_ids == sample_manifest_row_ids == figure_manifest_row_ids
        ),
        "figure_entries_match_visual_bundle": (
            sorted(expected_visual_row_ids) == figure_entry_row_ids
            and list(sample_manifest.get("sample_ids", [])) == list(figure_manifest.get("sample_ids", []))
            and figure_entry_sample_ids == sorted(int(item) for item in sample_manifest.get("sample_ids", []))
        ),
    }
    _write_json(output_root / "bundle_validation.json", validation_payload)
    return {
        "bundle_kind": bundle_kind,
        "audit_outcome": audit_payload["audit_outcome"],
        "table_json_path": str(table_json),
        "table_csv_path": str(table_csv),
        "table_tex_path": str(table_tex),
        "figure_manifest_path": figure_payload["figure_manifest_path"],
        "validation_path": str(output_root / "bundle_validation.json"),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked-rows-path", type=Path, default=DEFAULT_LOCKED_ROWS_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--search-root", dest="search_roots", action="append", type=Path)
    parser.add_argument("--execute-missing-1024-reruns", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_cns_paper_bundle(
        locked_rows_path=args.locked_rows_path,
        output_root=args.output_root,
        search_roots=args.search_roots,
        execute_missing_1024_reruns=args.execute_missing_1024_reruns,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
