from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RESULTS_HEADER = (
    "session_id\ttimestamp_utc\tref_or_commit\tdecision\tamp_ssim\t"
    "compared_to_ref\tcompared_to_amp_ssim\tdelta_amp_ssim\toutput_root\t"
    "comparison_png\tcommand_or_source\tnotes"
)

WORKFLOW_QUEUE_OUTCOME_DIRS = {
    "KEEP": "accepted",
    "DISCARD": "discarded",
    "BLOCKED": "blocked",
    "CRASH": "crashed",
    "TIMEOUT": "timed_out",
}

COMMON_PROPOSAL_FIELDS = (
    "candidate_kind",
    "base_ref",
    "smoke_command",
    "smoke_output_root",
    "smoke_log_path",
    "run_command",
    "output_root",
    "log_path",
    "comparison_png_path",
    "note",
    "hypothesis",
)

DEFAULT_MODEL = "gpt-5.4"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_MAX_ITERATIONS = 100

SEARCH_SUMMARY_FLAGS = {
    "fno_modes": "--fno-modes",
    "fno_width": "--fno-width",
    "fno_blocks": "--fno-blocks",
    "hybrid_downsample_steps": "--hybrid-downsample-steps",
    "hybrid_downsample_op": "--hybrid-downsample-op",
    "hybrid_resnet_blocks": "--hybrid-resnet-blocks",
    "hybrid_skip_style": "--hybrid-skip-style",
    "torch_resnet_width": "--torch-resnet-width",
    "learning_rate": "--learning-rate",
    "plateau_min_lr": "--plateau-min-lr",
    "hybrid_encoder_spectral_hidden_scale": "--hybrid-encoder-spectral-hidden-scale",
    "hybrid_encoder_conv_hidden_scale": "--hybrid-encoder-conv-hidden-scale",
}

DISCARD_STREAK_DECISIONS = {"discard", "timeout", "crash"}
HYPOTHESIS_FAMILIES = (
    "capacity",
    "downsampling",
    "skip_fusion",
    "optimizer_schedule",
    "encoder_hidden_scale",
    "other",
)


@dataclass
class SessionState:
    session_id: str
    session_branch: str
    session_root: Path
    session_json_path: Path
    results_tsv_path: Path
    accepted_state_path: Path
    protected_local_paths_path: Path
    outputs_root: Path
    comparison_gallery_dir: Path
    baseline_output_root: Path
    baseline_log_path: Path
    baseline_result_path: Path
    baseline_comparison_png: Path
    current_phase: str
    iteration: int
    debug_attempts: int


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _session_root(repo_root: Path, session_id: str) -> Path:
    return repo_root / "state" / "lines_256_arch_improvement_v2" / "sessions" / session_id


def _session_branch_name(session_id: str) -> str:
    return f"lines256/session/{session_id}"


def _repo_root_for_session(session_root: Path) -> Path:
    return session_root.parents[3]


def _results_tsv_path(session_root: Path) -> Path:
    return session_root / "results.tsv"


def _accepted_state_path(session_root: Path) -> Path:
    return session_root / "accepted_state.json"


def _protected_local_paths_path(session_root: Path) -> Path:
    return session_root / "protected_local_paths.json"


def _outputs_root(repo_root: Path, session_id: str) -> Path:
    return repo_root / "outputs" / "lines_256_arch_improvement_v2" / "sessions" / session_id


def _comparison_gallery_dir(outputs_root: Path) -> Path:
    return outputs_root / "comparison_pngs"


def _baseline_output_root(outputs_root: Path) -> Path:
    return outputs_root / "baseline"


def _proposal_context_path(session_root: Path) -> Path:
    return session_root / "proposal_context.json"


def _workflow_queue_root(repo_root: Path) -> Path:
    return repo_root / "docs" / "workflow_queue"


def _workflow_queue_active_dir(repo_root: Path) -> Path:
    return _workflow_queue_root(repo_root) / "active"


def _load_active_workflow_queue_item(repo_root: Path) -> dict[str, str] | None:
    active_dir = _workflow_queue_active_dir(repo_root)
    if not active_dir.exists():
        return None
    candidates = sorted(path for path in active_dir.glob("*.md") if path.is_file())
    if not candidates:
        return None
    selected = candidates[0]
    return {
        "path": _relative_to_repo(repo_root, selected),
        "content": selected.read_text(encoding="utf-8"),
    }


def _iterations_root(session_root: Path) -> Path:
    return session_root / "iterations"


def _iteration_root(session: SessionState, iteration: int | None = None) -> Path:
    index = session.iteration if iteration is None else iteration
    return _iterations_root(session.session_root) / f"{index:03d}"


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_to_repo(repo_root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(repo_root.resolve()).as_posix())


def _read_results_rows(session_root: Path) -> list[dict[str, str]]:
    results_tsv_path = _results_tsv_path(session_root)
    if not results_tsv_path.exists():
        return []
    with results_tsv_path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _write_results_header(results_tsv_path: Path) -> None:
    results_tsv_path.parent.mkdir(parents=True, exist_ok=True)
    if not results_tsv_path.exists():
        results_tsv_path.write_text(RESULTS_HEADER + "\n", encoding="utf-8")


def _write_session_json(session: SessionState) -> None:
    repo_root = _repo_root_for_session(session.session_root)
    payload = {
        "session_id": session.session_id,
        "session_branch": session.session_branch,
        "current_phase": session.current_phase,
        "iteration": session.iteration,
        "debug_attempts": session.debug_attempts,
        "session_root": _relative_to_repo(repo_root, session.session_root),
        "results_tsv_path": _relative_to_repo(repo_root, session.results_tsv_path),
        "accepted_state_path": _relative_to_repo(repo_root, session.accepted_state_path),
        "protected_local_paths_path": _relative_to_repo(
            repo_root, session.protected_local_paths_path
        ),
        "outputs_root": _relative_to_repo(repo_root, session.outputs_root),
        "comparison_gallery_dir": _relative_to_repo(repo_root, session.comparison_gallery_dir),
        "baseline_output_root": _relative_to_repo(repo_root, session.baseline_output_root),
        "baseline_log_path": _relative_to_repo(repo_root, session.baseline_log_path),
        "baseline_result_path": _relative_to_repo(repo_root, session.baseline_result_path),
        "baseline_comparison_png": _relative_to_repo(repo_root, session.baseline_comparison_png),
    }
    session.session_json_path.parent.mkdir(parents=True, exist_ok=True)
    session.session_json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def initialize_session(repo_root: Path, session_id: str | None = None) -> SessionState:
    repo_root = repo_root.resolve()
    session_id = session_id or _utc_timestamp()
    session_root = _session_root(repo_root, session_id)
    outputs_root = _outputs_root(repo_root, session_id)
    session = SessionState(
        session_id=session_id,
        session_branch=_session_branch_name(session_id),
        session_root=session_root,
        session_json_path=session_root / "session.json",
        results_tsv_path=_results_tsv_path(session_root),
        accepted_state_path=_accepted_state_path(session_root),
        protected_local_paths_path=_protected_local_paths_path(session_root),
        outputs_root=outputs_root,
        comparison_gallery_dir=_comparison_gallery_dir(outputs_root),
        baseline_output_root=_baseline_output_root(outputs_root),
        baseline_log_path=session_root / f"{session_id}_baseline.log",
        baseline_result_path=session_root / "baseline_run_result.json",
        baseline_comparison_png=_comparison_gallery_dir(outputs_root)
        / f"{session_id}__baseline__compare_amp_phase_probe.png",
        current_phase="init",
        iteration=0,
        debug_attempts=0,
    )
    session_root.mkdir(parents=True, exist_ok=True)
    session.comparison_gallery_dir.mkdir(parents=True, exist_ok=True)
    _iterations_root(session_root).mkdir(parents=True, exist_ok=True)
    _write_results_header(session.results_tsv_path)
    _write_json(session.protected_local_paths_path, [])
    _write_session_json(session)
    return session


def load_session(session_root: Path) -> SessionState:
    payload = json.loads((session_root / "session.json").read_text(encoding="utf-8"))
    repo_root = _repo_root_for_session(session_root)
    return SessionState(
        session_id=payload["session_id"],
        session_branch=str(payload.get("session_branch", _session_branch_name(payload["session_id"]))),
        session_root=session_root,
        session_json_path=session_root / "session.json",
        results_tsv_path=repo_root / payload["results_tsv_path"],
        accepted_state_path=repo_root / payload["accepted_state_path"],
        protected_local_paths_path=repo_root / payload["protected_local_paths_path"],
        outputs_root=repo_root / payload["outputs_root"],
        comparison_gallery_dir=repo_root / payload["comparison_gallery_dir"],
        baseline_output_root=repo_root / payload["baseline_output_root"],
        baseline_log_path=repo_root / payload["baseline_log_path"],
        baseline_result_path=repo_root / payload["baseline_result_path"],
        baseline_comparison_png=repo_root / payload["baseline_comparison_png"],
        current_phase=payload["current_phase"],
        iteration=int(payload["iteration"]),
        debug_attempts=int(payload.get("debug_attempts", 0)),
    )


def _set_phase(session: SessionState, phase: str) -> None:
    session.current_phase = phase
    _write_session_json(session)


def _git_status_lines(repo_root: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = []
    for line in proc.stdout.splitlines():
        line = line.rstrip()
        if line and len(line) >= 4:
            lines.append(line[3:].strip())
    return lines


def capture_protected_local_paths(repo_root: Path, session: SessionState) -> list[str]:
    protected = _git_status_lines(repo_root.resolve())
    _write_json(session.protected_local_paths_path, protected)
    return protected


def build_baseline_command(session: SessionState) -> list[str]:
    repo_root = _repo_root_for_session(session.session_root)
    return [
        "python",
        "scripts/studies/run_lines_256_arch_experiment.py",
        "--output-dir",
        str(session.baseline_output_root.relative_to(repo_root)),
        "--fno-modes",
        "12",
        "--fno-width",
        "32",
        "--fno-blocks",
        "4",
        "--no-hybrid-skip-connections",
        "--hybrid-downsample-steps",
        "2",
        "--hybrid-downsample-op",
        "stride_conv",
        "--hybrid-resnet-blocks",
        "6",
        "--hybrid-skip-style",
        "add",
    ]


def _find_metrics_path(output_root: Path) -> Path:
    direct = output_root / "runs" / "pinn_hybrid_resnet" / "metrics.json"
    if direct.exists():
        return direct
    candidates = sorted(output_root.rglob("metrics.json"))
    if not candidates:
        raise SystemExit(f"Missing metrics under {output_root}")
    return candidates[0]


def _find_randomness_path(output_root: Path) -> Path:
    direct = output_root / "runs" / "pinn_hybrid_resnet" / "randomness_contract.json"
    if direct.exists():
        return direct
    candidates = sorted(output_root.rglob("randomness_contract.json"))
    if not candidates:
        raise SystemExit(f"Missing randomness contract under {output_root}")
    return candidates[0]


def _find_compare_png(output_root: Path) -> Path:
    direct = output_root / "visuals" / "compare_amp_phase_probe.png"
    if direct.exists():
        return direct
    candidates = sorted(output_root.rglob("compare_amp_phase_probe.png"))
    if not candidates:
        raise SystemExit(f"Missing comparison PNG under {output_root}")
    return candidates[0]


def _find_plain_compare_png(output_root: Path) -> Path | None:
    direct = output_root / "visuals" / "compare_amp_phase.png"
    if direct.exists():
        return direct
    candidates = sorted(output_root.rglob("compare_amp_phase.png"))
    if not candidates:
        return None
    return candidates[0]


def _read_visual_publication_status(output_root: Path) -> dict[str, object] | None:
    path = output_root / "visual_publication_status.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Malformed visual publication status at {path}")
    return payload


def _resolve_optional_comparison_artifact(
    output_root: Path,
    publication_status: dict[str, object] | None,
) -> tuple[Path | None, str, str | None]:
    status = str(publication_status.get("status", "unknown")) if publication_status else "unknown"
    warning = (
        str(publication_status["warning"])
        if publication_status and publication_status.get("warning")
        else None
    )

    if publication_status and publication_status.get("published_compare_path"):
        candidate = output_root / str(publication_status["published_compare_path"])
        if candidate.exists():
            return candidate, status, warning

    probe_compare = output_root / "visuals" / "compare_amp_phase_probe.png"
    if probe_compare.exists():
        return probe_compare, ("published" if status == "unknown" else status), warning

    plain_compare = _find_plain_compare_png(output_root)
    if plain_compare is not None:
        if status == "unknown":
            status = "fallback_plain_compare"
        if warning is None:
            warning = "Optional probe-inclusive comparison PNG was unavailable; plain compare was used."
        return plain_compare, status, warning

    if status == "unknown":
        status = "missing_nonfatal"
    if warning is None:
        warning = "Optional probe-inclusive comparison PNG was unavailable."
    return None, status, warning


def _extract_amp_ssim(metrics_path: Path) -> float:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    amp_ssim = metrics.get("amp_ssim")
    if amp_ssim is None:
        ssim = metrics.get("ssim")
        if isinstance(ssim, list) and len(ssim) >= 1:
            amp_ssim = ssim[0]
    if amp_ssim is None:
        raise SystemExit("Metrics did not expose amp_ssim")
    return float(amp_ssim)


def _append_baseline_row(session: SessionState, accepted_ref: str, amp_ssim: float, command: str) -> None:
    repo_root = _repo_root_for_session(session.session_root)
    with session.results_tsv_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(
            [
                session.session_id,
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                accepted_ref,
                "baseline",
                amp_ssim,
                "na",
                "na",
                "na",
                _relative_to_repo(repo_root, session.baseline_output_root),
                _relative_to_repo(repo_root, session.baseline_comparison_png),
                command,
                "fresh session baseline",
            ]
        )


def _git_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _git_current_branch(repo_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _git_branch_exists(repo_root: Path, branch: str) -> bool:
    completed = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0


def _attach_or_reset_session_branch(repo_root: Path, branch: str, ref: str) -> None:
    current_branch = _git_current_branch(repo_root)
    if current_branch == branch:
        if _git_head(repo_root) != ref:
            _run_git_mutation(repo_root, "reset", "--mixed", ref)
        return

    if _git_branch_exists(repo_root, branch):
        _run_git_mutation(repo_root, "branch", "-f", branch, ref)
    else:
        _run_git_mutation(repo_root, "branch", branch, ref)
    _run_git_mutation(repo_root, "checkout", branch)
    if _git_head(repo_root) != ref:
        _run_git_mutation(repo_root, "reset", "--mixed", ref)


def _ensure_session_checkout_state(
    repo_root: Path,
    session: SessionState,
    *,
    expected_ref: str,
) -> None:
    current_branch = _git_current_branch(repo_root)
    if current_branch == session.session_branch:
        if _git_head(repo_root) != expected_ref:
            raise SystemExit(
                f"HEAD {_git_head(repo_root)} does not match expected session ref {expected_ref}"
            )
        return

    if current_branch is None:
        _attach_or_reset_session_branch(repo_root, session.session_branch, expected_ref)
        if _git_current_branch(repo_root) != session.session_branch:
            raise SystemExit(
                f"Failed to reattach detached checkout to session branch {session.session_branch}"
            )
        if _git_head(repo_root) != expected_ref:
            raise SystemExit(
                f"HEAD {_git_head(repo_root)} does not match expected session ref {expected_ref}"
            )
        return

    raise SystemExit(
        f"Checkout is on unexpected branch {current_branch}; expected {session.session_branch}"
    )


def harvest_baseline_outputs(
    repo_root: Path,
    session: SessionState,
    accepted_ref: str,
    run_result: dict[str, object],
) -> dict[str, object]:
    if run_result.get("status") != "completed" or int(run_result.get("exit_code", 1)) != 0:
        raise SystemExit(f"Baseline failed: {run_result}")

    _write_json(session.baseline_result_path, run_result)
    metrics_path = _find_metrics_path(session.baseline_output_root)
    randomness_path = _find_randomness_path(session.baseline_output_root)
    compare_path = _find_compare_png(session.baseline_output_root)
    amp_ssim = _extract_amp_ssim(metrics_path)
    randomness_contract = json.loads(randomness_path.read_text(encoding="utf-8"))

    session.baseline_comparison_png.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(compare_path, session.baseline_comparison_png)

    accepted_state = {
        "accepted_ref": accepted_ref,
        "accepted_amp_ssim": amp_ssim,
        "accepted_state_path": _relative_to_repo(repo_root, session.accepted_state_path),
        "session_id": session.session_id,
        "baseline_output_root": _relative_to_repo(repo_root, session.baseline_output_root),
        "baseline_comparison_png": _relative_to_repo(repo_root, session.baseline_comparison_png),
        "baseline_command": run_result["command"],
        "accepted_run_command": run_result["command"],
        "accepted_candidate_kind": "baseline",
        "accepted_randomness_contract": randomness_contract,
    }
    _write_json(session.accepted_state_path, accepted_state)
    _append_baseline_row(session, accepted_ref, amp_ssim, str(run_result["command"]))
    return accepted_state


def run_baseline(repo_root: Path, session: SessionState) -> dict[str, object]:
    repo_root = repo_root.resolve()
    _set_phase(session, "baseline_running")
    command = build_baseline_command(session)
    completed = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    session.baseline_log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    run_result = {
        "status": "completed" if completed.returncode == 0 else "failed",
        "exit_code": completed.returncode,
        "command": " ".join(command),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "log_path": _relative_to_repo(repo_root, session.baseline_log_path),
    }
    accepted_state = harvest_baseline_outputs(
        repo_root=repo_root,
        session=session,
        accepted_ref=_git_head(repo_root),
        run_result=run_result,
    )
    _attach_or_reset_session_branch(
        repo_root,
        session.session_branch,
        str(accepted_state["accepted_ref"]),
    )
    _set_phase(session, "baseline_complete")
    return accepted_state


def _dry_run_payload(session: SessionState, mode: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "session_id": session.session_id,
        "mode": mode,
        "session_root": str(session.session_root),
    }
    if mode == "baseline-only":
        payload["baseline_command"] = " ".join(build_baseline_command(session))
    return payload


def _extract_attempted_values(command_or_source: str) -> dict[str, set[str]]:
    attempted = {knob: set() for knob in SEARCH_SUMMARY_FLAGS}
    for knob, flag in SEARCH_SUMMARY_FLAGS.items():
        pattern = re.compile(rf"{re.escape(flag)}\s+([^\s]+)")
        for match in pattern.findall(command_or_source):
            attempted[knob].add(match)
    if "--hybrid-skip-connections" in command_or_source:
        attempted["hybrid_skip_connections"] = {"on"}
    elif "--no-hybrid-skip-connections" in command_or_source:
        attempted["hybrid_skip_connections"] = {"off"}
    else:
        attempted["hybrid_skip_connections"] = set()
    return attempted


def _classify_hypothesis_family(command_or_source: str, note: str = "", hypothesis: str = "") -> str:
    text = " ".join([command_or_source, note, hypothesis]).lower()
    if any(token in text for token in ("--learning-rate", "--plateau-min-lr", "scheduler", "learning rate", "plateau")):
        return "optimizer_schedule"
    if any(
        token in text
        for token in (
            "--hybrid-encoder-spectral-hidden-scale",
            "--hybrid-encoder-conv-hidden-scale",
            "encoder hidden scale",
            "spectral hidden scale",
            "conv hidden scale",
        )
    ):
        return "encoder_hidden_scale"
    if any(
        token in text
        for token in (
            "--hybrid-downsample-steps",
            "--hybrid-downsample-op",
            "downsample",
            "avgpool",
            "blurpool",
            "stride_conv",
        )
    ):
        return "downsampling"
    if any(
        token in text
        for token in (
            "--hybrid-skip-style",
            "--hybrid-skip-connections",
            "--no-hybrid-skip-connections",
            "skip fusion",
            "skip connection",
            "gated_add",
            "concat",
        )
    ):
        return "skip_fusion"
    if any(
        token in text
        for token in (
            "--fno-modes",
            "--fno-width",
            "--fno-blocks",
            "--hybrid-resnet-blocks",
            "--torch-resnet-width",
            "bottleneck",
            "width",
            "modes",
            "blocks",
        )
    ):
        return "capacity"
    return "other"


def _recent_discard_streak(rows: list[dict[str, str]]) -> int:
    streak = 0
    for row in reversed(rows):
        if row.get("decision", "").lower() in DISCARD_STREAK_DECISIONS:
            streak += 1
            continue
        break
    return streak


def build_recent_history_summary(session_root: Path, max_rows: int = 10) -> dict[str, object]:
    rows = _read_results_rows(session_root)
    if not rows:
        return {"recent_attempts": [], "recent_outcomes": []}

    recent_attempts = rows[-max_rows:] if max_rows > 0 else []
    return {
        "recent_attempts": recent_attempts,
        "recent_outcomes": [row["decision"] for row in recent_attempts],
    }


def build_search_summary(session_root: Path) -> dict[str, object]:
    rows = _read_results_rows(session_root)
    attempted_values = {knob: set() for knob in (*SEARCH_SUMMARY_FLAGS.keys(), "hybrid_skip_connections")}
    family_counts = Counter()
    for row in rows:
        values = _extract_attempted_values(row.get("command_or_source", ""))
        for knob, parsed in values.items():
            attempted_values[knob].update(parsed)
        family = _classify_hypothesis_family(
            row.get("command_or_source", ""),
            row.get("notes", ""),
            "",
        )
        family_counts[family] += 1

    serialized_attempted_values = {
        knob: sorted(values)
        for knob, values in attempted_values.items()
    }
    recent_rows = rows[-5:]
    recent_families = [
        _classify_hypothesis_family(row.get("command_or_source", ""), row.get("notes", ""), "")
        for row in recent_rows
    ]
    dominant_recent_family = Counter(recent_families).most_common(1)[0][0] if recent_families else None
    preferred_exploration_families = [
        family
        for family in HYPOTHESIS_FAMILIES
        if family != dominant_recent_family and family_counts.get(family, 0) == 0
    ]
    return {
        "total_attempts": len(rows),
        "decision_counts": dict(Counter(row["decision"] for row in rows)),
        "family_counts": dict(family_counts),
        "recent_families": recent_families,
        "dominant_recent_family": dominant_recent_family,
        "attempted_values": serialized_attempted_values,
        "underexplored_knobs": [
            knob for knob, values in serialized_attempted_values.items() if len(values) <= 1
        ],
        "recent_discard_streak": _recent_discard_streak(rows),
        "preferred_exploration_families": preferred_exploration_families,
    }


def _select_proposal_mode(search_summary: dict[str, object]) -> tuple[str, str]:
    recent_discard_streak = int(search_summary.get("recent_discard_streak", 0))
    dominant_recent_family = search_summary.get("dominant_recent_family")
    if recent_discard_streak >= 5 and dominant_recent_family:
        return (
            "explore",
            f"Recent discard streak suggests local saturation around the {dominant_recent_family} family.",
        )
    return (
        "exploit",
        "Default exploit mode: no sustained discard streak suggests an exploratory jump yet.",
    )


def build_proposal_context(session: SessionState, max_rows: int = 10) -> dict[str, object]:
    repo_root = _repo_root_for_session(session.session_root)
    accepted_state = json.loads(session.accepted_state_path.read_text(encoding="utf-8"))
    recent_history = build_recent_history_summary(session.session_root, max_rows=max_rows)
    search_summary = build_search_summary(session.session_root)
    proposal_mode, proposal_mode_reason = _select_proposal_mode(search_summary)
    queued_workflow_idea = _load_active_workflow_queue_item(repo_root)
    queue_priority_active = queued_workflow_idea is not None
    if queue_priority_active:
        proposal_mode_reason = (
            f"Queued workflow idea {queued_workflow_idea['path']} has priority for this proposal."
        )
    proposal_context_path = _proposal_context_path(session.session_root)
    payload = {
        "session_id": session.session_id,
        "iteration": session.iteration,
        "accepted_state": accepted_state,
        "recent_history": recent_history,
        "search_summary": search_summary,
        "proposal_mode": proposal_mode,
        "proposal_mode_reason": proposal_mode_reason,
        "queue_priority_active": queue_priority_active,
        "queued_workflow_idea": queued_workflow_idea,
        "proposal_context_path": str(proposal_context_path),
    }
    proposal_context_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _candidate_context_path(session: SessionState, iteration: int | None = None) -> Path:
    return _iteration_root(session, iteration) / "candidate_context.json"


def _candidate_metadata_path(session: SessionState, iteration: int | None = None) -> Path:
    return _iteration_root(session, iteration) / "candidate_metadata.json"


def _candidate_paths_file(session: SessionState, iteration: int | None = None) -> Path:
    return _iteration_root(session, iteration) / "candidate_paths.json"


def _candidate_run_result_path(session: SessionState, iteration: int | None = None) -> Path:
    return _iteration_root(session, iteration) / "candidate_run_result.json"


def _candidate_assessment_path(session: SessionState, iteration: int | None = None) -> Path:
    return _iteration_root(session, iteration) / "candidate_assessment.json"


def _debug_candidate_metadata_path(session: SessionState, iteration: int | None = None) -> Path:
    return _iteration_root(session, iteration) / "debug_candidate_metadata.json"


def _debug_candidate_paths_file(session: SessionState, iteration: int | None = None) -> Path:
    return _iteration_root(session, iteration) / "debug_candidate_paths.json"


def _debug_candidate_run_result_path(session: SessionState, iteration: int | None = None) -> Path:
    return _iteration_root(session, iteration) / "debug_candidate_run_result.json"


def _debug_candidate_assessment_path(session: SessionState, iteration: int | None = None) -> Path:
    return _iteration_root(session, iteration) / "debug_candidate_assessment.json"


def _iteration_agent_log_path(
    session: SessionState, log_name: str, iteration: int | None = None
) -> Path:
    return _iteration_root(session, iteration) / log_name


def build_candidate_context(session: SessionState, iteration: int | None = None) -> dict[str, object]:
    repo_root = _repo_root_for_session(session.session_root)
    iter_root = _iteration_root(session, iteration)
    iter_root.mkdir(parents=True, exist_ok=True)
    proposal_context = (
        _read_json(_proposal_context_path(session.session_root))
        if _proposal_context_path(session.session_root).exists()
        else {}
    )
    payload = {
        "timestamp_utc": _utc_timestamp(),
        "candidate_metadata_path": _relative_to_repo(repo_root, _candidate_metadata_path(session, iteration)),
        "candidate_paths_file": _relative_to_repo(repo_root, _candidate_paths_file(session, iteration)),
        "candidate_run_result_path": _relative_to_repo(
            repo_root, _candidate_run_result_path(session, iteration)
        ),
        "candidate_assessment_path": _relative_to_repo(
            repo_root, _candidate_assessment_path(session, iteration)
        ),
        "debug_candidate_metadata_path": _relative_to_repo(
            repo_root, _debug_candidate_metadata_path(session, iteration)
        ),
        "debug_candidate_paths_file": _relative_to_repo(
            repo_root, _debug_candidate_paths_file(session, iteration)
        ),
        "debug_candidate_run_result_path": _relative_to_repo(
            repo_root, _debug_candidate_run_result_path(session, iteration)
        ),
        "debug_candidate_assessment_path": _relative_to_repo(
            repo_root, _debug_candidate_assessment_path(session, iteration)
        ),
        "output_root_base": _relative_to_repo(repo_root, session.outputs_root / "candidates"),
        "log_root": _relative_to_repo(repo_root, iter_root),
        "comparison_gallery_dir": _relative_to_repo(repo_root, session.comparison_gallery_dir),
        "queue_priority_active": bool(proposal_context.get("queue_priority_active", False)),
        "queued_workflow_idea": proposal_context.get("queued_workflow_idea"),
    }
    _write_json(_candidate_context_path(session, iteration), payload)
    return payload


def normalize_candidate_proposal(proposal: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for field in COMMON_PROPOSAL_FIELDS:
        value = proposal.get(field)
        if not value:
            raise SystemExit(f"Proposal missing required field: {field}")
        normalized[field] = value

    candidate_kind = str(normalized["candidate_kind"])
    if candidate_kind not in {"source", "run_config"}:
        raise SystemExit(f"Unsupported candidate_kind: {candidate_kind}")

    if candidate_kind == "source":
        for field in ("candidate_commit", "candidate_paths_file"):
            value = proposal.get(field)
            if not value:
                raise SystemExit(f"Proposal missing required field: {field}")
            normalized[field] = value

    return normalized


def _load_accepted_state(session: SessionState) -> dict[str, object]:
    return json.loads(session.accepted_state_path.read_text(encoding="utf-8"))


def _write_accepted_state(session: SessionState, accepted_state: dict[str, object]) -> None:
    _write_json(session.accepted_state_path, accepted_state)


def _git_commit_exists(repo_root: Path, commitish: str) -> bool:
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", commitish],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0


def _protected_local_paths(session: SessionState) -> list[str]:
    return list(_read_json(session.protected_local_paths_path))


def _assert_tracked_paths_preserved(repo_root: Path, session: SessionState) -> None:
    current = _git_status_lines(repo_root)
    expected = _protected_local_paths(session)
    if current != expected:
        raise SystemExit(
            "Tracked working tree diverged from protected local paths: "
            f"current={current!r} expected={expected!r}"
        )


def _load_relative_json_list(repo_root: Path, relpath: str) -> list[str]:
    path = repo_root / relpath
    payload = _read_json(path)
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise SystemExit(f"Expected JSON list[str] at {path}")
    return payload


def _queued_workflow_idea_for_iteration(session: SessionState) -> dict[str, object] | None:
    candidate_context_path = _candidate_context_path(session)
    if not candidate_context_path.exists():
        proposal_context_path = _proposal_context_path(session.session_root)
        if not proposal_context_path.exists():
            return None
        payload = _read_json(proposal_context_path)
    else:
        payload = _read_json(candidate_context_path)
    queued = payload.get("queued_workflow_idea")
    return queued if isinstance(queued, dict) else None


def _move_queued_workflow_idea(
    repo_root: Path,
    session: SessionState,
    decision: str,
) -> None:
    queued = _queued_workflow_idea_for_iteration(session)
    if not queued:
        return
    dest_dir_name = WORKFLOW_QUEUE_OUTCOME_DIRS.get(str(decision).upper())
    if not dest_dir_name:
        return
    relpath = str(queued.get("path", "")).strip()
    if not relpath:
        return

    source = repo_root / relpath
    destination = _workflow_queue_root(repo_root) / dest_dir_name / Path(relpath).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    if source.exists():
        shutil.move(str(source), str(destination))


def _candidate_paths_for_source_proposal(repo_root: Path, proposal: dict[str, object]) -> list[str]:
    candidate_paths_file = proposal.get("candidate_paths_file")
    if not candidate_paths_file:
        raise SystemExit("Source proposal is missing candidate_paths_file")
    return _load_relative_json_list(repo_root, str(candidate_paths_file))


def _run_git_mutation(repo_root: Path, *args: str) -> None:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"Failed to run git {' '.join(args)}: {completed.stdout}{completed.stderr}".strip()
        )


def _path_exists_in_ref(repo_root: Path, ref: str, relpath: str) -> bool:
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{ref}:{relpath}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0


def _remove_worktree_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _assert_candidate_scoped_cleanup(
    repo_root: Path,
    session: SessionState,
    candidate_paths: list[str],
) -> None:
    current_paths = set(_git_status_lines(repo_root))
    protected_set = set(_protected_local_paths(session))
    candidate_set = set(candidate_paths)

    missing_protected = sorted(protected_set - current_paths)
    lingering_candidate = sorted(candidate_set & current_paths)

    if missing_protected or lingering_candidate:
        extra_dirty = sorted(current_paths - protected_set - candidate_set)
        details = []
        if missing_protected:
            details.append(f"missing_protected={missing_protected}")
        if lingering_candidate:
            details.append(f"lingering_candidate={lingering_candidate}")
        if extra_dirty:
            details.append(f"extra_dirty={extra_dirty}")
        raise SystemExit(
            "Candidate cleanup disturbed protected or candidate-scoped paths:\n"
            + "\n".join(details)
        )


def _cleanup_source_candidate(
    repo_root: Path,
    session: SessionState,
    proposal: dict[str, object],
) -> None:
    candidate_paths = _candidate_paths_for_source_proposal(repo_root, proposal)
    overlap = sorted(set(candidate_paths) & set(_protected_local_paths(session)))
    if overlap:
        raise SystemExit(
            "Candidate paths overlap protected local paths and cannot be handled safely:\n"
            + "\n".join(overlap)
        )

    current_ref = _git_head(repo_root)
    _attach_or_reset_session_branch(repo_root, session.session_branch, current_ref)
    base_ref = str(proposal["base_ref"])
    _run_git_mutation(repo_root, "reset", "--mixed", base_ref)
    existing_paths = [path for path in candidate_paths if _path_exists_in_ref(repo_root, base_ref, path)]
    missing_paths = [path for path in candidate_paths if path not in existing_paths]
    if existing_paths:
        _run_git_mutation(
            repo_root,
            "restore",
            "--source",
            base_ref,
            "--staged",
            "--worktree",
            "--",
            *existing_paths,
        )
    for relpath in missing_paths:
        _remove_worktree_path(repo_root / relpath)
    _assert_candidate_scoped_cleanup(repo_root, session, candidate_paths)


def validate_ready_proposal(repo_root: Path, session: SessionState, proposal: dict[str, object]) -> None:
    accepted = _load_accepted_state(session)
    if str(proposal["base_ref"]) != str(accepted["accepted_ref"]):
        raise SystemExit(
            f"Proposal base_ref {proposal['base_ref']} does not match accepted_ref {accepted['accepted_ref']}"
        )

    smoke_output_root = repo_root / str(proposal["smoke_output_root"])
    smoke_log_path = repo_root / str(proposal["smoke_log_path"])
    if not smoke_output_root.exists():
        raise SystemExit(f"Missing smoke output root: {smoke_output_root}")
    if not smoke_log_path.exists():
        raise SystemExit(f"Missing smoke log path: {smoke_log_path}")

    smoke_randomness = json.loads(_find_randomness_path(smoke_output_root).read_text(encoding="utf-8"))
    if smoke_randomness != accepted["accepted_randomness_contract"]:
        raise SystemExit(
            "Smoke randomness contract does not match accepted session contract: "
            f"{smoke_randomness!r} != {accepted['accepted_randomness_contract']!r}"
        )

    candidate_kind = str(proposal["candidate_kind"])
    if candidate_kind == "source":
        if not _git_commit_exists(repo_root, str(proposal["candidate_commit"])):
            raise SystemExit(f"Missing candidate commit: {proposal['candidate_commit']}")
        if _git_head(repo_root) != str(proposal["candidate_commit"]):
            raise SystemExit(
                f"HEAD {_git_head(repo_root)} does not match source candidate commit {proposal['candidate_commit']}"
            )
        _load_relative_json_list(repo_root, str(proposal["candidate_paths_file"]))
        _assert_tracked_paths_preserved(repo_root, session)
        return

    if "candidate_commit" in proposal or "candidate_paths_file" in proposal:
        raise SystemExit("run_config proposal must not include candidate commit or candidate paths")
    _assert_tracked_paths_preserved(repo_root, session)


def _render_injected_prompt(
    repo_root: Path,
    prompt_paths: list[Path],
    injected_paths: list[Path],
    instruction: str,
) -> str:
    prompt_parts = [instruction.rstrip(), ""]
    for path in injected_paths:
        rel = _relative_to_repo(repo_root, path)
        prompt_parts.append(f"--- BEGIN INJECTED FILE: {rel} ---")
        prompt_parts.append(path.read_text(encoding="utf-8").rstrip())
        prompt_parts.append(f"--- END INJECTED FILE: {rel} ---")
        prompt_parts.append("")
    for prompt_path in prompt_paths:
        prompt_parts.append(prompt_path.read_text(encoding="utf-8").rstrip())
        prompt_parts.append("")
    return "\n".join(prompt_parts)


def _resolve_codex_command(
    codex_cmd: str,
    model: str,
    reasoning_effort: str,
) -> list[str]:
    codex_bin = shutil.which(codex_cmd) or codex_cmd
    if not codex_bin:
        raise SystemExit(f"Codex CLI not found: {codex_cmd}")
    return [
        codex_bin,
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "--model",
        model,
        "--config",
        f"reasoning_effort={reasoning_effort}",
    ]


def _run_codex_prompt(
    repo_root: Path,
    prompt_text: str,
    log_path: Path,
    codex_cmd: str,
    model: str,
    reasoning_effort: str,
) -> subprocess.CompletedProcess[str]:
    command = _resolve_codex_command(codex_cmd, model, reasoning_effort)
    completed = subprocess.run(
        command,
        cwd=repo_root,
        input=prompt_text,
        capture_output=True,
        text=True,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = [f"$ {' '.join(command)}", completed.stdout, completed.stderr]
    log_path.write_text("".join(rendered), encoding="utf-8")
    return completed


def _proposal_injected_paths(repo_root: Path, session: SessionState, candidate_context_path: Path) -> list[Path]:
    return [
        repo_root / "docs/studies/lines_256_dataset.md",
        repo_root / "docs/studies/lines_256_arch_improvement_loop.md",
        session.protected_local_paths_path,
        session.accepted_state_path,
        candidate_context_path,
        _proposal_context_path(session.session_root),
    ]


def _proposal_prompt_paths(repo_root: Path, proposal_context: dict[str, object]) -> list[Path]:
    prompt_root = repo_root / "prompts/workflows/lines_256_arch_improvement"
    proposal_mode = str(proposal_context.get("proposal_mode", "exploit")).strip().lower()
    if proposal_mode not in {"exploit", "explore"}:
        raise SystemExit(f"Unsupported proposal_mode: {proposal_mode}")
    common_prompt = prompt_root / "experiment_step_common.md"
    mode_prompt = prompt_root / f"experiment_step_{proposal_mode}.md"
    if common_prompt.exists() and mode_prompt.exists():
        return [common_prompt, mode_prompt]

    legacy_prompt = prompt_root / "experiment_step.md"
    if legacy_prompt.exists():
        return [legacy_prompt]
    raise SystemExit(
        "Missing proposal prompt files: expected either the split prompt bundle "
        f"({common_prompt.name} + {mode_prompt.name}) or legacy {legacy_prompt.name}"
    )


def _debug_injected_paths(repo_root: Path, session: SessionState, candidate_context_path: Path) -> list[Path]:
    return [
        repo_root / "docs/studies/lines_256_dataset.md",
        repo_root / "docs/studies/lines_256_arch_improvement_loop.md",
        session.protected_local_paths_path,
        session.accepted_state_path,
        candidate_context_path,
        _proposal_context_path(session.session_root),
        _candidate_metadata_path(session),
        _candidate_assessment_path(session),
    ]


def _load_candidate_metadata(path: Path) -> dict[str, object]:
    if not path.exists():
        raise SystemExit(f"Missing candidate metadata: {path}")
    metadata = _read_json(path)
    status = str(metadata.get("status"))
    if status not in {"READY", "BLOCKED"}:
        raise SystemExit(f"Candidate metadata has unsupported status: {status}")
    return metadata


def run_proposal_step(
    repo_root: Path,
    session: SessionState,
    codex_cmd: str,
    model: str,
    reasoning_effort: str,
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    proposal_context = build_proposal_context(session, max_rows=10)
    build_candidate_context(session)
    candidate_context_path = _candidate_context_path(session)
    prompt_text = _render_injected_prompt(
        repo_root=repo_root,
        prompt_paths=_proposal_prompt_paths(repo_root, proposal_context),
        injected_paths=_proposal_injected_paths(repo_root, session, candidate_context_path),
        instruction="Use these files as the authoritative experiment-session inputs.",
    )
    _set_phase(session, "proposal_running")
    completed = _run_codex_prompt(
        repo_root=repo_root,
        prompt_text=prompt_text,
        log_path=_iteration_agent_log_path(session, "proposal_agent.log"),
        codex_cmd=codex_cmd,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    metadata = _load_candidate_metadata(_candidate_metadata_path(session))
    if completed.returncode != 0 and metadata.get("status") != "BLOCKED":
        raise SystemExit(f"Proposal agent failed with exit code {completed.returncode}")
    if metadata.get("status") == "BLOCKED":
        _assert_tracked_paths_preserved(repo_root, session)
        _set_phase(session, "proposal_complete")
        return metadata

    proposal = normalize_candidate_proposal(metadata)
    validate_ready_proposal(repo_root, session, proposal)
    _set_phase(session, "proposal_complete")
    return proposal


def run_debug_step(
    repo_root: Path,
    session: SessionState,
    codex_cmd: str,
    model: str,
    reasoning_effort: str,
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    candidate_context_path = _candidate_context_path(session)
    prompt_text = _render_injected_prompt(
        repo_root=repo_root,
        prompt_paths=[repo_root / "prompts/workflows/lines_256_arch_improvement/debug_crash.md"],
        injected_paths=_debug_injected_paths(repo_root, session, candidate_context_path),
        instruction="Use these files as the authoritative crash-debug inputs.",
    )
    _set_phase(session, "debug_running")
    completed = _run_codex_prompt(
        repo_root=repo_root,
        prompt_text=prompt_text,
        log_path=_iteration_agent_log_path(session, "debug_agent.log"),
        codex_cmd=codex_cmd,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    metadata = _load_candidate_metadata(_debug_candidate_metadata_path(session))
    if completed.returncode != 0 and metadata.get("status") != "BLOCKED":
        raise SystemExit(f"Debug agent failed with exit code {completed.returncode}")
    if metadata.get("status") == "BLOCKED":
        _assert_tracked_paths_preserved(repo_root, session)
        _set_phase(session, "debug_complete")
        return metadata

    proposal = normalize_candidate_proposal(metadata)
    validate_ready_proposal(repo_root, session, proposal)
    _set_phase(session, "debug_complete")
    return proposal


def _write_scored_artifacts(
    session: SessionState,
    assessment: dict[str, object],
    debug: bool = False,
) -> None:
    assessment_path = (
        _debug_candidate_assessment_path(session)
        if debug
        else _candidate_assessment_path(session)
    )
    assessment_payload = {
        "decision": assessment["decision"],
        "comparison_png_path": assessment.get("comparison_png_path"),
    }
    if "amp_ssim" in assessment:
        assessment_payload["amp_ssim"] = assessment["amp_ssim"]
    if "randomness_contract" in assessment:
        assessment_payload["randomness_contract"] = assessment["randomness_contract"]
    if "blocker_reason" in assessment:
        assessment_payload["blocker_reason"] = assessment["blocker_reason"]
    if "publication_status" in assessment:
        assessment_payload["publication_status"] = assessment["publication_status"]
    if "warning" in assessment:
        assessment_payload["warning"] = assessment["warning"]
    if "crash_excerpt" in assessment:
        assessment_payload["crash_excerpt"] = assessment["crash_excerpt"]
    _write_json(assessment_path, assessment_payload)


def _coerce_subprocess_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _controller_runtime_env() -> dict[str, str]:
    env = dict(os.environ)
    python_dir = str(Path(sys.executable).resolve().parent)
    current_path = env.get("PATH", "")
    parts = [part for part in current_path.split(os.pathsep) if part]
    normalized_python_dir = str(Path(python_dir).resolve())
    filtered_parts = [part for part in parts if str(Path(part).resolve()) != normalized_python_dir]
    env["PATH"] = os.pathsep.join([python_dir, *filtered_parts])
    return env


def _summarize_crash_text(stdout_text: str, stderr_text: str) -> str | None:
    combined = "\n".join(part for part in [stderr_text.strip(), stdout_text.strip()] if part)
    if not combined:
        return None
    return combined.splitlines()[0][:400]


def _run_scored_command(repo_root: Path, command: str, timeout_sec: int) -> dict[str, object]:
    repo_root = repo_root.resolve()
    try:
        completed = subprocess.run(
            ["bash", "-c", command],
            cwd=repo_root,
            env=_controller_runtime_env(),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        return {
            "launcher_status": "completed",
            "exit_code": completed.returncode,
            "timed_out": False,
            "stdout_text": _coerce_subprocess_text(completed.stdout),
            "stderr_text": _coerce_subprocess_text(completed.stderr),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "launcher_status": "timeout",
            "exit_code": 124,
            "timed_out": True,
            "stdout_text": _coerce_subprocess_text(exc.stdout),
            "stderr_text": _coerce_subprocess_text(exc.stderr),
        }


def _write_run_result_artifact(
    session: SessionState,
    launch_result: dict[str, object],
    debug: bool = False,
) -> None:
    run_result = {
        "launcher_status": launch_result["launcher_status"],
        "exit_code": launch_result["exit_code"],
        "timed_out": launch_result["timed_out"],
    }
    run_result_path = (
        _debug_candidate_run_result_path(session)
        if debug
        else _candidate_run_result_path(session)
    )
    _write_json(run_result_path, run_result)


def run_scored_candidate(
    repo_root: Path,
    session: SessionState,
    proposal: dict[str, object],
    *,
    debug: bool = False,
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    accepted_state = _load_accepted_state(session)
    output_root = repo_root / str(proposal["output_root"])
    log_path = repo_root / str(proposal["log_path"])
    comparison_png_path = repo_root / str(proposal["comparison_png_path"])

    launch_result = _run_scored_command(
        repo_root=repo_root,
        command=str(proposal["run_command"]),
        timeout_sec=1770,
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        str(launch_result["stdout_text"]) + str(launch_result["stderr_text"]),
        encoding="utf-8",
    )
    _write_run_result_artifact(session, launch_result, debug=debug)

    if bool(launch_result["timed_out"]):
        return {
            "decision": "TIMEOUT",
            "launcher_status": launch_result["launcher_status"],
            "timed_out": True,
            "exit_code": launch_result["exit_code"],
            "comparison_png_path": str(proposal["comparison_png_path"]),
        }

    if int(launch_result["exit_code"]) == 124:
        return {
            "decision": "TIMEOUT",
            "launcher_status": launch_result["launcher_status"],
            "timed_out": True,
            "exit_code": launch_result["exit_code"],
            "comparison_png_path": str(proposal["comparison_png_path"]),
        }

    if int(launch_result["exit_code"]) != 0:
        return {
            "decision": "CRASH",
            "launcher_status": launch_result["launcher_status"],
            "timed_out": False,
            "exit_code": launch_result["exit_code"],
            "comparison_png_path": str(proposal["comparison_png_path"]),
            "crash_excerpt": _summarize_crash_text(
                str(launch_result["stdout_text"]),
                str(launch_result["stderr_text"]),
            ),
        }

    try:
        randomness_path = _find_randomness_path(output_root)
        randomness_contract = json.loads(randomness_path.read_text(encoding="utf-8"))
    except SystemExit:
        return {
            "decision": "CRASH",
            "launcher_status": launch_result["launcher_status"],
            "timed_out": False,
            "exit_code": launch_result["exit_code"],
            "comparison_png_path": None,
            "warning": "Required randomness contract was missing from the scored run.",
        }

    accepted_randomness = accepted_state.get("accepted_randomness_contract")
    if randomness_contract != accepted_randomness:
        return {
            "decision": "BLOCKED",
            "launcher_status": launch_result["launcher_status"],
            "timed_out": False,
            "exit_code": launch_result["exit_code"],
            "comparison_png_path": str(proposal["comparison_png_path"]),
            "randomness_contract": randomness_contract,
            "blocker_reason": "Scored run randomness contract diverged from accepted session contract.",
        }

    try:
        metrics_path = _find_metrics_path(output_root)
    except SystemExit:
        return {
            "decision": "CRASH",
            "launcher_status": launch_result["launcher_status"],
            "timed_out": False,
            "exit_code": launch_result["exit_code"],
            "comparison_png_path": None,
            "warning": "Required metrics were missing from the scored run.",
        }

    amp_ssim = _extract_amp_ssim(metrics_path)
    publication_status = _read_visual_publication_status(output_root)
    compare_path, compare_status, compare_warning = _resolve_optional_comparison_artifact(
        output_root, publication_status
    )
    final_comparison_png_path: str | None = None
    if compare_path is not None:
        comparison_png_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(compare_path, comparison_png_path)
        final_comparison_png_path = str(proposal["comparison_png_path"])

    assessment = {
        "decision": "KEEP"
        if amp_ssim > float(accepted_state["accepted_amp_ssim"])
        else "DISCARD",
        "amp_ssim": amp_ssim,
        "randomness_contract": randomness_contract,
        "launcher_status": launch_result["launcher_status"],
        "timed_out": False,
        "exit_code": launch_result["exit_code"],
        "comparison_png_path": final_comparison_png_path,
        "publication_status": compare_status,
    }
    if compare_warning is not None:
        assessment["warning"] = compare_warning
    return assessment


def _append_candidate_row(
    session: SessionState,
    accepted_state_before: dict[str, object],
    proposal: dict[str, object],
    assessment: dict[str, object],
) -> None:
    amp_ssim = assessment.get("amp_ssim")
    compared_to_amp = float(accepted_state_before["accepted_amp_ssim"])
    delta = float(amp_ssim) - compared_to_amp if amp_ssim is not None else None
    ref_or_commit = str(
        proposal.get("candidate_commit") or proposal.get("base_ref") or "na"
    )
    with session.results_tsv_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(
            [
                session.session_id,
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                ref_or_commit,
                str(assessment["decision"]).lower(),
                amp_ssim if amp_ssim is not None else "na",
                str(accepted_state_before["accepted_ref"]),
                compared_to_amp,
                delta if delta is not None else "na",
                str(proposal["output_root"]),
                str(assessment.get("comparison_png_path") or "na"),
                str(proposal["run_command"]),
                str(proposal.get("note", "")),
            ]
        )


def apply_candidate_assessment(
    repo_root: Path,
    session: SessionState,
    proposal: dict[str, object],
    assessment: dict[str, object],
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    accepted_state = _load_accepted_state(session)
    _append_candidate_row(session, accepted_state, proposal, assessment)

    decision = str(assessment["decision"]).upper()
    candidate_kind = str(proposal["candidate_kind"])

    if decision == "KEEP":
        next_accepted = dict(accepted_state)
        next_accepted["accepted_ref"] = (
            str(proposal["candidate_commit"])
            if candidate_kind == "source"
            else str(accepted_state["accepted_ref"])
        )
        next_accepted["accepted_amp_ssim"] = float(assessment["amp_ssim"])
        next_accepted["accepted_run_command"] = str(proposal["run_command"])
        next_accepted["accepted_candidate_kind"] = candidate_kind
        next_accepted["accepted_randomness_contract"] = assessment["randomness_contract"]
        next_accepted["accepted_output_root"] = str(proposal["output_root"])
        next_accepted["accepted_comparison_png"] = (
            str(assessment["comparison_png_path"])
            if assessment.get("comparison_png_path")
            else "na"
        )
        _write_accepted_state(session, next_accepted)
        _move_queued_workflow_idea(repo_root, session, decision)
        return next_accepted

    if candidate_kind == "source":
        _cleanup_source_candidate(repo_root, session, proposal)

    _move_queued_workflow_idea(repo_root, session, decision)
    return accepted_state


def resume_session(session_root: Path) -> SessionState:
    return load_session(session_root)


def should_attempt_debug(assessment: dict[str, object], debug_attempts: int) -> bool:
    return str(assessment.get("decision", "")).upper() == "CRASH" and debug_attempts == 0


def _bootstrap_from_accepted_state(
    repo_root: Path,
    session: SessionState,
    bootstrap_path: Path,
) -> dict[str, object]:
    source = _read_json(bootstrap_path)
    accepted_state = dict(source)
    accepted_state["accepted_state_path"] = _relative_to_repo(repo_root, session.accepted_state_path)
    accepted_state["session_id"] = session.session_id
    _write_accepted_state(session, accepted_state)
    _attach_or_reset_session_branch(
        repo_root,
        session.session_branch,
        str(accepted_state["accepted_ref"]),
    )
    return accepted_state


def _expected_session_ref_for_phase(repo_root: Path, session: SessionState) -> str | None:
    if not session.accepted_state_path.exists():
        return None

    accepted_ref = str(_load_accepted_state(session)["accepted_ref"])
    if session.current_phase in {"proposal_pending", "baseline_complete", "completed", "init"}:
        return accepted_ref
    return _git_head(repo_root)


def _load_existing_ready_proposal(session: SessionState, debug: bool = False) -> dict[str, object] | None:
    metadata_path = _debug_candidate_metadata_path(session) if debug else _candidate_metadata_path(session)
    if not metadata_path.exists():
        return None
    metadata = _load_candidate_metadata(metadata_path)
    if metadata.get("status") != "READY":
        return metadata
    return normalize_candidate_proposal(metadata)


def _load_existing_assessment(session: SessionState, debug: bool = False) -> dict[str, object] | None:
    assessment_path = _debug_candidate_assessment_path(session) if debug else _candidate_assessment_path(session)
    if not assessment_path.exists():
        return None
    return _read_json(assessment_path)


def execute_iteration(
    repo_root: Path,
    session: SessionState,
    codex_cmd: str,
    model: str,
    reasoning_effort: str,
) -> str:
    repo_root = repo_root.resolve()
    if session.current_phase in {"debug_running", "debug_complete"}:
        existing_debug = _load_existing_ready_proposal(session, debug=True)
        debug_proposal_or_blocked = existing_debug or run_debug_step(
            repo_root=repo_root,
            session=session,
            codex_cmd=codex_cmd,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        if debug_proposal_or_blocked.get("status") == "BLOCKED":
            _move_queued_workflow_idea(repo_root, session, "BLOCKED")
            session.debug_attempts = 0
            _write_session_json(session)
            _set_phase(session, "completed")
            return "STOP"

        debug_proposal = debug_proposal_or_blocked
        debug_assessment = (
            _load_existing_assessment(session, debug=True)
            if session.current_phase == "debug_complete"
            else None
        )
        if debug_assessment is None:
            debug_assessment = run_scored_candidate(
                repo_root, session, debug_proposal, debug=True
            )
            _write_scored_artifacts(session, debug_assessment, debug=True)
        apply_candidate_assessment(repo_root, session, debug_proposal, debug_assessment)
        session.debug_attempts = 0
        _write_session_json(session)
        if str(debug_assessment["decision"]).upper() in {"KEEP", "DISCARD", "TIMEOUT"}:
            session.iteration += 1
            _set_phase(session, "proposal_pending")
            return "CONTINUE"
        _set_phase(session, "completed")
        return "STOP"

    existing = (
        _load_existing_ready_proposal(session)
        if session.current_phase in {"proposal_complete", "scored_running"}
        else None
    )
    proposal_or_blocked = existing or run_proposal_step(
        repo_root=repo_root,
        session=session,
        codex_cmd=codex_cmd,
        model=model,
        reasoning_effort=reasoning_effort,
    )

    if proposal_or_blocked.get("status") == "BLOCKED":
        _move_queued_workflow_idea(repo_root, session, "BLOCKED")
        _set_phase(session, "completed")
        return "STOP"

    proposal = proposal_or_blocked

    if session.current_phase == "scored_running":
        assessment = _load_existing_assessment(session) or run_scored_candidate(
            repo_root, session, proposal, debug=False
        )
    else:
        _set_phase(session, "scored_running")
        assessment = run_scored_candidate(repo_root, session, proposal, debug=False)
    _write_scored_artifacts(session, assessment, debug=False)

    if should_attempt_debug(assessment, session.debug_attempts):
        session.debug_attempts = 1
        _write_session_json(session)
        existing_debug = (
            _load_existing_ready_proposal(session, debug=True)
            if session.current_phase in {"debug_running", "debug_complete"}
            else None
        )
        debug_proposal_or_blocked = existing_debug or run_debug_step(
            repo_root=repo_root,
            session=session,
            codex_cmd=codex_cmd,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        if debug_proposal_or_blocked.get("status") == "BLOCKED":
            _move_queued_workflow_idea(repo_root, session, "BLOCKED")
            session.debug_attempts = 0
            _write_session_json(session)
            _set_phase(session, "completed")
            return "STOP"

        debug_proposal = debug_proposal_or_blocked
        debug_assessment = (
            _load_existing_assessment(session, debug=True)
            if session.current_phase == "debug_complete"
            else None
        )
        if debug_assessment is None:
            debug_assessment = run_scored_candidate(
                repo_root, session, debug_proposal, debug=True
            )
        _write_scored_artifacts(session, debug_assessment, debug=True)
        apply_candidate_assessment(repo_root, session, debug_proposal, debug_assessment)
        session.debug_attempts = 0
        _write_session_json(session)
        if str(debug_assessment["decision"]).upper() in {"KEEP", "DISCARD", "TIMEOUT"}:
            session.iteration += 1
            _set_phase(session, "proposal_pending")
            return "CONTINUE"
        _set_phase(session, "completed")
        return "STOP"

    apply_candidate_assessment(repo_root, session, proposal, assessment)
    if str(assessment["decision"]).upper() in {"KEEP", "DISCARD", "TIMEOUT"}:
        session.iteration += 1
        _set_phase(session, "proposal_pending")
        return "CONTINUE"
    _set_phase(session, "completed")
    return "STOP"


def run_full_session(
    repo_root: Path,
    session: SessionState,
    *,
    max_iterations: int,
    codex_cmd: str,
    model: str,
    reasoning_effort: str,
    bootstrap_accepted_state: Path | None = None,
) -> SessionState:
    repo_root = repo_root.resolve()
    capture_protected_local_paths(repo_root, session)

    if bootstrap_accepted_state is not None and not session.accepted_state_path.exists():
        _bootstrap_from_accepted_state(repo_root, session, bootstrap_accepted_state)
        _set_phase(session, "proposal_pending")

    if not session.accepted_state_path.exists():
        run_baseline(repo_root, session)
        _set_phase(session, "proposal_pending")
    elif session.current_phase == "baseline_complete":
        _set_phase(session, "proposal_pending")

    expected_ref = _expected_session_ref_for_phase(repo_root, session)
    if expected_ref is not None:
        _ensure_session_checkout_state(repo_root, session, expected_ref=expected_ref)

    while session.iteration < max_iterations and session.current_phase != "completed":
        decision = execute_iteration(
            repo_root=repo_root,
            session=session,
            codex_cmd=codex_cmd,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        if decision == "STOP":
            break

    if session.current_phase != "completed":
        _set_phase(session, "completed")
    return session


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start")
    start.add_argument("--repo-root", default=".")
    start.add_argument("--session-id")
    start.add_argument("--dry-run", action="store_true")
    start.add_argument("--mode", default="full", choices=["baseline-only", "full"])
    start.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    start.add_argument("--codex-cmd", default="codex")
    start.add_argument("--model", default=DEFAULT_MODEL)
    start.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    start.add_argument("--bootstrap-accepted-state")

    resume = subparsers.add_parser("resume")
    resume.add_argument("--session-root", required=True)
    resume.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    resume.add_argument("--codex-cmd", default="codex")
    resume.add_argument("--model", default=DEFAULT_MODEL)
    resume.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "start":
        repo_root = Path(args.repo_root).resolve()
        session = initialize_session(repo_root, session_id=args.session_id)
        if args.dry_run:
            print(json.dumps(_dry_run_payload(session, args.mode), indent=2))
            return 0
        if args.mode == "baseline-only":
            capture_protected_local_paths(repo_root, session)
            run_baseline(repo_root, session)
        else:
            bootstrap = (
                Path(args.bootstrap_accepted_state).resolve()
                if args.bootstrap_accepted_state
                else None
            )
            run_full_session(
                repo_root=repo_root,
                session=session,
                max_iterations=args.max_iterations,
                codex_cmd=args.codex_cmd,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                bootstrap_accepted_state=bootstrap,
            )
        print(json.dumps(asdict(session), indent=2, default=str))
        return 0

    session = load_session(Path(args.session_root))
    run_full_session(
        repo_root=_repo_root_for_session(session.session_root),
        session=session,
        max_iterations=args.max_iterations,
        codex_cmd=args.codex_cmd,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
    )
    print(json.dumps(asdict(session), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
