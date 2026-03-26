from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RESULTS_HEADER = (
    "session_id\ttimestamp_utc\tref_or_commit\tdecision\tamp_ssim\t"
    "compared_to_ref\tcompared_to_amp_ssim\tdelta_amp_ssim\toutput_root\t"
    "comparison_png\tcommand_or_source\tnotes"
)

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


@dataclass
class SessionState:
    session_id: str
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


def _write_results_header(results_tsv_path: Path) -> None:
    results_tsv_path.parent.mkdir(parents=True, exist_ok=True)
    if not results_tsv_path.exists():
        results_tsv_path.write_text(RESULTS_HEADER + "\n", encoding="utf-8")


def _write_session_json(session: SessionState) -> None:
    repo_root = _repo_root_for_session(session.session_root)
    payload = {
        "session_id": session.session_id,
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


def build_recent_history_summary(session_root: Path, max_rows: int = 10) -> dict[str, object]:
    results_tsv_path = _results_tsv_path(session_root)
    if not results_tsv_path.exists():
        return {"recent_attempts": [], "recent_outcomes": []}

    with results_tsv_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        rows = list(reader)

    recent_attempts = rows[-max_rows:] if max_rows > 0 else []
    return {
        "recent_attempts": recent_attempts,
        "recent_outcomes": [row["decision"] for row in recent_attempts],
    }


def build_proposal_context(session: SessionState, max_rows: int = 10) -> dict[str, object]:
    accepted_state = json.loads(session.accepted_state_path.read_text(encoding="utf-8"))
    recent_history = build_recent_history_summary(session.session_root, max_rows=max_rows)
    proposal_context_path = _proposal_context_path(session.session_root)
    payload = {
        "session_id": session.session_id,
        "iteration": session.iteration,
        "accepted_state": accepted_state,
        "recent_history": recent_history,
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
    prompt_path: Path,
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
    prompt_parts.append(prompt_path.read_text(encoding="utf-8"))
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
    build_proposal_context(session, max_rows=10)
    build_candidate_context(session)
    candidate_context_path = _candidate_context_path(session)
    prompt_text = _render_injected_prompt(
        repo_root=repo_root,
        prompt_path=repo_root / "prompts/workflows/lines_256_arch_improvement/experiment_step.md",
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
        prompt_path=repo_root / "prompts/workflows/lines_256_arch_improvement/debug_crash.md",
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
    run_result = {
        "launcher_status": assessment["launcher_status"],
        "exit_code": assessment["exit_code"],
        "timed_out": assessment["timed_out"],
    }
    run_result_path = (
        _debug_candidate_run_result_path(session)
        if debug
        else _candidate_run_result_path(session)
    )
    assessment_path = (
        _debug_candidate_assessment_path(session)
        if debug
        else _candidate_assessment_path(session)
    )
    _write_json(run_result_path, run_result)
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
    _write_json(assessment_path, assessment_payload)


def run_scored_candidate(
    repo_root: Path,
    session: SessionState,
    proposal: dict[str, object],
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    accepted_state = _load_accepted_state(session)
    output_root = repo_root / str(proposal["output_root"])
    log_path = repo_root / str(proposal["log_path"])
    comparison_png_path = repo_root / str(proposal["comparison_png_path"])

    try:
        completed = subprocess.run(
            ["bash", "-lc", str(proposal["run_command"])],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=1770,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = completed.returncode
        timed_out = False
        launcher_status = "completed"
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        returncode = 124
        timed_out = True
        launcher_status = "timeout"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(stdout + stderr, encoding="utf-8")

    if timed_out:
        return {
            "decision": "TIMEOUT",
            "launcher_status": launcher_status,
            "timed_out": True,
            "exit_code": returncode,
            "comparison_png_path": str(proposal["comparison_png_path"]),
        }

    if returncode == 124:
        return {
            "decision": "TIMEOUT",
            "launcher_status": launcher_status,
            "timed_out": True,
            "exit_code": returncode,
            "comparison_png_path": str(proposal["comparison_png_path"]),
        }

    if returncode != 0:
        return {
            "decision": "CRASH",
            "launcher_status": launcher_status,
            "timed_out": False,
            "exit_code": returncode,
            "comparison_png_path": str(proposal["comparison_png_path"]),
        }

    try:
        randomness_path = _find_randomness_path(output_root)
        randomness_contract = json.loads(randomness_path.read_text(encoding="utf-8"))
    except SystemExit:
        return {
            "decision": "BLOCKED",
            "launcher_status": launcher_status,
            "timed_out": False,
            "exit_code": returncode,
            "comparison_png_path": str(proposal["comparison_png_path"]),
            "blocker_reason": "Missing randomness contract from scored run.",
        }

    accepted_randomness = accepted_state.get("accepted_randomness_contract")
    if randomness_contract != accepted_randomness:
        return {
            "decision": "BLOCKED",
            "launcher_status": launcher_status,
            "timed_out": False,
            "exit_code": returncode,
            "comparison_png_path": str(proposal["comparison_png_path"]),
            "randomness_contract": randomness_contract,
            "blocker_reason": "Scored run randomness contract diverged from accepted session contract.",
        }

    try:
        metrics_path = _find_metrics_path(output_root)
        compare_path = _find_compare_png(output_root)
    except SystemExit:
        return {
            "decision": "CRASH",
            "launcher_status": launcher_status,
            "timed_out": False,
            "exit_code": returncode,
            "comparison_png_path": str(proposal["comparison_png_path"]),
        }

    amp_ssim = _extract_amp_ssim(metrics_path)
    comparison_png_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(compare_path, comparison_png_path)

    return {
        "decision": "KEEP"
        if amp_ssim > float(accepted_state["accepted_amp_ssim"])
        else "DISCARD",
        "amp_ssim": amp_ssim,
        "randomness_contract": randomness_contract,
        "launcher_status": launcher_status,
        "timed_out": False,
        "exit_code": returncode,
        "comparison_png_path": str(proposal["comparison_png_path"]),
    }


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
                str(proposal["comparison_png_path"]),
                str(proposal["run_command"]),
                str(proposal.get("note", "")),
            ]
        )


def _checkout_ref(repo_root: Path, ref: str) -> None:
    completed = subprocess.run(
        ["git", "checkout", "--detach", ref],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"Failed to checkout {ref}: {completed.stdout}{completed.stderr}".strip()
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
        next_accepted["accepted_comparison_png"] = str(proposal["comparison_png_path"])
        _write_accepted_state(session, next_accepted)
        return next_accepted

    if candidate_kind == "source":
        _checkout_ref(repo_root, str(proposal["base_ref"]))
        _assert_tracked_paths_preserved(repo_root, session)

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
    return accepted_state


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
    existing = (
        _load_existing_ready_proposal(session)
        if session.current_phase in {"proposal_complete", "scored_running", "debug_running", "debug_complete"}
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
        _set_phase(session, "completed")
        return "STOP"

    proposal = proposal_or_blocked

    if session.current_phase == "scored_running":
        assessment = _load_existing_assessment(session) or run_scored_candidate(repo_root, session, proposal)
    else:
        _set_phase(session, "scored_running")
        assessment = run_scored_candidate(repo_root, session, proposal)
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
            debug_assessment = run_scored_candidate(repo_root, session, debug_proposal)
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
