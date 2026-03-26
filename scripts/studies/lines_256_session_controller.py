from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


RESULTS_HEADER = (
    "session_id\ttimestamp_utc\tref_or_commit\tdecision\tamp_ssim\t"
    "compared_to_ref\tcompared_to_amp_ssim\tdelta_amp_ssim\toutput_root\t"
    "comparison_png\tcommand_or_source\tnotes"
)


@dataclass
class SessionState:
    session_id: str
    session_root: Path
    session_json_path: Path
    results_tsv_path: Path
    accepted_state_path: Path
    outputs_root: Path
    comparison_gallery_dir: Path
    baseline_output_root: Path
    baseline_log_path: Path
    baseline_result_path: Path
    baseline_comparison_png: Path
    current_phase: str
    iteration: int


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


def _outputs_root(repo_root: Path, session_id: str) -> Path:
    return repo_root / "outputs" / "lines_256_arch_improvement_v2" / "sessions" / session_id


def _comparison_gallery_dir(outputs_root: Path) -> Path:
    return outputs_root / "comparison_pngs"


def _baseline_output_root(outputs_root: Path) -> Path:
    return outputs_root / "baseline"


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
        "session_root": str(session.session_root.relative_to(repo_root)),
        "results_tsv_path": str(session.results_tsv_path.relative_to(repo_root)),
        "accepted_state_path": str(session.accepted_state_path.relative_to(repo_root)),
        "outputs_root": str(session.outputs_root.relative_to(repo_root)),
        "comparison_gallery_dir": str(session.comparison_gallery_dir.relative_to(repo_root)),
        "baseline_output_root": str(session.baseline_output_root.relative_to(repo_root)),
        "baseline_log_path": str(session.baseline_log_path.relative_to(repo_root)),
        "baseline_result_path": str(session.baseline_result_path.relative_to(repo_root)),
        "baseline_comparison_png": str(session.baseline_comparison_png.relative_to(repo_root)),
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
        outputs_root=outputs_root,
        comparison_gallery_dir=_comparison_gallery_dir(outputs_root),
        baseline_output_root=_baseline_output_root(outputs_root),
        baseline_log_path=session_root / f"{session_id}_baseline.log",
        baseline_result_path=session_root / "baseline_run_result.json",
        baseline_comparison_png=_comparison_gallery_dir(outputs_root)
        / f"{session_id}__baseline__compare_amp_phase_probe.png",
        current_phase="init",
        iteration=0,
    )
    session_root.mkdir(parents=True, exist_ok=True)
    session.comparison_gallery_dir.mkdir(parents=True, exist_ok=True)
    _write_results_header(session.results_tsv_path)
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
        outputs_root=repo_root / payload["outputs_root"],
        comparison_gallery_dir=repo_root / payload["comparison_gallery_dir"],
        baseline_output_root=repo_root / payload["baseline_output_root"],
        baseline_log_path=repo_root / payload["baseline_log_path"],
        baseline_result_path=repo_root / payload["baseline_result_path"],
        baseline_comparison_png=repo_root / payload["baseline_comparison_png"],
        current_phase=payload["current_phase"],
        iteration=int(payload["iteration"]),
    )


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
        raise SystemExit(f"Missing baseline metrics under {output_root}")
    return candidates[0]


def _find_randomness_path(output_root: Path) -> Path:
    direct = output_root / "runs" / "pinn_hybrid_resnet" / "randomness_contract.json"
    if direct.exists():
        return direct
    candidates = sorted(output_root.rglob("randomness_contract.json"))
    if not candidates:
        raise SystemExit(f"Missing baseline randomness contract under {output_root}")
    return candidates[0]


def _find_compare_png(output_root: Path) -> Path:
    direct = output_root / "visuals" / "compare_amp_phase_probe.png"
    if direct.exists():
        return direct
    candidates = sorted(output_root.rglob("compare_amp_phase_probe.png"))
    if not candidates:
        raise SystemExit(f"Missing baseline comparison PNG under {output_root}")
    return candidates[0]


def _extract_amp_ssim(metrics_path: Path) -> float:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    amp_ssim = metrics.get("amp_ssim")
    if amp_ssim is None:
        ssim = metrics.get("ssim")
        if isinstance(ssim, list) and len(ssim) >= 1:
            amp_ssim = ssim[0]
    if amp_ssim is None:
        raise SystemExit("Baseline metrics did not expose amp_ssim")
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
                str(session.baseline_output_root.relative_to(repo_root)),
                str(session.baseline_comparison_png.relative_to(repo_root)),
                command,
                "",
            ]
        )


def _set_phase(session: SessionState, phase: str) -> None:
    session.current_phase = phase
    _write_session_json(session)


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

    session.baseline_result_path.write_text(
        json.dumps(run_result, indent=2) + "\n",
        encoding="utf-8",
    )
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
        "accepted_state_path": str(session.accepted_state_path.relative_to(repo_root)),
        "session_id": session.session_id,
        "baseline_output_root": str(session.baseline_output_root.relative_to(repo_root)),
        "baseline_comparison_png": str(session.baseline_comparison_png.relative_to(repo_root)),
        "baseline_command": run_result["command"],
        "accepted_run_command": run_result["command"],
        "accepted_candidate_kind": "baseline",
        "accepted_randomness_contract": randomness_contract,
    }
    session.accepted_state_path.write_text(
        json.dumps(accepted_state, indent=2) + "\n",
        encoding="utf-8",
    )
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
        "log_path": str(session.baseline_log_path.relative_to(repo_root)),
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start")
    start.add_argument("--repo-root", default=".")
    start.add_argument("--session-id")
    start.add_argument("--dry-run", action="store_true")
    start.add_argument("--mode", default="full")

    resume = subparsers.add_parser("resume")
    resume.add_argument("--session-root", required=True)

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
            run_baseline(repo_root, session)
        else:
            print(json.dumps(asdict(session), indent=2, default=str))
        return 0

    load_session(Path(args.session_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
