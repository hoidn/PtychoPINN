from __future__ import annotations

import argparse
import json
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
    current_phase: str
    iteration: int


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _session_root(repo_root: Path, session_id: str) -> Path:
    return repo_root / "state" / "lines_256_arch_improvement_v2" / "sessions" / session_id


def _results_tsv_path(session_root: Path) -> Path:
    return session_root / "results.tsv"


def _accepted_state_path(session_root: Path) -> Path:
    return session_root / "accepted_state.json"


def _outputs_root(repo_root: Path, session_id: str) -> Path:
    return repo_root / "outputs" / "lines_256_arch_improvement_v2" / "sessions" / session_id


def _comparison_gallery_dir(outputs_root: Path) -> Path:
    return outputs_root / "comparison_pngs"


def _write_results_header(results_tsv_path: Path) -> None:
    results_tsv_path.parent.mkdir(parents=True, exist_ok=True)
    if not results_tsv_path.exists():
        results_tsv_path.write_text(RESULTS_HEADER + "\n", encoding="utf-8")


def _write_session_json(session: SessionState) -> None:
    payload = {
        "session_id": session.session_id,
        "current_phase": session.current_phase,
        "iteration": session.iteration,
        "session_root": str(session.session_root.relative_to(session.session_root.parents[3])),
        "results_tsv_path": str(session.results_tsv_path.relative_to(session.session_root.parents[3])),
        "accepted_state_path": str(
            session.accepted_state_path.relative_to(session.session_root.parents[3])
        ),
        "outputs_root": str(session.outputs_root.relative_to(session.session_root.parents[3])),
        "comparison_gallery_dir": str(
            session.comparison_gallery_dir.relative_to(session.session_root.parents[3])
        ),
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
    repo_root = session_root.parents[3]
    return SessionState(
        session_id=payload["session_id"],
        session_root=session_root,
        session_json_path=session_root / "session.json",
        results_tsv_path=repo_root / payload["results_tsv_path"],
        accepted_state_path=repo_root / payload["accepted_state_path"],
        outputs_root=repo_root / payload["outputs_root"],
        comparison_gallery_dir=repo_root / payload["comparison_gallery_dir"],
        current_phase=payload["current_phase"],
        iteration=int(payload["iteration"]),
    )


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
        session = initialize_session(Path(args.repo_root), session_id=args.session_id)
        if args.dry_run:
            print(json.dumps(asdict(session), indent=2, default=str))
        return 0

    load_session(Path(args.session_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
