import argparse
import json
import subprocess
from pathlib import Path


MODE_CONFIG = {
    "initial": {
        "outcome_file": "initial_candidate_outcome.json",
        "outcome_key": "initial_outcome",
        "noop_values": {"BLOCKED"},
        "metadata_file": "candidate_metadata.json",
        "assessment_file": "candidate_assessment.json",
    },
    "debugged": {
        "outcome_file": "debugged_candidate_outcome.json",
        "outcome_key": "debugged_outcome",
        "noop_values": {"NONE"},
        "metadata_file": "debug_candidate_metadata.json",
        "assessment_file": "debug_candidate_assessment.json",
    },
}


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _resolve_path(repo_root: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_root / path


def _git(repo_root: Path, *args: str, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=capture_output,
        text=True,
    )


def _tracked_dirty_paths(repo_root: Path) -> list[str]:
    result = _git(repo_root, "status", "--short", "--untracked-files=no", capture_output=True)
    paths = []
    for line in result.stdout.splitlines():
        line = line.rstrip()
        if line:
            paths.append(line[3:].strip())
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(MODE_CONFIG), required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--session-state-dir", required=True)
    parser.add_argument("--repo-root", default=".")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    state_dir = _resolve_path(repo_root, args.state_dir)
    session_state_dir = _resolve_path(repo_root, args.session_state_dir)
    config = MODE_CONFIG[args.mode]

    outcome = _load_json(state_dir / config["outcome_file"])[config["outcome_key"]]
    if outcome in config["noop_values"]:
        return 0

    metadata = _load_json(state_dir / config["metadata_file"])
    protected = _load_json(session_state_dir / "protected_local_paths.json")
    accepted_path = session_state_dir / "accepted_state.json"
    accepted = _load_json(accepted_path)

    candidate_paths = _load_json(_resolve_path(repo_root, metadata["candidate_paths_file"]))
    overlap = sorted(set(protected) & set(candidate_paths))
    if overlap:
        raise SystemExit(
            "Candidate paths overlap protected local paths and cannot be handled safely:\n"
            + "\n".join(overlap)
        )

    if outcome == "KEEP":
        assessment = _load_json(state_dir / config["assessment_file"])
        accepted["accepted_ref"] = metadata["candidate_commit"]
        accepted["accepted_amp_ssim"] = float(assessment["candidate_amp_ssim"])
        _write_json(accepted_path, accepted)
        return 0

    if outcome not in {"DISCARD", "CRASH"}:
        raise SystemExit(f"Unsupported outcome {outcome!r} for mode {args.mode!r}")

    _git(repo_root, "reset", "--mixed", "HEAD^")
    if candidate_paths:
        _git(repo_root, "restore", "--source=HEAD", "--staged", "--worktree", "--", *candidate_paths)

    current_paths = set(_tracked_dirty_paths(repo_root))
    protected_set = set(protected)
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
            "Candidate outcome handling disturbed protected or candidate-scoped paths:\n"
            + "\n".join(details)
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
