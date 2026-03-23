import json
import subprocess
from pathlib import Path


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _init_repo(repo: Path) -> str:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    return _git(repo, "rev-parse", "--show-toplevel").stdout.strip()


def test_handle_candidate_outcome_tolerates_new_unrelated_dirty_paths_on_discard(tmp_path):
    from scripts.studies import lines_256_handle_candidate_outcome as outcome_helper

    repo = tmp_path / "repo"
    _init_repo(repo)

    candidate_file = repo / "candidate.py"
    unrelated_file = repo / "unrelated.md"
    candidate_file.write_text("value = 1\n", encoding="utf-8")
    unrelated_file.write_text("baseline\n", encoding="utf-8")
    _git(repo, "add", "candidate.py", "unrelated.md")
    _git(repo, "commit", "-m", "baseline")
    baseline_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

    session_dir = repo / "state" / "lines_256_arch_improvement"
    state_dir = session_dir / "iter1"
    _write_json(session_dir / "protected_local_paths.json", [])
    _write_json(
        session_dir / "accepted_state.json",
        {"accepted_ref": baseline_sha, "accepted_amp_ssim": 0.8},
    )

    candidate_file.write_text("value = 2\n", encoding="utf-8")
    _git(repo, "add", "candidate.py")
    _git(repo, "commit", "-m", "candidate")
    candidate_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

    unrelated_file.write_text("changed after session start\n", encoding="utf-8")

    candidate_paths_file = state_dir / "candidate_paths.json"
    _write_json(candidate_paths_file, ["candidate.py"])
    _write_json(
        state_dir / "candidate_metadata.json",
        {
            "status": "READY",
            "candidate_commit": candidate_sha,
            "candidate_paths_file": str(candidate_paths_file.relative_to(repo)),
        },
    )
    _write_json(state_dir / "candidate_assessment.json", {"candidate_amp_ssim": 0.79})
    _write_json(state_dir / "initial_candidate_outcome.json", {"initial_outcome": "DISCARD"})

    rc = outcome_helper.main(
        [
            "--mode",
            "initial",
            "--repo-root",
            str(repo),
            "--state-dir",
            str(state_dir.relative_to(repo)),
            "--session-state-dir",
            str(session_dir.relative_to(repo)),
        ]
    )

    assert rc == 0
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == baseline_sha
    assert candidate_file.read_text(encoding="utf-8") == "value = 1\n"
    status_lines = _git(repo, "status", "--short", "--untracked-files=no").stdout.splitlines()
    assert status_lines == [" M unrelated.md"]


def test_handle_candidate_outcome_updates_accepted_state_on_keep(tmp_path):
    from scripts.studies import lines_256_handle_candidate_outcome as outcome_helper

    repo = tmp_path / "repo"
    _init_repo(repo)

    candidate_file = repo / "candidate.py"
    candidate_file.write_text("value = 1\n", encoding="utf-8")
    _git(repo, "add", "candidate.py")
    _git(repo, "commit", "-m", "baseline")
    baseline_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

    session_dir = repo / "state" / "lines_256_arch_improvement"
    state_dir = session_dir / "iter1"
    _write_json(session_dir / "protected_local_paths.json", [])
    _write_json(
        session_dir / "accepted_state.json",
        {"accepted_ref": baseline_sha, "accepted_amp_ssim": 0.8},
    )

    candidate_file.write_text("value = 2\n", encoding="utf-8")
    _git(repo, "add", "candidate.py")
    _git(repo, "commit", "-m", "candidate")
    candidate_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

    candidate_paths_file = state_dir / "candidate_paths.json"
    _write_json(candidate_paths_file, ["candidate.py"])
    _write_json(
        state_dir / "candidate_metadata.json",
        {
            "status": "READY",
            "candidate_commit": candidate_sha,
            "candidate_paths_file": str(candidate_paths_file.relative_to(repo)),
        },
    )
    _write_json(state_dir / "candidate_assessment.json", {"candidate_amp_ssim": 0.91})
    _write_json(state_dir / "initial_candidate_outcome.json", {"initial_outcome": "KEEP"})

    rc = outcome_helper.main(
        [
            "--mode",
            "initial",
            "--repo-root",
            str(repo),
            "--state-dir",
            str(state_dir.relative_to(repo)),
            "--session-state-dir",
            str(session_dir.relative_to(repo)),
        ]
    )

    assert rc == 0
    accepted_state = json.loads((session_dir / "accepted_state.json").read_text(encoding="utf-8"))
    assert accepted_state["accepted_ref"] == candidate_sha
    assert accepted_state["accepted_amp_ssim"] == 0.91
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == candidate_sha
