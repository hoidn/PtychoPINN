import json
import subprocess
from pathlib import Path


RESULTS_HEADER = (
    "session_id\ttimestamp_utc\tref_or_commit\tdecision\tamp_ssim\t"
    "compared_to_ref\tcompared_to_amp_ssim\tdelta_amp_ssim\toutput_root\t"
    "comparison_png\tcommand_or_source\tnotes"
)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(repo: Path) -> str:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "README.md").write_text("test repo\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "initial")
    return _git(repo, "rev-parse", "--show-toplevel").stdout.strip()


def test_controller_initializes_session_root(tmp_path):
    from scripts.studies.lines_256_session_controller import initialize_session

    repo = tmp_path / "repo"
    _init_repo(repo)

    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")

    assert session.session_root == (
        repo / "state" / "lines_256_arch_improvement_v2" / "sessions" / "20260325T000000Z"
    )
    assert (session.session_root / "session.json").exists()
    assert (session.session_root / "results.tsv").exists()


def test_controller_creates_results_ledger_with_exact_header(tmp_path):
    from scripts.studies.lines_256_session_controller import initialize_session

    repo = tmp_path / "repo"
    _init_repo(repo)

    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")

    header = (session.session_root / "results.tsv").read_text(encoding="utf-8").strip()
    assert header == RESULTS_HEADER


def test_controller_keeps_accepted_state_session_local(tmp_path):
    from scripts.studies.lines_256_session_controller import initialize_session

    repo = tmp_path / "repo"
    _init_repo(repo)

    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")

    payload = json.loads((session.session_root / "session.json").read_text(encoding="utf-8"))
    assert payload["accepted_state_path"] == str(
        Path("state/lines_256_arch_improvement_v2/sessions/20260325T000000Z/accepted_state.json")
    )
    assert payload["results_tsv_path"] == str(
        Path("state/lines_256_arch_improvement_v2/sessions/20260325T000000Z/results.tsv")
    )
