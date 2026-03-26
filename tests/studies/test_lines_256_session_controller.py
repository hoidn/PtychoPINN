import json
import subprocess
from pathlib import Path

import pytest


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


def test_build_baseline_command_uses_fixed_lines_256_control(tmp_path):
    from scripts.studies.lines_256_session_controller import build_baseline_command, initialize_session

    repo = tmp_path / "repo"
    _init_repo(repo)
    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")

    command = build_baseline_command(session)

    assert command[:3] == [
        "python",
        "scripts/studies/run_lines_256_arch_experiment.py",
        "--output-dir",
    ]
    assert "--fno-modes" in command and "12" in command
    assert "--fno-width" in command and "32" in command
    assert "--fno-blocks" in command and "4" in command
    assert "--no-hybrid-skip-connections" in command
    assert "--hybrid-downsample-op" in command and "stride_conv" in command


def test_harvest_baseline_writes_session_local_accepted_state_and_ledger(tmp_path):
    from scripts.studies.lines_256_session_controller import (
        build_baseline_command,
        harvest_baseline_outputs,
        initialize_session,
    )

    repo = tmp_path / "repo"
    _init_repo(repo)
    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")
    baseline_output = session.outputs_root / "baseline"
    metrics_dir = baseline_output / "runs" / "pinn_hybrid_resnet"
    visuals_dir = baseline_output / "visuals"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "metrics.json").write_text(
        json.dumps({"amp_ssim": 0.91}, indent=2) + "\n",
        encoding="utf-8",
    )
    (metrics_dir / "randomness_contract.json").write_text(
        json.dumps(
            {
                "requested_seed": 3,
                "effective_subsample_seed": 3,
                "effective_lightning_seed": 3,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (visuals_dir / "compare_amp_phase_probe.png").write_text("png", encoding="utf-8")

    result = {
        "status": "completed",
        "exit_code": 0,
        "command": " ".join(build_baseline_command(session)),
    }
    accepted_ref = _git(repo, "rev-parse", "HEAD").stdout.strip()

    accepted_state = harvest_baseline_outputs(
        repo_root=repo,
        session=session,
        accepted_ref=accepted_ref,
        run_result=result,
    )

    assert accepted_state["accepted_ref"] == accepted_ref
    assert accepted_state["accepted_amp_ssim"] == 0.91
    assert accepted_state["accepted_candidate_kind"] == "baseline"
    assert accepted_state["accepted_randomness_contract"]["requested_seed"] == 3
    assert session.accepted_state_path.exists()
    assert session.baseline_result_path.exists()
    ledger_lines = session.results_tsv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(ledger_lines) == 2
    assert "\tbaseline\t0.91\t" in ledger_lines[1]
    assert (
        session.comparison_gallery_dir
        / f"{session.session_id}__baseline__compare_amp_phase_probe.png"
    ).exists()


def test_run_baseline_executes_wrapper_and_updates_session_phase(tmp_path, monkeypatch):
    from scripts.studies.lines_256_session_controller import initialize_session, run_baseline

    repo = tmp_path / "repo"
    _init_repo(repo)
    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")
    real_run = subprocess.run

    def fake_run(command, *, cwd, capture_output, text, check):
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return real_run(
                command,
                cwd=cwd,
                capture_output=capture_output,
                text=text,
                check=check,
            )
        assert cwd == repo
        assert capture_output is True
        assert text is True
        assert check is False
        metrics_dir = session.baseline_output_root / "runs" / "pinn_hybrid_resnet"
        visuals_dir = session.baseline_output_root / "visuals"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        visuals_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "metrics.json").write_text(
            json.dumps({"amp_ssim": 0.88}, indent=2) + "\n",
            encoding="utf-8",
        )
        (metrics_dir / "randomness_contract.json").write_text(
            json.dumps(
                {
                    "requested_seed": 3,
                    "effective_subsample_seed": 3,
                    "effective_lightning_seed": 3,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (visuals_dir / "compare_amp_phase_probe.png").write_text("png", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="baseline ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    accepted_state = run_baseline(repo, session)

    assert accepted_state["accepted_amp_ssim"] == 0.88
    assert session.current_phase == "baseline_complete"
    session_payload = json.loads(session.session_json_path.read_text(encoding="utf-8"))
    assert session_payload["current_phase"] == "baseline_complete"
    assert session.baseline_result_path.exists()


def test_start_baseline_only_dry_run_prints_baseline_command(tmp_path, capsys):
    from scripts.studies.lines_256_session_controller import main

    repo = tmp_path / "repo"
    _init_repo(repo)

    exit_code = main(
        [
            "start",
            "--repo-root",
            str(repo),
            "--session-id",
            "20260325T000000Z",
            "--mode",
            "baseline-only",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "scripts/studies/run_lines_256_arch_experiment.py" in captured.out
    assert "--fno-modes 12" in captured.out
    assert "20260325T000000Z" in captured.out


def test_build_recent_history_summary_limits_to_recent_rows(tmp_path):
    from scripts.studies.lines_256_session_controller import (
        build_recent_history_summary,
        initialize_session,
    )

    repo = tmp_path / "repo"
    _init_repo(repo)
    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")
    with session.results_tsv_path.open("a", encoding="utf-8") as fh:
        fh.write(
            "20260325T000000Z\t2026-03-25T00:00:00Z\tbase\tbaseline\t0.8\tna\tna\tna\tout/a\tpng/a\tcmd a\t\n"
        )
        fh.write(
            "20260325T000000Z\t2026-03-25T00:10:00Z\tcand1\tdiscard\t0.7\tbase\t0.8\t-0.1\tout/b\tpng/b\tcmd b\tnote b\n"
        )
        fh.write(
            "20260325T000000Z\t2026-03-25T00:20:00Z\tcand2\tkeep\t0.9\tbase\t0.8\t0.1\tout/c\tpng/c\tcmd c\tnote c\n"
        )

    summary = build_recent_history_summary(session.session_root, max_rows=2)

    assert len(summary["recent_attempts"]) == 2
    assert [row["ref_or_commit"] for row in summary["recent_attempts"]] == ["cand1", "cand2"]
    assert summary["recent_outcomes"] == ["discard", "keep"]


def test_build_proposal_context_reads_session_local_accepted_state(tmp_path):
    from scripts.studies.lines_256_session_controller import (
        build_proposal_context,
        initialize_session,
    )

    repo = tmp_path / "repo"
    _init_repo(repo)
    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")
    session.accepted_state_path.write_text(
        json.dumps(
            {
                "accepted_ref": "abc123",
                "accepted_amp_ssim": 0.91,
                "accepted_candidate_kind": "baseline",
                "accepted_randomness_contract": {
                    "requested_seed": 3,
                    "effective_subsample_seed": 3,
                    "effective_lightning_seed": 3,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    context = build_proposal_context(session, max_rows=5)

    assert context["accepted_state"]["accepted_ref"] == "abc123"
    assert context["recent_history"]["recent_attempts"] == []
    assert context["proposal_context_path"] == str(
        session.session_root / "proposal_context.json"
    )
    assert (session.session_root / "proposal_context.json").exists()


def test_normalize_candidate_proposal_accepts_run_config_without_commit():
    from scripts.studies.lines_256_session_controller import normalize_candidate_proposal

    proposal = normalize_candidate_proposal(
        {
            "candidate_kind": "run_config",
            "base_ref": "abc123",
            "smoke_command": "python smoke.py",
            "run_command": "python run.py --fno-modes 24",
            "output_root": "outputs/v2/candidate",
            "log_path": "state/v2/candidate.log",
            "comparison_png_path": "outputs/v2/candidate.png",
            "note": "Try 24 global modes",
            "hypothesis": "Higher spectral capacity may improve amp_ssim.",
        }
    )

    assert proposal["candidate_kind"] == "run_config"
    assert "candidate_commit" not in proposal
    assert "candidate_paths_file" not in proposal


def test_normalize_candidate_proposal_requires_commit_for_source():
    from scripts.studies.lines_256_session_controller import normalize_candidate_proposal

    with pytest.raises(SystemExit, match="candidate_commit"):
        normalize_candidate_proposal(
            {
                "candidate_kind": "source",
                "base_ref": "abc123",
                "smoke_command": "python smoke.py",
                "run_command": "python run.py",
                "output_root": "outputs/v2/candidate",
                "log_path": "state/v2/candidate.log",
                "comparison_png_path": "outputs/v2/candidate.png",
                "note": "Edit source",
                "hypothesis": "Source change may improve amp_ssim.",
            }
        )
