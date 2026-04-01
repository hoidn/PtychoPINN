import json
import os
import subprocess
import sys
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

    def fake_run(command, *, cwd, capture_output, text, check, timeout=None, env=None):
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


def test_build_search_summary_tracks_attempted_values_and_discard_streak(tmp_path):
    from scripts.studies.lines_256_session_controller import (
        build_search_summary,
        initialize_session,
    )

    repo = tmp_path / "repo"
    _init_repo(repo)
    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")
    with session.results_tsv_path.open("a", encoding="utf-8") as fh:
        fh.write(
            "20260325T000000Z\t2026-03-25T00:00:00Z\tcand1\tdiscard\t0.7\tbase\t0.8\t-0.1\tout/a\tpng/a\tpython run.py --fno-modes 24 --fno-width 32 --hybrid-skip-connections --hybrid-resnet-blocks 8\tnote a\n"
        )
        fh.write(
            "20260325T000000Z\t2026-03-25T00:10:00Z\tcand2\ttimeout\tna\tbase\t0.8\tna\tout/b\tpng/b\tpython run.py --fno-modes 32 --fno-width 32 --hybrid-skip-connections --hybrid-resnet-blocks 10\tnote b\n"
        )
        fh.write(
            "20260325T000000Z\t2026-03-25T00:20:00Z\tcand3\tdiscard\t0.75\tbase\t0.8\t-0.05\tout/c\tpng/c\tpython run.py --fno-modes 32 --fno-width 48 --hybrid-skip-connections --hybrid-resnet-blocks 10 --torch-resnet-width 192\tnote c\n"
        )

    summary = build_search_summary(session.session_root)

    assert summary["total_attempts"] == 3
    assert summary["decision_counts"] == {"discard": 2, "timeout": 1}
    assert summary["attempted_values"]["fno_modes"] == ["24", "32"]
    assert summary["attempted_values"]["fno_width"] == ["32", "48"]
    assert summary["attempted_values"]["hybrid_skip_connections"] == ["on"]
    assert summary["attempted_values"]["torch_resnet_width"] == ["192"]
    assert summary["recent_discard_streak"] == 3
    assert "hybrid_encoder_spectral_hidden_scale" in summary["underexplored_knobs"]


def test_build_search_summary_tracks_family_saturation_after_local_discard_cluster(tmp_path):
    from scripts.studies.lines_256_session_controller import (
        build_search_summary,
        initialize_session,
    )

    repo = tmp_path / "repo"
    _init_repo(repo)
    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")
    with session.results_tsv_path.open("a", encoding="utf-8") as fh:
        for i, command in enumerate(
            [
                "python run.py --fno-modes 24 --fno-width 32",
                "python run.py --fno-modes 32 --fno-width 48",
                "python run.py --fno-modes 40 --fno-width 32",
                "python run.py --fno-blocks 6 --fno-modes 32",
                "python run.py --hybrid-resnet-blocks 12 --fno-modes 32",
            ]
        ):
            fh.write(
                f"20260325T000000Z\t2026-03-25T00:{i:02d}:00Z\tcand{i}\tdiscard\t0.7\tbase\t0.8\t-0.1\tout/{i}\tpng/{i}\t{command}\tnote {i}\n"
            )

    summary = build_search_summary(session.session_root)

    assert summary["recent_discard_streak"] == 5
    assert summary["dominant_recent_family"] == "capacity"
    assert summary["family_counts"]["capacity"] == 5
    assert "optimizer_schedule" in summary["preferred_exploration_families"]
    assert "encoder_hidden_scale" in summary["preferred_exploration_families"]


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
    assert context["search_summary"]["total_attempts"] == 0
    assert "fno_modes" in context["search_summary"]["attempted_values"]
    assert "hybrid_encoder_spectral_hidden_scale" in context["search_summary"]["underexplored_knobs"]
    assert context["proposal_mode"] == "exploit"
    assert "Default exploit mode" in context["proposal_mode_reason"]
    assert context["proposal_context_path"] == str(
        session.session_root / "proposal_context.json"
    )
    assert (session.session_root / "proposal_context.json").exists()


def test_build_proposal_context_switches_to_explore_mode_after_local_discard_cluster(tmp_path):
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
                "accepted_ref": _git(repo, "rev-parse", "HEAD").stdout.strip(),
                "accepted_amp_ssim": 0.9,
                "accepted_run_command": "python run.py --fno-modes 32 --fno-width 32 --fno-blocks 5",
                "accepted_candidate_kind": "run_config",
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
    with session.results_tsv_path.open("a", encoding="utf-8") as fh:
        for i, command in enumerate(
            [
                "python run.py --fno-modes 24 --fno-width 32",
                "python run.py --fno-modes 32 --fno-width 48",
                "python run.py --fno-modes 40 --fno-width 32",
                "python run.py --fno-blocks 6 --fno-modes 32",
                "python run.py --hybrid-resnet-blocks 12 --fno-modes 32",
            ]
            ):
                fh.write(
                    f"20260325T000000Z\t2026-03-25T00:{i:02d}:00Z\tcand{i}\tdiscard\t0.7\tbase\t0.8\t-0.1\tout/{i}\tpng/{i}\t{command}\tnote {i}\n"
                )

    context = build_proposal_context(session, max_rows=10)

    assert context["proposal_mode"] == "explore"
    assert "discard streak" in context["proposal_mode_reason"].lower()
    assert context["search_summary"]["dominant_recent_family"] == "capacity"


def test_build_proposal_context_includes_first_active_workflow_queue_item(tmp_path):
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
    active = repo / "docs" / "workflow_queue" / "active"
    active.mkdir(parents=True, exist_ok=True)
    first = active / "2026-03-31-a-first.md"
    second = active / "2026-03-31-z-second.md"
    first.write_text("# First idea\n\nTry the first queued idea.\n", encoding="utf-8")
    second.write_text("# Second idea\n\nTry the second queued idea.\n", encoding="utf-8")

    context = build_proposal_context(session, max_rows=5)

    assert context["queue_priority_active"] is True
    assert context["queued_workflow_idea"]["path"] == str(
        Path("docs/workflow_queue/active/2026-03-31-a-first.md")
    )
    assert "First idea" in context["queued_workflow_idea"]["content"]
    assert "queued workflow idea" in context["proposal_mode_reason"].lower()


def test_build_proposal_context_defaults_workflow_queue_candidate_factory_to_direct(tmp_path):
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
    active = repo / "docs" / "workflow_queue" / "active"
    active.mkdir(parents=True, exist_ok=True)
    (active / "2026-03-31-a-first.md").write_text(
        "# First idea\n\nTry the first queued idea.\n",
        encoding="utf-8",
    )

    context = build_proposal_context(session, max_rows=5)

    assert context["queued_workflow_idea"]["candidate_factory"] == "direct"


def test_build_proposal_context_reads_workflow_queue_candidate_factory_frontmatter(tmp_path):
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
    active = repo / "docs" / "workflow_queue" / "active"
    active.mkdir(parents=True, exist_ok=True)
    (active / "2026-03-31-a-first.md").write_text(
        "---\n"
        "candidate_factory: redesign\n"
        "---\n"
        "# First idea\n\nTry the first queued idea.\n",
        encoding="utf-8",
    )

    context = build_proposal_context(session, max_rows=5)

    assert context["queued_workflow_idea"]["candidate_factory"] == "redesign"


def test_build_proposal_context_reports_no_workflow_queue_item_when_empty(tmp_path):
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

    assert context["queue_priority_active"] is False
    assert context["queued_workflow_idea"] is None


def test_validate_ready_proposal_does_not_reject_same_family_during_explore_mode(tmp_path):
    from scripts.studies.lines_256_session_controller import (
        initialize_session,
        validate_ready_proposal,
    )

    repo = tmp_path / "repo"
    _init_repo(repo)
    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")
    base_ref = _git(repo, "rev-parse", "HEAD").stdout.strip()
    session.accepted_state_path.write_text(
        json.dumps(
            {
                "accepted_ref": base_ref,
                "accepted_amp_ssim": 0.9,
                "accepted_run_command": "python run.py --fno-modes 32 --fno-width 32 --fno-blocks 5",
                "accepted_candidate_kind": "run_config",
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
    smoke_output = repo / "outputs" / "smoke"
    randomness = smoke_output / "runs" / "pinn_hybrid_resnet" / "randomness_contract.json"
    randomness.parent.mkdir(parents=True, exist_ok=True)
    randomness.write_text(
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
    smoke_log = repo / "state" / "smoke.log"
    smoke_log.parent.mkdir(parents=True, exist_ok=True)
    smoke_log.write_text("ok\n", encoding="utf-8")
    with session.results_tsv_path.open("a", encoding="utf-8") as fh:
        for i, command in enumerate(
            [
                "python run.py --fno-modes 24 --fno-width 32",
                "python run.py --fno-modes 32 --fno-width 48",
                "python run.py --fno-modes 40 --fno-width 32",
                "python run.py --fno-blocks 6 --fno-modes 32",
                "python run.py --hybrid-resnet-blocks 12 --fno-modes 32",
            ]
            ):
                fh.write(
                    f"20260325T000000Z\t2026-03-25T00:{i:02d}:00Z\tcand{i}\tdiscard\t0.7\tbase\t0.8\t-0.1\tout/{i}\tpng/{i}\t{command}\tnote {i}\n"
                )

    proposal = {
        "candidate_kind": "run_config",
        "base_ref": base_ref,
        "smoke_output_root": "outputs/smoke",
        "smoke_log_path": "state/smoke.log",
        "run_command": "python run.py --fno-modes 32 --fno-width 64",
        "output_root": "outputs/candidate",
        "log_path": "state/candidate.log",
        "comparison_png_path": "outputs/candidate.png",
        "note": "another capacity bump",
        "hypothesis": "increase width again",
    }

    validate_ready_proposal(repo, session, proposal)


def test_proposal_prompt_paths_switch_between_exploit_and_explore(tmp_path):
    from scripts.studies.lines_256_session_controller import _proposal_prompt_paths

    repo = tmp_path / "repo"
    prompts = repo / "prompts" / "workflows" / "lines_256_arch_improvement"
    prompts.mkdir(parents=True, exist_ok=True)
    for name in (
        "experiment_step_common.md",
        "experiment_step_exploit.md",
        "experiment_step_explore.md",
    ):
        (prompts / name).write_text(f"{name}\n", encoding="utf-8")

    exploit_paths = _proposal_prompt_paths(repo, {"proposal_mode": "exploit"})
    explore_paths = _proposal_prompt_paths(repo, {"proposal_mode": "explore"})

    assert [path.name for path in exploit_paths] == [
        "experiment_step_common.md",
        "experiment_step_exploit.md",
    ]
    assert [path.name for path in explore_paths] == [
        "experiment_step_common.md",
        "experiment_step_explore.md",
    ]


def test_proposal_prompt_paths_fall_back_to_legacy_single_prompt(tmp_path):
    from scripts.studies.lines_256_session_controller import _proposal_prompt_paths

    repo = tmp_path / "repo"
    prompts = repo / "prompts" / "workflows" / "lines_256_arch_improvement"
    prompts.mkdir(parents=True, exist_ok=True)
    (prompts / "experiment_step.md").write_text("legacy prompt\n", encoding="utf-8")

    prompt_paths = _proposal_prompt_paths(repo, {"proposal_mode": "explore"})

    assert [path.name for path in prompt_paths] == ["experiment_step.md"]


def test_direct_candidate_factory_writes_candidate_metadata_and_result(tmp_path, monkeypatch):
    from scripts.studies.lines_256_session_controller import (
        _candidate_metadata_path,
        _candidate_context_path,
        _proposal_result_path,
        build_candidate_context,
        build_proposal_context,
        direct_candidate_factory,
        initialize_session,
    )

    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_minimal_lines256_docs(repo)
    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")
    base_ref = _git(repo, "rev-parse", "HEAD").stdout.strip()
    session.accepted_state_path.write_text(
        json.dumps(
            {
                "accepted_ref": base_ref,
                "accepted_amp_ssim": 0.8,
                "accepted_run_command": "python baseline.py",
                "accepted_candidate_kind": "baseline",
                "accepted_randomness_contract": {"requested_seed": 3},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    proposal_context = build_proposal_context(session, max_rows=5)
    build_candidate_context(session)
    candidate_context_path = _candidate_context_path(session)
    metadata_path = _candidate_metadata_path(session)

    def fake_run_codex_prompt(**kwargs):
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(
                {
                    "status": "READY",
                    "candidate_kind": "run_config",
                    "base_ref": base_ref,
                    "smoke_command": "python smoke.py --fno-modes 24",
                    "smoke_output_root": "outputs/v2/candidate_smoke",
                    "smoke_log_path": "state/v2/candidate_smoke.log",
                    "run_command": "python run.py --fno-modes 24",
                    "output_root": "outputs/v2/candidate",
                    "log_path": "state/v2/candidate.log",
                    "comparison_png_path": "outputs/v2/candidate.png",
                    "note": "Try 24 global modes",
                    "hypothesis": "Higher spectral capacity may improve amp_ssim.",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(["codex", "exec"], 0, stdout="READY\n", stderr="")

    monkeypatch.setattr(
        "scripts.studies.lines_256_session_controller._run_codex_prompt",
        fake_run_codex_prompt,
    )

    proposal = direct_candidate_factory(
        repo_root=repo,
        session=session,
        proposal_context=proposal_context,
        candidate_context_path=candidate_context_path,
        codex_cmd="codex",
        model="gpt-5.4",
        reasoning_effort="high",
    )

    assert proposal["candidate_kind"] == "run_config"
    assert metadata_path.exists()
    proposal_result = json.loads(_proposal_result_path(session).read_text(encoding="utf-8"))
    assert proposal_result["metadata_validated"] is True
    assert proposal_result["retryable"] is False


def test_redesign_candidate_factory_invokes_workflow_and_returns_ready_proposal(
    tmp_path, monkeypatch
):
    from scripts.studies.lines_256_session_controller import (
        _candidate_context_path,
        _candidate_metadata_path,
        _proposal_result_path,
        build_candidate_context,
        build_proposal_context,
        initialize_session,
        redesign_candidate_factory,
    )

    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_minimal_lines256_docs(repo)
    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")
    base_ref = _git(repo, "rev-parse", "HEAD").stdout.strip()
    session.accepted_state_path.write_text(
        json.dumps(
            {
                "accepted_ref": base_ref,
                "accepted_amp_ssim": 0.8,
                "accepted_run_command": "python baseline.py",
                "accepted_candidate_kind": "baseline",
                "accepted_randomness_contract": {"requested_seed": 3},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    active = repo / "docs" / "workflow_queue" / "active"
    active.mkdir(parents=True, exist_ok=True)
    (active / "2026-03-31-redesign.md").write_text(
        "---\n"
        "candidate_factory: redesign\n"
        "---\n"
        "# Redesign idea\n",
        encoding="utf-8",
    )
    proposal_context = build_proposal_context(session, max_rows=5)
    build_candidate_context(session)
    candidate_context_path = _candidate_context_path(session)
    metadata_path = _candidate_metadata_path(session)

    captured = {}

    def fake_run_redesign_workflow(**kwargs):
        captured.update(kwargs)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(
                {
                    "status": "READY",
                    "candidate_kind": "run_config",
                    "base_ref": base_ref,
                    "smoke_command": "python smoke.py --candidate redesign",
                    "smoke_output_root": "outputs/v2/redesign_smoke",
                    "smoke_log_path": "state/v2/redesign_smoke.log",
                    "run_command": "python run.py --candidate redesign",
                    "output_root": "outputs/v2/redesign",
                    "log_path": "state/v2/redesign.log",
                    "comparison_png_path": "outputs/v2/redesign.png",
                    "note": "Try a redesign candidate.",
                    "hypothesis": "A redesign may improve amp_ssim.",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        _proposal_result_path(session).write_text(
            json.dumps(
                {
                    "status": "READY",
                    "candidate_factory": "redesign",
                    "metadata_validated": True,
                    "retryable": False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(["python", "-m", "orchestrator"], 0, "", "")

    monkeypatch.setattr(
        "scripts.studies.lines_256_session_controller._run_redesign_candidate_workflow",
        fake_run_redesign_workflow,
    )

    proposal = redesign_candidate_factory(
        repo_root=repo,
        session=session,
        proposal_context=proposal_context,
        candidate_context_path=candidate_context_path,
    )

    assert proposal["candidate_kind"] == "run_config"
    assert captured["candidate_context_path"] == candidate_context_path
    assert captured["proposal_context"]["queued_workflow_idea"]["candidate_factory"] == "redesign"


def test_run_proposal_step_redesign_factory_failure_before_metadata_is_retryable(
    tmp_path, monkeypatch
):
    from scripts.studies.lines_256_session_controller import (
        _proposal_result_path,
        initialize_session,
        run_proposal_step,
    )

    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_minimal_lines256_docs(repo)
    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")
    base_ref = _git(repo, "rev-parse", "HEAD").stdout.strip()
    session.accepted_state_path.write_text(
        json.dumps(
            {
                "accepted_ref": base_ref,
                "accepted_amp_ssim": 0.8,
                "accepted_run_command": "python baseline.py",
                "accepted_candidate_kind": "baseline",
                "accepted_randomness_contract": {"requested_seed": 3},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    active = repo / "docs" / "workflow_queue" / "active"
    active.mkdir(parents=True, exist_ok=True)
    idea = active / "2026-03-31-redesign.md"
    idea.write_text(
        "---\n"
        "candidate_factory: redesign\n"
        "---\n"
        "# Redesign idea\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "scripts.studies.lines_256_session_controller._run_redesign_candidate_workflow",
        lambda **kwargs: subprocess.CompletedProcess(
            ["python", "-m", "orchestrator"],
            1,
            "",
            "workflow failed before writing metadata",
        ),
    )

    result = run_proposal_step(
        repo_root=repo,
        session=session,
        codex_cmd="codex",
        model="gpt-5.4",
        reasoning_effort="high",
    )

    assert result == {"status": "RETRYABLE_FAILURE"}
    assert idea.exists()
    proposal_result = json.loads(_proposal_result_path(session).read_text(encoding="utf-8"))
    assert proposal_result["retryable"] is True


@pytest.mark.parametrize(
    ("decision", "dest_dir"),
    [
        ("KEEP", "accepted"),
        ("DISCARD", "discarded"),
        ("TIMEOUT", "timed_out"),
        ("CRASH", "crashed"),
    ],
)
def test_apply_candidate_assessment_moves_active_workflow_queue_item_on_terminal_outcome(
    tmp_path, decision, dest_dir
):
    from scripts.studies.lines_256_session_controller import (
        _candidate_context_path,
        apply_candidate_assessment,
        initialize_session,
    )

    repo = tmp_path / "repo"
    _init_repo(repo)
    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")
    base_ref = _git(repo, "rev-parse", "HEAD").stdout.strip()
    session.accepted_state_path.write_text(
        json.dumps(
            {
                "accepted_ref": base_ref,
                "accepted_amp_ssim": 0.8,
                "accepted_run_command": "python baseline.py",
                "accepted_candidate_kind": "baseline",
                "accepted_randomness_contract": {"requested_seed": 3},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    active = repo / "docs" / "workflow_queue" / "active"
    active.mkdir(parents=True, exist_ok=True)
    idea = active / "2026-03-31-queued-idea.md"
    idea.write_text("# Queued idea\n", encoding="utf-8")
    _candidate_context_path(session).parent.mkdir(parents=True, exist_ok=True)
    _candidate_context_path(session).write_text(
        json.dumps(
            {
                "queued_workflow_idea": {
                    "path": "docs/workflow_queue/active/2026-03-31-queued-idea.md",
                    "content": "# Queued idea\n",
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    accepted_state = apply_candidate_assessment(
        repo_root=repo,
        session=session,
        proposal={
            "candidate_kind": "run_config",
            "base_ref": base_ref,
            "run_command": "python run.py --fno-modes 24",
            "output_root": "outputs/v2/candidate-queue",
            "comparison_png_path": "outputs/v2/candidate-queue.png",
            "note": "queued idea",
            "hypothesis": "queue-backed candidate",
        },
        assessment={
            "decision": decision,
            "amp_ssim": 0.91 if decision == "KEEP" else 0.75,
            "randomness_contract": {"requested_seed": 3},
        },
    )

    assert not idea.exists()
    moved = repo / "docs" / "workflow_queue" / dest_dir / idea.name
    assert moved.exists()
    if decision == "KEEP":
        assert accepted_state["accepted_amp_ssim"] == 0.91


def test_execute_iteration_moves_active_workflow_queue_item_to_blocked(tmp_path, monkeypatch):
    from scripts.studies.lines_256_session_controller import (
        build_proposal_context,
        execute_iteration,
        initialize_session,
    )

    repo = tmp_path / "repo"
    _init_repo(repo)
    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")
    active = repo / "docs" / "workflow_queue" / "active"
    active.mkdir(parents=True, exist_ok=True)
    idea = active / "2026-03-31-queued-idea.md"
    idea.write_text("# Queued idea\n", encoding="utf-8")
    session.accepted_state_path.write_text(
        json.dumps(
            {
                "accepted_ref": _git(repo, "rev-parse", "HEAD").stdout.strip(),
                "accepted_amp_ssim": 0.8,
                "accepted_run_command": "python baseline.py",
                "accepted_candidate_kind": "baseline",
                "accepted_randomness_contract": {"requested_seed": 3},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    build_proposal_context(session, max_rows=5)

    monkeypatch.setattr(
        "scripts.studies.lines_256_session_controller.run_proposal_step",
        lambda **kwargs: {"status": "BLOCKED", "blocker_reason": "idea not viable"},
    )

    decision = execute_iteration(
        repo_root=repo,
        session=session,
        codex_cmd="codex",
        model="gpt-5.4",
        reasoning_effort="high",
    )

    assert decision == "STOP"
    assert not idea.exists()
    moved = repo / "docs" / "workflow_queue" / "blocked" / idea.name
    assert moved.exists()


def test_execute_iteration_invalid_workflow_queue_candidate_factory_yields_without_consuming_item(
    tmp_path, monkeypatch
):
    from scripts.studies.lines_256_session_controller import execute_iteration, initialize_session

    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_minimal_lines256_docs(repo)
    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")
    active = repo / "docs" / "workflow_queue" / "active"
    active.mkdir(parents=True, exist_ok=True)
    idea = active / "2026-03-31-queued-idea.md"
    idea.write_text(
        "---\n"
        "candidate_factory: unsupported\n"
        "---\n"
        "# Queued idea\n",
        encoding="utf-8",
    )
    session.accepted_state_path.write_text(
        json.dumps(
            {
                "accepted_ref": _git(repo, "rev-parse", "HEAD").stdout.strip(),
                "accepted_amp_ssim": 0.8,
                "accepted_run_command": "python baseline.py",
                "accepted_candidate_kind": "baseline",
                "accepted_randomness_contract": {"requested_seed": 3},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "scripts.studies.lines_256_session_controller._run_codex_prompt",
        lambda *args, **kwargs: pytest.fail("provider should not run for invalid candidate_factory"),
    )

    decision = execute_iteration(
        repo_root=repo,
        session=session,
        codex_cmd="codex",
        model="gpt-5.4",
        reasoning_effort="high",
    )

    assert decision == "YIELD"
    assert idea.exists()
    proposal_result = json.loads(
        (session.session_root / "iterations" / "000" / "proposal_result.json").read_text(
            encoding="utf-8"
        )
    )
    assert proposal_result["retryable"] is True


def test_normalize_candidate_proposal_accepts_run_config_without_commit():
    from scripts.studies.lines_256_session_controller import normalize_candidate_proposal

    proposal = normalize_candidate_proposal(
        {
            "candidate_kind": "run_config",
            "base_ref": "abc123",
            "smoke_command": "python smoke.py",
            "smoke_output_root": "outputs/v2/candidate_smoke",
            "smoke_log_path": "state/v2/candidate_smoke.log",
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
                "smoke_output_root": "outputs/v2/source_smoke",
                "smoke_log_path": "state/v2/source_smoke.log",
                "run_command": "python run.py",
                "output_root": "outputs/v2/candidate",
                "log_path": "state/v2/candidate.log",
                "comparison_png_path": "outputs/v2/candidate.png",
                "note": "Edit source",
                "hypothesis": "Source change may improve amp_ssim.",
            }
        )


def test_source_candidate_discard_resets_to_accepted_ref(tmp_path):
    from scripts.studies.lines_256_session_controller import (
        apply_candidate_assessment,
        initialize_session,
    )

    repo = tmp_path / "repo"
    _init_repo(repo)
    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")
    (repo / "runtime.py").write_text("print('runtime base')\n", encoding="utf-8")
    _git(repo, "add", "runtime.py")
    _git(repo, "commit", "-m", "runtime base")
    base_ref = _git(repo, "rev-parse", "HEAD").stdout.strip()
    session.accepted_state_path.write_text(
        json.dumps(
            {
                "accepted_ref": base_ref,
                "accepted_amp_ssim": 0.8,
                "accepted_run_command": "python baseline.py",
                "accepted_candidate_kind": "baseline",
                "accepted_randomness_contract": {"requested_seed": 3},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (repo / "model.py").write_text("print('candidate')\n", encoding="utf-8")
    _git(repo, "add", "model.py")
    _git(repo, "commit", "-m", "candidate")
    candidate_commit = _git(repo, "rev-parse", "HEAD").stdout.strip()
    candidate_paths_file = session.session_root / "candidate_paths.json"
    candidate_paths_file.write_text(
        json.dumps(["model.py"], indent=2) + "\n",
        encoding="utf-8",
    )

    accepted_state = apply_candidate_assessment(
        repo_root=repo,
        session=session,
        proposal={
            "candidate_kind": "source",
            "base_ref": base_ref,
            "candidate_commit": candidate_commit,
            "candidate_paths_file": str(candidate_paths_file.relative_to(repo)),
            "run_command": "python run.py",
            "output_root": "outputs/v2/candidate1",
            "comparison_png_path": "outputs/v2/candidate1.png",
            "note": "candidate",
            "hypothesis": "test source candidate",
        },
        assessment={
            "decision": "DISCARD",
            "amp_ssim": 0.7,
            "randomness_contract": {"requested_seed": 3},
        },
    )

    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == base_ref
    assert accepted_state["accepted_ref"] == base_ref


def test_source_candidate_discard_preserves_unrelated_runtime_edits(tmp_path):
    from scripts.studies.lines_256_session_controller import (
        apply_candidate_assessment,
        initialize_session,
    )

    repo = tmp_path / "repo"
    _init_repo(repo)
    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")
    (repo / "runtime.py").write_text("print('runtime base')\n", encoding="utf-8")
    _git(repo, "add", "runtime.py")
    _git(repo, "commit", "-m", "runtime base")
    base_ref = _git(repo, "rev-parse", "HEAD").stdout.strip()
    session.accepted_state_path.write_text(
        json.dumps(
            {
                "accepted_ref": base_ref,
                "accepted_amp_ssim": 0.8,
                "accepted_run_command": "python baseline.py",
                "accepted_candidate_kind": "baseline",
                "accepted_randomness_contract": {"requested_seed": 3},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (repo / "runtime.py").write_text("print('runtime v1')\n", encoding="utf-8")
    _git(repo, "add", "runtime.py")
    _git(repo, "commit", "-m", "runtime change")

    (repo / "model.py").write_text("print('candidate')\n", encoding="utf-8")
    _git(repo, "add", "model.py")
    _git(repo, "commit", "-m", "candidate")
    candidate_commit = _git(repo, "rev-parse", "HEAD").stdout.strip()

    (repo / "runtime.py").write_text("print('runtime hotfix')\n", encoding="utf-8")
    candidate_paths_file = session.session_root / "candidate_paths.json"
    candidate_paths_file.write_text(
        json.dumps(["model.py"], indent=2) + "\n",
        encoding="utf-8",
    )

    accepted_state = apply_candidate_assessment(
        repo_root=repo,
        session=session,
        proposal={
            "candidate_kind": "source",
            "base_ref": base_ref,
            "candidate_commit": candidate_commit,
            "candidate_paths_file": str(candidate_paths_file.relative_to(repo)),
            "run_command": "python run.py",
            "output_root": "outputs/v2/candidate1",
            "comparison_png_path": "outputs/v2/candidate1.png",
            "note": "candidate",
            "hypothesis": "test source candidate",
        },
        assessment={
            "decision": "DISCARD",
            "amp_ssim": 0.7,
            "randomness_contract": {"requested_seed": 3},
        },
    )

    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == base_ref
    assert accepted_state["accepted_ref"] == base_ref
    status = _git(repo, "status", "--short", "--untracked-files=no").stdout.splitlines()
    assert status == [" M runtime.py"]


def test_run_config_discard_does_not_move_head(tmp_path):
    from scripts.studies.lines_256_session_controller import (
        apply_candidate_assessment,
        initialize_session,
    )

    repo = tmp_path / "repo"
    _init_repo(repo)
    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")
    base_ref = _git(repo, "rev-parse", "HEAD").stdout.strip()
    session.accepted_state_path.write_text(
        json.dumps(
            {
                "accepted_ref": base_ref,
                "accepted_amp_ssim": 0.8,
                "accepted_run_command": "python baseline.py",
                "accepted_candidate_kind": "baseline",
                "accepted_randomness_contract": {"requested_seed": 3},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    accepted_state = apply_candidate_assessment(
        repo_root=repo,
        session=session,
        proposal={
            "candidate_kind": "run_config",
            "base_ref": base_ref,
            "run_command": "python run.py --fno-modes 24",
            "output_root": "outputs/v2/candidate2",
            "comparison_png_path": "outputs/v2/candidate2.png",
            "note": "run config",
            "hypothesis": "test run config",
        },
        assessment={
            "decision": "DISCARD",
            "amp_ssim": 0.75,
            "randomness_contract": {"requested_seed": 3},
        },
    )

    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == base_ref
    assert accepted_state["accepted_ref"] == base_ref


def test_source_candidate_keep_advances_accepted_ref(tmp_path):
    from scripts.studies.lines_256_session_controller import (
        apply_candidate_assessment,
        initialize_session,
    )

    repo = tmp_path / "repo"
    _init_repo(repo)
    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")
    base_ref = _git(repo, "rev-parse", "HEAD").stdout.strip()
    session.accepted_state_path.write_text(
        json.dumps(
            {
                "accepted_ref": base_ref,
                "accepted_amp_ssim": 0.8,
                "accepted_run_command": "python baseline.py",
                "accepted_candidate_kind": "baseline",
                "accepted_randomness_contract": {"requested_seed": 3},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (repo / "model.py").write_text("print('better')\n", encoding="utf-8")
    _git(repo, "add", "model.py")
    _git(repo, "commit", "-m", "better candidate")
    candidate_commit = _git(repo, "rev-parse", "HEAD").stdout.strip()

    accepted_state = apply_candidate_assessment(
        repo_root=repo,
        session=session,
        proposal={
            "candidate_kind": "source",
            "base_ref": base_ref,
            "candidate_commit": candidate_commit,
            "run_command": "python run.py",
            "output_root": "outputs/v2/candidate3",
            "comparison_png_path": "outputs/v2/candidate3.png",
            "note": "source keep",
            "hypothesis": "keep source candidate",
        },
        assessment={
            "decision": "KEEP",
            "amp_ssim": 0.9,
            "randomness_contract": {"requested_seed": 3},
        },
    )

    assert accepted_state["accepted_ref"] == candidate_commit
    assert accepted_state["accepted_amp_ssim"] == 0.9


def test_run_config_keep_preserves_ref_and_updates_run_command(tmp_path):
    from scripts.studies.lines_256_session_controller import (
        apply_candidate_assessment,
        initialize_session,
    )

    repo = tmp_path / "repo"
    _init_repo(repo)
    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")
    base_ref = _git(repo, "rev-parse", "HEAD").stdout.strip()
    session.accepted_state_path.write_text(
        json.dumps(
            {
                "accepted_ref": base_ref,
                "accepted_amp_ssim": 0.8,
                "accepted_run_command": "python baseline.py",
                "accepted_candidate_kind": "baseline",
                "accepted_randomness_contract": {"requested_seed": 3},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    accepted_state = apply_candidate_assessment(
        repo_root=repo,
        session=session,
        proposal={
            "candidate_kind": "run_config",
            "base_ref": base_ref,
            "run_command": "python run.py --fno-modes 24",
            "output_root": "outputs/v2/candidate4",
            "comparison_png_path": "outputs/v2/candidate4.png",
            "note": "run config keep",
            "hypothesis": "keep run config candidate",
        },
        assessment={
            "decision": "KEEP",
            "amp_ssim": 0.91,
            "randomness_contract": {"requested_seed": 3},
        },
    )

    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == base_ref
    assert accepted_state["accepted_ref"] == base_ref
    assert accepted_state["accepted_run_command"] == "python run.py --fno-modes 24"
    assert accepted_state["accepted_amp_ssim"] == 0.91


def test_run_config_candidate_scored_run_harvests_outputs(tmp_path, monkeypatch):
    from scripts.studies.lines_256_session_controller import (
        initialize_session,
        run_scored_candidate,
    )

    repo = tmp_path / "repo"
    _init_repo(repo)
    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")
    base_ref = _git(repo, "rev-parse", "HEAD").stdout.strip()
    session.accepted_state_path.write_text(
        json.dumps(
            {
                "accepted_ref": base_ref,
                "accepted_amp_ssim": 0.8,
                "accepted_run_command": "python baseline.py",
                "accepted_candidate_kind": "baseline",
                "accepted_randomness_contract": {"requested_seed": 3},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    proposal = {
        "candidate_kind": "run_config",
        "base_ref": base_ref,
        "run_command": "python run.py --fno-modes 24",
        "output_root": "outputs/v2/candidate5",
        "log_path": "state/v2/candidate5.log",
        "comparison_png_path": "outputs/v2/candidate5.png",
        "note": "scored run",
        "hypothesis": "score run config candidate",
    }
    real_run = subprocess.run

    def fake_run(command, *, cwd, capture_output, text, check, timeout=None, env=None):
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return real_run(
                command,
                cwd=cwd,
                capture_output=capture_output,
                text=text,
                check=check,
            )
        assert command[:3] == ["bash", "-c", "python run.py --fno-modes 24"]
        output_root = repo / proposal["output_root"]
        metrics_dir = output_root / "runs" / "pinn_hybrid_resnet"
        visuals_dir = output_root / "visuals"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        visuals_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "metrics.json").write_text(
            json.dumps({"amp_ssim": 0.82}, indent=2) + "\n",
            encoding="utf-8",
        )
        (metrics_dir / "randomness_contract.json").write_text(
            json.dumps({"requested_seed": 3}, indent=2) + "\n",
            encoding="utf-8",
        )
        (visuals_dir / "compare_amp_phase_probe.png").write_text("png", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="scored ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assessment = run_scored_candidate(repo, session, proposal)

    assert assessment["decision"] == "KEEP"
    assert assessment["amp_ssim"] == 0.82
    assert assessment["comparison_png_path"] == proposal["comparison_png_path"]
    assert (repo / proposal["log_path"]).exists()


def test_run_scored_candidate_marks_source_provenance_mismatch_invalid_execution(
    tmp_path, monkeypatch
):
    from scripts.studies.lines_256_session_controller import (
        initialize_session,
        run_scored_candidate,
    )

    repo = tmp_path / "repo"
    _init_repo(repo)
    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")
    base_ref = _git(repo, "rev-parse", "HEAD").stdout.strip()
    session.accepted_state_path.write_text(
        json.dumps(
            {
                "accepted_ref": base_ref,
                "accepted_amp_ssim": 0.8,
                "accepted_run_command": "python baseline.py",
                "accepted_candidate_kind": "baseline",
                "accepted_randomness_contract": {"requested_seed": 3},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    proposal = {
        "candidate_kind": "source",
        "base_ref": base_ref,
        "candidate_commit": base_ref,
        "candidate_paths_file": "state/v2/source_paths.json",
        "run_command": "python run.py --source-candidate",
        "output_root": "outputs/v2/source-provenance-mismatch",
        "log_path": "state/v2/source-provenance-mismatch.log",
        "comparison_png_path": "outputs/v2/source-provenance-mismatch.png",
        "note": "source run with foreign provenance",
        "hypothesis": "source candidate should not score if runtime provenance points elsewhere",
    }

    def fake_run(command, *, cwd, capture_output, text, check, timeout=None, env=None):
        assert command[:3] == ["bash", "-c", "python run.py --source-candidate"]
        output_root = repo / proposal["output_root"]
        metrics_dir = output_root / "runs" / "pinn_hybrid_resnet"
        visuals_dir = output_root / "visuals"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        visuals_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "metrics.json").write_text(
            json.dumps({"amp_ssim": 0.81}, indent=2) + "\n",
            encoding="utf-8",
        )
        (metrics_dir / "randomness_contract.json").write_text(
            json.dumps({"requested_seed": 3}, indent=2) + "\n",
            encoding="utf-8",
        )
        (visuals_dir / "compare_amp_phase_probe.png").write_text("png", encoding="utf-8")
        wrapper_invocation = {
            "cwd": str(repo),
            "extra": {
                "runtime_provenance": {
                    "python_executable": "/usr/bin/python3",
                    "cwd": str(repo),
                    "pythonpath": str(tmp_path / "foreign_repo"),
                    "ptycho_torch_file": str(tmp_path / "foreign_repo" / "ptycho_torch" / "__init__.py"),
                }
            },
        }
        runner_invocation = {
            "cwd": str(repo),
            "extra": {
                "runtime_provenance": {
                    "python_executable": "/usr/bin/python3",
                    "cwd": str(repo),
                    "pythonpath": str(tmp_path / "foreign_repo"),
                    "ptycho_torch_file": str(tmp_path / "foreign_repo" / "ptycho_torch" / "__init__.py"),
                }
            },
        }
        (output_root / "invocation.json").write_text(
            json.dumps(wrapper_invocation, indent=2) + "\n",
            encoding="utf-8",
        )
        (metrics_dir / "invocation.json").write_text(
            json.dumps(runner_invocation, indent=2) + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="scored ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assessment = run_scored_candidate(repo, session, proposal)

    assert assessment["decision"] == "INVALID_EXECUTION"
    assert "integrity_reasons" in assessment
    assert any("PYTHONPATH" in reason or "ptycho_torch" in reason for reason in assessment["integrity_reasons"])


def test_run_scored_candidate_warns_on_exact_source_tie_with_valid_provenance(
    tmp_path, monkeypatch
):
    from scripts.studies.lines_256_session_controller import (
        initialize_session,
        run_scored_candidate,
    )

    repo = tmp_path / "repo"
    _init_repo(repo)
    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")
    base_ref = _git(repo, "rev-parse", "HEAD").stdout.strip()
    session.accepted_state_path.write_text(
        json.dumps(
            {
                "accepted_ref": base_ref,
                "accepted_amp_ssim": 0.8,
                "accepted_run_command": "python baseline.py",
                "accepted_candidate_kind": "baseline",
                "accepted_randomness_contract": {"requested_seed": 3},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    proposal = {
        "candidate_kind": "source",
        "base_ref": base_ref,
        "candidate_commit": base_ref,
        "candidate_paths_file": "state/v2/source_paths.json",
        "run_command": "python run.py --source-tie",
        "output_root": "outputs/v2/source-tie",
        "log_path": "state/v2/source-tie.log",
        "comparison_png_path": "outputs/v2/source-tie.png",
        "note": "source run with exact tie",
        "hypothesis": "exact source ties should be flagged as suspicious",
    }

    def fake_run(command, *, cwd, capture_output, text, check, timeout=None, env=None):
        assert command[:3] == ["bash", "-c", "python run.py --source-tie"]
        output_root = repo / proposal["output_root"]
        metrics_dir = output_root / "runs" / "pinn_hybrid_resnet"
        visuals_dir = output_root / "visuals"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        visuals_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "metrics.json").write_text(
            json.dumps({"amp_ssim": 0.8}, indent=2) + "\n",
            encoding="utf-8",
        )
        (metrics_dir / "randomness_contract.json").write_text(
            json.dumps({"requested_seed": 3}, indent=2) + "\n",
            encoding="utf-8",
        )
        (visuals_dir / "compare_amp_phase_probe.png").write_text("png", encoding="utf-8")
        provenance = {
            "python_executable": "/usr/bin/python3",
            "cwd": str(repo.resolve()),
            "pythonpath": str(repo.resolve()),
            "ptycho_torch_file": str(repo / "ptycho_torch" / "__init__.py"),
        }
        (output_root / "invocation.json").write_text(
            json.dumps({"cwd": str(repo.resolve()), "extra": {"runtime_provenance": provenance}}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        (metrics_dir / "invocation.json").write_text(
            json.dumps({"cwd": str(repo.resolve()), "extra": {"runtime_provenance": provenance}}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="scored ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assessment = run_scored_candidate(repo, session, proposal)

    assert assessment["decision"] == "DISCARD"
    assert "suspicious_tie_warning" in assessment


def test_run_scored_candidate_treats_optional_visual_fallback_as_advisory(tmp_path, monkeypatch):
    from scripts.studies.lines_256_session_controller import (
        initialize_session,
        run_scored_candidate,
    )

    repo = tmp_path / "repo"
    _init_repo(repo)
    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")
    base_ref = _git(repo, "rev-parse", "HEAD").stdout.strip()
    session.accepted_state_path.write_text(
        json.dumps(
            {
                "accepted_ref": base_ref,
                "accepted_amp_ssim": 0.8,
                "accepted_run_command": "python baseline.py",
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
    proposal = {
        "candidate_kind": "run_config",
        "base_ref": base_ref,
        "run_command": "python run.py --fno-modes 24",
        "output_root": "outputs/v2/candidate5-fallback",
        "log_path": "state/v2/candidate5-fallback.log",
        "comparison_png_path": "outputs/v2/candidate5-fallback.png",
        "note": "scored run with optional visual fallback",
        "hypothesis": "metrics should govern scoring even if probe-inclusive publication falls back",
    }

    def fake_run(command, *, cwd, capture_output, text, check, timeout=None, env=None):
        assert command[:3] == ["bash", "-c", "python run.py --fno-modes 24"]
        output_root = repo / proposal["output_root"]
        metrics_dir = output_root / "runs" / "pinn_hybrid_resnet"
        visuals_dir = output_root / "visuals"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        visuals_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "metrics.json").write_text(
            json.dumps({"amp_ssim": 0.82}, indent=2) + "\n",
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
        (visuals_dir / "compare_amp_phase.png").write_text("plain-png", encoding="utf-8")
        (output_root / "visual_publication_status.json").write_text(
            json.dumps(
                {
                    "status": "fallback_plain_compare",
                    "published_compare_path": "visuals/compare_amp_phase_probe.png",
                    "warning": "Probe-inclusive compare was unavailable; plain compare copied instead.",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="scored ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assessment = run_scored_candidate(repo, session, proposal)

    assert assessment["decision"] == "KEEP"
    assert assessment["publication_status"] == "fallback_plain_compare"
    assert assessment["comparison_png_path"] == proposal["comparison_png_path"]
    assert "warning" in assessment
    assert (repo / proposal["comparison_png_path"]).read_text(encoding="utf-8") == "plain-png"


def test_run_scored_candidate_timeout_is_classified_without_git_reset(tmp_path, monkeypatch):
    from scripts.studies.lines_256_session_controller import (
        initialize_session,
        run_scored_candidate,
    )

    repo = tmp_path / "repo"
    _init_repo(repo)
    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")
    base_ref = _git(repo, "rev-parse", "HEAD").stdout.strip()
    session.accepted_state_path.write_text(
        json.dumps(
            {
                "accepted_ref": base_ref,
                "accepted_amp_ssim": 0.8,
                "accepted_run_command": "python baseline.py",
                "accepted_candidate_kind": "baseline",
                "accepted_randomness_contract": {"requested_seed": 3},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    proposal = {
        "candidate_kind": "run_config",
        "base_ref": base_ref,
        "run_command": "python run.py --slow",
        "output_root": "outputs/v2/timeout",
        "log_path": "state/v2/timeout.log",
        "comparison_png_path": "outputs/v2/timeout.png",
        "note": "timeout",
        "hypothesis": "slow run",
    }
    real_run = subprocess.run

    def fake_run(command, *, cwd, capture_output, text, check, timeout=None, env=None):
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return real_run(
                command,
                cwd=cwd,
                capture_output=capture_output,
                text=text,
                check=check,
            )
        assert command[:3] == ["bash", "-c", "python run.py --slow"]
        return subprocess.CompletedProcess(command, 124, stdout="", stderr="timed out")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assessment = run_scored_candidate(repo, session, proposal)

    assert assessment["decision"] == "TIMEOUT"
    assert assessment["timed_out"] is True
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == base_ref


def test_run_scored_candidate_timeout_from_exception_decodes_partial_output(tmp_path, monkeypatch):
    from scripts.studies.lines_256_session_controller import (
        initialize_session,
        run_scored_candidate,
    )

    repo = tmp_path / "repo"
    _init_repo(repo)
    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")
    base_ref = _git(repo, "rev-parse", "HEAD").stdout.strip()
    session.accepted_state_path.write_text(
        json.dumps(
            {
                "accepted_ref": base_ref,
                "accepted_amp_ssim": 0.8,
                "accepted_run_command": "python baseline.py",
                "accepted_candidate_kind": "baseline",
                "accepted_randomness_contract": {"requested_seed": 3},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    proposal = {
        "candidate_kind": "run_config",
        "base_ref": base_ref,
        "run_command": "python run.py --slow-timeout",
        "output_root": "outputs/v2/timeout-exc",
        "log_path": "state/v2/timeout-exc.log",
        "comparison_png_path": "outputs/v2/timeout-exc.png",
        "note": "timeout exception",
        "hypothesis": "slow run timed out via TimeoutExpired",
    }

    def fake_run(command, *, cwd, capture_output, text, check, timeout=None, env=None):
        assert command[:3] == ["bash", "-c", "python run.py --slow-timeout"]
        raise subprocess.TimeoutExpired(
            cmd=command,
            timeout=timeout or 1770,
            output=b"partial stdout\n",
            stderr=b"partial stderr\n",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assessment = run_scored_candidate(repo, session, proposal)

    assert assessment["decision"] == "TIMEOUT"
    log_text = (repo / proposal["log_path"]).read_text(encoding="utf-8")
    assert "partial stdout" in log_text
    assert "partial stderr" in log_text


def test_run_scored_candidate_nonzero_exit_is_classified_as_crash(tmp_path, monkeypatch):
    from scripts.studies.lines_256_session_controller import (
        initialize_session,
        run_scored_candidate,
    )

    repo = tmp_path / "repo"
    _init_repo(repo)
    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")
    base_ref = _git(repo, "rev-parse", "HEAD").stdout.strip()
    session.accepted_state_path.write_text(
        json.dumps(
            {
                "accepted_ref": base_ref,
                "accepted_amp_ssim": 0.8,
                "accepted_run_command": "python baseline.py",
                "accepted_candidate_kind": "baseline",
                "accepted_randomness_contract": {"requested_seed": 3},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    proposal = {
        "candidate_kind": "source",
        "base_ref": base_ref,
        "candidate_commit": base_ref,
        "run_command": "python run.py --boom",
        "output_root": "outputs/v2/crash",
        "log_path": "state/v2/crash.log",
        "comparison_png_path": "outputs/v2/crash.png",
        "note": "crash",
        "hypothesis": "crashy run",
    }

    def fake_run(command, *, cwd, capture_output, text, check, timeout=None, env=None):
        assert command[:3] == ["bash", "-c", "python run.py --boom"]
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assessment = run_scored_candidate(repo, session, proposal)

    assert assessment["decision"] == "CRASH"
    assert assessment["timed_out"] is False


def test_run_scored_command_uses_controller_runtime_python_when_path_is_poisoned(
    tmp_path, monkeypatch
):
    from scripts.studies.lines_256_session_controller import _run_scored_command

    repo_root = tmp_path / "session_repo"
    repo_root.mkdir()
    (repo_root / "scripts").mkdir()
    (repo_root / "ptycho_torch").mkdir()
    (repo_root / "ptycho_torch" / "__init__.py").write_text(
        "MARKER = 'session_repo'\n", encoding="utf-8"
    )
    (repo_root / "scripts" / "show_import_root.py").write_text(
        "import ptycho_torch\nprint(ptycho_torch.MARKER)\nprint(ptycho_torch.__file__)\n",
        encoding="utf-8",
    )

    foreign_root = tmp_path / "foreign_repo"
    foreign_root.mkdir()
    (foreign_root / "ptycho_torch").mkdir()
    (foreign_root / "ptycho_torch" / "__init__.py").write_text(
        "MARKER = 'foreign_repo'\n", encoding="utf-8"
    )

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text("#!/bin/sh\nexit 91\n", encoding="utf-8")
    fake_python.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("PYTHONPATH", str(foreign_root))

    result = _run_scored_command(
        repo_root=repo_root,
        command="python scripts/show_import_root.py",
        timeout_sec=5,
    )

    assert result["exit_code"] == 0
    assert "session_repo" in str(result["stdout_text"])
    assert str(repo_root / "ptycho_torch" / "__init__.py") in str(result["stdout_text"])
    assert "foreign_repo" not in str(result["stdout_text"])


def test_run_scored_candidate_persists_crash_excerpt(tmp_path, monkeypatch):
    from scripts.studies.lines_256_session_controller import (
        _candidate_assessment_path,
        _write_scored_artifacts,
        initialize_session,
        run_scored_candidate,
    )

    repo = tmp_path / "repo"
    _init_repo(repo)
    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")
    base_ref = _git(repo, "rev-parse", "HEAD").stdout.strip()
    session.accepted_state_path.write_text(
        json.dumps(
            {
                "accepted_ref": base_ref,
                "accepted_amp_ssim": 0.8,
                "accepted_run_command": "python baseline.py",
                "accepted_candidate_kind": "baseline",
                "accepted_randomness_contract": {"requested_seed": 3},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    proposal = {
        "candidate_kind": "run_config",
        "base_ref": base_ref,
        "run_command": "python run.py --boom",
        "output_root": "outputs/v2/crash-excerpt",
        "log_path": "state/v2/crash-excerpt.log",
        "comparison_png_path": "outputs/v2/crash-excerpt.png",
        "note": "crash excerpt",
        "hypothesis": "persist first crash line",
    }

    def fake_run(command, *, cwd, capture_output, text, check, timeout=None, env=None):
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="ModuleNotFoundError: No module named 'tensorflow'\nextra context\n",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assessment = run_scored_candidate(repo, session, proposal)

    assert assessment["decision"] == "CRASH"
    assert "ModuleNotFoundError" in assessment["crash_excerpt"]
    _write_scored_artifacts(session, assessment)
    payload = json.loads(_candidate_assessment_path(session).read_text(encoding="utf-8"))
    assert payload["crash_excerpt"] == assessment["crash_excerpt"]


def test_resume_session_uses_persisted_phase_and_iteration(tmp_path):
    from scripts.studies.lines_256_session_controller import initialize_session, resume_session

    repo = tmp_path / "repo"
    _init_repo(repo)
    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")
    payload = json.loads(session.session_json_path.read_text(encoding="utf-8"))
    payload["current_phase"] = "score_candidate"
    payload["iteration"] = 3
    session.session_json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    resumed = resume_session(session.session_root)

    assert resumed.current_phase == "score_candidate"
    assert resumed.iteration == 3


def test_initialize_session_persists_session_branch(tmp_path):
    from scripts.studies.lines_256_session_controller import initialize_session

    repo = tmp_path / "repo"
    _init_repo(repo)

    session = initialize_session(repo_root=repo, session_id="20260406T000000Z")

    payload = json.loads((session.session_root / "session.json").read_text(encoding="utf-8"))
    assert payload["session_branch"] == "lines256/session/20260406T000000Z"


def test_attach_or_reset_session_branch_checks_out_named_branch(tmp_path):
    from scripts.studies.lines_256_session_controller import (
        _attach_or_reset_session_branch,
        _git_current_branch,
        initialize_session,
    )

    repo = tmp_path / "repo"
    _init_repo(repo)
    base_ref = _git(repo, "rev-parse", "HEAD").stdout.strip()
    session = initialize_session(repo_root=repo, session_id="20260406T000000Z")

    _attach_or_reset_session_branch(repo, session.session_branch, base_ref)

    assert _git_current_branch(repo) == "lines256/session/20260406T000000Z"


def test_cleanup_source_candidate_restores_session_branch_to_accepted_ref(tmp_path):
    from scripts.studies.lines_256_session_controller import (
        _attach_or_reset_session_branch,
        _cleanup_source_candidate,
        _git_current_branch,
        _git_head,
        initialize_session,
    )

    repo = tmp_path / "repo"
    _init_repo(repo)
    base_ref = _git(repo, "rev-parse", "HEAD").stdout.strip()
    session = initialize_session(repo_root=repo, session_id="20260406T000000Z")
    (session.protected_local_paths_path).write_text("[]\n", encoding="utf-8")

    _attach_or_reset_session_branch(repo, session.session_branch, base_ref)
    candidate_file = repo / "candidate.txt"
    candidate_file.write_text("candidate\n", encoding="utf-8")
    _git(repo, "add", "candidate.txt")
    _git(repo, "commit", "-m", "candidate")

    candidate_paths_file = session.session_root / "iterations" / "000" / "candidate_paths.json"
    candidate_paths_file.parent.mkdir(parents=True, exist_ok=True)
    candidate_paths_file.write_text(json.dumps(["candidate.txt"]) + "\n", encoding="utf-8")

    _cleanup_source_candidate(
        repo_root=repo,
        session=session,
        proposal={
            "candidate_kind": "source",
            "base_ref": base_ref,
            "candidate_paths_file": str(candidate_paths_file.relative_to(repo)),
        },
    )

    assert _git_current_branch(repo) == session.session_branch
    assert _git_head(repo) == base_ref


def test_ensure_session_checkout_state_reattaches_detached_checkout_to_session_branch(tmp_path):
    from scripts.studies.lines_256_session_controller import (
        _attach_or_reset_session_branch,
        _ensure_session_checkout_state,
        _git_current_branch,
        initialize_session,
    )

    repo = tmp_path / "repo"
    _init_repo(repo)
    base_ref = _git(repo, "rev-parse", "HEAD").stdout.strip()
    session = initialize_session(repo_root=repo, session_id="20260406T000000Z")

    _attach_or_reset_session_branch(repo, session.session_branch, base_ref)
    _git(repo, "checkout", base_ref)

    _ensure_session_checkout_state(repo, session, expected_ref=base_ref)

    assert _git_current_branch(repo) == session.session_branch


def test_branch_reset_preserves_protected_local_paths(tmp_path):
    from scripts.studies.lines_256_session_controller import (
        _attach_or_reset_session_branch,
        _cleanup_source_candidate,
        initialize_session,
    )

    repo = tmp_path / "repo"
    base_ref = _init_repo(repo)
    protected_path = repo / "protected.txt"
    protected_path.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "protected.txt")
    _git(repo, "commit", "-m", "add protected file")
    base_ref = _git(repo, "rev-parse", "HEAD").stdout.strip()

    session = initialize_session(repo_root=repo, session_id="20260406T000000Z")
    _attach_or_reset_session_branch(repo, session.session_branch, base_ref)

    protected_path.write_text("user local change\n", encoding="utf-8")
    session.protected_local_paths_path.write_text(
        json.dumps(["protected.txt"]) + "\n",
        encoding="utf-8",
    )

    candidate_file = repo / "candidate.txt"
    candidate_file.write_text("candidate\n", encoding="utf-8")
    _git(repo, "add", "candidate.txt")
    _git(repo, "commit", "-m", "candidate")

    candidate_paths_file = session.session_root / "iterations" / "000" / "candidate_paths.json"
    candidate_paths_file.parent.mkdir(parents=True, exist_ok=True)
    candidate_paths_file.write_text(json.dumps(["candidate.txt"]) + "\n", encoding="utf-8")

    _cleanup_source_candidate(
        repo_root=repo,
        session=session,
        proposal={
            "candidate_kind": "source",
            "base_ref": base_ref,
            "candidate_paths_file": str(candidate_paths_file.relative_to(repo)),
        },
    )

    assert protected_path.read_text(encoding="utf-8") == "user local change\n"


def test_should_attempt_debug_only_on_first_crash():
    from scripts.studies.lines_256_session_controller import should_attempt_debug

    assert should_attempt_debug({"decision": "CRASH"}, debug_attempts=0) is True
    assert should_attempt_debug({"decision": "CRASH"}, debug_attempts=1) is False
    assert should_attempt_debug({"decision": "TIMEOUT"}, debug_attempts=0) is False


def _write_run_outputs(root: Path, amp_ssim: float) -> None:
    metrics_dir = root / "runs" / "pinn_hybrid_resnet"
    visuals_dir = root / "visuals"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "metrics.json").write_text(
        json.dumps({"amp_ssim": amp_ssim}, indent=2) + "\n",
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


def _write_minimal_lines256_docs(repo: Path) -> None:
    (repo / "docs/studies").mkdir(parents=True, exist_ok=True)
    (repo / "prompts/workflows/lines_256_arch_improvement").mkdir(parents=True, exist_ok=True)
    (repo / "docs/studies/lines_256_dataset.md").write_text("dataset\n", encoding="utf-8")
    (repo / "docs/studies/lines_256_arch_improvement_loop.md").write_text(
        "loop doc\n",
        encoding="utf-8",
    )
    (repo / "prompts/workflows/lines_256_arch_improvement/experiment_step.md").write_text(
        "compat prompt\n",
        encoding="utf-8",
    )
    (repo / "prompts/workflows/lines_256_arch_improvement/experiment_step_common.md").write_text(
        "common prompt\n",
        encoding="utf-8",
    )
    (repo / "prompts/workflows/lines_256_arch_improvement/experiment_step_exploit.md").write_text(
        "exploit prompt\n",
        encoding="utf-8",
    )
    (repo / "prompts/workflows/lines_256_arch_improvement/experiment_step_explore.md").write_text(
        "explore prompt\n",
        encoding="utf-8",
    )
    (repo / "prompts/workflows/lines_256_arch_improvement/debug_crash.md").write_text(
        "debug prompt\n",
        encoding="utf-8",
    )


def test_resume_proposal_running_without_metadata_yields_retryable_pending_state(
    tmp_path, monkeypatch
):
    from scripts.studies.lines_256_session_controller import initialize_session, main

    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_minimal_lines256_docs(repo)
    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")
    base_ref = _git(repo, "rev-parse", "HEAD").stdout.strip()
    session.accepted_state_path.write_text(
        json.dumps(
            {
                "accepted_ref": base_ref,
                "accepted_amp_ssim": 0.8,
                "accepted_run_command": "python baseline.py",
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
    (session.protected_local_paths_path).write_text("[]\n", encoding="utf-8")
    session.current_phase = "proposal_running"
    _write_json = lambda payload: session.session_json_path.write_text(  # noqa: E731
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_json(
        {
            "session_id": session.session_id,
            "current_phase": session.current_phase,
            "iteration": session.iteration,
            "debug_attempts": session.debug_attempts,
            "session_root": str(session.session_root.relative_to(repo)),
            "results_tsv_path": str(session.results_tsv_path.relative_to(repo)),
            "accepted_state_path": str(session.accepted_state_path.relative_to(repo)),
            "protected_local_paths_path": str(session.protected_local_paths_path.relative_to(repo)),
            "outputs_root": str(session.outputs_root.relative_to(repo)),
            "comparison_gallery_dir": str(session.comparison_gallery_dir.relative_to(repo)),
            "baseline_output_root": str(session.baseline_output_root.relative_to(repo)),
            "baseline_log_path": str(session.baseline_log_path.relative_to(repo)),
            "baseline_result_path": str(session.baseline_result_path.relative_to(repo)),
            "baseline_comparison_png": str(session.baseline_comparison_png.relative_to(repo)),
        }
    )

    def fake_run(command, **kwargs):
        if command[:3] == ["git", "status", "--short"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if len(command) >= 2 and command[1] == "exec":
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="ERROR: Selected model is at capacity\n",
            )
        raise AssertionError(command)

    monkeypatch.setattr(subprocess, "run", fake_run)

    exit_code = main(
        [
            "resume",
            "--session-root",
            str(session.session_root),
            "--max-iterations",
            "1",
            "--codex-cmd",
            "codex",
        ]
    )

    assert exit_code == 0
    resumed = json.loads(session.session_json_path.read_text(encoding="utf-8"))
    assert resumed["current_phase"] == "proposal_pending"
    assert resumed["iteration"] == 0
    proposal_result_path = session.session_root / "iterations" / "000" / "proposal_result.json"
    assert proposal_result_path.exists()
    payload = json.loads(proposal_result_path.read_text(encoding="utf-8"))
    assert payload["retryable"] is True
    assert payload["metadata_validated"] is False


def test_start_full_runs_controller_owned_smoke_before_scored_candidate(tmp_path, monkeypatch):
    from scripts.studies.lines_256_session_controller import main

    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_minimal_lines256_docs(repo)
    session_id = "20260325T000000Z"
    session_root = (
        repo / "state" / "lines_256_arch_improvement_v2" / "sessions" / session_id
    )
    outputs_root = (
        repo / "outputs" / "lines_256_arch_improvement_v2" / "sessions" / session_id
    )
    comparison_gallery = outputs_root / "comparison_pngs"
    baseline_output_root = outputs_root / "baseline"
    iter_root = session_root / "iterations" / "000"
    smoke_output_root = outputs_root / "candidates" / "20260325T001000Z_runconfig_smoke"
    scored_output_root = outputs_root / "candidates" / "20260325T001000Z_runconfig"
    smoke_log_path = iter_root / "20260325T001000Z_runconfig_smoke.log"
    candidate_log_path = iter_root / "20260325T001000Z_runconfig.log"
    comparison_png_path = (
        comparison_gallery / "20260325T001000Z_runconfig__compare_amp_phase_probe.png"
    )
    real_run = subprocess.run
    seen_commands = []

    def fake_run(command, **kwargs):
        cwd = kwargs.get("cwd")
        assert cwd == repo
        seen_commands.append(command)
        if command[:3] == ["git", "status", "--short"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return real_run(command, **kwargs)
        if command[:2] == ["python", "scripts/studies/run_lines_256_arch_experiment.py"]:
            _write_run_outputs(baseline_output_root, 0.81)
            return subprocess.CompletedProcess(command, 0, stdout="baseline ok\n", stderr="")
        if len(command) >= 2 and command[1] == "exec":
            base_ref = _git(repo, "rev-parse", "HEAD").stdout.strip()
            candidate_metadata = {
                "status": "READY",
                "candidate_kind": "run_config",
                "base_ref": base_ref,
                "smoke_command": "python smoke.py --fno-modes 24",
                "smoke_output_root": str(smoke_output_root.relative_to(repo)),
                "smoke_log_path": str(smoke_log_path.relative_to(repo)),
                "run_command": "python run.py --fno-modes 24",
                "output_root": str(scored_output_root.relative_to(repo)),
                "log_path": str(candidate_log_path.relative_to(repo)),
                "comparison_png_path": str(comparison_png_path.relative_to(repo)),
                "note": "Try a global 24-mode rerun.",
                "hypothesis": "Higher spectral capacity may improve amp_ssim.",
            }
            (iter_root / "candidate_metadata.json").parent.mkdir(parents=True, exist_ok=True)
            (iter_root / "candidate_metadata.json").write_text(
                json.dumps(candidate_metadata, indent=2) + "\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, stdout="READY\n", stderr="")
        if command[:3] == ["bash", "-c", "python smoke.py --fno-modes 24"]:
            _write_run_outputs(smoke_output_root, 0.5)
            return subprocess.CompletedProcess(command, 0, stdout="smoke ok\n", stderr="")
        if command[:3] == ["bash", "-c", "python run.py --fno-modes 24"]:
            _write_run_outputs(scored_output_root, 0.93)
            return subprocess.CompletedProcess(command, 0, stdout="scored ok\n", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(subprocess, "run", fake_run)

    exit_code = main(
        [
            "start",
            "--repo-root",
            str(repo),
            "--session-id",
            session_id,
            "--mode",
            "full",
            "--max-iterations",
            "1",
            "--codex-cmd",
            "codex",
        ]
    )

    assert exit_code == 0
    assert ["bash", "-c", "python smoke.py --fno-modes 24"] in seen_commands
    assert smoke_log_path.exists()
    accepted_state = json.loads((session_root / "accepted_state.json").read_text(encoding="utf-8"))
    assert accepted_state["accepted_candidate_kind"] == "run_config"
    assert accepted_state["accepted_amp_ssim"] == 0.93


def test_start_full_runs_baseline_then_one_run_config_keep(tmp_path, monkeypatch):
    from scripts.studies.lines_256_session_controller import main

    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "docs/studies").mkdir(parents=True, exist_ok=True)
    (repo / "prompts/workflows/lines_256_arch_improvement").mkdir(parents=True, exist_ok=True)
    (repo / "docs/studies/lines_256_dataset.md").write_text("dataset\n", encoding="utf-8")
    (repo / "docs/studies/lines_256_arch_improvement_loop.md").write_text(
        "loop doc\n",
        encoding="utf-8",
    )
    (repo / "prompts/workflows/lines_256_arch_improvement/experiment_step.md").write_text(
        "compat prompt\n",
        encoding="utf-8",
    )
    (repo / "prompts/workflows/lines_256_arch_improvement/experiment_step_common.md").write_text(
        "common prompt\n",
        encoding="utf-8",
    )
    (repo / "prompts/workflows/lines_256_arch_improvement/experiment_step_exploit.md").write_text(
        "exploit prompt\n",
        encoding="utf-8",
    )
    (repo / "prompts/workflows/lines_256_arch_improvement/experiment_step_explore.md").write_text(
        "explore prompt\n",
        encoding="utf-8",
    )
    (repo / "prompts/workflows/lines_256_arch_improvement/debug_crash.md").write_text(
        "debug prompt\n",
        encoding="utf-8",
    )
    session_id = "20260325T000000Z"
    session_root = (
        repo / "state" / "lines_256_arch_improvement_v2" / "sessions" / session_id
    )
    outputs_root = (
        repo / "outputs" / "lines_256_arch_improvement_v2" / "sessions" / session_id
    )
    comparison_gallery = outputs_root / "comparison_pngs"
    baseline_output_root = outputs_root / "baseline"
    iter_root = session_root / "iterations" / "000"
    smoke_output_root = outputs_root / "candidates" / "20260325T001000Z_runconfig_smoke"
    scored_output_root = outputs_root / "candidates" / "20260325T001000Z_runconfig"
    smoke_log_path = iter_root / "20260325T001000Z_runconfig_smoke.log"
    candidate_log_path = iter_root / "20260325T001000Z_runconfig.log"
    comparison_png_path = (
        comparison_gallery / "20260325T001000Z_runconfig__compare_amp_phase_probe.png"
    )
    real_run = subprocess.run

    def fake_run(command, **kwargs):
        cwd = kwargs.get("cwd")
        assert cwd == repo
        if command and command[0] == "git":
            return real_run(command, **kwargs)
        if command[:2] == ["python", "scripts/studies/run_lines_256_arch_experiment.py"]:
            _write_run_outputs(baseline_output_root, 0.81)
            return subprocess.CompletedProcess(command, 0, stdout="baseline ok\n", stderr="")
        if len(command) >= 2 and command[1] == "exec":
            assert "candidate_context.json" in kwargs["input"]
            _write_run_outputs(smoke_output_root, 0.5)
            smoke_log_path.parent.mkdir(parents=True, exist_ok=True)
            smoke_log_path.write_text("smoke ok\n", encoding="utf-8")
            base_ref = real_run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            candidate_metadata = {
                "status": "READY",
                "candidate_kind": "run_config",
                "base_ref": base_ref,
                "smoke_command": "python smoke.py --fno-modes 24",
                "smoke_output_root": str(smoke_output_root.relative_to(repo)),
                "smoke_log_path": str(smoke_log_path.relative_to(repo)),
                "run_command": "python run.py --fno-modes 24",
                "output_root": str(scored_output_root.relative_to(repo)),
                "log_path": str(candidate_log_path.relative_to(repo)),
                "comparison_png_path": str(comparison_png_path.relative_to(repo)),
                "note": "Try a global 24-mode rerun.",
                "hypothesis": "Higher spectral capacity may improve amp_ssim.",
            }
            (iter_root / "candidate_metadata.json").parent.mkdir(parents=True, exist_ok=True)
            (iter_root / "candidate_metadata.json").write_text(
                json.dumps(candidate_metadata, indent=2) + "\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, stdout="READY\n", stderr="")
        if command[:3] == ["bash", "-c", "python run.py --fno-modes 24"]:
            _write_run_outputs(scored_output_root, 0.93)
            return subprocess.CompletedProcess(command, 0, stdout="scored ok\n", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(subprocess, "run", fake_run)

    exit_code = main(
        [
            "start",
            "--repo-root",
            str(repo),
            "--session-id",
            session_id,
            "--mode",
            "full",
            "--max-iterations",
            "1",
            "--codex-cmd",
            "codex",
        ]
    )

    assert exit_code == 0
    accepted_state = json.loads((session_root / "accepted_state.json").read_text(encoding="utf-8"))
    assert accepted_state["accepted_candidate_kind"] == "run_config"
    assert accepted_state["accepted_amp_ssim"] == 0.93
    assert accepted_state["accepted_run_command"] == "python run.py --fno-modes 24"
    session_payload = json.loads((session_root / "session.json").read_text(encoding="utf-8"))
    assert session_payload["current_phase"] == "completed"
    assert session_payload["iteration"] == 1
    ledger_lines = (session_root / "results.tsv").read_text(encoding="utf-8").strip().splitlines()
    assert len(ledger_lines) == 3
    assert "\tbaseline\t0.81\t" in ledger_lines[1]
    assert "\tkeep\t0.93\t" in ledger_lines[2]


def test_resume_proposal_complete_scores_existing_candidate(tmp_path, monkeypatch):
    from scripts.studies.lines_256_session_controller import (
        _attach_or_reset_session_branch,
        initialize_session,
        main,
    )

    repo = tmp_path / "repo"
    _init_repo(repo)
    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")
    base_ref = _git(repo, "rev-parse", "HEAD").stdout.strip()
    session.accepted_state_path.write_text(
        json.dumps(
            {
                "accepted_ref": base_ref,
                "accepted_amp_ssim": 0.8,
                "accepted_run_command": "python baseline.py",
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
    (session.protected_local_paths_path).write_text("[]\n", encoding="utf-8")
    _attach_or_reset_session_branch(repo, session.session_branch, base_ref)
    session.current_phase = "proposal_complete"
    session.iteration = 0
    session.debug_attempts = 0
    session.session_json_path.write_text(
        json.dumps(
            {
                "session_id": session.session_id,
                "session_branch": session.session_branch,
                "current_phase": session.current_phase,
                "iteration": session.iteration,
                "debug_attempts": session.debug_attempts,
                "session_root": str(session.session_root.relative_to(repo)),
                "results_tsv_path": str(session.results_tsv_path.relative_to(repo)),
                "accepted_state_path": str(session.accepted_state_path.relative_to(repo)),
                "protected_local_paths_path": str(session.protected_local_paths_path.relative_to(repo)),
                "outputs_root": str(session.outputs_root.relative_to(repo)),
                "comparison_gallery_dir": str(session.comparison_gallery_dir.relative_to(repo)),
                "baseline_output_root": str(session.baseline_output_root.relative_to(repo)),
                "baseline_log_path": str(session.baseline_log_path.relative_to(repo)),
                "baseline_result_path": str(session.baseline_result_path.relative_to(repo)),
                "baseline_comparison_png": str(session.baseline_comparison_png.relative_to(repo)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    iter_root = session.session_root / "iterations" / "000"
    smoke_output_root = session.outputs_root / "candidates" / "20260325T001000Z_resume_smoke"
    scored_output_root = session.outputs_root / "candidates" / "20260325T001000Z_resume"
    smoke_log_path = iter_root / "20260325T001000Z_resume_smoke.log"
    candidate_log_path = iter_root / "20260325T001000Z_resume.log"
    comparison_png_path = (
        session.comparison_gallery_dir / "20260325T001000Z_resume__compare_amp_phase_probe.png"
    )
    _write_run_outputs(smoke_output_root, 0.5)
    smoke_log_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_log_path.write_text("smoke ok\n", encoding="utf-8")
    (iter_root / "candidate_metadata.json").write_text(
        json.dumps(
            {
                "status": "READY",
                "candidate_kind": "run_config",
                "base_ref": base_ref,
                "smoke_command": "python smoke.py --fno-modes 24",
                "smoke_output_root": str(smoke_output_root.relative_to(repo)),
                "smoke_log_path": str(smoke_log_path.relative_to(repo)),
                "run_command": "python run.py --fno-modes 24",
                "output_root": str(scored_output_root.relative_to(repo)),
                "log_path": str(candidate_log_path.relative_to(repo)),
                "comparison_png_path": str(comparison_png_path.relative_to(repo)),
                "note": "Resume-scored run_config candidate",
                "hypothesis": "Resume should score the prepared candidate.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    real_run = subprocess.run

    def fake_run(command, **kwargs):
        cwd = kwargs.get("cwd")
        assert cwd == repo
        if command and command[0] == "git":
            return real_run(command, **kwargs)
        if command[:3] == ["bash", "-c", "python run.py --fno-modes 24"]:
            _write_run_outputs(scored_output_root, 0.92)
            return subprocess.CompletedProcess(command, 0, stdout="resume scored\n", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(subprocess, "run", fake_run)

    exit_code = main(
        [
            "resume",
            "--session-root",
            str(session.session_root),
            "--max-iterations",
            "1",
        ]
    )

    assert exit_code == 0
    accepted_state = json.loads(session.accepted_state_path.read_text(encoding="utf-8"))
    assert accepted_state["accepted_amp_ssim"] == 0.92
    assert accepted_state["accepted_candidate_kind"] == "run_config"
    resumed = json.loads(session.session_json_path.read_text(encoding="utf-8"))
    assert resumed["current_phase"] == "completed"
    assert resumed["iteration"] == 1


def test_resume_scored_running_timeout_persists_terminal_artifacts(tmp_path, monkeypatch):
    from scripts.studies.lines_256_session_controller import (
        _attach_or_reset_session_branch,
        initialize_session,
        main,
    )

    repo = tmp_path / "repo"
    _init_repo(repo)
    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")
    base_ref = _git(repo, "rev-parse", "HEAD").stdout.strip()
    session.accepted_state_path.write_text(
        json.dumps(
            {
                "accepted_ref": base_ref,
                "accepted_amp_ssim": 0.8,
                "accepted_run_command": "python baseline.py",
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
    _attach_or_reset_session_branch(repo, session.session_branch, base_ref)
    session.current_phase = "scored_running"
    session.iteration = 0
    session.debug_attempts = 0
    session.session_json_path.write_text(
        json.dumps(
            {
                "session_id": session.session_id,
                "session_branch": session.session_branch,
                "current_phase": session.current_phase,
                "iteration": session.iteration,
                "debug_attempts": session.debug_attempts,
                "session_root": str(session.session_root.relative_to(repo)),
                "results_tsv_path": str(session.results_tsv_path.relative_to(repo)),
                "accepted_state_path": str(session.accepted_state_path.relative_to(repo)),
                "protected_local_paths_path": str(session.protected_local_paths_path.relative_to(repo)),
                "outputs_root": str(session.outputs_root.relative_to(repo)),
                "comparison_gallery_dir": str(session.comparison_gallery_dir.relative_to(repo)),
                "baseline_output_root": str(session.baseline_output_root.relative_to(repo)),
                "baseline_log_path": str(session.baseline_log_path.relative_to(repo)),
                "baseline_result_path": str(session.baseline_result_path.relative_to(repo)),
                "baseline_comparison_png": str(session.baseline_comparison_png.relative_to(repo)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (session.protected_local_paths_path).write_text("[]\n", encoding="utf-8")
    real_run = subprocess.run
    iter_root = session.session_root / "iterations" / "000"
    output_root = session.outputs_root / "candidates" / "20260325T001000Z_resume_timeout"
    candidate_log_path = iter_root / "20260325T001000Z_resume_timeout.log"
    comparison_png_path = (
        session.comparison_gallery_dir
        / "20260325T001000Z_resume_timeout__compare_amp_phase_probe.png"
    )
    iter_root.mkdir(parents=True, exist_ok=True)
    (iter_root / "candidate_metadata.json").write_text(
        json.dumps(
            {
                "status": "READY",
                "candidate_kind": "run_config",
                "base_ref": base_ref,
                "smoke_command": "python smoke.py --fno-width 64",
                "smoke_output_root": str((session.outputs_root / "smoke").relative_to(repo)),
                "smoke_log_path": str((iter_root / "smoke.log").relative_to(repo)),
                "run_command": "python run.py --fno-width 64",
                "output_root": str(output_root.relative_to(repo)),
                "log_path": str(candidate_log_path.relative_to(repo)),
                "comparison_png_path": str(comparison_png_path.relative_to(repo)),
                "note": "Resume-scored timeout candidate",
                "hypothesis": "Resume should persist timeout outcomes.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_run(command, **kwargs):
        if command and command[0] == "git":
            return real_run(command, **kwargs)
        if command[:3] == ["bash", "-c", "python run.py --fno-width 64"]:
            raise subprocess.TimeoutExpired(
                cmd=command,
                timeout=kwargs.get("timeout", 1770),
                output=b"timed stdout\n",
                stderr=b"timed stderr\n",
            )
        raise AssertionError(command)

    monkeypatch.setattr(subprocess, "run", fake_run)

    exit_code = main(
        [
            "resume",
            "--session-root",
            str(session.session_root),
            "--max-iterations",
            "1",
        ]
    )

    assert exit_code == 0
    run_result = json.loads(
        (iter_root / "candidate_run_result.json").read_text(encoding="utf-8")
    )
    assessment = json.loads(
        (iter_root / "candidate_assessment.json").read_text(encoding="utf-8")
    )
    assert run_result["timed_out"] is True
    assert assessment["decision"] == "TIMEOUT"
    assert "timed stdout" in candidate_log_path.read_text(encoding="utf-8")
    ledger_lines = session.results_tsv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(ledger_lines) == 2
    assert "\ttimeout\tna\t" in ledger_lines[1]


def test_resume_debug_complete_uses_existing_debug_assessment(tmp_path, monkeypatch):
    from scripts.studies.lines_256_session_controller import (
        _attach_or_reset_session_branch,
        initialize_session,
        main,
    )

    repo = tmp_path / "repo"
    _init_repo(repo)
    session = initialize_session(repo_root=repo, session_id="20260325T000000Z")
    base_ref = _git(repo, "rev-parse", "HEAD").stdout.strip()
    session.accepted_state_path.write_text(
        json.dumps(
            {
                "accepted_ref": base_ref,
                "accepted_amp_ssim": 0.8,
                "accepted_run_command": "python baseline.py",
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
    (repo / "README.md").write_text("debug candidate\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "debug candidate")
    debug_candidate_commit = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _attach_or_reset_session_branch(repo, session.session_branch, debug_candidate_commit)
    session.current_phase = "debug_complete"
    session.iteration = 0
    session.debug_attempts = 1
    session.session_json_path.write_text(
        json.dumps(
            {
                "session_id": session.session_id,
                "session_branch": session.session_branch,
                "current_phase": session.current_phase,
                "iteration": session.iteration,
                "debug_attempts": session.debug_attempts,
                "session_root": str(session.session_root.relative_to(repo)),
                "results_tsv_path": str(session.results_tsv_path.relative_to(repo)),
                "accepted_state_path": str(session.accepted_state_path.relative_to(repo)),
                "protected_local_paths_path": str(session.protected_local_paths_path.relative_to(repo)),
                "outputs_root": str(session.outputs_root.relative_to(repo)),
                "comparison_gallery_dir": str(session.comparison_gallery_dir.relative_to(repo)),
                "baseline_output_root": str(session.baseline_output_root.relative_to(repo)),
                "baseline_log_path": str(session.baseline_log_path.relative_to(repo)),
                "baseline_result_path": str(session.baseline_result_path.relative_to(repo)),
                "baseline_comparison_png": str(session.baseline_comparison_png.relative_to(repo)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (session.protected_local_paths_path).write_text("[]\n", encoding="utf-8")
    iter_root = session.session_root / "iterations" / "000"
    iter_root.mkdir(parents=True, exist_ok=True)
    (iter_root / "candidate_metadata.json").write_text(
        json.dumps(
            {
                "status": "READY",
                "candidate_kind": "run_config",
                "base_ref": base_ref,
                "smoke_command": "python smoke.py --regular",
                "smoke_output_root": "outputs/v2/regular-smoke",
                "smoke_log_path": "state/v2/regular-smoke.log",
                "run_command": "python regular.py",
                "output_root": "outputs/v2/regular",
                "log_path": "state/v2/regular.log",
                "comparison_png_path": "outputs/v2/regular.png",
                "note": "regular candidate",
                "hypothesis": "regular candidate should not rerun during debug_complete resume",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (iter_root / "debug_candidate_paths.json").write_text(
        json.dumps(["README.md"], indent=2) + "\n",
        encoding="utf-8",
    )
    (iter_root / "debug_candidate_metadata.json").write_text(
        json.dumps(
            {
                "status": "READY",
                "candidate_kind": "source",
                "base_ref": base_ref,
                "candidate_commit": debug_candidate_commit,
                "candidate_paths_file": str((iter_root / "debug_candidate_paths.json").relative_to(repo)),
                "smoke_command": "python smoke.py --debug",
                "smoke_output_root": "outputs/v2/debug-smoke",
                "smoke_log_path": "state/v2/debug-smoke.log",
                "run_command": "python debug.py",
                "output_root": "outputs/v2/debug",
                "log_path": "state/v2/debug.log",
                "comparison_png_path": "outputs/v2/debug.png",
                "note": "debug candidate",
                "hypothesis": "existing debug assessment should be applied directly",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (iter_root / "debug_candidate_run_result.json").write_text(
        json.dumps(
            {"launcher_status": "completed", "exit_code": 1, "timed_out": False},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (iter_root / "debug_candidate_assessment.json").write_text(
        json.dumps(
            {
                "decision": "CRASH",
                "comparison_png_path": "outputs/v2/debug.png",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    real_run = subprocess.run

    def fake_run(command, **kwargs):
        if command and command[0] == "git":
            return real_run(command, **kwargs)
        if command[:3] == ["bash", "-c", "python regular.py"]:
            raise AssertionError("regular candidate should not rerun during debug_complete resume")
        if command[:3] == ["bash", "-c", "python debug.py"]:
            raise AssertionError("debug candidate should not rerun when assessment already exists")
        raise AssertionError(command)

    monkeypatch.setattr(subprocess, "run", fake_run)

    exit_code = main(
        [
            "resume",
            "--session-root",
            str(session.session_root),
            "--max-iterations",
            "1",
        ]
    )

    assert exit_code == 0
    resumed = json.loads(session.session_json_path.read_text(encoding="utf-8"))
    assert resumed["current_phase"] == "completed"
    assert resumed["iteration"] == 0
    assert resumed["debug_attempts"] == 0
    ledger_lines = session.results_tsv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(ledger_lines) == 2
    assert "\tcrash\tna\t" in ledger_lines[1]


def test_start_full_can_bootstrap_existing_accepted_state(tmp_path, monkeypatch):
    from scripts.studies.lines_256_session_controller import main

    repo = tmp_path / "repo"
    _init_repo(repo)
    bootstrap_ref = _git(repo, "rev-parse", "HEAD").stdout.strip()
    real_run = subprocess.run
    bootstrap_state = tmp_path / "legacy_accepted_state.json"
    bootstrap_state.write_text(
        json.dumps(
            {
                "accepted_ref": bootstrap_ref,
                "accepted_amp_ssim": 0.91,
                "accepted_run_command": "python run.py --fno-modes 24",
                "accepted_candidate_kind": "run_config",
                "accepted_randomness_contract": {
                    "requested_seed": 3,
                    "effective_subsample_seed": 3,
                    "effective_lightning_seed": 3,
                },
                "accepted_output_root": "outputs/legacy/candidate",
                "accepted_comparison_png": "outputs/legacy/candidate.png",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_run(command, **kwargs):
        if command and command[0] == "git":
            return real_run(command, **kwargs)
        raise AssertionError(command)

    monkeypatch.setattr(subprocess, "run", fake_run)

    session_id = "20260325T000000Z"
    exit_code = main(
        [
            "start",
            "--repo-root",
            str(repo),
            "--session-id",
            session_id,
            "--mode",
            "full",
            "--max-iterations",
            "0",
            "--bootstrap-accepted-state",
            str(bootstrap_state),
        ]
    )

    assert exit_code == 0
    accepted_state_path = (
        repo
        / "state"
        / "lines_256_arch_improvement_v2"
        / "sessions"
        / session_id
        / "accepted_state.json"
    )
    accepted_state = json.loads(accepted_state_path.read_text(encoding="utf-8"))
    assert accepted_state["accepted_ref"] == bootstrap_ref
    assert accepted_state["accepted_amp_ssim"] == 0.91
    assert accepted_state["session_id"] == session_id
