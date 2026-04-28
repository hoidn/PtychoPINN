import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "tests/fixtures/neurips_steered_backlog"


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "repo"
    shutil.copytree(FIXTURE_ROOT, workspace)
    return workspace


def _run_script(workspace: Path, relpath: str, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / relpath), *args],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=check,
    )


def test_build_manifest_preserves_check_commands_and_optional_fields(tmp_path):
    workspace = _workspace(tmp_path)
    output_path = workspace / "state/manifest.json"

    _run_script(
        workspace,
        "workflows/library/scripts/build_neurips_backlog_manifest.py",
        "--backlog-root",
        "docs/backlog/active",
        "--output",
        str(output_path),
    )

    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    assert manifest["active_count"] == 2
    ready = manifest["items"][0]
    blocked = manifest["items"][1]

    assert ready["item_id"] == "2026-04-22-ready-item"
    assert ready["check_commands"] == ['python -c "print(\'ready-check\')"']
    assert ready["prerequisites"] == ["phase-0-evidence-inventory"]
    assert blocked["prerequisites"] == ["phase-99-not-done"]


def test_build_manifest_rejects_malformed_frontmatter(tmp_path):
    workspace = _workspace(tmp_path)
    bad_item = workspace / "docs/backlog/active/2026-04-22-bad-item.md"
    bad_item.write_text("# no frontmatter\n", encoding="utf-8")

    result = _run_script(
        workspace,
        "workflows/library/scripts/build_neurips_backlog_manifest.py",
        "--backlog-root",
        "docs/backlog/active",
        "--output",
        str(workspace / "state/manifest.json"),
        check=False,
    )

    assert result.returncode != 0
    assert "frontmatter" in result.stderr.lower()


def test_materialize_selected_item_inputs_writes_check_command_artifacts(tmp_path):
    workspace = _workspace(tmp_path)
    manifest_path = workspace / "state/manifest.json"
    _run_script(
        workspace,
        "workflows/library/scripts/build_neurips_backlog_manifest.py",
        "--backlog-root",
        "docs/backlog/active",
        "--output",
        str(manifest_path),
    )

    selector_root = workspace / "state/selector"
    selector_root.mkdir(parents=True, exist_ok=True)
    selection_path = selector_root / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "selection_status": "SELECTED",
                "selected_item_id": "2026-04-22-ready-item",
                "selected_item_path": "docs/backlog/active/2026-04-22-ready-item.md",
                "selection_rationale": "ready",
                "roadmap_sync_hint": "NO_CHANGE",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = workspace / "state/materialized.json"

    _run_script(
        workspace,
        "workflows/library/scripts/materialize_neurips_selected_item_inputs.py",
        "--selection-path",
        str(selection_path),
        "--manifest-path",
        str(manifest_path),
        "--state-root",
        "state/NEURIPS-HYBRID-RESNET-2026/backlog_drain",
        "--output",
        str(output_path),
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    check_commands_path = workspace / payload["check_commands_path"]
    context_path = workspace / payload["selected_item_context_path"]

    assert payload["selection_mode"] == "ACTIVE_SELECTION"
    assert check_commands_path.is_file()
    assert json.loads(check_commands_path.read_text(encoding="utf-8")) == ['python -c "print(\'ready-check\')"']
    assert context_path.is_file()
    context_text = context_path.read_text(encoding="utf-8")
    assert "Required Check Commands" in context_text
    assert "authoritative_item_path: `docs/backlog/in_progress/2026-04-22-ready-item.md`" in context_text
    assert "selection_source_path: `docs/backlog/active/2026-04-22-ready-item.md`" in context_text
    assert "selected_item_path:" not in context_text


def test_move_item_allows_legal_transitions_and_rejects_illegal_ones(tmp_path):
    workspace = _workspace(tmp_path)
    active_path = "docs/backlog/active/2026-04-22-ready-item.md"

    result = _run_script(
        workspace,
        "workflows/library/scripts/move_neurips_backlog_item.py",
        "--item-path",
        active_path,
        "--dest-state",
        "in_progress",
    )
    assert result.stdout.strip() == "docs/backlog/in_progress/2026-04-22-ready-item.md"

    illegal = _run_script(
        workspace,
        "workflows/library/scripts/move_neurips_backlog_item.py",
        "--item-path",
        "docs/backlog/in_progress/2026-04-22-ready-item.md",
        "--dest-state",
        "in_progress",
        check=False,
    )
    assert illegal.returncode != 0
    assert "illegal queue transition" in illegal.stderr.lower()

    done = _run_script(
        workspace,
        "workflows/library/scripts/move_neurips_backlog_item.py",
        "--item-path",
        "docs/backlog/in_progress/2026-04-22-ready-item.md",
        "--dest-state",
        "done",
    )
    assert done.stdout.strip() == "docs/backlog/done/2026-04-22-ready-item.md"


def test_reconcile_selected_item_recovers_active_path_drift_and_rewrites_plan_path(tmp_path):
    workspace = _workspace(tmp_path)
    output_path = workspace / "state/reconciled_item_path.txt"
    plan_path = "docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-22-ready-item/execution_plan.md"

    result = _run_script(
        workspace,
        "workflows/library/scripts/reconcile_neurips_selected_item.py",
        "--active-path",
        "docs/backlog/active/2026-04-22-ready-item.md",
        "--in-progress-path",
        "docs/backlog/in_progress/2026-04-22-ready-item.md",
        "--plan-path",
        plan_path,
        "--output-path",
        str(output_path),
    )

    assert result.stdout.strip() == "docs/backlog/in_progress/2026-04-22-ready-item.md"
    assert not (workspace / "docs/backlog/active/2026-04-22-ready-item.md").exists()
    item_path = workspace / "docs/backlog/in_progress/2026-04-22-ready-item.md"
    assert item_path.is_file()
    assert output_path.read_text(encoding="utf-8").strip() == item_path.relative_to(workspace).as_posix()
    assert f"plan_path: {plan_path}" in item_path.read_text(encoding="utf-8")


def test_run_state_persists_blockers_and_current_roadmap(tmp_path):
    workspace = _workspace(tmp_path)
    state_path = "state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/run_state.json"

    _run_script(
        workspace,
        "workflows/library/scripts/update_neurips_backlog_run_state.py",
        "--state-path",
        state_path,
        "init",
        "--run-id",
        "run-123",
        "--roadmap-path",
        "docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md",
    )
    _run_script(
        workspace,
        "workflows/library/scripts/update_neurips_backlog_run_state.py",
        "--state-path",
        state_path,
        "select",
        "--run-id",
        "run-123",
        "--item-id",
        "2026-04-22-ready-item",
        "--item-path",
        "docs/backlog/in_progress/2026-04-22-ready-item.md",
        "--roadmap-path",
        "docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md",
    )
    _run_script(
        workspace,
        "workflows/library/scripts/update_neurips_backlog_run_state.py",
        "--state-path",
        state_path,
        "block",
        "--item-id",
        "2026-04-22-ready-item",
        "--item-path",
        "docs/backlog/in_progress/2026-04-22-ready-item.md",
        "--stage",
        "plan_phase",
        "--reason",
        "plan not approved",
        "--plan-path",
        "docs/plans/fresh-plan.md",
        "--roadmap-path",
        "docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md",
    )

    state = json.loads((workspace / state_path).read_text(encoding="utf-8"))
    assert state["run_id"] == "run-123"
    assert state["current_roadmap_path"] == "docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md"
    assert state["current_item"] is None
    assert state["blocked_items"]["2026-04-22-ready-item"]["stage"] == "plan_phase"


def test_run_neurips_backlog_checks_reports_failures_without_process_failure(tmp_path):
    workspace = _workspace(tmp_path)
    checks_path = workspace / "state/checks.json"
    checks_path.parent.mkdir(parents=True, exist_ok=True)
    checks_path.write_text(
        json.dumps(
            [
                'python -c "print(\'ok\')"',
                'python -c "import sys; sys.exit(3)"',
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = workspace / "artifacts/checks/report.json"

    result = _run_script(
        workspace,
        "workflows/library/scripts/run_neurips_backlog_checks.py",
        "--checks-path",
        str(checks_path),
        "--report-path",
        str(report_path),
    )
    assert result.returncode == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"
    assert report["failed_count"] == 1
    assert report["results"][1]["exit_code"] == 3
