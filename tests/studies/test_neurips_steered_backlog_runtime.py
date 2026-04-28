import json
import shutil
from dataclasses import is_dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from orchestrator.loader import WorkflowLoader
from orchestrator.providers.executor import ProviderExecutor
from orchestrator.state import StateManager
from orchestrator.workflow.executor import WorkflowExecutor
from orchestrator.workflow.loaded_bundle import workflow_context, workflow_input_contracts
from orchestrator.workflow.signatures import bind_workflow_inputs


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = ROOT / "tests/fixtures/neurips_steered_backlog"


def _thaw(value):
    if isinstance(value, dict):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, list):
        return [_thaw(item) for item in value]
    if hasattr(value, "items"):
        return {str(key): _thaw(item) for key, item in value.items()}
    if is_dataclass(value):
        return {str(key): _thaw(item) for key, item in vars(value).items()}
    return value


def _bundle_context_dict(bundle) -> dict:
    return _thaw(workflow_context(bundle))


def _copy_repo_file_to_workspace(workspace: Path, repo_relpath: str) -> None:
    src = ROOT / repo_relpath
    dest = workspace / repo_relpath
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def _copy_runtime_files(workspace: Path) -> Path:
    shutil.copytree(FIXTURE_ROOT, workspace, dirs_exist_ok=True)
    files = [
        "workflows/examples/neurips_steered_backlog_drain.yaml",
        "workflows/library/neurips_backlog_selector.yaml",
        "workflows/library/neurips_backlog_gap_drafter.yaml",
        "workflows/library/neurips_backlog_roadmap_sync_phase.yaml",
        "workflows/library/neurips_backlog_seeded_plan_phase.yaml",
        "workflows/library/neurips_backlog_implementation_phase.yaml",
        "workflows/library/neurips_selected_backlog_item.yaml",
        "workflows/library/scripts/build_neurips_backlog_manifest.py",
        "workflows/library/scripts/materialize_neurips_selected_item_inputs.py",
        "workflows/library/scripts/move_neurips_backlog_item.py",
        "workflows/library/scripts/reconcile_neurips_selected_item.py",
        "workflows/library/scripts/reconcile_neurips_backlog_roadmap_gate.py",
        "workflows/library/scripts/validate_neurips_backlog_gap_draft.py",
        "workflows/library/scripts/run_neurips_backlog_checks.py",
        "workflows/library/scripts/update_neurips_backlog_run_state.py",
    ]
    files.extend(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "workflows/library/prompts").rglob("*.md")
    )
    for relpath in sorted(set(files)):
        _copy_repo_file_to_workspace(workspace, relpath)
    return workspace / "workflows/examples/neurips_steered_backlog_drain.yaml"


def _target_from_pointer(workspace: Path, pointer_relpath: str) -> Path:
    target_relpath = (workspace / pointer_relpath).read_text(encoding="utf-8").strip()
    target = workspace / target_relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _next_missing_selector_dir(workspace: Path) -> Path:
    for selector_dir in sorted(workspace.glob("state/**/selector")):
        if not (selector_dir / "selection.json").exists():
            return selector_dir
    raise AssertionError("No pending selector directory found")


def _write_first_selector(workspace: Path) -> None:
    selector_dir = _next_missing_selector_dir(workspace)
    (selector_dir / "selection.json").write_text(
        json.dumps(
            {
                "selection_status": "SELECTED",
                "selected_item_id": "2026-04-22-ready-item",
                "selected_item_path": "docs/backlog/active/2026-04-22-ready-item.md",
                "selection_rationale": "The ready item already satisfies the prerequisite and directly advances the current objective.",
                "roadmap_sync_hint": "NO_CHANGE",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_second_selector_blocked(workspace: Path) -> None:
    selector_dir = _next_missing_selector_dir(workspace)
    (selector_dir / "selection.json").write_text(
        json.dumps(
            {
                "selection_status": "BLOCKED",
                "selection_rationale": "The remaining active item is blocked by an unmet prerequisite.",
                "blocking_reasons": ["phase-99-not-done is not completed"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_roadmap_sync_outputs(workspace: Path) -> None:
    pointer_candidates = sorted(workspace.glob("state/**/roadmap-sync/roadmap_sync_report_path.txt"))
    for pointer in pointer_candidates:
        if (pointer.parent / "roadmap_sync_status.txt").exists():
            continue
        target = _target_from_pointer(workspace, pointer.relative_to(workspace).as_posix())
        target.write_text("NO_CHANGE\n", encoding="utf-8")
        (pointer.parent / "roadmap_sync_status.txt").write_text("NO_CHANGE\n", encoding="utf-8")
        return
    raise AssertionError("No pending roadmap-sync report pointer found")


def _write_plan_draft(workspace: Path) -> None:
    pointer_candidates = sorted(workspace.glob("state/**/plan-phase/plan_path.txt"))
    for pointer in pointer_candidates:
        if (pointer.parent / "final_plan_path.txt").exists():
            continue
        target = _target_from_pointer(workspace, pointer.relative_to(workspace).as_posix())
        target.write_text(
            "\n".join(
                [
                    "# Fresh Ready Item Plan",
                    "",
                    "## Scope",
                    "- Execute only the selected ready backlog item.",
                    "",
                    "## Constraints",
                    "- Preserve roadmap order and steering intent.",
                    "",
                    "## Verification",
                    "- `python -c \"print('ready-check')\"`",
                    "",
                    "## Tasks",
                    "- Update the ready item path and produce an execution report.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return
    raise AssertionError("No pending plan draft pointer found")


def _write_plan_review(workspace: Path) -> None:
    pointer_candidates = sorted(workspace.glob("state/**/plan-phase/plan_review_report_path.txt"))
    for pointer in pointer_candidates:
        if (pointer.parent / "plan_review_decision.txt").exists():
            continue
        target = _target_from_pointer(workspace, pointer.relative_to(workspace).as_posix())
        target.write_text(
            json.dumps(
                {
                    "decision": "APPROVE",
                    "summary": "Ready item plan is self-contained and preserves the backlog verification contract.",
                    "unresolved_high_count": 0,
                    "unresolved_medium_count": 0,
                    "findings": [],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        phase_root = pointer.parent
        (phase_root / "plan_review_decision.txt").write_text("APPROVE\n", encoding="utf-8")
        (phase_root / "unresolved_high_count.txt").write_text("0\n", encoding="utf-8")
        (phase_root / "unresolved_medium_count.txt").write_text("0\n", encoding="utf-8")
        return
    raise AssertionError("No pending plan review pointer found")


def _restore_ready_item_to_active(workspace: Path) -> None:
    in_progress = workspace / "docs/backlog/in_progress/2026-04-22-ready-item.md"
    active = workspace / "docs/backlog/active/2026-04-22-ready-item.md"
    if in_progress.exists() and not active.exists():
        active.parent.mkdir(parents=True, exist_ok=True)
        in_progress.rename(active)


def _write_plan_review_and_restore_active_item(workspace: Path) -> None:
    _write_plan_review(workspace)
    _restore_ready_item_to_active(workspace)


def _write_execution_report(workspace: Path) -> None:
    state_roots = sorted(workspace.glob("state/**/implementation-phase"))
    for state_root in state_roots:
        bundle_path = state_root / "implementation_state.json"
        if bundle_path.exists():
            continue
        target = _target_from_pointer(
            workspace, (state_root / "execution_report_target_path.txt").relative_to(workspace).as_posix()
        )
        target.write_text(
            "\n".join(
                [
                    "# Execution Report",
                    "",
                    "## Completed In This Pass",
                    "- Implemented the ready backlog item fixture.",
                    "",
                    "## Completed Plan Tasks",
                    "- Produced a fresh implementation artifact for the selected item.",
                    "",
                    "## Remaining Required Plan Tasks",
                    "- None.",
                    "",
                    "## Verification",
                    "- ready-check command is expected to pass.",
                    "",
                    "## Residual Risks",
                    "- Fixture smoke only.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        bundle_path.write_text(
            json.dumps(
                {
                    "implementation_state": "COMPLETED",
                    "execution_report_path": target.relative_to(workspace).as_posix(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return
    raise AssertionError("No pending execution report pointer found")


def _write_running_execution_state(workspace: Path) -> None:
    state_roots = sorted(workspace.glob("state/**/implementation-phase"))
    for state_root in state_roots:
        bundle_path = state_root / "implementation_state.json"
        if bundle_path.exists():
            continue
        progress_report = (
            workspace
            / "artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-22-ready-item/progress_report.md"
        )
        progress_report.parent.mkdir(parents=True, exist_ok=True)
        progress_report.write_text(
            "\n".join(
                [
                    "# Progress Report",
                    "",
                    "## Active Work",
                    "- Long-running readiness or training task is still in progress.",
                    "",
                    "## Next Resume Condition",
                    "- Resume when the external execution finishes and the final report can be written.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        bundle_path.write_text(
            json.dumps(
                {
                    "implementation_state": "RUNNING",
                    "progress_report_path": progress_report.relative_to(workspace).as_posix(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return
    raise AssertionError("No pending implementation-phase state root found for RUNNING bundle")


def _write_blocked_execution_state(workspace: Path) -> None:
    state_roots = sorted(workspace.glob("state/**/implementation-phase"))
    for state_root in state_roots:
        bundle_path = state_root / "implementation_state.json"
        if bundle_path.exists():
            continue
        progress_report = (
            workspace
            / "artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-22-ready-item/blocked_progress_report.md"
        )
        progress_report.parent.mkdir(parents=True, exist_ok=True)
        progress_report.write_text(
            "\n".join(
                [
                    "# Blocked Progress Report",
                    "",
                    "## Blocker",
                    "- Required upstream artifact is unavailable.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        bundle_path.write_text(
            json.dumps(
                {
                    "implementation_state": "BLOCKED",
                    "progress_report_path": progress_report.relative_to(workspace).as_posix(),
                    "block_reason": "Required upstream artifact is unavailable.",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return
    raise AssertionError("No pending implementation-phase state root found for BLOCKED bundle")


def _replace_active_backlog_with_future_phase_gap_fixture(workspace: Path) -> None:
    active_root = workspace / "docs/backlog/active"
    for item in active_root.glob("*.md"):
        item.unlink()
    (active_root / "2026-04-27-cdi-item.md").write_text(
        "\n".join(
            [
                "---",
                "priority: 5",
                "plan_path: docs/plans/legacy-ready-plan.md",
                "check_commands:",
                "  - python -c \"print('cdi-check')\"",
                "related_roadmap_phases:",
                "  - phase-3-cdi-anchor-regeneration",
                "---",
                "",
                "# Backlog Item: CDI Item",
                "",
                "## Objective",
                "- Represent future CDI work.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (active_root / "2026-03-13-support-item.md").write_text(
        "\n".join(
            [
                "---",
                "priority: 10",
                "plan_path: docs/plans/legacy-ready-plan.md",
                "check_commands:",
                "  - python -c \"print('support-check')\"",
                "---",
                "",
                "# Backlog Item: Support Item",
                "",
                "## Objective",
                "- Represent support work without a roadmap phase.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_gap_draft(workspace: Path) -> None:
    state_roots = sorted(workspace.glob("state/**/gap-drafter"))
    for state_root in state_roots:
        bundle_path = state_root / "draft-bundle.json"
        if bundle_path.exists():
            continue
        plan_path = (
            workspace
            / "docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog-gaps/2026-04-28-phase2-gap.md"
        )
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(
            "\n".join(
                [
                    "# Phase 2 Gap Plan",
                    "",
                    "## Objective",
                    "- Close the missing Phase 2 full-training evidence gate.",
                    "",
                    "## Verification",
                    "- `python -c \"print('phase2-gap-check')\"`",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        item_path = workspace / "docs/backlog/active/2026-04-28-phase2-gap.md"
        item_path.write_text(
            "\n".join(
                [
                    "---",
                    "priority: 1",
                    "plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog-gaps/2026-04-28-phase2-gap.md",
                    "check_commands:",
                    "  - python -c \"print('phase2-gap-check')\"",
                    "related_roadmap_phases:",
                    "  - phase-2-pdebench-full-training-evidence",
                    "---",
                    "",
                    "# Backlog Item: Phase 2 Gap",
                    "",
                    "## Objective",
                    "- Close the missing Phase 2 full-training evidence gate.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        bundle_path.write_text(
            json.dumps(
                {
                    "draft_status": "DRAFTED",
                    "backlog_item_path": item_path.relative_to(workspace).as_posix(),
                    "seed_plan_path": plan_path.relative_to(workspace).as_posix(),
                    "summary": "Drafted missing Phase 2 backlog work.",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return
    raise AssertionError("No pending gap-drafter state root found")


def _write_selector_blocked_after_gap(workspace: Path) -> None:
    manifests = sorted(workspace.glob("state/**/eligible_manifest.json"))
    assert manifests, "No eligible manifest produced after gap draft"
    latest = manifests[-1]
    manifest = json.loads(latest.read_text(encoding="utf-8"))
    assert any(item["item_id"] == "2026-04-28-phase2-gap" for item in manifest["items"])
    selector_dirs = sorted(path for path in workspace.glob("state/**/selector") if not (path / "selection.json").exists())
    assert selector_dirs, "No pending selector directory found"
    selector_dir = selector_dirs[-1]
    (selector_dir / "selection.json").write_text(
        json.dumps(
            {
                "selection_status": "BLOCKED",
                "selection_rationale": "Runtime smoke stops after proving the drafted Phase 2 item became selectable.",
                "blocking_reasons": ["smoke stops before executing drafted item"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_implementation_review(workspace: Path) -> None:
    pointer_candidates = sorted(workspace.glob("state/**/implementation-phase/implementation_review_report_path.txt"))
    for pointer in pointer_candidates:
        if (pointer.parent / "implementation_review_decision.txt").exists():
            continue
        state_path = pointer.parent / "implementation_state.txt"
        if not state_path.exists() or state_path.read_text(encoding="utf-8").strip() != "COMPLETED":
            continue
        target = _target_from_pointer(workspace, pointer.relative_to(workspace).as_posix())
        target.write_text(
            "\n".join(
                [
                    "# Implementation Review",
                    "",
                    "No blocking findings.",
                    "",
                    "## Follow-Up Work",
                    "- None.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (pointer.parent / "implementation_review_decision.txt").write_text("APPROVE\n", encoding="utf-8")
        return
    raise AssertionError("No pending implementation review pointer found")


def _write_implementation_review_and_restore_active_item(workspace: Path) -> None:
    _write_implementation_review(workspace)
    _restore_ready_item_to_active(workspace)


def _run_with_mocked_providers(workspace: Path, workflow_path: Path) -> dict:
    loader = WorkflowLoader(workspace)
    workflow = loader.load(workflow_path)
    workflow_relpath = workflow_path.relative_to(workspace).as_posix()
    bound_inputs = bind_workflow_inputs(workflow_input_contracts(workflow), {}, workspace)
    state_manager = StateManager(workspace=workspace, run_id="test-run")
    state_manager.initialize(workflow_relpath, _bundle_context_dict(workflow), bound_inputs=bound_inputs)
    executor = WorkflowExecutor(workflow, workspace, state_manager)

    provider_sequence = [
        ("SelectNextItem", _write_first_selector),
        ("ReviewOrUpdateRoadmap", _write_roadmap_sync_outputs),
        ("DraftPlan", _write_plan_draft),
        ("ReviewPlanTracked", _write_plan_review),
        ("ExecuteImplementation", _write_execution_report),
        ("ReviewImplementation", _write_implementation_review),
    ]
    call_index = {"value": 0}

    def _prepare_invocation(_self, *args, **kwargs):
        return SimpleNamespace(input_mode="stdin", prompt=kwargs.get("prompt_content", "")), None

    def _execute(_self, _invocation, **kwargs):
        expected_step, writer = provider_sequence[call_index["value"]]
        actual_step = kwargs.get("step_name")
        if actual_step is not None:
            assert actual_step == expected_step
        call_index["value"] += 1
        writer(workspace)
        return SimpleNamespace(
            exit_code=0,
            stdout=b"ok",
            stderr=b"",
            duration_ms=1,
            error=None,
            missing_placeholders=None,
            invalid_prompt_placeholder=False,
            raw_stdout=None,
            normalized_stdout=None,
            provider_session=None,
        )

    with patch.object(ProviderExecutor, "prepare_invocation", _prepare_invocation), patch.object(
        ProviderExecutor, "execute", _execute
    ):
        state = executor.execute()
    state["__provider_calls"] = call_index["value"]
    return state


def test_neurips_steered_backlog_runtime_smoke(tmp_path):
    workspace = tmp_path / "workspace"
    workflow_path = _copy_runtime_files(workspace)

    state = _run_with_mocked_providers(workspace, workflow_path)

    assert state["__provider_calls"] == 6
    summary = json.loads(
        (workspace / "artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog-drain-summary.json").read_text(
            encoding="utf-8"
        )
    )
    run_state = json.loads(
        (workspace / "state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/run_state.json").read_text(
            encoding="utf-8"
        )
    )

    assert summary["drain_status"] == "BLOCKED"
    assert summary["completed_item_count"] == 1
    assert run_state["completed_items"] == ["2026-04-22-ready-item"]
    assert run_state["current_roadmap_path"] == "docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md"
    assert (workspace / "docs/backlog/done/2026-04-22-ready-item.md").is_file()
    assert (workspace / "docs/backlog/active/2026-04-22-blocked-item.md").is_file()

    done_text = (workspace / "docs/backlog/done/2026-04-22-ready-item.md").read_text(encoding="utf-8")
    assert "plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-22-ready-item/execution_plan.md" in done_text

    checks_report = json.loads(
        (
            workspace
            / "artifacts/checks/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-22-ready-item-checks.json"
        ).read_text(encoding="utf-8")
    )
    assert checks_report["status"] == "PASS"
    assert checks_report["failed_count"] == 0


def test_neurips_steered_backlog_runtime_recovers_queue_path_drift_after_plan_review(tmp_path):
    workspace = tmp_path / "workspace"
    workflow_path = _copy_runtime_files(workspace)

    provider_sequence = [
        ("SelectNextItem", _write_first_selector),
        ("ReviewOrUpdateRoadmap", _write_roadmap_sync_outputs),
        ("DraftPlan", _write_plan_draft),
        ("ReviewPlanTracked", _write_plan_review_and_restore_active_item),
        ("ExecuteImplementation", _write_execution_report),
        ("ReviewImplementation", _write_implementation_review_and_restore_active_item),
    ]
    call_index = {"value": 0}

    def _prepare_invocation(_self, *args, **kwargs):
        return SimpleNamespace(input_mode="stdin", prompt=kwargs.get("prompt_content", "")), None

    def _execute(_self, _invocation, **kwargs):
        expected_step, writer = provider_sequence[call_index["value"]]
        actual_step = kwargs.get("step_name")
        if actual_step is not None:
            assert actual_step == expected_step
        call_index["value"] += 1
        writer(workspace)
        return SimpleNamespace(
            exit_code=0,
            stdout=b"ok",
            stderr=b"",
            duration_ms=1,
            error=None,
            missing_placeholders=None,
            invalid_prompt_placeholder=False,
            raw_stdout=None,
            normalized_stdout=None,
            provider_session=None,
        )

    loader = WorkflowLoader(workspace)
    workflow = loader.load(workflow_path)
    workflow_relpath = workflow_path.relative_to(workspace).as_posix()
    bound_inputs = bind_workflow_inputs(workflow_input_contracts(workflow), {}, workspace)
    state_manager = StateManager(workspace=workspace, run_id="test-run")
    state_manager.initialize(workflow_relpath, _bundle_context_dict(workflow), bound_inputs=bound_inputs)
    executor = WorkflowExecutor(workflow, workspace, state_manager)

    with patch.object(ProviderExecutor, "prepare_invocation", _prepare_invocation), patch.object(
        ProviderExecutor, "execute", _execute
    ):
        executor.execute()

    assert call_index["value"] == 6
    assert not (workspace / "docs/backlog/active/2026-04-22-ready-item.md").exists()
    done_path = workspace / "docs/backlog/done/2026-04-22-ready-item.md"
    assert done_path.is_file()
    assert "plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-22-ready-item/execution_plan.md" in (
        done_path.read_text(encoding="utf-8")
    )


def test_neurips_steered_backlog_runtime_recovers_running_item_without_relaunch(tmp_path):
    workspace = tmp_path / "workspace"
    workflow_path = _copy_runtime_files(workspace)

    provider_sequence = [
        ("SelectNextItem", _write_first_selector),
        ("ReviewOrUpdateRoadmap", _write_roadmap_sync_outputs),
        ("DraftPlan", _write_plan_draft),
        ("ReviewPlanTracked", _write_plan_review),
        ("ExecuteImplementation", _write_running_execution_state),
        ("ReviewOrUpdateRoadmap", _write_roadmap_sync_outputs),
        ("DraftPlan", _write_plan_draft),
        ("ReviewPlanTracked", _write_plan_review),
        ("ExecuteImplementation", _write_execution_report),
        ("ReviewImplementation", _write_implementation_review),
    ]
    call_index = {"value": 0}

    def _prepare_invocation(_self, *args, **kwargs):
        return SimpleNamespace(input_mode="stdin", prompt=kwargs.get("prompt_content", "")), None

    def _execute(_self, _invocation, **kwargs):
        expected_step, writer = provider_sequence[call_index["value"]]
        actual_step = kwargs.get("step_name")
        if actual_step is not None:
            assert actual_step == expected_step
        call_index["value"] += 1
        writer(workspace)
        return SimpleNamespace(
            exit_code=0,
            stdout=b"ok",
            stderr=b"",
            duration_ms=1,
            error=None,
            missing_placeholders=None,
            invalid_prompt_placeholder=False,
            raw_stdout=None,
            normalized_stdout=None,
            provider_session=None,
        )

    loader = WorkflowLoader(workspace)
    workflow = loader.load(workflow_path)
    workflow_relpath = workflow_path.relative_to(workspace).as_posix()
    bound_inputs = bind_workflow_inputs(workflow_input_contracts(workflow), {}, workspace)
    state_manager = StateManager(workspace=workspace, run_id="test-run")
    state_manager.initialize(workflow_relpath, _bundle_context_dict(workflow), bound_inputs=bound_inputs)
    executor = WorkflowExecutor(workflow, workspace, state_manager)

    with patch.object(ProviderExecutor, "prepare_invocation", _prepare_invocation), patch.object(
        ProviderExecutor, "execute", _execute
    ):
        state = executor.execute()

    assert call_index["value"] == 10
    summary = json.loads(
        (workspace / "artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog-drain-summary.json").read_text(
            encoding="utf-8"
        )
    )
    run_state = json.loads(
        (workspace / "state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/run_state.json").read_text(
            encoding="utf-8"
        )
    )

    assert summary["drain_status"] == "BLOCKED"
    assert summary["completed_item_count"] == 1
    assert run_state["blocked_items"] == {}
    assert run_state["current_item"] is None
    assert run_state["completed_items"] == ["2026-04-22-ready-item"]
    assert (workspace / "docs/backlog/done/2026-04-22-ready-item.md").is_file()
    assert not (workspace / "docs/backlog/in_progress/2026-04-22-ready-item.md").exists()


def test_neurips_steered_backlog_runtime_marks_semantic_block_only_on_explicit_block(tmp_path):
    workspace = tmp_path / "workspace"
    workflow_path = _copy_runtime_files(workspace)

    provider_sequence = [
        ("SelectNextItem", _write_first_selector),
        ("ReviewOrUpdateRoadmap", _write_roadmap_sync_outputs),
        ("DraftPlan", _write_plan_draft),
        ("ReviewPlanTracked", _write_plan_review),
        ("ExecuteImplementation", _write_blocked_execution_state),
    ]
    call_index = {"value": 0}

    def _prepare_invocation(_self, *args, **kwargs):
        return SimpleNamespace(input_mode="stdin", prompt=kwargs.get("prompt_content", "")), None

    def _execute(_self, _invocation, **kwargs):
        expected_step, writer = provider_sequence[call_index["value"]]
        actual_step = kwargs.get("step_name")
        if actual_step is not None:
            assert actual_step == expected_step
        call_index["value"] += 1
        writer(workspace)
        return SimpleNamespace(
            exit_code=0,
            stdout=b"ok",
            stderr=b"",
            duration_ms=1,
            error=None,
            missing_placeholders=None,
            invalid_prompt_placeholder=False,
            raw_stdout=None,
            normalized_stdout=None,
            provider_session=None,
        )

    loader = WorkflowLoader(workspace)
    workflow = loader.load(workflow_path)
    workflow_relpath = workflow_path.relative_to(workspace).as_posix()
    bound_inputs = bind_workflow_inputs(workflow_input_contracts(workflow), {}, workspace)
    state_manager = StateManager(workspace=workspace, run_id="test-run")
    state_manager.initialize(workflow_relpath, _bundle_context_dict(workflow), bound_inputs=bound_inputs)
    executor = WorkflowExecutor(workflow, workspace, state_manager)

    with patch.object(ProviderExecutor, "prepare_invocation", _prepare_invocation), patch.object(
        ProviderExecutor, "execute", _execute
    ):
        state = executor.execute()

    assert call_index["value"] == 5
    summary = json.loads(
        (workspace / "artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog-drain-summary.json").read_text(
            encoding="utf-8"
        )
    )
    run_state = json.loads(
        (workspace / "state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/run_state.json").read_text(
            encoding="utf-8"
        )
    )

    assert summary["drain_status"] == "BLOCKED"
    assert (
        run_state["blocked_items"]["2026-04-22-ready-item"]["reason"]
        == "Implementation phase reported a semantic blocker. See progress report."
    )


def test_neurips_steered_backlog_runtime_drafts_gap_item_and_continues_without_relaunch(tmp_path):
    workspace = tmp_path / "workspace"
    workflow_path = _copy_runtime_files(workspace)
    _replace_active_backlog_with_future_phase_gap_fixture(workspace)

    provider_sequence = [
        ("DraftMissingBacklogItem", _write_gap_draft),
        ("SelectNextItem", _write_selector_blocked_after_gap),
    ]
    call_index = {"value": 0}

    def _prepare_invocation(_self, *args, **kwargs):
        return SimpleNamespace(input_mode="stdin", prompt=kwargs.get("prompt_content", "")), None

    def _execute(_self, _invocation, **kwargs):
        expected_step, writer = provider_sequence[call_index["value"]]
        actual_step = kwargs.get("step_name")
        if actual_step is not None:
            assert actual_step == expected_step
        call_index["value"] += 1
        writer(workspace)
        return SimpleNamespace(
            exit_code=0,
            stdout=b"ok",
            stderr=b"",
            duration_ms=1,
            error=None,
            missing_placeholders=None,
            invalid_prompt_placeholder=False,
            raw_stdout=None,
            normalized_stdout=None,
            provider_session=None,
        )

    loader = WorkflowLoader(workspace)
    workflow = loader.load(workflow_path)
    workflow_relpath = workflow_path.relative_to(workspace).as_posix()
    bound_inputs = bind_workflow_inputs(workflow_input_contracts(workflow), {}, workspace)
    state_manager = StateManager(workspace=workspace, run_id="test-run")
    state_manager.initialize(workflow_relpath, _bundle_context_dict(workflow), bound_inputs=bound_inputs)
    executor = WorkflowExecutor(workflow, workspace, state_manager)

    with patch.object(ProviderExecutor, "prepare_invocation", _prepare_invocation), patch.object(
        ProviderExecutor, "execute", _execute
    ):
        executor.execute()

    assert call_index["value"] == 2
    assert (workspace / "docs/backlog/active/2026-04-27-cdi-item.md").is_file()
    assert (workspace / "docs/backlog/active/2026-04-28-phase2-gap.md").is_file()
    summary = json.loads(
        (workspace / "artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog-drain-summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["drain_status"] == "BLOCKED"
