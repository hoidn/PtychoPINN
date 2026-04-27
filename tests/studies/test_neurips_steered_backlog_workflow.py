from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(relpath: str) -> dict:
    return yaml.safe_load((REPO_ROOT / relpath).read_text(encoding="utf-8"))


def _walk_steps(steps: list[dict], prefix: str = ""):
    for step in steps:
        name = step["name"]
        path = f"{prefix} > {name}" if prefix else name
        yield path, step
        repeat_until = step.get("repeat_until")
        if isinstance(repeat_until, dict):
            yield from _walk_steps(repeat_until.get("steps", []), path)
        match = step.get("match")
        if isinstance(match, dict):
            for case_name, case in match.get("cases", {}).items():
                case_steps = case.get("steps", case) if isinstance(case, dict) else case
                yield from _walk_steps(case_steps, f"{path} > case {case_name}")
        if_branch = step.get("if")
        if isinstance(if_branch, dict):
            then_block = step.get("then", {})
            then_steps = then_block.get("steps", []) if isinstance(then_block, dict) else []
            else_block = step.get("else", {})
            else_steps = else_block.get("steps", []) if isinstance(else_block, dict) else []
            yield from _walk_steps(then_steps, f"{path} > then")
            yield from _walk_steps(else_steps, f"{path} > else")


def _step_by_name(workflow: dict, name: str) -> dict:
    for _, step in _walk_steps(workflow["steps"]):
        if step["name"] == name:
            return step
    raise AssertionError(f"Missing step {name}")


def _on_config(step: dict) -> dict:
    return step.get("on", step.get(True, {}))


def test_neurips_drain_workflow_uses_local_backlog_implementation_phase():
    workflow = _load_yaml("workflows/examples/neurips_steered_backlog_drain.yaml")
    selected_item_workflow = _load_yaml("workflows/library/neurips_selected_backlog_item.yaml")

    assert workflow["imports"] == {
        "selector": "../library/neurips_backlog_selector.yaml",
        "selected_item": "../library/neurips_selected_backlog_item.yaml",
    }
    assert selected_item_workflow["imports"] == {
        "roadmap_sync_phase": "./neurips_backlog_roadmap_sync_phase.yaml",
        "plan_phase": "./neurips_backlog_seeded_plan_phase.yaml",
        "implementation_phase": "./neurips_backlog_implementation_phase.yaml",
    }
    assert workflow["inputs"]["backlog_root"]["default"] == "docs/backlog/active"


def test_neurips_drain_workflow_tracks_authoritative_current_roadmap_path():
    workflow = _load_yaml("workflows/examples/neurips_steered_backlog_drain.yaml")
    selected_item_workflow = _load_yaml("workflows/library/neurips_selected_backlog_item.yaml")

    assert workflow["artifacts"]["roadmap"]["pointer"] == "${inputs.drain_state_root}/current_roadmap_path.txt"

    initialize = _step_by_name(workflow, "InitializeBacklogDrain")
    init_script = initialize["command"][-1]
    assert "current_roadmap_path.txt" in init_script
    assert "--roadmap-path" in init_script

    read_pointer = _step_by_name(workflow, "ReadCurrentRoadmapPointer")
    assert read_pointer["expected_outputs"][0]["path"].endswith("current_roadmap_path.txt")

    apply_sync = _step_by_name(selected_item_workflow, "ApplyRoadmapSyncDecision")
    apply_script = apply_sync["command"][2]
    assert "emitted_status == \"BLOCKED\"" in apply_script
    assert apply_sync["command"][3] == "${inputs.current_roadmap_pointer_path}"

    run_selected = _step_by_name(workflow, "RunSelectedItem")
    assert run_selected["with"]["current_roadmap_path"] == {
        "ref": "parent.steps.ReadCurrentRoadmapPointer.artifacts.current_roadmap_path"
    }

    run_plan = _step_by_name(selected_item_workflow, "RunFreshPlanPhase")
    assert run_plan["with"]["roadmap_path"] == {
        "ref": "root.steps.ApplyRoadmapSyncDecision.artifacts.current_roadmap_path"
    }


def test_neurips_drain_workflow_materializes_selected_item_inputs_and_rewrites_plan_path():
    workflow = _load_yaml("workflows/library/neurips_selected_backlog_item.yaml")

    materialize = _step_by_name(workflow, "MaterializeSelectedItemInputs")
    assert materialize["command"][:2] == [
        "python",
        "workflows/library/scripts/materialize_neurips_selected_item_inputs.py",
    ]
    field_names = {field["name"] for field in materialize["output_bundle"]["fields"]}
    assert {
        "selected_item_active_path",
        "selected_item_in_progress_path",
        "selected_item_context_path",
        "check_commands_path",
        "checks_report_target_path",
        "plan_target_path",
    } <= field_names
    assert "item_id" not in field_names

    rewrite = _step_by_name(workflow, "RewriteSelectedItemPlanPath")
    rewrite_script = rewrite["command"][2]
    assert "plan_path:" in rewrite_script
    assert "Missing frontmatter" in rewrite_script


def test_neurips_drain_workflow_dispatches_selected_items_through_local_selected_item_subworkflow():
    workflow = _load_yaml("workflows/examples/neurips_steered_backlog_drain.yaml")

    prepare_call_inputs = _step_by_name(workflow, "PrepareSelectedItemCallInputs")
    field_names = {field["name"] for field in prepare_call_inputs["output_bundle"]["fields"]}
    assert field_names == {"state_root", "current_roadmap_pointer_path"}

    run_selected = _step_by_name(workflow, "RunSelectedItem")
    assert run_selected["call"] == "selected_item"
    assert run_selected["with"]["state_root"] == {
        "ref": "parent.steps.PrepareSelectedItemCallInputs.artifacts.state_root"
    }
    assert run_selected["with"]["current_roadmap_pointer_path"] == {
        "ref": "parent.steps.PrepareSelectedItemCallInputs.artifacts.current_roadmap_pointer_path"
    }
    assert run_selected["with"]["current_roadmap_path"] == {
        "ref": "parent.steps.ReadCurrentRoadmapPointer.artifacts.current_roadmap_path"
    }
    assert run_selected["with"]["selector_state_root"] == {
        "ref": "parent.steps.PrepareSelectorInputs.artifacts.selector_state_root"
    }
    assert run_selected["with"]["manifest_path"] == {
        "ref": "parent.steps.BuildBacklogManifest.artifacts.manifest_path"
    }


def test_neurips_drain_workflow_supports_waiting_as_a_clean_terminal_outcome():
    workflow = _load_yaml("workflows/examples/neurips_steered_backlog_drain.yaml")

    assert workflow["outputs"]["drain_status"]["allowed"] == [
        "CONTINUE",
        "DONE",
        "WAITING",
        "BLOCKED",
    ]

    drain_loop = _step_by_name(workflow, "DrainBacklogItems")["repeat_until"]
    assert drain_loop["outputs"]["drain_status"]["allowed"] == [
        "CONTINUE",
        "DONE",
        "WAITING",
        "BLOCKED",
    ]

    waiting_compares = [
        predicate["compare"]["right"]
        for predicate in drain_loop["condition"]["any_of"]
        if isinstance(predicate, dict) and "compare" in predicate
    ]
    assert "WAITING" in waiting_compares


def test_neurips_drain_workflow_asserts_plan_and_implementation_approval_before_completion():
    workflow = _load_yaml("workflows/library/neurips_selected_backlog_item.yaml")

    assert _on_config(_step_by_name(workflow, "AssertPlanApproved"))["failure"]["goto"] == "RecordPlanBlocked"

    record_completed = _step_by_name(workflow, "RecordCompletedItem")
    assert record_completed["when"]["all_of"][0]["compare"]["right"] == "COMPLETED"
    assert record_completed["when"]["all_of"][1]["compare"]["right"] == "APPROVE"

    record_waiting = _step_by_name(workflow, "RecordImplementationWaiting")
    assert record_waiting["when"]["compare"]["right"] == "RUNNING"

    finalize = _step_by_name(workflow, "FinalizeSelectedItemOutcome")
    assert finalize["expected_outputs"][0]["allowed"] == ["CONTINUE", "WAITING", "BLOCKED"]

    script = record_completed["command"][2]
    assert "final_checks_report_path.txt" in script
    assert record_completed["command"][-1] == "${steps.ApplyRoadmapSyncDecision.artifacts.current_roadmap_path}"


def test_neurips_backlog_implementation_phase_emits_explicit_execution_state():
    workflow = _load_yaml("workflows/library/neurips_backlog_implementation_phase.yaml")

    assert workflow["inputs"]["check_commands_path"]["under"] == "state"
    assert workflow["inputs"]["checks_report_target_path"]["under"] == "artifacts/checks"
    assert workflow["artifacts"]["execution_report_target"]["pointer"] == (
        "${inputs.state_root}/execution_report_target_path.txt"
    )
    assert workflow["outputs"]["implementation_state"]["allowed"] == ["COMPLETED", "RUNNING", "BLOCKED"]
    assert workflow["outputs"]["implementation_review_decision"]["allowed"] == [
        "APPROVE",
        "REVISE",
        "NOT_APPLICABLE",
    ]

    initialize = _step_by_name(workflow, "InitializeImplementationPhasePaths")
    init_outputs = {entry["name"] for entry in initialize["expected_outputs"]}
    assert "execution_report_target_path" in init_outputs
    assert "progress_report_target_path" in init_outputs
    assert "execution_report_path" not in init_outputs

    execute = _step_by_name(workflow, "ExecuteImplementation")
    assert execute["asset_file"] == "prompts/neurips_backlog_implementation_phase/implement_implementation.md"
    assert execute["prompt_consumes"] == ["design", "plan", "execution_report_target", "progress_report_target"]
    assert "output_bundle" in execute
    bundle_fields = {field["name"]: field for field in execute["output_bundle"]["fields"]}
    assert bundle_fields["implementation_state"]["allowed"] == ["COMPLETED", "RUNNING", "BLOCKED"]
    assert bundle_fields["progress_report_path"]["required"] is False
    assert bundle_fields["execution_report_path"]["required"] is False

    publish_execution = _step_by_name(workflow, "PublishExecutionReport")
    assert publish_execution["when"]["compare"]["right"] == "COMPLETED"

    publish_progress = _step_by_name(workflow, "PublishProgressReport")
    assert publish_progress["when"]["compare"]["right"] == "RUNNING"

    publish_blocked_progress = _step_by_name(workflow, "PublishBlockedProgressReport")
    assert publish_blocked_progress["when"]["compare"]["right"] == "BLOCKED"

    review_loop_step = _step_by_name(workflow, "ImplementationReviewLoop")
    loop = review_loop_step["repeat_until"]["steps"]
    assert loop[0]["name"] == "RouteIterationWork"

    route_iteration = _step_by_name(workflow, "RouteIterationWork")
    assert route_iteration["if"]["compare"]["right"] == "COMPLETED"

    run_checks = _step_by_name(workflow, "RunTargetedChecks")
    assert run_checks["command"][:2] == [
        "python",
        "workflows/library/scripts/run_neurips_backlog_checks.py",
    ]
    assert {publish["artifact"] for publish in run_checks["publishes"]} == {"checks_report"}

    publish_execution_report = _step_by_name(workflow, "PublishExecutionReport")
    assert publish_execution_report["publishes"] == [
        {"artifact": "execution_report", "from": "execution_report_path"}
    ]
    assert publish_execution_report["expected_outputs"][0]["path"].endswith("execution_report_path.txt")

    review = _step_by_name(workflow, "ReviewImplementation")
    assert review["asset_file"] == "prompts/neurips_backlog_implementation_phase/review_implementation.md"
    assert review["prompt_consumes"] == ["design", "plan", "execution_report", "checks_report"]

    fix = _step_by_name(workflow, "FixImplementation")
    assert fix["asset_file"] == "prompts/neurips_backlog_implementation_phase/fix_implementation.md"
    assert fix["prompt_consumes"] == [
        "design",
        "plan",
        "execution_report",
        "execution_report_target",
        "checks_report",
        "implementation_review_report",
    ]
    assert "expected_outputs" not in fix
    assert "publishes" not in fix

    publish_updated_execution_report = _step_by_name(workflow, "PublishUpdatedExecutionReport")
    assert publish_updated_execution_report["publishes"] == [
        {"artifact": "execution_report", "from": "execution_report_path"}
    ]


def test_docs_and_template_register_new_queue_and_runbook_contract():
    docs_index = (REPO_ROOT / "docs/index.md").read_text(encoding="utf-8")
    template = (REPO_ROOT / "docs/backlog/templates/backlog_item_workflow.md").read_text(encoding="utf-8")

    assert "steering.md" in docs_index
    assert "workflows/neurips_steered_backlog_drain.md" in docs_index
    assert "docs/backlog/in_progress/" in template
    assert "may be rewritten" in template
