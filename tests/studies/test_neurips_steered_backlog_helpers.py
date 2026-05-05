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


def test_build_manifest_records_malformed_frontmatter_as_invalid_item(tmp_path):
    workspace = _workspace(tmp_path)
    bad_item = workspace / "docs/backlog/active/2026-04-22-bad-item.md"
    bad_item.write_text("# no frontmatter\n", encoding="utf-8")
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
    invalid = {item["item_id"]: item for item in manifest["invalid_items"]}
    assert "2026-04-22-bad-item" in invalid
    assert any("frontmatter" in reason.lower() for reason in invalid["2026-04-22-bad-item"]["invalid_reasons"])


def test_roadmap_gate_filters_out_phase3_cdi_when_phase2_required(tmp_path):
    workspace = _workspace(tmp_path)
    cdi_item = workspace / "docs/backlog/active/2026-04-27-cdi-item.md"
    cdi_item.write_text(
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
    manifest_path = workspace / "state/manifest.json"
    _run_script(
        workspace,
        "workflows/library/scripts/build_neurips_backlog_manifest.py",
        "--backlog-root",
        "docs/backlog/active",
        "--output",
        str(manifest_path),
    )

    output_path = workspace / "state/gate.json"
    _run_script(
        workspace,
        "workflows/library/scripts/reconcile_neurips_backlog_roadmap_gate.py",
        "--manifest-path",
        str(manifest_path),
        "--gate-policy-path",
        "docs/backlog/roadmap_gate.json",
        "--progress-ledger-path",
        "state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json",
        "--run-state-path",
        "state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/run_state.json",
        "--output",
        str(output_path),
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["gate_status"] == "ELIGIBLE"
    assert {item["item_id"] for item in payload["eligible_items"]} == {"2026-04-22-ready-item"}
    assert "2026-04-27-cdi-item" in {item["item_id"] for item in payload["ineligible_items"]}
    eligible_manifest = json.loads((workspace / payload["eligible_manifest_path"]).read_text(encoding="utf-8"))
    assert eligible_manifest["active_count"] == 1
    assert eligible_manifest["items"][0]["item_id"] == "2026-04-22-ready-item"


def test_roadmap_gate_reports_backlog_gap_when_only_future_phase_items_remain(tmp_path):
    workspace = _workspace(tmp_path)
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
    manifest_path = workspace / "state/manifest.json"
    _run_script(
        workspace,
        "workflows/library/scripts/build_neurips_backlog_manifest.py",
        "--backlog-root",
        "docs/backlog/active",
        "--output",
        str(manifest_path),
    )

    output_path = workspace / "state/gate.json"
    _run_script(
        workspace,
        "workflows/library/scripts/reconcile_neurips_backlog_roadmap_gate.py",
        "--manifest-path",
        str(manifest_path),
        "--gate-policy-path",
        "docs/backlog/roadmap_gate.json",
        "--progress-ledger-path",
        "state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json",
        "--run-state-path",
        "state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/run_state.json",
        "--output",
        str(output_path),
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["gate_status"] == "BACKLOG_GAP"
    assert payload["eligible_count"] == 0
    gap_request = json.loads((workspace / payload["gap_request_path"]).read_text(encoding="utf-8"))
    assert gap_request["required_scope_summary"] == "Remaining Phase 2 PDEBench full-training evidence gate"
    assert gap_request["gap_item_target_dir"] == "docs/backlog/active"


def test_roadmap_gate_reports_done_for_empty_manifest(tmp_path):
    workspace = _workspace(tmp_path)
    active_root = workspace / "docs/backlog/active"
    for item in active_root.glob("*.md"):
        item.unlink()
    manifest_path = workspace / "state/manifest.json"
    _run_script(
        workspace,
        "workflows/library/scripts/build_neurips_backlog_manifest.py",
        "--backlog-root",
        "docs/backlog/active",
        "--output",
        str(manifest_path),
    )

    output_path = workspace / "state/gate.json"
    _run_script(
        workspace,
        "workflows/library/scripts/reconcile_neurips_backlog_roadmap_gate.py",
        "--manifest-path",
        str(manifest_path),
        "--gate-policy-path",
        "docs/backlog/roadmap_gate.json",
        "--progress-ledger-path",
        "state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json",
        "--run-state-path",
        "state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/run_state.json",
        "--output",
        str(output_path),
    )

    assert json.loads(output_path.read_text(encoding="utf-8"))["gate_status"] == "DONE"


def test_roadmap_gate_rejects_policy_without_allowed_prefixes(tmp_path):
    workspace = _workspace(tmp_path)
    bad_policy = workspace / "docs/backlog/bad_gate.json"
    bad_policy.write_text(
        json.dumps(
            {
                "gate_version": 1,
                "allowed_roadmap_phase_prefixes": [],
                "disallowed_roadmap_phase_prefixes": ["phase-3-"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = workspace / "state/manifest.json"
    _run_script(
        workspace,
        "workflows/library/scripts/build_neurips_backlog_manifest.py",
        "--backlog-root",
        "docs/backlog/active",
        "--output",
        str(manifest_path),
    )

    result = _run_script(
        workspace,
        "workflows/library/scripts/reconcile_neurips_backlog_roadmap_gate.py",
        "--manifest-path",
        str(manifest_path),
        "--gate-policy-path",
        bad_policy.relative_to(workspace).as_posix(),
        "--progress-ledger-path",
        "state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json",
        "--run-state-path",
        "state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/run_state.json",
        "--output",
        str(workspace / "state/gate.json"),
        check=False,
    )

    assert result.returncode != 0
    assert "allowed_roadmap_phase_prefixes" in result.stderr


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
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["source_manifest_path"] = "state/raw_manifest.json"
    manifest_payload["roadmap_gate_status"] = "ELIGIBLE"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")

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


def test_materialize_selected_active_item_fails_when_plan_target_is_missing(tmp_path):
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
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["source_manifest_path"] = "state/raw_manifest.json"
    manifest_payload["roadmap_gate_status"] = "ELIGIBLE"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
    (workspace / "docs/plans/legacy-ready-plan.md").unlink()

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

    result = _run_script(
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
        check=False,
    )

    assert result.returncode != 0
    assert "plan_path target does not exist" in result.stderr
    assert not output_path.exists()




def test_gap_draft_validator_accepts_valid_phase2_item(tmp_path):
    workspace = _workspace(tmp_path)
    gap_request = workspace / "state/gap_request.json"
    gap_request.parent.mkdir(parents=True, exist_ok=True)
    gap_request.write_text(
        json.dumps(
            {
                "current_gate_id": "phase-2-pdebench-full-training-evidence",
                "allowed_roadmap_phase_prefixes": ["phase-2-pdebench-"],
                "disallowed_roadmap_phase_prefixes": ["phase-3-"],
                "gap_item_target_dir": "docs/backlog/active",
                "gap_plan_target_root": "docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog-gaps",
                "required_scope_summary": "Remaining Phase 2 PDEBench full-training evidence gate",
                "roadmap_path": "docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    plan_path = workspace / "docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog-gaps/2026-04-28-phase2.md"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text("# Phase 2 Gap Plan\n", encoding="utf-8")
    item_path = workspace / "docs/backlog/active/2026-04-28-phase2-gap.md"
    item_path.write_text(
        "\n".join(
            [
                "---",
                "priority: 5",
                "plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog-gaps/2026-04-28-phase2.md",
                "check_commands:",
                "  - python -m pytest tests/studies/test_neurips_steered_backlog_helpers.py -q",
                "related_roadmap_phases:",
                "  - phase-2-pdebench-full-training-evidence",
                "---",
                "",
                "# Backlog Item: Phase 2 Gap",
                "",
                "## Objective",
                "- Close the missing Phase 2 evidence gap.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    draft_bundle = workspace / "state/draft_bundle.json"
    draft_bundle.write_text(
        json.dumps(
            {
                "draft_status": "DRAFTED",
                "backlog_item_path": item_path.relative_to(workspace).as_posix(),
                "seed_plan_path": plan_path.relative_to(workspace).as_posix(),
                "summary": "Drafted Phase 2 gap item.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    output = workspace / "state/draft_validation.json"
    _run_script(
        workspace,
        "workflows/library/scripts/validate_neurips_backlog_gap_draft.py",
        "--gap-request-path",
        str(gap_request),
        "--draft-bundle-path",
        str(draft_bundle),
        "--gate-policy-path",
        "docs/backlog/roadmap_gate.json",
        "--output",
        str(output),
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["draft_validation_status"] == "VALID"
    assert payload["backlog_item_path"] == "docs/backlog/active/2026-04-28-phase2-gap.md"


def test_gap_draft_validator_accepts_block_scalar_check_commands(tmp_path):
    workspace = _workspace(tmp_path)
    gap_request = workspace / "state/gap_request.json"
    gap_request.parent.mkdir(parents=True, exist_ok=True)
    gap_request.write_text(
        json.dumps(
            {
                "allowed_roadmap_phase_prefixes": ["phase-2-pdebench-"],
                "disallowed_roadmap_phase_prefixes": ["phase-3-"],
                "gap_item_target_dir": "docs/backlog/active",
                "gap_plan_target_root": "docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog-gaps",
                "roadmap_path": "docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    plan_path = workspace / "docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog-gaps/2026-04-28-block.md"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text("# Phase 2 Block Command Plan\n", encoding="utf-8")
    item_path = workspace / "docs/backlog/active/2026-04-28-block-command-gap.md"
    item_path.write_text(
        "\n".join(
            [
                "---",
                "priority: 5",
                "plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog-gaps/2026-04-28-block.md",
                "check_commands:",
                "  - |",
                "    python - <<'PY'",
                "    print('valid block command')",
                "    PY",
                "related_roadmap_phases:",
                "  - phase-2-pdebench-full-training-evidence",
                "---",
                "",
                "# Backlog Item: Phase 2 Block Command Gap",
                "",
                "## Objective",
                "- Close the missing Phase 2 evidence gap.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    draft_bundle = workspace / "state/draft_bundle.json"
    draft_bundle.write_text(
        json.dumps(
            {
                "draft_status": "DRAFTED",
                "backlog_item_path": item_path.relative_to(workspace).as_posix(),
                "seed_plan_path": plan_path.relative_to(workspace).as_posix(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    output = workspace / "state/draft_validation.json"
    _run_script(
        workspace,
        "workflows/library/scripts/validate_neurips_backlog_gap_draft.py",
        "--gap-request-path",
        str(gap_request),
        "--draft-bundle-path",
        str(draft_bundle),
        "--gate-policy-path",
        "docs/backlog/roadmap_gate.json",
        "--output",
        str(output),
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["draft_validation_status"] == "VALID"
    assert payload["backlog_item_path"] == "docs/backlog/active/2026-04-28-block-command-gap.md"


def test_gap_draft_validator_rejects_future_phase_item(tmp_path):
    workspace = _workspace(tmp_path)
    gap_request = workspace / "state/gap_request.json"
    gap_request.parent.mkdir(parents=True, exist_ok=True)
    gap_request.write_text(
        json.dumps(
            {
                "allowed_roadmap_phase_prefixes": ["phase-2-pdebench-"],
                "disallowed_roadmap_phase_prefixes": ["phase-3-"],
                "gap_item_target_dir": "docs/backlog/active",
                "gap_plan_target_root": "docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog-gaps",
                "roadmap_path": "docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    plan_path = workspace / "docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog-gaps/2026-04-28-future.md"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text("# Future Plan\n", encoding="utf-8")
    item_path = workspace / "docs/backlog/active/2026-04-28-future.md"
    item_path.write_text(
        "\n".join(
            [
                "---",
                "priority: 5",
                "plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog-gaps/2026-04-28-future.md",
                "check_commands:",
                "  - python -c \"print('future')\"",
                "related_roadmap_phases:",
                "  - phase-3-cdi-anchor-regeneration",
                "---",
                "",
                "# Backlog Item: Future",
                "",
                "## Objective",
                "- Future work.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    draft_bundle = workspace / "state/draft_bundle.json"
    draft_bundle.write_text(
        json.dumps(
            {
                "draft_status": "DRAFTED",
                "backlog_item_path": item_path.relative_to(workspace).as_posix(),
                "seed_plan_path": plan_path.relative_to(workspace).as_posix(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = _run_script(
        workspace,
        "workflows/library/scripts/validate_neurips_backlog_gap_draft.py",
        "--gap-request-path",
        str(gap_request),
        "--draft-bundle-path",
        str(draft_bundle),
        "--gate-policy-path",
        "docs/backlog/roadmap_gate.json",
        "--output",
        str(workspace / "state/draft_validation.json"),
        check=False,
    )

    assert result.returncode != 0
    assert "disallowed" in result.stderr.lower() or "allowed" in result.stderr.lower()
    failure_payload = json.loads((workspace / "state/draft_validation.json").read_text(encoding="utf-8"))
    assert failure_payload["draft_validation_status"] == "INVALID"
    assert "disallowed" in failure_payload["reason"].lower() or "allowed" in failure_payload["reason"].lower()


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
