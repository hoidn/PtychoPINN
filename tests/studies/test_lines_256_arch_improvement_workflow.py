from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_prompt_index_registers_lines_256_experiment_prompt():
    index = (REPO_ROOT / "prompts/index.md").read_text(encoding="utf-8")

    assert "workflows/lines_256_arch_improvement/experiment_step.md" in index
    assert "workflows/lines_256_arch_improvement/debug_crash.md" in index

    prompt = REPO_ROOT / "prompts/workflows/lines_256_arch_improvement/experiment_step.md"
    assert prompt.exists()
    debug_prompt = REPO_ROOT / "prompts/workflows/lines_256_arch_improvement/debug_crash.md"
    assert debug_prompt.exists()


def test_studies_index_registers_lines_256_session_workflow():
    index = (REPO_ROOT / "docs/studies/index.md").read_text(encoding="utf-8")

    assert "lines_256_arch_improvement_session_loop.yaml" in index
    assert "lines_256_arch_improvement_session_loop_v2_call.yaml" in index


def test_lines_256_arch_improvement_workflow_uses_v27_repeat_until_and_timeout_outcome_budgeting():
    workflow_path = REPO_ROOT / Path(
        "workflows/agent_orchestration/lines_256_arch_improvement_session_loop.yaml"
    )
    assert workflow_path.exists()

    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))

    assert workflow["version"] == "2.7"

    run_baseline = next(step for step in workflow["steps"] if step["name"] == "RunBaseline")
    assert run_baseline["timeout_sec"] == 1800

    experiment_loop = next(
        step for step in workflow["steps"] if step["name"] == "ExperimentLoop"
    )
    assert "repeat_until" in experiment_loop

    loop_steps = experiment_loop["repeat_until"]["steps"]
    assert (
        experiment_loop["repeat_until"]["outputs"]["loop_decision"]["from"]["ref"]
        == "self.steps.FinalizeIterationDecision.artifacts.loop_decision"
    )

    prepare_context = next(step for step in loop_steps if step["name"] == "PrepareCandidateContext")
    bundle_fields = prepare_context["output_bundle"]["fields"]
    bundle_field_names = {field["name"] for field in bundle_fields}
    assert "debug_candidate_metadata_path" in bundle_field_names
    assert "debug_candidate_paths_file" in bundle_field_names

    run_candidate = next(step for step in loop_steps if step["name"] == "RunCandidateExperiment")
    assert run_candidate["timeout_sec"] == 1860
    assert "timeout=1770" in run_candidate["command"][-1]

    harvest_candidate = next(step for step in loop_steps if step["name"] == "HarvestCandidateOutputs")
    assert "decision = \"TIMEOUT\"" in harvest_candidate["command"][-1]
    decision_field = next(
        field
        for field in harvest_candidate["output_bundle"]["fields"]
        if field["name"] == "decision"
    )
    assert "TIMEOUT" in decision_field["allowed"]

    debug_step = next(step for step in loop_steps if step["name"] == "DebugCandidateCrash")
    assert debug_step["input_file"] == "prompts/workflows/lines_256_arch_improvement/debug_crash.md"
    assert (
        debug_step["output_bundle"]["path"]
        == "state/lines_256_arch_improvement/debug_candidate_metadata.json"
    )

    handle_initial = next(
        step for step in loop_steps if step["name"] == "HandleInitialCandidateOutcome"
    )
    assert handle_initial["command"][:2] == [
        "python",
        "scripts/studies/lines_256_handle_candidate_outcome.py",
    ]
    assert handle_initial["command"][-4:] == [
        "--session-state-dir",
        "state/lines_256_arch_improvement",
        "--state-dir",
        "state/lines_256_arch_improvement",
    ]

    handle_debugged = next(
        step for step in loop_steps if step["name"] == "HandleDebuggedCandidateOutcome"
    )
    assert handle_debugged["command"][:2] == [
        "python",
        "scripts/studies/lines_256_handle_candidate_outcome.py",
    ]
    assert handle_debugged["command"][-4:] == [
        "--session-state-dir",
        "state/lines_256_arch_improvement",
        "--state-dir",
        "state/lines_256_arch_improvement",
    ]

    rerun_step = next(step for step in loop_steps if step["name"] == "RunDebuggedCandidateExperiment")
    assert rerun_step["timeout_sec"] == 1860
    assert "timeout=1770" in rerun_step["command"][-1]

    assess_initial = next(
        step for step in loop_steps if step["name"] == "AssessInitialCandidateOutcome"
    )
    initial_outcome_field = assess_initial["output_bundle"]["fields"][0]
    assert "TIMEOUT" in initial_outcome_field["allowed"]


def test_lines_256_workflow_preflights_dataset_metadata_and_probe_gallery_contract():
    workflow_path = REPO_ROOT / Path(
        "workflows/agent_orchestration/lines_256_arch_improvement_session_loop.yaml"
    )
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))

    validate_inputs = next(step for step in workflow["steps"] if step["name"] == "ValidateStudyInputs")
    validate_script = validate_inputs["command"][-1]
    assert "outputs/lines_256_arch_improvement/datasets/N256/gs1/train.npz" in validate_script
    assert "outputs/lines_256_arch_improvement/datasets/N256/gs1/test.npz" in validate_script
    assert "allow_pickle=True" in validate_script
    assert "probe_scale_mode" in validate_script
    assert "pad_preserve" in validate_script
    assert "probe_npz" in validate_script
    assert "Run1084_recon3_postPC_shrunk_3.npz" in validate_script

    create_session = next(step for step in workflow["steps"] if step["name"] == "CreateSession")
    create_session_script = create_session["command"][-1]
    assert "__baseline__compare_amp_phase_probe.png" in create_session_script

    harvest_baseline = next(step for step in workflow["steps"] if step["name"] == "HarvestBaselineOutputs")
    harvest_script = harvest_baseline["command"][-1]
    assert "compare_amp_phase_probe.png" in harvest_script

    loop_doc = (
        REPO_ROOT / "docs/studies/lines_256_arch_improvement_loop.md"
    ).read_text(encoding="utf-8")
    assert "TIMEOUT" in loop_doc
    assert "only `CRASH` should trigger the focused debug path" in loop_doc


def test_lines_256_prompts_pin_smoke_target_normalization_and_probe_gallery_name():
    experiment_prompt = (
        REPO_ROOT / "prompts/workflows/lines_256_arch_improvement/experiment_step.md"
    ).read_text(encoding="utf-8")
    debug_prompt = (
        REPO_ROOT / "prompts/workflows/lines_256_arch_improvement/debug_crash.md"
    ).read_text(encoding="utf-8")

    assert "--torch-mae-pred-l2-match-target" in experiment_prompt
    assert "--torch-mae-pred-l2-match-target" in debug_prompt
    assert "compare_amp_phase_probe.png" in experiment_prompt
    assert "compare_amp_phase_probe.png" in debug_prompt


def test_lines_256_arch_improvement_v2_call_workflow_extracts_iteration_into_subworkflow():
    workflow_path = REPO_ROOT / Path(
        "workflows/agent_orchestration/lines_256_arch_improvement_session_loop_v2_call.yaml"
    )
    assert workflow_path.exists()

    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))

    assert workflow["version"] == "2.7"
    assert workflow["imports"]["iteration_phase"] == "../library/lines_256_arch_improvement_iteration.yaml"

    experiment_loop = next(
        step for step in workflow["steps"] if step["name"] == "ExperimentLoop"
    )
    repeat_until = experiment_loop["repeat_until"]
    assert (
        repeat_until["outputs"]["loop_decision"]["from"]["ref"]
        == "self.steps.RunIteration.artifacts.loop_decision"
    )

    loop_steps = repeat_until["steps"]
    assert [step["name"] for step in loop_steps] == ["PrepareIterationCallInputs", "RunIteration"]
    prepare_inputs = loop_steps[0]
    assert prepare_inputs["output_bundle"]["fields"][0]["name"] == "write_root"
    assert prepare_inputs["output_bundle"]["fields"][0]["type"] == "relpath"
    run_iteration = loop_steps[1]
    assert run_iteration["call"] == "iteration_phase"
    assert (
        run_iteration["with"]["write_root"]["ref"]
        == "self.steps.PrepareIterationCallInputs.artifacts.write_root"
    )


def test_lines_256_arch_improvement_iteration_library_preserves_crash_debug_path():
    workflow_path = REPO_ROOT / Path(
        "workflows/library/lines_256_arch_improvement_iteration.yaml"
    )
    assert workflow_path.exists()

    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))

    assert workflow["version"] == "2.7"
    assert (
        workflow["outputs"]["loop_decision"]["from"]["ref"]
        == "root.steps.FinalizeIterationDecision.artifacts.loop_decision"
    )

    steps = workflow["steps"]
    names = [step["name"] for step in steps]
    assert "DebugCandidateCrash" in names
    assert "RunDebuggedCandidateExperiment" in names
    handle_initial = next(step for step in steps if step["name"] == "HandleInitialCandidateOutcome")
    assert handle_initial["command"][:2] == [
        "python",
        "scripts/studies/lines_256_handle_candidate_outcome.py",
    ]
    assert handle_initial["command"][-4:] == [
        "--session-state-dir",
        "state/lines_256_arch_improvement",
        "--state-dir",
        "${inputs.write_root}",
    ]
    handle_debugged = next(step for step in steps if step["name"] == "HandleDebuggedCandidateOutcome")
    assert handle_debugged["command"][:2] == [
        "python",
        "scripts/studies/lines_256_handle_candidate_outcome.py",
    ]
    assert handle_debugged["command"][-4:] == [
        "--session-state-dir",
        "state/lines_256_arch_improvement",
        "--state-dir",
        "${inputs.write_root}",
    ]
    run_candidate = next(step for step in steps if step["name"] == "RunCandidateExperiment")
    assert run_candidate["timeout_sec"] == 1860
    assert "timeout=1770" in run_candidate["command"][-1]

    harvest_candidate = next(step for step in steps if step["name"] == "HarvestCandidateOutputs")
    decision_field = next(
        field
        for field in harvest_candidate["output_bundle"]["fields"]
        if field["name"] == "decision"
    )
    assert "TIMEOUT" in decision_field["allowed"]

    rerun_step = next(step for step in steps if step["name"] == "RunDebuggedCandidateExperiment")
    assert rerun_step["timeout_sec"] == 1860
    assert "timeout=1770" in rerun_step["command"][-1]
