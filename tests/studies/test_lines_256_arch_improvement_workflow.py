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


def test_lines_256_arch_improvement_workflow_uses_v27_repeat_until_and_30m_run_budget():
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
    assert run_candidate["timeout_sec"] == 1800

    debug_step = next(step for step in loop_steps if step["name"] == "DebugCandidateCrash")
    assert debug_step["input_file"] == "prompts/workflows/lines_256_arch_improvement/debug_crash.md"
    assert (
        debug_step["output_bundle"]["path"]
        == "state/lines_256_arch_improvement/debug_candidate_metadata.json"
    )

    rerun_step = next(step for step in loop_steps if step["name"] == "RunDebuggedCandidateExperiment")
    assert rerun_step["timeout_sec"] == 1800
