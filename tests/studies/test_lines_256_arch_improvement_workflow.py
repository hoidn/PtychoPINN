from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_prompt_index_registers_lines_256_experiment_prompt():
    index = (REPO_ROOT / "prompts/index.md").read_text(encoding="utf-8")

    assert "workflows/lines_256_arch_improvement/experiment_step.md" in index

    prompt = REPO_ROOT / "prompts/workflows/lines_256_arch_improvement/experiment_step.md"
    assert prompt.exists()


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
    route_step = next(step for step in loop_steps if step["name"] == "RouteExperimentReadiness")
    ready_case = route_step["match"]["cases"]["READY"]["steps"]
    run_candidate = next(step for step in ready_case if step["name"] == "RunCandidateExperiment")
    assert run_candidate["timeout_sec"] == 1800
