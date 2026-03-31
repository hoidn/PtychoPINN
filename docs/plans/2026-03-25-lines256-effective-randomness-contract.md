# Lines 256 Effective Randomness Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the brittle literal-seed blocker in the `lines_256` experiment loop with a session-level effective-randomness contract, while also fixing the runner so the requested seed propagates into the Torch training stack.

**Architecture:** The Torch runner will publish a small `randomness_contract.json` artifact alongside each scored run. The workflow baseline harvest will persist that contract into `accepted_state.json`, and the loop contract will compare candidate effective randomness metadata to the accepted session baseline instead of treating any internal seed mismatch as an automatic blocker. Prompts/docs will describe the same higher-level rule without re-owning the deterministic comparison logic.

**Tech Stack:** Python, pytest, agent-orchestration DSL v2.7, YAML, Markdown

---

### Task 1: Add Failing Tests For Seed Propagation And Randomness Artifact Emission

**Files:**
- Modify: `tests/torch/test_grid_lines_torch_runner.py`
- Modify: `tests/studies/test_lines_256_arch_improvement_workflow.py`

- [ ] **Step 1: Add a failing runner-config test**

Add a test to `tests/torch/test_grid_lines_torch_runner.py` asserting `setup_torch_configs()` threads `cfg.seed` into `training_config.subsample_seed`.

- [ ] **Step 2: Add a failing runner-artifact test**

Add a test to `tests/torch/test_grid_lines_torch_runner.py` asserting `run_grid_lines_torch()` writes `runs/pinn_<arch>/randomness_contract.json` with `requested_seed`, `effective_subsample_seed`, and `effective_lightning_seed`.

- [ ] **Step 3: Add a failing workflow contract test**

Add a test to `tests/studies/test_lines_256_arch_improvement_workflow.py` asserting the `lines_256` workflow harvest path records baseline randomness metadata in `accepted_state.json` and that candidate harvest references randomness-contract comparison rather than only literal seed wording.

- [ ] **Step 4: Run the new tests to verify RED**

Run:
```bash
cd /home/ollie/Documents/tmp/PtychoPINN
pytest tests/torch/test_grid_lines_torch_runner.py tests/studies/test_lines_256_arch_improvement_workflow.py -k "randomness or subsample_seed" -v
```
Expected: FAIL because the runner does not yet publish the contract and the workflow does not yet consume it.

### Task 2: Implement The Minimal Runner And Workflow Changes

**Files:**
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `workflows/agent_orchestration/lines_256_arch_improvement_session_loop.yaml`
- Modify: `workflows/library/lines_256_arch_improvement_iteration.yaml`

- [ ] **Step 1: Fix seed propagation**

In `scripts/studies/grid_lines_torch_runner.py`, set `training_config.subsample_seed = cfg.seed` inside `setup_torch_configs()` so the requested study seed reaches grouped-data generation and Lightning seeding.

- [ ] **Step 2: Publish effective randomness metadata**

In `scripts/studies/grid_lines_torch_runner.py`, compute a `randomness_contract` from the actual training config used by `run_torch_training()` and write it to `runs/pinn_<arch>/randomness_contract.json` during artifact save.

- [ ] **Step 3: Persist baseline randomness metadata**

In `workflows/agent_orchestration/lines_256_arch_improvement_session_loop.yaml`, make `HarvestBaselineOutputs` require the baseline `randomness_contract.json`, include it in `accepted_state.json`, and expose it through the accepted-state artifact.

- [ ] **Step 4: Gate candidate comparability on effective randomness parity**

In both `workflows/agent_orchestration/lines_256_arch_improvement_session_loop.yaml` and `workflows/library/lines_256_arch_improvement_iteration.yaml`, update candidate harvest so it reads the candidate `randomness_contract.json` and treats baseline-vs-candidate effective-randomness mismatch or missing metadata as `BLOCKED`, while leaving requested-vs-effective mismatch itself non-fatal if baseline and candidate agree.

- [ ] **Step 5: Run the focused tests to verify GREEN**

Run:
```bash
cd /home/ollie/Documents/tmp/PtychoPINN
pytest tests/torch/test_grid_lines_torch_runner.py tests/studies/test_lines_256_arch_improvement_workflow.py -k "randomness or subsample_seed" -v
```
Expected: PASS.

### Task 3: Align Prompts And Study Docs With The New Contract

**Files:**
- Modify: `prompts/workflows/lines_256_arch_improvement/experiment_step.md`
- Modify: `prompts/workflows/lines_256_arch_improvement/debug_crash.md`
- Modify: `docs/studies/lines_256_arch_improvement_loop.md`

- [ ] **Step 1: Update the study loop contract**

Revise `docs/studies/lines_256_arch_improvement_loop.md` so keepability depends on matching the accepted session’s effective randomness contract, not literal internal seed values, while still documenting the wrapper’s requested `seed=3`.

- [ ] **Step 2: Update the candidate prompt**

Revise `prompts/workflows/lines_256_arch_improvement/experiment_step.md` so smoke viability uses session-level effective randomness parity with `accepted_state.json`, and only blocks when that metadata is missing or inconsistent with the accepted reference.

- [ ] **Step 3: Update the crash-debug prompt**

Revise `prompts/workflows/lines_256_arch_improvement/debug_crash.md` to use the same effective-randomness parity rule for repaired candidates.

- [ ] **Step 4: Run prompt/workflow verification**

Run:
```bash
cd /home/ollie/Documents/tmp/PtychoPINN
pytest --collect-only tests/torch/test_grid_lines_torch_runner.py tests/studies/test_lines_256_arch_improvement_workflow.py -q
pytest tests/torch/test_grid_lines_torch_runner.py tests/studies/test_lines_256_arch_improvement_workflow.py tests/studies/test_run_lines_256_arch_experiment.py -v
PYTHONPATH=/home/ollie/Documents/agent-orchestration python -m orchestrator run workflows/agent_orchestration/lines_256_arch_improvement_session_loop.yaml --dry-run --stream-output
```
Expected: collection succeeds, targeted tests pass, workflow validation succeeds.
