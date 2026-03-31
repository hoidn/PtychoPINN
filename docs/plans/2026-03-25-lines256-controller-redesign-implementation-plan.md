# lines_256 Controller Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a parallel `lines_256` v2 experiment path that replaces the brittle YAML loop with a deterministic Python session controller while keeping the legacy workflow intact until the new path proves itself.

**Architecture:** Add a study-specific controller script that owns session state, baseline/scored runs, keep-discard logic, and resume semantics. Keep the agent only for candidate proposal/debug, store v2 state in isolated session-local roots, and add at most a thin workflow wrapper that launches the controller rather than re-encoding the loop in YAML.

**Tech Stack:** Python 3.11, pytest, agent-orchestration CLI, existing `scripts/studies/run_lines_256_arch_experiment.py` wrapper, tmux launch/runbook conventions, JSON/TSV session artifacts

---

## File Map

### New files

- `scripts/studies/lines_256_session_controller.py`
  Main v2 controller CLI. Owns session init, baseline, loop phases, keep/discard, timeout/crash handling, and resume.

- `tests/studies/test_lines_256_session_controller.py`
  Focused unit/integration-style tests for controller state transitions and candidate-kind behavior.

- `workflows/agent_orchestration/lines_256_session_controller.yaml`
  Optional thin orchestrator wrapper that validates inputs and launches the controller.

- `docs/studies/lines_256_controller_loop.md`
  Study/runbook doc for the v2 controller path, separate from the legacy loop contract.

### Existing files to modify

- `docs/studies/index.md`
  Register the new v2 controller loop doc and keep the legacy loop labeled as legacy.

- `docs/index.md`
  Add discoverability for the new controller path and parallel-rollout status.

- `prompts/workflows/lines_256_arch_improvement/experiment_step.md`
  Keep task-local contract, but align with the controller-owned proposal artifact format if needed.

- `prompts/workflows/lines_256_arch_improvement/debug_crash.md`
  Align with the controller-owned crash-debug artifact format if needed.

- `tests/studies/test_lines_256_arch_improvement_workflow.py`
  Add thin-wrapper contract assertions for the new controller workflow without deleting legacy coverage.

## Task 1: Scaffold v2 session model and failing controller tests

**Files:**
- Create: `scripts/studies/lines_256_session_controller.py`
- Create: `tests/studies/test_lines_256_session_controller.py`
- Modify: `docs/plans/2026-03-25-lines256-controller-redesign.md` only if a clarifying implementation note is needed

- [ ] **Step 1: Write failing tests for the controller session skeleton**

Add tests for:
- session root creation under `state/lines_256_arch_improvement_v2/sessions/<session_id>/`
- `session.json` phase initialization
- isolated `results.tsv` creation with the exact header
- accepted-state path staying session-local instead of reusing legacy paths

Example test sketch:

```python
def test_controller_initializes_session_root(tmp_path):
    session = initialize_session(repo_root=tmp_path, session_id="20260325T000000Z")
    assert session.session_root == tmp_path / "state" / "lines_256_arch_improvement_v2" / "sessions" / "20260325T000000Z"
    assert (session.session_root / "session.json").exists()
    assert (session.session_root / "results.tsv").exists()
```

- [ ] **Step 2: Run the new tests to confirm they fail**

Run:

```bash
pytest tests/studies/test_lines_256_session_controller.py -k "initializes_session_root or creates_results_ledger" -v
```

Expected: FAIL because the controller module or helpers do not exist yet.

- [ ] **Step 3: Add the minimal controller/session scaffolding**

Implement:
- argument parsing for `start` and `resume`
- session-root path helpers
- `session.json` writer
- ledger bootstrap helper

Suggested public surface:

```python
def initialize_session(repo_root: Path, session_id: str | None = None) -> SessionState: ...
def load_session(session_root: Path) -> SessionState: ...
```

- [ ] **Step 4: Run the focused tests again**

Run:

```bash
pytest tests/studies/test_lines_256_session_controller.py -k "initializes_session_root or creates_results_ledger" -v
```

Expected: PASS.

- [ ] **Step 5: Commit the scaffolding**

```bash
git add scripts/studies/lines_256_session_controller.py tests/studies/test_lines_256_session_controller.py
git commit -m "Add lines 256 v2 session controller scaffold"
```

## Task 2: Implement fresh baseline run and harvest in the controller

**Files:**
- Modify: `scripts/studies/lines_256_session_controller.py`
- Modify: `tests/studies/test_lines_256_session_controller.py`
- Reuse: `scripts/studies/run_lines_256_arch_experiment.py`

- [ ] **Step 1: Write failing tests for baseline harvest**

Add tests for:
- baseline command creation
- baseline run-result parsing
- accepted-state JSON writing
- session-local comparison gallery copy
- randomness-contract capture

Example sketch:

```python
def test_controller_harvests_baseline_into_session_local_accepted_state(tmp_path):
    result = harvest_baseline(session, baseline_output_root)
    assert result.accepted_state["accepted_candidate_kind"] == "baseline"
    assert result.accepted_state["accepted_randomness_contract"]["requested_seed"] == 3
```

- [ ] **Step 2: Run the failing baseline tests**

Run:

```bash
pytest tests/studies/test_lines_256_session_controller.py -k "baseline" -v
```

Expected: FAIL until harvest behavior exists.

- [ ] **Step 3: Implement baseline launch and harvest**

Requirements:
- launch the existing thin wrapper with the fixed baseline control
- write `baseline_run_result.json` under the session root
- read `metrics.json`, `randomness_contract.json`, and `compare_amp_phase_probe.png`
- write session-local `accepted_state.json`
- append the baseline ledger row
- checkpoint `session.json` phase transitions around the baseline

- [ ] **Step 4: Run the baseline tests**

Run:

```bash
pytest tests/studies/test_lines_256_session_controller.py -k "baseline" -v
```

Expected: PASS.

- [ ] **Step 5: Run a narrow dry smoke of the controller baseline path**

Run:

```bash
python scripts/studies/lines_256_session_controller.py start --mode baseline-only --dry-run
```

Expected: exit `0`, session root created, baseline command printed without launching training.

- [ ] **Step 6: Commit the baseline slice**

```bash
git add scripts/studies/lines_256_session_controller.py tests/studies/test_lines_256_session_controller.py
git commit -m "Add lines 256 v2 baseline controller flow"
```

## Task 3: Define proposal artifacts and recent-history summary

**Files:**
- Modify: `scripts/studies/lines_256_session_controller.py`
- Modify: `tests/studies/test_lines_256_session_controller.py`
- Modify: `prompts/workflows/lines_256_arch_improvement/experiment_step.md`
- Modify: `prompts/workflows/lines_256_arch_improvement/debug_crash.md`

- [ ] **Step 1: Write failing tests for proposal-context generation**

Cover:
- compact recent-history summary artifact creation
- accepted-state injection into proposal context
- support for `candidate_kind in {"source", "run_config"}`

Example sketch:

```python
def test_controller_builds_compact_recent_history_summary(tmp_path):
    summary = build_recent_history_summary(session_root, max_rows=10)
    assert "discard" in summary["recent_outcomes"]
    assert len(summary["recent_attempts"]) <= 10
```

- [ ] **Step 2: Run the failing proposal-context tests**

Run:

```bash
pytest tests/studies/test_lines_256_session_controller.py -k "proposal_context or recent_history" -v
```

Expected: FAIL.

- [ ] **Step 3: Implement proposal-context and candidate-artifact schema**

Requirements:
- session-local `proposal_context.json`
- compact recent-history summary derived from v2 session ledger
- normalized proposal artifact that always includes:
  - `candidate_kind`
  - `base_ref`
  - `smoke_command`
  - `run_command`
  - `output_root`
  - `log_path`
  - `comparison_png_path`
  - `note`
  - `hypothesis`
- require `candidate_commit` and `candidate_paths_file` only for `source`

- [ ] **Step 4: Update prompts to match the controller-owned proposal artifact**

Keep prompts task-local:
- “prepare one viable candidate package”
- no workflow-loop language
- explicit `source` vs `run_config` forms
- no ledger/keep-discard ownership in prompts

- [ ] **Step 5: Run the proposal-context tests**

Run:

```bash
pytest tests/studies/test_lines_256_session_controller.py -k "proposal_context or recent_history" -v
```

Expected: PASS.

- [ ] **Step 6: Commit the proposal-artifact slice**

```bash
git add scripts/studies/lines_256_session_controller.py tests/studies/test_lines_256_session_controller.py prompts/workflows/lines_256_arch_improvement/experiment_step.md prompts/workflows/lines_256_arch_improvement/debug_crash.md
git commit -m "Add lines 256 v2 proposal artifact contract"
```

## Task 4: Implement candidate execution for `source` and `run_config`

**Files:**
- Modify: `scripts/studies/lines_256_session_controller.py`
- Modify: `tests/studies/test_lines_256_session_controller.py`

- [ ] **Step 1: Write failing tests for candidate-kind behavior**

Add tests for:
- `source` candidate rejection resets to `accepted_ref`
- `run_config` rejection performs no git reset
- `source` keep advances `accepted_ref`
- `run_config` keep keeps `accepted_ref` unchanged but updates accepted run command and metric

Example sketch:

```python
def test_run_config_discard_does_not_move_head(tmp_path):
    before = git_head(repo)
    apply_rejected_candidate(session, proposal={"candidate_kind": "run_config"}, assessment=assessment)
    assert git_head(repo) == before
```

- [ ] **Step 2: Run the failing candidate-kind tests**

Run:

```bash
pytest tests/studies/test_lines_256_session_controller.py -k "run_config or source_candidate" -v
```

Expected: FAIL.

- [ ] **Step 3: Implement scored-run execution and accept/reject logic**

Requirements:
- run scored command
- harvest metric, randomness contract, and comparison PNG
- append one row per attempt
- branch on `candidate_kind`
- update accepted-state JSON deterministically
- reset only for rejected `source` candidates

- [ ] **Step 4: Run the candidate-kind tests**

Run:

```bash
pytest tests/studies/test_lines_256_session_controller.py -k "run_config or source_candidate" -v
```

Expected: PASS.

- [ ] **Step 5: Commit the candidate-execution slice**

```bash
git add scripts/studies/lines_256_session_controller.py tests/studies/test_lines_256_session_controller.py
git commit -m "Add lines 256 v2 source and run-config execution"
```

## Task 5: Implement timeout, crash-debug, and resume checkpoints

**Files:**
- Modify: `scripts/studies/lines_256_session_controller.py`
- Modify: `tests/studies/test_lines_256_session_controller.py`

- [ ] **Step 1: Write failing tests for timeout/crash/resume**

Cover:
- timed-out candidate recorded as `TIMEOUT`
- timed-out `run_config` continues without git rollback
- crash triggers one debug attempt
- resume continues from the last persisted phase instead of replaying the whole session

Example sketch:

```python
def test_resume_continues_from_last_completed_phase(tmp_path):
    write_session_json(session_root, {"phase": "score", "iteration": 3})
    state = resume_session(session_root)
    assert state.phase == "score"
    assert state.iteration == 3
```

- [ ] **Step 2: Run the failing timeout/crash/resume tests**

Run:

```bash
pytest tests/studies/test_lines_256_session_controller.py -k "timeout or crash or resume" -v
```

Expected: FAIL.

- [ ] **Step 3: Implement phase checkpoints and debug attempt handling**

Requirements:
- explicit phase writes in `session.json`
- deterministic timeout classification
- single focused crash-debug attempt
- resume from the recorded phase

- [ ] **Step 4: Run the timeout/crash/resume tests**

Run:

```bash
pytest tests/studies/test_lines_256_session_controller.py -k "timeout or crash or resume" -v
```

Expected: PASS.

- [ ] **Step 5: Commit the resilience slice**

```bash
git add scripts/studies/lines_256_session_controller.py tests/studies/test_lines_256_session_controller.py
git commit -m "Add lines 256 v2 timeout crash and resume handling"
```

## Task 6: Add thin workflow wrapper and legacy-parallel docs

**Files:**
- Create: `workflows/agent_orchestration/lines_256_session_controller.yaml`
- Modify: `tests/studies/test_lines_256_arch_improvement_workflow.py`
- Create: `docs/studies/lines_256_controller_loop.md`
- Modify: `docs/studies/index.md`
- Modify: `docs/index.md`

- [ ] **Step 1: Write failing workflow/doc tests**

Add tests for:
- new thin wrapper existence and loadability
- wrapper delegates to the controller script instead of encoding the full loop
- docs index includes both legacy and v2 paths

- [ ] **Step 2: Run the failing wrapper/doc tests**

Run:

```bash
pytest tests/studies/test_lines_256_arch_improvement_workflow.py -k "session_controller or legacy_and_v2_docs" -v
```

Expected: FAIL.

- [ ] **Step 3: Add the thin wrapper and v2 docs**

Requirements:
- validate authoritative docs and controller script
- run the controller
- keep state/output roots separate from legacy
- document that legacy remains supported during validation rollout

- [ ] **Step 4: Run workflow/doc tests and dry-runs**

Run:

```bash
pytest tests/studies/test_lines_256_arch_improvement_workflow.py -k "session_controller or legacy_and_v2_docs" -v
PYTHONPATH=/home/ollie/Documents/agent-orchestration python -m orchestrator run workflows/agent_orchestration/lines_256_session_controller.yaml --dry-run --stream-output
```

Expected: all PASS.

- [ ] **Step 5: Commit the wrapper/docs slice**

```bash
git add workflows/agent_orchestration/lines_256_session_controller.yaml tests/studies/test_lines_256_arch_improvement_workflow.py docs/studies/lines_256_controller_loop.md docs/studies/index.md docs/index.md
git commit -m "Add lines 256 v2 controller wrapper and docs"
```

## Task 7: End-to-end validation against the legacy path

**Files:**
- Modify: `docs/studies/lines_256_controller_loop.md`
- Modify: `docs/plans/2026-03-25-lines256-controller-redesign.md` only if rollout notes need to be tightened

- [ ] **Step 1: Run a fresh v2 baseline-only session**

Run:

```bash
python scripts/studies/lines_256_session_controller.py start --mode baseline-only
```

Expected:
- fresh session root under `state/lines_256_arch_improvement_v2/sessions/...`
- baseline row written
- accepted-state JSON written

- [ ] **Step 2: Run one `run_config` candidate session slice**

Run:

```bash
python scripts/studies/lines_256_session_controller.py resume --session-root state/lines_256_arch_improvement_v2/sessions/<session_id> --max-iterations 1
```

Expected:
- one parameter-only candidate recorded
- no git reset performed for reject/timeout

- [ ] **Step 3: Run one `source` candidate session slice**

Run:

```bash
python scripts/studies/lines_256_session_controller.py resume --session-root state/lines_256_arch_improvement_v2/sessions/<session_id> --max-iterations 1
```

Expected:
- one source-changing candidate recorded
- correct reset on reject

- [ ] **Step 4: Run the full focused test suite and wrapper dry-runs**

Run:

```bash
pytest --collect-only tests/studies/test_lines_256_session_controller.py tests/studies/test_lines_256_arch_improvement_workflow.py -q
pytest tests/studies/test_lines_256_session_controller.py tests/studies/test_lines_256_arch_improvement_workflow.py -v
PYTHONPATH=/home/ollie/Documents/agent-orchestration python -m orchestrator run workflows/agent_orchestration/lines_256_session_controller.yaml --dry-run --stream-output
PYTHONPATH=/home/ollie/Documents/agent-orchestration python -m orchestrator run workflows/agent_orchestration/lines_256_arch_improvement_session_loop.yaml --dry-run --stream-output
```

Expected:
- collection succeeds
- tests pass
- both v2 and legacy wrappers validate

- [ ] **Step 5: Commit the validation/reporting updates**

```bash
git add tests/studies/test_lines_256_session_controller.py tests/studies/test_lines_256_arch_improvement_workflow.py docs/studies/lines_256_controller_loop.md
git commit -m "Validate lines 256 v2 controller against legacy path"
```

## Notes for Implementers

- Keep the legacy workflow intact during this plan. Do not delete or deprecate it prematurely.
- Keep v2 state and outputs completely separate from legacy roots.
- Prefer small controller helpers over one giant script body if the file starts becoming hard to hold in context.
- Do not reintroduce loop mechanics into prompts.
- Do not use git ancestry shortcuts like `HEAD^` in the new path; always use explicit recorded refs.
- The v2 controller should be operable directly from the PtychoPINN repo root even if the thin workflow wrapper does not yet exist.

## Final Verification Checklist

- [ ] V2 baseline works from a fresh session root
- [ ] V2 `run_config` candidate works end-to-end
- [ ] V2 `source` candidate works end-to-end
- [ ] Resume works from an interrupted phase
- [ ] V2 state does not collide with legacy state
- [ ] Legacy workflow still validates
- [ ] New thin wrapper validates
