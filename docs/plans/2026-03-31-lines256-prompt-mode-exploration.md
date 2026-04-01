# Lines 256 Prompt-Mode Exploration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the brittle controller-enforced exploration gate with prompt-driven `explore` / `exploit` proposal modes for the `lines_256` v2 controller.

**Architecture:** Keep deterministic session mechanics in `lines_256_session_controller.py`, but limit the controller to selecting a proposal mode and surfacing broader search context. Move exploratory-vs-exploitative judgment back into specialized prompt files so the agent decides how to vary hypotheses without controller-side family policing.

**Tech Stack:** Python 3.11, pytest, agent-orchestration prompt injection, JSON session artifacts

---

### Task 1: Lock the behavior change down in tests

**Files:**
- Modify: `tests/studies/test_lines_256_session_controller.py`

- [ ] **Step 1: Replace brittle hard-gate expectations with soft-mode expectations**

Add/adjust tests so they verify:
- `build_search_summary()` records useful soft exploration signals such as `recent_discard_streak`, `family_counts`, and `preferred_exploration_families`
- `build_proposal_context()` records `proposal_mode` and `proposal_mode_reason`
- `validate_ready_proposal()` does **not** reject proposals based on hypothesis-family inference alone

- [ ] **Step 2: Run the narrow test slice to verify RED**

Run:
```bash
cd /home/ollie/Documents/tmp/PtychoPINN && pytest tests/studies/test_lines_256_session_controller.py -k "proposal_mode or search_summary or validate_ready_proposal" -v
```

Expected:
- failures because the controller still exposes or enforces the old hard-gate behavior

### Task 2: Implement prompt-mode selection in the controller

**Files:**
- Modify: `scripts/studies/lines_256_session_controller.py`

- [ ] **Step 1: Keep search-summary context, remove hard validation**

Update the controller so:
- `build_search_summary()` can keep soft family/context summaries
- `build_proposal_context()` emits:
  - `proposal_mode`
  - `proposal_mode_reason`
- `validate_ready_proposal()` no longer rejects a proposal for being from the “wrong” hypothesis family

- [ ] **Step 2: Select proposal prompt by mode**

Change proposal-step prompt rendering so it uses:
- `experiment_step_common.md` + `experiment_step_exploit.md` when `proposal_mode == "exploit"`
- `experiment_step_common.md` + `experiment_step_explore.md` when `proposal_mode == "explore"`

Keep debug/crash prompt behavior unchanged.

- [ ] **Step 3: Run the narrow test slice to verify GREEN**

Run:
```bash
cd /home/ollie/Documents/tmp/PtychoPINN && pytest tests/studies/test_lines_256_session_controller.py -k "proposal_mode or search_summary or validate_ready_proposal" -v
```

Expected:
- targeted tests pass

### Task 3: Split the proposal prompt into common and mode-specific overlays

**Files:**
- Create: `prompts/workflows/lines_256_arch_improvement/experiment_step_common.md`
- Create: `prompts/workflows/lines_256_arch_improvement/experiment_step_exploit.md`
- Create: `prompts/workflows/lines_256_arch_improvement/experiment_step_explore.md`
- Modify: `prompts/workflows/lines_256_arch_improvement/experiment_step.md`

- [ ] **Step 1: Move shared contract text into a common prompt body**

Shared prompt should keep:
- task contract
- candidate-kind rules
- smoke requirements
- output metadata rules
- simplicity/compute tradeoff guidance

- [ ] **Step 2: Add specialized mode overlays**

`experiment_step_exploit.md` should:
- optimize near the accepted champion
- avoid obvious repeats
- prefer high-confidence local improvements or simplifications

`experiment_step_explore.md` should:
- prefer a different hypothesis class from recent saturated local attempts
- explicitly invite broader but coherent ideas
- still respect the fixed dataset and `20`-epoch scored contract

- [ ] **Step 3: Keep compatibility**

Retain `experiment_step.md` as a compatibility wrapper or pointer so older paths are not left with a missing file while the v2 controller migrates.

### Task 4: Update docs and live-checkout compatibility

**Files:**
- Modify: `docs/studies/lines_256_controller_loop.md`
- Modify: `docs/plans/2026-03-31-lines256-exploration-steering.md`
- Optionally modify live run-checkout mirrors under:
  - `.../prompts/workflows/lines_256_arch_improvement/`
  - `.../scripts/studies/lines_256_session_controller.py`

- [ ] **Step 1: Document the prompt-mode split**

Document that:
- controller chooses `proposal_mode`
- prompts own exploratory vs exploitative judgment
- controller does not hard-reject proposals by hypothesis-family heuristics

- [ ] **Step 2: Mirror the minimal changes into the active run checkout**

If the current live session may be resumed, mirror the prompt/controller changes needed for future resumes and proposal steps.

### Task 5: Verify end to end

**Files:**
- Test: `tests/studies/test_lines_256_session_controller.py`

- [ ] **Step 1: Collect the touched test module**

Run:
```bash
cd /home/ollie/Documents/tmp/PtychoPINN && pytest --collect-only tests/studies/test_lines_256_session_controller.py -q
```

- [ ] **Step 2: Run the full relevant unit module**

Run:
```bash
cd /home/ollie/Documents/tmp/PtychoPINN && pytest tests/studies/test_lines_256_session_controller.py -v
```

- [ ] **Step 3: Validate the orchestrator wrapper**

Run:
```bash
cd /home/ollie/Documents/tmp/PtychoPINN && PYTHONPATH=/home/ollie/Documents/agent-orchestration python -m orchestrator run workflows/agent_orchestration/lines_256_session_controller.yaml --dry-run --stream-output
```

- [ ] **Step 4: Check prompt/controller diffs for whitespace or syntax problems**

Run:
```bash
cd /home/ollie/Documents/tmp/PtychoPINN && git diff --check -- scripts/studies/lines_256_session_controller.py tests/studies/test_lines_256_session_controller.py prompts/workflows/lines_256_arch_improvement docs/studies/lines_256_controller_loop.md docs/plans/2026-03-31-lines256-prompt-mode-exploration.md
```
