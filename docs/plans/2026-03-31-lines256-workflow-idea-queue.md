# Lines 256 Workflow Idea Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated workflow-idea queue for `lines_256` so the v2 controller prioritizes human-authored experiment ideas from `docs/workflow_queue/active/` before free-form exploration, and moves each idea file to an outcome-specific folder only after a terminal result.

**Architecture:** Keep the queue deterministic and controller-owned. The controller discovers queue items in lexicographic order, injects the selected idea into proposal context and prompt inputs, and moves the source markdown file from `active/` to an outcome-specific folder only after `KEEP`, `DISCARD`, `TIMEOUT`, `CRASH`, or `BLOCKED`. Prompts use the queued idea as proposal guidance but do not own queue mutation or routing.

**Tech Stack:** Python 3.11, pytest, pathlib/shutil filesystem operations, agent-orchestration prompt injection, Markdown queue items

---

### Task 1: Define the queue filesystem contract in tests

**Files:**
- Modify: `tests/studies/test_lines_256_session_controller.py`

- [ ] **Step 1: Write failing tests for queue discovery and proposal-context injection**

Add tests covering:
- queue roots under `docs/workflow_queue/{active,accepted,discarded,blocked,crashed,timed_out}`
- lexicographic selection of the first `active/*.md` item
- `proposal_context.json` including queue metadata such as selected idea path and text
- free-form mode when no queued ideas are active

- [ ] **Step 2: Write failing tests for queue file movement on terminal outcomes**

Add tests covering:
- queued idea stays in `active/` while the iteration is only proposed or scored
- queued idea moves after terminal outcome to the correct destination folder:
  - `KEEP -> accepted/`
  - `DISCARD -> discarded/`
  - `BLOCKED -> blocked/`
  - `CRASH -> crashed/`
  - `TIMEOUT -> timed_out/`

- [ ] **Step 3: Run the narrow test slice to verify RED**

Run:
```bash
cd /home/ollie/Documents/tmp/PtychoPINN && pytest tests/studies/test_lines_256_session_controller.py -k "workflow_queue or queued_idea or proposal_context" -v
```

Expected:
- failures because queue discovery, prompt injection, and move-on-terminal-outcome behavior do not exist yet

### Task 2: Implement deterministic queue ownership in the controller

**Files:**
- Modify: `scripts/studies/lines_256_session_controller.py`

- [ ] **Step 1: Add queue path helpers and active-item discovery**

Implement helpers for:
- queue root paths under `docs/workflow_queue/`
- finding the first active idea file in lexicographic order
- reading queued idea text as plain Markdown guidance

- [ ] **Step 2: Add queue metadata to proposal context**

Extend `build_proposal_context()` to include:
- whether a queued workflow idea is active
- selected queue item path
- selected queue item content
- `proposal_mode_reason` updated to explain when queue priority is in effect

Keep queue choice deterministic and controller-owned.

- [ ] **Step 3: Move queue items only after terminal outcomes**

Update the controller’s terminal-outcome path so:
- queued file remains in `active/` until a terminal result is persisted
- then moves to the outcome-specific folder
- the move happens once per terminal outcome and is idempotent on resume

- [ ] **Step 4: Run the narrow test slice to verify GREEN**

Run:
```bash
cd /home/ollie/Documents/tmp/PtychoPINN && pytest tests/studies/test_lines_256_session_controller.py -k "workflow_queue or queued_idea or proposal_context" -v
```

Expected:
- targeted queue tests pass

### Task 3: Teach proposal prompts how to use queued ideas without owning the queue

**Files:**
- Modify: `prompts/workflows/lines_256_arch_improvement/experiment_step_common.md`
- Optionally modify: `prompts/workflows/lines_256_arch_improvement/experiment_step_exploit.md`
- Optionally modify: `prompts/workflows/lines_256_arch_improvement/experiment_step_explore.md`

- [ ] **Step 1: Add queue guidance to the common prompt**

Prompt should state:
- if `proposal_context.json` includes an active queued workflow idea, treat it as the highest-priority candidate direction for this step
- interpret the idea faithfully but still apply coherence, smoke, and study-contract standards
- if the queued idea is not viable, return `BLOCKED` or a clean failed attempt rather than silently replacing it with some other idea

- [ ] **Step 2: Keep prompt boundaries clean**

Do not put file-moving or queue bookkeeping in prompt text.
Queue consumption remains controller-owned.

### Task 4: Document the workflow-idea queue

**Files:**
- Modify: `docs/studies/lines_256_controller_loop.md`
- Create: `docs/workflow_queue/README.md`
- Create: `docs/workflow_queue/templates/workflow_idea.md`

- [ ] **Step 1: Document queue semantics in the v2 study doc**

Document:
- active queue items override free-form proposal selection
- lexicographic order
- move after terminal outcome only
- outcome folder mapping

- [ ] **Step 2: Add a queue README and lightweight idea template**

The template should stay Markdown-light and human-authored, not schema-heavy.

### Task 5: Mirror the behavior into the active run checkout if needed

**Files:**
- Modify run-checkout mirrors only if the live session will be resumed with queue support:
  - `.../scripts/studies/lines_256_session_controller.py`
  - `.../prompts/workflows/lines_256_arch_improvement/experiment_step_common.md`
  - optional related prompt files

- [ ] **Step 1: Mirror minimal runtime files**

Only after source-of-truth verification passes, copy the needed files into the active run checkout so resume can see the queue behavior.

- [ ] **Step 2: Reconcile protected local paths if needed**

If mirroring changes tracked dirt in the run checkout, update the session-local protected-path snapshot before resume.

### Task 6: Verify end to end

**Files:**
- Test: `tests/studies/test_lines_256_session_controller.py`

- [ ] **Step 1: Collect the touched test module**

Run:
```bash
cd /home/ollie/Documents/tmp/PtychoPINN && pytest --collect-only tests/studies/test_lines_256_session_controller.py -q
```

- [ ] **Step 2: Run the full relevant module**

Run:
```bash
cd /home/ollie/Documents/tmp/PtychoPINN && pytest tests/studies/test_lines_256_session_controller.py -v
```

- [ ] **Step 3: Validate the controller workflow wrapper**

Run:
```bash
cd /home/ollie/Documents/tmp/PtychoPINN && PYTHONPATH=/home/ollie/Documents/agent-orchestration python -m orchestrator run workflows/agent_orchestration/lines_256_session_controller.yaml --dry-run --stream-output
```

- [ ] **Step 4: Check whitespace and patch cleanliness**

Run:
```bash
cd /home/ollie/Documents/tmp/PtychoPINN && git diff --check -- scripts/studies/lines_256_session_controller.py tests/studies/test_lines_256_session_controller.py prompts/workflows/lines_256_arch_improvement docs/studies/lines_256_controller_loop.md docs/workflow_queue docs/plans/2026-03-31-lines256-workflow-idea-queue.md
```
