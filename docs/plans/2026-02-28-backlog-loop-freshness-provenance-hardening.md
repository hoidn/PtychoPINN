# Backlog Loop Freshness & Provenance Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent stale artifact reuse across review/fix cycles by making handoff artifacts run/cycle scoped and enforcing consume freshness, while keeping semantic completion decisions in LLM review.

**Architecture:** Keep DSL and executor semantics unchanged. Harden workflow reliability by using pre-steps that deterministically prepare per-cycle output paths, rewiring producer/consumer edges, and splitting artifact aliases so `since_last_consume` can be enforced without deadlocking shared artifact names. Keep prompts thin by consuming prepared destination artifacts instead of embedding path-derivation logic in prompt text.

**Tech Stack:** Workflow DSL YAML (`v1.2`), prompt markdown, bash/python command steps, `agent-orchestration` CLI (`python -m orchestrator.cli.main`).

---

## Scope and Non-Goals

In scope:
- Run-scoped and cycle-scoped artifact paths for execution/check/review outputs.
- Prompt contracts that reference consumed destination artifacts, not hardcoded `latest-*` files.
- `since_last_consume` on loop-critical consumes.
- Runbook updates for new artifact layout.

Out of scope:
- Deterministic semantic “plan complete” gate.
- Executor or DSL schema changes.
- New reviewer decision criteria.

## Pathing Contract (Target State)

- Execute session log path:
  - `artifacts/work/runs/${run.id}/c<cycle>-execute-session.md`
- Fix session log path:
  - `artifacts/work/runs/${run.id}/c<cycle>-fix-session.md`
- Targeted checks log path:
  - `artifacts/checks/runs/${run.id}/c<cycle>-checks.log`
- Review report path:
  - `artifacts/review/runs/${run.id}/c<cycle>-review.md`

All pointers remain in `state/*.txt`; only pointer values become run/cycle scoped.

---

### Task 1: Add Deterministic Pre-Steps for Cycle-Scoped Output Paths

**Files:**
- Modify: `workflows/agent_orchestration/backlog_plan_slice_impl_review_loop.yaml`

**Step 1: Add prepare step before ExecutePlan**

Add command step `PrepareExecuteSessionLogPath` that:
- reads `state/review_cycle.txt`
- writes deterministic path into `state/execution_session_log_path.txt`
- removes any pre-existing target file

Skeleton:

```yaml
- name: PrepareExecuteSessionLogPath
  command:
    - bash
    - -lc
    - |
      mkdir -p state artifacts/work/runs/${run.id}
      cycle="$(cat state/review_cycle.txt)"
      target="artifacts/work/runs/${run.id}/c${cycle}-execute-session.md"
      rm -f "$target"
      printf '%s\n' "$target" > state/execution_session_log_path.txt
  expected_outputs:
    - name: execution_session_log_path
      path: state/execution_session_log_path.txt
      type: relpath
      under: artifacts/work
```

**Step 2: Add prepare step before RunTargetedChecks**

Add `PrepareCheckLogPath` that writes `state/check_log_path.txt` with `c<cycle>-checks.log` under `artifacts/checks/runs/${run.id}/`.

**Step 3: Add prepare step before ReviewImplVsPlan**

Add `PrepareCodeReviewPath` that writes `state/code_review_path.txt` with `c<cycle>-review.md` under `artifacts/review/runs/${run.id}/`.

**Step 4: Add prepare step before FixIssues**

Add `PrepareFixSessionLogPath` for fix-phase session logs using cycle-stamped filename `c<cycle>-fix-session.md` (or `c<cycle+1>` if cycle increments later; pick one convention and apply consistently in docs and flow).

**Step 5: Rewire `goto`/step ordering**

Rewire flow:
- `InitializeReviewCycle -> PrepareExecuteSessionLogPath -> ExecutePlan`
- `ExecutePlan -> PrepareCheckLogPath -> RunTargetedChecks`
- `RunTargetedChecks -> PrepareCodeReviewPath -> ReviewImplVsPlan`
- `ReviewCycleGate(success) -> PrepareFixSessionLogPath -> FixIssues`
- `IncrementReviewCycle -> PrepareCheckLogPath -> RunTargetedChecks`

**Step 6: Commit**

```bash
git add workflows/agent_orchestration/backlog_plan_slice_impl_review_loop.yaml
git commit -m "workflow: add cycle-scoped pre-steps for log/review paths"
```

---

### Task 2: Keep Prompts Thin (Use Canonical State Pointer Files)

**Files:**
- Modify: `prompts/workflows/backlog_plan_loop/execute_plan.md`
- Modify: `prompts/workflows/backlog_plan_loop/fix_issues.md`
- Modify: `prompts/workflows/backlog_plan_loop/review_impl_vs_plan.md`

**Step 1: Update ExecutePlan prompt contract**

Replace hardcoded artifact write target with:
- “Read destination path from `state/execution_session_log_path.txt` and write the session log there.”

**Step 2: Update FixIssues prompt contract**

Use `state/execution_session_log_path.txt` as destination path source.

**Step 3: Update ReviewImplVsPlan prompt contract**

Replace `artifacts/review/latest-review.md` with:
- “Read destination path from `state/code_review_path.txt` and write the review there.”

**Step 4: Add anti-footgun guidance**

For all three prompts:
- “Do not write to `latest-*` fallback files.”
- “Do not derive destination paths yourself; use the state pointer file value exactly.”

**Step 5: Commit**

```bash
git add prompts/workflows/backlog_plan_loop/execute_plan.md \
        prompts/workflows/backlog_plan_loop/fix_issues.md \
        prompts/workflows/backlog_plan_loop/review_impl_vs_plan.md
git commit -m "prompts: use state pointer files for output paths"
```

---

### Task 3: Freshness Tuning Without Artifact Alias Proliferation

**Files:**
- Modify: `workflows/agent_orchestration/backlog_plan_slice_impl_review_loop.yaml`

**Step 1: Keep only canonical artifact names**

Use only:
- `execution_session_log`
- `check_log`
- `code_review`

**Step 2: Publish canonical artifacts from producer steps**

Update `publishes:`:
- `ExecutePlan` and `FixIssues` publish `execution_session_log`.
- `RunTargetedChecks` publishes `check_log`.
- `ReviewImplVsPlan` publishes `code_review`.

**Step 3: Apply `since_last_consume` selectively**

Update consumes:
- `ReviewImplVsPlan`:
  - `execution_session_log`: `freshness: since_last_consume`
  - `check_log`: `freshness: since_last_consume`
- `FixIssues`:
  - `execution_session_log`: `freshness: any` (avoid deadlock after review consumes)
  - `check_log`: `freshness: any` (avoid deadlock after review consumes)
  - `code_review`: `freshness: since_last_consume`
- `plan` remains `freshness: any`.

**Step 4: Commit**

```bash
git add workflows/agent_orchestration/backlog_plan_slice_impl_review_loop.yaml
git commit -m "workflow: simplify artifacts and tune selective consume freshness"
```

---

### Task 4: Update RunTargetedChecks to Respect Prepared Check Path

**Files:**
- Modify: `workflows/agent_orchestration/backlog_plan_slice_impl_review_loop.yaml`

**Step 1: Read output path from prepared pointer**

Inside `RunTargetedChecks` inline Python:
- read `state/check_log_path.txt`
- resolve and write log to that path
- do not hardcode `artifacts/checks/latest-checks.log`

Pseudo-snippet:

```python
check_log_rel = (state_dir / "check_log_path.txt").read_text(encoding="utf-8").strip()
log_path = Path(check_log_rel)
log_path.parent.mkdir(parents=True, exist_ok=True)
```

**Step 2: Stop rewriting pointer with `latest`**

Remove line that overwrites `state/check_log_path.txt` to `latest-checks.log`.

**Step 3: Commit**

```bash
git add workflows/agent_orchestration/backlog_plan_slice_impl_review_loop.yaml
git commit -m "workflow: run targeted checks writes to prepared check_log_path"
```

---

### Task 5: Runbook and Operator-Facing Docs Update

**Files:**
- Modify: `docs/workflows/agent_orchestration_backlog_loop.md`

**Step 1: Update “What the Workflow Does”**

Document pre-step path preparation and cycle-scoped artifact outputs.

**Step 2: Update “Monitoring” + state/output section**

Replace references implying only `latest-*` artifacts. Document:
- run/cycle-scoped paths under `artifacts/*/runs/<run_id>/`
- pointers in `state/*.txt` as current selected artifacts

**Step 3: Commit**

```bash
git add docs/workflows/agent_orchestration_backlog_loop.md
git commit -m "docs: document cycle-scoped artifact paths and pointer-driven handoff"
```

---

### Task 6: Verification (Dry-Run + Smoke Loop)

**Files:**
- Modify if needed: `workflows/agent_orchestration/backlog_plan_slice_impl_review_loop.yaml`
- Verify prompts/docs from prior tasks

**Step 1: Schema/loader dry-run**

Run:

```bash
cd ~/Documents/tmp/PtychoPINN
PYTHONPATH=~/Documents/agent-orchestration \
python -m orchestrator.cli.main run \
  workflows/agent_orchestration/backlog_plan_slice_impl_review_loop.yaml \
  --dry-run
```

Expected:
- workflow validates
- no contract/loader errors

**Step 2: Static grep checks for hardcoded latest paths**

Run:

```bash
rg -n "latest-execution-session-log|latest-checks|latest-review" \
  workflows/agent_orchestration/backlog_plan_slice_impl_review_loop.yaml \
  prompts/workflows/backlog_plan_loop \
  docs/workflows/agent_orchestration_backlog_loop.md
```

Expected:
- no hardcoded latest paths in workflow/prompt contracts
- any remaining mentions are explanatory legacy notes only (if kept)

**Step 3: One live workflow smoke run (manual)**

Run with normal launch path (tmux recommended) and observe one review/fix cycle.

Acceptance checks:
- `state/execution_session_log_path.txt`, `state/check_log_path.txt`, `state/code_review_path.txt` point to run/cycle-scoped files.
- Files exist at those targets.
- After one loop turn, consumed artifacts are new versions (no stale consume error unless producer truly failed to produce new artifact).

**Step 4: Final commit**

```bash
git add workflows/agent_orchestration/backlog_plan_slice_impl_review_loop.yaml \
        prompts/workflows/backlog_plan_loop/*.md \
        docs/workflows/agent_orchestration_backlog_loop.md
git commit -m "workflow: harden run/cycle artifact provenance and consume freshness"
```

---

## Verification Commands

```bash
cd ~/Documents/tmp/PtychoPINN
PYTHONPATH=~/Documents/agent-orchestration \
python -m orchestrator.cli.main run \
  workflows/agent_orchestration/backlog_plan_slice_impl_review_loop.yaml \
  --dry-run

rg -n "latest-execution-session-log|latest-checks|latest-review" \
  workflows/agent_orchestration/backlog_plan_slice_impl_review_loop.yaml \
  prompts/workflows/backlog_plan_loop \
  docs/workflows/agent_orchestration_backlog_loop.md
```

## Completion Criteria

- [ ] Workflow uses pre-steps to prepare run/cycle-scoped output paths for execute/check/review/fix artifacts.
- [ ] `ExecutePlan`, `FixIssues`, and `ReviewImplVsPlan` prompts write to pointer-prepared targets (no hardcoded `latest-*` contract paths).
- [ ] `ReviewImplVsPlan` uses `freshness: since_last_consume` for execution/check artifacts; `FixIssues` uses selective freshness to avoid consume deadlocks.
- [ ] Workflow keeps canonical artifact names (`execution_session_log`, `check_log`, `code_review`) without alias proliferation.
- [ ] Dry-run validation passes and one live smoke cycle confirms pointer/path updates per cycle.
- [ ] Runbook reflects new provenance contract and monitoring locations.
