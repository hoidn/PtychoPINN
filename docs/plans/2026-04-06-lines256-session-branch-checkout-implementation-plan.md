# Lines256 Session Branch Checkout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace detached-`HEAD` runtime behavior in the `lines_256` controller with a session-local branch model while preserving SHA-based provenance, protected local changes, and deterministic resume/cleanup behavior.

**Architecture:** Add a controller-owned `session_branch` field to session state and make session start/resume explicitly manage that branch inside the dedicated run checkout. Source proposals will still be resolved to exact commit SHAs for scientific provenance, but `KEEP`/`DISCARD`/resume semantics will now operate on a named session branch that always points at the current accepted state between candidates. Detached-but-recoverable checkouts become reattachable instead of being the normal runtime mode.

**Tech Stack:** Python 3.11, Git CLI (`checkout`, `switch`, `branch`, `rev-parse`, `symbolic-ref`), pytest, JSON session artifacts, orchestrator dry-run validation

---

## File Map

- Modify: `scripts/studies/lines_256_session_controller.py`
  - Persist `session_branch` in session state and `session.json`
  - Add branch helpers for create/attach/validate/reset
  - Rework start/resume/cleanup paths to operate on a named session branch
  - Keep `resolved_candidate_commit` as the authoritative source-candidate identity
- Modify: `tests/studies/test_lines_256_session_controller.py`
  - Add coverage for session-branch creation, discard reset, resume reattachment, and protected-local-path survival
- Modify: `docs/studies/lines_256_controller_loop.md`
  - Document the named session-branch runtime contract
- Modify: `docs/findings.md`
  - Record the rule that branch names are ergonomic handles and SHAs remain authoritative
- Create: `docs/plans/2026-04-06-lines256-session-branch-checkout-implementation-plan.md`
  - This implementation plan

## Design Boundaries

- The dedicated run checkout remains required.
- The branch name is not scientific provenance.
- `accepted_ref` and resolved candidate SHAs remain the authoritative session lineage.
- Provider-reported `candidate_commit` remains advisory; resolved `git rev-parse HEAD` wins.
- Phase 1 does not add persistent hidden refs for discarded candidates.

---

### Task 1: Lock the New Session-Branch Contract With Failing Tests

**Files:**
- Modify: `tests/studies/test_lines_256_session_controller.py`

- [ ] **Step 1: Add a failing test for session initialization on a named branch**

Add a focused test that initializes a session in a temporary git repo and asserts:

```python
def test_initialize_session_persists_session_branch(tmp_path):
    session = initialize_session(repo_root, session_id="20260406T000000Z")
    payload = json.loads((session.session_root / "session.json").read_text())
    assert payload["session_branch"] == "lines256/session/20260406T000000Z"
```

and a branch-aware runtime check such as:

```python
def test_start_attaches_checkout_to_session_branch(tmp_path):
    session = initialize_session(repo_root, session_id="20260406T000000Z")
    _ensure_session_branch(repo_root, session, base_ref)
    assert _git_current_branch(repo_root) == "lines256/session/20260406T000000Z"
```

- [ ] **Step 2: Add a failing test for discard reset on the session branch**

Add a test that:

1. creates an accepted base commit
2. creates a source-candidate commit on the session branch
3. calls the branch-aware cleanup/reset path
4. asserts:

```python
assert _git_current_branch(repo_root) == session.session_branch
assert _git_head(repo_root) == accepted_ref
```

- [ ] **Step 3: Add a failing resume/reattach test**

Add a test that leaves the repo detached at the accepted SHA and asserts resume-time validation can reattach it:

```python
def test_resume_reattaches_detached_checkout_to_session_branch(tmp_path):
    ...
    _ensure_session_checkout_state(repo_root, session, expected_ref=accepted_ref)
    assert _git_current_branch(repo_root) == session.session_branch
```

- [ ] **Step 4: Add a failing protected-local-path regression**

Add a test proving branch-based reset still preserves tracked protected dirt:

```python
def test_branch_reset_preserves_protected_local_paths(tmp_path):
    ...
    assert protected_path.read_text() == "user local change"
```

- [ ] **Step 5: Run collect-only on the controller test module**

Run: `pytest --collect-only tests/studies/test_lines_256_session_controller.py -q`

Expected: the new tests collect cleanly.

- [ ] **Step 6: Run the narrow failing selectors**

Run:
- `pytest tests/studies/test_lines_256_session_controller.py -k session_branch -v`
- `pytest tests/studies/test_lines_256_session_controller.py -k reattach -v`
- `pytest tests/studies/test_lines_256_session_controller.py -k protected_local_paths -v`

Expected: FAIL for the expected missing-branch-behavior reasons.

- [ ] **Step 7: Commit the red tests**

```bash
git add tests/studies/test_lines_256_session_controller.py
git commit -m "test: lock lines256 session branch checkout contract"
```

---

### Task 2: Persist Session Branch Metadata and Add Git Helpers

**Files:**
- Modify: `scripts/studies/lines_256_session_controller.py`
- Test: `tests/studies/test_lines_256_session_controller.py`

- [ ] **Step 1: Extend `SessionState` and session JSON persistence**

Add `session_branch: str` to the session state model and include it in:

- `initialize_session(...)`
- `_write_session_json(...)`
- `load_session(...)`

Use the deterministic naming rule:

```python
session_branch = f"lines256/session/{session_id}"
```

- [ ] **Step 2: Add focused Git helpers**

Implement small helpers in `scripts/studies/lines_256_session_controller.py`:

```python
def _git_current_branch(repo_root: Path) -> str | None: ...
def _git_branch_exists(repo_root: Path, branch: str) -> bool: ...
def _checkout_branch_at_ref(repo_root: Path, branch: str, ref: str) -> None: ...
def _attach_or_reset_session_branch(repo_root: Path, branch: str, ref: str) -> None: ...
```

Behavior requirements:

- if the branch does not exist, create it at `ref`
- if it exists, force-reset it to `ref` only in controller-owned reset paths
- use non-interactive Git commands only

- [ ] **Step 3: Add a compact branch-state validator**

Implement a helper such as:

```python
def _ensure_session_checkout_state(repo_root: Path, session: SessionState, expected_ref: str) -> None:
    ...
```

It should:

- verify current branch == `session.session_branch`, or reattach if detached and recoverable
- verify `HEAD == expected_ref` after attachment
- raise deterministically if the checkout is on the wrong named branch or at the wrong commit in a non-recoverable way

- [ ] **Step 4: Run the targeted session-branch selectors**

Run:
- `pytest tests/studies/test_lines_256_session_controller.py -k session_branch -v`
- `pytest tests/studies/test_lines_256_session_controller.py -k reattach -v`

Expected: PASS.

- [ ] **Step 5: Commit the metadata/helper changes**

```bash
git add scripts/studies/lines_256_session_controller.py tests/studies/test_lines_256_session_controller.py
git commit -m "feat: add lines256 session branch metadata and git helpers"
```

---

### Task 3: Move Start, Proposal Resolution, and Resume to the Session Branch Model

**Files:**
- Modify: `scripts/studies/lines_256_session_controller.py`
- Test: `tests/studies/test_lines_256_session_controller.py`

- [ ] **Step 1: Attach the session branch at session start**

Update startup flow so that when baseline/accepted state exists, the controller ensures the run checkout is attached to `session.session_branch` at the correct ref before proposal/scoring work begins.

Touch the actual control points:

- `initialize_session(...)`
- `run_full_session(...)`
- `main(...)` `start` path

- [ ] **Step 2: Make resume branch-aware**

Update the `resume` path so it:

- loads `session_branch` from `session.json`
- reattaches detached-but-recoverable checkouts to the session branch
- validates branch/ref before continuing `proposal_running`, `proposal_complete`, `scored_running`, or `debug_*` phases

Touch:

- `load_session(...)`
- `resume_session(...)` if needed
- `main(...)` `resume` path
- any helper invoked before `execute_iteration(...)`

- [ ] **Step 3: Keep resolved SHAs authoritative for source proposals**

In `resolve_ready_proposal(...)`, keep the current logic that resolves candidate identity from:

```python
resolved_candidate_commit = _git_head(repo_root)
```

but ensure it now runs while the repo is on `session.session_branch`, not detached `HEAD`.

Do not replace SHA provenance with branch-name provenance.

- [ ] **Step 4: Preserve provider mismatch diagnostics**

Keep the advisory mismatch signal:

- provider `candidate_commit`
- resolved `candidate_commit`

This remains useful for proposal hallucination/debugging even after the branch shift.

- [ ] **Step 5: Run focused proposal/resume selectors**

Run:
- `pytest tests/studies/test_lines_256_session_controller.py -k "resolve_ready_proposal or resume" -v`

Expected: PASS.

- [ ] **Step 6: Commit the start/resume integration**

```bash
git add scripts/studies/lines_256_session_controller.py tests/studies/test_lines_256_session_controller.py
git commit -m "fix: run lines256 proposals and resumes on session branches"
```

---

### Task 4: Make Candidate Cleanup and Assessment Reset the Session Branch Correctly

**Files:**
- Modify: `scripts/studies/lines_256_session_controller.py`
- Test: `tests/studies/test_lines_256_session_controller.py`

- [ ] **Step 1: Update source-candidate cleanup to preserve branch attachment**

Rework `_cleanup_source_candidate(...)` so it:

- still restores candidate-scoped paths from `base_ref`
- still removes candidate-created paths when absent in `base_ref`
- ends with the checkout attached to `session.session_branch`
- leaves `HEAD` at `base_ref`

Do not let cleanup leave the repo detached.

- [ ] **Step 2: Ensure `KEEP` semantics are branch-stable**

After a `KEEP`, the session branch should already point at the new accepted commit.

Verify that `apply_candidate_assessment(...)` and subsequent accepted-state writes do not inadvertently reset away the kept commit.

- [ ] **Step 3: Ensure `DISCARD` / `TIMEOUT` / `CRASH` reset back to accepted state**

For non-keeping source outcomes, the controller should:

- perform candidate-scoped cleanup
- ensure the branch tip and `HEAD` are back at the accepted SHA
- leave protected tracked local changes intact

- [ ] **Step 4: Run cleanup-focused selectors**

Run:
- `pytest tests/studies/test_lines_256_session_controller.py -k "discard or cleanup or protected_local_paths" -v`

Expected: PASS.

- [ ] **Step 5: Commit the cleanup/reset changes**

```bash
git add scripts/studies/lines_256_session_controller.py tests/studies/test_lines_256_session_controller.py
git commit -m "fix: reset lines256 source candidates on session branch"
```

---

### Task 5: Update Docs and Verify the Workflow Contract

**Files:**
- Modify: `docs/studies/lines_256_controller_loop.md`
- Modify: `docs/findings.md`

- [ ] **Step 1: Update the controller loop doc**

Document:

- the dedicated run checkout is still required
- the checkout now stays on a named session branch
- branch names are ergonomic handles only
- exact SHAs remain the authoritative provenance for accepted state and source candidates
- detached checkouts are recoverable but no longer the intended steady state

- [ ] **Step 2: Add a concise finding entry**

Add a finding that captures the rule:

- use session-local branches for operator-facing checkout state
- use commit SHAs for scientific provenance

- [ ] **Step 3: Run a full controller test pass**

Run: `pytest tests/studies/test_lines_256_session_controller.py -q`

Expected: PASS.

- [ ] **Step 4: Run syntax and workflow validation**

Run:
- `python -m py_compile scripts/studies/lines_256_session_controller.py`
- `PYTHONPATH=/home/ollie/Documents/agent-orchestration python -m orchestrator run workflows/agent_orchestration/lines_256_session_controller.yaml --dry-run --stream-output`

Expected:

- `py_compile` succeeds
- orchestrator validation succeeds

- [ ] **Step 5: Run a narrow formatting check**

Run:

```bash
git diff --check -- \
  scripts/studies/lines_256_session_controller.py \
  tests/studies/test_lines_256_session_controller.py \
  docs/studies/lines_256_controller_loop.md \
  docs/findings.md
```

Expected: clean.

- [ ] **Step 6: Commit docs and final verification state**

```bash
git add scripts/studies/lines_256_session_controller.py \
        tests/studies/test_lines_256_session_controller.py \
        docs/studies/lines_256_controller_loop.md \
        docs/findings.md \
        docs/plans/2026-04-06-lines256-session-branch-checkout-implementation-plan.md
git commit -m "feat: run lines256 sessions on named checkout branches"
```
