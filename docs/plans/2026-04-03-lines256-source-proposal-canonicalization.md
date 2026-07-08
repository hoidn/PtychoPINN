# Lines256 Source Proposal Canonicalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `lines_256` source-candidate proposal handoff robust when the provider writes a wrong or hallucinated `candidate_commit`, by moving source-commit authority to the controller and making proposal-state failures recoverable instead of session-fatal.

**Architecture:** Keep provider outputs task-local and judgment-heavy: hypothesis, candidate kind, candidate paths, smoke/run commands. Add a controller-owned proposal-resolution step that canonicalizes provider metadata into a deterministic `proposal_resolution.json`, resolves the actual source candidate commit from repo state, and feeds downstream validation/execution from that controller-owned artifact. Provider `candidate_commit` becomes optional/advisory, and mismatches become warnings unless the controller cannot prove a valid candidate state.

**Tech Stack:** Python 3.11, git subprocess helpers in `lines_256_session_controller.py`, pytest, orchestrator dry-run validation

---

## File Map

- Modify: `scripts/studies/lines_256_session_controller.py`
  - Add controller-owned source-proposal canonicalization.
  - Persist `proposal_resolution.json`.
  - Validate and score against controller-resolved provenance instead of raw provider `candidate_commit`.
  - Turn proposal-resolution failures into retryable proposal-state transitions instead of stale `proposal_running`.
- Modify: `tests/studies/test_lines_256_session_controller.py`
  - Add regression tests for hallucinated/mismatched `candidate_commit` handling.
  - Add failure tests for unprovable source candidate state.
  - Assert behavior/artifacts, not literal prompt text.
- Modify: `prompts/workflows/lines_256_arch_improvement/experiment_step_common.md`
  - Remove provider authority over final `candidate_commit`.
  - Keep `candidate_paths_file` as the provider-owned source-path contract.
- Modify: `prompts/workflows/lines_256_arch_improvement/experiment_step.md`
  - Keep the legacy prompt surface aligned with the split prompt contract.
- Modify: `prompts/workflows/lines_256_arch_improvement/debug_crash.md`
  - Remove provider authority over final `candidate_commit` for replacement source candidates.
- Modify: `prompts/workflows/lines_256_arch_improvement/redesign_candidate_implement.md`
  - Keep redesign factory on the same narrowed source-candidate contract.
- Modify: `docs/studies/lines_256_controller_loop.md`
  - Document `proposal_resolution.json` and controller-owned source canonicalization.
- Modify: `docs/findings.md`
  - Record the new failure class and the controller-owned fix.
- Create: `docs/plans/2026-04-03-lines256-source-proposal-canonicalization.md`
  - This plan document.

## Task 1: Lock the Failure in Tests First

**Files:**
- Modify: `tests/studies/test_lines_256_session_controller.py`

- [ ] **Step 1: Add a failing test for a hallucinated source `candidate_commit`**

Create a controller test where:
- provider metadata says `candidate_kind="source"`
- `candidate_paths_file` is valid
- provider `candidate_commit` is a bogus/non-existent SHA
- repo `HEAD` is on a valid source candidate commit
- protected tracked paths are preserved

Expected behavior:
- proposal does **not** die in `validate_ready_proposal(...)`
- controller writes `proposal_resolution.json`
- `proposal_resolution.json` records both `provider_candidate_commit` and `resolved_candidate_commit`
- `resolved_candidate_commit` equals the real `git rev-parse HEAD`
- mismatch is captured as a warning, not a fatal error

- [ ] **Step 2: Run the narrow test to verify it fails on current behavior**

Run:
```bash
pytest --collect-only tests/studies/test_lines_256_session_controller.py -q
pytest tests/studies/test_lines_256_session_controller.py -k hallucinated_source_candidate_commit -v
```

Expected:
- collect-only succeeds
- the new test fails because the controller still treats the bogus provider SHA as authoritative

- [ ] **Step 3: Add a failing test for a mismatched-but-existing provider SHA**

Create a second test where:
- provider `candidate_commit` points to a real commit
- repo `HEAD` points to a different real commit that is the actual candidate state

Expected behavior:
- controller resolves from `HEAD`
- writes a mismatch warning
- continues with the resolved commit

- [ ] **Step 4: Add a failing test for an unprovable source candidate state**

Create a failure-path test where:
- provider says `candidate_kind="source"`
- `candidate_paths_file` is present
- repo `HEAD` never moved off `base_ref`, or otherwise cannot prove a source candidate exists

Expected behavior:
- controller writes `proposal_result.json`
- marks the proposal handoff retryable
- resets to `proposal_pending`
- does **not** leave the session stuck at stale `proposal_running`

- [ ] **Step 5: Run the focused failure set**

Run:
```bash
pytest tests/studies/test_lines_256_session_controller.py -k "hallucinated_source_candidate_commit or mismatched_source_candidate_commit or unprovable_source_candidate_state" -v
```

Expected:
- all new tests fail before implementation

- [ ] **Step 6: Commit the red tests**

```bash
git add tests/studies/test_lines_256_session_controller.py
git commit -m "test: capture lines256 source proposal commit drift"
```

## Task 2: Add Controller-Owned Proposal Resolution

**Files:**
- Modify: `scripts/studies/lines_256_session_controller.py`

- [ ] **Step 1: Add artifact-path helpers and a controller-owned resolution writer**

Add a new iteration artifact such as:
- `iterations/<n>/proposal_resolution.json`

Recommended payload shape:
```json
{
  "timestamp_utc": "...",
  "iteration": 52,
  "candidate_kind": "source",
  "base_ref": "22bb...",
  "provider_candidate_commit": "c2e37204a935...",
  "resolved_candidate_commit": "c2e372041f52...",
  "provider_commit_matches_resolved": false,
  "warnings": ["provider_candidate_commit_mismatch"],
  "candidate_paths_file": "state/.../candidate_paths.json"
}
```

- [ ] **Step 2: Implement controller-side source proposal canonicalization**

Add a helper like:
```python
def resolve_ready_proposal(repo_root: Path, session: SessionState, proposal: dict[str, object]) -> dict[str, object]:
    ...
```

For `source` proposals it should:
- load the provider metadata
- resolve `actual_head = _git_head(repo_root)`
- load and validate `candidate_paths_file`
- verify protected tracked paths are still preserved
- treat provider `candidate_commit` as advisory only
- write `proposal_resolution.json`
- return a canonical proposal dict that uses `resolved_candidate_commit`

For `run_config` proposals it should still:
- validate tracked-path preservation
- write a lightweight `proposal_resolution.json`
- return the original config proposal with controller-owned resolution metadata

- [ ] **Step 3: Fail only when the controller cannot prove a valid source candidate**

Canonicalization should hard-fail only if controller-owned evidence is insufficient, for example:
- `HEAD` did not move off `base_ref` for a claimed source candidate
- `candidate_paths_file` is missing or malformed
- candidate paths overlap protected local paths
- protected tracked paths were disturbed

Those failures should be represented as deterministic controller errors, not inferred from provider text.

- [ ] **Step 4: Run the focused tests again**

Run:
```bash
pytest tests/studies/test_lines_256_session_controller.py -k "hallucinated_source_candidate_commit or mismatched_source_candidate_commit or unprovable_source_candidate_state" -v
```

Expected:
- the hallucinated/mismatched commit tests pass
- the unprovable-state test now fails later on retryability/state transition details if those are not implemented yet

- [ ] **Step 5: Commit the canonicalization core**

```bash
git add scripts/studies/lines_256_session_controller.py tests/studies/test_lines_256_session_controller.py
git commit -m "fix: canonicalize lines256 source proposals from repo state"
```

## Task 3: Make Proposal Handoff Recovery Use the Canonicalized Artifact

**Files:**
- Modify: `scripts/studies/lines_256_session_controller.py`

- [ ] **Step 1: Route direct and redesign factories through canonicalization before proposal success**

In both `direct_candidate_factory(...)` and `redesign_candidate_factory(...)`:
- load provider metadata
- normalize proposal
- canonicalize with controller-owned resolution
- only after successful canonicalization:
  - write `proposal_result.json` with `metadata_validated=true`
  - return the canonical proposal

Do not let raw provider metadata bypass this step.

- [ ] **Step 2: Make proposal-resolution failures retryable when appropriate**

When canonicalization fails because the controller cannot prove a valid candidate state:
- write `proposal_result.json`
- include a clear controller-owned reason
- set phase back to `proposal_pending`
- return `RETRYABLE_FAILURE`

Do **not** leave the session at `proposal_running` with only partial artifacts.

- [ ] **Step 3: Ensure downstream source-candidate consumers use the resolved commit**

Audit the controller’s later source-candidate code paths and switch them from raw provider metadata to the controller-resolved proposal dict/artifact, including:
- scored execution provenance
- source cleanup/reset paths
- result recording
- invalid-execution checks that refer to candidate commit identity

- [ ] **Step 4: Add/adjust tests for retryable recovery semantics**

Cover:
- proposal attempt + metadata + failed canonicalization resets to `proposal_pending`
- `proposal_result.json` exists with retryable classification
- resume can continue cleanly from the same iteration later

- [ ] **Step 5: Run the controller test slice for proposal recovery**

Run:
```bash
pytest tests/studies/test_lines_256_session_controller.py -k "proposal_running or retryable or hallucinated_source_candidate_commit or mismatched_source_candidate_commit or unprovable_source_candidate_state" -v
```

Expected:
- all targeted tests pass

- [ ] **Step 6: Commit the recovery behavior**

```bash
git add scripts/studies/lines_256_session_controller.py tests/studies/test_lines_256_session_controller.py
git commit -m "fix: recover lines256 proposal handoff from provider commit drift"
```

## Task 4: Narrow the Provider Contract

**Files:**
- Modify: `prompts/workflows/lines_256_arch_improvement/experiment_step_common.md`
- Modify: `prompts/workflows/lines_256_arch_improvement/experiment_step.md`
- Modify: `prompts/workflows/lines_256_arch_improvement/debug_crash.md`
- Modify: `prompts/workflows/lines_256_arch_improvement/redesign_candidate_implement.md`

- [ ] **Step 1: Remove `candidate_commit` from the required provider-written source metadata**

Prompt contract should say:
- for `source`, provider must write `candidate_paths_file`
- provider must create exactly one candidate commit
- controller will resolve and record the authoritative commit after the provider step returns

Do not ask the provider to manufacture a final commit SHA string.

- [ ] **Step 2: Keep prompt behavior task-local**

Make the prompt changes narrow:
- no workflow-loop language
- no new controller logic in prompt text
- only adjust the source-candidate package contract so the provider stops owning deterministic git provenance

- [ ] **Step 3: Run a prompt-adjacent workflow validation**

From the `PtychoPINN` repo root:
```bash
PYTHONPATH=/home/ollie/Documents/agent-orchestration python -m orchestrator run workflows/agent_orchestration/lines_256_session_controller.yaml --dry-run --stream-output
```

Expected:
- workflow validation successful

- [ ] **Step 4: Commit the prompt narrowing**

```bash
git add prompts/workflows/lines_256_arch_improvement/experiment_step_common.md prompts/workflows/lines_256_arch_improvement/experiment_step.md prompts/workflows/lines_256_arch_improvement/debug_crash.md prompts/workflows/lines_256_arch_improvement/redesign_candidate_implement.md
git commit -m "docs: narrow lines256 source proposal commit contract"
```

## Task 5: Document the New Controller Contract

**Files:**
- Modify: `docs/studies/lines_256_controller_loop.md`
- Modify: `docs/findings.md`

- [ ] **Step 1: Document `proposal_resolution.json` as a controller-owned artifact**

Update the controller-loop doc to state:
- provider metadata is not the final authority for `source` commits
- controller writes `proposal_resolution.json`
- the controller-resolved commit is the authoritative scored candidate provenance
- provider commit mismatches are warnings unless controller-owned evidence cannot prove candidate state

- [ ] **Step 2: Record the finding in `docs/findings.md`**

Add a finding capturing:
- the failure mode: provider hallucinated `candidate_commit` killed proposal handoff
- the fix: controller-owned canonicalization of source proposal provenance
- the rule: prompt/provider owns semantic candidate selection, controller owns final git provenance

- [ ] **Step 3: Run diff hygiene on the touched docs**

Run:
```bash
git diff --check -- docs/studies/lines_256_controller_loop.md docs/findings.md
```

Expected:
- no whitespace or patch-format errors

- [ ] **Step 4: Commit the docs**

```bash
git add docs/studies/lines_256_controller_loop.md docs/findings.md
git commit -m "docs: record lines256 source proposal canonicalization"
```

## Task 6: Full Verification and Real-Session Recovery Check

**Files:**
- Modify: none expected unless verification exposes defects

- [ ] **Step 1: Run the full controller test module**

Run:
```bash
pytest --collect-only tests/studies/test_lines_256_session_controller.py -q
pytest tests/studies/test_lines_256_session_controller.py -q
```

Expected:
- collect-only succeeds
- full module passes

- [ ] **Step 2: Compile the controller**

Run:
```bash
python -m py_compile scripts/studies/lines_256_session_controller.py
```

Expected:
- no syntax errors

- [ ] **Step 3: Re-run orchestrator validation**

Run:
```bash
PYTHONPATH=/home/ollie/Documents/agent-orchestration python -m orchestrator run workflows/agent_orchestration/lines_256_session_controller.yaml --dry-run --stream-output
```

Expected:
- validation successful

- [ ] **Step 4: Recover the real stale iteration `52`**

Use the existing stale session:
- session root: `state/lines_256_arch_improvement_v2/sessions/20260331T015545Z`

Goal:
- resume iteration `52`
- controller should ignore the bogus provider `candidate_commit`
- resolve the actual `HEAD` commit from repo state
- write `proposal_resolution.json`
- either proceed to scored execution or, if repo state has drifted further, reset to `proposal_pending` with a retryable `proposal_result.json`

This step is the real-world proof that the brittleness is fixed.

- [ ] **Step 5: Inspect the recovery artifacts**

Verify the iteration directory now contains:
- `proposal_attempt.json`
- `proposal_result.json`
- `proposal_resolution.json`

And that:
- `proposal_resolution.json["resolved_candidate_commit"]` matches `git rev-parse HEAD` for the actual scored candidate state
- any provider mismatch is captured as a warning rather than a fatal session stop

- [ ] **Step 6: Final commit if the recovery check required small follow-up fixes**

```bash
git add scripts/studies/lines_256_session_controller.py tests/studies/test_lines_256_session_controller.py prompts/workflows/lines_256_arch_improvement/experiment_step_common.md prompts/workflows/lines_256_arch_improvement/experiment_step.md prompts/workflows/lines_256_arch_improvement/debug_crash.md prompts/workflows/lines_256_arch_improvement/redesign_candidate_implement.md docs/studies/lines_256_controller_loop.md docs/findings.md
git commit -m "fix: harden lines256 source proposal provenance"
```
