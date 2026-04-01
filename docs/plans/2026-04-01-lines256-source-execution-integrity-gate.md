# Lines256 Source Execution Integrity Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent `lines_256` source candidates from being treated as normal scored outcomes when the intended source tree was not provably what executed.

**Architecture:** Keep this fix controller-owned and minimal. Add a deterministic post-score integrity check for `source` candidates that validates runtime provenance from invocation artifacts, introduce an `INVALID_EXECUTION` outcome distinct from `DISCARD`, and record a non-fatal suspicious-tie warning when a semantic source change exactly matches the accepted score.

**Tech Stack:** Python 3.11, pytest, JSON session artifacts, TSV ledger, agent-orchestration dry-run validation

---

### Task 1: Lock the New Behavior with Controller Tests

**Files:**
- Modify: `tests/studies/test_lines_256_session_controller.py`

- [ ] **Step 1: Add a failing test for source-candidate provenance mismatch**

Add a controller-level test that creates:
- accepted state
- `source` candidate proposal metadata
- wrapper/runner invocation artifacts whose `runtime_provenance` resolves to a foreign repo

Expected result:
- scored assessment is classified as `INVALID_EXECUTION`
- assessment carries an integrity reason mentioning import provenance drift

- [ ] **Step 2: Add a failing test for suspicious exact source tie**

Add a controller-level test where:
- provenance is valid
- a `source` candidate exactly ties accepted `amp_ssim`

Expected result:
- normal scored outcome remains `DISCARD`
- assessment includes a `suspicious_tie_warning`

- [ ] **Step 3: Run collect-only on the controller test module**

Run: `pytest --collect-only tests/studies/test_lines_256_session_controller.py -q`

Expected: the new tests collect cleanly.

- [ ] **Step 4: Run the two narrow selectors and confirm they fail**

Run:
- `pytest tests/studies/test_lines_256_session_controller.py -k invalid_execution -v`
- `pytest tests/studies/test_lines_256_session_controller.py -k suspicious_tie -v`

Expected: both fail before implementation for the right reason.

### Task 2: Add the Minimal Integrity Gate in the Controller

**Files:**
- Modify: `scripts/studies/lines_256_session_controller.py`
- Test: `tests/studies/test_lines_256_session_controller.py`

- [ ] **Step 1: Add helpers to load wrapper/runner invocation provenance**

Implement small helpers that read:
- `output_root / invocation.json`
- `output_root / runs/pinn_hybrid_resnet/invocation.json`

and normalize the nested `extra.runtime_provenance` payload.

- [ ] **Step 2: Add a source-only execution-integrity validator**

Implement a helper such as:

```python
def _validate_source_execution_integrity(
    repo_root: Path,
    proposal: dict[str, object],
    output_root: Path,
) -> dict[str, object]:
    ...
```

Rules:
- only meaningful for `candidate_kind == "source"`
- require wrapper and runner invocation artifacts to exist
- require runtime provenance to show:
  - `cwd == repo_root`
  - `pythonpath == str(repo_root.resolve())`
  - `ptycho_torch_file` under `repo_root`

Return a small structured result with:
- `status`: `ok` or `invalid_execution`
- `reasons`: list of strings

- [ ] **Step 3: Add `INVALID_EXECUTION` handling to scored assessment**

In `run_scored_candidate(...)`:
- after successful metrics/randomness validation
- before final `KEEP`/`DISCARD`
- if `candidate_kind == "source"` and integrity fails:
  - return assessment with `decision = "INVALID_EXECUTION"`
  - include `warning` or `integrity_reasons`

- [ ] **Step 4: Add an advisory suspicious-tie warning**

For `source` candidates only:
- if `amp_ssim == accepted_amp_ssim`
- attach `suspicious_tie_warning`
- do not change decision on that basis alone

- [ ] **Step 5: Persist the new assessment fields**

Update assessment writing so `INVALID_EXECUTION` rows and suspicious-tie warnings persist into:
- `candidate_assessment.json`
- `results.tsv`

- [ ] **Step 6: Run the narrow controller selectors again**

Run:
- `pytest tests/studies/test_lines_256_session_controller.py -k invalid_execution -v`
- `pytest tests/studies/test_lines_256_session_controller.py -k suspicious_tie -v`

Expected: PASS.

### Task 3: Document the New Outcome Semantics

**Files:**
- Modify: `docs/studies/lines_256_controller_loop.md`
- Modify: `docs/findings.md`

- [ ] **Step 1: Document `INVALID_EXECUTION` in the controller loop doc**

Clarify that:
- `INVALID_EXECUTION` is an infra/runtime-integrity outcome for `source` candidates
- it means the run completed but the controller could not prove the intended source tree executed
- it is not scientific evidence against the hypothesis

- [ ] **Step 2: Add a finding entry**

Add a short finding explaining:
- false ties can occur when runtime provenance is wrong
- exact source ties are suspicious but not sufficient proof alone
- the controller now validates execution provenance and can emit `INVALID_EXECUTION`

### Task 4: Verify End-to-End

**Files:**
- Modify: none

- [ ] **Step 1: Run the touched controller test module**

Run:
- `pytest tests/studies/test_lines_256_session_controller.py -v`

Expected: PASS.

- [ ] **Step 2: Compile the controller**

Run:
- `python -m py_compile scripts/studies/lines_256_session_controller.py`

Expected: no output, exit `0`.

- [ ] **Step 3: Run workflow validation**

Run:
- `PYTHONPATH=/home/ollie/Documents/agent-orchestration python -m orchestrator run workflows/agent_orchestration/lines_256_session_controller.yaml --dry-run --stream-output`

Expected: workflow validation successful.

- [ ] **Step 4: Run diff hygiene**

Run:
- `git diff --check -- scripts/studies/lines_256_session_controller.py tests/studies/test_lines_256_session_controller.py docs/studies/lines_256_controller_loop.md docs/findings.md`

Expected: clean.
