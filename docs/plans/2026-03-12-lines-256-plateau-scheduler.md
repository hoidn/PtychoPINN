# Lines 256 Plateau Scheduler Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the `lines_256` study always use `ReduceLROnPlateau` with `plateau_min_lr=2e-4`.

**Architecture:** Pin the scheduler at the thin-wrapper boundary so every `lines_256` baseline and candidate run gets the same LR policy without changing the generic Torch runner defaults. Update the study docs so the wrapper contract matches the actual run behavior.

**Tech Stack:** Python, pytest, Markdown docs

---

### Task 1: Add a failing wrapper contract test

**Files:**
- Modify: `tests/studies/test_run_lines_256_arch_experiment.py`

**Step 1: Write the failing test**

Extend the wrapper command-contract test to assert:
- `--scheduler ReduceLROnPlateau`
- `--plateau-min-lr 0.0002`

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/studies/test_run_lines_256_arch_experiment.py::test_build_runner_cmd_pins_lines_256_invariants -q
```

Expected: FAIL because the wrapper does not yet pass those flags.

### Task 2: Implement the wrapper change

**Files:**
- Modify: `scripts/studies/run_lines_256_arch_experiment.py`

**Step 1: Add fixed scheduler constants**

Add study-level constants for:
- `FIXED_SCHEDULER = "ReduceLROnPlateau"`
- `FIXED_PLATEAU_MIN_LR = 2e-4`

**Step 2: Pass the flags through the runner command**

Update `build_runner_cmd()` to append:
- `--scheduler ReduceLROnPlateau`
- `--plateau-min-lr 0.0002`

**Step 3: Persist the fixed settings in invocation metadata**

Add the scheduler values to the wrapper’s invocation `extra` payload.

### Task 3: Align study docs

**Files:**
- Modify: `docs/studies/lines_256_dataset.md`
- Modify: `docs/studies/lines_256_arch_improvement_loop.md`

**Step 1: Update pinned-wrapper bullets**

Document that the wrapper also pins:
- `scheduler=ReduceLROnPlateau`
- `plateau_min_lr=2e-4`

### Task 4: Verify

**Files:**
- Verify: `tests/studies/test_run_lines_256_arch_experiment.py`
- Verify: `tests/studies/test_lines_256_arch_improvement_workflow.py`

**Step 1: Run targeted tests**

Run:

```bash
pytest tests/studies/test_run_lines_256_arch_experiment.py tests/studies/test_lines_256_arch_improvement_workflow.py -q
```

Expected: PASS.

**Step 2: Wrapper help smoke**

Run:

```bash
python scripts/studies/run_lines_256_arch_experiment.py --help
```

Expected: command succeeds.
