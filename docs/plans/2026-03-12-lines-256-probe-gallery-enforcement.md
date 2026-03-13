# Lines 256 Probe Gallery Enforcement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the `lines_256` experiment wrapper guarantee that the comparison PNG copied into the session gallery includes the probe column.

**Architecture:** Keep the contract local to the `lines_256` wrapper. After the underlying runner finishes successfully, rerender visuals from saved recon artifacts using the existing grid-lines visual helper so `visuals/compare_amp_phase.png` is probe-inclusive before the orchestration workflow copies it into the session gallery.

**Tech Stack:** Python, pytest, YAML orchestration contract, grid-lines visual helpers

---

### Task 1: Add a failing wrapper test

**Files:**
- Modify: `tests/studies/test_run_lines_256_arch_experiment.py`
- Modify: `scripts/studies/run_lines_256_arch_experiment.py`

**Step 1: Write the failing test**

Add a unit test that:
- monkeypatches the subprocess runner to simulate a successful wrapped experiment
- creates minimal `recons/gt/recon.npz` and `recons/pinn_hybrid_resnet/recon.npz` artifacts under a temp output dir
- monkeypatches the visual rerender helper so the test can observe that it is called
- asserts the wrapper invokes probe-inclusive rerendering after a successful run

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/studies/test_run_lines_256_arch_experiment.py -q
```

Expected: FAIL because the wrapper does not yet rerender the comparison PNG.

### Task 2: Implement the minimal wrapper fix

**Files:**
- Modify: `scripts/studies/run_lines_256_arch_experiment.py`

**Step 1: Add a small helper**

Add a helper that:
- calls the existing grid-lines visual rerender path with `order=("gt", "pinn_hybrid_resnet")`
- verifies the resulting `visuals/compare_amp_phase.png` exists

**Step 2: Call it only on success**

After the wrapped subprocess returns `0`, invoke the helper before returning from `main()`.

**Step 3: Keep failure behavior unchanged**

Do not attempt rerendering on failed wrapped runs.

### Task 3: Verify the contract

**Files:**
- Modify: `tests/studies/test_run_lines_256_arch_experiment.py`
- Verify: `tests/studies/test_lines_256_arch_improvement_workflow.py`

**Step 1: Run targeted tests**

Run:

```bash
pytest tests/studies/test_run_lines_256_arch_experiment.py tests/studies/test_lines_256_arch_improvement_workflow.py -q
```

Expected: PASS.

**Step 2: Optional wrapper smoke**

Run:

```bash
python scripts/studies/run_lines_256_arch_experiment.py --help
```

Expected: command succeeds.
