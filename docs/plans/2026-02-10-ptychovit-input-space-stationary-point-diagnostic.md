# PtychoViT Input-Space Stationary-Point Diagnostic Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a reproducible diagnostic that performs constrained input optimization on the PtychoViT model to identify stationary points in diffraction-input space and report normalization/scale sensitivity.

**Architecture:** Implement a standalone study script that loads the existing bridge-ready model/data path, optimizes an input tensor under physical constraints, and records objective/gradient trajectories. Keep this strictly diagnostic: no training-path or inference-contract changes. Use bounded input-domain projection and optional regularizers so findings are interpretable against current interop contract assumptions.

**Tech Stack:** Python 3, PyTorch, NumPy, h5py, pytest, `scripts/studies`, existing PtychoViT bridge/runtime config and dataset loaders.

**Execution discipline:**
- Use `@superpowers:test-driven-development` for every behavior change.
- Use `@superpowers:systematic-debugging` for any unexpected optimizer/stability failures.
- Use `@superpowers:verification-before-completion` before any success claims.

---

### Task 0: Red/Green Unit Tests for Optimization Core

**Files:**
- Create: `tests/test_ptychovit_input_optimization_diagnostic.py`
- Create: `scripts/studies/ptychovit_input_optimization_diagnostic.py`

**Step 1: Write failing tests**

```python
def test_project_input_enforces_nonnegative_and_max_bound():
    ...

def test_stationary_criterion_triggers_on_small_gradient_norm():
    ...

def test_objective_components_reported_with_expected_keys():
    ...
```

**Step 2: Run tests to verify RED**

Run:
- `pytest tests/test_ptychovit_input_optimization_diagnostic.py -k "project_input or stationary_criterion or objective_components" -v`

Expected: FAIL (new module/functions missing).

**Step 3: Write minimal implementation**

Implement in `scripts/studies/ptychovit_input_optimization_diagnostic.py`:
- `project_input(x, min_value=0.0, max_value=None)`
- `is_stationary(grad_norm, threshold)`
- `compute_objective(pred_amp, pred_phase, weights)` returning component dict + scalar objective

**Step 4: Run tests to verify GREEN**

Run:
- `pytest tests/test_ptychovit_input_optimization_diagnostic.py -k "project_input or stationary_criterion or objective_components" -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_ptychovit_input_optimization_diagnostic.py scripts/studies/ptychovit_input_optimization_diagnostic.py
git commit -m "test: add ptychovit stationary-point diagnostic optimization primitives"
```

### Task 1: CLI Contract and Artifact Schema

**Files:**
- Modify: `scripts/studies/ptychovit_input_optimization_diagnostic.py`
- Modify: `tests/test_ptychovit_input_optimization_diagnostic.py`

**Step 1: Write failing tests**

```python
def test_cli_writes_json_report_with_required_fields(tmp_path):
    # fields: objective_history, grad_norm_history, stationary_step,
    # input_stats, normalization_context, config
    ...

def test_cli_rejects_invalid_bounds_and_weights(tmp_path):
    ...
```

**Step 2: Run tests to verify RED**

Run:
- `pytest tests/test_ptychovit_input_optimization_diagnostic.py -k "json_report or invalid_bounds" -v`

Expected: FAIL.

**Step 3: Write minimal implementation**

Add CLI args:
- `--output-dir`, `--checkpoint`, `--ptychovit-repo`, `--test-dp`, `--test-para`
- `--steps`, `--lr`, `--stationary-threshold`, `--input-max`
- objective weights (`--w-amp-var`, `--w-phase-var`, `--w-tv`, `--w-forward-consistency`)

Write `diagnostic_report.json` with required schema and deterministic metadata.

**Step 4: Run tests to verify GREEN**

Run:
- `pytest tests/test_ptychovit_input_optimization_diagnostic.py -k "json_report or invalid_bounds" -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_ptychovit_input_optimization_diagnostic.py scripts/studies/ptychovit_input_optimization_diagnostic.py
git commit -m "feat: add ptychovit input-optimization diagnostic cli and report schema"
```

### Task 2: Bridge-Compatible Model/Data Integration Smoke

**Files:**
- Modify: `scripts/studies/ptychovit_input_optimization_diagnostic.py`
- Modify: `tests/test_ptychovit_input_optimization_diagnostic.py`

**Step 1: Write failing integration-style test**

```python
def test_diagnostic_runs_with_bridge_artifacts_and_stops_at_stationary(tmp_path):
    # Use minimal synthetic bridge-like files and a tiny model stub.
    # Assert finite histories and stationary_step <= steps.
    ...
```

**Step 2: Run test to verify RED**

Run:
- `pytest tests/test_ptychovit_input_optimization_diagnostic.py -k bridge_artifacts_and_stops -v`

Expected: FAIL.

**Step 3: Write minimal implementation**

- Reuse bridge-compatible loading path (same dataset expectations as `scripts/studies/ptychovit_bridge_entrypoint.py`).
- Optimize one selected scan (or batch) with gradient ascent.
- Capture per-step: objective, gradient norm, input min/max/mean/std, output amp/phase stats.

**Step 4: Run test to verify GREEN**

Run:
- `pytest tests/test_ptychovit_input_optimization_diagnostic.py -k bridge_artifacts_and_stops -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_ptychovit_input_optimization_diagnostic.py scripts/studies/ptychovit_input_optimization_diagnostic.py
git commit -m "feat: integrate stationary-point diagnostic with bridge-compatible ptychovit inputs"
```

### Task 3: Documentation and Workflow Guidance

**Files:**
- Modify: `docs/workflows/ptychovit.md`
- Modify: `docs/findings.md` (if a concrete normalization expectation pattern is observed)
- Modify: `tests/test_docs_ptychovit_workflow.py`

**Step 1: Write failing docs test**

```python
def test_ptychovit_workflow_documents_stationary_point_diagnostic_runbook():
    text = Path("docs/workflows/ptychovit.md").read_text()
    assert "stationary-point diagnostic" in text
    assert "input optimization" in text
    assert "diagnostic only" in text
```

**Step 2: Run tests to verify RED**

Run:
- `pytest tests/test_docs_ptychovit_workflow.py -k stationary_point_diagnostic_runbook -v`

Expected: FAIL.

**Step 3: Write minimal docs updates**

- Add runbook section with command example.
- Define interpretation guidance for normalization expectation signals (saturation, preferred scale window, collapse patterns).
- State explicitly that this does not modify reconstruction behavior.

**Step 4: Run tests to verify GREEN**

Run:
- `pytest tests/test_docs_ptychovit_workflow.py -k stationary_point_diagnostic_runbook -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add docs/workflows/ptychovit.md tests/test_docs_ptychovit_workflow.py
git commit -m "docs: add ptychovit stationary-point normalization diagnostic runbook"
```

### Task 4: Execute Diagnostic and Archive Evidence

**Files:**
- Modify: `docs/plans/2026-02-10-ptychovit-input-space-stationary-point-diagnostic.md`
- Modify/Create (gitignored): `.artifacts/ptychovit_stationary_input_diag/README.md`

**Step 1: Run diagnostic on fresh bridge artifacts**

Run (example):

```bash
python scripts/studies/ptychovit_input_optimization_diagnostic.py \
  --checkpoint tmp/ptychovit_initial_fresh_post_contract_fix/runs/pinn_ptychovit/best_model.pth \
  --ptychovit-repo /home/ollie/Documents/ptycho-vit \
  --test-dp tmp/ptychovit_initial_fresh_post_contract_fix/runs/pinn_ptychovit/bridge_work/data/test_dp.hdf5 \
  --test-para tmp/ptychovit_initial_fresh_post_contract_fix/runs/pinn_ptychovit/bridge_work/data/test_para.hdf5 \
  --output-dir tmp/ptychovit_stationary_diag \
  --steps 200 --lr 1e-2 --stationary-threshold 1e-5 --input-max 100.0
```

Expected:
- Exit 0
- `tmp/ptychovit_stationary_diag/diagnostic_report.json` exists

**Step 2: Summarize normalization expectation indicators**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
r = json.loads(Path('tmp/ptychovit_stationary_diag/diagnostic_report.json').read_text())
print('stationary_step', r['stationary_step'])
print('grad_norm_final', r['grad_norm_history'][-1])
print('input_stats_final', r['input_stats'][-1])
print('normalization_context', r['normalization_context'])
PY
```

Expected:
- finite values and explicit normalization context for interpretation.

**Step 3: Archive evidence pointer**

- Add commands, report paths, and interpretation notes to `.artifacts/ptychovit_stationary_input_diag/README.md`.

**Step 4: Update plan evidence section**

- Append RED/GREEN log summary and diagnostic outputs.

**Step 5: Commit**

```bash
git add docs/plans/2026-02-10-ptychovit-input-space-stationary-point-diagnostic.md

git commit -m "chore: record stationary-point normalization diagnostic evidence"
```

## Expected Outputs

- `scripts/studies/ptychovit_input_optimization_diagnostic.py`
- `tests/test_ptychovit_input_optimization_diagnostic.py`
- `tmp/ptychovit_stationary_diag/diagnostic_report.json`
- `docs/workflows/ptychovit.md` stationary-point diagnostic section
- `.artifacts/ptychovit_stationary_input_diag/README.md`

## Non-Goals (YAGNI)

- No changes to PtychoViT training/inference production behavior.
- No automatic use of optimized inputs in reconstruction pipelines.
- No upstream `ptycho-vit` repository edits.
