# Grid-Lines Baseline Opt-In Selection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `baseline` fully opt-in in `grid_lines_compare_wrapper.py` so it is selected exactly like other models and never run implicitly.

**Architecture:** Introduce explicit TF model selection (`pinn`, `baseline`) end-to-end, from CLI parsing to TF workflow execution. Keep backward compatibility for callers of `run_grid_lines_workflow` by defaulting to both TF models when no selector is provided. In the wrapper, split dataset preparation from model execution so Torch-only runs do not trigger TF baseline work.

**Tech Stack:** Python 3, `argparse`, dataclasses, existing grid-lines TF/Torch workflows, `pytest`.

---

## Principled Design

### Problem Statement

Current behavior violates selection symmetry:
- CLI default architectures include `baseline` by default.
- TF workflow (`run_grid_lines_workflow`) always trains/evaluates `baseline` and `pinn` together.
- Wrapper sometimes invokes TF workflow just to prepare datasets, which implicitly runs baseline.

Result: baseline can run even when the user did not request it.

### Design Principles

1. **Explicitness over implicit defaults**
- `baseline` must run only when explicitly selected via `--architectures` or `--models`.

2. **Selection symmetry across models**
- `baseline` should follow the same selection mechanics as `pinn`, `fno`, `hybrid`, etc.

3. **Separation of concerns**
- Dataset generation should be callable without triggering TF model training/inference.

4. **Backward compatibility for direct workflow callers**
- Existing direct calls to `run_grid_lines_workflow(cfg)` should preserve current behavior unless a selector is provided.

5. **Deterministic outputs and provenance**
- Selected-model behavior must match saved metrics/recons/visual orders exactly; no hidden model outputs.

### Decision

Implement a model selector in TF workflow and wire wrapper selection into it:
- Add optional `tf_models` parameter to `run_grid_lines_workflow`.
- Update wrapper default architectures to exclude baseline.
- When no TF models are selected, call dataset builder only (no TF training).
- In `--models` path, execute TF once per required `N` with only requested TF models.

This is the smallest coherent design that removes implicit baseline execution while preserving compatibility.

---

### Task 1: Lock Selection Semantics with RED Tests

**Files:**
- Modify: `tests/test_grid_lines_compare_wrapper.py`

**Step 1: Add failing test for default architecture set (baseline excluded)**

```python
def test_parse_args_default_architectures_excludes_baseline(tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import parse_args

    args = parse_args([
        "--N", "64",
        "--gridsize", "1",
        "--output-dir", str(tmp_path),
    ])

    assert "baseline" not in args.architectures
```

**Step 2: Add failing test that `cnn`-only run does not merge baseline metrics**

```python
def test_wrapper_cnn_only_excludes_baseline_metrics(monkeypatch, tmp_path):
    ...
    result = run_grid_lines_compare(..., architectures=("cnn",), ...)
    merged = json.loads((tmp_path / "metrics.json").read_text())
    assert "pinn" in merged
    assert "baseline" not in merged
```

**Step 3: Add failing test that Torch-only run uses dataset builder, not TF workflow**

```python
def test_wrapper_torch_only_uses_dataset_builder_not_tf_workflow(monkeypatch, tmp_path):
    ...
    run_grid_lines_compare(..., architectures=("fno",), ...)
    assert called["build_datasets"] is True
    assert called["tf_workflow"] is False
```

**Step 4: Run RED tests**

Run:
```bash
pytest tests/test_grid_lines_compare_wrapper.py -k "default_architectures_excludes_baseline or cnn_only_excludes_baseline or torch_only_uses_dataset_builder" -v
```
Expected: FAIL.

**Step 5: Commit test scaffolding**

```bash
git add tests/test_grid_lines_compare_wrapper.py
git commit -m "test(grid-lines-wrapper): codify baseline opt-in selection semantics"
```

### Task 2: Add TF Workflow Model Selector

**Files:**
- Modify: `ptycho/workflows/grid_lines_workflow.py`
- Modify: `tests/test_grid_lines_workflow.py`

**Step 1: Add failing workflow tests for selective TF model execution**

Add tests for:
- `tf_models=("pinn",)` skips baseline training/inference/metrics/recon.
- `tf_models=("baseline",)` skips pinn training/inference/metrics/recon.
- default call still returns both metrics for backward compatibility.

**Step 2: Run RED**

Run:
```bash
pytest tests/test_grid_lines_workflow.py -k "tf_models or baseline_only or pinn_only" -v
```
Expected: FAIL.

**Step 3: Implement selector in workflow**

Update signature:
```python
def run_grid_lines_workflow(
    cfg: GridLinesConfig,
    tf_models: Tuple[str, ...] = ("pinn", "baseline"),
) -> Dict[str, Any]:
```

Implementation requirements:
- Validate `tf_models` subset of `{"pinn", "baseline"}` and non-empty.
- Gate step 4/5 execution by membership.
- Build `metrics_payload` only for selected models.
- Save only selected recon artifacts.
- Build visual order dynamically from selected models.
- Preserve default behavior when called without `tf_models`.

**Step 4: Run GREEN**

Run:
```bash
pytest tests/test_grid_lines_workflow.py -k "tf_models or baseline_only or pinn_only" -v
```
Expected: PASS.

**Step 5: Commit**

```bash
git add ptycho/workflows/grid_lines_workflow.py tests/test_grid_lines_workflow.py
git commit -m "feat(grid-lines-workflow): add explicit TF model selector for pinn/baseline"
```

### Task 3: Wire Wrapper to Explicit TF Selection

**Files:**
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Modify: `tests/test_grid_lines_compare_wrapper.py`

**Step 1: Update CLI default architectures to remove baseline**

Change parse default:
```python
default="cnn,fno,hybrid,stable_hybrid,fno_vanilla,hybrid_resnet"
```

**Step 2: Derive TF selection from requested models/architectures**

Add helper logic:
- TF models: `("pinn", "baseline")` intersection with selected set.
- Torch models unchanged.
- PtychoViT unchanged.

**Step 3: Replace implicit TF workflow calls**

In architecture path:
- If TF models selected: call `run_grid_lines_workflow(cfg, tf_models=...)`.
- If no TF models selected and dataset missing: call `build_grid_lines_datasets(cfg)`.

In `--models` path:
- Group selected TF models by required `N` and call workflow once per `N` with only those TF models.
- Do not call TF workflow inside per-model loop for torch/ptychovit IDs.

**Step 4: Ensure merged metrics/recon order strictly follows selection**

- No baseline key unless selected.
- `render_grid_lines_visuals` order includes baseline only when selected.

**Step 5: Run GREEN tests**

Run:
```bash
pytest tests/test_grid_lines_compare_wrapper.py -k "baseline or architectures or torch_only_uses_dataset_builder" -v
```
Expected: PASS.

**Step 6: Commit**

```bash
git add scripts/studies/grid_lines_compare_wrapper.py tests/test_grid_lines_compare_wrapper.py
git commit -m "feat(grid-lines-wrapper): make baseline opt-in and selection-symmetric"
```

### Task 4: Regression Coverage and Compatibility Checks

**Files:**
- Modify (if needed): `tests/test_grid_lines_compare_wrapper.py`
- Modify (if needed): `tests/test_grid_lines_workflow.py`

**Step 1: Add compatibility test for legacy direct workflow call**

```python
def test_run_grid_lines_workflow_default_runs_both_models(...):
    ...
    result = run_grid_lines_workflow(cfg)
    assert "pinn" in result["metrics"]
    assert "baseline" in result["metrics"]
```

**Step 2: Add wrapper test for explicit baseline inclusion**

```python
def test_wrapper_baseline_included_only_when_requested(...):
    ...
```

**Step 3: Run focused regression suite**

Run:
```bash
pytest tests/test_grid_lines_compare_wrapper.py -v
pytest tests/test_grid_lines_workflow.py -v
```
Expected: PASS.

**Step 4: Commit**

```bash
git add tests/test_grid_lines_compare_wrapper.py tests/test_grid_lines_workflow.py
git commit -m "test(grid-lines): add regression coverage for baseline opt-in behavior"
```

### Task 5: Docs and CLI Reference Alignment

**Files:**
- Modify: `docs/COMMANDS_REFERENCE.md`
- Modify: `docs/studies/index.md` (if examples depend on defaults)
- Modify: `scripts/studies/README.md` (if architecture defaults are documented)

**Step 1: Update docs to state baseline is opt-in**

Document:
- default architecture set excludes baseline
- include baseline explicitly via `--architectures ... ,baseline` or `--models baseline`

**Step 2: Add explicit baseline example**

```bash
python scripts/studies/grid_lines_compare_wrapper.py \
  --N 64 --gridsize 1 --output-dir outputs/x \
  --architectures cnn,baseline,fno
```

**Step 3: Verify docs references**

Run:
```bash
rg -n "baseline.*opt-in|architectures|grid_lines_compare_wrapper" docs/COMMANDS_REFERENCE.md docs/studies/index.md scripts/studies/README.md
```
Expected: references present and consistent.

**Step 4: Commit**

```bash
git add docs/COMMANDS_REFERENCE.md docs/studies/index.md scripts/studies/README.md
git commit -m "docs(grid-lines): document baseline as explicit opt-in selection"
```

### Task 6: Final Verification Sweep

**Step 1: Run targeted tests**

```bash
pytest tests/test_grid_lines_compare_wrapper.py -v
pytest tests/test_grid_lines_workflow.py -v
```

**Step 2: Run related Torch wrapper sanity tests**

```bash
pytest tests/torch/test_grid_lines_torch_runner.py -k invocation_artifacts -v
```

**Step 3: Store logs**

- Save outputs under `.artifacts/studies/baseline_opt_in_selection/`.

**Step 4: Final status check**

```bash
git status
```

