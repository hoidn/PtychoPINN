# Grid-Lines External Dataset Mode Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable grid-lines Torch-style studies to run on non-synthetic datasets (for example fly64) via explicit dataset modes, without overloading the existing synthetic grid-lines code path.

**Architecture:** Introduce a single dataset-resolution layer that chooses between (a) existing synthetic grid-lines dataset generation and (b) external NPZ train/test inputs. Keep the Torch runner contract unchanged by requiring a strict, validated NPZ split format. In phase 1, allow external mode only for Torch model IDs and fail fast for TF/PtychoViT arms to avoid semantic drift.

**Tech Stack:** Python 3.11, argparse, pathlib, NumPy NPZ I/O, `ptycho.metadata.MetadataManager`, pytest

---

### Task 1: Define External Dataset Mode CLI Contract (RED)

**Files:**
- Modify: `tests/test_grid_lines_compare_wrapper.py`
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Test: `tests/test_grid_lines_compare_wrapper.py`

**Step 1: Write the failing tests**

Add tests asserting:

```python
def test_parse_args_accepts_external_npz_mode(tmp_path):
    args = parse_args([
        "--N", "64",
        "--gridsize", "1",
        "--output-dir", str(tmp_path),
        "--dataset-mode", "external_npz",
        "--train-npz", "datasets/fly64/train.npz",
        "--test-npz", "datasets/fly64/test.npz",
        "--models", "pinn_hybrid_resnet",
    ])
    assert args.dataset_mode == "external_npz"
    assert str(args.train_npz).endswith("train.npz")
    assert str(args.test_npz).endswith("test.npz")
```

```python
def test_parse_args_external_mode_requires_train_test_npz(tmp_path):
    with pytest.raises(SystemExit):
        parse_args([
            "--N", "64",
            "--gridsize", "1",
            "--output-dir", str(tmp_path),
            "--dataset-mode", "external_npz",
            "--models", "pinn_hybrid_resnet",
        ])
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_grid_lines_compare_wrapper.py -k "external_npz_mode" -v`
Expected: FAIL (unknown CLI args / missing validation behavior)

**Step 3: Write minimal implementation**

Add CLI args and parse-time validation:
- `--dataset-mode synthetic_grid_lines|external_npz` (default `synthetic_grid_lines`)
- `--train-npz`, `--test-npz` (required when `external_npz`)

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_grid_lines_compare_wrapper.py -k "external_npz_mode" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_grid_lines_compare_wrapper.py scripts/studies/grid_lines_compare_wrapper.py
git commit -m "feat(grid-lines): add external_npz dataset mode CLI contract"
```

---

### Task 2: Add Dataset Resolver Module (Single Entry Point)

**Files:**
- Create: `scripts/studies/grid_dataset_resolver.py`
- Create: `tests/studies/test_grid_dataset_resolver.py`
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Test: `tests/studies/test_grid_dataset_resolver.py`

**Step 1: Write the failing tests**

Add tests for resolver behavior:

```python
def test_resolver_external_npz_validates_required_keys(tmp_path):
    # create minimal train/test npz with required keys
    # call resolve_study_datasets(... dataset_mode="external_npz")
    # assert returned train/test paths match inputs
```

```python
def test_resolver_external_npz_rejects_missing_required_key(tmp_path):
    # omit Y_phi
    # expect ValueError mentioning missing key
```

```python
def test_resolver_synthetic_delegates_to_grid_builder(monkeypatch, tmp_path):
    # monkeypatch build_grid_lines_datasets(...)
    # assert resolver calls builder and returns generated paths
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/studies/test_grid_dataset_resolver.py -v`
Expected: FAIL (module/function missing)

**Step 3: Write minimal implementation**

Implement resolver API:

```python
def resolve_study_datasets(*, dataset_mode, output_dir, N, gridsize, ...):
    ...
```

Behavior:
- `synthetic_grid_lines`: delegate to existing `grid_lines_workflow` builders.
- `external_npz`: validate both NPZ files contain required keys:
  - `diffraction`, `Y_I`, `Y_phi`, `coords_nominal`
  - optional for downstream recon/eval: `probeGuess`, `YY_full|YY_ground_truth`, `norm_Y_I`
- return normalized `Path` bundle `{train_npz, test_npz, gt_recon(optional), metadata}`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/studies/test_grid_dataset_resolver.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/studies/grid_dataset_resolver.py tests/studies/test_grid_dataset_resolver.py scripts/studies/grid_lines_compare_wrapper.py
git commit -m "feat(grid-lines): add dataset resolver for synthetic vs external npz"
```

---

### Task 3: Integrate Resolver Into Wrapper + Add Model Compatibility Guardrails

**Files:**
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Modify: `tests/test_grid_lines_compare_wrapper.py`
- Test: `tests/test_grid_lines_compare_wrapper.py`

**Step 1: Write the failing tests**

Add tests asserting:

```python
def test_external_npz_mode_rejects_tf_models(tmp_path):
    with pytest.raises(ValueError, match="external_npz.*torch models"):
        run_grid_lines_compare(..., dataset_mode="external_npz", models=("pinn",), ...)
```

```python
def test_external_npz_mode_allows_torch_models(monkeypatch, tmp_path):
    # monkeypatch resolver + torch runner + eval path
    # run with models=("pinn_hybrid_resnet",)
    # assert no synthetic dataset builder calls occur
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_grid_lines_compare_wrapper.py -k "external_npz_mode or torch models" -v`
Expected: FAIL

**Step 3: Write minimal implementation**

- Replace direct synthetic-dataset branching with resolver call.
- Add explicit guard in external mode:
  - allowed: `pinn_fno`, `pinn_hybrid`, `pinn_stable_hybrid`, `pinn_fno_vanilla`, `pinn_hybrid_resnet`
  - reject: `pinn`, `baseline`, `pinn_ptychovit`
- Error messages must state what is supported and why.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_grid_lines_compare_wrapper.py -k "external_npz_mode or torch models" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_compare_wrapper.py tests/test_grid_lines_compare_wrapper.py
git commit -m "feat(grid-lines): route wrapper dataset flow through resolver with external mode guardrails"
```

---

### Task 4: Add Optional Adapter Script for Legacy/Generalization NPZ Inputs (Phase 2, Scoped)

**Files:**
- Create: `scripts/studies/prepare_grid_study_external_splits.py`
- Create: `tests/studies/test_prepare_grid_study_external_splits.py`
- Modify: `scripts/studies/README.md`
- Test: `tests/studies/test_prepare_grid_study_external_splits.py`

**Step 1: Write the failing tests**

Add tests for a narrow, explicit adapter contract:

```python
def test_adapter_normalizes_key_aliases_and_writes_train_test_splits(tmp_path):
    # input: diffraction or diff3d + coords inputs
    # output: train.npz/test.npz with grid-runner required keys
```

```python
def test_adapter_refuses_inputs_missing_minimum_contract(tmp_path):
    # missing coords/object context
    # expect actionable ValueError
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/studies/test_prepare_grid_study_external_splits.py -v`
Expected: FAIL (script/module missing)

**Step 3: Write minimal implementation**

Create a standalone preparation tool that:
- validates incoming NPZ,
- emits contract-compliant `train.npz` and `test.npz` for Torch runner,
- writes metadata via `MetadataManager.save_with_metadata` including `coords_type`, `nimgs_test`, `outer_offset_test` (or explicit null-safe fallback fields if not available).

**Step 4: Run test to verify it passes**

Run: `pytest tests/studies/test_prepare_grid_study_external_splits.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/studies/prepare_grid_study_external_splits.py tests/studies/test_prepare_grid_study_external_splits.py scripts/studies/README.md
git commit -m "feat(grid-lines): add external split preparation tool for torch studies"
```

---

### Task 5: Update Documentation With Capability Matrix and Non-Confusing Usage

**Files:**
- Modify: `docs/COMMANDS_REFERENCE.md`
- Modify: `docs/workflows/pytorch.md`
- Modify: `scripts/studies/README.md`
- Test: `tests/test_docs_ptychovit_workflow.py` (and any doc-link/lint checks already used in repo)

**Step 1: Write the failing test / assertion update (if applicable)**

Add or adjust doc test(s) to assert new external mode flags/commands are documented.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_docs_ptychovit_workflow.py -v`
Expected: FAIL if new docs are required by tests

**Step 3: Write minimal documentation updates**

Document:
- dataset mode matrix (`synthetic_grid_lines` vs `external_npz`),
- supported model IDs per mode,
- example commands for fly64-style external inputs,
- explicit note that external mode is Torch-only in phase 1.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_docs_ptychovit_workflow.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add docs/COMMANDS_REFERENCE.md docs/workflows/pytorch.md scripts/studies/README.md tests/test_docs_ptychovit_workflow.py
git commit -m "docs(grid-lines): document external dataset mode and model compatibility matrix"
```

---

### Task 6: End-to-End Verification and Study Index Provenance

**Files:**
- Modify: `docs/studies/index.md`
- Optional Create: `.artifacts/studies/grid_lines_external_npz_smoke/` helper scripts
- Test: targeted wrapper and runner pytest selectors

**Step 1: Write/prepare verification checklist**

Create a reproducible smoke sequence:
1. prepare external splits (or use existing contract-compliant NPZs),
2. run compare wrapper in `external_npz` mode with `pinn_hybrid_resnet`,
3. confirm `metrics.json`, `metrics_by_model.json`, visuals, and tables are emitted.

**Step 2: Run targeted tests**

Run:
- `pytest tests/studies/test_grid_dataset_resolver.py -v`
- `pytest tests/test_grid_lines_compare_wrapper.py -k "external_npz_mode" -v`
- `pytest tests/torch/test_grid_lines_torch_runner.py -k "invocation or metadata" -v`

Expected: PASS

**Step 3: Run smoke command**

```bash
python scripts/studies/grid_lines_compare_wrapper.py \
  --N 64 --gridsize 1 \
  --output-dir outputs/grid_lines_external_fly64_smoke \
  --dataset-mode external_npz \
  --train-npz <prepared_train_npz> \
  --test-npz <prepared_test_npz> \
  --models pinn_hybrid_resnet \
  --nepochs 5 --batch-size 16 --seed 3
```

Expected:
- `outputs/.../metrics.json`
- `outputs/.../metrics_by_model.json`
- `outputs/.../visuals/compare_amp_phase.png`
- `outputs/.../metrics_table.tex`

**Step 4: Update studies provenance**

Add a short entry in `docs/studies/index.md` with:
- exact CLI invocation,
- script path(s),
- output directory.

**Step 5: Commit**

```bash
git add docs/studies/index.md
# plus any smoke helper script references if intentionally tracked
git commit -m "chore(studies): add external npz grid-lines smoke provenance"
```

---

## Scope Guardrails (No Debt / No Confusion)

1. Keep synthetic grid-lines behavior unchanged unless `--dataset-mode external_npz` is explicitly selected.
2. Keep one dataset resolution entry point (`grid_dataset_resolver.py`), not duplicate conditional logic in wrapper/runner.
3. External mode in phase 1 is Torch-only; do not silently run TF/PtychoViT against unsupported dataset semantics.
4. Prefer explicit errors over implicit fallbacks.
5. Keep dataset prep as a separate command/tool, not hidden wrapper side-effects.
