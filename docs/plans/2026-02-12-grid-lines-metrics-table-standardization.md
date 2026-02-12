# Grid Lines Metrics Table Standardization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `.tex` metrics table generation a standard, automatic part of the grid-lines comparison wrapper output flow (alongside `metrics.json` and `visuals/`), with hierarchical grouping by `N` so one output can contain zero or more model rows per `N` value and remain extensible to future merged multi-directory tables.

**Architecture:** Add a reusable table-writer utility that accepts model metrics plus optional model→`N` mapping and emits multi-level LaTeX tables grouped by `N`. Then route both wrapper execution branches through a single finalize path that writes metrics, renders visuals, and writes table artifacts. Keep default behavior dependency-light (`.tex` required, image/PDF optional) and enforce via wrapper unit tests.

**Tech Stack:** Python 3.11, existing `scripts/studies/` orchestration, pytest, JSON/Pathlib.

---

### Task 1: Lock Behavior With Failing Wrapper Tests

**Files:**
- Modify: `tests/test_grid_lines_compare_wrapper.py`
- Reference: `scripts/studies/grid_lines_compare_wrapper.py`

**Step 1: Write failing tests for standard table artifacts**

Add tests that run `run_grid_lines_compare(...)` via existing monkeypatch-style stubs and assert:
- `output_dir/metrics_table.tex` is created.
- table contains expected hierarchical header cells (for example `N`, `Model`, metric headers) and expected row labels for selected models.
- table includes correct `N` grouping values for rows in both single-`N` and model-override paths.
- (optional, if included in implementation) `metrics_table_best.tex` is created and non-empty.

**Step 2: Run targeted tests to confirm RED**

Run:
```bash
pytest tests/test_grid_lines_compare_wrapper.py -k "metrics_table" -v
```

Expected: FAIL because wrapper currently does not generate `.tex` table artifacts.

**Step 3: Commit failing-test scaffold (optional checkpoint commit)**

```bash
git add tests/test_grid_lines_compare_wrapper.py
git commit -m "test(grid-lines): require metrics table artifacts from compare wrapper"
```

---

### Task 2: Add Reusable Metrics Table Utility

**Files:**
- Create: `scripts/studies/metrics_tables.py`
- Modify: `tests/test_grid_lines_compare_wrapper.py` (if direct unit coverage is added)

**Step 1: Implement minimal writer API**

Create a small utility with explicit API, e.g.:
- `write_metrics_tables(output_dir: Path, merged_metrics: dict, model_ns: dict[str, int] | None = None) -> dict`
- Writes both `metrics_table.tex` and `metrics_table_best.tex` in one call.

Core behavior:
- deterministic model ordering compatible with wrapper outputs (e.g., `pinn`, `baseline`, `pinn_fno_vanilla`, `pinn_fno`, `pinn_hybrid`, `pinn_stable_hybrid`, `pinn_hybrid_resnet`, `pinn_ptychovit` when present).
- deterministic metric ordering (`mae`, `mse`, `psnr`, `ssim`, `ms_ssim`, `frc50`).
- deterministic `N` grouping/order with support for 0+ rows per `N`.
- robust formatting for both numeric strings and numeric values.
- no external rendering dependency in the core utility.

**Step 2: Add/adjust utility-level assertions via wrapper tests**

If separate unit tests are added, validate:
- known input dict writes expected `.tex` rows/cells.
- `N`-grouped layout is correct when multiple `N` values are provided.
- missing models/metrics are skipped gracefully (no crash).

**Step 3: Run focused tests**

Run:
```bash
pytest tests/test_grid_lines_compare_wrapper.py -k "metrics_table" -v
```

Expected: still RED until wrapper is wired.

**Step 4: Commit utility**

```bash
git add scripts/studies/metrics_tables.py tests/test_grid_lines_compare_wrapper.py
git commit -m "feat(studies): add reusable grid-lines metrics table tex writer"
```

---

### Task 3: Integrate Utility Into Standard Wrapper Codepath

**Files:**
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Reference: `scripts/studies/invocation_logging.py`

**Step 1: Introduce one shared finalize helper**

Refactor wrapper to use a single internal finalize function (name flexible) that:
- writes `metrics.json`
- renders visuals using existing `render_grid_lines_visuals(...)`
- writes `metrics_table.tex` (and optionally `metrics_table_best.tex`) with `N`-grouped rows.

Apply this finalize helper in all relevant paths:
- models-mode branch
- architecture-mode branch
- reuse-existing-recons early return path

Ensure finalize helper receives (or derives) per-model `N` metadata so table grouping works for mixed-`N` runs created via `--model-n`.

**Step 2: Keep behavior stable**

Preserve current output contract:
- no change to `metrics.json` schema
- no change to recon artifact layout
- no change to invocation logging

**Step 3: Run targeted tests**

Run:
```bash
pytest tests/test_grid_lines_compare_wrapper.py -v
```

Expected: PASS for updated and existing wrapper tests.

**Step 4: Commit integration**

```bash
git add scripts/studies/grid_lines_compare_wrapper.py tests/test_grid_lines_compare_wrapper.py
git commit -m "feat(grid-lines): auto-generate metrics tex tables in compare wrapper finalize flow"
```

---

### Task 4: Documentation + CLI Reference Alignment

**Files:**
- Modify: `docs/COMMANDS_REFERENCE.md`
- Optional Modify: `docs/workflows/pytorch.md` (if this is the canonical grid-lines torch reference section in this repo)

**Step 1: Document standard outputs**

In the grid-lines compare wrapper section, document that standard artifacts now include:
- `metrics.json`
- `visuals/compare_amp_phase.png` (and related visuals)
- `metrics_table.tex` (grouped by `N`; plus `metrics_table_best.tex` if implemented)

**Step 2: Add one reproducible command example**

Use existing style; include output directory and expected artifact list.

**Step 3: Run docs sanity check (lightweight)**

Run:
```bash
rg -n "metrics_table\\.tex|grid_lines_compare_wrapper" docs/COMMANDS_REFERENCE.md docs/workflows/pytorch.md
```

Expected: updated docs include the new artifact expectations.

**Step 4: Commit docs**

```bash
git add docs/COMMANDS_REFERENCE.md docs/workflows/pytorch.md
git commit -m "docs(grid-lines): document automatic metrics tex table artifacts"
```

---

### Task 5: Final Verification Before Completion

**Files:**
- Verify: `scripts/studies/grid_lines_compare_wrapper.py`
- Verify: `scripts/studies/metrics_tables.py`
- Verify: `tests/test_grid_lines_compare_wrapper.py`
- Verify: `docs/COMMANDS_REFERENCE.md`

**Step 1: Run required tests**

Run:
```bash
pytest tests/test_grid_lines_compare_wrapper.py -v
```

**Step 2: Run broader smoke checks (recommended)**

```bash
pytest tests/test_grid_lines_compare_wrapper.py tests/torch/test_grid_lines_torch_runner.py -v
```

**Step 3: Validate formatting/lint if project requires it**

Use repo-standard lint command if configured.

**Step 4: Final commit (if any changes remained unstaged)**

```bash
git add -A
git commit -m "chore(grid-lines): standardize compare wrapper metrics table generation"
```
