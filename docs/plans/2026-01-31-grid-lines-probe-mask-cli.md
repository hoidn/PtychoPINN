# Grid Lines Probe Mask CLI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose `probe_mask_diameter` on grid-lines CLIs so masked probes can be generated and evaluated via the standard grid_lines workflow.

**Architecture:** Add a single CLI flag (`--probe-mask-diameter`) to both `grid_lines_workflow.py` and `grid_lines_compare_wrapper.py`, thread it into `GridLinesConfig`, and validate via lightweight unit tests. Keep masking in `ptycho.workflows.grid_lines_workflow.run_grid_lines_workflow`, which already applies the mask during probe prep.

**Tech Stack:** Python, argparse, pytest

---

## Preconditions / Notes

- **Baseline tests currently failing/time out** in this worktree (`pytest -q` timed out with failures). Use targeted tests only unless asked to debug the existing failures.
- Requirement: keep evaluation on the **standard grid_lines path** (no custom stitching).

---

### Task 1: Add CLI flag to grid_lines_workflow and test config wiring

**Files:**
- Modify: `scripts/studies/grid_lines_workflow.py`
- Modify: `tests/test_grid_lines_workflow.py`

**Step 1: Write the failing test**

Add a test that asserts the CLI flag is parsed and wired into `GridLinesConfig`.

```python
# tests/test_grid_lines_workflow.py

def test_grid_lines_cli_probe_mask_diameter(tmp_path):
    from scripts.studies import grid_lines_workflow as cli

    args = cli.parse_args([
        "--N", "64",
        "--gridsize", "1",
        "--output-dir", str(tmp_path),
        "--probe-mask-diameter", "64",
    ])
    cfg = cli.build_config(args)
    assert cfg.probe_mask_diameter == 64
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_grid_lines_workflow.py::test_grid_lines_cli_probe_mask_diameter -v`
Expected: FAIL (missing parse_args/build_config or flag not wired).

**Step 3: Write minimal implementation**

In `scripts/studies/grid_lines_workflow.py`:
- Add a `parse_args(argv=None)` helper that returns `argparse.Namespace`.
- Add a `build_config(args)` helper that returns `GridLinesConfig`.
- Add argparse flag:
  ```python
  parser.add_argument("--probe-mask-diameter", type=int, default=None)
  ```
- Pass `probe_mask_diameter=args.probe_mask_diameter` into `GridLinesConfig`.
- Update `main()` to call `args = parse_args()` and `cfg = build_config(args)`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_grid_lines_workflow.py::test_grid_lines_cli_probe_mask_diameter -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_workflow.py tests/test_grid_lines_workflow.py
git commit -m "feat(grid-lines): add probe mask flag to workflow CLI"
```

---

### Task 2: Add CLI flag to grid_lines_compare_wrapper and test passthrough

**Files:**
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Modify: `tests/test_grid_lines_compare_wrapper.py`

**Step 1: Write the failing test**

Add a test that ensures the new flag is parsed and passed into `GridLinesConfig`.

```python
# tests/test_grid_lines_compare_wrapper.py

def test_compare_wrapper_probe_mask_diameter_passthrough(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare, parse_args

    args = parse_args([
        "--N", "64",
        "--gridsize", "1",
        "--output-dir", str(tmp_path),
        "--probe-mask-diameter", "64",
    ])
    assert args.probe_mask_diameter == 64

    captured = {}
    def fake_tf_run(cfg):
        captured["probe_mask_diameter"] = cfg.probe_mask_diameter
        datasets_dir = cfg.output_dir / "datasets" / f"N{cfg.N}" / f"gs{cfg.gridsize}"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        (datasets_dir / "train.npz").write_bytes(b"stub")
        (datasets_dir / "test.npz").write_bytes(b"stub")
        (cfg.output_dir / "metrics.json").write_text("{}")
        return {"train_npz": str(datasets_dir / "train.npz"), "test_npz": str(datasets_dir / "test.npz")}

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.run_grid_lines_workflow", fake_tf_run)

    run_grid_lines_compare(
        N=64,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("cnn",),
        probe_mask_diameter=64,
    )

    assert captured["probe_mask_diameter"] == 64
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_grid_lines_compare_wrapper.py::test_compare_wrapper_probe_mask_diameter_passthrough -v`
Expected: FAIL (arg missing / not threaded).

**Step 3: Write minimal implementation**

In `scripts/studies/grid_lines_compare_wrapper.py`:
- Add argparse flag:
  ```python
  parser.add_argument("--probe-mask-diameter", type=int, default=None)
  ```
- Thread into `run_grid_lines_compare` signature and pass into `GridLinesConfig` when constructing `tf_cfg`.
- Include `probe_mask_diameter=args.probe_mask_diameter` in CLI parsing call.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_grid_lines_compare_wrapper.py::test_compare_wrapper_probe_mask_diameter_passthrough -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_compare_wrapper.py tests/test_grid_lines_compare_wrapper.py
git commit -m "feat(grid-lines): add probe mask flag to compare wrapper"
```

---

### Task 3: Update CLI documentation

**Files:**
- Modify: `scripts/studies/README.md`
- Modify: `docs/COMMANDS_REFERENCE.md`
- Modify: `docs/workflows/pytorch.md`

**Step 1: Update docs**

- Add `--probe-mask-diameter` option to the grid_lines CLI sections.
- Update any example commands to include the new flag where relevant.

**Step 2: Commit**

```bash
git add scripts/studies/README.md docs/COMMANDS_REFERENCE.md docs/workflows/pytorch.md
git commit -m "docs(grid-lines): document probe mask CLI"
```

---

### Task 4: Validation via standard grid_lines runs (after merge to main repo)

**Files:**
- None (runtime validation)

**Step 1: End-to-end N=64 run (standard path)**

Run in **main repo (after merge)**:

```bash
PYTHONPATH=. python scripts/studies/grid_lines_compare_wrapper.py \
  --N 64 \
  --gridsize 1 \
  --output-dir outputs/grid_lines_n64_hybrid_resnet_mask64_lr2e4_plateau \
  --architectures hybrid_resnet \
  --set-phi \
  --probe-mask-diameter 64 \
  --probe-source custom \
  --probe-scale-mode pad_extrapolate \
  --torch-epochs 20 \
  --torch-learning-rate 2e-4 \
  --torch-scheduler ReduceLROnPlateau \
  --nepochs 1
```

Expected artifacts:
- `outputs/grid_lines_n64_hybrid_resnet_mask64_lr2e4_plateau/metrics.json`
- `outputs/grid_lines_n64_hybrid_resnet_mask64_lr2e4_plateau/visuals/*`

**Step 2: Longer stability run (pick one)**

Option A (N=64):
```bash
PYTHONPATH=. python scripts/studies/grid_lines_compare_wrapper.py \
  --N 64 \
  --gridsize 1 \
  --output-dir outputs/grid_lines_n64_hybrid_resnet_mask64_lr2e4_plateau_e40 \
  --architectures hybrid_resnet \
  --set-phi \
  --probe-mask-diameter 64 \
  --probe-source custom \
  --probe-scale-mode pad_extrapolate \
  --torch-epochs 40 \
  --torch-learning-rate 2e-4 \
  --torch-scheduler ReduceLROnPlateau \
  --nepochs 1
```

Option B (N=128):
```bash
PYTHONPATH=. python scripts/studies/grid_lines_compare_wrapper.py \
  --N 128 \
  --gridsize 1 \
  --output-dir outputs/grid_lines_n128_hybrid_resnet_mask64_lr2e4_plateau_e20 \
  --architectures hybrid_resnet \
  --set-phi \
  --probe-mask-diameter 64 \
  --probe-source custom \
  --probe-scale-mode pad_extrapolate \
  --torch-epochs 20 \
  --torch-learning-rate 2e-4 \
  --torch-scheduler ReduceLROnPlateau \
  --nepochs 1
```

Capture:
- Metrics JSON and recon visuals in each output directory.

---

### Task 5: Final test run + merge prep

**Files:**
- None (tests)

**Step 1: Run targeted tests**

```bash
pytest tests/test_grid_lines_workflow.py::test_grid_lines_cli_probe_mask_diameter -v
pytest tests/test_grid_lines_compare_wrapper.py::test_compare_wrapper_probe_mask_diameter_passthrough -v
```

**Step 2: Report baseline test failures**

Note: Full `pytest -q` timed out with failures earlier. Do not claim full-suite success unless it passes.

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-01-31-grid-lines-probe-mask-cli.md`.

Two execution options:

1. Subagent-Driven (this session) — I dispatch a fresh subagent per task, review between tasks.
2. Parallel Session (separate) — Open a new session with executing-plans and batch execution.

Which approach?
