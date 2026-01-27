# Grid Lines Compare Wrapper Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a thin wrapper that runs the TF grid-lines workflow (cnn + baseline), runs Torch FNO/hybrid on the cached dataset, and writes a single merged comparison metrics JSON.

**Architecture:** Implement a new CLI under `scripts/studies/` that orchestrates: (1) `run_grid_lines_workflow()` to generate the dataset and TF metrics, (2) `run_grid_lines_torch()` for each requested Torch architecture using the cached `train/test.npz`, and (3) merge metrics into `output_dir/metrics.json` while keeping per‑run metrics in `output_dir/runs/pinn_<arch>/metrics.json`. Keep the wrapper parameterized for future custom runs.

**Tech Stack:** Python, pytest, NumPy, existing TF/Torch workflow modules.

---

### Task 1: Create wrapper CLI for full comparison

**Files:**
- Create: `scripts/studies/grid_lines_compare_wrapper.py`
- Test: `tests/test_grid_lines_compare_wrapper.py`

**Step 1: Write the failing test**

```python
def test_wrapper_merges_metrics(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare
    from pathlib import Path
    import json

    # Fake TF workflow result
    def fake_tf_run(cfg):
        datasets_dir = cfg.output_dir / "datasets" / f"N{cfg.N}" / f"gs{cfg.gridsize}"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        (datasets_dir / "train.npz").write_bytes(b"stub")
        (datasets_dir / "test.npz").write_bytes(b"stub")
        (cfg.output_dir / "metrics.json").write_text(json.dumps({
            "pinn": {"mse": 0.1},
            "baseline": {"mse": 0.2},
        }))
        return {"train_npz": str(datasets_dir / "train.npz"), "test_npz": str(datasets_dir / "test.npz")}

    # Fake Torch runner result
    def fake_torch_run(cfg):
        run_dir = cfg.output_dir / "runs" / f"pinn_{cfg.architecture}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text(json.dumps({"mse": 0.3}))
        return {"metrics": {"mse": 0.3}}

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.run_grid_lines_workflow", fake_tf_run)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)

    cfg = {
        "N": 64,
        "gridsize": 1,
        "output_dir": tmp_path,
        "architectures": ("cnn", "baseline", "fno", "hybrid"),
    }
    result = run_grid_lines_compare(**cfg)
    merged = json.loads((tmp_path / "metrics.json").read_text())
    assert "pinn" in merged
    assert "baseline" in merged
    assert "pinn_fno" in merged
    assert "pinn_hybrid" in merged
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_merges_metrics -v`  
Expected: FAIL (wrapper not implemented).

**Step 3: Write minimal implementation**

```python
# scripts/studies/grid_lines_compare_wrapper.py
def run_grid_lines_compare(...):
    # 1) run_grid_lines_workflow if cnn/baseline requested
    # 2) find train/test npz under output_dir/datasets/N{N}/gs{gridsize}
    # 3) run_grid_lines_torch for fno/hybrid
    # 4) merge metrics into output_dir/metrics.json
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_merges_metrics -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_compare_wrapper.py tests/test_grid_lines_compare_wrapper.py
git commit -m "feat(studies): add grid-lines compare wrapper"
```

---

### Task 2: Add CLI options for future parametrization

**Files:**
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Test: `tests/test_grid_lines_compare_wrapper.py`

**Step 1: Write the failing test**

```python
def test_wrapper_accepts_architecture_list(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import parse_args
    args = parse_args(["--N", "64", "--gridsize", "1", "--output-dir", str(tmp_path),
                       "--architectures", "cnn,fno"])
    assert args.architectures == ("cnn", "fno")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_accepts_architecture_list -v`  
Expected: FAIL (parser missing).

**Step 3: Write minimal implementation**

```python
def parse_args(argv=None):
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--architectures", default="cnn,baseline,fno,hybrid")
    ...
    args = parser.parse_args(argv)
    args.architectures = tuple(a.strip() for a in args.architectures.split(",") if a.strip())
    return args
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_accepts_architecture_list -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_compare_wrapper.py tests/test_grid_lines_compare_wrapper.py
git commit -m "feat(studies): support architecture selection in wrapper"
```

---

### Task 3: Documentation update

**Files:**
- Modify: `docs/plans/2026-01-27-grid-lines-workflow.md`

**Step 1: Add wrapper usage note**
- Mention `scripts/studies/grid_lines_compare_wrapper.py` as the canonical entry point for cnn+baseline+fno+hybrid comparisons.
- Note that metrics are merged into `output_dir/metrics.json`, while per‑run metrics remain in `output_dir/runs/pinn_<arch>/metrics.json`.

**Step 2: Commit**

```bash
git add docs/plans/2026-01-27-grid-lines-workflow.md
git commit -m "docs: add grid-lines compare wrapper entry point"
```

---

### Task 4: Verification & Evidence

**Required tests (per TESTING_GUIDE.md):**
- `pytest tests/test_grid_lines_compare_wrapper.py -v`

**Commands:**
```bash
pytest tests/test_grid_lines_compare_wrapper.py -v
```

**Evidence capture:**
- Save logs under `.artifacts/grid_lines_compare_wrapper/`
- Add a short note in this plan pointing to the log paths.

**Evidence (2026-01-27):**
- `.artifacts/grid_lines_compare_wrapper/pytest_grid_lines_compare_wrapper.log`

---

## Execution Handoff
Plan complete and saved to `docs/plans/2026-01-27-grid-lines-compare-wrapper.md`.

Two execution options:
1. **Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks
2. **Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
