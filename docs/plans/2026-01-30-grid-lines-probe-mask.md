# Grid-Lines Probe Masking Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an optional centered disk mask (diameter in pixels) to the grid‑lines probe after upscaling, use it for diffraction simulation, and persist the masked probe in the saved NPZ datasets.

**Architecture:** Extend `GridLinesConfig` with a `probe_mask_diameter` knob (default None). Add a small mask helper (centered disk) and an `apply_probe_mask` helper in `ptycho/workflows/grid_lines_workflow.py`. In `run_grid_lines_workflow`, apply the mask after `scale_probe()` and before `configure_legacy_params()/simulate_grid_data()`. Persist the value in MetadataManager additional parameters so downstream consumers can reproduce the mask.

**Tech Stack:** Python, NumPy, existing grid‑lines workflow helpers, pytest.

---

### Task 1: Add mask helper + unit tests (TDD)

**Files:**
- Modify: `ptycho/workflows/grid_lines_workflow.py`
- Modify: `tests/test_grid_lines_workflow.py`

**Step 1: Write the failing test**

```python
# tests/test_grid_lines_workflow.py

def test_apply_probe_mask_centered_disk():
    probe = (np.ones((8, 8)) + 1j * np.ones((8, 8))).astype(np.complex64)
    masked = apply_probe_mask(probe, diameter=4)
    assert masked.shape == (8, 8)
    assert masked.dtype == np.complex64
    # Center pixel inside disk should remain nonzero
    assert masked[4, 4] != 0
    # Corner pixel outside disk should be zero
    assert masked[0, 0] == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_grid_lines_workflow.py::TestProbeHelpers::test_apply_probe_mask_centered_disk -v`

Expected: FAIL (apply_probe_mask not defined).

**Step 3: Write minimal implementation**

```python
# ptycho/workflows/grid_lines_workflow.py

def make_disk_mask(N: int, diameter: int) -> np.ndarray:
    radius = diameter / 2.0
    yy, xx = np.ogrid[:N, :N]
    cy = (N - 1) / 2.0
    cx = (N - 1) / 2.0
    return (((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2).astype(np.float32)


def apply_probe_mask(probe: np.ndarray, diameter: Optional[int]) -> np.ndarray:
    if diameter is None:
        return probe
    mask = make_disk_mask(probe.shape[0], diameter)
    return (probe * mask).astype(probe.dtype)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_grid_lines_workflow.py::TestProbeHelpers::test_apply_probe_mask_centered_disk -v`

Expected: PASS

**Step 5: Commit**

```bash
git add ptycho/workflows/grid_lines_workflow.py tests/test_grid_lines_workflow.py
git commit -m "feat: add probe masking helper for grid-lines" 
```

---

### Task 2: Wire config + apply mask in workflow (TDD)

**Files:**
- Modify: `ptycho/workflows/grid_lines_workflow.py`
- Modify: `tests/test_grid_lines_workflow.py`

**Step 1: Write failing test**

```python
# tests/test_grid_lines_workflow.py

def test_run_grid_lines_workflow_applies_mask(monkeypatch, tmp_path):
    # stub out simulation to inspect probe guess
    captured = {}
    def fake_sim(cfg, probe_np):
        captured["probe"] = probe_np
        return {"train": {}, "test": {}}

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.simulate_grid_data", fake_sim)
    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.configure_legacy_params", lambda *args, **kwargs: None)
    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.save_split_npz", lambda *args, **kwargs: None)
    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.render_grid_lines_visuals", lambda *args, **kwargs: {})

    cfg = GridLinesConfig(
        N=8,
        gridsize=1,
        output_dir=tmp_path,
        probe_npz=tmp_path / "probe.npz",
        probe_mask_diameter=4,
    )
    probe = (np.ones((8, 8)) + 1j * np.ones((8, 8))).astype(np.complex64)
    np.savez(cfg.probe_npz, probeGuess=probe)

    run_grid_lines_workflow(cfg)

    assert captured["probe"][0, 0] == 0
    assert captured["probe"][4, 4] != 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_grid_lines_workflow.py::TestProbeHelpers::test_run_grid_lines_workflow_applies_mask -v`

Expected: FAIL (probe_mask_diameter not wired).

**Step 3: Write minimal implementation**

- Add `probe_mask_diameter: Optional[int] = None` to `GridLinesConfig`.
- In `run_grid_lines_workflow()`, after `scale_probe(...)`:
  - `probe_scaled = apply_probe_mask(probe_scaled, cfg.probe_mask_diameter)`
- Update metadata creation in `save_split_npz` (or metadata call site) to include `probe_mask_diameter` in `additional_parameters`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_grid_lines_workflow.py::TestProbeHelpers::test_run_grid_lines_workflow_applies_mask -v`

Expected: PASS

**Step 5: Commit**

```bash
git add ptycho/workflows/grid_lines_workflow.py tests/test_grid_lines_workflow.py
git commit -m "feat: apply optional probe mask in grid-lines workflow"
```

---

### Task 3: Metadata persistence for mask diameter (TDD)

**Files:**
- Modify: `ptycho/workflows/grid_lines_workflow.py`
- Modify: `tests/test_grid_lines_workflow.py`

**Step 1: Write failing test**

```python
# tests/test_grid_lines_workflow.py

def test_metadata_includes_probe_mask_diameter(monkeypatch, tmp_path):
    captured = {}
    def fake_save_with_metadata(path, payload, metadata):
        captured["metadata"] = metadata

    monkeypatch.setattr("ptycho.metadata.MetadataManager.save_with_metadata", fake_save_with_metadata)
    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.simulate_grid_data", lambda *args, **kwargs: {"train": {}, "test": {}})
    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.configure_legacy_params", lambda *args, **kwargs: None)

    cfg = GridLinesConfig(
        N=8,
        gridsize=1,
        output_dir=tmp_path,
        probe_npz=tmp_path / "probe.npz",
        probe_mask_diameter=4,
    )
    probe = (np.ones((8, 8)) + 1j * np.ones((8, 8))).astype(np.complex64)
    np.savez(cfg.probe_npz, probeGuess=probe)

    run_grid_lines_workflow(cfg)

    assert captured["metadata"]["additional_parameters"]["probe_mask_diameter"] == 4
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_grid_lines_workflow.py::TestProbeHelpers::test_metadata_includes_probe_mask_diameter -v`

Expected: FAIL (metadata missing field).

**Step 3: Write minimal implementation**

Add `probe_mask_diameter=cfg.probe_mask_diameter` to the `MetadataManager.create_metadata(...)` call for train/test dataset persistence.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_grid_lines_workflow.py::TestProbeHelpers::test_metadata_includes_probe_mask_diameter -v`

Expected: PASS

**Step 5: Commit**

```bash
git add ptycho/workflows/grid_lines_workflow.py tests/test_grid_lines_workflow.py
git commit -m "feat: persist probe mask diameter in grid-lines metadata"
```

---

### Task 4: Full test pass (targeted)

**Files:**
- Verify: `tests/test_grid_lines_workflow.py`

**Step 1: Run targeted test module**

Run: `pytest tests/test_grid_lines_workflow.py -q`

Expected: PASS

**Step 2: Commit (if any final fixes)**

```bash
git add ptycho/workflows/grid_lines_workflow.py tests/test_grid_lines_workflow.py
git commit -m "test: grid-lines probe mask coverage"
```

---

Plan complete and saved to `docs/plans/2026-01-30-grid-lines-probe-mask.md`.

Two execution options:

1. Subagent‑Driven (this session) — I dispatch a fresh subagent per task, review between tasks, fast iteration
2. Parallel Session (separate) — Open new session with executing‑plans, batch execution with checkpoints

Which approach? 
