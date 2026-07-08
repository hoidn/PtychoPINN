# Grid Lines Probe Padding + Phase Extrapolation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make grid_lines probe scaling default to nearest-neighbor amplitude padding with quadratic phase extrapolation, while keeping cubic interpolation as an opt-in mode.

**Architecture:** Extend `GridLinesConfig` with a probe scaling mode, add a new pad+extrapolate path in `scale_probe()`, and expose the mode via grid_lines CLIs. Keep interpolation logic untouched for compatibility. Update unit tests to cover both modes.

**Tech Stack:** NumPy, SciPy (existing interpolate helper), pytest, grid_lines workflow + study scripts.

---

## Task 1: Add failing tests for pad+extrapolate and mode selection

**Files:**
- Modify: `tests/test_grid_lines_workflow.py`

**Step 1: Add a pad+extrapolate test (expects edge-padded amplitude + near-zero phase)**

```python
def test_scale_probe_pad_extrapolate_pads_amplitude_and_extrapolates_phase(self):
    amplitude = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    phase = np.zeros_like(amplitude)
    probe = (amplitude * np.exp(1j * phase)).astype(np.complex64)

    scaled = scale_probe(
        probe,
        target_N=4,
        smoothing_sigma=0.0,
        scale_mode="pad_extrapolate",
    )

    expected_amp = np.pad(amplitude, pad_width=1, mode="edge")
    np.testing.assert_allclose(np.abs(scaled), expected_amp, atol=1e-6)
    assert np.allclose(np.angle(scaled), 0.0, atol=1e-5)
```

**Step 2: Update the interpolation test to call interpolate mode explicitly**

```python
scaled = scale_probe(probe, target_N=8, smoothing_sigma=0.5, scale_mode="interpolate")
```

**Step 3: Add a guard test for pad_extrapolate when target_N < input size**

```python
with pytest.raises(ValueError, match="pad_extrapolate requires target_N >= probe size"):
    scale_probe(probe, target_N=4, smoothing_sigma=0.0, scale_mode="pad_extrapolate")
```

**Step 4: Run tests to confirm RED**

Run: `pytest tests/test_grid_lines_workflow.py -q`
Expected: FAIL (new arguments not supported yet).

**Step 5: Commit**

```bash
git add tests/test_grid_lines_workflow.py
git commit -m "test: cover pad/extrapolate probe scaling"
```

---

## Task 2: Implement pad+extrapolate scaling in grid_lines workflow

**Files:**
- Modify: `ptycho/workflows/grid_lines_workflow.py`

**Step 1: Extend GridLinesConfig with scale_mode**

```python
probe_scale_mode: str = "pad_extrapolate"  # or "interpolate"
```

**Step 2: Add helper to fit quadratic phase vs r^2 and extrapolate**

```python
def _fit_quadratic_phase(phase: np.ndarray) -> tuple[float, float]:
    # Fit phase ≈ a*r^2 + b in least-squares sense.
```

**Step 3: Add helper to pad amplitude with nearest-neighbor edge mode**

```python
def _pad_amplitude(amplitude: np.ndarray, target_N: int) -> np.ndarray:
    # Use np.pad(..., mode="edge") with symmetric pad widths.
```

**Step 4: Update scale_probe() signature + logic**

```python
def scale_probe(
    probe: np.ndarray,
    target_N: int,
    smoothing_sigma: float,
    scale_mode: str = "pad_extrapolate",
) -> np.ndarray:
    if scale_mode == "interpolate":
        # Existing interpolate_array + smooth_complex_array
    elif scale_mode == "pad_extrapolate":
        # Guard: target_N >= probe.shape[0]
        # amplitude pad + quadratic phase extrapolation
        # optional smooth_complex_array
    else:
        raise ValueError("Unknown scale_mode...")
```

**Step 5: Wire scale_mode into run_grid_lines_workflow()**

```python
probe_scaled = scale_probe(
    probe_guess,
    cfg.N,
    cfg.probe_smoothing_sigma,
    scale_mode=cfg.probe_scale_mode,
)
```

**Step 6: Run tests (GREEN)**

Run: `pytest tests/test_grid_lines_workflow.py -q`
Expected: PASS.

**Step 7: Commit**

```bash
git add ptycho/workflows/grid_lines_workflow.py
git commit -m "feat(grid-lines): add pad/extrapolate probe scaling"
```

---

## Task 3: Expose probe scale mode in grid_lines CLIs

**Files:**
- Modify: `scripts/studies/grid_lines_workflow.py`
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`

**Step 1: Add CLI flag**

```python
parser.add_argument(
    "--probe-scale-mode",
    choices=["pad_extrapolate", "interpolate"],
    default="pad_extrapolate",
)
```

**Step 2: Pass into GridLinesConfig**

```python
probe_scale_mode=args.probe_scale_mode,
```

**Step 3: Update compare wrapper signature + wiring**

- Add `probe_scale_mode: str = "pad_extrapolate"` param to `run_grid_lines_compare()`.
- Thread through `GridLinesConfig`.
- Add CLI flag in `parse_args()` with same choices/default.

**Step 4: Run CLI smoke (no execution)**

Run: `python scripts/studies/grid_lines_workflow.py --help`
Expected: shows `--probe-scale-mode` with default `pad_extrapolate`.

Run: `python scripts/studies/grid_lines_compare_wrapper.py --help`
Expected: shows `--probe-scale-mode` with default `pad_extrapolate`.

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_workflow.py scripts/studies/grid_lines_compare_wrapper.py
git commit -m "feat(grid-lines): expose probe scale mode in CLIs"
```

---

## Task 4: Update documentation note for probe scaling behavior

**Files:**
- Modify: `docs/plans/2026-01-27-grid-lines-workflow.md`

**Step 1: Update the “Probe scaling” bullet**

Change from interpolation-only to:
- Default: pad+quadratic phase extrapolation (edge amplitude padding + quadratic phase fit)
- Option: cubic interpolation (`--probe-scale-mode interpolate`)

**Step 2: Commit**

```bash
git add docs/plans/2026-01-27-grid-lines-workflow.md
git commit -m "docs(grid-lines): document probe scale modes"
```

---

## Task 5: Full verification (per TESTING_GUIDE.md)

**Step 1: Run unit tests**

Run: `pytest tests/test_grid_lines_workflow.py -q`
Expected: PASS.

**Step 2: Run integration marker (required for workflow changes)**

Run: `pytest -v -m integration`
Expected: PASS.

**Step 3: Archive logs**

Save logs under `.artifacts/` and link from the active plan summary per repo conventions.

---

## Notes & Edge Cases

- `pad_extrapolate` should only accept `target_N >= probe.shape[0]`.
- For quadratic fit, use unwrapped phase on the original probe, fit `phase ≈ a*r^2 + b`, and wrap output to `[-π, π]` after extrapolation.
- Use `np.pad(..., mode="edge")` for nearest-neighbor amplitude padding.
- Keep interpolation path unchanged to preserve legacy behavior.
