# Probe Mischaracterization Probe Context Panels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not create a worktree for this initiative.

**Goal:** Add paper-facing probe context panels to the probe mischaracterization stress figure so the figure shows the unperturbed probe, an amplitude-blurred amplitude example, and a phase-noise phase example alongside the quantitative stress curves.

**Architecture:** Keep the plotting path study-local in `scripts/studies/probe_mischaracterization_stress_test.py`. Extend the existing `write_stress_figure(...)` function to compose separate panels, not insets: probe amplitude thumbnails using the amplitude-blur corruption, probe phase thumbnails using the phase-noise corruption, and the current amplitude SSIM/PSNR stress curves in a bottom row. Reuse the already-written `probes/true_probe.npz` and `conditions/<condition_id>/assumed_probe.npz` artifacts; do not rerun the stress study just to build the figure.

**Tech Stack:** Python, NumPy, matplotlib, pytest, existing probe stress study artifacts, and existing paper asset export in `export_paper_assets(...)`.

---

## Initiative

- ID: `probe-mischaracterization-probe-context-panels`
- Status: done
- Source request: Use separate panels, not insets, to show ground-truth probe, amplitude-blurred amplitude, and phase-noise phase for the paper figure.
- Primary implementation file: `scripts/studies/probe_mischaracterization_stress_test.py`
- Primary tests: `tests/studies/test_probe_mischaracterization_stress_test.py`
- Existing source run to regenerate from, if needed: `.artifacts/revision_studies/probe_mischaracterization/full_process_isolated_20260413T020739Z/`
- Existing paper figure target: `/home/ollie/Documents/ptychopinnpaper2/figures/probe_mischaracterization_stress.png`

## Compliance Matrix

- [ ] **Reviewer-facing visual contract:** Use explicit neighboring panels, not metric-plot insets, so the metric curves stay readable.
- [ ] **Scientific scope contract:** The visual panels are explanatory context only. They must not change the numeric metrics, perturbation grid, mild-perturbation gate, baseline comparability gate, or reviewer claims.
- [ ] **Representative condition contract:** The amplitude example defaults to `amplitude_blur_sigma_px_1p0`; the phase example defaults to `phase_noise_sigma_rad_0p2pi_seed17`. Do not show phase from the amplitude-blur probe unless no phase-noise artifact exists, and record any fallback.
- [ ] **Complex-probe display contract:** Show true amplitude against amplitude-blurred amplitude, and true phase against phase-noise phase. Use a shared amplitude scale across the amplitude pair and a fixed `[-pi, pi]` phase scale across the phase pair.
- [ ] **Artifact contract:** The paper export still writes `figures/probe_mischaracterization_stress.png`; the figure now contains probe-context panels plus stress curves. Manifest metadata records the amplitude and phase probe conditions used.
- [ ] **Finding ID:** `GRIDLINES-PROBE-PIPELINE-001` - preserve the existing probe transform provenance, such as `smooth:0.5|pad:64`; do not add a new probe preprocessing path.
- [ ] **Finding ID:** `TF-REPEATED-MODEL-OOM-001` - do not rerun training just to update the visualization; regenerate from existing artifacts when possible.

## File Structure

- Modify: `scripts/studies/probe_mischaracterization_stress_test.py`
  - Add `DEFAULT_PROBE_FIGURE_AMPLITUDE_CONDITION_IDS = ("amplitude_blur_sigma_px_1p0",)` and `DEFAULT_PROBE_FIGURE_PHASE_CONDITION_IDS = ("phase_noise_sigma_rad_0p2pi_seed17",)`.
  - Add a small `ProbeFigureContext` dataclass or dictionary with `true_probe`, `amplitude_probe`, `phase_probe`, `amplitude_condition_id`, `phase_condition_id`, and per-channel fallback flags.
  - Add `load_probe_figure_context(output_root, amplitude_condition_ids=..., phase_condition_ids=...)`.
  - Add `draw_probe_context_panels(...)` or keep the rendering inline if it stays concise.
  - Update `write_stress_figure(...)` to build a composite figure when probe context is available and fall back to the current metrics-only figure when it is not.
  - Update the main manifest with `stress_figure_visual_context`.
  - Keep `save_probe_visuals(...)` as standalone artifact generation; do not make the paper figure depend on reading those PNGs.

- Modify: `tests/studies/test_probe_mischaracterization_stress_test.py`
  - Add focused tests for amplitude/phase probe context loading and composite figure generation.
  - Extend existing stress-figure tests; do not assert exact prompt text or brittle pixel-perfect image contents.

- Optional paper-side file after code verification:
  - Update only if the manuscript currently describes the figure as metrics-only: `/home/ollie/Documents/ptychopinnpaper2/...`

## Context Priming

Read before edits:

- `docs/index.md`
- `docs/findings.md`
- `docs/TESTING_GUIDE.md`
- `docs/development/TEST_SUITE_INDEX.md`
- `scripts/studies/probe_mischaracterization_stress_test.py`
- `tests/studies/test_probe_mischaracterization_stress_test.py`

Relevant existing behavior:

- `save_probe_visuals(...)` already writes per-condition amplitude/phase PNGs.
- `persist_true_measurements(...)` writes `probes/true_probe.npz`.
- Each condition writes `conditions/<condition_id>/assumed_probe.npz`.
- `write_stress_figure(...)` currently writes `figures/probe_mischaracterization_stress.png` and is the right insertion point for the composite figure.

## Proposed Figure Layout

Use one paper figure file with separate panels:

- Top row:
  - true probe amplitude
  - amplitude-blurred probe amplitude
  - true probe phase
  - phase-noise probe phase
- Bottom row:
  - amplitude SSIM vs perturbation magnitude
  - amplitude PSNR vs perturbation magnitude

Implementation detail:

```python
fig = plt.figure(figsize=(12, 7), constrained_layout=True)
grid = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.25])
probe_axes = [fig.add_subplot(grid[0, col]) for col in range(4)]
ssim_axis = fig.add_subplot(grid[1, 0:2])
psnr_axis = fig.add_subplot(grid[1, 2:4])
```

Use labels that are short enough for the paper:

- `True amp`
- `<amplitude_condition_id> amp`
- `True phase`
- `<phase_condition_id> phase`

Prefer the full condition ID in manifest metadata. If figure titles get too long, format the title as `Amplitude blur 1.0 phase` or `Phase noise 0.2 pi phase` and keep the exact condition ID in the caption/data payload.

## Task 1: Add Probe Context Selection Tests

**Files:**
- Modify: `tests/studies/test_probe_mischaracterization_stress_test.py`

- [ ] **Step 1: Write a failing preferred-condition test**

Add:

```python
def test_load_probe_figure_context_uses_preferred_corrupted_probe(tmp_path):
    from scripts.studies import probe_mischaracterization_stress_test as study

    true_probe = _toy_probe()
    amplitude_corrupted = (0.8 * true_probe).astype(np.complex64)
    phase_corrupted = (true_probe * np.exp(0.25j)).astype(np.complex64)
    (tmp_path / "probes").mkdir()
    amplitude_dir = tmp_path / "conditions" / "amplitude_blur_sigma_px_1p0"
    amplitude_dir.mkdir(parents=True)
    phase_dir = tmp_path / "conditions" / "phase_noise_sigma_rad_0p2pi_seed17"
    phase_dir.mkdir(parents=True)
    np.savez(tmp_path / "probes" / "true_probe.npz", probe=true_probe)
    np.savez(amplitude_dir / "assumed_probe.npz", probe=amplitude_corrupted)
    np.savez(phase_dir / "assumed_probe.npz", probe=phase_corrupted)

    context = study.load_probe_figure_context(tmp_path)

    assert context is not None
    assert context["amplitude_condition_id"] == "amplitude_blur_sigma_px_1p0"
    assert context["phase_condition_id"] == "phase_noise_sigma_rad_0p2pi_seed17"
    assert context["amplitude_fallback_used"] is False
    assert context["phase_fallback_used"] is False
    assert np.array_equal(context["true_probe"], true_probe)
    assert np.array_equal(context["amplitude_probe"], amplitude_corrupted)
    assert np.array_equal(context["phase_probe"], phase_corrupted)
```

- [ ] **Step 2: Run the failing preferred-condition test**

Run:

```bash
python -m pytest tests/studies/test_probe_mischaracterization_stress_test.py::test_load_probe_figure_context_uses_preferred_corrupted_probe -q
```

Expected: fail because `load_probe_figure_context(...)` does not exist.

- [ ] **Step 3: Write a fallback test**

Add:

```python
def test_load_probe_figure_context_falls_back_to_available_nonbaseline_probe(tmp_path):
    from scripts.studies import probe_mischaracterization_stress_test as study

    true_probe = _toy_probe()
    corrupted = (true_probe * np.exp(0.10j)).astype(np.complex64)
    (tmp_path / "probes").mkdir()
    condition_dir = tmp_path / "conditions" / "phase_noise_sigma_rad_0p1pi_seed11"
    condition_dir.mkdir(parents=True)
    np.savez(tmp_path / "probes" / "true_probe.npz", probe=true_probe)
    np.savez(condition_dir / "assumed_probe.npz", probe=corrupted)

    context = study.load_probe_figure_context(tmp_path)

    assert context is not None
    assert context["amplitude_condition_id"] == "phase_noise_sigma_rad_0p1pi_seed11"
    assert context["phase_condition_id"] == "phase_noise_sigma_rad_0p1pi_seed11"
    assert context["amplitude_fallback_used"] is True
    assert context["phase_fallback_used"] is True
```

- [ ] **Step 4: Run the failing fallback test**

Run:

```bash
python -m pytest tests/studies/test_probe_mischaracterization_stress_test.py -k "load_probe_figure_context" -q
```

Expected: fail because context loading is not implemented.

## Task 2: Implement Probe Context Loading

**Files:**
- Modify: `scripts/studies/probe_mischaracterization_stress_test.py`
- Test: `tests/studies/test_probe_mischaracterization_stress_test.py`

- [ ] **Step 1: Add constants and context loader**

Implement near the existing figure constants:

```python
DEFAULT_PROBE_FIGURE_AMPLITUDE_CONDITION_IDS = (
    "amplitude_blur_sigma_px_1p0",
)
DEFAULT_PROBE_FIGURE_PHASE_CONDITION_IDS = (
    "phase_noise_sigma_rad_0p2pi_seed17",
)
```

Implement near `_probe_from_npz(...)` or near the plotting helpers. The loader should select amplitude and phase probes separately:

```python
def load_probe_figure_context(
    output_root: Path,
    amplitude_condition_ids: tuple[str, ...] = DEFAULT_PROBE_FIGURE_AMPLITUDE_CONDITION_IDS,
    phase_condition_ids: tuple[str, ...] = DEFAULT_PROBE_FIGURE_PHASE_CONDITION_IDS,
) -> dict[str, Any] | None:
    # Select amplitude from amplitude blur and phase from phase noise.
    ...
    return {
        "true_probe": np.asarray(_probe_from_npz(true_probe_path), dtype=np.complex64),
        "amplitude_condition_id": amplitude_condition_id,
        "amplitude_probe": np.asarray(_probe_from_npz(amplitude_path), dtype=np.complex64),
        "amplitude_fallback_used": amplitude_fallback_used,
        "phase_condition_id": phase_condition_id,
        "phase_probe": np.asarray(_probe_from_npz(phase_path), dtype=np.complex64),
        "phase_fallback_used": phase_fallback_used,
    }
```

- [ ] **Step 2: Run probe context tests**

Run:

```bash
python -m pytest tests/studies/test_probe_mischaracterization_stress_test.py -k "load_probe_figure_context" -q
```

Expected: pass.

## Task 3: Add Composite Figure Rendering

**Files:**
- Modify: `scripts/studies/probe_mischaracterization_stress_test.py`
- Test: `tests/studies/test_probe_mischaracterization_stress_test.py`

- [ ] **Step 1: Add a failing composite-figure smoke test**

Add:

```python
def test_write_stress_figure_with_probe_context_writes_composite_png(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    from PIL import Image
    from scripts.studies import probe_mischaracterization_stress_test as study

    true_probe = _toy_probe()
    corrupted = (true_probe * np.exp(0.25j)).astype(np.complex64)
    (tmp_path / "probes").mkdir()
    condition_dir = tmp_path / "conditions" / "amplitude_blur_sigma_px_1p0"
    condition_dir.mkdir(parents=True)
    np.savez(tmp_path / "probes" / "true_probe.npz", probe=true_probe)
    np.savez(condition_dir / "assumed_probe.npz", probe=corrupted)

    condition_results = {
        "baseline": {"condition_id": "baseline", "perturbation_type": "baseline", "status": "ok", "amp_ssim": 0.91, "amp_psnr": 68.0},
        "amplitude_blur_sigma_px_1p0": {"condition_id": "amplitude_blur_sigma_px_1p0", "perturbation_type": "amplitude_blur_sigma_px", "value": 1.0, "status": "ok", "amp_ssim": 0.86, "amp_psnr": 67.0},
    }

    figure_path = study.write_stress_figure(tmp_path, condition_results)

    assert figure_path == tmp_path / "figures" / "probe_mischaracterization_stress.png"
    with Image.open(figure_path) as image:
        assert image.size[0] > image.size[1]
        assert image.size[1] >= 1000
```

Expected before implementation: may still pass path existence but should fail the height assertion if the figure remains metrics-only at 2200x800.

- [ ] **Step 2: Implement `draw_probe_context_panels(...)`**

Implement a narrow helper:

```python
def draw_probe_context_panels(fig, axes: list[Any], context: dict[str, Any]) -> None:
    true_probe = context["true_probe"]
    amplitude_probe = context["amplitude_probe"]
    phase_probe = context["phase_probe"]
    amplitude_label = format_probe_condition_label(context["amplitude_condition_id"])
    phase_label = format_probe_condition_label(context["phase_condition_id"])
    amp_vmax = float(max(np.abs(true_probe).max(initial=0.0), np.abs(amplitude_probe).max(initial=0.0)))
    images = [
        (np.abs(true_probe), "True amp", "viridis", 0.0, amp_vmax),
        (np.abs(amplitude_probe), f"{amplitude_label} amp", "viridis", 0.0, amp_vmax),
        (np.angle(true_probe), "True phase", "twilight", -math.pi, math.pi),
        (np.angle(phase_probe), f"{phase_label} phase", "twilight", -math.pi, math.pi),
    ]
    for axis, (image, title, cmap, vmin, vmax) in zip(axes, images):
        axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
```

Add a small `format_probe_condition_label(condition_id)` helper so the fallback phase-noise case does not inherit amplitude-blur titles. Unit-test at least the two defaults:

```python
assert study.format_probe_condition_label("amplitude_blur_sigma_px_1p0") == "Amplitude blur 1.0"
assert study.format_probe_condition_label("phase_noise_sigma_rad_0p2pi_seed17") == "Phase noise 0.2 pi"
```

- [ ] **Step 3: Update `write_stress_figure(...)` layout**

Change `write_stress_figure(...)` to:

1. load context with `load_probe_figure_context(output_root)`;
2. use the 2-row/4-column `GridSpec` layout when context exists;
3. call the existing stress-series plotting loop for bottom axes;
4. keep the current 1-row/2-axis fallback when context is `None`.

- [ ] **Step 4: Run composite rendering tests**

Run:

```bash
MPLBACKEND=Agg python -m pytest tests/studies/test_probe_mischaracterization_stress_test.py -k "stress_figure or probe_figure_context" -q
```

Expected: pass.

## Task 4: Record Manifest Context and Preserve Paper Export

**Files:**
- Modify: `scripts/studies/probe_mischaracterization_stress_test.py`
- Test: `tests/studies/test_probe_mischaracterization_stress_test.py`

- [ ] **Step 1: Add manifest metadata**

After `fig_path = write_stress_figure(...)`, add:

```python
probe_context = load_probe_figure_context(output_root)
if probe_context is not None:
    manifest["stress_figure_visual_context"] = {
        "amplitude_condition_id": probe_context["amplitude_condition_id"],
        "phase_condition_id": probe_context["phase_condition_id"],
        "amplitude_fallback_used": bool(probe_context["amplitude_fallback_used"]),
        "phase_fallback_used": bool(probe_context["phase_fallback_used"]),
        "panel_policy": "separate_probe_context_panels_not_insets",
        "display_channels": ["amplitude", "phase"],
    }
```

If calling the loader twice feels awkward, let `write_stress_figure(...)` return `(fig_path, visual_context_metadata)` instead, but only do that if it keeps the call site clearer.

- [ ] **Step 2: Extend or add manifest test**

Add a test that runs the cheap dry-run path or a direct manifest helper if one exists, then asserts `stress_figure_visual_context["panel_policy"] == "separate_probe_context_panels_not_insets"` when probe context artifacts are present.

- [ ] **Step 3: Preserve export path**

Do not change:

```python
shutil.copy2(source_figure, figures_dir / "probe_mischaracterization_stress.png")
```

The paper keeps consuming the same figure path.

- [ ] **Step 4: Run export-related tests**

Run:

```bash
python -m pytest tests/studies/test_probe_mischaracterization_stress_test.py -k "export_paper_assets or stress_figure" -q
```

Expected: pass.

## Task 5: Regenerate From Existing Complete Run

**Files/artifacts:**
- Update: `.artifacts/revision_studies/probe_mischaracterization/full_process_isolated_20260413T020739Z/figures/probe_mischaracterization_stress.png`
- Update: `tmp/probe/source_run/probe_mischaracterization_stress.png`
- Update: `tmp/probe/paper/probe_mischaracterization_stress.png`
- Update: `/home/ollie/Documents/ptychopinnpaper2/figures/probe_mischaracterization_stress.png`

- [ ] **Step 1: Regenerate the figure only**

Run:

```bash
MPLBACKEND=Agg python - <<'PY'
import json
import shutil
from pathlib import Path
from scripts.studies import probe_mischaracterization_stress_test as study

root = Path(".artifacts/revision_studies/probe_mischaracterization/full_process_isolated_20260413T020739Z")
condition_results = json.loads((root / "metrics.json").read_text(encoding="utf-8"))["conditions"]
fig_path = study.write_stress_figure(root, condition_results)
for target in [
    Path("tmp/probe/source_run/probe_mischaracterization_stress.png"),
    Path("tmp/probe/paper/probe_mischaracterization_stress.png"),
    Path("/home/ollie/Documents/ptychopinnpaper2/figures/probe_mischaracterization_stress.png"),
]:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fig_path, target)
    print(target)
print(fig_path)
PY
```

Expected: figure files are rewritten without rerunning model training.

- [ ] **Step 2: Inspect generated PNG dimensions**

Run:

```bash
python - <<'PY'
from pathlib import Path
from PIL import Image
for path in [
    Path(".artifacts/revision_studies/probe_mischaracterization/full_process_isolated_20260413T020739Z/figures/probe_mischaracterization_stress.png"),
    Path("tmp/probe/source_run/probe_mischaracterization_stress.png"),
    Path("/home/ollie/Documents/ptychopinnpaper2/figures/probe_mischaracterization_stress.png"),
]:
    with Image.open(path) as image:
        print(path, image.size, image.mode)
PY
```

Expected: image height is larger than the previous metrics-only `2200x800` PNG, and mode is readable by PIL.

- [ ] **Step 3: Visually inspect the paper copy**

Open:

```bash
xdg-open /home/ollie/Documents/ptychopinnpaper2/figures/probe_mischaracterization_stress.png
```

Manual check:

- top row contains true/amplitude-blurred amplitude and true/phase-noise phase panels;
- phase color scale is fixed and not autoscaled condition-by-condition;
- metric curves still show zero-perturbation baseline anchor points;
- labels are readable at expected paper size;
- there are no cramped insets or overlapping titles.

## Verification Commands

Run targeted tests:

```bash
python -m pytest tests/studies/test_probe_mischaracterization_stress_test.py -k "load_probe_figure_context or stress_figure or export_paper_assets" -q
```

Run the full probe stress test module:

```bash
python -m pytest tests/studies/test_probe_mischaracterization_stress_test.py -q
```

Run collection because this plan adds tests:

```bash
python -m pytest --collect-only tests/studies/test_probe_mischaracterization_stress_test.py -q
```

Run whitespace check:

```bash
git diff --check -- scripts/studies/probe_mischaracterization_stress_test.py tests/studies/test_probe_mischaracterization_stress_test.py docs/plans/revision-studies/probe-mischaracterization-probe-context-panels-plan.md
```

## Completion Criteria

- [ ] The paper figure uses separate probe-context panels, not insets.
- [ ] The true and amplitude-blurred probe amplitudes are shown as the amplitude pair.
- [ ] The true and phase-noise probe phases are shown as the phase pair.
- [ ] The amplitude panel defaults to `amplitude_blur_sigma_px_1p0`, the phase panel defaults to `phase_noise_sigma_rad_0p2pi_seed17`, and any per-channel fallback is recorded.
- [ ] The metric panels retain zero-perturbation baseline anchor points.
- [ ] Existing numeric metrics, gates, and perturbation grid are unchanged.
- [ ] The regenerated paper figure is written to `/home/ollie/Documents/ptychopinnpaper2/figures/probe_mischaracterization_stress.png`.
- [ ] Targeted and full probe stress tests pass.

## Artifacts Index

- Plan: `docs/plans/revision-studies/probe-mischaracterization-probe-context-panels-plan.md`
- Primary figure source run: `.artifacts/revision_studies/probe_mischaracterization/full_process_isolated_20260413T020739Z/`
- Paper figure path: `/home/ollie/Documents/ptychopinnpaper2/figures/probe_mischaracterization_stress.png`
- Inspection copy: `tmp/probe/paper/probe_mischaracterization_stress.png`
