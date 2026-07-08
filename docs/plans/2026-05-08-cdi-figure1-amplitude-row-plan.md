# CDI Figure 1 Amplitude Row Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not create a worktree; the repo policy forbids worktrees for this project.

**Goal:** Replace the current phase-only CDI Figure 1 with a same-column two-row figure showing amplitude and phase reconstructions from the existing `lines128` reconstruction artifacts.

**Architecture:** Keep the current Figure 1 row identities, crop policy, phase alignment, and final-FFNO override. Extend the paper refresh script so the figure writer loads both `amp` and `phase` from the same `recon.npz` files, renders a 2x6 grid, and records metadata that makes the amplitude and phase display policies explicit. Update the manuscript and style guide to reference the new amplitude/phase figure rather than the old phase-only image.

**Tech Stack:** Python, NumPy, Matplotlib, pytest, LaTeX manuscript assets under `docs/plans/NEURIPS-HYBRID-RESNET-2026/`.

---

## Scope

In scope:
- Add an amplitude row to the current CDI qualitative Figure 1.
- Use the same six visible columns as the current phase figure:
  `gt`, `pinn`, `pinn_fno_vanilla`, `pinn_ffno`, `pinn_neuralop_uno`, `pinn_hybrid_resnet`.
- Preserve the active corrected FFNO row override from `scripts/studies/cdi_final_ffno_pair.py`.
- Use the same center crop fraction for amplitude and phase.
- Use shared, GT-anchored amplitude display bounds across all columns.
- Keep phase alignment as global circular offset to GT before wrapping.
- Regenerate paper assets and the paper zip after the manuscript/image update.

Out of scope:
- Retraining any CDI model.
- Adding error maps to Figure 1.
- Changing CDI metrics tables or benchmark authority.
- Promoting depth-24 FFNO into Figure 1 unless `cdi_final_ffno_pair.py` is separately changed by an approved paper-refresh item.

## File Structure

- Modify: `scripts/studies/paper_results_refresh.py`
  - Add amplitude loading and a combined amplitude/phase Figure 1 writer.
  - Keep the current phase-only writer available if tests or fallback docs still use it.
  - Add metadata fields for `display_channels`, amplitude scale, phase scale, crop bounds, and source recon paths.
- Modify: `tests/studies/test_paper_results_refresh.py`
  - Add tests for combined amplitude/phase figure generation and metadata.
  - Update phase-only assumptions only where the tested paper-facing default changes.
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`
  - Update `figure_metadata`, `\includegraphics`, and caption for amplitude plus phase.
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_presentation_style_guide.md`
  - Update Figure 1 guidance from phase-only to amplitude/phase.
- Generated/update: `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/cdi_lines128_amp_phase_zoom_cnn_fno_ffno_uno_srunet.png`
  - New paper-facing Figure 1 image.
- Generated/update: paper zip under the existing paper build location.
  - Rebuild after script/manuscript changes so the zip does not contain stale `.tex` or figure assets.

## Display Contract

- Layout: 2 rows x 6 columns.
- Columns: unchanged from current Figure 1.
- Row labels: left-side labels `Amplitude` and `Phase`.
- Column titles: show titles only once on the top row.
- Amplitude colormap: grayscale or viridis is acceptable; prefer grayscale if it makes reconstruction texture easier to compare in print.
- Amplitude bounds: derive from the GT amplitude crop using robust upper quantile, e.g. `min` to `p99`, and apply to every amplitude panel.
- Phase bounds: preserve current `gt_crop_min_to_gt_crop_p99_after_alignment`.
- No per-panel colorbars.
- No independent per-panel amplitude scaling in the paper-facing default.

## Task 1: Add Amplitude Loading Tests

**Files:**
- Modify: `tests/studies/test_paper_results_refresh.py`
- Modify later: `scripts/studies/paper_results_refresh.py`

- [ ] **Step 1: Add a fixture that writes both `amp` and `phase` arrays**

Add or update the existing phase-zoom test fixture so every row writes:

```python
np.savez(row_dir / "recon.npz", amp=amp, phase=phase)
```

Use deliberately different amplitude ranges for GT and model rows so the test can prove the display bounds are GT-anchored rather than model-outlier anchored.

- [ ] **Step 2: Add a failing test for amplitude array validation**

Add a test for a new helper such as `_load_amp_array`:

```python
def test_load_amp_array_requires_2d_amp(tmp_path):
    path = tmp_path / "recon.npz"
    np.savez(path, amp=np.zeros((2, 2, 1), dtype=np.float32), phase=np.zeros((2, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="amp image must be 2D"):
        _load_amp_array(path)
```

Expected before implementation: import or name failure for `_load_amp_array`.

- [ ] **Step 3: Run the narrow failing test**

Run:

```bash
pytest -q tests/studies/test_paper_results_refresh.py::test_load_amp_array_requires_2d_amp
```

Expected: FAIL because the helper does not exist yet.

## Task 2: Implement Amplitude Loading And Bounds

**Files:**
- Modify: `scripts/studies/paper_results_refresh.py`
- Test: `tests/studies/test_paper_results_refresh.py`

- [ ] **Step 1: Add `_load_amp_array` near `_load_phase_array`**

Implementation requirements:
- Raise `FileNotFoundError` if the file is missing.
- Raise `KeyError` if `amp` is absent.
- Convert to `np.float32`.
- Require a 2D array.

- [ ] **Step 2: Add GT-anchored amplitude bounds helper**

Add a helper parallel to `gt_anchored_phase_bounds`, for example:

```python
def gt_anchored_amplitude_bounds(
    display_amplitudes: Mapping[str, np.ndarray],
    crop_bounds: Sequence[int],
    *,
    reference_row: str = "gt",
    upper_quantile: float = 0.99,
) -> tuple[float, float]:
    ...
```

Rules:
- Use only the cropped GT amplitude to choose bounds.
- Use `nanmin` and `nanquantile(..., 0.99)`.
- Fall back to `shared_display_bounds` if GT is missing or non-finite.
- Pad degenerate bounds through `shared_display_bounds`.

- [ ] **Step 3: Run helper tests**

Run:

```bash
pytest -q tests/studies/test_paper_results_refresh.py::test_load_amp_array_requires_2d_amp
```

Expected: PASS.

## Task 3: Add Combined Figure Writer

**Files:**
- Modify: `scripts/studies/paper_results_refresh.py`
- Test: `tests/studies/test_paper_results_refresh.py`

- [ ] **Step 1: Add a new default output constant**

Add:

```python
CDI_AMP_PHASE_ZOOM_FIGURE = (
    FIGURES_DIR / "cdi_lines128_amp_phase_zoom_cnn_fno_ffno_uno_srunet.png"
)
```

Do not reuse the old `cdi_lines128_phase_zoom...` filename for the combined figure.

- [ ] **Step 2: Add a data preparation helper**

Add a helper such as `_cdi_amp_phase_zoom_display_arrays(...)` that:
- Resolves recon paths with `_resolve_cdi_phase_zoom_recon_paths`.
- Loads `amp` and `phase` for every `CDI_PHASE_ZOOM_ROWS` row.
- Verifies all amplitude and phase arrays share the same shape.
- Computes one crop bound tuple from that shared shape.
- Aligns phase exactly as `_cdi_phase_zoom_display_phases` does today.
- Returns `display_amplitudes`, `display_phases`, and `crop_bounds`.

- [ ] **Step 3: Add the Matplotlib renderer**

Add `_save_cdi_amp_phase_zoom_figure(...)`:
- Create `plt.subplots(2, panel_count, figsize=(1.75 * panel_count, 4.0), constrained_layout=True)`.
- Top row renders amplitude with shared amplitude bounds.
- Bottom row renders phase with shared phase bounds.
- Top row gets column titles.
- Leftmost panels get row labels via `set_ylabel("Amplitude", fontsize=8)` and `set_ylabel("Phase", fontsize=8)`.
- All axes call `set_xticks([])` and `set_yticks([])` instead of fully hiding the left-side y-label axes.
- Save with `dpi=300`, `bbox_inches="tight"`, and `pad_inches=0.02`.

- [ ] **Step 4: Add `write_cdi_amp_phase_zoom_figure`**

The public writer should mirror `write_cdi_phase_zoom_figure` parameters:

```python
def write_cdi_amp_phase_zoom_figure(
    *,
    recons_root: Path | None = None,
    recon_paths: Mapping[str, Path] | None = None,
    output_path: Path = CDI_AMP_PHASE_ZOOM_FIGURE,
    final_ffno_pair: CdiFinalFfnoPair = FOUR_BLOCK_NO_REFINER_PAIR,
    final_output_stem: str | None = None,
    crop_fraction: float = 0.5,
) -> dict[str, object]:
    ...
```

Returned metadata must include:
- `figure`
- `versioned_figure`
- `source_recon_paths`
- `final_ffno_pair`
- `visible_rows`
- `display_channels: ["amp", "phase"]`
- `crop_fraction`
- `crop_bounds`
- `amplitude_colormap`
- `amplitude_display_scale`
- `amplitude_display_bounds`
- `phase_alignment`
- `phase_colormap`
- `phase_display_scale`
- `phase_display_bounds`

- [ ] **Step 5: Add a failing public-writer test**

Add a test such as:

```python
def test_write_cdi_amp_phase_zoom_figure_records_amp_and_phase_metadata(tmp_path):
    ...
    meta = write_cdi_amp_phase_zoom_figure(recons_root=recons, output_path=output)
    assert output.exists()
    assert meta["display_channels"] == ["amp", "phase"]
    assert meta["amplitude_display_scale"] == "gt_crop_min_to_gt_crop_p99"
    assert meta["phase_display_scale"] == "gt_crop_min_to_gt_crop_p99_after_alignment"
    assert meta["visible_rows"] == [...]
```

Also assert that the amplitude display upper bound is below an intentionally extreme model-only outlier, proving the scale is GT-anchored.

- [ ] **Step 6: Run the new test**

Run:

```bash
pytest -q tests/studies/test_paper_results_refresh.py::test_write_cdi_amp_phase_zoom_figure_records_amp_and_phase_metadata
```

Expected: PASS after implementation.

## Task 4: Wire The Combined Figure Into Asset Refresh

**Files:**
- Modify: `scripts/studies/paper_results_refresh.py`
- Test: `tests/studies/test_paper_results_refresh.py`

- [ ] **Step 1: Find the asset refresh entry point**

Inspect the function that currently calls `write_cdi_phase_zoom_figure`, likely `write_cdi_extended_assets` or the script `main`.

- [ ] **Step 2: Replace the paper-facing default with the combined figure**

Call `write_cdi_amp_phase_zoom_figure` for the manuscript-facing Figure 1 output.

Keep the phase-only writer callable for fallback/per-panel artifacts unless removing it is explicitly approved.

- [ ] **Step 3: Update asset-output tests**

If tests currently assert the paper-facing figure path is phase-only, update them to expect:

```python
figures/cdi_lines128_amp_phase_zoom_cnn_fno_ffno_uno_srunet.png
```

Do not weaken tests that verify final FFNO pair provenance.

- [ ] **Step 4: Run the paper refresh test module**

Run:

```bash
pytest -q tests/studies/test_paper_results_refresh.py
```

Expected: PASS.

## Task 5: Update Manuscript And Style Guide

**Files:**
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_presentation_style_guide.md`

- [ ] **Step 1: Update the manuscript figure metadata**

Replace the Figure 1 metadata line with fields equivalent to:

```tex
% figure_metadata: task=CDI_lines128; paper_copy=figures/cdi_lines128_amp_phase_zoom_cnn_fno_ffno_uno_srunet.png; visible_columns=gt,pinn,pinn_fno_vanilla,pinn_ffno,pinn_neuralop_uno,pinn_hybrid_resnet; display_channels=amplitude,phase; crop_fraction=0.5; amplitude_display_scale=gt_crop_min_to_gt_crop_p99; phase_alignment=global_circular_offset_to_gt_before_wrapping; phase_colormap=twilight; phase_display_scale=gt_crop_min_to_gt_crop_p99_after_alignment; source=.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-table-extension/runs/complete_table_plus_uno_20260504T100347Z/recons
```

- [ ] **Step 2: Update `\includegraphics`**

Use:

```tex
\includegraphics[width=\linewidth]{figures/cdi_lines128_amp_phase_zoom_cnn_fno_ffno_uno_srunet.png}
```

- [ ] **Step 3: Update the caption**

Caption requirements:
- Say amplitude and phase reconstructions are shown.
- Preserve the model list.
- State that phase predictions are globally aligned to the target before display.
- State that amplitude panels share a target-anchored display scale.
- Avoid internal/process wording such as "now includes", "paper-facing", "artifact", or "contract".

- [ ] **Step 4: Update the style guide**

Change the main CDI qualitative guidance from phase-only to amplitude/phase.

The guide should say:
- Figure 1 uses a fixed center crop.
- The top row is amplitude and the bottom row is phase.
- Amplitude uses shared target-anchored scaling.
- Phase uses the existing global circular alignment.
- The per-panel-scaled phase figure, if retained, is an alternate diagnostic, not the main Figure 1.

## Task 6: Regenerate Figure Assets And Paper Zip

**Files:**
- Generated/update: `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/cdi_lines128_amp_phase_zoom_cnn_fno_ffno_uno_srunet.png`
- Generated/update: existing paper zip

- [ ] **Step 1: Run the paper refresh command**

Use the existing command path for `scripts/studies/paper_results_refresh.py`. If unsure, inspect the script `main` help first:

```bash
python scripts/studies/paper_results_refresh.py --help
```

Then run the narrow command that refreshes CDI/paper assets without retraining.

- [ ] **Step 2: Verify the new figure exists and has the expected aspect**

Run:

```bash
python - <<'PY'
from pathlib import Path
from PIL import Image
p = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/cdi_lines128_amp_phase_zoom_cnn_fno_ffno_uno_srunet.png")
im = Image.open(p)
print(p, im.size)
assert im.size[0] > im.size[1]
assert im.size[1] > 900
PY
```

Expected: command exits `0`. The exact dimensions may vary with Matplotlib layout, but height should be materially larger than the old phase-only figure.

- [ ] **Step 3: Rebuild the paper zip**

Use the existing paper packaging command already used for this project. If it is not obvious, inspect `Makefile`, paper scripts, or prior plan commands before running.

- [ ] **Step 4: Verify zip freshness**

Run a zip listing check that proves both the `.tex` and new PNG inside the zip are fresh and match the repo-side filenames:

```bash
python - <<'PY'
from pathlib import Path
import zipfile
zip_path = Path("PATH/TO/PAPER.zip")
with zipfile.ZipFile(zip_path) as zf:
    names = set(zf.namelist())
    assert any(name.endswith("hybrid_resnet_neurips_first_draft.tex") for name in names)
    assert any(name.endswith("figures/cdi_lines128_amp_phase_zoom_cnn_fno_ffno_uno_srunet.png") for name in names)
print(zip_path)
PY
```

Replace `PATH/TO/PAPER.zip` with the actual zip path.

## Task 7: Final Verification

**Files:**
- All modified/generated files from previous tasks.

- [ ] **Step 1: Run targeted tests**

Run:

```bash
pytest -q tests/studies/test_paper_results_refresh.py
```

Expected: PASS.

- [ ] **Step 2: Check manuscript references**

Run:

```bash
rg -n "cdi_lines128_phase_zoom_cnn_fno_ffno_uno_srunet|display_channel=phase|cdi_lines128_amp_phase_zoom" docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_presentation_style_guide.md
```

Expected:
- The active Figure 1 manuscript reference uses `cdi_lines128_amp_phase_zoom_cnn_fno_ffno_uno_srunet.png`.
- No active Figure 1 metadata still says `display_channel=phase`.
- Commented fallback references may remain only if clearly labeled as alternate diagnostics.

- [ ] **Step 3: Run diff hygiene**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 4: Review generated image manually**

Open or inspect the PNG before committing. Confirm:
- 2 rows x 6 columns.
- Top row is amplitude.
- Bottom row is phase.
- Column order matches the current Figure 1 order.
- Amplitude row is not independently contrast-normalized per model.
- Phase row is not visibly wrapped incorrectly relative to the current phase-only figure.

## Commit Plan

Use one scoped commit:

```bash
git add \
  scripts/studies/paper_results_refresh.py \
  tests/studies/test_paper_results_refresh.py \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_presentation_style_guide.md \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/cdi_lines128_amp_phase_zoom_cnn_fno_ffno_uno_srunet.png \
  docs/plans/2026-05-08-cdi-figure1-amplitude-row-plan.md
git commit -m "paper: add amplitude row to CDI figure"
```

Add the rebuilt paper zip only if this repo already tracks that zip. Do not add bulky new archives if they are intentionally ignored or externally stored.
