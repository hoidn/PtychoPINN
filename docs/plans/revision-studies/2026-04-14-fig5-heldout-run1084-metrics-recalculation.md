# Table 3 / Fig. 5 Held-Out Run1084 Metrics Recalculation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Project override: do not create a git worktree. Session-policy override: do not spawn subagents unless the user explicitly asks for delegated agent work.

**Goal:** Recalculate the manuscript Table 3 / Fig. 5 ID-OOD Run1084 metrics on only the held-out spatial half that was not used for training, then update the paper table, table caption, surrounding Results/Discussion text, data provenance, changelog, and reviewer checklist.

**Architecture:** Extend the bounded revision-study script `scripts/studies/ood_fig5_metrics.py` with an explicit held-out spatial evaluation mode. The implementation must recover or validate the train/test split, align each reconstruction and the full Run1084 ePIE reference into the same full coordinate crop, then apply the held-out window as a relative crop to both arrays. Do not substitute a center crop of the reconstruction against a smaller held-out reference crop; that would select the wrong spatial region.

**Tech Stack:** PATH `python`, NumPy, existing `ptycho.image.cropping` conventions, `ptycho.evaluation.eval_reconstruction`, pytest, JSON/CSV/TeX artifacts, `latexmk` for paper verification.

---

## Initiative

- ID: `fig5-heldout-run1084-metrics-recalculation`
- Title: Recalculate Table 3 / Fig. 5 metrics on held-out Run1084 half
- Status: completed
- Owner: optional
- Spec/Source: user correction on 2026-04-14 that only the half not used for training should be used for evaluation
- Current stale run: `/home/ollie/Documents/PtychoPINN/.artifacts/revision_studies/fig5_ood_metrics_low_frequency_phase_20260413T071529Z/run_coordinate_primary_20260413T202230Z`

## Compliance Matrix

- [x] **Project instruction:** Consulted `docs/index.md` and relevant project docs/findings before planning.
- [x] **Paper-revision process:** This plan must update `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md` as work progresses.
- [x] **Finding/Policy ID:** `PYTHON-ENV-001` - invoke Python through PATH `python`; record interpreter provenance in the run manifest.
- [x] **Finding/Policy ID:** `CONFIG-001` - do not change legacy `params.cfg` initialization semantics; preserve the script's explicit `params.cfg["offset"]` handling around `eval_reconstruction`.
- [x] **Finding/Policy ID:** `DATA-001` - metric output key/shape changes must be intentional and documented for downstream paper assets.
- [x] **Project override:** Do not create worktrees.

## Context Priming

Read before editing:

- `docs/index.md`
- `docs/findings.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/TESTING_GUIDE.md`
- `docs/DATA_GENERATION_GUIDE.md`
- `docs/COMMANDS_REFERENCE.md`
- `specs/data_contracts.md`
- `/home/ollie/Documents/ptychopinnpaper2/ptychopinn_2025.tex`
- `/home/ollie/Documents/ptychopinnpaper2/data/README.md`
- `/home/ollie/Documents/ptychopinnpaper2/data/fig5_ood_metrics.json`
- `/home/ollie/Documents/ptychopinnpaper2/tables/fig5_ood_metrics.tex`
- `/home/ollie/Documents/ptychopinnpaper2/changelog.txt`
- `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`
- `/home/ollie/Documents/ptychopinnpaper2/revision_designs/fig5_ood_metrics_low_frequency_phase.md`
- `/home/ollie/Documents/ptychopinnpaper2/docs/backlog/2026-04-13-fig5-ood-metrics-presentation.md`
- `/home/ollie/Documents/ptychopinnpaper2/revision_plan.md`
- `scripts/studies/ood_fig5_metrics.py`
- `tests/studies/test_ood_fig5_metrics.py`
- `ptycho/image/cropping.py`
- Prior Fig. 5 OOD design/plan artifacts under `.artifacts/revision_studies/fig5_ood_metrics_low_frequency_phase_20260413T071529Z/`

Current state to account for:

- The paper says Siemens-star experiments use a spatial holdout: top half for training and bottom half for testing.
- Post-run split audit (2026-04-14): no persisted Run1084 training split manifest was found. The Aug. 2025 Run1084 model artifacts were produced before the independent-sampling rewrite, and the run-date code at `ea895c01:ptycho/workflows/components.py` used sequential slicing for `gridsize=1`, so `load_data(..., n_images=512)` implies the first 512 Run1084 frames. Those inferred training frames are all high-y coordinates (`y=58.8747..78.0384`). The accepted lower-y held-out run has zero coordinate-index overlap with that inferred first-512 training set; the opposite high-y diagnostic overlaps 512 of 544 coordinates with the inferred training selection.
- The exact first-512 complement (`512..1086`) is non-contiguous in scan space. A diagnostic `from_indices` run at `.artifacts/revision_studies/fig5_ood_metrics_low_frequency_phase_20260413T071529Z/run_heldout_first512_complement_20260414T012933Z` showed that the current bounding-box crop expands back to nearly the full scan and reproduces the old all-scan metrics. Do not use that diagnostic as paper evidence without implementing a masked/non-rectangular evaluation path.
- The current accepted Table 3 values were produced with full Run1084 scan-coordinate alignment, not an explicit held-out-half evaluation crop.
- `scripts/studies/ood_fig5_metrics.py` currently calls `align_for_evaluation(reconstruction, full_reference, scan_coords_yx, stitch_patch_size)` with all Run1084 coordinates.
- `align_for_evaluation` crops the full reference to the coordinate bbox, then center-crops both arrays to the smallest common shape.
- Passing only held-out coordinates to `align_for_evaluation` is not sufficient if the reconstruction image remains the full coordinate crop, because the reconstruction would be center-cropped rather than sliced to the held-out spatial window.
- `/home/ollie/Documents/ptychopinnpaper2/ptychopinn_2025.tex` contains the manuscript table inline, while `/home/ollie/Documents/ptychopinnpaper2/tables/fig5_ood_metrics.tex` is also exported. Both must be kept in sync unless the manuscript is refactored to `\input{tables/fig5_ood_metrics.tex}`.

## Scope

In scope:

- Add a reproducible held-out spatial-evaluation mode to `scripts/studies/ood_fig5_metrics.py`.
- Add tests that fail on the current all-scan or wrong-center-crop behavior.
- Recompute the four existing rows:
  - ID PtychoPINN
  - ID supervised baseline
  - OOD PtychoPINN
  - OOD supervised baseline
- Preserve the existing metric contract unless a stop gate triggers:
  - no primary fine registration before scoring
  - no support/background mask
  - `params.cfg["offset"]=4` legacy evaluation trim
  - phase-plane alignment
  - amplitude metrics with the legacy mean-scaling convention
  - PSNR as the MSE-derived value, with explanatory text in the manuscript
- Recompute reference-based FRC diagnostics/artifacts on the held-out field of view. Keep FRC out of the manuscript table unless the same-FOV acceptance gate passes and the paper text is explicitly updated.
- Export updated paper-side JSON/table assets and update the manuscript table/text, data README, changelog, and reviewer checklist.

Out of scope unless explicitly approved after a stop gate:

- Retraining models.
- Regenerating the reconstruction NPZs through a new inference run.
- Changing the primary registration policy.
- Changing Fig. 5 source panels, except for caption/text clarification about the table field of view.
- Adding new metrics beyond the held-out recalculation and its provenance diagnostics.

## Ambiguities and Stop Gates

- [x] **Exact split provenance:** No explicit persisted Run1084 split-index file was found; the fallback split policy and source evidence are documented in Task 4 and in the paper data README. A post-run audit found the run-date historical loader semantics: `gridsize=1` plus `n_images=512` selected the first 512 frames.
- [x] **Axis direction:** The run-date first 512 training frames are all high-y coordinates. The accepted lower-y sorted held-out half has 0/512 overlap with that training selection; the opposite high-y sorted diagnostic has 512/544 training overlap and is therefore the wrong spatial half for evaluation.
- [x] **Odd coordinate count:** Run1084 has 1087 scan positions; the deterministic lower-y held-out split records 543 evaluation coordinates and 544 training-side coordinates. The historical first-512 complement has 575 non-contiguous coordinates and is not safe with the current rectangular bounding-box crop.
- [x] **OOD comparability:** ID and OOD rows were evaluated on the same held-out Run1084 half.
- [x] **Reconstruction coordinate frame:** Panel-reference validation passed for all rows before metrics; the held-out crop is applied as a relative crop after full coordinate alignment.
- [x] **Figure/table field-of-view mismatch:** Fig. 5 panels remain full coordinate-crop images; the manuscript caption/text now state that Table 3 metrics use the held-out Run1084 test half.
- [x] **Conclusion changes:** The manuscript claim was narrowed to emphasize held-out phase/structural comparisons and to disclose that mean-scaled amplitude MSE favors the supervised baseline slightly while unscaled amplitude MSE favors PtychoPINN.

## Files and Responsibilities

- `scripts/studies/ood_fig5_metrics.py`: add split provenance, held-out crop-window construction, CLI flags, manifest fields, metric policy text, and paper asset emission metadata.
- `tests/studies/test_ood_fig5_metrics.py`: add failing tests for split selection, relative held-out cropping, manifest provenance, and paper-mode refusal without held-out provenance.
- `/home/ollie/Documents/ptychopinnpaper2/data/fig5_ood_metrics.json`: replace stale all-scan metric payload with held-out metric payload.
- `/home/ollie/Documents/ptychopinnpaper2/tables/fig5_ood_metrics.tex`: replace stale all-scan table rows.
- `/home/ollie/Documents/ptychopinnpaper2/ptychopinn_2025.tex`: update the inline Table 3 rows, table caption, and nearby Results/Discussion claims.
- `/home/ollie/Documents/ptychopinnpaper2/data/README.md`: record held-out split policy, source run root, manifest path, and stale all-scan run supersession.
- `/home/ollie/Documents/ptychopinnpaper2/changelog.txt`: add a correction entry for held-out Table 3 metrics.
- `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`: keep the R3 OOD metrics item reopened until held-out artifacts, paper update, and PDF inspection pass.
- `/home/ollie/Documents/ptychopinnpaper2/revision_designs/fig5_ood_metrics_low_frequency_phase.md`: update the design record so it no longer recommends or records the stale all-scan metric contract as final.
- `/home/ollie/Documents/ptychopinnpaper2/docs/backlog/2026-04-13-fig5-ood-metrics-presentation.md`: update the paper-backlog item with the held-out recomputation requirement and accepted/pending status.
- `/home/ollie/Documents/ptychopinnpaper2/revision_plan.md`: update the main revision plan to record the held-out Table 3 metric correction and remove stale instructions that conflict with the corrected metric contract.

## Implementation Design

### Split Provenance

Add CLI arguments:

```text
--evaluation-region {all_scan,heldout_test_half}
--heldout-split-source PATH
--heldout-split-policy {from_indices,bottom_half_by_sorted_y,top_half_by_sorted_y,paper_top_train_bottom_test_by_high_y}
--require-heldout-eval
--min-post-eval-trim-dim INT
```

Default behavior can remain `all_scan` for exploratory/backward-compatible runs, but paper-facing runs must pass `--require-heldout-eval --evaluation-region heldout_test_half`. `--emit-paper-assets` must refuse to export unless the source manifest records `evaluation_region="heldout_test_half"` and accepted split provenance.

If exact index files are found, `--heldout-split-policy from_indices` should read explicit test indices. Because the current crop is a rectangular bounding box, only use an explicit-index split for paper-facing metrics if the evaluation indices form a contiguous spatial region or a masked/non-rectangular evaluator has been implemented. If exact indices are not recoverable or the exact complement is non-contiguous, use a deterministic coordinate fallback only after documenting it:

```python
order = np.argsort(scan_coords_yx[:, 0], kind="mergesort")
top = order[: len(order) // 2]
bottom = order[len(order) // 2 :]
```

Then choose the paper-facing held-out half only after verifying coordinate orientation against training provenance. The executed held-out run used `paper_top_train_bottom_test_by_high_y`: because no persisted Run1084 training split manifest was found, the first 512 training frames are inferred from the historical Aug. 2025 loader for `gridsize=1`; those inferred frames are all high-y, and the lower-y sorted half has zero coordinate-index overlap with them. Record `n_total_scan_positions`, `n_eval_scan_positions`, `train_half`, `eval_half`, `axis`, `axis_direction_evidence`, and the selected coordinate ranges.

### Crop-Window Algorithm

Implement a helper that computes explicit crop windows instead of relying on center crop side effects:

```python
def coordinate_bbox(scan_coords_yx, stitch_patch_size, object_shape):
    radius = stitch_patch_size // 2
    min_y, min_x = scan_coords_yx.min(axis=0)
    max_y, max_x = scan_coords_yx.max(axis=0)
    return (
        max(0, int(min_y) - radius),
        min(object_shape[0], int(max_y) + radius),
        max(0, int(min_x) - radius),
        min(object_shape[1], int(max_x) + radius),
    )
```

Use the same bbox convention as `align_for_evaluation` to preserve comparability with the stale run. Then:

- Compute the all-scan bbox.
- Create the all-scan aligned reconstruction and reference exactly as current `align_for_evaluation` would, but record the GT origin after final center-cropping.
- Compute the held-out bbox from held-out coordinates.
- Intersect the held-out bbox with the all-scan aligned GT bbox.
- Convert that intersection to a relative slice.
- Apply the same relative slice to the already aligned reconstruction and aligned GT.
- Reject empty slices. Use the default `MIN_POST_EVAL_TRIM_DIM=56` for full-FOV runs, but allow the paper-facing held-out run to record a lower explicit `--min-post-eval-trim-dim` because the true held-out half is 42 pixels tall before the legacy `eval_offset=4` trim and 38 pixels tall after it.

Required manifest fields:

```json
{
  "evaluation_region": "heldout_test_half",
  "heldout_split_policy": "...",
  "heldout_split_source": "...",
  "total_coordinate_count": 1087,
  "eval_coordinate_count": 544,
  "all_scan_bbox_rows_cols": [start_row, end_row, start_col, end_col],
  "aligned_full_bbox_rows_cols": [start_row, end_row, start_col, end_col],
  "heldout_bbox_rows_cols": [start_row, end_row, start_col, end_col],
  "heldout_relative_slice_rows_cols": [row0, row1, col0, col1],
  "heldout_crop_shape": [height, width],
  "min_post_eval_trim_dim": 32,
  "heldout_only": true
}
```

### Metric Policy Text

Update `metric_policy(args)` so paper-facing output says:

- coordinate alignment against Run1084 ePIE reference
- held-out test-half spatial crop after coordinate alignment
- no primary fine registration
- no support/background mask
- `params.cfg["offset"]=4` two-pixel evaluation trim
- amplitude mean scaling
- phase-plane alignment
- PSNR derived from MSE on the post-trim arrays
- reference-based FRC is artifact-only unless same-FOV acceptance passes

## Tasks

### Task 1: Reopen and Audit the Existing Claim

**Files:**
- Modify: `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`
- Read-only: current run manifest under `.artifacts/revision_studies/fig5_ood_metrics_low_frequency_phase_20260413T071529Z/run_coordinate_primary_20260413T202230Z`

- [x] Add a checklist progress note that the all-scan coordinate-aligned Table 3 metrics are superseded pending held-out-half recomputation.
- [x] Confirm the stale run manifest has no `evaluation_region="heldout_test_half"` and used all 1087 Run1084 coordinates.
- [x] Record the stale source run root as historical only; do not delete it.

Audit note (2026-04-14): `.artifacts/revision_studies/fig5_ood_metrics_low_frequency_phase_20260413T071529Z/run_coordinate_primary_20260413T202230Z/fig5_ood_metrics_manifest.json` has `status="metrics_complete"` but no top-level or row-level `evaluation_region`; each of the four rows records `coordinate_count=1087`. Treat this run as historical superseded evidence only.

### Task 2: Add Failing Tests for Held-Out Evaluation

**Files:**
- Modify: `tests/studies/test_ood_fig5_metrics.py`

- [x] Add `test_paper_mode_requires_heldout_test_half_evaluation`.
- [x] Add `test_sorted_y_split_assigns_odd_count_deterministically` with synthetic coordinates and explicit expected indices.
- [x] Add `test_heldout_crop_uses_relative_scan_bbox_not_center_crop` using a synthetic reconstruction with unique row markers so a center crop would fail.
- [x] Add `test_alignment_manifest_records_heldout_bbox_and_coordinate_counts`.
- [x] Add `test_emit_paper_assets_refuses_all_scan_metrics`.

Run:

```bash
python -m pytest tests/studies/test_ood_fig5_metrics.py -k 'heldout or paper_mode_requires or emit_paper_assets_refuses_all_scan' -q
```

Result (2026-04-14): red run failed for missing CLI args, held-out alignment parameters, and all-scan paper export refusal; after implementation the targeted selector passed with `5 passed, 42 deselected`.

### Task 3: Implement Held-Out Split and Crop Support

**Files:**
- Modify: `scripts/studies/ood_fig5_metrics.py`

- [x] Add split-selection helpers and CLI args.
- [x] Add explicit bbox/window helpers that mirror `align_for_evaluation`.
- [x] Update `align_row_for_metrics(...)` so `all_scan` preserves current behavior and `heldout_test_half` applies the relative crop after full coordinate alignment.
- [x] Record the new manifest fields for each row and in the top-level metric policy.
- [x] Make `--require-heldout-eval` and `--emit-paper-assets` refuse all-scan artifacts.
- [x] Keep the existing panel-reference validation and no-primary-registration policy intact.

Run:

```bash
python -m pytest tests/studies/test_ood_fig5_metrics.py -q
python -m py_compile scripts/studies/ood_fig5_metrics.py tests/studies/test_ood_fig5_metrics.py
```

Result (2026-04-14): `python -m pytest tests/studies/test_ood_fig5_metrics.py -q` initially passed with `47 passed`; after adding the paper-specific split alias and configurable held-out post-trim floor, the full selector passed with `49 passed`; `python -m py_compile scripts/studies/ood_fig5_metrics.py tests/studies/test_ood_fig5_metrics.py` exited `0`.

### Task 4: Recover or Approve Split Provenance

**Files:**
- Read-only until a decision is made: training logs/configs and reconstruction/source manifests under `experiment_outputs/`, `.artifacts/`, and paper-side provenance.

- [x] Search for explicit Run1084 train/test index files, split manifests, data-bundle configs, and training invocation logs.
- [x] If exact test indices exist, use `--heldout-split-source` and `--heldout-split-policy from_indices`.
- [x] If exact indices do not exist, document the fallback `paper_top_train_bottom_test_by_high_y` policy and the evidence for axis direction before running paper-facing metrics.
- [x] Stop if the split cannot be justified from local evidence or user approval.

Result (2026-04-14): No explicit persisted Run1084 split-index file was found in the searched artifacts/logs/configs. A later audit of the run-date history found that `ea895c01:ptycho/workflows/components.py`, the code state immediately before the 2025-08-17 Run1084 artifacts, used sequential slicing for `gridsize=1`; therefore `n_images=512` implies the first 512 Run1084 frames. Those inferred training frames are all high-y (`y=58.8747..78.0384`). The accepted lower-y held-out split has 0/512 coordinate-index overlap with that inferred training selection, while the opposite high-y split overlaps 512/544 coordinates with the inferred training selection. The paper-facing fallback therefore remains `paper_top_train_bottom_test_by_high_y`: evaluate the lower-y held-out spatial half. Run1084 has 1087 coordinates, producing 543 lower-y held-out evaluation coordinates and 544 training-side coordinates. The exact first-512 complement branch is not used because the complement is non-contiguous and the current bounding-box crop leaks back to nearly the full scan.

Suggested search commands:

```bash
rg -n "Run1084|train.*half|test.*half|bottom|top|split|holdout|ycoords|xcoords" experiment_outputs .artifacts /home/ollie/Documents/ptychopinnpaper2
find experiment_outputs .artifacts -iname '*split*' -o -iname '*indices*' -o -iname '*manifest*'
```

### Task 5: Run Held-Out Metrics

**Files:**
- Writes new artifacts under `.artifacts/revision_studies/fig5_ood_metrics_low_frequency_phase_20260413T071529Z/`

- [x] Choose a fresh output root; do not reuse or overwrite the stale `run_coordinate_primary_20260413T202230Z`.
- [x] Run a validate-references-only pass with held-out evaluation enabled.
- [x] Run the full metric pass only after validation passes.
- [x] Verify the tracked PID exits `0` and required artifacts are freshly written.
- [x] Confirm every row manifest records `heldout_only=true`, `evaluation_region="heldout_test_half"`, `eval_coordinate_count < total_coordinate_count`, and non-empty held-out crop slices.

Result (2026-04-14): Accepted held-out run root is `.artifacts/revision_studies/fig5_ood_metrics_low_frequency_phase_20260413T071529Z/run_heldout_test_half_20260414T011335Z`. The tmux run tracked validate PID `574256` and metric PID `574435`; both exited `0`, and `fig5_ood_metrics_manifest.json`, `fig5_ood_metrics.json`, and `fig5_ood_metrics_table.tex` were freshly written. Each row records `evaluation_region="heldout_test_half"`, `heldout_only=true`, `heldout_split_policy="paper_top_train_bottom_test_by_high_y"`, `eval_coordinate_count=543`, and held-out crop shapes `[42, 65]` or `[42, 64]`. A later opposite-half diagnostic at `run_heldout_opposite_half_20260414T012635Z` confirmed the high-y split is the training-overlap half, not the reviewer-facing held-out half.

Use tmux for the full metric command if it is long-running, and track the exact launched PID:

```bash
tmux new-session -d -s fig5-heldout-metrics '
cd /home/ollie/Documents/PtychoPINN
set -euo pipefail
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate ptycho311
fi
RUN_ROOT=".artifacts/revision_studies/fig5_ood_metrics_low_frequency_phase_20260413T071529Z/run_heldout_test_half_$(date -u +%Y%m%dT%H%M%SZ)"
python scripts/studies/ood_fig5_metrics.py \
  --output-root "$RUN_ROOT" \
  --paper-root /home/ollie/Documents/ptychopinnpaper2 \
  --evaluation-region heldout_test_half \
  --heldout-split-policy bottom_half_by_sorted_y \
  --require-heldout-eval \
  --validate-references-only &
pid=$!
wait "$pid"
python scripts/studies/ood_fig5_metrics.py \
  --output-root "$RUN_ROOT" \
  --paper-root /home/ollie/Documents/ptychopinnpaper2 \
  --evaluation-region heldout_test_half \
  --heldout-split-policy bottom_half_by_sorted_y \
  --require-heldout-eval &
pid=$!
wait "$pid"
test -f "$RUN_ROOT/fig5_ood_metrics_manifest.json"
test -f "$RUN_ROOT/fig5_ood_metrics.json"
test -f "$RUN_ROOT/fig5_ood_metrics_table.tex"
'
```

Adjust `--heldout-split-policy` or add `--heldout-split-source` based on Task 4. Do not run the full command if Task 4 has not resolved the split.

Audit command:

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path("REPLACE_WITH_RUN_ROOT")
manifest = json.loads((root / "fig5_ood_metrics_manifest.json").read_text())
assert manifest["status"] == "metrics_complete"
for row in manifest["rows"]:
    assert row["evaluation_region"] == "heldout_test_half"
    assert row["heldout_only"] is True
    assert row["eval_coordinate_count"] < row["total_coordinate_count"]
    assert row["heldout_crop_shape"][0] > 0 and row["heldout_crop_shape"][1] > 0
print("held-out manifest audit ok")
PY
```

### Task 6: Update Design and Main Planning Docs

**Files:**
- Modify: `/home/ollie/Documents/ptychopinnpaper2/revision_designs/fig5_ood_metrics_low_frequency_phase.md`
- Modify: `/home/ollie/Documents/ptychopinnpaper2/docs/backlog/2026-04-13-fig5-ood-metrics-presentation.md`
- Modify: `/home/ollie/Documents/ptychopinnpaper2/revision_plan.md`
- Modify: this plan file

- [x] Update the Fig. 5 OOD metrics design doc to state that the 2026-04-13 all-scan coordinate-aligned values are superseded and that final Table 3 metrics must use the held-out Run1084 test half.
- [x] Update the paper backlog item so its current blocker/acceptance criteria require held-out split provenance, held-out metric artifacts, paper asset update, and PDF inspection.
- [x] Update the main revision plan with the corrected Table 3 / Fig. 5 metric contract and link the held-out implementation plan.
- [x] Keep this implementation plan's task checklist synchronized as execution progresses.

Result (2026-04-14): Updated `/home/ollie/Documents/ptychopinnpaper2/revision_designs/fig5_ood_metrics_low_frequency_phase.md`, `/home/ollie/Documents/ptychopinnpaper2/docs/backlog/2026-04-13-fig5-ood-metrics-presentation.md`, `/home/ollie/Documents/ptychopinnpaper2/revision_plan.md`, and this implementation plan with the held-out correction and run root.

### Task 7: Update Paper Assets and Manuscript

**Files:**
- Modify: `/home/ollie/Documents/ptychopinnpaper2/data/fig5_ood_metrics.json`
- Modify: `/home/ollie/Documents/ptychopinnpaper2/tables/fig5_ood_metrics.tex`
- Modify: `/home/ollie/Documents/ptychopinnpaper2/ptychopinn_2025.tex`
- Modify: `/home/ollie/Documents/ptychopinnpaper2/data/README.md`
- Modify: `/home/ollie/Documents/ptychopinnpaper2/changelog.txt`
- Modify: `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`

- [x] Use the script's paper asset emission only after the held-out manifest passes.
- [x] Replace inline Table 3 rows in `ptychopinn_2025.tex` to match `tables/fig5_ood_metrics.tex`, or refactor the manuscript to input the generated table and verify no duplicate stale rows remain.
- [x] Update the table caption to state the metrics are on the held-out Run1084 test half.
- [x] Update the Results paragraph to remove stale all-scan values and restate the conclusion using held-out values only.
- [x] Update the unscaled-amplitude sensitivity sentence with held-out values, or remove it if it no longer applies.
- [x] Update Discussion text about OOD phase artifacts if held-out phase metrics no longer support the current wording.
- [x] Update `data/README.md` with the new run root, manifest path, split policy, and supersession of `run_coordinate_primary_20260413T202230Z`.
- [x] Update `changelog.txt` with the correction.
- [x] Keep the reviewer checklist item open until PDF verification passes; then mark it complete with the held-out run root and compile evidence.

Result (2026-04-14): Emitted paper assets from the held-out run, updated the inline Table 3 values/caption and Results text in `/home/ollie/Documents/ptychopinnpaper2/ptychopinn_2025.tex`, updated `/home/ollie/Documents/ptychopinnpaper2/data/README.md`, added the changelog correction, and marked the reviewer checklist item complete after PDF verification.

Emission command after Task 5 audit:

```bash
python scripts/studies/ood_fig5_metrics.py \
  --output-root REPLACE_WITH_RUN_ROOT \
  --paper-root /home/ollie/Documents/ptychopinnpaper2 \
  --evaluation-region heldout_test_half \
  --heldout-split-policy REPLACE_WITH_POLICY \
  --require-heldout-eval \
  --emit-paper-assets
```

### Task 8: Verify

**Files:**
- Read/write build artifacts only under the paper repo's existing build/artifact paths.

- [x] Re-run targeted tests after paper asset emission:

```bash
python -m pytest tests/studies/test_ood_fig5_metrics.py -q
```

- [x] Verify no stale all-scan metric values remain in the paper text/table:

```bash
rg -n "0\\.01521|66\\.311|0\\.5349|0\\.02728|63\\.772|0\\.1869|run_coordinate_primary_20260413T202230Z|coordinate-aligned Fig\\. 5" /home/ollie/Documents/ptychopinnpaper2
```

- [x] Build the paper:

```bash
cd /home/ollie/Documents/ptychopinnpaper2
latexmk -pdf -interaction=nonstopmode -halt-on-error ptychopinn_2025.tex
```

- [x] Inspect the page containing Table 3 and the surrounding OOD text in `ptychopinn_2025.pdf`.
- [x] Record final evidence in `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`.

Result (2026-04-14): `python -m pytest tests/studies/test_ood_fig5_metrics.py -q` passed with `49 passed`; `python -m py_compile scripts/studies/ood_fig5_metrics.py tests/studies/test_ood_fig5_metrics.py` exited `0`; the paper-side Fig. 5 metrics JSON audit passed; the stale-value scan found no stale Table 3 metric values in `/home/ollie/Documents/ptychopinnpaper2/ptychopinn_2025.tex`, `/home/ollie/Documents/ptychopinnpaper2/data/fig5_ood_metrics.json`, or `/home/ollie/Documents/ptychopinnpaper2/tables/fig5_ood_metrics.tex`. `latexmk` was unavailable (`command not found`), so the paper was verified with two tracked `pdflatex -interaction=nonstopmode -halt-on-error ptychopinn_2025.tex` passes, both exiting `0`. `pdftotext` inspection confirmed the compiled PDF contains the held-out Table 3 caption and values (`0.02412`, `0.02182`, `0.5137`, `0.3805`). Existing overfull hbox warnings remain at lines 233--252, 670--687, and 694--715.

## Completion Criteria

- Held-out split provenance is either exact or explicitly approved/documented.
- Tests cover the held-out crop behavior and all-scan export refusal.
- New metrics are generated in a fresh artifact root with accepted manifest status.
- Design and main planning docs record the held-out correction and no longer point workers at stale all-scan Table 3 metrics as final evidence.
- Paper-side JSON/table, inline manuscript table, caption, Results/Discussion text, data README, changelog, and checklist all cite the same held-out run.
- PDF compiles and Table 3 inspection passes.
- No final paper claim depends on the stale all-scan coordinate-aligned metric values.
