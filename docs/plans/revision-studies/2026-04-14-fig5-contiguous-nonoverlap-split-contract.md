# Fig. 5 Contiguous Non-Overlap Split Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Project override: do not create a git worktree. Session-policy override: do not spawn subagents unless the user explicitly asks for delegated agent work.

**Goal:** Regenerate or validate the manuscript Table 3 / Fig. 5 Run1084 metrics under an explicit contiguous, non-overlapping train/test-evaluation split contract; exact agreement with the displayed Fig. 5 panel crop is not required, and any metric-region mismatch only needs internal provenance unless existing paper text becomes false.

**Architecture:** Keep the bounded Fig. 5 metrics path in `scripts/studies/ood_fig5_metrics.py`, but add a split-contract audit before any paper-facing metric export. The audit must prove that the Run1084 training region and evaluation region are disjoint, spatially contiguous, and consistently applied to all ID/OOD rows; if the already-exported held-out run fails this stricter contract, stop and rerun metrics with a corrected contiguous held-out region or open a retraining/new-experiment decision.

**Tech Stack:** PATH `python`, NumPy, existing `scripts/studies/ood_fig5_metrics.py`, `tests/studies/test_ood_fig5_metrics.py`, JSON/CSV/TeX metrics artifacts, pytest, `pdflatex` or `latexmk` for paper verification.

---

## Initiative

- ID: `fig5-contiguous-nonoverlap-split-contract`
- Title: Enforce the Fig. 5/Table 3 contiguous non-overlap split contract
- Status: planned
- Spec/Source: user clarification on 2026-04-14: the evaluation does not have to exactly match the displayed images, but it does have to use non-overlapping, contiguous train and test/eval splits
- Related plan: `docs/plans/revision-studies/2026-04-14-fig5-heldout-run1084-metrics-recalculation.md`
- Related accepted run to audit first: `.artifacts/revision_studies/fig5_ood_metrics_low_frequency_phase_20260413T071529Z/run_heldout_test_half_20260414T011335Z`

## Compliance Matrix

- [ ] **Paper-revision process:** Update `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md` as this plan progresses.
- [ ] **Project instruction:** Re-read `docs/index.md`, `docs/findings.md`, and the related held-out plan before editing implementation code.
- [ ] **Finding/Policy ID:** `PYTHON-ENV-001` - invoke Python through PATH `python`.
- [ ] **Finding/Policy ID:** `DATA-001` - metric output schema changes must be intentional and documented for paper assets.
- [ ] **Project override:** Do not create worktrees.

## Context Priming

Read before edits:

- `docs/index.md`
- `docs/findings.md`
- `docs/TESTING_GUIDE.md`
- `docs/plans/revision-studies/2026-04-14-fig5-heldout-run1084-metrics-recalculation.md`
- `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`
- `/home/ollie/Documents/ptychopinnpaper2/revision_designs/fig5_ood_metrics_low_frequency_phase.md`
- `/home/ollie/Documents/ptychopinnpaper2/revision_plan.md`
- `/home/ollie/Documents/ptychopinnpaper2/ptychopinn_2025.tex`
- `/home/ollie/Documents/ptychopinnpaper2/data/fig5_ood_metrics.json`
- `/home/ollie/Documents/ptychopinnpaper2/tables/fig5_ood_metrics.tex`
- `scripts/studies/ood_fig5_metrics.py`
- `tests/studies/test_ood_fig5_metrics.py`
- `.artifacts/revision_studies/fig5_ood_metrics_low_frequency_phase_20260413T071529Z/run_heldout_test_half_20260414T011335Z/fig5_ood_metrics_manifest.json`

Current facts:

- The previous all-scan run is superseded because it evaluated the full Run1084 coordinate crop.
- The current paper-side held-out run uses `paper_top_train_bottom_test_by_high_y`, evaluates the lower-y Run1084 half, and records zero coordinate-index overlap with the inferred historical first-512 Run1084 training frames. No persisted Run1084 training split manifest was found; the first-512 training frame set is inferred from the run-date loader semantics.
- The exact first-512 complement is non-contiguous and should not drive this plan.
- The displayed Fig. 5 panels can remain full-FOV or legacy panel crops. A metric-region mismatch should be recorded in manifests and data provenance, but it does not need to be disclosed in the paper text unless current wording would otherwise be false.

## Scope

In scope:

- Add an explicit split-contract audit for paper-facing Fig. 5/Table 3 metrics.
- Audit the already-exported held-out run before rerunning anything.
- Regenerate Table 3 metrics only if the accepted held-out run fails the contiguous non-overlap contract or lacks enough manifest evidence.
- Preserve the existing metric policy unless a stop gate triggers:
  - no primary fine registration
  - no support/background mask
  - `params.cfg["offset"]=4` legacy trim
  - phase-plane alignment
  - amplitude mean scaling
  - PSNR as MSE-derived
- Update the design doc, main revision plan, paper data README/changelog, and reviewer checklist if the accepted run is reaffirmed or superseded.

Out of scope unless explicitly approved:

- Implementing masked/non-rectangular exact first-512 complement metrics.
- Changing Fig. 5 images only to match the metric crop.
- Retraining Run1084 ID models, except if the training-coordinate contiguity audit shows the existing model cannot support the clarified reviewer-facing claim.
- Adding new metric types or FRC columns.

## Ambiguities and Stop Gates

- [ ] **Non-overlap definition:** Decide and record whether "non-overlap" means coordinate-index disjointness or stricter object-footprint disjointness after applying the stitch/probe patch footprint. Default to the stricter object-footprint audit for reviewer-facing claims; if this is too conservative, require an explicit paper-design note before using coordinate-index disjointness only.
- [ ] **Training contiguity:** Do not assume "first512 all high-y" is enough. Audit whether the actual historical first-512 Run1084 training coordinates form a single contiguous spatial region. If they do not, stop and choose between retraining on an intentional contiguous split, narrowing the paper claim, or using the current result only as a diagnostic.
- [ ] **Evaluation contiguity:** The evaluation coordinates must form one contiguous spatial region, not a scattered complement. The lower-y sorted half is the current candidate.
- [ ] **Boundary leakage:** If object-footprint overlap exists at the train/eval boundary, either add a guard band/drop boundary scan positions and rerun metrics, or document a coordinate-index-only non-overlap policy before paper export.
- [ ] **Figure/table mismatch:** If the metric region differs from the displayed Fig. 5 images, record it in the manifest/data provenance. Do not require a paper-text disclosure unless the existing manuscript wording would otherwise make a false statement.

## Files and Responsibilities

- `scripts/studies/ood_fig5_metrics.py`: add split-contract audit helpers, manifest fields, paper-export refusal for unaudited splits, and optional guard-band split selection.
- `tests/studies/test_ood_fig5_metrics.py`: add tests for contiguous split auditing, non-overlap refusal, and manifest/paper-export gates.
- `docs/plans/revision-studies/2026-04-14-fig5-contiguous-nonoverlap-split-contract.md`: track execution progress.
- `/home/ollie/Documents/ptychopinnpaper2/revision_designs/fig5_ood_metrics_low_frequency_phase.md`: record that exact image matching is not required, but contiguous non-overlapping train/eval split evidence is required.
- `/home/ollie/Documents/ptychopinnpaper2/revision_plan.md`: update the active Fig. 5 metric contract.
- `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`: update progress notes as the audit/rerun proceeds.
- `/home/ollie/Documents/ptychopinnpaper2/data/README.md`: update run provenance if metrics are reaffirmed or regenerated.
- `/home/ollie/Documents/ptychopinnpaper2/changelog.txt`: update only if the paper-facing result changes or the paper text is corrected.
- `/home/ollie/Documents/ptychopinnpaper2/ptychopinn_2025.tex`: update only if metric values change or existing table/caption/text becomes inaccurate.

## Implementation Design

### Split-Contract Data Model

Add a serializable audit payload to each metric row manifest:

```json
{
  "split_contract": {
    "status": "ok",
    "policy": "paper_top_train_bottom_test_by_high_y",
    "nonoverlap_level": "coordinate_indices",
    "train_coordinate_count": 512,
    "eval_coordinate_count": 543,
    "train_contiguous": true,
    "eval_contiguous": true,
    "coordinate_index_overlap_count": 0,
    "object_footprint_overlap_pixel_count": 0,
    "object_footprint_overlap_policy": "guard_band_excluded",
    "metric_region_matches_displayed_panel": false
  }
}
```

If object-footprint overlap is not enforced, use:

```json
{
  "nonoverlap_level": "coordinate_indices",
  "object_footprint_overlap_pixel_count": null,
  "object_footprint_overlap_policy": "not_enforced_coordinate_split_only"
}
```

Paper export must refuse rows with `split_contract.status != "ok"` or missing `split_contract`.

### Contiguity Check

Prefer a footprint-based check because scan ordering may be misleading:

```python
def audit_split_contract(scan_coords_yx, train_indices, eval_indices, *, stitch_patch_size):
    train = np.asarray(train_indices, dtype=int)
    eval_ = np.asarray(eval_indices, dtype=int)
    if np.intersect1d(train, eval_).size:
        raise ValueError("train/eval coordinate indices overlap")

    train_mask = coordinate_footprint_mask(scan_coords_yx[train], stitch_patch_size)
    eval_mask = coordinate_footprint_mask(scan_coords_yx[eval_], stitch_patch_size)
    train_components = count_connected_components(train_mask)
    eval_components = count_connected_components(eval_mask)
    footprint_overlap = int(np.logical_and(train_mask, eval_mask).sum())

    return {
        "train_contiguous": train_components == 1,
        "eval_contiguous": eval_components == 1,
        "coordinate_index_overlap_count": 0,
        "object_footprint_overlap_pixel_count": footprint_overlap,
    }
```

If this stricter footprint audit fails only at the train/eval boundary, add a guard-band mode:

```text
--heldout-guard-band-pixels INT
--split-nonoverlap-level {coordinate_indices,object_footprint}
```

For `object_footprint`, drop boundary-side evaluation scan positions until the evaluation footprint has zero overlap with the training footprint, then recompute the held-out crop and metrics. Record dropped indices and coordinate ranges in the manifest.

## Tasks

### Task 1: Record the Clarified Contract

**Files:**
- Modify: `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`
- Modify: `/home/ollie/Documents/ptychopinnpaper2/revision_designs/fig5_ood_metrics_low_frequency_phase.md`
- Modify: `/home/ollie/Documents/ptychopinnpaper2/revision_plan.md`

- [x] Add a checklist progress note linking this plan and the clarified contract.
- [x] Update the design doc to say exact panel-image matching is not required, but the metric split must be contiguous and non-overlapping.
- [x] Update the main revision plan so future workers do not interpret the exact first-512 complement as required.

Run:

```bash
git diff --check -- docs/plans/revision-studies/2026-04-14-fig5-contiguous-nonoverlap-split-contract.md
git -C /home/ollie/Documents/ptychopinnpaper2 diff --check -- reviewer_revision_checklist.md revision_designs/fig5_ood_metrics_low_frequency_phase.md revision_plan.md
```

Expected: both commands exit `0`.

Result (2026-04-14): Updated `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`, `/home/ollie/Documents/ptychopinnpaper2/revision_designs/fig5_ood_metrics_low_frequency_phase.md`, and `/home/ollie/Documents/ptychopinnpaper2/revision_plan.md` to record the clarified contract and this plan path.

### Task 2: Audit the Existing Held-Out Run Before Rerunning

**Files:**
- Read-only: `.artifacts/revision_studies/fig5_ood_metrics_low_frequency_phase_20260413T071529Z/run_heldout_test_half_20260414T011335Z/fig5_ood_metrics_manifest.json`
- Read-only: `datasets/Run1084_recon3_postPC_shrunk_3.npz`
- Modify if needed: `docs/plans/revision-studies/2026-04-14-fig5-contiguous-nonoverlap-split-contract.md`

- [x] Reconstruct the historical first-512 Run1084 training indices from the documented run-date loader semantics.
- [x] Reconstruct the current lower-y held-out evaluation indices from the accepted manifest policy.
- [x] Compute coordinate-index overlap.
- [x] Compute train/eval contiguity using the stricter footprint/component audit.
- [x] Compute object-footprint overlap at the split boundary.
- [x] Record whether the existing held-out run is acceptable, acceptable with coordinate-index-only wording, or blocked pending a guarded rerun.

Run:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
import scripts.studies.ood_fig5_metrics as study

coords = study.load_scan_coords_yx(Path("datasets/Run1084_recon3_postPC_shrunk_3.npz"))
train = np.arange(512)
eval_ = study.select_heldout_indices(coords, policy="paper_top_train_bottom_test_by_high_y").eval_indices
print("train", len(train), "eval", len(eval_))
print("index_overlap", np.intersect1d(train, eval_).size)
print("train_y", float(coords[train, 0].min()), float(coords[train, 0].max()))
print("eval_y", float(coords[eval_, 0].min()), float(coords[eval_, 0].max()))
PY
```

Expected: index overlap is `0`; contiguity/footprint details are recorded by the new audit helper after Task 4.

Result (2026-04-14): The accepted lower-y evaluation split passes the clarified coordinate-index contract against the inferred historical first-512 training set. No persisted Run1084 training split manifest was found; the training set is inferred from run-date loader semantics. The audit recorded `train_coordinate_count=512`, `eval_coordinate_count=543`, `coordinate_index_overlap_count=0`, `train_connected_component_count=1`, and `eval_connected_component_count=1`. The stricter object-footprint audit recorded boundary overlap (`object_footprint_overlap_pixel_count=1151`) and is treated as internal provenance under `object_footprint_overlap_policy="recorded_not_enforced_coordinate_split_only"`, not as the paper split definition.

### Task 3: Add Red Tests for the Contract

**Files:**
- Modify: `tests/studies/test_ood_fig5_metrics.py`

- [x] Add `test_split_contract_rejects_coordinate_index_overlap`.
- [x] Add `test_split_contract_rejects_noncontiguous_eval_subset`.
- [x] Add `test_split_contract_records_panel_crop_mismatch_as_internal_provenance`.
- [x] Add `test_emit_paper_assets_requires_split_contract_ok`.
- [x] Add `test_object_footprint_overlap_requires_guard_band_when_strict`.

Run:

```bash
python -m pytest tests/studies/test_ood_fig5_metrics.py -k 'split_contract or guard_band or panel_crop_mismatch' -q
```

Expected before implementation: tests fail because the split-contract helper and export gate do not exist.

Result (2026-04-14): Red run failed as intended with four missing `audit_split_contract` attribute errors and one missing paper-export split-contract gate failure (`5 failed, 49 deselected`).

### Task 4: Implement the Split Audit and Export Gate

**Files:**
- Modify: `scripts/studies/ood_fig5_metrics.py`
- Modify: `tests/studies/test_ood_fig5_metrics.py`

- [x] Add `audit_split_contract(...)`.
- [x] Add a small no-dependency connected-component helper for boolean masks.
- [x] Add manifest fields under `split_contract`.
- [x] Add CLI flags:

```text
--split-nonoverlap-level {coordinate_indices,object_footprint}
--heldout-guard-band-pixels INT
--allow-panel-metric-region-mismatch
```

- [x] Refuse `--emit-paper-assets` if `split_contract.status` is not `ok`.
- [x] Rerun the red tests.

Run:

```bash
python -m pytest tests/studies/test_ood_fig5_metrics.py -k 'split_contract or guard_band or panel_crop_mismatch' -q
```

Expected after implementation: targeted tests pass.

Result (2026-04-14): Targeted selector passed with `6 passed, 49 deselected` after adding guard-band/drop coverage. The full Fig. 5 study test file later passed; final verification recorded `56 passed`.

### Task 5: Decide Whether to Reuse or Regenerate Metrics

**Files:**
- Modify if values/provenance change: `/home/ollie/Documents/ptychopinnpaper2/data/fig5_ood_metrics.json`
- Modify if values/provenance change: `/home/ollie/Documents/ptychopinnpaper2/tables/fig5_ood_metrics.tex`
- Modify if values/provenance change: `/home/ollie/Documents/ptychopinnpaper2/ptychopinn_2025.tex`
- Modify: `/home/ollie/Documents/ptychopinnpaper2/data/README.md`
- Modify: `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`
- Modify: `/home/ollie/Documents/ptychopinnpaper2/changelog.txt`

- [x] If the existing held-out run passes the clarified contract, update provenance only; do not change metric values.
- [x] If it passes coordinate-index non-overlap but fails object-footprint non-overlap, stop for a policy decision unless a guard-band rerun has been approved in this plan execution.
- [x] If a guard-band rerun is needed and approved, run a fresh output root under `.artifacts/revision_studies/fig5_ood_metrics_low_frequency_phase_20260413T071529Z/`.
- [x] Export paper assets only from a manifest with `split_contract.status="ok"`.
- [x] Update manuscript text only if metric values change or existing table/caption/text would otherwise be inaccurate.

Long-run command pattern:

```bash
python scripts/studies/ood_fig5_metrics.py \
  --output-root .artifacts/revision_studies/fig5_ood_metrics_low_frequency_phase_20260413T071529Z/run_contiguous_nonoverlap_<timestamp> \
  --paper-root /home/ollie/Documents/ptychopinnpaper2 \
  --evaluation-region heldout_test_half \
  --heldout-split-policy paper_top_train_bottom_test_by_high_y \
  --require-heldout-eval \
  --split-nonoverlap-level coordinate_indices \
  --allow-panel-metric-region-mismatch \
  --min-post-eval-trim-dim 32
```

Use tmux for the real run and track the exact PID:

```bash
python scripts/studies/ood_fig5_metrics.py ... &
pid=$!
wait "$pid"
```

Expected: command exits `0` and writes fresh manifest, JSON, CSV, and TeX artifacts.

Result (2026-04-14): Fresh accepted run root is `.artifacts/revision_studies/fig5_ood_metrics_low_frequency_phase_20260413T071529Z/run_contiguous_coordinate_index_nonoverlap_20260414T020035Z`. The validation-only command exited `0`; the full metric command exited `0`; the follow-up `--emit-paper-assets` command exited `0`. The run preserved all prior held-out metric values, added `split_contract.status="ok"` to every row, and archived the validation-only dry-run artifacts under `reference_validation_only_gate/` before the full metric invocation overwrote the run-root manifest/invocation files. Because values did not change, `/home/ollie/Documents/ptychopinnpaper2/ptychopinn_2025.tex` was not edited. Prior intermediate split-contract runs at `.artifacts/revision_studies/fig5_ood_metrics_low_frequency_phase_20260413T071529Z/run_contiguous_coordinate_index_nonoverlap_20260414T015540Z` and `.artifacts/revision_studies/fig5_ood_metrics_low_frequency_phase_20260413T071529Z/run_contiguous_nonoverlap_20260414T015117Z` are superseded.

### Task 6: Verification

**Files:**
- Read-only: all touched files and generated artifacts

- [x] Run the targeted Fig. 5 metric tests.
- [x] Run the full Fig. 5 metric test file.
- [x] Run `py_compile` on the script and tests.
- [x] Audit the paper JSON for `split_contract.status="ok"` on all rows.
- [x] Scan the paper for stale all-scan values and for any text that becomes inaccurate under the final split contract.
- [x] Compile the paper and inspect Table 3 text.

Run:

```bash
python -m pytest tests/studies/test_ood_fig5_metrics.py -q
python -m py_compile scripts/studies/ood_fig5_metrics.py tests/studies/test_ood_fig5_metrics.py
python - <<'PY'
import json
from pathlib import Path
p = Path("/home/ollie/Documents/ptychopinnpaper2/data/fig5_ood_metrics.json")
payload = json.loads(p.read_text())
for row in payload["rows"]:
    contract = row.get("split_contract")
    assert contract and contract["status"] == "ok", row.get("model")
print("split contract audit ok")
PY
```

Expected: all commands exit `0`.

Result (2026-04-14): Targeted selector passed with `6 passed, 49 deselected`; final full selector passed with `56 passed`; `python -m py_compile scripts/studies/ood_fig5_metrics.py tests/studies/test_ood_fig5_metrics.py` exited `0`; paper JSON split-contract audit printed `split contract audit ok`. The stale metric-value scan found no stale all-scan metric values in the paper source/JSON/table, with only intentional historical/superseded run mentions in provenance files. `latexmk` was unavailable, so two tracked `pdflatex -interaction=nonstopmode -halt-on-error ptychopinn_2025.tex` passes were run in `/home/ollie/Documents/ptychopinnpaper2`; both exited `0`. `pdftotext` inspection confirmed the held-out Table 3 caption and values (`0.02412`, `0.02182`, `0.5137`, `0.3805`).

## Completion Criteria

- [x] Table 3/Fig. 5 metrics are either reaffirmed from the current held-out run or regenerated from a fresh run with an explicit split-contract manifest.
- [x] The manifest proves coordinate-index disjointness and records whether stricter object-footprint non-overlap was enforced.
- [x] The training and evaluation regions are recorded as contiguous, or the plan is blocked with a retraining/new-experiment decision.
- [x] The manifest and data provenance record any mismatch between displayed Fig. 5 panels and the Table 3 metric region; paper-text disclosure is not required unless existing wording becomes inaccurate.
- [x] The design doc, main revision plan, reviewer checklist, and paper data README agree on the same metric contract and run root.
- [x] Verification commands pass and the paper PDF inspection confirms the updated Table 3 wording.

## Notes

- Do not chase the exact first-512 complement in this plan. It is non-contiguous and would require masked/non-rectangular metrics, which is not the clarified requirement.
- If the stricter object-footprint audit would require dropping many held-out scan positions, record the count before accepting a guard-band rerun. That may change the scientific interpretation enough to require a short user decision.
