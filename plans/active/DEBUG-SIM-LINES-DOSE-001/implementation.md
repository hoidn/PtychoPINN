# Implementation Plan (Phased)

## Initiative
- ID: DEBUG-SIM-LINES-DOSE-001
- Title: Isolate sim_lines_4x vs dose_experiments sim->recon discrepancy
- Owner: Codex + user
- Spec Owner: docs/specs/spec-ptycho-workflow.md
- Status: in_progress — Phase C verification (C2 ideal scenarios)

## Goals
- Identify whether the failure is caused by a core regression, nongrid pipeline differences, or a workflow/config mismatch.
- Produce a minimal repro that cleanly distinguishes grid vs nongrid and probe/normalization effects.
- Land a targeted fix (or document workflow change) with verification evidence.

## Phases Overview
- Phase A -- Evidence capture: pin down baseline behavior and parameters.
- Phase B -- Differential experiments: isolate the breaking dimension(s).
- Phase C -- Fix + verification: implement minimal correction and validate.

## Exit Criteria
1. A/B results captured for grid vs nongrid, probe normalization, and grouping parameters.
2. Clear root-cause statement with evidence (logs + params snapshot + artifacts).
3. Targeted fix or workflow change applied, with recon success and no NaNs.
4. Visual inspection success gate (when metrics are unavailable):
   - Reconstructed amplitude/phase show coherent structure (non-blank, non-NaN).
   - Correct canvas size and no obvious shifts/tiling artifacts.
   - Side-by-side PNGs captured with brief inspection notes.
5. **Test coverage verified:**
   - All cited selectors collect >0 tests (`pytest --collect-only`)
   - All cited selectors pass
   - Logs saved to `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/<timestamp>/`

## Compliance Matrix (Mandatory)
- [ ] Spec Constraint: docs/specs/spec-ptycho-core.md (Forward model + normalization invariants)
- [ ] Spec Constraint: docs/specs/spec-ptycho-workflow.md (load->group->infer->stitch sequence)
- [ ] Spec Constraint: specs/spec-inference-pipeline.md (stitching/offset contracts)
- [ ] Spec Constraint: specs/data_contracts.md (RawData/NPZ requirements)
- [ ] Fix-Plan Link: docs/fix_plan.md -- SIM-LINES-4X-001
- [ ] Finding/Policy ID: CONFIG-001
- [ ] Finding/Policy ID: MODULE-SINGLETON-001
- [ ] Finding/Policy ID: NORMALIZATION-001
- [ ] Finding/Policy ID: BUG-TF-001
- [ ] Test Strategy: plans/active/DEBUG-SIM-LINES-DOSE-001/test_strategy.md

## Spec Alignment
- Normative Spec: docs/specs/spec-ptycho-core.md, docs/specs/spec-ptycho-workflow.md
- Key Clauses: forward model + normalization rules; inference load->group->stitch contracts.

## Testing Integration

Principle: Every checklist item that adds or modifies observable behavior MUST specify its test artifact.

Format for checklist items:
```
- [ ] <ID>: <implementation task>
      Test: <pytest selector> | N/A: <justification>
```

Guidelines:
- <selector>: pytest selector covering this change (e.g., tests/unit/test_foo.py::test_bar)
- N/A: Valid only for pure refactoring with existing coverage, documentation-only changes, or infrastructure with no behavior change

## Architecture / Interfaces
- Key Data Types / Protocols: RawData, GroupedDataDict, PtychoDataContainer, TrainingConfig, InferenceConfig
- Boundary Definitions: sim_lines_4x pipeline (nongrid) -> workflows -> reassembly
- Sequence Sketch (Happy Path): simulate (nongrid) -> split -> group -> train -> infer -> stitch
- Data-Flow Notes: Raw diffraction + coords -> grouped patches + offsets -> model -> stitched object

## Context Priming (read before edits)
- Primary docs/specs to re-read:
  - docs/DATA_GENERATION_GUIDE.md (nongrid vs grid simulation)
  - docs/DEVELOPER_GUIDE.md (inference pipeline patterns)
  - docs/architecture_inference.md (stitching/offset flow)
  - docs/DATA_NORMALIZATION_GUIDE.md (physics/statistical/display scaling)
  - docs/debugging/QUICK_REFERENCE_PARAMS.md (CONFIG-001)
  - specs/data_contracts.md (RawData/NPZ contracts)
- Required findings/case law: CONFIG-001, MODULE-SINGLETON-001, NORMALIZATION-001, BUG-TF-001
- Related telemetry/attempts: plans/active/SIM-LINES-4X-001/reports/
- Data dependencies to verify:
  - ptycho/datasets/Run1084_recon3_postPC_shrunk_3.npz (custom probe)
  - outputs/sim_lines_4x/

## Phase A -- Evidence Capture
### Checklist
- [ ] A0: **Nucleus / Test-first gate:** Capture minimal failing repro or justify deferral.
      Test: N/A -- evidence-only while reproducing the failure in code
- [x] A1: Extract dose_experiments codepath without checkout:
      - Use `git ls-tree -r dose_experiments` to locate files.
      - Use `git show dose_experiments:<path>` for sim, training, inference, stitching modules.
      - Record parameter defaults (probe_mask, probe_big, probe_scale, default_probe_scale, gridsize, nphotons, split, grouping).
      Test: N/A -- evidence capture only
- [ ] A2: Verify data-contract expectations for any RawData/NPZ outputs used in comparison.
      Test: N/A -- evidence capture only
- [x] A3: Capture sim_lines_4x params snapshot with full config/params dump.
      Test: N/A -- evidence capture only
- [x] A4: Compare parameter tables (dose_experiments vs sim_lines_4x) and log probe stats + intensity_scale.
      - Outputs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T003217Z/{comparison_draft.md,comparison_diff.json}`
      Test: N/A -- evidence capture only
      Test: N/A -- evidence capture only

### Notes & Risks
- Ensure CONFIG-001 sync before any param readout.
- Avoid editing core modules during evidence-only steps.

## Phase B -- Differential Experiments
### Checklist
- [x] B1: Grid vs nongrid A/B in current codebase with identical seeds + probe settings.
      - Instrument grouping/splitting so we can run gs1/gs2 scenarios with the legacy `gridsize=2, nimgs_train=test=2` constraints even if the sim_lines pipeline defaults to much larger counts.
      - Target artifact: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/<timestamp>/grouping_comparison_{mode}.json` (summary of requested vs achieved groups, coord ranges, offset stats for both configurations).
      - Evidence captured in `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T020000Z/`:
        - `grouping_sim_lines_default.json`/`.md` show the snapshot defaults produce 1000/1000 groups with clean offset ranges.
        - `grouping_dose_experiments_legacy.json`/`.md` record the KDTree failure (`Dataset has only 2 points but 4 coordinates per group requested.`) when forcing `gridsize=2` and only two images per split, matching the legacy config.
      Test: N/A -- evidence run; log training stability + recon size
- [x] B2: Probe normalization A/B (set_default_probe path vs make_probe path) holding everything else constant.
      - Deliverable: plan-local CLI `bin/probe_normalization_report.py` that loads the Phase A snapshot, reconstructs both normalization paths (legacy `set_default_probe()` vs sim_lines `make_probe`/`normalize_probe_guess`), and emits JSON + Markdown summaries with amplitude min/max/mean, L2 norm, and ratio deltas.
      - Scenarios: run for `gs1_custom`, `gs1_ideal`, `gs2_custom`, `gs2_ideal` so we capture both custom-probe and idealized-probe cases.
      - Artifacts: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T031500Z/probe_stats_<scenario>.{json,md}` plus CLI log; the branches are numerically identical (max amplitude delta ≈5e-7) so probe scaling is no longer a suspect.
      Test: N/A -- evidence run; log probe stats + intensity_scale
- [x] B3: Grouping A/B (neighbor_count, group_count, gridsize) with fixed seeds to compare offsets and grouping shapes.
      - Extend `bin/grouping_summary.py` so each run records per-axis stats (min/max/mean/std) for `coords_offsets`/`coords_relative` and the min/max of `nn_indices`, giving us richer telemetry to compare gs1 vs gs2 behavior.
      - Generate three runs (identical seeds) and archive the JSON/Markdown outputs under `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T041700Z/`:
        1. `gs1_custom` default to refresh the baseline with the new stats.
        2. `gs2_custom` default (2000 images, gridsize=2) to prove KDTree grouping succeeds when enough raw points exist.
        3. `gs2_custom` override `--neighbor-count 1` (keeping total images high) to capture the “insufficient neighbors” failure signature that mirrors the legacy grid assumption.
      - Summarize the success/error contrast in the initiative summary; include the CLI log in the hub.
      - Evidence in `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T041700Z/` shows gs2 offsets reach ~382 px on axis 0/1, while gs1 offsets remain ≤ 195 px. These magnitudes far exceed the default `get_padded_size()` (~78 px when `offset=4`, `buffer=10`), so B4 now focuses on proving the padded-size shortfall.
      Test: N/A -- evidence run; log coords ranges and offset distributions; rerun `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`
- [x] B4: Reassembly A/B using a fixed synthetic container to isolate stitch math (M/padded_size).
      - Author `bin/reassembly_limits_report.py` to rebuild the nongrid simulation from the Phase A snapshot, regroup it, and emit JSON/Markdown summaries that compare:
        - observed max offsets per axis (train/test, gs1 vs gs2) vs. the derived `get_padded_size()` after CONFIG-001 bridging
        - the theoretical canvas requirement `ceil(N + 2·max_offset)` vs both the hard-coded `reassemble_M` from the snapshot and the legacy padded size
        - sampled reassembly sums when `size=get_padded_size()` vs `size=required_canvas` (use dummy patches backed by actual `coords_relative` so we can quantify how much signal is clipped)
      - Inputs: scenario name + overrides for gridsize/neighbor_count/total_images + `--group-limit` (default 64) to keep TensorFlow memory bounded when running `reassemble_whole_object`.
      - Outputs: `{scenario}_{subset}_reassembly.json/.md` detailing offsets, padded-size math, `fits_canvas` booleans, and ratios such as `(sum_default / sum_required)` so we can prove the gs2 configs lose ~80% of the energy when `size` is too small.
      - Evidence captured under `reports/2026-01-16T050500Z/` shows gs1/gs2 offsets reach ≈382 px while the legacy padded size remains 74–78 px, and the dummy reassembly sums lose 95–100 % of the signal when `size` stays at the legacy value.
      Test: N/A -- evidence run; add test in Phase C if fix touches core logic

### Notes & Risks
- If grid works and nongrid fails, focus on nongrid grouping/offsets.
- If both fail, suspect core regression in probe normalization, reassembly, or translation.

## Phase C -- Fix + Verification
### Checklist
- [x] C1: Implement padded-size update at the workflow layer so grouped datasets derive `max_position_jitter` from actual `coords_offsets` before the Keras model is built, and add a regression test for the factory helper.
      - Modify `ptycho/workflows/components.py::create_ptycho_data_container` to analyze grouped offsets, bump `params.cfg['max_position_jitter']` to cover `ceil(N + 2·max(|dx|,|dy|))`, and ensure repeated calls keep the maximum across train/test splits.
      - Share the helper with the plan-local CLI so reassembly_limits_report proves the new padded size matches the spec requirement.
      Test: `pytest tests/test_workflow_components.py::TestCreatePtychoDataContainer::test_updates_max_position_jitter -v`
- [x] C2: Rerun gs2_ideal + gs1_ideal and confirm:
      - Author `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py` to replay any snapshot scenario end-to-end (simulate → split → train → infer) while emitting:
        * `.npy` dumps for amplitude/phase plus lightweight PNGs for manual inspection
        * JSON stats (min/max/mean/std, NaN counts, padded size / `fits_canvas` flag) and a streaming log
      - Execute the runner for `gs1_ideal` and `gs2_ideal`, archiving both CLI logs and inspection notes in the current report hub.
      - Re-run `bin/reassembly_limits_report.py` for each scenario (same seeds, `--group-limit 64`) to prove the jitter-driven padded size keeps `fits_canvas=True`.
      - Record visual inspection notes (what structure is visible, any artifacts) in Markdown alongside thumbnails.
      Test: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`
- [x] C2b: Encode the reduced-load profile (`gs1_ideal`: base_total_images=512, group_count=256, batch_size=8; `gs2_ideal`: base_total_images=256, group_count=128, batch_size=4, neighbor_count=4) directly into `run_phase_c2_scenario.py` so reruns don’t require manual overrides, then regenerate both scenarios under a new hub with refreshed reassembly telemetry/logs and metadata tags documenting the profile. Evidence in `reports/2026-01-20T063500Z/` shows gs2 remains healthy while gs1 still collapses to NaNs even with the baked profile.
      Test: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`
- [x] C3: Instrument the Phase C2 runner to persist per-epoch training history + NaN detection (JSON + Markdown summary), rerun gs1_ideal and gs2_ideal under the baked profiles, and archive the new telemetry so we can pinpoint when gs1 diverges relative to gs2.
      Test: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`
- [x] C3b: Extend the Phase C2 runner with ground-truth comparison utilities that crop the stitched reconstructions to the simulated object size, compute amplitude/phase error metrics (MAE/RMSE/max/pearson), emit diff PNGs, and surface the metrics + artifact paths in `run_metadata.json` so gs1 vs gs2 divergence can be quantified without manual inspection. (`reports/2026-01-20T083000Z/` contains comparison_metrics.json, diff PNGs, and metadata pointers for gs1_ideal/gs2_ideal plus refreshed reassembly telemetry.)
      Test: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`
- [ ] C3c: Quantify amplitude/phase bias versus ground truth by augmenting `run_phase_c2_scenario.py` to record per-scenario prediction vs truth stats (mean, median, percentiles) and emit Markdown summaries so we can prove whether the gs1/gs2 collapses stem from a shared intensity offset.
      Test: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`
- [ ] C3d: Inspect the intensity-scaling pipeline by loading the gs1_ideal/gs2_ideal training checkpoints, dumping the `IntensityScaler`/`IntensityScaler_inv` weights and params (`params.cfg['intensity_scale']`), and comparing them to the observed amplitude bias so we can decide whether the workflow or the model needs the fix.
      Test: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`
- [ ] C4: Update docs/fix_plan.md Attempts History with evidence and root-cause summary.
      Test: N/A

### Notes & Risks
- Keep changes scoped; do not modify stable physics modules unless evidence pins regression there.

## Artifacts Index
- Reports root: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/
- Latest run: <YYYY-MM-DDTHHMMSSZ>/
