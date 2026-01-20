# Implementation Plan (Phased)

## Initiative
- ID: DEBUG-SIM-LINES-DOSE-001
- Title: Isolate sim_lines_4x vs dose_experiments sim->recon discrepancy
- Owner: Codex + user
- Spec Owners: specs/spec-ptycho-workflow.md; specs/spec-ptycho-core.md; specs/spec-inference-pipeline.md; specs/data_contracts.md
- Status: **NaN DEBUGGING COMPLETE** — CONFIG-001 bridging (C4f) confirmed as root cause fix; amplitude bias (separate issue) remains open

## Goals
- Identify whether the failure is caused by a core regression, nongrid pipeline differences, or a workflow/config mismatch.
- Produce a minimal repro that cleanly distinguishes grid vs nongrid and probe/normalization effects.
- Land a targeted fix (or document workflow change) with verification evidence.

## Phases Overview
- Phase A -- Evidence capture: pin down baseline behavior and parameters.
- Phase B0 -- Hypothesis enumeration: systematically enumerate and rank root-cause candidates.
- Phase B -- Differential experiments: isolate the breaking dimension(s).
- Phase C -- Fix + verification: implement minimal correction and validate.

## Exit Criteria
1. A/B results captured for grid vs nongrid, probe normalization, and grouping parameters.
2. Clear root-cause statement with evidence (logs + params snapshot + artifacts).
3. Targeted fix or workflow change applied, with recon success and no NaNs.
4. Amplitude bias/intensity-scale alignment verified (metrics + scaler evidence or corrective fix).
5. Visual inspection success gate (when metrics are unavailable):
   - Reconstructed amplitude/phase show coherent structure (non-blank, non-NaN).
   - Correct canvas size and no obvious shifts/tiling artifacts.
   - Side-by-side PNGs captured with brief inspection notes.
6. **Test coverage verified:**
   - All cited selectors collect >0 tests (`pytest --collect-only`)
   - All cited selectors pass
   - Logs saved to `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/<timestamp>/`

## Compliance Matrix (Mandatory)
- [x] Spec Authority: Specs are consolidated under `specs/`; list applicable specs from this root.
- [x] Spec Constraint: specs/spec-ptycho-core.md (Forward model + normalization invariants)
- [x] Spec Constraint: specs/spec-ptycho-workflow.md (load->group->infer->stitch sequence)
- [x] Spec Constraint: specs/spec-inference-pipeline.md (stitching/offset contracts)
- [x] Spec Constraint: specs/data_contracts.md (RawData/NPZ requirements)
- [x] Fix-Plan Link: docs/fix_plan.md -- DEBUG-SIM-LINES-DOSE-001
- [x] Finding/Policy ID: CONFIG-001
- [x] Finding/Policy ID: MODULE-SINGLETON-001
- [x] Finding/Policy ID: NORMALIZATION-001
- [x] Finding/Policy ID: BUG-TF-001
- [x] Test Strategy: plans/active/DEBUG-SIM-LINES-DOSE-001/test_strategy.md

## Spec Alignment
- Normative Specs (authoritative under `specs/`):
  - specs/spec-ptycho-core.md
  - specs/spec-ptycho-workflow.md
  - specs/spec-inference-pipeline.md
  - specs/data_contracts.md
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

### Checklist
- [x] A0: **Nucleus / Test-first gate:** Capture minimal failing repro or justify deferral.
      - Completed via the SIM-LINES snapshot + grouping CLI captured in `reports/2026-01-16T000353Z/` (params snapshot, `dose_experiments_tree.txt`, smoke-test pytest) which reproduces the sim_lines vs dose_experiments mismatch without new code.
      Test: N/A -- evidence-only while reproducing the failure in code
- [x] A1: Extract dose_experiments codepath without checkout:
      - Use `git ls-tree -r dose_experiments` to locate files.
      - Use `git show dose_experiments:<path>` for sim, training, inference, stitching modules.
      - Record parameter defaults (probe_mask, probe_big, probe_scale, default_probe_scale, gridsize, nphotons, split, grouping).
      Test: N/A -- evidence capture only
- [x] A1b: Run dose_experiments ground truth from the local checkout at `/home/ollie/Documents/PtychoPINN`.
      - **Status:** BLOCKED (Keras 3.x incompatibility) — documented in `reports/2026-01-20T143500Z/a1b_closure_rationale.md`
      - **Blocker:** Legacy `ptycho/model.py` uses `tf.shape()` directly on Keras tensors, which is prohibited in Keras 3.x (`KerasTensor cannot be used as input to a TensorFlow function`)
      - **Partial success:** Simulation stage completed with 512 diffraction patterns; training stage fails at model construction
      - **Resolution:** A1b is no longer required for NaN debugging scope:
        - NaN root cause identified (CONFIG-001 violation)
        - Fix applied and verified (C4f bridging)
        - All scenarios (gs1/gs2 × ideal/custom) train without NaN
      - **Evidence:** `reports/2026-01-20T092411Z/simulation_clamped4.log` (simulation success + training failure)
      Test: N/A -- blocked; evidence captured
- [x] A2: Verify data-contract expectations for any RawData/NPZ outputs used in comparison.
      - Completed alongside the Phase A4 comparison diff: `reports/2026-01-16T003217Z/{comparison_draft.md,comparison_diff.json}` enumerate the RawData/grouped dict keys + dtypes pulled from the SIM-LINES snapshot.
      Test: N/A -- evidence capture only
- [x] A3: Capture sim_lines_4x params snapshot with full config/params dump.
      Test: N/A -- evidence capture only
- [x] A4: Compare parameter tables (dose_experiments vs sim_lines_4x) and log probe stats + intensity_scale.
      - Outputs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T003217Z/{comparison_draft.md,comparison_diff.json}`
      Test: N/A -- evidence capture only

### Notes & Risks
- Ensure CONFIG-001 sync before any param readout.
- Avoid editing core modules during evidence-only steps.

## Phase B0 -- Hypothesis Enumeration (Root Cause Brainstorm)

**Purpose:** Systematically enumerate all plausible root causes before running differential experiments. This prevents tunnel vision and ensures test ordering reflects likelihood/impact.

**Observed Symptoms (as of 2026-01-20):**
- sim_lines_4x reconstructions show ~12× amplitude undershoot vs ground truth
- gs1_ideal (gridsize=1) collapses to NaN at epoch 3
- gs2_ideal (gridsize=2) healthy after CONFIG-001 bridging fix
- Constant rescaling (C4e) does not close the amplitude gap (pearson_r stuck at ~0.10)

**Critical Context:** Ideal probe **worked in dose_experiments** but is broken locally. This rules out inherent numeric instability with ideal probe normalization and points to a **regression or code divergence** between dose_experiments and sim_lines_4x.

### Hypothesis Candidates (Ranked by Likelihood)

| Rank | ID | Hypothesis | Rationale | Test | Status |
|------|----|------------|-----------|------|--------|
| 1 | H-CONFIG | Stale `params.cfg` values (CONFIG-001 violation) | Legacy modules read unsynced globals; known cause of gridsize/intensity drift | C4f: Add bridging calls before train/infer | **✅ CONFIRMED** — Root cause of all NaN failures; C4f fix enabled all scenarios (gs1/gs2 × ideal/custom) to train without NaN |
| 2 | H-PROBE-IDEAL-REGRESSION | Ideal probe handling regressed between dose_experiments and sim_lines_4x | Ideal probe worked in dose_experiments but fails locally; something changed in how ideal probe is processed (not inherent numeric instability since it worked before) | Diff dose_experiments vs sim_lines ideal probe code paths; check probe loading, normalization application timing, intensity_scale derivation | **❌ RULED OUT** — B0f showed both ideal and custom probes work after CONFIG-001 fix; no probe-specific regression |
| 3 | H-GRIDSIZE-NUMERIC | gridsize=1 triggers degenerate numeric paths | gs1 fails while gs2 succeeds with identical CONFIG-001 fix; gridsize=1 may hit edge cases in grouping/reassembly math | Compare gs1 vs gs2 loss curves, gradient norms, intermediate activations | **❌ RULED OUT** — B0f showed gs1_custom works; all gridsize=1 scenarios work after CONFIG-001 fix |
| 4 | H-PROBE-SCALE-CUSTOM | Custom probe scale factor (4.0 vs 10.0) mismatch | dose_experiments may have used different probe_scale; scaling inconsistency propagates to intensity | Check dose_experiments defaults vs sim_lines | [x] A1/A4 captured params |
| 5 | H-INTENSITY-SCALE | `intensity_scale` computation differs between pipelines | The recorded `intensity_scale` may not match what legacy code expects | C3d: Dump IntensityScaler weights | [x] Inspected — no obvious mismatch |
| 6 | H-OFFSET-OVERFLOW | Reassembly offsets exceed padded canvas | B4 showed offsets reach ~382px vs legacy ~78px padded size | C1: Implement jitter-based padding | [x] Fixed — `fits_canvas=True` now |
| 7 | H-GROUPING-KDTREE | KDTree grouping fails with small datasets | B1 showed "only 2 points but 4 coordinates per group requested" | B1/B3 grouping A/B | [x] Confirmed — nongrid grouping works with sufficient points |
| 8 | H-LOSS-WIRING | Loss function receives incorrectly scaled inputs | Double-scaling or missing normalization in loss computation | Instrument loss inputs pre/post normalization | [ ] Open |
| 9 | H-LEGACY-CODEPATH | Legacy sim/train/infer codepath differs fundamentally | dose_experiments may use different module versions | A1b: Run ground-truth from legacy checkout | [ ] Blocked — Keras 3.x incompatibility |
| 10 | H-SEED-DIVERGENCE | Random seed handling differs between pipelines | Probe/noise initialization may diverge | Control seeds in A/B experiments | [x] Seeds controlled in B1-B4 |

### Hypotheses NOT Tested by B2 (Probe Normalization A/B)

**Critical gap identified:** B2 tested whether `set_default_probe()` vs `make_probe/normalize_probe_guess` produce identical outputs. They do (≤5e-7 delta). **But B2 did NOT test:**

1. **H-PROBE-IDEAL-REGRESSION:** What changed between dose_experiments (where ideal probe worked) and sim_lines_4x (where it fails)? The issue is NOT inherent numeric instability with small norm factors — it's a regression.

2. **Cross-probe comparison:** The test compared legacy vs sim_lines *within* each probe type, not *across* probe types. The dramatic differences between ideal and custom probes were never analyzed for causal impact:

   | Probe | Norm factor | Amp mean | L2 norm |
   |-------|-------------|----------|---------|
   | ideal | 0.009766 | 0.100 | 102.77 |
   | custom | 1.100300 | 0.292 | 42.65 |

3. **Regression candidates to investigate:**
   - Probe loading sequence (when is probe applied relative to intensity normalization?)
   - `intensity_scale` derivation (does sim_lines compute it differently for ideal vs custom?)
   - Forward model probe application (any changes to how probe amplitude is used?)
   - CONFIG-001 timing (is `update_legacy_dict` called at the right point for ideal probe scenarios?)

### Test Priority (Recommended Order)

Based on the current state (gs2_ideal healthy, gs1_ideal NaN) and the fact that **ideal probe worked in dose_experiments**, the next experiments should be:

1. **H-PROBE-IDEAL-REGRESSION:** Diff the ideal probe code paths between dose_experiments and sim_lines_4x:
   - Compare probe loading/normalization timing
   - Compare `intensity_scale` computation for ideal vs custom probes
   - Check if CONFIG-001 bridging happens at the same point in both pipelines
   - Look for any conditional logic that treats ideal probes differently

2. **Isolation test:** Run gs1_custom (gridsize=1 with custom probe) to determine whether:
   - If gs1_custom works → problem is specific to ideal probe handling (regression)
   - If gs1_custom also fails → problem is gridsize=1 (H-GRIDSIZE-NUMERIC)

3. **H-GRIDSIZE-NUMERIC:** If isolation test points to gridsize, add targeted logging at pipeline boundaries (forward pass input/output, loss value, gradient norms) to identify where NaN first appears. Escalate to full instrumentation (e.g., `tf.debugging.enable_check_numerics()`) only if boundary logging is insufficient.

4. **H-LOSS-WIRING:** Instrument the loss function to log input/output scales and verify NORMALIZATION-001 compliance.

### Decision Tree

```
gs2_ideal healthy, gs1_ideal NaN after CONFIG-001 fix
+ KEY FACT: ideal probe WORKED in dose_experiments (not inherent numeric issue)
│
├─ Step 1: Run gs1_custom (gridsize=1 + custom probe) to isolate variable  ✅ EXECUTED
│   │
│   ├─ gs1_custom WORKS → Problem is ideal probe handling (H-PROBE-IDEAL-REGRESSION)
│   │   └─ ...
│   │
│   └─ gs1_custom FAILS → Problem is gridsize=1 (H-GRIDSIZE-NUMERIC)
│       └─ ...
│
└─ ACTUAL RESULT: gs1_custom WORKS *and* gs1_ideal ALSO WORKS after CONFIG-001 fix!
    │
    └─ **ROOT CAUSE IDENTIFIED: CONFIG-001 violation**
        │
        ├─ The gs1_ideal NaN at epoch 3 was observed BEFORE C4f bridging
        ├─ After C4f, both gs1_ideal and gs1_custom train without NaN
        ├─ H-PROBE-IDEAL-REGRESSION: RULED OUT (both probe types work)
        ├─ H-GRIDSIZE-NUMERIC: RULED OUT (gridsize=1 works fine)
        │
        └─ CONCLUSION: NaN debugging COMPLETE
            Remaining amplitude bias (~3-6x) is a SEPARATE issue from NaN stability
```

### Checklist
- [x] B0a: Document observed symptoms and failure modes.
- [x] B0b: Enumerate hypothesis candidates with rationale.
- [x] B0c: Rank hypotheses by likelihood and testability.
- [x] B0d: Identify gaps in existing tests (B2 did not test cross-probe scaling).
- [x] B0e: Add critical context: ideal probe worked in dose_experiments → regression, not inherent numeric issue.
- [x] B0f: **Isolation test:** Run gs1_custom (gridsize=1 + custom probe) to determine if problem is probe-specific or gridsize-specific.
      - Executed gs1_custom with same workload as gs1_ideal (512 images, 256 groups, batch_size=8)
      - Result: **gs1_custom trains without NaN** (has_nan=false), matching gs1_ideal after CONFIG-001 bridging
      - gs1_custom pearson_r=0.155 (vs gs1_ideal 0.102) — slightly better correlation with custom probe
      - gs1_custom amplitude pred mean=0.704, truth mean=2.71 (~3.8x undershoot)
      - Conclusion: **NaN failures were caused by CONFIG-001 violations, not probe type**
      - Decision tree resolution: H-PROBE-IDEAL-REGRESSION NOT confirmed; problem was workflow-wide
      - Evidence: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T102300Z/{gs1_custom/*,summary.md}`
      Test: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`
- [x] B0g: Closed — B0f showed both probe types work after CONFIG-001 bridging; no ideal-probe-specific regression to investigate.
- [x] B0h: Closed — B0f showed both gridsize=1 scenarios work after CONFIG-001 bridging; no gridsize-specific numeric issue to investigate.

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
      - Artifacts: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T031500Z/probe_stats_<scenario>.{json,md}` plus CLI log; the branches are numerically identical (max amplitude delta ≈5e-7).
      - **Scope limitation (B0 review):** This test proved legacy vs sim_lines *code paths* are equivalent for a given probe type. It did NOT test whether the dramatically different normalization characteristics between ideal (norm=0.0098) and custom (norm=1.1) probes cause downstream amplitude issues. See H-PROBE-SCALE-IDEAL in Phase B0.
      - **Corrected conclusion:** "Legacy vs sim_lines code paths are equivalent" — the ideal vs custom probe scaling question remains open (B0e).
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
- [x] C3c: Quantify amplitude/phase bias versus ground truth by augmenting `run_phase_c2_scenario.py` to record per-scenario prediction vs truth stats (mean, median, percentiles) and emit Markdown summaries so we can prove whether the gs1/gs2 collapses stem from a shared intensity offset.
      Evidence: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T093000Z/` (bias summaries + metrics)
      Test: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`
- [x] C3d: Inspect the intensity-scaling pipeline by loading the gs1_ideal/gs2_ideal training checkpoints, dumping the `IntensityScaler`/`IntensityScaler_inv` weights and params (`params.cfg['intensity_scale']`), and comparing them to the observed amplitude bias so we can decide whether the workflow or the model needs the fix. (Evidence: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T103000Z/`)
      Test: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`
- [x] C4a: Instrument the Phase C2 runner with intensity-normalization stats (raw `diff3d`, grouped diffraction, container tensors, recorded `intensity_scale`) so each scenario writes a JSON summary alongside the existing bias metrics.
      Test: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`
- [x] C4b: Re-run the gs1_ideal and gs2_ideal scenarios under a fresh artifacts hub with the new instrumentation enabled so the intensity stats, amplitude/phase metrics, and training evidence all live together for analysis.
      Test: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`
- [x] C4c: Update docs/fix_plan.md Attempts History (plus summary.md) with the new intensity telemetry and document the decision tree for the next remediation step.
      Test: N/A
- [x] C4d: Extend the bias analyzer to compute per-scenario prediction↔truth scaling diagnostics (best-fit scalar, ratio distribution, rescaled error) so we can prove whether a constant amplitude factor explains the ≈12× drop before touching loader/workflow code.
      - Evidence: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T143000Z/{bias_summary.json,bias_summary.md}` now records ratio distributions + least-squares scalars (gs1_ideal best scalar≈1.878, gs2_ideal ratios empty due to NaNs).
      - Guard: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v` (1 passed)
      Test: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`
- [x] C4e: Prototype amplitude rescaling in the workflow layer by applying the recorded scaling factor (bundle `intensity_scale` or analyzer least-squares scalar) before saving stitched outputs, then rerun gs1_ideal + gs2_ideal to confirm whether a deterministic scalar closes the ≈12× gap.
      - Implementation: add an opt-in rescaling hook to `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py` (and mirror it in `scripts/studies/sim_lines_4x/pipeline.py::run_inference`) so predictions can be multiplied by either the bundle `intensity_scale` or the analyzer-derived scalar before stats/PNG emission, then persist the chosen scalar in `run_metadata`.
      - Verification: reran gs1_ideal + gs2_ideal under `reports/2026-01-20T150500Z/` with rescaled amplitude `.npy`/PNG outputs and refreshed analyzer summaries; least-squares scaling (~1.86×) reduced normalized-to-prediction ratio but amplitude bias/pearson_r (≈0.10) remained unchanged, so constant rescaling alone cannot close the gap.
      Test: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`
- [x] C4f: Enforce CONFIG-001 bridging before every training/inference handoff (pipeline + plan-local runner) so legacy modules read the current dataclass settings prior to loader/model usage, then rerun gs1_ideal + gs2_ideal with analyzer evidence to confirm whether synced `params.cfg` resolves the amplitude drift.
      - **ROOT CAUSE FIX CONFIRMED**: Added `update_legacy_dict(params.cfg, config)` calls in both `scripts/studies/sim_lines_4x/pipeline.py` and `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py` before all training/inference handoffs
      - All four scenarios (gs1_ideal, gs1_custom, gs2_ideal, gs2_custom) now train without NaN
      - Evidence: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T160000Z/` (gs*_ideal runs), `reports/2026-01-20T102300Z/` (gs1_custom isolation test)
      - Implementation: call `update_legacy_dict(params.cfg, train_config)` and `update_legacy_dict(params.cfg, infer_config)` inside both `scripts/studies/sim_lines_4x/pipeline.py` and `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py` right before invoking the backend selectors so loader/model see the new gridsize/n_groups/intensity parameters.
      - Verification: rerun the Phase C2 runner for gs1_ideal + gs2_ideal (stable profiles) with `--prediction-scale-source least_squares`, regenerate `bias_summary.{json,md}`, and guard with the CLI smoke pytest selector; archive logs under `reports/2026-01-20T160000Z/`.
      Test: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`

### Notes & Risks
- Keep changes scoped; do not modify stable physics modules unless evidence pins regression there.

## Phase D -- Amplitude Bias Root Cause

**Problem Statement:** Even after CONFIG-001 bridging and padded-canvas fixes, every sim_lines_4x scenario undershoots ground-truth amplitude by ≈2.3–2.6× and needs ≈1.7–2.0× least-squares scaling to partially close the gap, while dose_experiments reconstructions at comparable doses did not exhibit this failure. Phase D reopens the initiative to find the actual workflow mismatch (loss configuration, normalization pipeline, hyperparameters, or architecture deltas) and recover dose_experiments-quality reconstructions instead of treating “no NaNs” as success.

**Open Hypotheses (tracked from fix_plan.md §DEBUG-SIM-LINES-DOSE-001 Phase D):**
- H-LOSS-WEIGHT — Loss weights/regression targets differ between pipelines.
- H-NORMALIZATION — Intensity normalization introduces bias in sim_lines_4x only.
- H-TRAINING-PARAMS — Hyperparameters (lr, epochs, batch size) diverge from legacy runs.
- H-ARCHITECTURE — Model/forward-path wiring changed between repositories.

### Checklist
- [x] D1: **Validate loss configuration parity vs dose_experiments.** *(COMPLETE 2026-01-20T121449Z — H-LOSS-WEIGHT ruled out)*  
      Extended `bin/compare_sim_lines_params.py` with runtime cfg capture and `--output-dose-loss-weights-markdown` flag. Regenerated all artifacts under `reports/2026-01-20T121449Z/`. **Corrected finding:** legacy dose_experiments does NOT set explicit loss weights under default NLL mode — it relies on `ptycho/params.cfg` defaults (`mae_weight=0.0, nll_weight=1.0`), which match sim_lines TrainingConfig defaults exactly.
      - [x] **D1a — Capture legacy runtime configs.** Executed with stubbed cfg for both `loss_fn='nll'` and `loss_fn='mae'`; persisted JSON + Markdown.
      - [x] **D1b — Fix the comparison CLI.** Added `--output-dose-loss-weights-markdown` flag and runtime cfg capture; conditional assignments labeled explicitly.
      - [x] **D1c — Re-evaluate the hypothesis.** Reran the diff; both pipelines use identical loss weights (`mae_weight=0.0, nll_weight=1.0`). H-LOSS-WEIGHT is ruled out.
      - Evidence path: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T121449Z/{dose_loss_weights.json,dose_loss_weights.md,loss_config_diff.md,loss_config_diff.json,legacy_params_cfg_defaults.json,pytest_cli_smoke.log,summary.md}`
      - Guard: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v` (1 passed)
- [x] D2: **Normalization pipeline parity.** *(PARTIALLY COMPLETE 2026-01-20T123330Z — normalization-invariant instrumentation landed)*
      - Once loss weights are reconciled/documented, add instrumentation (plan-local first) to log normalization gains for both the simulator and loader across sim_lines and the legacy dose_experiments scripts so we can prove whether scaling symmetry (spec-ptycho-core.md §Normalization Invariants) holds end-to-end.
      - Capture RawData → grouped → normalized → container stage distributions side-by-side with the legacy pipeline, including any pre-scaling applied inside loss wiring.
      - [x] D2a — *sim_lines telemetry refresh*: extend the plan-local runner/analyzer pipeline so `intensity_stats.json` explicitly records stage-level ratios (raw→grouped, grouped→normalized, normalized→prediction) plus the exact `normalize_data` gain values for each scenario. Confirm stage-measurement helpers emit both JSON and Markdown in the artifacts hub. *(COMPLETE — `intensity_stats.json` already emits ratios and normalize_gain)*
      - [ ] D2b — *dose_experiments parity capture*: add a dedicated CLI (plan-local) that replays the `dose_experiments_param_scan.md` configuration through the same simulation→group→normalize stack (via compatibility runner or locally replicated parameters) and writes `dose_normalization_stats.{json,md}` mirroring the sim_lines format so ratios can be compared without rerunning the legacy training loop.
      - [x] D2c — *Comparison report*: update `bin/analyze_intensity_bias.py` (or a new helper) so it ingests both sim_lines and dose_experiments stats, emits a Markdown table highlighting per-stage deltas vs the spec requirement, and flags whichever stage first breaks symmetry. The summary must cite `specs/spec-ptycho-core.md §Normalization Invariants`. *(COMPLETE 2026-01-20T123330Z — added `compute_normalization_invariant_check()` that multiplies raw→truth stage ratios, computes cumulative products, flags 5% tolerance violations, identifies primary deviation sources, and cites the spec. Evidence: `reports/2026-01-20T122937Z/bias_summary.{json,md}` show dose_legacy_gs2 fails with full_chain_product=1.986 and symmetry_violated=true)*
      Test: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`
- [ ] D3: **Hyperparameter delta audit.**
      - Diff nepochs, batch sizes, group limits, optimizer knobs, and scheduler behavior between dose_experiments and sim_lines_4x for each scenario profile; document any mismatches and run targeted re-trains (plan-local scope) if differences plausibly explain the amplitude collapse.
      - Archive training-history overlays plus config dumps under a fresh hub.
      Test: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`
- [ ] D4: **Architecture / loss-wiring verification.**
      - If loss weights + normalization still match legacy behavior, inspect the current TensorFlow training graph (loss composition, intensity scaler placement, probe application) against the `dose_experiments` script to identify code-level divergences; add focused diagnostic tests as needed.
      Test: `pytest -m integration`

### Notes & Risks
- Avoid touching `ptycho/model.py`, `ptycho/diffsim.py`, or other core physics modules without explicit approval (per CLAUDE.md directive #6). Start with plan-local instrumentation and documentation diffs.
- Legacy dose_experiments scripts are frozen to TensorFlow 1-era semantics; keep compatibility wrappers under `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/` and do not reintroduce TF addons as production deps.

## Artifacts Index
- Reports root: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/
- Latest run: <YYYY-MM-DDTHHMMSSZ>/
