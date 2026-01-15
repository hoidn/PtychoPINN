# Implementation Plan (Phased)

## Initiative
- ID: DEBUG-SIM-LINES-DOSE-001
- Title: Isolate sim_lines_4x vs dose_experiments sim->recon discrepancy
- Owner: Codex + user
- Spec Owner: docs/specs/spec-ptycho-workflow.md
- Status: pending

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
- [ ] A1: Extract dose_experiments codepath without checkout:
      - Use `git ls-tree -r dose_experiments` to locate files.
      - Use `git show dose_experiments:<path>` for sim, training, inference, stitching modules.
      - Record parameter defaults (probe_mask, probe_big, probe_scale, default_probe_scale, gridsize, nphotons, split, grouping).
      Test: N/A -- evidence capture only
- [ ] A2: Verify data-contract expectations for any RawData/NPZ outputs used in comparison.
      Test: N/A -- evidence capture only
- [ ] A3: Capture sim_lines_4x params snapshot with full config/params dump.
      Test: N/A -- evidence capture only
- [ ] A4: Compare parameter tables (dose_experiments vs sim_lines_4x) and log probe stats + intensity_scale.
      Test: N/A -- evidence capture only

### Notes & Risks
- Ensure CONFIG-001 sync before any param readout.
- Avoid editing core modules during evidence-only steps.

## Phase B -- Differential Experiments
### Checklist
- [ ] B1: Grid vs nongrid A/B in current codebase with identical seeds + probe settings.
      Test: N/A -- evidence run; log training stability + recon size
- [ ] B2: Probe normalization A/B (set_default_probe path vs make_probe path) holding everything else constant.
      Test: N/A -- evidence run; log probe stats + intensity_scale
- [ ] B3: Grouping A/B (neighbor_count, group_count, gridsize) with fixed seeds to compare offsets and grouping shapes.
      Test: N/A -- evidence run; log coords ranges and offset distributions
- [ ] B4: Reassembly A/B using a fixed synthetic container to isolate stitch math (M/padded_size).
      Test: N/A -- evidence run; add test in Phase C if fix touches core logic

### Notes & Risks
- If grid works and nongrid fails, focus on nongrid grouping/offsets.
- If both fail, suspect core regression in probe normalization, reassembly, or translation.

## Phase C -- Fix + Verification
### Checklist
- [ ] C1: Implement minimal fix (probe normalization alignment, reassembly size handling, or workflow sync).
      Test: If code changes, add targeted pytest selector in this phase; otherwise N/A with rationale
- [ ] C2: Rerun gs2_ideal + gs1_ideal and confirm:
      - no NaNs
      - expected canvas size
      - visual inspection pass
      Test: N/A -- evidence run
- [ ] C3: Update docs/fix_plan.md Attempts History with evidence and root-cause summary.
      Test: N/A

### Notes & Risks
- Keep changes scoped; do not modify stable physics modules unless evidence pins regression there.

## Artifacts Index
- Reports root: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/
- Latest run: <YYYY-MM-DDTHHMMSSZ>/
