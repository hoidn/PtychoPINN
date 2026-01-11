# Implementation Plan

## Initiative
- ID: SIM-LINES-4X-001
- Title: Four-scenario nongrid sim + TF reconstruction (lines object)
- Owner: Codex + user
- Spec Owner: specs/spec-inference-pipeline.md (inference contract)
- Status: pending

## Goals
- Simulate four datasets from a synthetic "lines" object: (gridsize=1,2) x (idealized probe, integration-test probe).
- Run TF training + inference on the test split only, with a contiguous spatial split.

## Phases Overview
- Phase A -- Design: lock parameters, data flow, and module boundaries.
- Phase B -- Implementation: build core pipeline + thin per-scenario runners.
- Phase C -- Validation: run all scenarios and confirm outputs.

## Exit Criteria
1. Four scenario runs complete: gs1/gs2 x idealized/integration probe.
2. Each run produces a trained bundle + reconstructed amplitude/phase for the test split.
3. Run logs capture key parameters (N=64, object_size=392, split=0.5, test_n=1000).
4. Workflow ledger and summary artifacts updated (docs/fix_plan.md + plans/active/SIM-LINES-4X-001/summary.md).
5. Test registry updated if any tests are added.

## Compliance Matrix (Mandatory)
- [ ] Spec Constraint: specs/data_contracts.md -- only if NPZ splits are persisted.
- [ ] Finding/Policy ID: CONFIG-001 (update_legacy_dict before legacy usage).
- [ ] Finding/Policy ID: MODULE-SINGLETON-001 (fresh model per run).
- [ ] Finding/Policy ID: ANTIPATTERN-001 (no import-time side effects).
- [ ] Finding/Policy ID: BUG-TF-001 (gridsize sync before grouping).

## Spec Alignment
- Normative Spec: specs/spec-inference-pipeline.md
- Supporting Spec: specs/ptychodus_api_spec.md (bundle loading + inference contract)
- Supporting Spec: specs/data_contracts.md (only if NPZ splits are produced)
- Key Clauses: load->infer->stitch contract, bundle load path, config restore (CONFIG-001), NPZ keys/dtypes if persisted.

## Architecture / Interfaces
- Key Data Types / Protocols:
  - TrainingConfig, RawData, PtychoDataContainer
- Boundary Definitions:
  - core pipeline module (pure functions, adapters over ptycho.workflows) -> thin scenario runner scripts (fresh process).
- Reuse Story:
  - Training: ptycho.workflows.components.train_cdi_model via backend_selector (TF only).
  - Inference: backend_selector.load_inference_bundle_with_backend + nbutils.reconstruct_image + tf_helper.reassemble_position.
  - Persistence: model bundle + recon images only; no new orchestration beyond thin wrappers.
- Sequence Sketch (Happy Path):
  - Build lines object -> build probe -> nongrid simulate -> spatial split -> train -> save bundle -> load bundle -> infer test split -> save outputs.
- Data-Flow Notes:
  - RawData (diff3d + coords) -> grouped data -> TF model -> stitched recon outputs.
  - Persistence policy locked: in-memory train/test split; only model bundle and recon images saved (no NPZ splits unless explicitly added later).

## Context Priming (read before edits)
- Primary docs/specs to re-read:
  - docs/DATA_GENERATION_GUIDE.md (nongrid pipeline + grouping)
  - docs/DEVELOPER_GUIDE.md (inference pipeline patterns + params ordering)
  - docs/architecture_inference.md (load->infer->stitch flow)
  - docs/COMMANDS_REFERENCE.md (workflow guardrails)
  - docs/debugging/QUICK_REFERENCE_PARAMS.md (CONFIG-001, MODULE-SINGLETON-001)
  - docs/findings.md (required pre-read)
  - specs/ptychodus_api_spec.md (bundle load/inference contract)
  - specs/data_contracts.md (only if persisting NPZ splits)
- Required findings/case law:
  - CONFIG-001, MODULE-SINGLETON-001, ANTIPATTERN-001, BUG-TF-001
- Related telemetry/attempts: none
- Data dependencies to verify:
  - Integration probe source: ptycho/datasets/Run1084_recon3_postPC_shrunk_3.npz

## Phase A -- Design
### Checklist
- [x] A0: Create/update plans/active/SIM-LINES-4X-001/summary.md (turn summary format).
- [x] A1: Append initiative entry to docs/fix_plan.md (scope + next action).
- [x] A2: No tests added; test strategy not required for this loop.
- [x] A3: Confirm lines object generator: ptycho.diffsim.sim_object_image with data_source='lines'.
- [x] A4: Lock parameters:
  - N=64
  - object_size=392
  - split axis: y
  - split fraction: 0.5
  - base_total_images=2000 (gridsize=1)
  - group_count=1000 per split
  - gridsize scaling: total_images = base_total_images * gridsize^2
- [x] A5: Lock persistence + inference entrypoint:
  - In-memory train/test split; no NPZ split files.
  - Save model bundle + recon images; inference via load_inference_bundle_with_backend + nbutils.reconstruct_image + tf_helper.reassemble_position.
- [x] A6: Define core API signatures and output layout per scenario.

### Dependency Analysis (Required for Refactors)
- Touched Modules: new core pipeline module, new scenario runner scripts.
- Reused Modules (no changes planned): ptycho/nongrid_simulation.py, ptycho/diffsim.py, ptycho/workflows/components.py, ptycho/workflows/backend_selector.py, ptycho/nbutils.py, ptycho/tf_helper.py.
- Circular Import Risks: keep core module independent of CLI modules to avoid side effects.
- State Migration: all params.cfg updates occur inside per-scenario process via update_legacy_dict.

### Notes & Risks
- nongrid simulation currently relies on RawData.from_simulation (gridsize=1 only). For gs2, simulate raw data with gridsize=1 then group with gridsize=2 at train/infer time.
- Stable-module policy: no changes planned for ptycho/diffsim.py, ptycho/model.py, or ptycho/tf_helper.py.

## Phase B -- Implementation
### Checklist
- [x] B1: Implement core pipeline module with pure functions wrapping existing workflow components (no new orchestration).
- [x] B2: Implement thin runner scripts per scenario (fresh process, minimal args).
- [x] B3: Enforce CONFIG-001 before grouping or model init.
- [x] B4: Bundle save + inference helper wiring (TF only, via backend_selector + nbutils.reconstruct_image + tf_helper.reassemble_position).

### Notes & Risks
- Ensure model singletons refreshed per run; avoid shared imports across scenarios.

## Phase C -- Validation
### Checklist
- [ ] C1: Run four scenarios (gs1 total_images=2000, gs2 total_images=8000; equal splits).
- [ ] C2: Verify output artifacts (bundle, reconstructed_amplitude.png, reconstructed_phase.png).
- [ ] C3: Capture per-scenario run summaries.
- [ ] C4: Update docs/index.md and any relevant workflow READMEs if new scripts/flows are introduced.

### Notes & Risks
- If recon outputs are blank, verify split contiguity and nphotons scaling.

## Artifacts Index
- Reports root: plans/active/SIM-LINES-4X-001/reports/
- Latest run: <YYYY-MM-DDTHHMMSSZ>/
