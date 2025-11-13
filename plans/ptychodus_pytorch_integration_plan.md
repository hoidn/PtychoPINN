<plan_update version="1.0">
  <trigger>Resume the paused PyTorch↔Ptychodus parity initiative so Ralph can work the Phase 1 bridge/persistence gaps again.</trigger>
  <focus_id>INTEGRATE-PYTORCH-PARITY-001</focus_id>
  <documents_read>docs/index.md, docs/specs/spec-ptychopinn.md, docs/specs/spec-ptycho-core.md, docs/specs/spec-ptycho-runtime.md, docs/specs/spec-ptycho-interfaces.md, docs/specs/spec-ptycho-workflow.md, docs/specs/spec-ptycho-tracing.md, docs/specs/spec-ptycho-config-bridge.md, docs/specs/spec-ptycho-conformance.md, docs/specs/overlap_metrics.md, docs/architecture.md, docs/workflows/pytorch.md, docs/findings.md, plans/ptychodus_pytorch_integration_plan.md, plans/pytorch_integration_test_plan.md, PYTORCH_INVENTORY_SUMMARY.txt, docs/fix_plan.md</documents_read>
  <current_plan_path>plans/ptychodus_pytorch_integration_plan.md</current_plan_path>
  <proposed_changes>- Document the reactivation scope with a Phase R immediate-focus checklist (config bridge invocation + persistence shim + regression test gate).
- Record the new reports hub path and testing guard so downstream loops land evidence consistently.
- Align the Do Now with the inventory quick wins (update_legacy_dict wiring, n_groups default, targeted pytest).</proposed_changes>
  <impacts>Re-enabling parity work pulls Ralph away from export/docs tasks and reintroduces PyTorch regression risk; requires pytest parity guard and hub evidence on every loop; future attempts must honor POLICY-001 torch requirements.</impacts>
  <ledger_updates>Add a high-priority focus row to docs/fix_plan.md plus a matching input.md brief and galph_memory entry pointing at this plan + hub.</ledger_updates>
  <status>approved</status>
</plan_update>

## Ptychodus ↔ PtychoPINN (PyTorch) Integration Plan

### Immediate Focus — Phase R (Bridge Reactivation, 2025-11-13)

1. **Wire the configuration bridge in runtime entry points.** In `ptycho_torch/train.py` and `ptycho_torch/inference.py`, instantiate the canonical dataclasses via `config_bridge`, call `update_legacy_dict(params.cfg, config)` before touching `RawData`/loader modules, and raise actionable errors when required overrides (e.g., `train_data_file`, `n_groups`) are missing.
2. **Backfill spec-mandated defaults in `ptycho_torch/config_params.py`.** Ensure `n_groups`, `test_data_file`, `gaussian_smoothing_sigma`, and related knobs required by `specs/ptychodus_api_spec.md §§5.1-5.3` exist with TensorFlow-parity defaults so the bridge can populate legacy consumers.
3. **Provide a native persistence shim.** Until the full `.h5.zip` adapter lands, teach `ptycho_torch/api/base_api.py::PtychoModel.save_pytorch()` to emit a Lightning checkpoint + manifest bundle and document how `load_*` surfaces rehydrate configs; keep the implementation small but spec-compliant (§4.6).
4. **Regression gate.** Every loop must run `pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity -vv` (or a stricter subset) and upload logs under the active hub `plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/`.
5. **Exit criteria for reactivation:** (a) update_legacy_dict invoked in both CLI entry points, (b) config defaults + persistence shim merged, (c) targeted pytest selector green with evidence, (d) hub `analysis/artifact_inventory.txt` + `summary/summary.md` list the code/test paths touched.

### 1. Scope & Goals

- Deliver a PyTorch implementation of the PtychoPINN backend that satisfies every contract defined in `specs/ptychodus_api_spec.md`.
- Keep the existing TensorFlow path fully operational while allowing runtime backend selection from ptychodus.
- Ensure configuration, data, training, inference, and persistence semantics remain identical for both backends so that third-party tooling can operate without divergence.
- **Dual Backend Surface:** The PyTorch implementation provides both a high-level API layer (`ptycho_torch/api/base_api.py`) for orchestration and low-level module access for direct integration. The integration strategy must select between these surfaces based on maintainability and spec alignment.
- **Configuration Bridge as First Milestone:** Establishing dataclass-driven configuration synchronization with `ptycho.params.cfg` is the critical dependency for all downstream workflows and must be completed in Phase 1 before data pipeline or training integration.

### 2. Authoritative References

| Topic | Spec Section | Key Files |
| --- | --- | --- |
| Reconstructor lifecycle & behaviour | `specs/ptychodus_api_spec.md:127-211` | `ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py`, `ptycho/workflows/components.py` |
| Configuration surface & legacy bridge | `specs/ptychodus_api_spec.md:20-125`, `213-291` | `ptycho/config/config.py`, `ptycho/params.py` |
| Data ingestion & grouping | `specs/ptychodus_api_spec.md:150-190` | `ptycho/raw_data.py`, `ptycho/loader.py`, `ptycho_torch/dset_loader_pt_mmap.py` |
| Model persistence contract | `specs/ptychodus_api_spec.md §4.6` | `ptycho/model_manager.py` |

### 3. Deliverables

1. PyTorch-backed configuration bridge that updates `ptycho.params.cfg` via the same dataclass pipeline as TensorFlow.
   - Configuration schema harmonization (resolve field name/type divergence: `grid_size` → `gridsize`, `mode` → `model_type`, add missing fields)
   - `KEY_MAPPINGS` translation layer for legacy dot-separated keys
   - Parameterized tests verifying all 75+ config fields propagate correctly
2. Data adapters producing grouping outputs and tensor layouts compatible with the reconstructor contract.
   - `RawDataTorch` shim delegating to `ptycho/raw_data.py` with memory-map bridging
   - `PtychoDataContainerTorch` matching TensorFlow tensor shapes/dtypes
   - Shared NPZ contract compliance with `datagen/` package
3. PyTorch model wrappers exposing the inference and training signatures referenced in the spec, including intensity scaling behaviour.
   - Barycentric reassembly modules with numeric parity validation vs TensorFlow
   - Multi-GPU inference path with DataParallel support
4. Workflow orchestration (training, inference, export) usable by ptychodus without changes to the UI layer.
   - Lightning + MLflow orchestration adapter or lower-level API exposure
   - Multi-stage training logic (stage_1/2/3_epochs with physics weight scheduling)
5. Save/load routines compatible with the existing archive semantics, including `params.cfg` restoration.
   - Lightning checkpoint + MLflow persistence adapter
   - Archive shim bundling `.ckpt` + `params.cfg` into `.h5.zip` format
   - Backend auto-detection for dual-format archives
6. Extended reconstructor capable of selecting the PyTorch backend, with regression tests demonstrating parity across backends.
   - Dual-backend reconstructor with runtime selection
   - Integration test suite (config bridge, data pipeline, training, persistence, Ptychodus integration)

### 4. Phased Work Breakdown

#### Phase 0 – Discovery & Design (1 sprint)

- Audit `ptycho_torch/` modules for reusable components and gaps relative to spec references.
- Confirm the capability to toggle backends from `PtychoPINNReconstructorLibrary` (`ptychodus/src/ptychodus/model/ptychopinn/core.py:22-59`).
- Produce a component parity map that lists each TensorFlow workflow dependency (`ptycho.probe`, `ptycho.train_pinn`, `ptycho.tf_helper`, `run_cdi_example`, `ModelManager`, etc.) alongside the existing or missing `ptycho_torch` counterpart. Flag gaps that need new code versus thin shims.
- Produce sequence diagrams mapping ptychodus calls to PyTorch equivalents, referencing both the reconstructor contract (`specs/ptychodus_api_spec.md §4`) and the parity map.
- **Decision Gate: High-Level API vs Low-Level Integration** — Determine whether `ptychodus` should invoke the `ptycho_torch/api/` layer (which provides `ConfigManager`, `PtychoModel`, `Trainer`, `InferenceEngine` classes with MLflow orchestration) or bypass it for direct module calls. Document ownership of MLflow persistence strategy and Lightning dependency policy. See `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T025000Z/delta_log.md` Delta-2 for analysis.
- **API Package Structure Documentation** — Document the new `ptycho_torch/api/base_api.py` layer (994 lines), which abstracts Lightning orchestration and provides MLflow-centric persistence (`save_mlflow()`, `load_from_mlflow()`). Cross-reference with `specs/ptychodus_api_spec.md:129-212` to verify contract alignment.
- Acceptance: signed-off architecture note describing module boundaries, extension points, and integration surface decision (API layer or low-level modules).

**Phase 0 Artifact – TensorFlow ↔ PyTorch Parity Map**

| TensorFlow / Legacy Component | Responsibility | PyTorch Counterpart | Status / Gaps |
| --- | --- | --- | --- |
| `ptycho.probe` (probe guess setup) | Load/stash probe guesses, handle masks | `ptycho_torch/dset_loader_pt_mmap.py:get_probes`, `ptycho_torch/model.ProbeIllumination` | Core functionality exists; need shim so reconstructor can reuse without direct CLI assumptions. |
| `ptycho.raw_data.RawData.generate_grouped_data` | NPZ ingestion, neighbor grouping | `ptycho_torch/dset_loader_pt_mmap.py`, `ptycho_torch/patch_generator.py` | Grouping logic present; add `RawDataTorch` wrapper to consume ptychodus-exported NPZs and expose spec-compliant dictionary. |
| `ptycho.loader.PtychoDataContainer` | Tensor packaging for model input | _No direct equivalent_ | Must implement `PtychoDataContainerTorch` mirroring TensorFlow shapes/dtypes. |
| `ptycho.tf_helper` utilities | Patch reassembly, translation, diffraction | `ptycho_torch/helper.py` | Most helpers ported; verify APIs and add thin adapters where TensorFlow-specific signatures differ. |
| `ptycho.model` (Keras models) | Core PINN network + loss wiring | `ptycho_torch/model.py` | Architectural parity achieved; integrate with dataclass-driven configs and inference wrapper. |
| `ptycho.train_pinn`, `run_cdi_example`, `train_cdi_model`, `save_outputs` | End-to-end training/inference orchestration | _Missing_ | Need PyTorch orchestration layer adhering to reconstructor contract and existing CLI expectations. |
| `ptycho.model_manager`, `load_inference_bundle` | Model persistence & params restoration | _Missing_ | Design PyTorch archive format (or adaptor) compatible with current loader side effects. |
| TensorFlow `Model.predict` signature | Inference entry `model.predict([X * scale, offsets])` | `ptycho_torch/model.PtychoPINN`, `ptycho_torch/train.PtychoPINN.forward` | Forward path available; require wrapper that matches TensorFlow call signature and intensity scaling semantics. |
| `update_legacy_dict` usage through dataclasses | Config propagation to `params.cfg` | _Pending integration_ | Reuse existing dataclasses; replace singleton configs in `ptycho_torch/config_params.py` with dataclass-backed factories. |

#### Phase 1 – Configuration Parity (1 sprint)

- **Task 1.1**: Introduce shared dataclasses for PyTorch by importing `ModelConfig`, `TrainingConfig`, `InferenceConfig` instead of singleton dictionaries. Map existing default dictionaries in `ptycho_torch/config_params.py` into dataclass factory helpers.
  - **Critical Schema Harmonization Required:** Current `ptycho_torch/config_params.py` uses divergent schema (e.g., `grid_size: Tuple[int, int]` vs spec-mandated `gridsize: int`, `mode: 'Supervised' | 'Unsupervised'` vs `model_type: 'pinn' | 'supervised'`). See `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T025000Z/delta_log.md` Delta-1 for full field mismatch inventory.
  - **Missing Spec-Mandated Fields:** Add `gaussian_smoothing_sigma`, `probe_scale`, `pad_object` to PyTorch config classes per `specs/ptychodus_api_spec.md:220-273`.
  - **Configuration Schema Mapping Table:** Create a mapping table documenting every PyTorch config field → spec-required field transformation (e.g., `grid_size[0]` → `gridsize`, `mode` → `model_type`) to guide implementation and testing.
- **Task 1.2**: Implement a PyTorch-friendly `update_legacy_dict()` invoker that calls the existing bridge (`specs/ptychodus_api_spec.md §§2-3`) immediately after dataclass instantiation.
  - **Add KEY_MAPPINGS for PyTorch:** Extend or create equivalent to `ptycho/config/config.py:KEY_MAPPINGS` to translate modern field names to legacy dot-separated keys (e.g., `object_big` → `object.big`, `probe_trainable` → `probe.trainable`).
  - **params.cfg Population Verification:** Ensure `ptycho.params.cfg` is correctly populated for all fields consumed by downstream modules (`ptycho/raw_data.py:365`, `ptycho/loader.py:178-181`, `ptycho/model.py:280`).
- **Task 1.3**: Auto-generate a parity checklist from every field in the configuration tables (`specs/ptychodus_api_spec.md §5`) and translate it into parameterized tests that set and round-trip values such as `probe_scale`, `gaussian_smoothing_sigma`, `sequential_sampling`, and other newly documented knobs.
  - **Test Coverage:** Parameterized tests must verify all 75+ fields across `ModelConfig`, `TrainingConfig`, `InferenceConfig` tables (§5.1-5.3) propagate correctly through dataclass → `update_legacy_dict` → `params.cfg` → downstream consumers.
- Acceptance: automated parity tests confirm each documented field maps identically into `ptycho.params.cfg`, and comparative snapshots show no differences between TensorFlow and PyTorch updates for a matrix of representative configurations.

#### Phase 2 – Data Ingestion & Grouping (2 sprints)

- **Task 2.1**: Implement a `RawDataTorch` shim that consumes the NPZ schema produced by `export_training_data()` (`specs/ptychodus_api_spec.md §4.5`) and exposes methods mirroring `RawData.generate_grouped_data` semantics.
  - **RawDataTorch Wrapper Scope:** Create adapter delegating to existing `ptycho/raw_data.py` for neighbor-aware grouping logic while bridging to PyTorch's memory-mapped dataset infrastructure.
  - **Memory-Map to Cache Bridging:** Reconcile PyTorch's `ptycho_torch/dset_loader_pt_mmap.py` memory-mapped tensor approach with TensorFlow's `.groups_cache.npz` caching strategy. Ensure cache reuse across backends for performance parity.
- **Task 2.2**: Map the Torch memory-mapped dataset outputs (`ptycho_torch/dset_loader_pt_mmap.py:1-260`) onto the dictionary keys enumerated in the contract (`specs/ptychodus_api_spec.md §4.3`).
  - **TensorDict Format Conversion:** PyTorch dataloader provides `TensorDict` format; must expose dictionary with keys `diffraction`, `coords_offsets`, `coords_relative`, `Y`, `nn_indices` to satisfy spec.
- **Task 2.3**: Provide a `PtychoDataContainerTorch` that matches TensorFlow tensor shapes and dtype expectations, ensuring compatibility with downstream reassembly helpers.
- **Task 2.4**: Document `ptycho_torch/datagen/` package for synthetic data parity.
  - **Synthetic Data Generation:** `ptycho_torch/datagen/datagen.py` provides `from_simulation()`, `simulate_multiple_experiments()` for dataset creation with Poisson scaling and beamstop. Verify outputs conform to `specs/data_contracts.md` NPZ schema.
  - **Shared NPZ Contract:** Ensure `datagen/` package produces NPZ files consumable by both TensorFlow and PyTorch backends without format divergence. Cross-reference with TensorFlow `ptycho.diffsim` equivalence for physics parity.
  - **Experimental Data Extraction:** Note `generate_data_from_experiment()` capability for supervised label extraction from experimental datasets, bypassing simulation workflow.
- Acceptance: integration tests load identical NPZ inputs through both backends and compare grouped dictionary keys, shapes, and coordinate values (float tolerances defined up front). Synthetic datasets from `datagen/` pass NPZ validation against `specs/data_contracts.md`.

#### Phase 3 – Inference Entry Point (1 sprint)

- **Task 3.1**: Wrap the PyTorch model so `predict([diffraction * intensity_scale, local_offsets])` is supported (per `specs/ptychodus_api_spec.md §4.4`).
- **Task 3.2**: Bridge the PyTorch intensity scaler to `params.cfg['intensity_scale']` and `params.cfg['intensity_scale.trainable']` using the logic in `ptycho_torch/model.py:480-565`.
- **Task 3.3**: Resolve probe initialisation: either reuse `ptycho.probe.set_probe_guess` via thin adapters or port equivalent functionality into `ptycho_torch` so the grouped data pipeline receives the expected probe tensor (`specs/ptychodus_api_spec.md §4.3`).
- **Task 3.4**: Document barycentric reassembly modules and establish parity with TensorFlow stitching.
  - **Alternative Reassembly Implementation:** PyTorch provides `ptycho_torch/reassembly_alpha.py`, `reassembly_beta.py`, `reassembly.py` implementing vectorized barycentric accumulator for patch stitching (alternative to `ptycho.tf_helper.reassemble_position`). See `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T025000Z/delta_log.md` Delta-4.
  - **Multi-GPU Inference Path:** `reassembly_alpha.py:VectorizedBarycentricAccumulator` includes DataParallel support for multi-GPU patch stitching with performance profiling (inference time vs assembly time tracking).
  - **Parity Requirements:** Establish numeric comparators to verify PyTorch barycentric output matches TensorFlow `reassemble_position()` outputs within acceptable tolerances on synthetic test fixtures.
  - **Adaptation Strategy:** Either (A) adapt `ptycho.tf_helper.reassemble_position` usage by providing data conversion utilities, or (B) validate PyTorch reassembly parity and use native implementation. Document decision and testing approach.
- Acceptance: reconstructor `reconstruct()` executes end-to-end with the PyTorch backend on sample data, successfully initialises probes, and produces an object array of the expected shape and dtype. Numeric parity tests confirm stitching outputs match TensorFlow within defined tolerances.

#### Phase 4 – Training Workflow Parity (2 sprints)

- **Task 4.1**: Expose orchestrators analogous to `run_cdi_example`, `train_cdi_model`, and `save_outputs` that operate on PyTorch models (`specs/ptychodus_api_spec.md §4.5`).
  - **Lightning + MLflow Orchestration Divergence:** PyTorch training uses `ptycho_torch/train.py` with PyTorch Lightning `Trainer` (callbacks, DataModule, DDP strategy) and MLflow autologging, diverging from TensorFlow's `ptycho.workflows.components.run_cdi_example()` direct orchestration. See `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T025000Z/delta_log.md` Delta-5.
  - **Multi-Stage Training Logic:** `train.py` embeds multi-stage training (`stage_1/2/3_epochs` with physics weight scheduling) in Lightning orchestration. Ensure spec alignment or document as PyTorch-specific enhancement.
  - **Orchestration Surface Decision:** Clarify whether `ptychodus` integration will invoke Lightning trainer directly or require lower-level model API for training. Cross-reference Phase 0 decision gate (API layer vs low-level integration).
- **Task 4.2**: Update `ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py:229-269` to dispatch to the PyTorch workflow when selected, preserving logging and output directory semantics.
- **Task 4.3**: Ensure NPZ exports from `export_training_data()` remain consumable by the PyTorch training path without schema divergence.
- **Task 4.4**: Integrate the probe strategy selected in Phase 3 into the training pipeline so probe guesses are initialised consistently before batching/grouping.
- Acceptance: training end-to-end test executes on a reduced dataset, writes artifacts to `output_dir`, restores the trained model for immediate inference, and probe initialisation runs without backend-specific fallbacks.

#### Phase 5 – Model Persistence & Archives (1 sprint)

- **Task 5.1**: Extend `ptycho.model_manager` with PyTorch-aware branches (or introduce a companion module) that produce bundle metadata alongside TensorFlow artefacts while sharing the `wts.h5.zip` packaging contract (`specs/ptychodus_api_spec.md §4.6`).
  - **Lightning Checkpoint + MLflow Workflow:** Current PyTorch implementation uses Lightning `.ckpt` format with MLflow artifact logging (`ptycho_torch/train.py:238-240`). Design adapter to wrap Lightning checkpoints in `.h5.zip`-compatible archives.
  - **Archive Shim Proposal:** Introduce persistence adapter that bundles Lightning checkpoint + `ptycho.params.cfg` snapshot + custom layer registry into unified `.h5.zip` format satisfying `MODEL_FILE_NAME = 'wts.h5.zip'` contract.
  - **MLflow Dependency Mitigation:** Define policy for optional MLflow/Lightning dependencies in CI and production environments. Consider graceful degradation or alternative persistence paths when MLflow unavailable.
- **Task 5.2**: Define the PyTorch payload layout (e.g., `diffraction_to_obj.pt`, optional optimizer state, serialized custom layers) and how it coexists with Keras assets inside the archive; document format versioning.
  - **Dual-Format Archive Structure:** Document how Lightning checkpoints and TensorFlow SavedModel assets coexist within `.h5.zip`, including versioning strategy to prevent loader collisions.
- **Task 5.3**: Update `load_inference_bundle` to inspect archive contents, dispatch to the appropriate loader (TensorFlow vs PyTorch), and restore `params.cfg` side effects for both paths.
  - **Backend Auto-Detection:** Implement archive introspection logic to detect TensorFlow vs PyTorch payloads and dispatch to appropriate loader without requiring user-specified backend flag.
- **Task 5.4**: Provide migration tooling or guidance for existing archives and adjust reconstructor file filters/tooltips to reflect dual-backend support.
- Acceptance: automated save→load tests confirm PyTorch models round-trip via `wts.h5.zip`, `load_inference_bundle` returns backend-specific objects with `params.cfg` restored, and legacy TensorFlow bundles remain unaffected.

#### Phase 6 – Reconstructor & UI Integration (1 sprint)

- **Task 6.1**: Extend `PtychoPINNReconstructorLibrary` to register PyTorch variants in addition to TensorFlow (`specs/ptychodus_api_spec.md §4.1`).
- **Task 6.2**: Provide backend selection controls (UI toggle or configuration flag) and ensure file filters remain accurate.
- **Task 6.3**: Update logging to indicate the active backend for traceability.
- Acceptance: switching between backends at runtime works without restarting ptychodus, and both paths respect the reconstructor contract.

#### Phase 7 – Validation & Regression Testing (1 sprint)

- **Task 7.1**: Develop regression suites comparing TensorFlow and PyTorch outputs on shared fixtures (object reconstructions, grouped data, loss curves) with acceptable tolerance thresholds.
- **Task 7.2**: Add integration tests covering save/load, inference, and training flows to prevent contract regressions.
- **Task 7.3**: Document manual verification steps aligned with the spec (e.g., verifying key mappings, checking NPZ contents).
- Acceptance: CI passes with new test coverage, and manual checklist is signed off.

#### Phase 8 – Spec & Ledger Synchronization

- **Task 8.1**: Update `specs/ptychodus_api_spec.md` to document PyTorch backend semantics once implementation finalizes.
  - **Configuration Schema Amendment:** If dual-schema approach is adopted (TensorFlow dataclasses vs PyTorch singletons), document both schemas and translation layer in spec §2-3 and field tables (§5.1-5.3).
  - **Persistence Contract Extension:** Document PyTorch archive format (Lightning checkpoint wrapper), backend auto-detection logic, and coexistence with TensorFlow bundles in §4.6.
  - **Workflow Divergence Notes:** Capture Lightning/MLflow orchestration differences vs TensorFlow direct orchestration in §4.5 (training workflow).
- **Task 8.2**: Update `docs/findings.md` knowledge ledger with PyTorch-specific lessons learned.
  - Document config schema harmonization patterns
  - Record reassembly parity validation approach
  - Capture persistence adapter design decisions
- **Task 8.3**: Cross-reference integration plan with downstream initiatives.
  - Ensure `plans/active/INTEGRATE-PYTORCH-001/implementation.md` consumes refreshed plan sections
  - Coordinate with `plans/pytorch_integration_test_plan.md` (TEST-PYTORCH-001) for fixture requirements
- **Task 8.4**: Update `docs/workflows/pytorch.md` with integration-specific usage patterns.
  - Document how to invoke PyTorch backend from Ptychodus
  - Provide configuration examples for dual-backend scenarios
  - Add troubleshooting guidance for common integration issues
- Acceptance: Spec updates are merged, knowledge ledger entries are validated against actual implementation outcomes, and downstream initiative plans reference refreshed canonical plan sections.

### 5. Dependencies & Risks

- **TensorFlow helper reuse**: section `4.4` of the spec assumes TensorFlow-specific helpers. Replacing them may require heavy refactoring or careful interop wrappers.
- **Archive compatibility**: deviating from `wts.h5.zip` could break existing saved models; introduce versioning and migration guidance if change is unavoidable.
- **Performance parity**: PyTorch data pipeline must be profiled against TensorFlow to ensure similar throughput, particularly for grouping operations.
- **API layer drift vs low-level integration**: The new `ptycho_torch/api/` high-level layer (ConfigManager, PtychoModel, InferenceEngine) may diverge from low-level module interfaces, creating maintenance burden if Ptychodus integration bypasses the API layer. Decision required in Phase 0 to commit to either (A) API-first integration with formal contract or (B) low-level integration with API layer as optional convenience. See `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T025000Z/delta_log.md` Delta-2 for analysis.
- **Configuration schema divergence**: PyTorch config fields (`grid_size`, `mode`) differ from spec-mandated TensorFlow fields (`gridsize`, `model_type`), risking silent contract violations. Requires explicit harmonization strategy (refactor PyTorch schema vs dual-schema documentation) to prevent downstream integration failures.
- **Lightning/MLflow dependency policy**: PyTorch training relies on Lightning and MLflow; CI/production environments may lack these dependencies. Requires mitigation strategy (optional dependency with graceful degradation, or mandatory dependency with environment setup guidance).
- **Reassembly parity validation complexity**: PyTorch barycentric reassembly diverges from TensorFlow `tf_helper.reassemble_position`; numeric parity testing across different tensor layouts and interpolation strategies may reveal edge cases requiring tolerance tuning or algorithm adjustments.

### 6. Communication & Handoff

- Weekly sync to review progress against the spec sections cited above.
- Maintain a living checklist mapping completed tasks to spec requirements to guarantee no contracts are overlooked.
- Final deliverable includes updated documentation (README, user guides) explaining backend selection and any new configuration knobs.
