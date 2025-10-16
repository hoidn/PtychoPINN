## Ptychodus ↔ PtychoPINN (PyTorch) Integration Plan

### 1. Scope & Goals

- Deliver a PyTorch implementation of the PtychoPINN backend that satisfies every contract defined in `specs/ptychodus_api_spec.md`.
- Keep the existing TensorFlow path fully operational while allowing runtime backend selection from ptychodus.
- Ensure configuration, data, training, inference, and persistence semantics remain identical for both backends so that third-party tooling can operate without divergence.

### 2. Authoritative References

| Topic | Spec Section | Key Files |
| --- | --- | --- |
| Reconstructor lifecycle & behaviour | `specs/ptychodus_api_spec.md:127-211` | `ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py`, `ptycho/workflows/components.py` |
| Configuration surface & legacy bridge | `specs/ptychodus_api_spec.md:20-125`, `213-291` | `ptycho/config/config.py`, `ptycho/params.py` |
| Data ingestion & grouping | `specs/ptychodus_api_spec.md:150-190` | `ptycho/raw_data.py`, `ptycho/loader.py`, `ptycho_torch/dset_loader_pt_mmap.py` |
| Model persistence contract | `specs/ptychodus_api_spec.md §4.6` | `ptycho/model_manager.py` |

### 3. Deliverables

1. PyTorch-backed configuration bridge that updates `ptycho.params.cfg` via the same dataclass pipeline as TensorFlow.
2. Data adapters producing grouping outputs and tensor layouts compatible with the reconstructor contract.
3. PyTorch model wrappers exposing the inference and training signatures referenced in the spec, including intensity scaling behaviour.
4. Workflow orchestration (training, inference, export) usable by ptychodus without changes to the UI layer.
5. Save/load routines compatible with the existing archive semantics, including `params.cfg` restoration.
6. Extended reconstructor capable of selecting the PyTorch backend, with regression tests demonstrating parity across backends.

### 4. Phased Work Breakdown

#### Phase 0 – Discovery & Design (1 sprint)

- Audit `ptycho_torch/` modules for reusable components and gaps relative to spec references.
- Confirm the capability to toggle backends from `PtychoPINNReconstructorLibrary` (`ptychodus/src/ptychodus/model/ptychopinn/core.py:22-59`).
- Produce a component parity map that lists each TensorFlow workflow dependency (`ptycho.probe`, `ptycho.train_pinn`, `ptycho.tf_helper`, `run_cdi_example`, `ModelManager`, etc.) alongside the existing or missing `ptycho_torch` counterpart. Flag gaps that need new code versus thin shims.
- Produce sequence diagrams mapping ptychodus calls to PyTorch equivalents, referencing both the reconstructor contract (`specs/ptychodus_api_spec.md §4`) and the parity map.
- Acceptance: signed-off architecture note describing module boundaries and extension points.

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
- **Task 1.2**: Implement a PyTorch-friendly `update_legacy_dict()` invoker that calls the existing bridge (`specs/ptychodus_api_spec.md §§2-3`) immediately after dataclass instantiation.
- **Task 1.3**: Auto-generate a parity checklist from every field in the configuration tables (`specs/ptychodus_api_spec.md §5`) and translate it into parameterized tests that set and round-trip values such as `probe_scale`, `gaussian_smoothing_sigma`, `sequential_sampling`, and other newly documented knobs.
- Acceptance: automated parity tests confirm each documented field maps identically into `ptycho.params.cfg`, and comparative snapshots show no differences between TensorFlow and PyTorch updates for a matrix of representative configurations.

#### Phase 2 – Data Ingestion & Grouping (2 sprints)

- **Task 2.1**: Implement a `RawDataTorch` shim that consumes the NPZ schema produced by `export_training_data()` (`specs/ptychodus_api_spec.md §4.5`) and exposes methods mirroring `RawData.generate_grouped_data` semantics.
- **Task 2.2**: Map the Torch memory-mapped dataset outputs (`ptycho_torch/dset_loader_pt_mmap.py:1-260`) onto the dictionary keys enumerated in the contract (`specs/ptychodus_api_spec.md §4.3`).
- **Task 2.3**: Provide a `PtychoDataContainerTorch` that matches TensorFlow tensor shapes and dtype expectations, ensuring compatibility with downstream reassembly helpers.
- Acceptance: integration tests load identical NPZ inputs through both backends and compare grouped dictionary keys, shapes, and coordinate values (float tolerances defined up front).

#### Phase 3 – Inference Entry Point (1 sprint)

- **Task 3.1**: Wrap the PyTorch model so `predict([diffraction * intensity_scale, local_offsets])` is supported (per `specs/ptychodus_api_spec.md §4.4`).
- **Task 3.2**: Bridge the PyTorch intensity scaler to `params.cfg['intensity_scale']` and `params.cfg['intensity_scale.trainable']` using the logic in `ptycho_torch/model.py:480-565`.
- **Task 3.3**: Resolve probe initialisation: either reuse `ptycho.probe.set_probe_guess` via thin adapters or port equivalent functionality into `ptycho_torch` so the grouped data pipeline receives the expected probe tensor (`specs/ptychodus_api_spec.md §4.3`).
- **Task 3.4**: Adapt `ptycho.tf_helper.reassemble_position` usage by providing PyTorch equivalents or data conversion utilities before invoking the TensorFlow helper (see `specs/ptychodus_api_spec.md §4.4`).
- Acceptance: reconstructor `reconstruct()` executes end-to-end with the PyTorch backend on sample data, successfully initialises probes, and produces an object array of the expected shape and dtype.

#### Phase 4 – Training Workflow Parity (2 sprints)

- **Task 4.1**: Expose orchestrators analogous to `run_cdi_example`, `train_cdi_model`, and `save_outputs` that operate on PyTorch models (`specs/ptychodus_api_spec.md §4.5`).
- **Task 4.2**: Update `ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py:229-269` to dispatch to the PyTorch workflow when selected, preserving logging and output directory semantics.
- **Task 4.3**: Ensure NPZ exports from `export_training_data()` remain consumable by the PyTorch training path without schema divergence.
- **Task 4.4**: Integrate the probe strategy selected in Phase 3 into the training pipeline so probe guesses are initialised consistently before batching/grouping.
- Acceptance: training end-to-end test executes on a reduced dataset, writes artifacts to `output_dir`, restores the trained model for immediate inference, and probe initialisation runs without backend-specific fallbacks.

#### Phase 5 – Model Persistence & Archives (1 sprint)

- **Task 5.1**: Extend `ptycho.model_manager` with PyTorch-aware branches (or introduce a companion module) that produce bundle metadata alongside TensorFlow artefacts while sharing the `wts.h5.zip` packaging contract (`specs/ptychodus_api_spec.md §4.6`).
- **Task 5.2**: Define the PyTorch payload layout (e.g., `diffraction_to_obj.pt`, optional optimizer state, serialized custom layers) and how it coexists with Keras assets inside the archive; document format versioning.
- **Task 5.3**: Update `load_inference_bundle` to inspect archive contents, dispatch to the appropriate loader (TensorFlow vs PyTorch), and restore `params.cfg` side effects for both paths.
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

### 5. Dependencies & Risks

- **TensorFlow helper reuse**: section `4.4` of the spec assumes TensorFlow-specific helpers. Replacing them may require heavy refactoring or careful interop wrappers.
- **Archive compatibility**: deviating from `wts.h5.zip` could break existing saved models; introduce versioning and migration guidance if change is unavoidable.
- **Performance parity**: PyTorch data pipeline must be profiled against TensorFlow to ensure similar throughput, particularly for grouping operations.

### 6. Communication & Handoff

- Weekly sync to review progress against the spec sections cited above.
- Maintain a living checklist mapping completed tasks to spec requirements to guarantee no contracts are overlooked.
- Final deliverable includes updated documentation (README, user guides) explaining backend selection and any new configuration knobs.
