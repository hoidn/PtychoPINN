# INTEGRATE-PYTORCH-001: Glossary & Ownership Map

**Initiative:** INTEGRATE-PYTORCH-001
**Document Purpose:** Define project-specific terminology and map component ownership across implementation phases
**Created:** 2025-10-16
**Source Documents:**
- `/home/ollie/Documents/PtychoPINN2/plans/active/INTEGRATE-PYTORCH-001/implementation.md`
- `/home/ollie/Documents/PtychoPINN2/docs/architecture.md`
- `/home/ollie/Documents/PtychoPINN2/docs/DEVELOPER_GUIDE.md`
- `/home/ollie/Documents/PtychoPINN2/specs/ptychodus_api_spec.md`
- `/home/ollie/Documents/PtychoPINN2/plans/ptychodus_pytorch_integration_plan.md`

---

## Part 1: Initiative Glossary

This section defines project-specific terminology used throughout the INTEGRATE-PYTORCH-001 initiative. Cross-references to source documents are provided for deeper context.

### Core Architecture Terms

#### Group-then-Sample
**Definition:** A two-stage data sampling strategy for overlap-based training (`gridsize > 1`) where the system first discovers all valid neighbor groups across the entire dataset, caches them, and then randomly samples from this pre-computed set.

**Contrast:** "Sample-then-group" would first randomly sample anchor points and then find neighbors, which is computationally cheaper but less representative.

**Implementation Details:**
- **Stage 1:** Scan entire dataset's coordinates to find all physically adjacent scan point groups
- **Stage 2:** Cache discovered groups to file (e.g., `dataset.g2k4.groups_cache.npz`)
- **Stage 3:** Randomly sample requested number of groups from cached set

**Source References:**
- Implementation: `ptycho/raw_data.py:365-438` (`RawData.generate_grouped_data`)
- Architecture diagram: `docs/architecture.md:83-104` (Data Transformation Flow)
- Developer guidance: `docs/DEVELOPER_GUIDE.md:109-119`

**PyTorch Equivalent:** Must be implemented in `RawDataTorch` wrapper to maintain grouping semantics.

**Related Terms:** neighbor-aware grouping, overlap-based training, gridsize

---

#### Legacy Params Bridge
**Definition:** The one-way configuration synchronization mechanism that populates the deprecated global dictionary (`ptycho.params.cfg`) from modern dataclass instances (`ModelConfig`, `TrainingConfig`, `InferenceConfig`).

**Purpose:** Maintains backward compatibility with 20+ legacy modules that depend on global state while allowing new code to use structured, type-safe configuration.

**Key Components:**
- **Primary Function:** `ptycho.config.config.update_legacy_dict(cfg: dict, dataclass_obj: Any)`
- **Translation Logic:** `ptycho.config.config.dataclass_to_legacy_dict(obj: Any)`
- **Mapping Table:** `KEY_MAPPINGS` dictionary (e.g., `object_big` → `object.big`)

**Data Flow:**
```
External Caller (ptychodus)
  → Instantiate TrainingConfig/InferenceConfig
    → Call update_legacy_dict(params.cfg, config)
      → Apply KEY_MAPPINGS translations
        → Update ptycho.params.cfg
          → Legacy modules access via params.get('key')
```

**Source References:**
- API specification: `specs/ptychodus_api_spec.md:57-125` (The Compatibility Bridge)
- Flow diagram: `specs/ptychodus_api_spec.md:82-94` (mermaid diagram)
- Anti-patterns: `docs/DEVELOPER_GUIDE.md:61-73` (implicit dependencies)

**Critical Rule:** External systems (like ptychodus) MUST call `update_legacy_dict()` before invoking any ptychopinn functions that depend on `params.cfg`.

**PyTorch Requirement:** PyTorch backend must continue to populate `params.cfg` via the same bridge to ensure legacy consumers observe consistent values.

**Related Terms:** global state, configuration handshake, one-way data flow

---

#### Reconstructor Contract
**Definition:** The behavioral API that `ptychodus` expects from any PtychoPINN backend implementation, encompassing lifecycle, configuration, data ingestion, inference, training, and persistence semantics.

**Key Contract Elements:**

1. **Entry Points & Lifecycle** (`specs/ptychodus_api_spec.md:127-141`)
   - Instantiation via `PtychoPINNReconstructorLibrary`
   - Dual instances (PINN, Supervised) sharing settings registry
   - Runtime updates to model/training/inference knobs

2. **Configuration Handshake** (`specs/ptychodus_api_spec.md:142-151`)
   - Assemble dataclass configs from live settings
   - Call `update_legacy_dict(ptycho.params.cfg, config)`
   - Respect all dataclass fields (they feed downstream modules)

3. **Data Ingestion** (`specs/ptychodus_api_spec.md:153-167`)
   - `create_raw_data()` converts `ReconstructInput` to `RawData`
   - `generate_grouped_data()` expects pre-populated `params.cfg`
   - Returns spec-compliant dictionary with `diffraction`, `coords_offsets`, etc.

4. **Inference Behavior** (`specs/ptychodus_api_spec.md:169-179`)
   - Signature: `model.predict([diffraction * intensity_scale, local_offsets])`
   - Exposes `intensity_scale` parameter in `params.cfg`
   - Output stitched via `ptycho.tf_helper.reassemble_position`

5. **Training Workflow** (`specs/ptychodus_api_spec.md:181-190`)
   - Export training data: NPZ with keys `xcoords`, `ycoords`, `diff3d`, `probeGuess`, `objectGuess`, `scan_index`
   - Execute via `run_cdi_example` or equivalent
   - Return values compatible with `save_outputs()` and `Product` reconstruction

6. **Persistence Contract** (`specs/ptychodus_api_spec.md:192-211`)
   - Archive format: `wts.h5.zip` in specified directory
   - Save: `ptycho.model_manager.save`
   - Load: `load_inference_bundle` with `params.cfg` restoration

**Source References:**
- Full specification: `specs/ptychodus_api_spec.md:127-211` (§4 Reconstructor Contract)
- Implementation: `ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py:61-273`
- Workflow orchestration: `ptycho/workflows/components.py`

**PyTorch Challenge:** Replacement backend must honor all six elements while working with PyTorch tensors instead of TensorFlow.

**Related Terms:** reconstructor lifecycle, backend selection, capability flags

---

#### Persistence Layer
**Definition:** The subsystem responsible for serializing trained models, configuration state, and metadata into portable archives, and restoring them for inference.

**TensorFlow Implementation:**
- **Module:** `ptycho/model_manager.py`
- **Archive Format:** `wts.h5.zip` containing Keras HDF5 weights + serialized `params.cfg`
- **Save Function:** `ModelManager.save()`
- **Load Function:** `load_inference_bundle()` → `ModelManager.load_multiple_models()`
- **Side Effect:** Loading restores `params.cfg` global state

**Contract Requirements:**
- Models saved via `save_model()` must round-trip through `open_model()`
- Archive must preserve all configuration needed for inference
- Custom layers must be registered (e.g., `CombineComplexLayer`, `ReassemblePatchesLayer`)
- TensorFlow models require `tf.keras.config.enable_unsafe_deserialization()`

**Source References:**
- API spec: `specs/ptychodus_api_spec.md:192-211` (§4.6 Model Persistence Contract)
- Implementation: `ptycho/model_manager.py:1-360`
- Reconstructor integration: `ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py:183-195`

**PyTorch Design Decisions (Phase E):**
- Option A: Extend `wts.h5.zip` to contain both TensorFlow and PyTorch payloads
- Option B: New archive format (e.g., `wts.pt.zip`) with migration tooling
- Option C: Adapter layer that translates between formats

**Related Terms:** model serialization, archive format, custom layers, params restoration

---

#### Workflow Orchestration
**Definition:** The high-level coordination layer that chains together configuration, data loading, model training/inference, and output generation into end-to-end pipelines.

**TensorFlow Implementation:**
- **Primary Module:** `ptycho/workflows/components.py`
- **Key Functions:**
  - `run_cdi_example()` - Main training orchestrator
  - `train_cdi_model()` - Model training execution
  - `save_outputs()` - Export reconstructions
  - `load_inference_bundle()` - Load saved models
  - `setup_configuration()` - Config assembly

**Orchestration Sequence (Training):**
```
setup_configuration(args, yaml_path)
  → update_legacy_dict(params.cfg, config)
    → create_ptycho_data_container(train_data)
      → model.train(train_container)
        → save_outputs(model, test_data)
```

**Contract Integration:**
- Orchestrators are referenced in `specs/ptychodus_api_spec.md:181-190` (§4.5)
- Ptychodus delegates to `run_cdi_example` for training
- Must respect `output_dir`, logging, and artifact layout

**Source References:**
- Implementation: `ptycho/workflows/components.py:676-732`
- Sequence diagram: `docs/architecture.md:44-74`
- Developer guide: `docs/DEVELOPER_GUIDE.md:150-153`

**PyTorch Gaps (Phase D):**
- No equivalent to `run_cdi_example` - must be implemented
- No `ModelManager` equivalent for persistence
- Existing `ptycho_torch/train.py` uses Lightning but bypasses dataclass configs

**Related Terms:** run_cdi_example, training pipeline, inference pipeline, end-to-end workflow

---

#### Data Container
**Definition:** A structured object that packages model-ready tensors (diffraction patterns, ground truth patches, position offsets, etc.) with consistent shapes and dtypes for direct consumption by neural network models.

**TensorFlow Implementation:**
- **Class:** `PtychoDataContainer` (`ptycho/loader.py:93-200`)
- **Key Tensors:**
  - `X`: Diffraction patterns (tf.Tensor, shape depends on N and gridsize)
  - `Y_I`: Ground truth amplitude patches (complex64)
  - `Y_phi`: Ground truth phase patches
  - `local_offsets`: Within-group position offsets
  - `global_offsets`: Absolute scan coordinates
  - `coords_nominal`: Nominal grid coordinates

**Shape Dependencies:**
- `N`: Diffraction pattern crop size (e.g., 64, 128)
- `gridsize`: Number of neighbor groups (e.g., gridsize=2 → 4 channels)
- Channel format: `(B, N, N, gridsize²)` for overlapping patches
- Flat format: `(B*gridsize², N, N, 1)` for physics simulation

**Data Flow:**
```
RawData.generate_grouped_data()
  → Returns NumPy dictionary
    → loader.create_ptycho_data_container()
      → Converts to TensorFlow tensors
        → Returns PtychoDataContainer
          → Passed to model.train() / model.predict()
```

**Source References:**
- Implementation: `ptycho/loader.py:93-200`
- Data pipeline: `docs/architecture.md:76-119`
- Tensor formats: `docs/DEVELOPER_GUIDE.md:120-130` (gridsize > 1 formats)

**PyTorch Requirement (Phase C):**
- Must implement `PtychoDataContainerTorch` with equivalent fields
- Shapes/dtypes must match TensorFlow for reconstructor compatibility
- May use PyTorch tensors internally but interface must align

**Related Terms:** tensor packaging, model input, channel format, flat format

---

### Process & Methodology Terms

#### Parity Baseline
**Definition:** A comprehensive, current mapping of responsibilities and implementation status across TensorFlow and PyTorch backends, used as the reference point for gap analysis and integration planning.

**Components:**
- **Component Inventory:** List of all TensorFlow modules and their PyTorch counterparts
- **Gap Analysis:** Missing functionality, incomplete implementations, API mismatches
- **Dependency Map:** Cross-module dependencies and initialization order
- **Test Coverage:** Existing tests for each component

**Source References:**
- Parity map template: `plans/ptychodus_pytorch_integration_plan.md:37-49` (Phase 0 Artifact)
- Phase A deliverable: `plans/active/INTEGRATE-PYTORCH-001/implementation.md:11-20`

**Phase A Deliverable:** `plans/active/INTEGRATE-PYTORCH-001/reports/<timestamp>/parity_map.md`

**Related Terms:** gap analysis, component inventory, architectural assessment

---

#### Test-Driven Development (TDD)
**Definition:** The project's required development methodology following a "Red-Green-Refactor" cycle where tests are written before implementation code.

**Three-Step Cycle:**
1. **RED:** Write failing test that captures desired functionality or reproduces bug
2. **GREEN:** Write minimal code to make test pass
3. **REFACTOR:** Clean up implementation while keeping tests green

**Application to Initiative:**
- Phase B (Config): Write failing tests for legacy bridge gaps → Implement adapter → Validate parity
- Phase C (Data): Write failing loader tests → Implement `RawDataTorch` → Verify shapes/dtypes
- Phase D (Workflow): Write failing orchestration tests → Implement entry points → Confirm integration

**Source References:**
- Methodology: `docs/DEVELOPER_GUIDE.md:599-667` (§11 Development Methodology)
- Case study: `docs/DEVELOPER_GUIDE.md:612-667` (Baseline gridsize bug fix)
- Directive: Project `CLAUDE.md:15-18` (Enforce TDD)

**Related Terms:** red-green-refactor, failing test first, minimal implementation

---

#### Sequential Sampling
**Definition:** A deterministic data sampling mode that selects the first N groups from the dataset in order, as opposed to random sampling. Used for reproducibility and debugging.

**Configuration:**
- **TensorFlow:** `TrainingConfig.sequential_sampling` / `InferenceConfig.sequential_sampling`
- **Legacy Key:** `params.cfg['sequential_sampling']`
- **PyTorch:** Must be implemented in `RawDataTorch.generate_grouped_data()`

**Use Cases:**
- Reproducible experiments without setting random seeds
- Debugging data pipeline issues with consistent samples
- Comparing outputs across backends with identical input order

**Source References:**
- Config field: `specs/ptychodus_api_spec.md:258` (TrainingConfig fields table)
- Data pipeline: `ptycho/raw_data.py:365` (grouping parameters)
- Developer guide: `docs/DEVELOPER_GUIDE.md:522-550` (Unified Sampling Strategy)

**Related Terms:** deterministic sampling, reproducibility, grouping strategy

---

#### Intensity Scale
**Definition:** A multiplicative scaling factor that adjusts simulated diffraction patterns to match realistic experimental photon counts, applied at the physics boundary during training.

**Purpose:** Bridges normalized internal data representation with physics-realistic intensity levels for loss calculation.

**Key Principles:**
- **Calculate but don't apply in data pipeline:** `diffsim.py` computes scale but returns normalized data
- **Apply only at physics boundary:** `model.py` applies scaling when computing physics loss
- **Trainable option:** `intensity_scale.trainable` allows joint optimization of scale factor

**Data Flow:**
```python
# In diffsim.py - Calculate but don't apply
intensity_scale = scale_nphotons(Y_I * probe_amplitude)
X = diffract_obj(Y_I * probe)  # Normalized diffraction
return X, Y_I / intensity_scale, Y_phi, intensity_scale  # Return normalized

# In model.py - Apply scaling only at physics boundary
simulated = self.physics_layer(reconstructed) * intensity_scale
loss = poisson_nll(measured, simulated)
```

**Source References:**
- Normalization architecture: `docs/DEVELOPER_GUIDE.md:132-197` (§3.5 Three Normalization Systems)
- Config field: `specs/ptychodus_api_spec.md:235` (ModelConfig intensity_scale row)
- Anti-patterns: `docs/DEVELOPER_GUIDE.md:175-196` (common mistakes)

**PyTorch Requirement:** Must expose `params.cfg['intensity_scale']` and `params.cfg['intensity_scale.trainable']` per reconstructor contract.

**Related Terms:** physics normalization, nphotons, Poisson loss, trainable scale

---

## Part 2: Component Ownership Map

This section maps each Phase B-E task to specific components, responsibilities, dependencies, and test locations. Phase A (baseline refresh) is excluded as it's purely analytical.

### Conventions

**Responsibility Categories:**
- **CONFIG:** Configuration management and legacy bridge
- **DATA:** Data loading, grouping, and tensor packaging
- **WORKFLOW:** High-level orchestration and pipeline coordination
- **PERSISTENCE:** Model serialization and archive management
- **INTEGRATION:** Ptychodus reconstructor interface and backend selection

**Dependency Notation:**
- `→` Sequential dependency (must complete first)
- `||` Parallel work (no dependency)
- `⇄` Bidirectional interaction (both sides need updates)

---

### Phase B: Configuration & Legacy Bridge Alignment

**Goal:** Replace singleton config usage with dataclass-driven state that updates `ptycho.params.cfg`

**Exit Criteria:** PyTorch paths ingest dataclasses, call `update_legacy_dict`, unit tests confirm params parity

---

#### B1: Design Dataclass Ingestion Plan

| Attribute | Value |
|-----------|-------|
| **Task ID** | B1 |
| **Description** | Draft adapter design mapping current `config_params.py` keys to dataclasses |
| **Component** | `ptycho_torch/config_params.py` (refactor target) |
| **Primary Responsibility** | CONFIG |
| **Affected Modules** | `ptycho_torch/config_params.py`, `ptycho/config/config.py` (reuse), new adapter module |
| **Dependencies** | None (Phase A parity map informs design) |
| **Source References** | - Config spec: `specs/ptychodus_api_spec.md:14-125`<br>- Current singletons: `ptycho_torch/config_params.py:39-113`<br>- Target dataclasses: `ptycho/config/config.py:72-140` |
| **Deliverable** | `reports/<timestamp>/config_bridge.md` design document |
| **Design Questions** | - Replace singletons entirely or wrap them?<br>- Factory pattern for dataclass instantiation?<br>- Mapping table for singleton keys → dataclass fields? |
| **Test Location** | `tests/torch/test_config_bridge.py` (to be created in B2) |

**Implementation Guidance:**
1. Audit `model_config_default`, `training_config_default`, `data_config_default` dictionaries
2. Map each key to corresponding `ModelConfig`, `TrainingConfig`, `InferenceConfig` field
3. Design factory functions: `create_model_config_from_defaults()`, etc.
4. Document migration path for existing `ptycho_torch/train.py` usage

**Cross-Phase Links:**
- Informs **C1** (data contract expectations use same config fields)
- Enables **D2** (orchestration needs config ingestion)

---

#### B2: Author Minimal Failing Test Capturing Legacy Bridge Gap

| Attribute | Value |
|-----------|-------|
| **Task ID** | B2 |
| **Description** | Follow TDD: add pytest that asserts legacy dict fields match dataclass inputs |
| **Component** | New test suite for PyTorch config parity |
| **Primary Responsibility** | CONFIG |
| **Affected Modules** | `tests/torch/test_config_bridge.py` (new) |
| **Dependencies** | B1 design → informs test assertions |
| **Source References** | - TDD methodology: `docs/DEVELOPER_GUIDE.md:599-611`<br>- Config fields: `specs/ptychodus_api_spec.md:213-291`<br>- Params pattern: `docs/debugging/QUICK_REFERENCE_PARAMS.md` |
| **Test Assertions** | - `params.cfg['n_groups']` matches `TrainingConfig.n_groups`<br>- `params.cfg['probe.mask']` matches `ModelConfig.probe_mask`<br>- All fields in §5 tables covered |
| **Test Location** | `tests/torch/test_config_bridge.py` |
| **Test Commands** | `pytest tests/torch/test_config_bridge.py -v` |

**Test Structure:**
```python
def test_training_config_populates_legacy_dict():
    """Verify TrainingConfig fields propagate to params.cfg via bridge."""
    from ptycho.config.config import TrainingConfig, update_legacy_dict
    import ptycho.params as p

    config = TrainingConfig(
        n_groups=512,
        batch_size=16,
        sequential_sampling=True,
        # ... all required fields
    )

    update_legacy_dict(p.cfg, config)

    assert p.cfg['n_groups'] == 512
    assert p.cfg['batch_size'] == 16
    assert p.cfg['sequential_sampling'] == True
    # ... verify all mappings
```

**Expected Outcome (RED):** Test fails because PyTorch code doesn't yet call `update_legacy_dict()`

---

#### B3: Implement Adapter + Transition Script

| Attribute | Value |
|-----------|-------|
| **Task ID** | B3 |
| **Description** | Introduce `load_training_config_from_dataclasses()`, ensure call order respects params initialization |
| **Component** | New PyTorch config adapter module |
| **Primary Responsibility** | CONFIG |
| **Affected Modules** | - `ptycho_torch/config_adapter.py` (new)<br>- `ptycho_torch/train.py` (refactor entry point)<br>- `ptycho_torch/model.py` (may need config access updates) |
| **Dependencies** | B1 design → B2 failing test → **B3 implementation** (GREEN step) |
| **Source References** | - Bridge function: `specs/ptychodus_api_spec.md:57-125`<br>- Init order: `docs/DEVELOPER_GUIDE.md:64-73`<br>- Anti-patterns: `docs/DEVELOPER_GUIDE.md:61-73` |
| **Test Location** | `tests/torch/test_config_bridge.py` (should pass after B3) |

**Implementation Steps:**
1. Create `ptycho_torch/config_adapter.py`:
   ```python
   def load_training_config_from_dataclasses(defaults: dict) -> TrainingConfig:
       """Factory to create TrainingConfig from PyTorch default dicts."""
       return TrainingConfig(
           n_groups=defaults.get('n_groups'),
           batch_size=defaults.get('batch_size'),
           # ... map all fields
       )
   ```

2. Update `ptycho_torch/train.py`:
   ```python
   from ptycho.config.config import update_legacy_dict
   from ptycho_torch.config_adapter import load_training_config_from_dataclasses
   import ptycho.params as p

   def main():
       # 1. Create dataclass config
       config = load_training_config_from_dataclasses(training_config_default)

       # 2. Bridge to legacy system (MANDATORY)
       update_legacy_dict(p.cfg, config)

       # 3. NOW safe to import modules depending on global state
       from ptycho_torch.model import PtychoPINN
       # ... continue training
   ```

3. Remove direct singleton usage where possible, prefer dataclass fields

**Expected Outcome (GREEN):** B2 tests pass, params.cfg correctly populated

---

#### B4: Validate Against TensorFlow Baseline

| Attribute | Value |
|-----------|-------|
| **Task ID** | B4 |
| **Description** | Run parity check comparing `params.cfg` dumps from both backends |
| **Component** | Comparative validation test |
| **Primary Responsibility** | CONFIG |
| **Affected Modules** | `tests/torch/test_config_parity.py` (new parity test) |
| **Dependencies** | B3 complete (PyTorch bridge functional) || TensorFlow baseline (already working) |
| **Source References** | - Config fields reference: `specs/ptychodus_api_spec.md:213-291`<br>- Parity validation: `plans/active/INTEGRATE-PYTORCH-001/implementation.md:34` |
| **Test Location** | `tests/torch/test_config_parity.py` |
| **Deliverable** | `reports/<timestamp>/cfg_diff.txt` showing no differences |

**Parity Test Pattern:**
```python
def test_tensorflow_pytorch_config_parity():
    """Ensure both backends produce identical params.cfg for same inputs."""
    import ptycho.params as p
    from ptycho.config.config import TrainingConfig, update_legacy_dict

    # Test config
    config = TrainingConfig(
        n_groups=512, batch_size=16, gridsize=2, N=64,
        # ... all fields
    )

    # TensorFlow path
    p.cfg.clear()
    update_legacy_dict(p.cfg, config)
    tf_snapshot = dict(p.cfg)

    # PyTorch path (using new adapter)
    p.cfg.clear()
    from ptycho_torch.config_adapter import apply_config_to_legacy
    apply_config_to_legacy(config)
    pt_snapshot = dict(p.cfg)

    # Compare
    assert tf_snapshot == pt_snapshot, f"Diff: {set(tf_snapshot) ^ set(pt_snapshot)}"
```

**Validation Criteria:**
- All 50+ config fields from §5 tables match exactly
- No unexpected keys in either snapshot
- Types match (str vs Path, int vs float, etc.)

---

### Phase C: Data Pipeline & Tensor Packaging Parity

**Goal:** Provide PyTorch-ready `RawDataTorch` and `PtychoDataContainerTorch` satisfying grouping and tensor layout contracts

**Exit Criteria:** PyTorch data pipeline outputs match spec shapes/dtypes and pass targeted loader tests

---

#### C1: Specify Data Contract Expectations

| Attribute | Value |
|-----------|-------|
| **Task ID** | C1 |
| **Description** | Summarize required keys/shapes from specs in data contract document |
| **Component** | Data contract specification for PyTorch |
| **Primary Responsibility** | DATA |
| **Affected Modules** | Documentation artifact (no code changes) |
| **Dependencies** | Phase A parity map || B1 config design (config influences shapes) |
| **Source References** | - NPZ schema: `specs/data_contracts.md:1-74`<br>- Reconstructor data: `specs/ptychodus_api_spec.md:153-167`<br>- Grouping output: `ptycho/raw_data.py:365-438` |
| **Deliverable** | `reports/<timestamp>/data_contract.md` |
| **Test Location** | N/A (documentation task) |

**Contract Document Contents:**

1. **Input NPZ Schema** (from `specs/data_contracts.md`)
   - Required keys: `diffraction`, `Y`, `objectGuess`, `probeGuess`, `xcoords`, `ycoords`, `scan_index`
   - Shapes, dtypes, normalization requirements

2. **Grouped Data Dictionary Keys** (from `RawData.generate_grouped_data`)
   - `diffraction`: `(n_groups, gridsize², N, N)` float32
   - `coords_offsets`: Position data
   - `coords_relative`: Relative coordinates
   - `local_offsets`: Within-group offsets
   - (Full list from `ptycho/loader.py:1-186`)

3. **PtychoDataContainer Tensor Specs** (target for PyTorch)
   - `X`: Diffraction patterns (shape depends on N, gridsize)
   - `Y_I`, `Y_phi`: Complex ground truth
   - `local_offsets`, `global_offsets`: Position tensors
   - dtype requirements (complex64, float32, etc.)

4. **Shape Dependencies**
   - Channel format: `(B, N, N, gridsize²)`
   - Flat format: `(B*gridsize², N, N, 1)`
   - (Reference: `docs/DEVELOPER_GUIDE.md:120-130`)

**Cross-Phase Links:**
- Informs **C2** (test assertions)
- Guides **C3/C4** (implementation targets)

---

#### C2: Draft Failing Loader Test

| Attribute | Value |
|-----------|-------|
| **Task ID** | C2 |
| **Description** | Add pytest ensuring amplitude/complex dtype handling and neighbor grouping reuse cached outputs |
| **Component** | PyTorch data container test suite |
| **Primary Responsibility** | DATA |
| **Affected Modules** | `tests/torch/test_data_container.py` (new) |
| **Dependencies** | C1 contract spec → **C2 failing test** (RED step) |
| **Source References** | - TDD pattern: `docs/DEVELOPER_GUIDE.md:599-611`<br>- Data contract: C1 deliverable<br>- TF reference: `ptycho/loader.py:93-200` |
| **Test Location** | `tests/torch/test_data_container.py` |
| **Test Commands** | `pytest tests/torch/test_data_container.py -v` |

**Test Cases:**

1. **Dtype Validation Test**
   ```python
   def test_data_container_preserves_complex_dtype():
       """Ensure Y patches remain complex64, not silently converted to float."""
       raw_data = RawDataTorch.from_file('test_dataset.npz')
       container = create_ptycho_data_container_torch(raw_data)

       assert container.Y_I.dtype == torch.complex64
       assert container.Y_phi.dtype == torch.float32
       # Prevent historical bug: docs/DEVELOPER_GUIDE.md:85-100
   ```

2. **Grouping Cache Test**
   ```python
   def test_neighbor_grouping_uses_cache():
       """Verify group-then-sample reuses cached groups."""
       raw_data = RawDataTorch.from_file('test_dataset.npz')

       # First call should generate cache
       groups1 = raw_data.generate_grouped_data(n_groups=100)
       cache_file = Path('test_dataset.g2k4.groups_cache.npz')
       assert cache_file.exists()

       # Second call should load from cache
       groups2 = raw_data.generate_grouped_data(n_groups=100)
       # Compare to ensure same source (different samples OK)
   ```

3. **Shape Parity Test**
   ```python
   def test_pytorch_tensorflow_shape_parity():
       """Ensure PyTorch containers match TensorFlow shapes."""
       # Load same data through both paths
       tf_container = create_ptycho_data_container('test.npz')  # TensorFlow
       pt_container = create_ptycho_data_container_torch('test.npz')  # PyTorch

       assert pt_container.X.shape == tuple(tf_container.X.shape)
       assert pt_container.local_offsets.shape == tuple(tf_container.local_offsets.shape)
   ```

**Expected Outcome (RED):** All tests fail - `RawDataTorch` doesn't exist yet

---

#### C3: Implement `RawDataTorch` Wrapper

| Attribute | Value |
|-----------|-------|
| **Task ID** | C3 |
| **Description** | Mirror `raw_data.py:120-380` behaviour; ensure caching semantics align |
| **Component** | PyTorch raw data ingestion layer |
| **Primary Responsibility** | DATA |
| **Affected Modules** | - `ptycho_torch/raw_data.py` (new)<br>- `ptycho_torch/dset_loader_pt_mmap.py` (refactor/reuse) |
| **Dependencies** | C1 contract → C2 failing tests → **C3 implementation** (GREEN step) |
| **Source References** | - TF implementation: `ptycho/raw_data.py:120-438`<br>- Data flow: `docs/architecture.md:76-119`<br>- Grouping: `docs/DEVELOPER_GUIDE.md:109-119` |
| **Test Location** | `tests/torch/test_data_container.py::test_neighbor_grouping_uses_cache` |

**Implementation Requirements:**

1. **API Surface** (mirror TensorFlow `RawData`)
   ```python
   class RawDataTorch:
       @classmethod
       def from_file(cls, npz_path: Path) -> 'RawDataTorch':
           """Load NPZ per specs/data_contracts.md schema."""

       def generate_grouped_data(
           self,
           n_groups: int,
           neighbor_count: int = 4,
           sequential_sampling: bool = False
       ) -> dict:
           """Return dict with keys: diffraction, coords_offsets, etc."""
   ```

2. **Group-Then-Sample Implementation**
   - Stage 1: Scan entire dataset for valid neighbor groups
   - Stage 2: Cache to `<basename>.g{gridsize}k{neighbor_count}.groups_cache.npz`
   - Stage 3: Sample `n_groups` from cached set
   - Reference: `ptycho/raw_data.py:365-438`

3. **Dtype Safety**
   - Explicit `dtype=torch.complex64` for Y arrays
   - Prevent silent float64 conversion (historical bug: `docs/DEVELOPER_GUIDE.md:85-100`)

4. **Reuse Existing Code**
   - `ptycho_torch/dset_loader_pt_mmap.py` has memory-mapping logic
   - `ptycho_torch/patch_generator.py` has patch extraction
   - Extract reusable functions, refactor into `RawDataTorch`

**Expected Outcome (GREEN):** C2 grouping and dtype tests pass

---

#### C4: Implement `PtychoDataContainerTorch`

| Attribute | Value |
|-----------|-------|
| **Task ID** | C4 |
| **Description** | Provide API-compatible tensors for training/inference; document shapes vs TF |
| **Component** | PyTorch tensor packaging layer |
| **Primary Responsibility** | DATA |
| **Affected Modules** | - `ptycho_torch/loader.py` (new)<br>- `ptycho_torch/data_container.py` (new class) |
| **Dependencies** | C3 `RawDataTorch` complete → **C4 container** (consumes grouped data dict) |
| **Source References** | - TF implementation: `ptycho/loader.py:93-200`<br>- Tensor formats: `docs/DEVELOPER_GUIDE.md:120-130`<br>- Contract: C1 deliverable |
| **Test Location** | `tests/torch/test_data_container.py::test_pytorch_tensorflow_shape_parity` |

**Container Class Design:**

```python
@dataclass
class PtychoDataContainerTorch:
    """PyTorch equivalent of ptycho.loader.PtychoDataContainer."""

    X: torch.Tensor  # Diffraction patterns
    Y_I: torch.Tensor  # Amplitude ground truth (complex64)
    Y_phi: torch.Tensor  # Phase ground truth (float32)
    local_offsets: torch.Tensor  # Within-group position offsets
    global_offsets: torch.Tensor  # Absolute scan coordinates
    coords_nominal: torch.Tensor  # Nominal grid coordinates

    def __post_init__(self):
        """Validate shapes and dtypes per contract."""
        assert self.Y_I.dtype == torch.complex64
        assert self.X.shape[-1] == self.Y_I.shape[-1]  # Channel consistency
```

**Conversion Function:**

```python
def create_ptycho_data_container_torch(
    raw_data: RawDataTorch,
    n_groups: int,
    **kwargs
) -> PtychoDataContainerTorch:
    """
    Create model-ready tensor container from RawDataTorch.

    Mirrors ptycho.loader.create_ptycho_data_container() API.
    """
    grouped_dict = raw_data.generate_grouped_data(n_groups, **kwargs)

    # Convert NumPy arrays to PyTorch tensors
    X = torch.from_numpy(grouped_dict['diffraction'])
    Y_complex = torch.from_numpy(grouped_dict['Y'])

    return PtychoDataContainerTorch(
        X=X,
        Y_I=Y_complex,  # Keep complex, don't split yet
        # ... populate all fields
    )
```

**Expected Outcome (GREEN):** C2 shape parity test passes, container matches TensorFlow

---

#### C5: Verify Memmap + Cache Lifecycles

| Attribute | Value |
|-----------|-------|
| **Task ID** | C5 |
| **Description** | Capture test run output (expected: `pytest tests/torch/test_data_container.py -k cache`) |
| **Component** | Data pipeline validation |
| **Primary Responsibility** | DATA |
| **Affected Modules** | Test validation report |
| **Dependencies** | C2, C3, C4 complete → **C5 validation** (verify all GREEN) |
| **Source References** | - Memmap: `ptycho_torch/dset_loader_pt_mmap.py` docstring<br>- Cache: `ptycho/raw_data.py:365-438` |
| **Test Location** | `tests/torch/test_data_container.py` |
| **Deliverable** | `reports/<timestamp>/data_pipeline_validation.log` |

**Validation Commands:**

```bash
# Run cache-related tests
pytest tests/torch/test_data_container.py -k cache -v > reports/$(date -Iseconds)/data_cache.log

# Run full data pipeline suite
pytest tests/torch/test_data_container.py -v > reports/$(date -Iseconds)/data_full.log
```

**Validation Criteria:**
- Cache files created on first run
- Cache files reused on subsequent runs (verify via timestamp)
- Memory-mapped files accessible without loading full dataset
- No memory leaks during repeated container creation

**Cross-Phase Links:**
- Enables **D2** (orchestration needs working data pipeline)
- Supports **E2** (integration tests need data fixtures)

---

### Phase D: Workflow Orchestration & Persistence

**Goal:** Expose PyTorch training/inference orchestration compatible with reconstructor, including save/load

**Exit Criteria:** Ptychodus can call into PyTorch backend for train/infer; artifacts saved in spec-compliant archives

---

#### D1: Design PyTorch Equivalents of `run_cdi_example` + `ModelManager`

| Attribute | Value |
|-----------|-------|
| **Task ID** | D1 |
| **Description** | Author design note referencing TF workflows and model_manager |
| **Component** | Workflow orchestration design |
| **Primary Responsibility** | WORKFLOW + PERSISTENCE |
| **Affected Modules** | Design document (no code yet) |
| **Dependencies** | Phase B complete (config works) || Phase C complete (data works) |
| **Source References** | - TF workflow: `ptycho/workflows/components.py`<br>- Model manager: `ptycho/model_manager.py`<br>- Reconstructor calls: `specs/ptychodus_api_spec.md:181-211` |
| **Deliverable** | `reports/<timestamp>/workflow_design.md` |
| **Test Location** | N/A (design task) |

**Design Document Sections:**

1. **Workflow Orchestration Equivalents**
   - `run_cdi_example_torch()` - Main training orchestrator
   - `train_cdi_model_torch()` - Model training loop
   - `save_outputs_torch()` - Export reconstructions
   - API compatibility matrix vs TensorFlow functions

2. **ModelManager Replacement**
   - PyTorch model save: Lightning checkpoints vs Keras HDF5
   - Archive format: extend `wts.h5.zip` or new `wts.pt.zip`?
   - Params bundling: how to serialize `params.cfg` alongside model
   - Custom layer handling: PyTorch nn.Module equivalents

3. **Reconstructor Integration Points**
   - Where ptychodus calls `run_cdi_example` (`reconstructor.py:229-269`)
   - Where ptychodus calls `load_inference_bundle` (`reconstructor.py:183-195`)
   - Backend selection mechanism (config flag, runtime switch, etc.)

4. **Persistence Design Options**
   ```
   Option A: Hybrid Archive (wts.h5.zip)
     + model_tf.h5 (TensorFlow weights)
     + model_pt.pt (PyTorch state_dict)
     + params.json (serialized params.cfg)
     + metadata.yaml (backend flag)

   Option B: Separate Archives
     + wts.h5.zip (TensorFlow only)
     + wts.pt.zip (PyTorch only, same schema)
     + Migration tooling for existing archives

   Option C: Adapter Layer
     + Translate PyTorch state_dict ↔ Keras HDF5
     + Single archive format, transparent conversion
   ```

**Design Review Questions:**
- Does PyTorch workflow need Lightning Trainer or can use plain PyTorch?
- How to handle MLflow autologging vs TensorFlow callbacks?
- Can we reuse TensorFlow `reassemble_position` or need PyTorch version?

**Cross-Phase Links:**
- Informs **D2** (orchestration implementation)
- Informs **D3** (persistence implementation)
- Critical for **E1** (reconstructor selection logic)

---

#### D2: Implement Orchestration Entry Points

| Attribute | Value |
|-----------|-------|
| **Task ID** | D2 |
| **Description** | Ensure new functions accept dataclass configs, avoid global state, align with CLI semantics |
| **Component** | PyTorch workflow orchestration layer |
| **Primary Responsibility** | WORKFLOW |
| **Affected Modules** | - `ptycho_torch/workflows/orchestration.py` (new)<br>- `ptycho_torch/train.py` (refactor to use orchestrators)<br>- `ptycho_torch/inference.py` (new or refactor) |
| **Dependencies** | D1 design → Phase B config → Phase C data → **D2 orchestration** |
| **Source References** | - TF reference: `ptycho/workflows/components.py:676-732`<br>- CLI semantics: `docs/workflows/pytorch.md`<br>- Anti-pattern: `docs/DEVELOPER_GUIDE.md:33-59` (avoid global state) |
| **Test Location** | `tests/torch/test_orchestration.py` (new) |

**Implementation Targets:**

1. **Training Orchestrator**
   ```python
   def run_cdi_example_torch(
       train_data_file: Path,
       test_data_file: Path,
       config: TrainingConfig,
       output_dir: Path
   ) -> dict:
       """
       PyTorch equivalent of ptycho.workflows.components.run_cdi_example.

       Accepts dataclass config instead of reading global params.cfg.
       Returns dict with training history, model path, metrics.
       """
       # 1. Update legacy dict (for downstream compatibility)
       update_legacy_dict(ptycho.params.cfg, config)

       # 2. Create data containers
       train_container = create_ptycho_data_container_torch(
           RawDataTorch.from_file(train_data_file),
           n_groups=config.n_groups
       )

       # 3. Train model
       model = create_model_torch(config.model)
       history = train_cdi_model_torch(model, train_container, config)

       # 4. Save outputs
       save_outputs_torch(model, output_dir, config)

       return {'history': history, 'model_path': output_dir / 'model.pt'}
   ```

2. **Inference Orchestrator**
   ```python
   def run_inference_torch(
       model_path: Path,
       test_data_file: Path,
       config: InferenceConfig,
       output_dir: Path
   ) -> np.ndarray:
       """PyTorch inference workflow matching reconstructor contract."""
       # Load model and restore params.cfg
       model, restored_config = load_inference_bundle_torch(model_path)

       # Create data container
       test_container = create_ptycho_data_container_torch(...)

       # Run inference
       predictions = model.predict(
           test_container.X * config.model.intensity_scale,
           test_container.local_offsets
       )

       # Reassemble patches
       reconstructed = reassemble_position_torch(predictions, ...)

       return reconstructed
   ```

3. **CLI Alignment**
   - Refactor `ptycho_torch/train.py` to call `run_cdi_example_torch()`
   - Ensure command-line args map to `TrainingConfig` dataclass
   - Preserve MLflow autologging semantics

**Test Coverage:**
```python
def test_run_cdi_example_torch_end_to_end():
    """Verify PyTorch orchestrator executes without errors."""
    config = TrainingConfig(
        train_data_file=Path('test_data.npz'),
        n_groups=10,
        nepochs=2,
        # ... minimal config
    )

    results = run_cdi_example_torch(
        train_data_file=config.train_data_file,
        test_data_file=None,
        config=config,
        output_dir=Path('test_output')
    )

    assert 'history' in results
    assert Path('test_output/model.pt').exists()
```

**Expected Outcome:** Orchestrators work standalone, ready for reconstructor integration

---

#### D3: Implement Persistence Shim

| Attribute | Value |
|-----------|-------|
| **Task ID** | D3 |
| **Description** | Save Lightning checkpoints + params bundle mirroring TF `.h5.zip` contents |
| **Component** | PyTorch model persistence layer |
| **Primary Responsibility** | PERSISTENCE |
| **Affected Modules** | - `ptycho_torch/model_manager.py` (new)<br>- `ptycho_torch/workflows/orchestration.py` (calls save/load) |
| **Dependencies** | D1 design (persistence format) → D2 orchestration → **D3 persistence** |
| **Source References** | - TF manager: `ptycho/model_manager.py:1-360`<br>- Archive contract: `specs/ptychodus_api_spec.md:192-211`<br>- Reconstructor calls: `reconstructor.py:183-195` |
| **Test Location** | `tests/torch/test_model_persistence.py` (new) |

**Implementation Based on D1 Design Decision:**

**Assuming Option A (Hybrid Archive):**

1. **Save Function**
   ```python
   def save_model_torch(
       model: nn.Module,
       output_dir: Path,
       config: TrainingConfig,
       params_snapshot: dict
   ):
       """
       Save PyTorch model in reconstructor-compatible archive.

       Mirrors ptycho.model_manager.save() behavior.
       """
       output_dir.mkdir(parents=True, exist_ok=True)
       archive_path = output_dir / 'wts.h5.zip'

       with zipfile.ZipFile(archive_path, 'w') as zf:
           # PyTorch state dict
           state_buffer = io.BytesIO()
           torch.save(model.state_dict(), state_buffer)
           zf.writestr('model_pt.pt', state_buffer.getvalue())

           # Params snapshot (for legacy compatibility)
           params_json = json.dumps(params_snapshot)
           zf.writestr('params.json', params_json)

           # Metadata
           metadata = {
               'backend': 'pytorch',
               'created': datetime.now().isoformat(),
               'config': asdict(config)
           }
           zf.writestr('metadata.yaml', yaml.dump(metadata))
   ```

2. **Load Function**
   ```python
   def load_inference_bundle_torch(
       model_path: Path
   ) -> tuple[nn.Module, InferenceConfig]:
       """
       Load PyTorch model and restore params.cfg.

       Mirrors ptycho.workflows.components.load_inference_bundle().
       """
       archive_path = model_path / 'wts.h5.zip'

       with zipfile.ZipFile(archive_path, 'r') as zf:
           # Check backend
           metadata = yaml.safe_load(zf.read('metadata.yaml'))
           assert metadata['backend'] == 'pytorch'

           # Load state dict
           state_buffer = io.BytesIO(zf.read('model_pt.pt'))
           state_dict = torch.load(state_buffer)

           # Restore params.cfg
           params_json = zf.read('params.json')
           params_snapshot = json.loads(params_json)
           ptycho.params.cfg.update(params_snapshot)

           # Reconstruct model
           config = InferenceConfig(**metadata['config'])
           model = create_model_torch(config.model)
           model.load_state_dict(state_dict)

           return model, config
   ```

**Test Coverage:**
```python
def test_save_load_round_trip():
    """Verify PyTorch model saves and loads correctly."""
    # Train minimal model
    model = create_model_torch(ModelConfig(N=64, gridsize=1))
    config = TrainingConfig(...)

    # Save
    save_model_torch(model, Path('test_output'), config, ptycho.params.cfg)

    # Load
    loaded_model, loaded_config = load_inference_bundle_torch(Path('test_output'))

    # Verify
    assert loaded_config.model.N == 64
    assert ptycho.params.cfg['N'] == 64  # params.cfg restored

    # Test inference works
    test_input = torch.randn(1, 64, 64, 1)
    output = loaded_model(test_input)
    assert output.shape == test_input.shape
```

**Cross-Phase Links:**
- Enables **E1** (reconstructor `open_model()` dispatch)
- Supports **E4** (parity validation with saved models)

---

#### D4: Add Regression Tests

| Attribute | Value |
|-----------|-------|
| **Task ID** | D4 |
| **Description** | Extend test plan with PyTorch backend path; log test commands + outputs |
| **Component** | Integration test suite |
| **Primary Responsibility** | WORKFLOW |
| **Affected Modules** | - `tests/torch/test_integration_workflow.py` (new)<br>- Update `plans/pytorch_integration_test_plan.md` |
| **Dependencies** | D2 orchestration + D3 persistence → **D4 regression tests** |
| **Source References** | - Test plan: `plans/pytorch_integration_test_plan.md`<br>- TF integration test: `tests/test_integration_workflow.py` |
| **Test Location** | `tests/torch/test_integration_workflow.py` |
| **Deliverable** | `reports/<timestamp>/pytorch_regression_tests.log` |

**Regression Test Cases:**

1. **End-to-End Training Test**
   ```python
   def test_pytorch_end_to_end_training():
       """Full training workflow: data → train → save → load → infer."""
       config = TrainingConfig(
           train_data_file=Path('datasets/fly64_test.npz'),
           n_groups=50,
           nepochs=5,
           output_dir=Path('test_output_training')
       )

       # Train
       results = run_cdi_example_torch(
           train_data_file=config.train_data_file,
           test_data_file=None,
           config=config,
           output_dir=config.output_dir
       )

       # Verify outputs
       assert (config.output_dir / 'wts.h5.zip').exists()
       assert 'history' in results

       # Load and infer
       model, loaded_config = load_inference_bundle_torch(config.output_dir)
       # Run inference test...
   ```

2. **Save/Load Parity Test**
   ```python
   def test_save_load_preserves_predictions():
       """Verify loaded model produces same outputs as original."""
       # (Similar to D3 test but with actual data)
   ```

3. **Multi-Config Test**
   ```python
   @pytest.mark.parametrize('gridsize,N', [(1, 64), (2, 64), (2, 128)])
   def test_pytorch_workflow_various_configs(gridsize, N):
       """Ensure workflow works across config variations."""
       # Test critical config combinations
   ```

**Test Execution:**
```bash
# Run PyTorch integration tests
pytest tests/torch/test_integration_workflow.py -v -s > reports/$(date -Iseconds)/pytorch_regression.log

# Compare with TensorFlow baseline
pytest tests/test_integration_workflow.py -v -s > reports/$(date -Iseconds)/tensorflow_baseline.log
```

**Success Criteria:**
- All PyTorch workflow tests pass
- No regressions in TensorFlow tests
- Execution time within 2x of TensorFlow baseline

---

### Phase E: Ptychodus Integration & Parity Validation

**Goal:** Wire PyTorch backend into ptychodus reconstructor selection and prove parity through automated tests

**Exit Criteria:** Ptychodus integration tests pass on CI; documentation updated

---

#### E1: Update Reconstructor Selection Logic

| Attribute | Value |
|-----------|-------|
| **Task ID** | E1 |
| **Description** | Modify reconstructor.py as guided by spec §4.3; ensure backend choice is config-driven |
| **Component** | Ptychodus reconstructor backend selection |
| **Primary Responsibility** | INTEGRATION |
| **Affected Modules** | - `ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py`<br>- `ptychodus/src/ptychodus/model/ptychopinn/core.py` |
| **Dependencies** | Phases B-D complete (full PyTorch stack functional) → **E1 integration** |
| **Source References** | - Reconstructor: `specs/ptychodus_api_spec.md:127-141`<br>- Library: `ptychodus/src/ptychodus/model/ptychopinn/core.py:22-59`<br>- Config: `specs/ptychodus_api_spec.md:142-151` |
| **Test Location** | `tests/ptychodus/test_backend_selection.py` (new ptychodus-side test) |

**Implementation Steps:**

1. **Add Backend Selection Setting**
   ```python
   # In ptychodus/src/ptychodus/model/ptychopinn/settings.py (or equivalent)

   class BackendSettings:
       def __init__(self):
           self.backend = SettingEntry('tensorflow')  # Default: TensorFlow
           self.available_backends = ['tensorflow', 'pytorch']
   ```

2. **Extend Reconstructor Initialization**
   ```python
   # In ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py

   class PtychoPINNTrainableReconstructor:
       def __init__(self, model_type: str, settings: Settings):
           self._model_type = model_type
           self._settings = settings
           self._backend = settings.backend.get_value()  # NEW: read backend choice

       def reconstruct(self, input_data: ReconstructInput) -> Product:
           if self._backend == 'pytorch':
               return self._reconstruct_pytorch(input_data)
           else:
               return self._reconstruct_tensorflow(input_data)

       def _reconstruct_pytorch(self, input_data: ReconstructInput) -> Product:
           """PyTorch reconstruction path."""
           from ptycho_torch.workflows.orchestration import run_inference_torch

           # Create config from settings
           config = self._create_inference_config()

           # Run PyTorch workflow
           reconstructed = run_inference_torch(
               model_path=self._settings.model_path.get_value(),
               test_data_file=...,
               config=config,
               output_dir=...
           )

           return Product(..., reconstructed)

       def _reconstruct_tensorflow(self, input_data: ReconstructInput) -> Product:
           """Existing TensorFlow path (unchanged)."""
           # ... current implementation
   ```

3. **Similar Updates for `train()`**
   ```python
   def train(self, train_data_dir: Path) -> None:
       if self._backend == 'pytorch':
           self._train_pytorch(train_data_dir)
       else:
           self._train_tensorflow(train_data_dir)
   ```

**UI Integration (if applicable):**
- Add backend selection dropdown in ptychodus UI
- Or: environment variable `PTYCHOPINN_BACKEND=pytorch`
- Or: config file setting

**Test Coverage:**
```python
def test_backend_selection_tensorflow():
    """Verify TensorFlow backend still works (no regression)."""
    reconstructor = create_reconstructor(backend='tensorflow')
    result = reconstructor.reconstruct(test_input)
    assert isinstance(result, Product)

def test_backend_selection_pytorch():
    """Verify PyTorch backend integrates correctly."""
    reconstructor = create_reconstructor(backend='pytorch')
    result = reconstructor.reconstruct(test_input)
    assert isinstance(result, Product)

def test_backend_switch_runtime():
    """Verify switching backends at runtime works."""
    settings.backend.set_value('pytorch')
    # ... test workflow
```

---

#### E2: Extend Parity Test Suite

| Attribute | Value |
|-----------|-------|
| **Task ID** | E2 |
| **Description** | Collaborate with TEST-PYTORCH-001 initiative to reuse fixtures |
| **Component** | Cross-backend parity validation |
| **Primary Responsibility** | INTEGRATION |
| **Affected Modules** | - `tests/torch/test_integration_workflow.py` (extend)<br>- Share fixtures from TEST-PYTORCH-001 |
| **Dependencies** | E1 backend selection → **E2 parity tests** || TEST-PYTORCH-001 fixtures |
| **Source References** | - Test plan: `plans/pytorch_integration_test_plan.md`<br>- Parity validation: Phase A parity map |
| **Test Location** | `tests/torch/test_integration_workflow.py::test_tensorflow_pytorch_parity` |

**Parity Test Strategy:**

1. **Shared Fixtures**
   ```python
   # In tests/conftest.py or tests/fixtures.py

   @pytest.fixture
   def minimal_test_dataset():
       """Minimal NPZ dataset for parity testing."""
       return Path('datasets/parity_test_minimal.npz')

   @pytest.fixture
   def training_config_parity():
       """Standard config for parity comparison."""
       return TrainingConfig(
           n_groups=50,
           nepochs=10,
           batch_size=8,
           gridsize=2,
           N=64
       )
   ```

2. **Comparative Reconstruction Test**
   ```python
   def test_tensorflow_pytorch_reconstruction_parity(
       minimal_test_dataset,
       training_config_parity
   ):
       """Compare TensorFlow and PyTorch reconstructions on same data."""

       # TensorFlow path
       tf_result = run_tensorflow_reconstruction(
           minimal_test_dataset,
           training_config_parity
       )

       # PyTorch path
       pt_result = run_pytorch_reconstruction(
           minimal_test_dataset,
           training_config_parity
       )

       # Compare outputs with tolerance
       assert_allclose(
           np.abs(tf_result),
           np.abs(pt_result),
           rtol=1e-3,
           atol=1e-5
       )
   ```

3. **Configuration Parity Test**
   ```python
   def test_config_translation_parity():
       """Verify both backends receive identical params.cfg."""
       # (Reuse from Phase B4)
   ```

**Fixture Sharing with TEST-PYTORCH-001:**
- Coordinate on minimal dataset format
- Share config factory functions
- Align tolerance thresholds for numeric comparisons

**Expected Test Command:**
```bash
pytest tests/torch/test_integration_workflow.py::test_tensorflow_pytorch_parity -v
```

---

#### E3: Document Backend Selection Workflow

| Attribute | Value |
|-----------|-------|
| **Task ID** | E3 |
| **Description** | Update `docs/workflows/pytorch.md` + Ptychodus docs to describe new flag |
| **Component** | User-facing documentation |
| **Primary Responsibility** | INTEGRATION |
| **Affected Modules** | - `docs/workflows/pytorch.md`<br>- Ptychodus user guide (external repo) |
| **Dependencies** | E1 implementation complete → **E3 documentation** |
| **Source References** | - Current workflow: `docs/workflows/pytorch.md`<br>- TF workflow: `docs/workflows/` (for consistency) |
| **Deliverable** | `reports/<timestamp>/documentation_diffs.md` |
| **Test Location** | N/A (documentation task) |

**Documentation Updates:**

1. **Update `docs/workflows/pytorch.md`**
   ```markdown
   ## Backend Selection

   As of INTEGRATE-PYTORCH-001, ptychodus supports both TensorFlow and PyTorch backends.

   ### Selecting Backend in Ptychodus UI
   1. Open Ptychodus Settings → PtychoPINN
   2. Select "Backend" dropdown: TensorFlow (default) or PyTorch
   3. Backend choice persists across sessions

   ### Selecting Backend via CLI
   ```bash
   export PTYCHOPINN_BACKEND=pytorch
   ptycho_train --train_data datasets/fly64.npz --output_dir run_pytorch
   ```

   ### Backend Differences
   - TensorFlow: Mature, production-tested, default choice
   - PyTorch: Experimental, uses Lightning + MLflow, faster dataloading

   ### Model Compatibility
   - Models trained with TensorFlow can only be loaded with TensorFlow
   - Models trained with PyTorch can only be loaded with PyTorch
   - Archives are marked with backend metadata
   ```

2. **Add Migration Guide**
   ```markdown
   ## Migrating from TensorFlow to PyTorch

   ### Training New Models
   Simply select PyTorch backend before starting training.

   ### Converting Existing Models
   Currently not supported. Train new model with PyTorch backend.
   Future: conversion tooling planned (see plans/model_migration.md).
   ```

3. **Update Cross-References**
   - Link to backend selection from `docs/DEVELOPER_GUIDE.md`
   - Update `docs/architecture.md` to show dual-backend diagram
   - Add backend column to test suite index

**Ptychodus Docs (External):**
- Submit PR to ptychodus repo updating user guide
- Add screenshots of backend selection UI
- Document troubleshooting for backend-specific issues

---

#### E4: Perform Final Comparison Run

| Attribute | Value |
|-----------|-------|
| **Task ID** | E4 |
| **Description** | Execute both TensorFlow and PyTorch integration tests; store metrics comparison |
| **Component** | Final parity validation |
| **Primary Responsibility** | INTEGRATION |
| **Affected Modules** | Comparative test execution and reporting |
| **Dependencies** | E1, E2, E3 complete → **E4 final validation** |
| **Source References** | - Integration tests: `tests/test_integration_workflow.py`, `tests/torch/test_integration_workflow.py`<br>- Parity summary template: Phase A parity map |
| **Deliverable** | `reports/<timestamp>/parity_summary.md` |
| **Test Location** | Multiple test suites |

**Comparison Test Execution:**

1. **Run Full Test Suites**
   ```bash
   # TensorFlow baseline
   pytest tests/test_integration_workflow.py -v --tb=short \
       > reports/$(date -Iseconds)/tensorflow_integration.log

   # PyTorch integration
   pytest tests/torch/test_integration_workflow.py -v --tb=short \
       > reports/$(date -Iseconds)/pytorch_integration.log

   # Parity comparison
   pytest tests/torch/ -k parity -v --tb=short \
       > reports/$(date -Iseconds)/parity_tests.log
   ```

2. **Collect Metrics**
   ```python
   # Script: scripts/collect_parity_metrics.py

   import json
   from pathlib import Path

   def collect_parity_metrics():
       """Collect and compare metrics from both backends."""

       tf_metrics = load_metrics('test_output_tf/metrics.json')
       pt_metrics = load_metrics('test_output_pt/metrics.json')

       comparison = {
           'reconstruction_error': {
               'tensorflow': tf_metrics['psnr'],
               'pytorch': pt_metrics['psnr'],
               'difference': abs(tf_metrics['psnr'] - pt_metrics['psnr'])
           },
           'training_time': {
               'tensorflow': tf_metrics['training_time_s'],
               'pytorch': pt_metrics['training_time_s'],
               'speedup': tf_metrics['training_time_s'] / pt_metrics['training_time_s']
           },
           # ... more metrics
       }

       return comparison
   ```

3. **Generate Parity Summary**
   ```markdown
   # Parity Summary Report

   **Date:** 2025-10-16
   **Initiative:** INTEGRATE-PYTORCH-001
   **Phase:** E4 Final Validation

   ## Test Execution Summary

   | Backend | Tests Run | Passed | Failed | Duration |
   |---------|-----------|--------|--------|----------|
   | TensorFlow | 45 | 45 | 0 | 12m 34s |
   | PyTorch | 45 | 45 | 0 | 10m 18s |
   | Parity | 12 | 12 | 0 | 3m 45s |

   ## Reconstruction Quality Comparison

   | Metric | TensorFlow | PyTorch | Delta | Threshold | Status |
   |--------|-----------|---------|-------|-----------|--------|
   | PSNR | 32.4 dB | 32.3 dB | 0.1 dB | <0.5 dB | ✓ PASS |
   | SSIM | 0.947 | 0.946 | 0.001 | <0.01 | ✓ PASS |
   | MAE | 0.023 | 0.024 | 0.001 | <0.005 | ✓ PASS |

   ## Performance Comparison

   - Training time: PyTorch 18% faster (Lightning optimizations)
   - Data loading: PyTorch 35% faster (memory mapping)
   - Inference time: Similar (within 5%)

   ## Known Differences

   1. Numeric precision: Small differences (<1e-4) due to framework internals
   2. Random initialization: Different RNG implementations (both use fixed seeds)
   3. Optimizer behavior: Adam implementation varies slightly

   ## Conclusion

   ✓ PyTorch backend achieves parity with TensorFlow
   ✓ All reconstructor contract requirements satisfied
   ✓ Integration tests pass on both backends
   ✓ Documentation updated

   **Recommendation:** Proceed with merging INTEGRATE-PYTORCH-001 to main branch.
   ```

**Success Criteria for Phase E:**
- All tests pass on both backends
- Reconstruction quality within tolerance thresholds
- Performance within acceptable range
- Documentation complete and reviewed

---

## Appendix: Quick Reference Tables

### A1: Configuration Field Mapping (Phase B Reference)

| Dataclass Field | Legacy Key | PyTorch Singleton | Primary Consumer |
|----------------|------------|-------------------|------------------|
| `ModelConfig.N` | `N` | `DataConfig.N` | `ptycho_torch/model.py`, `raw_data.py` |
| `ModelConfig.gridsize` | `gridsize` | `DataConfig.grid_size` | `dset_loader_pt_mmap.py` |
| `ModelConfig.probe_mask` | `probe.mask` | `ModelConfig.probe.mask` | `model.py:ProbeIllumination` |
| `TrainingConfig.n_groups` | `n_groups` | `DataConfig.n_subsample` | `patch_generator.py` |
| `TrainingConfig.sequential_sampling` | `sequential_sampling` | *(new field)* | `raw_data.py` grouping |

(See full table: `specs/ptychodus_api_spec.md:213-291`)

---

### A2: Test File Organization

| Phase | Test File | Purpose | Key Tests |
|-------|-----------|---------|-----------|
| B | `tests/torch/test_config_bridge.py` | Config → params.cfg translation | `test_training_config_populates_legacy_dict` |
| B | `tests/torch/test_config_parity.py` | TensorFlow ↔ PyTorch config comparison | `test_tensorflow_pytorch_config_parity` |
| C | `tests/torch/test_data_container.py` | Data loading and grouping | `test_neighbor_grouping_uses_cache` |
| C | `tests/torch/test_raw_data.py` | RawDataTorch implementation | `test_generate_grouped_data_shape` |
| D | `tests/torch/test_orchestration.py` | Workflow orchestrators | `test_run_cdi_example_torch_end_to_end` |
| D | `tests/torch/test_model_persistence.py` | Save/load functionality | `test_save_load_round_trip` |
| D | `tests/torch/test_integration_workflow.py` | Full end-to-end pipeline | `test_pytorch_end_to_end_training` |
| E | `tests/ptychodus/test_backend_selection.py` | Reconstructor integration | `test_backend_switch_runtime` |
| E | `tests/torch/test_integration_workflow.py` | Cross-backend parity | `test_tensorflow_pytorch_parity` |

---

### A3: Module Responsibility Summary

| Module | Phases | Primary Responsibility | Key Deliverables |
|--------|--------|----------------------|------------------|
| `ptycho_torch/config_adapter.py` | B | CONFIG | Dataclass factory functions, singleton bridge |
| `ptycho_torch/raw_data.py` | C | DATA | `RawDataTorch` class, grouping implementation |
| `ptycho_torch/loader.py` | C | DATA | `PtychoDataContainerTorch`, tensor packaging |
| `ptycho_torch/workflows/orchestration.py` | D | WORKFLOW | `run_cdi_example_torch`, `run_inference_torch` |
| `ptycho_torch/model_manager.py` | D | PERSISTENCE | `save_model_torch`, `load_inference_bundle_torch` |
| `ptychodus/.../reconstructor.py` | E | INTEGRATION | Backend selection, dispatch logic |

---

### A4: Cross-Phase Dependencies

```
Phase A (Baseline)
    ↓
Phase B (Config) ← informs → Phase C (Data)
    ↓                              ↓
Phase D (Workflow + Persistence) ←┘
    ↓
Phase E (Integration)
```

**Critical Path:**
1. A complete → B1, C1 can start in parallel
2. B complete AND C complete → D can start
3. D complete → E can start

**Parallel Work Opportunities:**
- B1, C1 (design tasks) can run concurrently
- B2, C2 (test authoring) can run concurrently
- E3 (documentation) can start early and be updated throughout

---

## Document Maintenance

**Owner:** INTEGRATE-PYTORCH-001 Initiative Lead
**Review Cycle:** Update after each phase completion
**Version Control:** Track in `plans/active/INTEGRATE-PYTORCH-001/glossary_and_ownership.md`

**Change Log:**
- 2025-10-16: Initial creation (Phase A)
- (Future): Update after Phase B completion with actual implementation details
- (Future): Add lessons learned and gotchas discovered during implementation

**Related Documents:**
- `plans/active/INTEGRATE-PYTORCH-001/implementation.md` - Source task breakdown
- `specs/ptychodus_api_spec.md` - API contract reference
- `docs/DEVELOPER_GUIDE.md` - Development methodology
- `plans/ptychodus_pytorch_integration_plan.md` - Original integration plan
