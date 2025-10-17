# Phase D1.A: TensorFlow Workflow Callchain Analysis

**Goal**: Document `run_cdi_example` orchestration to identify PyTorch replication requirements.

**Scope**: `ptycho.workflows.components.py:676` → complete training/inference/persistence flow.

---

## 1. Primary Entry Point: `run_cdi_example`

**Location**: `ptycho/workflows/components.py:676-723`

**Signature**:
```python
def run_cdi_example(
    train_data: Union[RawData, PtychoDataContainer],
    test_data: Optional[Union[RawData, PtychoDataContainer]],
    config: TrainingConfig,
    flip_x: bool = False,
    flip_y: bool = False,
    transpose: bool = False,
    M: int = 20,
    do_stitching: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]
```

**Call Flow**:
```
run_cdi_example
  ├─> update_legacy_dict(params.cfg, config)     [706] ← CRITICAL CONFIG BRIDGE
  ├─> train_cdi_model(train_data, test_data, config)  [709]
  │    ├─> create_ptycho_data_container(train_data, config)  [591]
  │    │    └─> loader.load(grouped_data, probe, ...)  [569]
  │    ├─> create_ptycho_data_container(test_data, config)  [593]
  │    ├─> probe.set_probe_guess(None, train_container.probe)  [598]
  │    └─> train_pinn.train_eval(PtychoDataset(...))  [604]
  │         ├─> train(train_data)  [85]
  │         │    ├─> calculate_intensity_scale(train_data)  [70]
  │         │    ├─> params.set('intensity_scale', ...)  [71]
  │         │    ├─> probe.set_probe_guess(None, train_data.probe)  [74]
  │         │    └─> model.train(nepochs, train_data)  [81]
  │         └─> eval(test_data, history, trained_model)  [91]
  │              └─> model.predict([X*intensity_scale, coords_nominal])  [137-139]
  │
  └─> reassemble_cdi_image(test_data, config, ...) [714-719] ← IF do_stitching
       ├─> create_ptycho_data_container(test_data, config)  [637]
       ├─> nbutils.reconstruct_image(test_container)  [640]
       └─> tf_helper.reassemble_position(obj_tensor_full, global_offsets, M)  [662]
```

---

## 2. Configuration Management Touchpoints

### 2.1. Bridge Invocation (CRITICAL)
**Location**: `ptycho/workflows/components.py:706`
```python
update_legacy_dict(params.cfg, config)  # One-way TrainingConfig → params.cfg
```
- **Purpose**: Populate global `params.cfg` before ANY legacy module calls
- **Side Effect**: Modifies `ptycho.params.cfg` dictionary in-place
- **Requirement**: PyTorch orchestrator MUST call this at entry to satisfy CONFIG-001

### 2.2. Parameters Accessed During Training
From `ptycho/train_pinn.py`:
- `params.cfg['nepochs']` (line 79) - training loop count
- `params.cfg['intensity_scale']` (set at line 71) - diffraction scaling
- `params.cfg` passed to `model.train()` implicitly

From `ptycho/model.py` (via params.get()):
- `N`, `gridsize`, `model_type`, `amp_activation`, `probe.trainable`, etc.

**Implication**: PyTorch must either:
(A) Call `update_legacy_dict` to populate `params.cfg`, OR
(B) Refactor all legacy calls to accept explicit config arguments

---

## 3. Data Flow Transformations

### 3.1. Input → Model-Ready Pipeline

```
RawData (NumPy arrays)
  ↓ generate_grouped_data(N, K, nsamples, gridsize)
  ↓ ptycho/raw_data.py:365-438
Grouped Data Dict (NumPy)
  {diffraction, coords_offsets, coords_relative, local_offsets, ...}
  ↓ loader.load(lambda: grouped_data, probe, ...)
  ↓ ptycho/loader.py:93-200
PtychoDataContainer (TensorFlow tensors)
  .X, .Y, .Y_I, .Y_phi, .coords_nominal, .probe, .nn_indices,
  .global_offsets, .local_offsets, .coords_true
  ↓ model.train(nepochs, train_data) OR model.predict(...)
  ↓ ptycho/model.py:217-560
Training/Inference Outputs
  reconstructed_obj, pred_amp, reconstructed_obj_cdi
```

**Key Shapes** (for N=64, gridsize=2):
- `X`: (n_groups, 128, 128, 4) - diffraction input (4 = gridsize²)
- `Y`: (n_groups, 128, 128, 4) - ground truth complex (if supervised)
- `coords_nominal`: (n_groups, 2, 2) - position offsets
- `reconstructed_obj`: (n_groups, 128, 128, 4) - model output

### 3.2. Output → Visualization Pipeline

```
Model Predictions
  reconstructed_obj (n_groups, H, W, channels)
  ↓ reassemble_patches() OR reassemble_position()
  ↓ ptycho/image/reassembly.py OR ptycho/tf_helper.py
Full-Field Reconstruction
  Complex 2D array (M_obj, M_obj)
  ↓ np.absolute(), np.angle()
Amplitude + Phase Images
  ↓ save_outputs(...) matplotlib.pyplot.imsave()
PNG Files
```

---

## 4. ModelManager Persistence Interactions

### 4.1. Save Flow (Training Exit)

**Triggered by**: `scripts/training/train.py` → `model_manager.save(out_prefix)`

**Location**: `ptycho/model_manager.py:425-466`
```python
save(out_prefix: str) -> None:
  ├─> models_dict = {'autoencoder': model.autoencoder,
  │                  'diffraction_to_obj': model.diffraction_to_obj}
  ├─> custom_objects = {ProbeIllumination, IntensityScaler, ...}
  └─> ModelManager.save_multiple_models(models_dict, model_path,
                                         custom_objects,
                                         params.get('intensity_scale'))
       ├─> For each model:
       │    ├─> model.save(model_dir/model.keras)           [Keras 3 format]
       │    ├─> dill.dump(custom_objects, 'custom_objects.dill')  [67-71]
       │    └─> dill.dump(params.cfg.copy(), 'params.dill')       [73-78]
       └─> zipfile.ZipFile(f"{base_path}.zip", 'w')        [373-378]
```

**Archive Structure**:
```
wts.h5.zip
├── manifest.dill            (model names list)
├── autoencoder/
│   ├── model.keras          (Keras 3 SavedModel)
│   ├── custom_objects.dill  (Lambda layers, custom classes)
│   └── params.dill          (params.cfg snapshot + intensity_scale)
└── diffraction_to_obj/
    ├── model.keras
    ├── custom_objects.dill
    └── params.dill
```

### 4.2. Load Flow (Inference Entry)

**Location**: `ptycho/workflows/components.py:102-184` (`load_inference_bundle`)

```python
load_inference_bundle(model_dir: Path) -> Tuple[tf.keras.Model, dict]:
  ├─> ModelManager.load_multiple_models(str(model_zip))
  │    ├─> zipfile.ZipFile.extractall(temp_dir)                    [396-399]
  │    ├─> dill.load('manifest.dill')                              [402-404]
  │    └─> For requested models:
  │         └─> ModelManager.load_model(model_subdir)               [420]
  │              ├─> dill.load('params.dill')                       [105-106]
  │              ├─> params.cfg.update(loaded_params)               [119] ← SIDE EFFECT
  │              ├─> dill.load('custom_objects.dill')               [122-123]
  │              ├─> create_model_with_gridsize(gridsize, N)        [176] ← BLANK MODEL
  │              └─> tf.keras.models.load_model('model.keras', ...)  [200-203]
  │
  ├─> model = models_dict['diffraction_to_obj']                     [171]
  └─> config = params.cfg.copy()                                    [175]
```

**Critical Side Effects**:
1. `params.cfg` is MUTATED on load (line 119) - PyTorch must replicate
2. Blank model created with architecture params BEFORE weight loading (line 176)
3. Custom objects registry required for TensorFlow Lambda layers

---

## 5. Critical Side Effects (Global State Modifications)

### 5.1. `params.cfg` Mutations

| Location | Operation | Purpose |
|----------|-----------|---------|
| `components.py:706` | `update_legacy_dict(params.cfg, config)` | Populate global params at workflow entry |
| `train_pinn.py:71` | `params.set('intensity_scale', ...)` | Store calculated scale for model |
| `model_manager.py:119` | `params.cfg.update(loaded_params)` | Restore saved params on model load |
| `probe.py` | `params.set('probe', ...)` | Register probe tensor globally |

### 5.2. File System Operations

**Training Outputs** (via `save_outputs`):
- `{output_dir}/wts.h5.zip` - model archive
- `{output_dir}/reconstructed_amplitude.png`
- `{output_dir}/reconstructed_phase.png`

**Temporary Files**:
- ModelManager uses `tempfile.TemporaryDirectory()` for zip extraction (auto-cleanup)

### 5.3. TensorFlow Graph Mutations

**Model Instantiation** (`ptycho/model.py`):
- `create_model_with_gridsize(gridsize, N)` builds TF computational graph
- Custom layers (`ProbeIllumination`, `IntensityScaler`) with trainable variables
- Graph structure FROZEN after first call (TF 2.x eager mode)

---

## 6. PyTorch Replication Requirements

### 6.1. MUST Replicate

1. **Config Bridge Call**: `update_legacy_dict(params.cfg, config)` at workflow entry (CONFIG-001)
2. **params.cfg Population**: All fields accessed by legacy modules (N, gridsize, nepochs, intensity_scale, etc.)
3. **Data Pipeline**: RawData → grouped_data → DataContainer (Phase C adapters handle this)
4. **Model Persistence**: Archive format compatible with `wts.h5.zip` structure OR separate PyTorch format
5. **Probe Initialization**: `probe.set_probe_guess()` side effect (or equivalent)

### 6.2. MAY Diverge

1. **Training Loop**: Lightning `Trainer.fit()` vs Keras `model.fit()` - different callbacks/logging
2. **Persistence Format**: PyTorch state_dict + Lightning checkpoint vs Keras SavedModel
3. **Custom Layers**: PyTorch `nn.Module` vs TensorFlow custom layers (no direct equivalence)
4. **Stitching Details**: Can reuse `reassemble_position` logic but tensor types differ

### 6.3. Open Questions for D1.C Decision

1. **MLflow Coupling**: PyTorch inference.py hardcodes MLflow loading - can we make optional?
2. **API Surface**: Should PyTorch expose `run_cdi_example_torch()` OR `ptycho_torch.api.train()`?
3. **Config Translation**: Where to insert config_bridge calls - API wrapper or low-level shims?
4. **Archive Compatibility**: Should PyTorch produce TensorFlow-compatible archives OR parallel format?

---

## 7. Minimal Working Example (Pseudo-PyTorch)

**Hypothetical PyTorch orchestrator** (to guide D2 implementation):

```python
# ptycho_torch/workflows/components.py (NEW)
from ptycho.config.config import TrainingConfig, update_legacy_dict
from ptycho import params
from ptycho_torch.config_bridge import to_training_config, to_model_config
from ptycho_torch.raw_data_bridge import RawDataTorch
from ptycho_torch.data_container_bridge import PtychoDataContainerTorch
import lightning as L

def run_cdi_example_torch(
    train_data: Union[RawDataTorch, PtychoDataContainerTorch],
    test_data: Optional[Union[RawDataTorch, PtychoDataContainerTorch]],
    config: TrainingConfig,  # TensorFlow-style config from Phase B bridge
    do_stitching: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """PyTorch equivalent of run_cdi_example."""

    # CRITICAL: Populate params.cfg (CONFIG-001 compliance)
    update_legacy_dict(params.cfg, config)

    # Train model using Lightning
    train_results = train_cdi_model_torch(train_data, test_data, config)

    recon_amp, recon_phase = None, None

    # Reassemble if requested
    if do_stitching and test_data is not None:
        from ptycho_torch.reassembly import reconstruct_image_barycentric
        recon_amp, recon_phase = reconstruct_image_barycentric(
            train_results['model'], test_data, config
        )

    return recon_amp, recon_phase, train_results

def train_cdi_model_torch(
    train_data, test_data, config: TrainingConfig
) -> Dict[str, Any]:
    """PyTorch training orchestration."""
    from ptycho_torch.train import PtychoPINN_Lightning
    from ptycho_torch.train_utils import PtychoDataModule

    # Convert TensorFlow config → PyTorch singleton configs (Phase B bridge)
    # (This step may be implicit if config_bridge already called)

    # Create Lightning components
    data_module = PtychoDataModule(...)
    model = PtychoPINN_Lightning(...)
    trainer = L.Trainer(max_epochs=config.nepochs, ...)

    # Train
    trainer.fit(model, datamodule=data_module)

    return {'model': model, 'trainer': trainer, 'checkpoint_path': ...}
```

---

## 8. Summary Table: TensorFlow vs PyTorch Parity

| Component | TensorFlow | PyTorch Equivalent | Parity Status |
|-----------|------------|-------------------|---------------|
| Entry point | `run_cdi_example` | `run_cdi_example_torch` (NEW) | ❌ Missing |
| Config bridge | `update_legacy_dict` at line 706 | MUST replicate | ✅ Phase B (but not called) |
| Data pipeline | RawData → loader.load → PtychoDataContainer | RawDataTorch → PtychoDataContainerTorch | ✅ Phase C |
| Training loop | `train_pinn.train_eval` → `model.train` | `train.main` → `Trainer.fit` | ⚠️ Exists but uncoupled |
| Probe init | `probe.set_probe_guess` (global) | Needs equivalent | ❌ Missing |
| Persistence | `ModelManager.save_multiple_models` | MLflow-only | ⚠️ Incompatible |
| Inference | `model.predict` → `reassemble_position` | `inference.load_and_predict` | ⚠️ MLflow-only |
| Stitching | `reassemble_position` (TF tensors) | `reconstruct_image_barycentric` (torch) | ✅ Exists |

**Legend**: ✅ Complete | ⚠️ Partial/Issues | ❌ Missing

---

**Next Steps** (D1.B): Inventory `ptycho_torch/` modules to assess reuse vs gaps.
