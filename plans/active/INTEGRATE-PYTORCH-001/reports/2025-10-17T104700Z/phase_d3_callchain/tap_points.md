# Phase D3 Persistence Trace — Tap Points Plan

**Initiative:** INTEGRATE-PYTORCH-001 | **Phase:** D3.A | **Date:** 2025-10-17

---

## Purpose

Define 5–7 numeric/structural tap points aligned to the persistence callgraph for Phase D3.B/C validation. Taps capture intermediate state to verify correctness without re-implementing algorithms.

---

## Tap Point Specifications

### Tap 1: Config Snapshot at Save Entry
- **Key:** `tap_config_snapshot_save`
- **Purpose:** Validate that `params.cfg` is fully populated before ModelManager.save captures snapshot
- **Owning Function:** `ptycho/model_manager.py::ModelManager.save_model:74`
- **Location:** Just before `params_dict = params.cfg.copy()`
- **Expected Output:**
  ```python
  {
    'N': 64,
    'gridsize': 2,
    'nphotons': 1e9,
    'model_type': 'pinn',
    'intensity_scale': 0.00123,  # Runtime-computed value
    # ... all other TrainingConfig fields
  }
  ```
- **Units/Type:** Dict[str, Any]
- **Validation:** Assert `gridsize` and `N` are not None; assert `intensity_scale` > 0

---

### Tap 2: Archive Manifest Contents
- **Key:** `tap_manifest_structure`
- **Purpose:** Verify dual-model bundle structure before zip creation
- **Owning Function:** `ptycho/model_manager.py::ModelManager.save_multiple_models:361-364`
- **Location:** After `dill.dump(manifest, f)` in save_multiple_models
- **Expected Output:**
  ```python
  {
    'models': ['autoencoder', 'diffraction_to_obj'],
    'version': '1.0'
  }
  ```
- **Units/Type:** Dict with 'models' list and 'version' string
- **Validation:** Assert len(models) == 2; assert 'diffraction_to_obj' in models

---

### Tap 3: Per-Model File Inventory
- **Key:** `tap_model_files_created`
- **Purpose:** Validate each model subdirectory contains required files before zipping
- **Owning Function:** `ptycho/model_manager.py::ModelManager.save_model:64-82`
- **Location:** End of save_model (after all file writes)
- **Expected Output (per model):**
  ```python
  {
    'model_dir': '/tmp/xyz/autoencoder',
    'files_created': [
      'model.keras',
      'custom_objects.dill',
      'params.dill',
      'model.h5'
    ],
    'params_snapshot_keys': ['N', 'gridsize', 'intensity_scale', '_version', ...],
    'custom_objects_count': 25
  }
  ```
- **Units/Type:** Dict with file list + metadata
- **Validation:** Assert all 4 files exist; assert params has '_version'; assert custom_objects_count > 20

---

### Tap 4: Loaded Params Before cfg.update
- **Key:** `tap_params_before_restore`
- **Purpose:** Capture loaded params.dill contents before mutating global params.cfg
- **Owning Function:** `ptycho/model_manager.py::ModelManager.load_model:105-106`
- **Location:** After `loaded_params = dill.load(f)`, before line 119 `params.cfg.update`
- **Expected Output:**
  ```python
  {
    'gridsize': 2,
    'N': 64,
    'intensity_scale': 0.00123,
    '_version': '1.0',
    # ... full training-time params
  }
  ```
- **Units/Type:** Dict[str, Any]
- **Validation:** Assert `gridsize` and `N` match expected training values; assert `_version` exists

---

### Tap 5: Model Reconstruction Inputs
- **Key:** `tap_model_reconstruction_inputs`
- **Purpose:** Verify gridsize/N extracted correctly for blank model creation
- **Owning Function:** `ptycho/model_manager.py::ModelManager.load_model:112-116`
- **Location:** After extracting gridsize & N, before line 176 `create_model_with_gridsize`
- **Expected Output:**
  ```python
  {
    'gridsize': 2,
    'N': 64,
    'source': 'loaded_params',
    'validation_passed': True  # gridsize and N are not None
  }
  ```
- **Units/Type:** Dict with gridsize, N, validation flag
- **Validation:** Assert gridsize and N are integers > 0; assert validation_passed == True

---

### Tap 6: Archive Extraction File List
- **Key:** `tap_extracted_files`
- **Purpose:** Validate archive contents after extraction in load path
- **Owning Function:** `ptycho/model_manager.py::ModelManager.load_multiple_models:398-399`
- **Location:** After `zf.extractall(temp_dir)`, before manifest load
- **Expected Output:**
  ```python
  {
    'temp_dir': '/tmp/abc123',
    'extracted_files': [
      'manifest.dill',
      'autoencoder/model.keras',
      'autoencoder/params.dill',
      'autoencoder/custom_objects.dill',
      'autoencoder/model.h5',
      'diffraction_to_obj/model.keras',
      'diffraction_to_obj/params.dill',
      'diffraction_to_obj/custom_objects.dill',
      'diffraction_to_obj/model.h5'
    ],
    'manifest_exists': True,
    'model_dirs': ['autoencoder', 'diffraction_to_obj']
  }
  ```
- **Units/Type:** Dict with file list + directory list
- **Validation:** Assert manifest.dill exists; assert 2 model dirs present; assert each dir has 4 files

---

### Tap 7: PyTorch Checkpoint Structure (Baseline)
- **Key:** `tap_pytorch_checkpoint_keys`
- **Purpose:** Document current Lightning checkpoint contents for delta analysis
- **Owning Function:** `ptycho_torch/train.py:123-128` (ModelCheckpoint callback output)
- **Location:** Post-training, after `.ckpt` file saved (requires `torch.load` inspection)
- **Expected Output:**
  ```python
  {
    'checkpoint_path': 'outputs/checkpoints/best-checkpoint.ckpt',
    'keys': [
      'state_dict',          # Model weights
      'optimizer_states',    # Optimizer state
      'lr_schedulers',       # LR scheduler state
      'epoch',               # Training epoch
      'global_step',         # Step counter
      'pytorch-lightning_version',
      'callbacks',           # Lightning callback state
      # ... other Lightning metadata
    ],
    'state_dict_sample_keys': ['encoder.conv1.weight', 'decoder.conv_final.bias', ...],
    'has_params_snapshot': False,  # Key finding: params.cfg not captured
    'has_custom_objects': False    # Key finding: no custom objects serialization
  }
  ```
- **Units/Type:** Dict with checkpoint metadata
- **Validation:** Highlight missing `params_snapshot` and `custom_objects` keys (confirms delta analysis)

---

## Implementation Notes

### Tap Insertion Strategy
- **Phase D3.B (Writer Tests):** Taps 1-3 + 7 — validate save path creates correct structure
- **Phase D3.C (Loader Tests):** Taps 4-6 — validate load path restores params correctly

### Instrumentation Approach
- **Option A (Preferred):** Unit tests with mocked intermediate state → assert tap values
- **Option B:** Pytest fixtures capturing real training run outputs → inspect tap values
- **Option C:** Temporary logging statements during dev (remove before commit)

### Tap Output Format
```python
# Example pytest fixture capturing Tap 3
@pytest.fixture
def tap_model_files_created(tmp_path):
    """Capture files created during save_model call."""
    model_dir = tmp_path / "autoencoder"
    # ... call save_model ...
    return {
        'model_dir': str(model_dir),
        'files_created': [f.name for f in model_dir.iterdir()],
        'params_snapshot_keys': list(dill.load(open(model_dir / 'params.dill', 'rb')).keys()),
        'custom_objects_count': len(dill.load(open(model_dir / 'custom_objects.dill', 'rb')))
    }

def test_save_model_creates_required_files(tap_model_files_created):
    assert 'model.keras' in tap_model_files_created['files_created']
    assert 'params.dill' in tap_model_files_created['files_created']
    assert tap_model_files_created['custom_objects_count'] > 20
```

---

## Validation Checklist

For each tap point in Phase D3.B/C tests:
- [ ] Tap captures actual intermediate value (not re-derived)
- [ ] Tap validates against expected schema/range
- [ ] Tap failure provides actionable diagnostic message
- [ ] Tap does not modify production code hot path
- [ ] Tap documented with file:line anchor + expected units

---

## Cross-Reference to Callgraph

| Tap | Callgraph Stage | Edge Anchor |
|-----|-----------------|-------------|
| Tap 1 | Training → Save Entry | `model_manager.save:425` → `save_multiple_models:467` |
| Tap 2 | Archive Creation | `save_multiple_models:361-364` manifest write |
| Tap 3 | Per-Model Save | `save_multiple_models:370` → `save_model:64-82` |
| Tap 4 | Load → Params Restore | `load_model:105-106` params.dill load |
| Tap 5 | Model Reconstruction | `load_model:112-116` → `create_model_with_gridsize:176` |
| Tap 6 | Archive Extraction | `load_multiple_models:398-399` extract + manifest load |
| Tap 7 | PyTorch Baseline | `ptycho_torch/train.py:123-128` checkpoint callback |

---

## Summary

**Seven tap points cover:**
1. **Save entry** (Tap 1) — config completeness validation
2. **Archive structure** (Tap 2-3) — manifest + file inventory
3. **Load entry** (Tap 4-5) — params restoration + model reconstruction inputs
4. **Archive contents** (Tap 6) — extraction validation
5. **PyTorch baseline** (Tap 7) — document current state for delta analysis

**Next:** Implement taps as pytest fixtures in Phase D3.B/C test modules.
