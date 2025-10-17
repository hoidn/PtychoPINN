# PyTorch Backend Module Inventory

**Generated:** 2025-10-17T02:50:00Z
**Commit Context:** bfc22e7 (rebased PyTorch stack)
**Scope:** `ptycho_torch/` tree up to 2 levels deep
**Purpose:** Phase A.A1 — Authoritative module snapshot for integration plan refresh

---

## Complete File Listing

```
ptycho_torch/api/base_api.py
ptycho_torch/api/example_train.py
ptycho_torch/api/example_use.py
ptycho_torch/api/__init__.py
ptycho_torch/api/trainer_api.py
ptycho_torch/api/usage_example.ipynb
ptycho_torch/beta_modules/model_test.py
ptycho_torch/config_params.py
ptycho_torch/datagen/datagen.py
ptycho_torch/datagen/__init__.py
ptycho_torch/datagen/objects.py
ptycho_torch/datagen/probe.py
ptycho_torch/datagen.py
ptycho_torch/dataloader.py
ptycho_torch/dset_loader_pt_mmap.py
ptycho_torch/eval/eval_metrics.py
ptycho_torch/eval/frc.py
ptycho_torch/helper.py
ptycho_torch/inference.py
ptycho_torch/__init__.py
ptycho_torch/model_attention.py
ptycho_torch/model.py
ptycho_torch/notebooks/analysis.py
ptycho_torch/notebooks/calculate_frc.ipynb
ptycho_torch/notebooks/calculate_probe_similarity.ipynb
ptycho_torch/notebooks/create_supervised_datasets.ipynb
ptycho_torch/notebooks/inductive_module_ablation.ipynb
ptycho_torch/patch_generator.py
ptycho_torch/reassembly_alpha.py
ptycho_torch/reassembly_beta.py
ptycho_torch/reassembly.py
ptycho_torch/train_dummy.py
ptycho_torch/train_full.py
ptycho_torch/train.py
ptycho_torch/train_utils.py
ptycho_torch/utils.py
```

---

## High-Impact Module Annotations

### 🚨 **Critical: New `api/` Package** (Post-Rebase Addition)

**Modules:**
- `ptycho_torch/api/base_api.py` — **High-impact**
  - Defines `ConfigManager` class with methods:
    - `_from_mlflow(run_id, mlflow_tracking_uri)` — Loads configs from MLflow artifacts
    - `_from_json(json_path)` — Loads configs from JSON file
    - `_flexible_load()` — Hybrid MLflow + JSON override strategy
  - Exposes `PtychoDataLoader` wrapper for Lightning DataModule and TensorDict formats
  - Defines `PtychoModel` wrapper with persistence layer:
    - `save(strategy='mlflow' | 'pytorch')` — Strategy-based persistence
    - `load_from_mlflow(run_id, mlflow_tracking_uri)` — MLflow artifact restoration
  - Provides `Trainer` class with Lightning-specific setup
  - Includes `Datagen` class for synthetic data generation
  - **Contains `InferenceEngine` with `predict_and_stitch()` — direct analog to TensorFlow inference workflows**

- `ptycho_torch/api/trainer_api.py` — Not read but referenced by `base_api.py:684` for `setup_lightning_trainer()`
- `ptycho_torch/api/example_train.py`, `example_use.py`, `usage_example.ipynb` — Example scripts/notebooks demonstrating API usage

**Plan Impact:**
- **Spec touchpoint:** This package introduces a high-level API surface not covered in `plans/ptychodus_pytorch_integration_plan.md`. The API abstracts MLflow orchestration, model persistence, and inference stitching.
- **Integration gap:** `specs/ptychodus_api_spec.md` Section 4 ("PtychoPINN Reconstructor Contract") expects direct interaction with lower-level modules (`ptycho_torch.model`, `ptycho_torch.loader`). The new `api/` layer may offer a cleaner integration path **or** introduce a mismatch if `ptychodus` bypasses it.
- **Action:** Flag for delta_log.md — determine whether `ptychodus` integration should target `api/` layer or continue calling legacy-style modules.

---

### 🚨 **Critical: New `datagen/` Package** (Post-Rebase Addition)

**Modules:**
- `ptycho_torch/datagen/datagen.py` — **High-impact**
  - Implements `from_simulation()` — batched diffraction pattern generation with Poisson scaling and beamstop
  - Includes `get_image_patches()` — coordinate-based patch extraction from full canvas (uses `Translation` helper)
  - Provides `generate_simulated_data()`, `simulate_multiple_experiments()` for dataset creation
  - **Offers `generate_data_from_experiment()` — extracts supervised labels from experimental data, bypassing simulation**

- `ptycho_torch/datagen/objects.py`, `probe.py` — Synthetic object and probe generation utilities

**Plan Impact:**
- **Spec touchpoint:** `specs/data_contracts.md` Section 1 defines NPZ format contracts. The `datagen` package satisfies these but was not mentioned in the legacy plan.
- **Integration gap:** No linkage to `ptychodus` export workflows (`ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py:200-224` calls `export_training_data()` to emit NPZ).
- **Action:** Flag for delta_log.md — `datagen/` package could consolidate simulation/export logic; note as potential consolidation opportunity.

---

### 🚨 **Critical: Configuration Dataclasses** (Modified Post-Rebase)

**Module:** `ptycho_torch/config_params.py` — **Spec-breaking change**

**Defines:**
- `DataConfig` — 24 fields including `nphotons`, `N`, `C`, `K`, `grid_size`, `neighbor_function`, etc.
- `ModelConfig` — 26 fields including `mode`, `intensity_scale_trainable`, `n_filters_scale`, `amp_activation`, `object_big`, `probe_big`, `loss_function`, etc.
- `TrainingConfig` — 24 fields including `learning_rate`, `epochs`, `batch_size`, `scheduler`, `stage_1/2/3_epochs`, `experiment_name`, etc.
- `InferenceConfig` — 5 fields: `middle_trim`, `batch_size`, `experiment_number`, `pad_eval`, `window`
- `DatagenConfig` — 5 fields for synthetic data generation

**Key Observation:**
- **Different schema than TensorFlow backend** (`ptycho/config/config.py` defines `ModelConfig`, `TrainingConfig`, `InferenceConfig` with different field names and mappings).
- **No `KEY_MAPPINGS` dictionary** — PyTorch config does **not** populate legacy `params.cfg`.
- **No `update_legacy_dict()` bridge** — PyTorch backend operates independently of global state.

**Plan Impact:**
- **Spec violation:** `specs/ptychodus_api_spec.md:11-29` mandates `ModelConfig` with specific fields (`N`, `gridsize`, `model_type`, `amp_activation`, etc.). PyTorch `ModelConfig` has overlapping but divergent schema:
  - `grid_size: Tuple[int, int]` instead of scalar `gridsize`
  - `mode: Literal['Supervised', 'Unsupervised']` instead of `model_type: 'pinn' | 'supervised'`
  - Missing fields: `gaussian_smoothing_sigma`, `probe_scale`, `pad_object`
- **Critical Action:** Delta_log.md must flag this as **Phase B.B1 blocker**. Either:
  1. Harmonize PyTorch config schema with TensorFlow to satisfy spec, **or**
  2. Update `specs/ptychodus_api_spec.md` to document dual schema and provide translation layer

---

### 📦 **New Reassembly Modules** (Post-Rebase Addition)

**Modules:**
- `ptycho_torch/reassembly_alpha.py` — **Vectorized barycentric accumulator** for patch stitching
  - Implements `VectorizedBarycentricAccumulator` class with GPU-accelerated scatter operations
  - Provides `reconstruct_image()` multi-GPU inference function with DataParallel support
  - **Includes performance profiling** (inference time vs assembly time tracking)

- `ptycho_torch/reassembly_beta.py` — Likely an alternative reassembly strategy (not read)
- `ptycho_torch/reassembly.py` — Canonical reassembly entry point (not read, but imported by `api/base_api.py:938`)

**Plan Impact:**
- **Spec touchpoint:** `specs/ptychodus_api_spec.md:176-178` requires model outputs to be stitched via `ptycho.tf_helper.reassemble_position`. PyTorch provides alternative implementation.
- **Action:** Delta_log.md should note that reassembly logic diverges from TensorFlow's `tf_helper` approach; parity testing required (flag for `TEST-PYTORCH-001`).

---

### 📦 **Training Orchestration** (Modified Post-Rebase)

**Module:** `ptycho_torch/train.py` — **Lightning + MLflow integration**

**Key Components:**
- Uses **PyTorch Lightning** `Trainer` with callbacks (`ModelCheckpoint`, `EarlyStopping`)
- Integrates **MLflow autologging** (`mlflow.pytorch.autolog`)
- Implements `PtychoDataModule` for train/val splitting
- Supports **multi-stage training** (`stage_1/2/3_epochs` with physics weight scheduling)
- **DDP synchronization** with `dist.barrier()` before return

**Plan Impact:**
- **Integration gap:** TensorFlow backend uses `ptycho.workflows.components.run_cdi_example()` orchestration. PyTorch uses Lightning conventions.
- **Action:** Delta_log.md should note orchestration divergence; `ptychodus` integration must decide whether to call Lightning workflows or lower-level model API.

---

### 📦 **Data Loading** (Modified Post-Rebase)

**Modules:**
- `ptycho_torch/dataloader.py` — **TensorDictDataLoader** and `PtychoDataset` classes
- `ptycho_torch/dset_loader_pt_mmap.py` — Memory-mapped dataset implementation

**Plan Impact:**
- **Spec touchpoint:** `specs/ptychodus_api_spec.md:153-166` requires `RawData.generate_grouped_data()` dictionary format. PyTorch dataloader provides TensorDict format instead.
- **Action:** Delta_log.md should flag data pipeline divergence; parity verification needed.

---

### 📂 **Notebooks** (Post-Rebase Addition)

**Modules:**
- `ptycho_torch/notebooks/analysis.py`
- `ptycho_torch/notebooks/calculate_frc.ipynb`
- `ptycho_torch/notebooks/calculate_probe_similarity.ipynb`
- `ptycho_torch/notebooks/create_supervised_datasets.ipynb`
- `ptycho_torch/notebooks/inductive_module_ablation.ipynb`

**Note:** Notebooks are informational; omit from integration plan scope but preserve for reference/examples.

---

## Summary Statistics

- **Total Python modules:** 30 (excluding notebooks)
- **Total notebooks:** 6
- **New packages since last plan:** `api/`, `datagen/`, reassembly suite
- **Config schema drift:** PyTorch dataclasses diverge from TensorFlow spec (critical blocker)
- **New API surface:** `base_api.py` introduces high-level orchestration layer (300+ lines)

---

## References for Phase A.A2

Use this inventory to annotate `plans/ptychodus_pytorch_integration_plan.md` sections:
- **Section 2 (Configuration & Legacy Bridge):** Flag `config_params.py` schema mismatch
- **Section 3 (Data Pipeline):** Note `datagen/` package addition, memory-mapped dataloader
- **Section 4 (Model & Inference):** Reference `api/` layer and reassembly modules
- **Section 5 (Persistence):** Document MLflow-centric save/load in `api/base_api.py`

Cross-reference with `specs/ptychodus_api_spec.md` Section 5 field table (lines 220-273) to identify missing/renamed config fields.
