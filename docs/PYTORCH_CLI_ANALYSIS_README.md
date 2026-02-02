# PyTorch CLI Entry Points: Complete Analysis Suite

**Analysis Date**: October 19, 2025  
**Status**: COMPLETE  
**Total Documentation**: 1,314 lines across 3 comprehensive documents  
**Scope**: train.py + inference.py CLI entry points with full config mapping

**Practical guide:** For a hands-on, minimal path to load and run PyTorch inference
(including state_dict-only models), see `ptycho_torch/README.md`.

---

## Document Suite Overview

This analysis provides exhaustive coverage of the PyTorch CLI infrastructure, from raw command-line flags through multi-layer config transformations to legacy CONFIG-001 compliance.

### Document 1: pytorch_cli_inventory.md (535 lines, 22 KB)

**Purpose**: Comprehensive reference for CLI flags, config field mappings, and manual wiring patterns

**Contents**:
- Part 1: Training CLI Flags (10 tables covering 18 flags)
- Part 2: Inference CLI Flags (11 tables covering 10 flags)
- Part 3: Config Dataclass Field Mapping (5 tables across all PyTorch dataclasses)
- Part 4: Config Bridge Transformations (5 critical transformations with code examples)
- Part 5: Execution-Only Knobs (10 parameters scattered across codebase)
- Part 6: Current Manual Wiring Patterns (7 factory replacement opportunities)
- Part 7: Summary of Gaps & Factory Opportunities (comprehensive replacement roadmap)
- Part 8: CLI Invocation Examples (ready-to-run commands)
- Appendix: File:Line Citation Index (complete source code references)

**Use Cases**:
- Finding specific CLI flag behavior
- Understanding config field transformations
- Identifying where to implement factory patterns
- Tracing data flow through config bridge

**Key Sections with Line Citations**:
```
train.py lines 366-570: CLI entrypoint
train.py lines 475-501: Manual config instantiation
config_bridge.py lines 79-306: Critical transformations
train.py lines 515-532: Bridge call with overrides
```

---

### Document 2: cli_flags_quick_reference.md (229 lines, 6.7 KB)

**Purpose**: Quick lookup guide for users and developers

**Contents**:
- Training CLI flags (summary table + examples)
- Inference CLI flags (summary table + examples)
- Configuration mapping summary
- Critical implementation details (6 key patterns)
- Factory refactoring roadmap
- Testing checklist

**Use Cases**:
- Quick flag lookup while using CLI
- Copy-paste command examples
- Understanding critical implementation patterns
- Testing checklist for validation

**Example Commands Provided**:
```bash
# Minimal training
python -m ptycho_torch.train --train_data_file data/train.npz --output_dir ./outputs

# Full-featured training
python -m ptycho_torch.train --train_data_file data/train.npz --test_data_file data/test.npz \
  --output_dir ./outputs --max_epochs 50 --batch_size 8 --gridsize 4 --n_images 256 \
  --device cuda --disable_mlflow

# Minimal inference
python -m ptycho_torch.inference --model_path training_outputs --test_data test.npz --output_dir inference_outputs

# Full-featured inference
python -m ptycho_torch.inference --model_path training_outputs --test_data datasets/Run1084.npz \
  --output_dir inference_outputs --n_images 64 --device cuda --quiet
```

---

### Document 3: cli_config_dataflow.md (550 lines, 19 KB)

**Purpose**: Deep technical documentation of complete data flow through config transformations

**Contents**:
- Section 1: Training Workflow (5 phases of transformation)
  - Phase 1A: CLI argument parsing
  - Phase 1B: Interface selection (legacy vs. new)
  - Phase 2A: PyTorch config instantiation
  - Phase 2B: Field assignments with source documentation
  - Phase 3A: Device resolution
  - Phase 3B: Config bridge transformation
  - Phase 4: params.cfg population (CONFIG-001)
  - Phase 5: Training execution

- Section 2: Inference Workflow (8 phases)
  - Argument parsing → validation → checkpoint discovery → model loading → data loading → 
  - dtype casting → forward pass → output saving

- Section 3: Config Transformation Summary Table (25 transformation steps documented)

- Section 4: Key Design Patterns (3 patterns explained)
  - Overrides dictionary pattern
  - Validation at bridge layer
  - Execution-only knobs

- Section 5: Critical Checkpoints for CONFIG-001 Compliance

**Use Cases**:
- Understanding complete data flow
- Debugging config-related issues
- Understanding bridge transformation logic
- Ensuring CONFIG-001 compliance in new code

**Visual Format**: ASCII art flow diagrams showing transformation pipelines

---

## Key Findings Summary

### 1. CLI Interfaces (train.py)

**New CLI (Phase E2.C1)** - 8 Primary Flags:
- `--train_data_file` (REQUIRED): Training dataset
- `--output_dir` (REQUIRED): Checkpoint directory
- `--max_epochs` (default: 100): Training epochs
- `--batch_size` (default: 16): Batch size
- `--n_images` (default: 512): Number of diffraction groups
- `--gridsize` (default: 2): Grid size
- `--device` (default: cpu): cpu or cuda
- `--disable_mlflow` (flag): Disable tracking

**Legacy CLI** (Backward Compatible) - 2 Flags:
- `--ptycho_dir` (REQUIRED): Ptycho directory
- `--config` (optional): JSON config file

**Interface Enforcement** (line 407-414):
- Mutually exclusive: Cannot mix new + legacy flags
- Fail-fast validation on both specified

---

### 2. Config Mapping Architecture

```
CLI Arguments (8 flags)
          ↓
PyTorch Configs (5 dataclasses: DataConfig, ModelConfig, TrainingConfig, InferenceConfig, DatagenConfig)
          ↓
Config Bridge (to_model_config, to_training_config)
          ↓
TensorFlow Configs (ModelConfig, TrainingConfig, InferenceConfig)
          ↓
Legacy params.cfg (update_legacy_dict)
```

**Critical Transformations**:
1. Grid size: `Tuple[int, int]` → `int` (line 107-113)
2. Mode: `'Unsupervised'` → `'pinn'` (line 116-124)
3. Activation: `'silu'` → `'swish'` (line 127-140)
4. NLL: `bool` → `float` (line 218)
5. Nphotons: Explicit override enforcement (line 261-269)

---

### 3. Probe Size Inference (NEW in Phase E2.C1)

**Location**: train.py lines 468-473  
**Method**: NPZ metadata reading without full array load  
**Function**: `_infer_probe_size(npz_file)` → reads `probeGuess.shape[0]`  
**Fallback**: Default N=64 if inference fails  
**Output**: Inferred_N → DataConfig.N

---

### 4. Device Resolution Strategy

**Location**: train.py line 493, train_utils.py lines 62-88  
**Logic**:
```python
if args.device == 'cpu':
    n_devices = 1
else:
    n_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
```

**Used For**:
- Setting `TrainingConfig.n_devices`
- Calling `get_training_strategy(n_devices)`
- Configuring Lightning Trainer

---

### 5. Manual Wiring Patterns Identified (7 Opportunities for Factories)

| # | Current Location | Pattern | Proposed Factory |
|---|---|---|---|
| 1 | train.py:475-501 | Manual DataConfig(...) instantiation | PyTorchConfigFactory.from_cli_args() |
| 2 | train.py:515-532 | Manual bridge call with dict overrides | TensorFlowConfigFactory.from_pytorch_configs() |
| 3 | train.py:493 | Inline device resolution ternary | DeviceResolver.compute_n_devices() |
| 4 | train.py:468-473 | Manual _infer_probe_size() + fallback | ProbeMetadataFactory.infer_N_from_npz() |
| 5 | inference.py:412-429 | Manual checkpoint candidate loop | CheckpointResolver.find_lightning_checkpoint() |
| 6 | inference.py:494-495 | Inline dtype casting | TorchDataFactory.cast_diffraction_data() |
| 7 | inference.py:499-501 | Manual shape detection + permute | DiffractionShapeFactory.normalize_shape() |

---

### 6. Execution-Only Knobs (NOT in Config Dataclasses)

**Training**:
- Learning rate scaling (computed from n_devices + batch_size)
- Training strategy selection (auto vs. DDPStrategy)
- Checkpoint root directory
- MLflow experiment management
- Validation split (5% hardcoded)

**Inference**:
- Checkpoint search path and candidate list
- Data type casting (float32, complex64)
- Diffraction shape transposition logic

---

### 7. Critical Validations in Config Bridge (config_bridge.py)

**Nphotons Default Divergence** (line 261-269):
- PyTorch default: 1e5
- TensorFlow default: 1e9
- Requires explicit override to prevent silent divergence
- RAISES ValueError if default not overridden

**N_groups Required Override** (line 271-278):
- Must be provided in overrides dict
- RAISES ValueError if missing
- Prevents params.cfg['n_groups'] = None

**Test_data_file Optional Warning** (line 282-290):
- WARNS if not provided
- Necessary for evaluation workflows

---

### 8. CONFIG-001 Compliance Checkpoint (train.py:535)

```python
update_legacy_dict(params.cfg, tf_training_config)  # Line 535
```

**Critical Step**: After this line, legacy modules can safely access params.cfg

**Before**: params.cfg is empty/uninitialized → Silent failures  
**After**: params.cfg is fully populated → CONFIG-001 compliant

**Guaranteed Fields Post-Population**:
- N, gridsize, n_groups, epochs, batch_size
- nphotons, neighbor_count, backend
- 40+ additional fields from TFTrainingConfig

---

## File:Line Citation Index (Quick Reference)

### train.py (ptycho_torch/train.py)

| Section | Lines | Purpose |
|---------|-------|---------|
| cli_main() function | 353-570 | Main CLI entrypoint |
| argparse setup | 366-403 | Flag definitions |
| Interface detection | 407-414 | Legacy vs. new selection |
| Probe inference | 468-473 | NPZ metadata reading |
| Config instantiation | 475-501 | Manual config creation |
| Bridge call | 515-532 | to_training_config() with overrides |
| params.cfg population | 535 | CONFIG-001 compliance |
| main() delegation | 549-555 | Training execution |

### inference.py (ptycho_torch/inference.py)

| Section | Lines | Purpose |
|---------|-------|---------|
| cli_main() function | 293-571 | Main CLI entrypoint |
| Mode detection | 576-578 | Lightning vs. MLflow routing |
| argparse setup | 319-380 | Flag definitions |
| Input validation | 399-409 | Path existence checks |
| Checkpoint search | 412-429 | Candidate list iteration |
| Model loading | 442-451 | Lightning checkpoint load |
| Data loading | 462-480 | NPZ file validation |
| Dtype casting | 494-495 | float32 + complex64 |
| Shape normalization | 499-501 | (H,W,n) → (n,H,W) |
| Forward pass | 531-538 | Model inference |
| Output saving | 559 | PNG file generation |

### config_bridge.py (ptycho_torch/config_bridge.py)

| Function | Lines | Key Transformations |
|----------|-------|-------------------|
| to_model_config() | 79-182 | gridsize, mode, activation, probe_mask |
| to_training_config() | 185-306 | epochs→nepochs, K→neighbor_count, nll→float, nphotons validation, n_groups validation |
| to_inference_config() | 309-360 | K→neighbor_count, model reference |

---

## Using This Analysis Suite

### For CLI Users
**Start with**: `cli_flags_quick_reference.md`
- Copy-paste command examples
- Check default values
- See critical implementation details

### For Developers
**For understanding flow**: 
1. Read `cli_flags_quick_reference.md` (5 minutes)
2. Study `cli_config_dataflow.md` Phase 1-5 (15 minutes)
3. Reference `pytorch_cli_inventory.md` for specific details

**For implementing factories**:
1. Identify pattern in Part 6 of `pytorch_cli_inventory.md`
2. Study current implementation in source files (line citations provided)
3. Check Part 5 of `pytorch_cli_inventory.md` for factory candidates

### For Debugging
**Config-related issues**:
1. Trace issue through Section 3 of `cli_config_dataflow.md`
2. Check critical transformations in Part 4 of `pytorch_cli_inventory.md`
3. Verify CONFIG-001 compliance checkpoint (Section 5 of `cli_config_dataflow.md`)

**CLI flag behavior**:
1. Find flag in training/inference table in `pytorch_cli_inventory.md` Part 1-2
2. Check quick reference for default value
3. Trace mapping through Part 3 for dataclass field

---

## Recommendations for Future Work

### Phase 1: Factory Pattern Implementation
Implement 7 factory classes identified in Part 7 of `pytorch_cli_inventory.md`
**Benefit**: Reduce manual wiring from ~60 lines to ~10 lines per entrypoint
**Time**: 3-4 development iterations

### Phase 2: Config Validation Enhancement
Extend config bridge validations (currently 3 validations in config_bridge.py:261-290)
**Benefit**: Catch misconfigurations earlier, better error messages
**Examples**: Type validation for all fields, range checking for numeric fields

### Phase 3: CLI Interface Modernization
Consider unified CLI interface that accepts both PyTorch and TensorFlow flags
**Benefit**: Reduced cognitive load for users, consistent experience
**Prerequisite**: Factory pattern implementation

---

## Document Metadata

| Property | Value |
|----------|-------|
| Total Lines | 1,314 |
| Total Size | 47.7 KB |
| Analysis Scope | train.py (576 lines) + inference.py (601 lines) |
| Config Files Analyzed | config_params.py, config_bridge.py |
| Total Flags Documented | 28 (18 training + 10 inference) |
| Config Fields Documented | 80+ |
| Transformations Detailed | 30+ |
| Code Citations | 100+ line:range references |
| Tables | 20+ reference tables |

---

**Created**: October 19, 2025  
**Analysis Depth**: EXHAUSTIVE (all code paths traced)  
**Validation**: Verified against source code with line citations  
**Ready for**: Architecture documentation, factory implementation planning, developer onboarding

For questions or corrections, cross-reference specific line citations in the source documents.
