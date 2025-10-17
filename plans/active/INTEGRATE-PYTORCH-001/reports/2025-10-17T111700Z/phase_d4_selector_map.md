# Phase D4 Selector Map — Authoritative Pytest Commands & Environment Overrides

**Initiative:** INTEGRATE-PYTORCH-001 Phase D4.A3
**Date:** 2025-10-17
**Status:** Reference Document
**Purpose:** Define pytest selectors, environment overrides, and artifact storage expectations for PyTorch regression tests

---

## Overview

This document provides the authoritative map of pytest selectors for PyTorch backend regression testing. All commands are torch-optional (execute without PyTorch runtime via `tests/conftest.py` whitelist exemption) and target <60s runtime on CPU.

**Usage:** Reference this document when executing Phase D4.B (red tests), Phase D4.C (green tests), and TEST-PYTORCH-001 integration harness authoring.

---

## Environment Configuration

### Required Environment Variables

```bash
# Force CPU-only execution (no CUDA)
export CUDA_VISIBLE_DEVICES=""

# Disable MLflow network calls in tests
export MLFLOW_TRACKING_URI="memory"
```

### Optional Environment Variables

```bash
# Increase pytest verbosity for debugging
export PYTEST_ADDOPTS="-vv --tb=short"

# Skip torch-dependent tests explicitly (alternative to conftest auto-skip)
export SKIP_TORCH_TESTS=1  # (not currently implemented; reserved for future use)
```

### Python Environment Requirements

- **PyTorch:** Optional (tests execute without torch via shim patterns)
- **Lightning:** Required for Phase D4.B orchestration tests (stub validation only)
- **MLflow:** Optional (disabled via `MLFLOW_TRACKING_URI=memory`)
- **TensorFlow:** Required (adapter delegation to `ptycho.raw_data` and `ptycho.loader`)

---

## Selector Categories

### Category 1: Configuration Bridge (Phase B — COMPLETE)

**Module:** `tests/torch/test_config_bridge.py`
**Purpose:** Validate PyTorch config → TensorFlow dataclass → params.cfg translation
**Status:** Green (43/43 tests passing as of 2025-10-17)

#### Selectors

```bash
# MVP baseline (9 core fields)
pytest tests/torch/test_config_bridge.py -k mvp -vv

# All parity tests (38 spec-required fields)
pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity -vv

# Direct field translations (no overrides required)
pytest tests/torch/test_config_bridge.py -k "parity_direct" -vv

# Override-required fields
pytest tests/torch/test_config_bridge.py -k "parity_override" -vv

# Probe mask translation
pytest tests/torch/test_config_bridge.py -k "probe_mask" -vv

# Nphotons divergence validation
pytest tests/torch/test_config_bridge.py -k "nphotons" -vv

# n_subsample semantic collision guard
pytest tests/torch/test_config_bridge.py -k "n_subsample" -vv

# Baseline comparison (params.cfg snapshot validation)
pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_params_cfg_matches_baseline -vv

# Full suite
pytest tests/torch/test_config_bridge.py -vv
```

**Expected Runtime:** ~3-7s per selector

**Artifacts:**
- Baseline snapshot: `tests/torch/baseline_params.json` (31 keys)
- Parity evidence: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T050930Z/`

**Environment Notes:**
- Tests execute without PyTorch runtime (torch-optional imports)
- No GPU required
- No MLflow calls

---

### Category 2: Data Pipeline (Phase C — COMPLETE)

**Modules:** `tests/torch/test_data_pipeline.py`
**Purpose:** Validate RawDataTorch, PtychoDataContainerTorch, MemmapDatasetBridge adapters
**Status:** Green (6/6 tests passing as of 2025-10-17)

#### Selectors

```bash
# RawDataTorch parity (delegation to TensorFlow grouping)
pytest tests/torch/test_data_pipeline.py -k raw_data -vv

# PtychoDataContainerTorch parity (13 attributes, shapes, dtypes)
pytest tests/torch/test_data_pipeline.py -k data_container -vv

# Y patches complex64 enforcement (DATA-001 compliance)
pytest tests/torch/test_data_pipeline.py -k y_dtype -vv

# Memory-mapped dataset bridge
pytest tests/torch/test_data_pipeline.py -k memmap -vv

# Deterministic generation validation (cache-free behavior)
pytest tests/torch/test_data_pipeline.py::TestMemmapBridgeParity::test_deterministic_generation_validation -vv

# Full suite
pytest tests/torch/test_data_pipeline.py -vv
```

**Expected Runtime:** ~5-7s per selector

**Artifacts:**
- Data contract documentation: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T070200Z/data_contract.md`
- Gap matrix: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T070200Z/torch_gap_matrix.md`
- Cache semantics: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T084500Z/cache_semantics.md`

**Environment Notes:**
- Synthetic fixtures generated in-memory (no NPZ files required)
- `params.cfg` snapshot/restore via `params_cfg_snapshot` fixture
- TensorFlow required for baseline comparison

---

### Category 3: Workflow Orchestration (Phase D2 — COMPLETE)

**Module:** `tests/torch/test_workflows_components.py`
**Purpose:** Validate PyTorch workflow entry points and CONFIG-001 guards
**Status:** Green (3/3 tests passing as of 2025-10-17)

#### Selectors

```bash
# Scaffold smoke test (CONFIG-001 invocation)
pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsScaffold::test_run_cdi_example_calls_update_legacy_dict -vv

# Training orchestration (_ensure_container + Lightning delegation stub)
pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_train_cdi_model_torch_invokes_lightning -vv

# Run orchestration (train → optional stitching)
pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_run_cdi_example_invokes_training -vv

# Full suite
pytest tests/torch/test_workflows_components.py -vv
```

**Expected Runtime:** ~3-5s per selector

**Artifacts:**
- D2.A scaffold: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T091450Z/phase_d2_scaffold.md`
- D2.B training: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T094500Z/phase_d2_training.md`
- D2.C orchestration: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T101500Z/phase_d2c_green.md`

**Environment Notes:**
- Lightning stubs return placeholder results (no actual training)
- Monkeypatch spies validate orchestration contract
- No model weights or NPZ data required

---

### Category 4: Persistence (Phase D3 — COMPLETE)

**Module:** `tests/torch/test_model_manager.py`
**Purpose:** Validate save_torch_bundle / load_torch_bundle wts.h5.zip compatibility
**Status:** Green (4/4 tests passing as of 2025-10-17)

#### Selectors

```bash
# Archive structure validation
pytest tests/torch/test_model_manager.py::TestSaveTorchBundle::test_archive_structure -vv

# CONFIG-001 params snapshot
pytest tests/torch/test_model_manager.py::TestSaveTorchBundle::test_params_snapshot -vv

# Loader CONFIG-001 restoration
pytest tests/torch/test_model_manager.py::TestLoadTorchBundle::test_load_bundle_restores_params_cfg -vv

# Malformed archive error handling
pytest tests/torch/test_model_manager.py::TestLoadTorchBundle::test_malformed_archive_raises_error -vv

# Full suite
pytest tests/torch/test_model_manager.py -vv
```

**Expected Runtime:** ~7-10s per selector (includes zip I/O)

**Artifacts:**
- D3.A callchain: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T104700Z/phase_d3_callchain/`
- D3.B save impl: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T110500Z/phase_d3b_summary.md`
- D3.C load impl: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T113200Z/phase_d3c_summary.md`

**Environment Notes:**
- Tests create temporary zip archives in `tempfile.TemporaryDirectory()`
- Dual-model bundle structure (autoencoder + diffraction_to_obj subdirectories)
- Requires `dill` for manifest serialization
- Model reconstruction (`.load_state_dict()`) stubbed in current implementation

---

### Category 5: Regression Coverage (Phase D4.B — PENDING)

**Purpose:** Extend adapter-level tests with comprehensive regression coverage
**Status:** Red phase (tests to be authored in D4.B)

#### Planned Selectors (D4.B)

```bash
# Persistence regression (D4.B1)
pytest tests/torch/test_model_manager.py::TestPersistenceRegression -vv

# Orchestration regression (D4.B2)
pytest tests/torch/test_workflows_components.py::TestOrchestrationRegression -vv

# Cross-backend compatibility (D4.B2)
pytest tests/torch/test_cross_backend_parity.py -vv  # (new module)
```

**Expected Tests (D4.B scope per phase_d4_regression.md Q3):**

1. **Persistence Tests:**
   - `test_archive_roundtrip`: Save dual-model bundle → load → verify params restoration
   - `test_manifest_validation`: Verify manifest.dill structure + version='2.0-pytorch'
   - `test_params_snapshot_all_fields`: Validate all CONFIG-001 critical fields in params.dill
   - `test_cross_backend_archive_compatibility`: TensorFlow can read PyTorch archives (if feasible)

2. **Orchestration Tests:**
   - `test_run_cdi_example_end_to_end`: Full train+infer stub execution (no Lightning.fit)
   - `test_ensure_container_all_input_types`: Validate RawData, RawDataTorch, PtychoDataContainerTorch normalization
   - `test_config_bridge_automatic_invocation`: Confirm `update_legacy_dict` called without explicit user action

**Artifact Storage (D4.B):**
- Red logs: `plans/active/INTEGRATE-PYTORCH-001/reports/<ts>/phase_d4_red_persistence.log`
- Red logs: `plans/active/INTEGRATE-PYTORCH-001/reports/<ts>/phase_d4_red_workflows.log`
- Summary: `plans/active/INTEGRATE-PYTORCH-001/reports/<ts>/phase_d4_red_summary.md`

**Artifact Storage (D4.C):**
- Green logs: `plans/active/INTEGRATE-PYTORCH-001/reports/<ts>/phase_d4_green_persistence.log`
- Green logs: `plans/active/INTEGRATE-PYTORCH-001/reports/<ts>/phase_d4_green_workflows.md`
- Handoff: `plans/active/INTEGRATE-PYTORCH-001/reports/<ts>/phase_d4_handoff.md`

---

## Integration Test Selectors (TEST-PYTORCH-001 — FUTURE)

**Module:** `tests/test_pytorch_integration.py` (to be authored)
**Purpose:** Subprocess-based train → save → load → infer cycle
**Status:** Not started (activation gated on INTEGRATE-PYTORCH-001 D4.C completion)

#### Planned Selector (TEST-PYTORCH-001)

```bash
# Full integration workflow (mirrors tests/test_integration_workflow.py)
pytest tests/test_pytorch_integration.py::TestPyTorchFullWorkflow::test_train_save_load_infer_cycle -vv --timeout=120
```

**Environment Requirements:**
- MLflow disabled: `export MLFLOW_TRACKING_URI=memory`
- CPU-only: `export CUDA_VISIBLE_DEVICES=""`
- Fixtures: `tests/fixtures/pytorch_integration/` (minimal NPZ + probe)

**Expected Runtime:** <120s (per acceptance criteria)

**Subprocess Commands (to be implemented):**
```bash
# Training subprocess
python -m ptycho_torch.train \
  --ptycho_dir <tmpdir>/ptycho \
  --probe_dir <tmpdir>/probes \
  --max_epochs 1 \
  --device cpu \
  --disable_mlflow

# Inference subprocess
python -m ptycho_torch.inference \
  --checkpoint <tmpdir>/checkpoints/last.ckpt \
  --data_dir <tmpdir>/ptycho \
  --output_dir <tmpdir>/outputs
```

**Handoff Requirements:**
- INTEGRATE-PYTORCH-001 D4.C delivers green adapter tests + handoff package
- Fixtures align with `specs/data_contracts.md` (canonical NPZ schema)
- MLflow toggle implemented (`--disable_mlflow` CLI flag)

---

## Full Regression Suite (All Phases)

### Complete Test Run (Pre-D4.B)

```bash
# All torch-optional tests (config + data + workflows + persistence)
pytest tests/torch/ -vv
```

**Expected Output (as of 2025-10-17):**
- **Total:** 195 tests collected
- **Passed:** 195
- **Skipped:** 13 (unrelated modules like `test_benchmark_throughput.py`, `test_run_baseline.py`)
- **Failed:** 0
- **Runtime:** ~204s

### Targeted Smoke Test (Quick Validation)

```bash
# MVP config + key adapters
pytest \
  tests/torch/test_config_bridge.py::TestConfigBridgeMVP \
  tests/torch/test_data_pipeline.py::TestRawDataTorchAdapter::test_raw_data_torch_matches_tensorflow \
  tests/torch/test_workflows_components.py::TestWorkflowsComponentsScaffold \
  tests/torch/test_model_manager.py::TestSaveTorchBundle::test_archive_structure \
  -vv
```

**Expected Runtime:** ~15-20s

---

## Artifact Storage Conventions

### Log File Naming

```bash
# Phase identifier + selector category + pass/fail state
phase_d4_red_persistence.log       # D4.B persistence tests (failing)
phase_d4_green_workflows.log       # D4.C orchestration tests (passing)
pytest_mvp.log                      # MVP selector run
pytest_full.log                     # Full suite run
```

### Directory Structure

```
plans/active/INTEGRATE-PYTORCH-001/reports/
├── 2025-10-17T111700Z/              # D4.A (this artifact directory)
│   ├── phase_d4_alignment.md        # Ownership narrative
│   └── phase_d4_selector_map.md     # This document
├── <future-ts>/                      # D4.B (red phase)
│   ├── phase_d4_red_persistence.log
│   ├── phase_d4_red_workflows.log
│   └── phase_d4_red_summary.md
└── <future-ts>/                      # D4.C (green phase + handoff)
    ├── phase_d4_green_persistence.log
    ├── phase_d4_green_workflows.md
    └── phase_d4_handoff.md
```

### Snapshot File Naming

```bash
# Configuration baselines
baseline_params.json                 # TensorFlow canonical params.cfg snapshot
pytorch_config_snapshot.json         # PyTorch singleton state snapshot

# Archive structure examples
archive_manifest_snapshot.json       # manifest.dill contents
archive_tree.txt                     # Directory listing of wts.h5.zip contents
```

---

## Skip Guards & Torch-Optional Patterns

### Conftest Whitelist

**File:** `tests/conftest.py:38-46`

```python
TORCH_OPTIONAL_MODULES = [
    'test_config_bridge',
    'test_data_pipeline',
    'test_workflows_components',
    'test_model_manager',
]
```

**Behavior:** Tests in these modules execute without PyTorch runtime. If torch unavailable, imports fall back to shims (`TORCH_AVAILABLE = False`).

### Adding New Torch-Optional Modules

1. Add module name to `TORCH_OPTIONAL_MODULES` list in `tests/conftest.py`
2. Use guarded imports in test module:
   ```python
   try:
       import torch
       TORCH_AVAILABLE = True
   except ImportError:
       TORCH_AVAILABLE = False
       # Define shims or skip tests appropriately
   ```
3. Validate test collection succeeds without torch: `pytest --collect-only tests/torch/new_module.py`

---

## Environment Validation Commands

### Pre-Flight Check

```bash
# Verify torch-optional execution
pytest tests/torch/test_config_bridge.py::TestConfigBridgeMVP::test_mvp_config_bridge_populates_params_cfg -vv

# Expected output (without torch):
# 1 passed in ~3s
# No import errors or collection failures

# Verify MLflow toggle
MLFLOW_TRACKING_URI=memory pytest tests/torch/test_workflows_components.py -vv

# Expected output:
# 3 passed in ~5s
# No network calls to MLflow server
```

### Dependency Check

```bash
# Required packages
python -c "import ptycho; import ptycho_torch; import lightning; print('OK')"

# Optional packages (not required for torch-optional tests)
python -c "import torch; import mlflow; print('OK')" || echo "Optional packages missing (OK for torch-optional tests)"
```

---

## Troubleshooting

### Issue: Tests fail with "ModuleNotFoundError: ptycho_torch"

**Solution:** Install project in editable mode:
```bash
pip install -e .
```

### Issue: Tests skip with "PyTorch not available"

**Expected Behavior:** Tests should execute without torch via conftest whitelist.

**Debug Steps:**
1. Verify module in `TORCH_OPTIONAL_MODULES` list (`tests/conftest.py:38-46`)
2. Check for hard `import torch` statements (should use guarded imports)
3. Validate test file exists in `tests/torch/` directory

### Issue: Tests fail with "params.cfg not initialized"

**Root Cause:** CONFIG-001 violation — `update_legacy_dict` not called before legacy module access.

**Solution:** Verify entry point calls `update_legacy_dict` before data operations:
```python
from ptycho.config.config import update_legacy_dict
import ptycho.params as p

# Always call before data loading
update_legacy_dict(p.cfg, config)
```

### Issue: MLflow network calls in CI

**Solution:** Set environment variable:
```bash
export MLFLOW_TRACKING_URI=memory
```

Or disable MLflow in test code:
```python
import os
os.environ['MLFLOW_TRACKING_URI'] = 'memory'
```

---

## References

**Planning Documents:**
- Phase D4 checklist: `plans/active/INTEGRATE-PYTORCH-001/phase_d4_regression.md`
- Alignment narrative: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T111700Z/phase_d4_alignment.md`
- TEST-PYTORCH-001 charter: `plans/pytorch_integration_test_plan.md`

**Test Suite Documentation:**
- Testing guide: `docs/TESTING_GUIDE.md`
- Test index: `docs/development/TEST_SUITE_INDEX.md`
- TensorFlow baseline test: `tests/test_integration_workflow.py`

**Evidence Artifacts:**
- Config parity: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T050930Z/`
- Data pipeline: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T070200Z/`
- Persistence: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T110500Z/`

**Specification Contracts:**
- Configuration: `specs/ptychodus_api_spec.md` §5
- Data format: `specs/data_contracts.md` §1-2
- Persistence: `specs/ptychodus_api_spec.md` §4.6

**Knowledge Base:**
- CONFIG-001: `docs/findings.md:9` (params.cfg initialization)
- DATA-001: `docs/findings.md:11` (NPZ complex64 requirement)

---

**Document Status:** Ready for Phase D4.B execution
**Next Use:** Reference during D4.B red phase test authoring and D4.C green phase validation
