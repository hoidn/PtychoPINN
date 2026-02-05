# Non-Integration Test Failures Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Get the non-integration pytest suite green by fixing broken implementations and updating obsolete test expectations uncovered in the 2026‑02‑05 run.

**Architecture:** Address core regressions first (NumPy NPZ header reading, TF reassembly padding, config override mapping), then repair CLI/config plumbing for patch stats, and finally align tests with updated APIs (ptychodus, adapters, programmatic Phase‑D). Each task is isolated, TDD‑style, with targeted selectors and small commits.

**Tech Stack:** Python, NumPy, TensorFlow, PyTorch, pytest.

---

### Task 1: Replace Removed NumPy Header API in NPZ Readers

**Files:**
- Modify: `ptycho_torch/dataloader.py`
- Modify: `ptycho_torch/train.py`
- (Optional) Create: `ptycho_torch/npz_utils.py`
- Test: `tests/torch/test_dataloader.py`
- Test: `tests/torch/test_train_probe_size.py`

**Step 1: Run failing tests to confirm current failure**

Run:
```bash
pytest tests/torch/test_dataloader.py::TestDataloaderCanonicalKeySupport::test_loads_canonical_diffraction -v
pytest tests/torch/test_train_probe_size.py::TestNPZProbeSizeExtraction::test_infer_probe_size_128 -v
```
Expected: FAIL with `AttributeError: numpy.lib.format has no attribute _read_array_header`.

**Step 2: Implement a public header reader helper**

If creating helper:
```python
# ptycho_torch/npz_utils.py
import numpy as np

def read_npy_shape(npy_file):
    version = np.lib.format.read_magic(npy_file)
    if version == (1, 0):
        shape, _, _ = np.lib.format.read_array_header_1_0(npy_file)
    elif version == (2, 0):
        shape, _, _ = np.lib.format.read_array_header_2_0(npy_file)
    elif version == (3, 0):
        shape, _, _ = np.lib.format.read_array_header_3_0(npy_file)
    else:
        raise ValueError(f"Unsupported .npy version: {version}")
    return shape
```
Then replace `_read_array_header` calls in `ptycho_torch/dataloader.py` and `ptycho_torch/train.py` with this helper.

**Step 3: Re-run tests**

Run:
```bash
pytest tests/torch/test_dataloader.py::TestDataloaderCanonicalKeySupport::test_loads_canonical_diffraction -v
pytest tests/torch/test_dataloader.py::TestDataloaderCanonicalKeySupport::test_backward_compat_legacy_diff3d -v
pytest tests/torch/test_dataloader.py::TestDataloaderFormatAutoTranspose::test_npz_headers_also_transposes_shape -v
pytest tests/torch/test_train_probe_size.py::TestNPZProbeSizeExtraction::test_infer_probe_size_128 -v
```
Expected: PASS.

**Step 4: Commit**
```bash
git add ptycho_torch/dataloader.py ptycho_torch/train.py ptycho_torch/npz_utils.py

git commit -m "fix(torch): replace removed npy header API"
```

---

### Task 2: Fix ReassemblePatchesLayer padded_size=None in batched path

**Files:**
- Modify: `ptycho/tf_helper.py`
- Test: `tests/study/test_dose_overlap_comparison.py::test_pinn_reconstruction_reassembles_full_train_split`
- Test: `tests/test_tf_helper.py`

**Step 1: Run failing test**

Run:
```bash
pytest tests/study/test_dose_overlap_comparison.py::test_pinn_reconstruction_reassembles_full_train_split -v
```
Expected: FAIL with `ValueError: Can't convert Python sequence with mixed types` from `tf.zeros((1, padded_size, ...))`.

**Step 2: Implement default padded_size handling**

In `mk_reassemble_position_batched_real`:
```python
if 'padded_size' not in merged_kwargs or merged_kwargs['padded_size'] is None:
    padded_size = get_padded_size()
else:
    padded_size = merged_kwargs.pop('padded_size')
```

**Step 3: Re-run tests**

Run:
```bash
pytest tests/study/test_dose_overlap_comparison.py::test_pinn_reconstruction_reassembles_full_train_split -v
```
Expected: PASS.

**Step 4: Commit**
```bash
git add ptycho/tf_helper.py

git commit -m "fix(tf): default padded_size in batched reassembly"
```

---

### Task 3: Update TF helper tests to use correct offsets shape

**Files:**
- Modify: `tests/test_tf_helper.py`

**Step 1: Run failing test**

Run:
```bash
pytest tests/test_tf_helper.py::TestReassemblePosition::test_basic_functionality -v
```
Expected: FAIL with `InvalidArgumentError: required broadcastable shapes`.

**Step 2: Update offsets shape in tests**

Change all `global_offsets` reshapes from `(B, 1, 2, 1)` to `(B, 1, 1, 2)` to match `reassemble_position()` contract.

**Step 3: Re-run tests**

Run:
```bash
pytest tests/test_tf_helper.py::TestReassemblePosition::test_basic_functionality -v
pytest tests/test_tf_helper.py::TestReassemblePosition::test_perfect_overlap_averages_to_identity -v
pytest tests/test_tf_helper.py::TestReassemblePosition::test_identical_patches_single_vs_double -v
pytest tests/test_tf_helper.py::TestReassemblePosition::test_different_patch_values_blend -v
```
Expected: PASS.

**Step 4: Commit**
```bash
git add tests/test_tf_helper.py

git commit -m "test(tf): fix offsets shape in reassemble_position tests"
```

---

### Task 4: Fix `train_cdi_model` test scaffold to include config.model

**Files:**
- Modify: `tests/study/test_dose_overlap_training.py`

**Step 1: Run failing test**

Run:
```bash
pytest tests/study/test_dose_overlap_training.py::test_train_cdi_model_normalizes_history -v
```
Expected: FAIL with `AttributeError: 'SimpleNamespace' object has no attribute model`.

**Step 2: Update test config to include model/architecture**

Use a minimal config object with `.model.architecture` so `resolve_generator()` works, e.g.:
```python
config = SimpleNamespace(model=SimpleNamespace(architecture="cnn"))
```

**Step 3: Re-run test**

Run:
```bash
pytest tests/study/test_dose_overlap_training.py::test_train_cdi_model_normalizes_history -v
```
Expected: PASS.

**Step 4: Commit**
```bash
git add tests/study/test_dose_overlap_training.py

git commit -m "test(tf): fix train_cdi_model scaffold config"
```

---

### Task 5: Update Phase‑G orchestrator tests for programmatic Phase‑D

**Files:**
- Modify: `tests/study/test_phase_g_dense_orchestrator.py`

**Step 1: Run failing test**

Run:
```bash
pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -v
```
Expected: FAIL due to missing `studies.fly64_dose_overlap.overlap` string.

**Step 2: Update expected output**

Replace the CLI expectation with programmatic marker, e.g. assert presence of:
- `__PHASE_D_PROGRAMMATIC__`
- `Programmatic Phase D overlap generation`

**Step 3: Re-run Phase‑G tests**

Run:
```bash
pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -v
pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_hooks -v
pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_invokes_reporting_helper -v
pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -v
pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -v
```
Expected: PASS.

**Step 4: Commit**
```bash
git add tests/study/test_phase_g_dense_orchestrator.py

git commit -m "test(study): align Phase G orchestrator expectations with programmatic Phase D"
```

---

### Task 6: Update ptychodus interop test to new API

**Files:**
- Modify: `tests/io/test_ptychodus_interop_h5.py`

**Step 1: Run failing test**

Run:
```bash
pytest tests/io/test_ptychodus_interop_h5.py::test_interop_h5_reader -v
```
Expected: FAIL with `Product has no attribute positions`.

**Step 2: Update test to use `probe_positions`**

Change:
```python
len(product.positions)
```
To:
```python
len(product.probe_positions)
```
Optionally support both to avoid future churn.

**Step 3: Re-run test**

Run:
```bash
pytest tests/io/test_ptychodus_interop_h5.py::test_interop_h5_reader -v
```
Expected: PASS.

**Step 4: Commit**
```bash
git add tests/io/test_ptychodus_interop_h5.py

git commit -m "test(io): update ptychodus interop for probe_positions"
```

---

### Task 7: Update inference bundle test for adapter wrapper

**Files:**
- Modify: `tests/test_workflow_components.py`

**Step 1: Run failing test**

Run:
```bash
pytest tests/test_workflow_components.py::TestLoadInferenceBundle::test_load_valid_model_directory -v
```
Expected: FAIL because model is `DiffractionToObjectAdapter`.

**Step 2: Accept adapter in test**

Adjust assertions:
```python
assert isinstance(model, DiffractionToObjectAdapter)
assert model._model is mock_model
```

**Step 3: Re-run test**

Run:
```bash
pytest tests/test_workflow_components.py::TestLoadInferenceBundle::test_load_valid_model_directory -v
```
Expected: PASS.

**Step 4: Commit**
```bash
git add tests/test_workflow_components.py

git commit -m "test(tf): accept adapter wrapper in inference bundle"
```

---

### Task 8: Map max_epochs → epochs and neighbor_count → K in factory

**Files:**
- Modify: `ptycho_torch/config_factory.py`
- Test: `tests/torch/test_config_factory.py::TestConfigBridgeTranslation::*`

**Step 1: Run failing tests**

Run:
```bash
pytest tests/torch/test_config_factory.py::TestConfigBridgeTranslation::test_epochs_to_nepochs_conversion -v
pytest tests/torch/test_config_factory.py::TestConfigBridgeTranslation::test_k_to_neighbor_count_conversion -v
```
Expected: FAIL with wrong values.

**Step 2: Implement alias mapping**

Before `update_existing_config()` calls, add:
```python
if 'max_epochs' in overrides and 'epochs' not in overrides:
    overrides['epochs'] = overrides['max_epochs']
if 'neighbor_count' in overrides and 'K' not in overrides:
    overrides['K'] = overrides['neighbor_count']
```

**Step 3: Re-run tests**

Run:
```bash
pytest tests/torch/test_config_factory.py::TestConfigBridgeTranslation::test_epochs_to_nepochs_conversion -v
pytest tests/torch/test_config_factory.py::TestConfigBridgeTranslation::test_k_to_neighbor_count_conversion -v
```
Expected: PASS.

**Step 4: Commit**
```bash
git add ptycho_torch/config_factory.py

git commit -m "fix(torch): map max_epochs and neighbor_count overrides"
```

---

### Task 9: Add patch stats flags + training payload inference config

**Files:**
- Modify: `ptycho_torch/config_params.py` (add fields to InferenceConfig)
- Modify: `ptycho_torch/config_factory.py` (add pt_inference_config to TrainingPayload + build it)
- Modify: `ptycho_torch/train.py` (argparse flags + overrides wiring)
- Test: `tests/torch/test_patch_stats_cli.py`
- Test: `tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump`

**Step 1: Run failing tests**

Run:
```bash
pytest tests/torch/test_patch_stats_cli.py::TestPatchStatsCLI::test_patch_stats_flags_accepted -v
pytest tests/torch/test_patch_stats_cli.py::TestPatchStatsCLI::test_factory_inference_config_defaults -v
pytest tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump -v
```
Expected: FAIL with missing CLI args and missing `pt_inference_config`.

**Step 2: Add fields to PT InferenceConfig**

In `ptycho_torch/config_params.py`:
```python
log_patch_stats: bool = False
patch_stats_limit: Optional[int] = None
```

**Step 3: Extend TrainingPayload + create_training_payload**

- Add `pt_inference_config: PTInferenceConfig` to `TrainingPayload`.
- In `create_training_payload()`, instantiate `PTInferenceConfig()` and update it with overrides (log_patch_stats, patch_stats_limit). Return it in payload.

**Step 4: Wire CLI flags in `ptycho_torch/train.py`**

Add argparse flags:
```
--log-patch-stats
--patch-stats-limit
```
Then inject into overrides dict passed to `create_training_payload()`.

**Step 5: Re-run tests**

Run:
```bash
pytest tests/torch/test_patch_stats_cli.py::TestPatchStatsCLI::test_patch_stats_flags_accepted -v
pytest tests/torch/test_patch_stats_cli.py::TestPatchStatsCLI::test_factory_inference_config_defaults -v
pytest tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump -v
```
Expected: PASS.

**Step 6: Commit**
```bash
git add ptycho_torch/config_params.py ptycho_torch/config_factory.py ptycho_torch/train.py

git commit -m "feat(torch): add patch stats flags and training payload inference config"
```

---

### Task 10: Initialize params.cfg before grid-lines stitching

**Files:**
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Test: `tests/torch/test_grid_lines_torch_runner.py::TestRunGridLinesTorchScaffold::test_runner_creates_run_directory_structure`

**Step 1: Run failing test**

Run:
```bash
pytest tests/torch/test_grid_lines_torch_runner.py::TestRunGridLinesTorchScaffold::test_runner_creates_run_directory_structure -v
```
Expected: FAIL with `KeyError: 'data_source'` in `params.validate()`.

**Step 2: Initialize legacy params before `p.set()`**

In `_configure_stitching_params()`, add:
```python
from ptycho.config.config import TrainingConfig, ModelConfig, update_legacy_dict
update_legacy_dict(p.cfg, TrainingConfig(model=ModelConfig(N=cfg.N, gridsize=cfg.gridsize)))
```
(or set `p.cfg.setdefault('data_source', 'generic')` before `p.set`).

**Step 3: Re-run test**

Run:
```bash
pytest tests/torch/test_grid_lines_torch_runner.py::TestRunGridLinesTorchScaffold::test_runner_creates_run_directory_structure -v
```
Expected: PASS.

**Step 4: Commit**
```bash
git add scripts/studies/grid_lines_torch_runner.py

git commit -m "fix(torch): init legacy params before stitching"
```

---

### Task 11: Update inference helper test mock to return torch tensor

**Files:**
- Modify: `tests/torch/test_cli_inference_torch.py`

**Step 1: Run failing test**

Run:
```bash
pytest tests/torch/test_cli_inference_torch.py::TestInferenceCLI::test_accelerator_flag_roundtrip -v
```
Expected: FAIL with `TypeError: only 0-dimensional arrays can be converted to Python scalars`.

**Step 2: Update mock forward_predict**

Return a torch tensor instead of MagicMock:
```python
mock_model.forward_predict = MagicMock(return_value=torch.rand(1, 1, 64, 64, dtype=torch.complex64))
```

**Step 3: Re-run test**

Run:
```bash
pytest tests/torch/test_cli_inference_torch.py::TestInferenceCLI::test_accelerator_flag_roundtrip -v
```
Expected: PASS.

**Step 4: Commit**
```bash
git add tests/torch/test_cli_inference_torch.py

git commit -m "test(torch): use tensor mock for inference helper"
```

---

### Task 12: Update workflow component test mocks for execution_config

**Files:**
- Modify: `tests/torch/test_workflows_components.py`

**Step 1: Run failing test**

Run:
```bash
pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsScaffold::test_run_cdi_example_calls_update_legacy_dict -v
```
Expected: FAIL with unexpected `execution_config` kwarg in mock.

**Step 2: Adjust mock signatures**

Update mock helpers to accept `execution_config=None` (and ignore it).

**Step 3: Re-run representative tests**

Run:
```bash
pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsScaffold::test_run_cdi_example_calls_update_legacy_dict -v
pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_dataloader_tensor_dict_structure -v
```
Expected: PASS.

**Step 4: Commit**
```bash
git add tests/torch/test_workflows_components.py

git commit -m "test(torch): update workflow mocks for execution_config"
```

---

### Task 13: Rerun focused subset and then full non-integration suite

**Files:**
- Test: `tests/` (all)

**Step 1: Run targeted regression group**

Run:
```bash
pytest tests/test_tf_helper.py::TestReassemblePosition -v
pytest tests/study/test_dose_overlap_comparison.py::test_pinn_reconstruction_reassembles_full_train_split -v
pytest tests/torch/test_dataloader.py -v
pytest tests/torch/test_train_probe_size.py -v
pytest tests/torch/test_patch_stats_cli.py -v
```
Expected: PASS.

**Step 2: Run full non-integration suite**

Run:
```bash
pytest tests/ -q -m "not integration" | tee .artifacts/test_runs/pytest_not_integration_final.log
```
Expected: PASS (or remaining failures limited to known skipped/expected).

**Step 3: Commit (if any final test-only changes)**

```bash
git add .artifacts/test_runs/pytest_not_integration_final.log

git commit -m "test: verify non-integration suite"
```

---

Plan complete and saved to `docs/plans/2026-02-05-non-integration-test-failures.md`. Two execution options:

1. **Subagent-Driven (this session)** — I dispatch a fresh subagent per task, review between tasks.
2. **Parallel Session (separate)** — Open a new session and use `superpowers:executing-plans` with checkpoints.

Which approach?
