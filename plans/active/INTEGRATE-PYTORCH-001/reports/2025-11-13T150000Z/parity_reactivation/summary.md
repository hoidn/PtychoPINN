### Turn Summary
Confirmed commit 3efa2dc3 landed GPU-first `PyTorchExecutionConfig` defaults plus GREEN `green/pytest_execution_config_defaults.log`, and captured the evidence in `analysis/artifact_inventory.txt`.
Updated `plans/ptychodus_pytorch_integration_plan.md`, docs/fix_plan.md, input.md, and this hub plan so the next increment adds dispatcher-level regression tests proving backend_selector inherits the GPU baseline and emits POLICY-001 warnings on CPU-only hosts.
Next: implement the backend-selector GPU/CPU tests in `tests/torch/test_execution_config_defaults.py`, rerun `pytest tests/torch/test_execution_config_defaults.py -vv`, and refresh hub summaries/inventory with the new log (blockers → `$HUB/red/`).
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/{green/pytest_execution_config_defaults.log,analysis/artifact_inventory.txt}, docs/fix_plan.md, plans/ptychodus_pytorch_integration_plan.md

### Turn Summary
Recorded DEVICE-MISMATCH-001 completion (commit 85478a67) and validated CUDA inference evidence, then reviewed `docs/workflows/pytorch.md` (§12) + `cli/pytorch_cli_smoke_training/train_clean.log` to confirm the CLIs still default to CPU when `--torch-accelerator` is omitted.
Updated the initiative plan, fix plan, and hub instructions so the next increment forces CUDA-by-default via argparse defaults + `resolve_accelerator` auto-detection, with refreshed backend-selector/CLI-shared tests and a new pytest log.
Next: implement the CUDA default change, run the targeted selectors, rerun the training/inference CLIs without explicit accelerator flags to capture GPU logs, and update the hub summaries/artifact inventory.
Artifacts: plans/ptychodus_pytorch_integration_plan.md, docs/fix_plan.md, analysis/artifact_inventory.txt, cli/pytorch_cli_smoke_training/train_clean.log

### Turn Summary
Validated config defaults backfill (commit dd0a5b0e): all spec-mandated fields now flow through PyTorch dataclasses without ad-hoc overrides, proven by 47 PASSED parity tests.
Reran PyTorch inference CLI against the trained bundle; succeeded on CPU accelerator with amplitude/phase PNGs generated, but CUDA path blocked by device mismatch (model weights on CPU vs inputs on GPU).
Next: either fix model.to(device) in load_torch_bundle or _run_inference_and_reconstruct, then retest CUDA inference before closing Phase R.
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/ (green/pytest_config_bridge.log, cli/pytorch_cli_smoke_training/{inference_cpu.log,inference_outputs/*.png}, red/blocked_*_device_mismatch.md, analysis/artifact_inventory.txt)

## Phase R Config Defaults Backfill — Summary

**Date:** 2025-11-12T19:31:00Z  
**Focus:** INTEGRATE-PYTORCH-PARITY-001 (Phase R)  
**Objective:** Backfill spec-mandated config defaults in PyTorch dataclasses and validate end-to-end config parity

### Code Changes (Commit dd0a5b0e)

Extended `ptycho_torch/config_params.py` dataclasses with spec-mandated defaults:
- DataConfig: `subsample_seed`
- ModelConfig: `pad_object`, `gaussian_smoothing_sigma`  
- TrainingConfig: `train_data_file`, `test_data_file`, `output_dir`, `n_groups`

Updated `ptycho_torch/config_bridge.py` to prefer dataclass defaults over hardcoded fallbacks, and extended `ptycho_torch/config_factory.py` payload constructors to populate new fields.

Added 2 new parity tests validating `subsample_seed` propagation with correct override precedence.

### Test Evidence

**Config Bridge Parity (GREEN):**
- Selector: `pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity -vv`
- Result: 47 PASSED, 18 warnings, 3.65s
- Log: `green/pytest_config_bridge.log`

**PyTorch CLI Smoke (PARTIAL):**
- Training: Previously completed (bundle at `cli/pytorch_cli_smoke_training/train_outputs/wts.h5.zip`)
- Inference (CPU): SUCCESS — generated `reconstructed_amplitude.png` (24K) and `reconstructed_phase.png` (18K)
- Inference (CUDA): BLOCKED — device mismatch (`torch.cuda.FloatTensor` vs `torch.FloatTensor`)
- Log: `cli/pytorch_cli_smoke_training/inference_cpu.log`
- Blocker: `red/blocked_20251112T193051Z_device_mismatch.md`

### Exit Criteria Assessment

Phase R objectives:
- [x] Spec-mandated defaults exist in PyTorch dataclasses
- [x] Config bridge propagates fields without ad-hoc overrides
- [x] Config factory supports new fields
- [x] Parity tests validate complete flow (47 PASSED)
- [x] PyTorch CLI smoke completes inference (CPU workaround applied)
- [x] Hub evidence documented

**Status:** Phase R config defaults parity COMPLETE. Device mismatch is a separate inference execution issue (not a config parity blocker).

### Known Issues

**CUDA Device Mismatch (documented in findings TBD):**
- Model loaded via `load_torch_bundle` remains on CPU even when `--torch-accelerator cuda` specified
- Inference code moves input tensors to CUDA but model weights stay on CPU
- Workaround: Use CPU accelerator for smoke tests
- Fix: Add `model.to(device)` in `load_torch_bundle` or `_run_inference_and_reconstruct`

### Next Actions

1. File finding DEVICE-MISMATCH-001 in `docs/findings.md`
2. Fix device placement in either:
   - `ptycho_torch/model_manager.py::load_torch_bundle` (accept device parameter), OR
   - `ptycho_torch/workflows/components.py::_run_inference_and_reconstruct` (move model to execution_config.accelerator)
3. Rerun CUDA inference smoke test
4. Update ledger to mark Phase R complete and pivot to next Do Now

### References
- Commit: dd0a5b0e
- Spec: docs/specs/spec-ptycho-config-bridge.md
- Plan: plans/ptychodus_pytorch_integration_plan.md (Phase R section)
- Finding: CONFIG-001 (config bridge flow)
