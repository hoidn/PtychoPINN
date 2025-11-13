# Phase R Backend Selector Integration — Hub Summary

# Phase R Backend Selector Integration — Hub Summary

### Turn Summary
Recorded that training CLI execution-config work (commit 04a016ad) shipped while the PyTorch training smoke now fails with `AttributeError: 'PtychoPINN_Lightning' object has no attribute 'loss_name'`, and stored the blocker plus artifact inventory updates (`analysis/artifact_inventory.txt`, `red/blocked_20251113T183500Z_loss_name.md`).
Plans/docs/input were updated so Ralph’s next loop maps supervised configs to a supported PyTorch loss function, adds regression tests for the factory/selector, reruns the targeted pytest selector, and repeats the PyTorch CLI smoke before refreshing the hub summaries and inventory.
Next: implement the supervised-loss mapping + tests, rerun the specified pytest command, execute the PyTorch training/inference CLI smoke with the documented flags, and update the hub evidence (logs/PNGs/inventory) accordingly.
Artifacts: analysis/artifact_inventory.txt, red/blocked_20251113T183500Z_loss_name.md

**Date:** 2025-11-13  
**Focus:** INTEGRATE-PYTORCH-PARITY-001 — PyTorch backend API parity reactivation  
**Phase:** CLI routing via backend selector  
**Commit:** a53f897b

## Deliverables

Successfully integrated backend_selector into production training and inference CLIs,
enabling PyTorch backend invocation from canonical entry points while maintaining
TensorFlow backward compatibility.

### 1. Training CLI (scripts/training/train.py)

- **Routing:** Replaced direct `run_cdi_example` import with `backend_selector.run_cdi_example_with_backend`
- **Persistence Guard:** Added `if config.backend == 'tensorflow':` check before calling:
  * `model_manager.save(str(config.output_dir))`
  * `save_outputs(recon_amp, recon_phase, results, str(config.output_dir))`
- **PyTorch Path:** Skips TensorFlow-only helpers; logs bundle_path from results dict
- **CONFIG-001:** Preserves `update_legacy_dict(params.cfg, config)` call at line 132

### 2. Inference CLI (scripts/inference/inference.py)

- **Routing:** Replaced direct `load_inference_bundle` import with `backend_selector.load_inference_bundle_with_backend`
- **Backend Dispatch:** Passes `InferenceConfig` to selector for automatic backend routing
- **CONFIG-001:** Documented that params.cfg restoration happens inside backend-specific loaders

### 3. Test Coverage

#### Training Tests (tests/scripts/test_training_backend_selector.py)

- `test_pytorch_backend_dispatch`:
  * Verifies CLI calls `run_cdi_example_with_backend` with `backend='pytorch'`
  * Asserts `model_manager.save()` and `save_outputs()` NOT called for PyTorch
  * Confirms `results['backend'] == 'pytorch'` and bundle_path logged

- `test_tensorflow_backend_persistence`:
  * Verifies TensorFlow backend still calls legacy persistence helpers
  * Confirms backward compatibility for existing TensorFlow workflows

**Result:** 2 passed in 3.70s ✅

#### Inference Tests (tests/scripts/test_inference_backend_selector.py)

- `test_pytorch_backend_dispatch`:
  * Verifies CLI calls `load_inference_bundle_with_backend` with `backend='pytorch'`
  * Confirms PyTorch model returned and params_dict restored

- `test_tensorflow_backend_dispatch`:
  * Verifies TensorFlow backend routing continues to work

- `test_backend_selector_preserves_config_001_compliance`:
  * Documents params.cfg restoration inside backend-specific loaders

**Result:** 3 passed in 3.66s ✅

## Evidence

- **Commit:** a53f897b
- **Green Logs:**
  * `green/pytest_training_backend_dispatch.log` (2 passed)
  * `green/pytest_inference_backend_dispatch.log` (3 passed)
- **Artifact Inventory:** `analysis/artifact_inventory.txt`

## Policy Compliance

- ✅ POLICY-001: PyTorch>=2.2 enforced (version 2.8.0+cu128 confirmed in test logs)
- ✅ CONFIG-001: update_legacy_dict called before backend dispatch in training CLI
- ✅ CONFIG-001: params.cfg restoration handled inside backend-specific loaders for inference
- ✅ TYPE-PATH-001: Path objects handled correctly in backend_selector

## Next Steps

Phase R quick wins are complete. The production CLIs now support `--backend pytorch`
via backend_selector routing. Next increment should focus on:

1. End-to-end CLI execution tests with real PyTorch backend invocation
2. Verify PyTorch workflows are fully CONFIG-001 compliant
3. Validate persistence parity between TensorFlow and PyTorch bundles
