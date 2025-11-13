### Turn Summary
Captured the supervised-loss fix plus the new manual-optimization (`accumulate_grad_batches`) and unlabeled-data blockers, then rewrote the plan/fix-plan/input.md so the next loop adds guardrails and reruns the PyTorch CLI smoke in PINN mode.
Filed findings EXEC-ACCUM-001 / DATA-SUP-001, updated the Do Now with the gradient-accumulation guard, supervised data-contract detection, targeted pytest selector, and refreshed CLI commands/log expectations.
Next: implement the guard in `_train_with_lightning`, add the regression test, fail fast when supervised data lacks labels, rerun the selector, and redo the CLI smoke / hub summaries.
Artifacts: plans/ptychodus_pytorch_integration_plan.md, docs/fix_plan.md (2025-11-13T185800Z Do Now), plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/{analysis/artifact_inventory.txt,red/blocked_*.md}

### Turn Summary
Fixed supervised model_type→MAE loss mapping to prevent loss_name AttributeError in PyTorch Lightning module; added regression test coverage and confirmed fix with GREEN pytest evidence (3 PASSED).
Supervised CLI smoke revealed two downstream blockers: manual optimization incompatibility with accumulate_grad_batches flag, and supervised mode requiring ground-truth labels (label_amp/label_phase) absent from experimental fly001 dataset.
Next: either switch smoke test to unsupervised PINN mode for execution-config validation, or defer supervised CLI smoke until labeled synthetic data is available, then refresh hub documentation.
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/ (green/pytest_backend_selector_cli.log ✅, red/blocked_*.md, analysis/artifact_inventory.txt)

### Turn Summary
Documented that the training CLI execution-config flags (commit 04a016ad) are in place but the PyTorch training smoke fails with `AttributeError: 'PtychoPINN_Lightning' object has no attribute 'loss_name'`, and logged the blocker plus artifact inventory updates in the active hub.
Rewrote plans/ptychodus_pytorch_integration_plan.md, docs/fix_plan.md, and input.md so the next Do Now maps supervised configs to a supported PyTorch loss, adds regression tests, and reruns the filtered pytest selector + CLI smoke before refreshing hub evidence.
Next: implement the supervised-loss mapping + tests, rerun `pytest tests/scripts/test_training_backend_selector.py::TestTrainingCliBackendDispatch::test_pytorch_execution_config_flags tests/scripts/test_inference_backend_selector.py::TestInferenceCliBackendDispatch::test_pytorch_execution_config_flags -vv`, then redo the PyTorch training/inference CLI smoke and update the hub summaries/inventory.
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/red/blocked_20251113T183500Z_loss_name.md, plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/analysis/artifact_inventory.txt

### Turn Summary
Exposed PyTorch execution config flags (--torch-accelerator, --torch-num-workers, --torch-learning-rate, --torch-scheduler, --torch-logger, etc.) in scripts/training/train.py, wired them through backend_selector→run_cdi_example_torch with torch_execution_config parameter.
Added test_pytorch_execution_config_flags to tests/scripts/test_training_backend_selector.py verifying flag propagation, reran backend selector tests (2 PASSED), and executed PyTorch training CLI smoke test.
Training smoke confirmed execution config successfully parsed and logged (accelerator=cpu, num_workers=0, learning_rate=0.001, logger_backend=csv), but failed downstream in PyTorch Lightning model initialization (pre-existing 'loss_name' attribute error, not related to execution config flags).
Artifacts: commit 04a016ad, green/pytest_backend_selector_cli.log, cli/pytorch_cli_smoke_training/SMOKE_TEST_RESULT.txt

### Turn Summary
Confirmed commit b218696a delivered the inference CLI execution-config flags plus GREEN pytest + CLI smoke evidence (scripts/inference/inference.py, tests/scripts/test_inference_backend_selector.py, hub `green/pytest_backend_selector_cli.log`).
Pivoted plans/ptychodus_pytorch_integration_plan.md and docs/fix_plan.md to a new Do Now that surfaces the same execution-config knobs on scripts/training/train.py, adds backend-selector tests, and reruns the PyTorch CLI smoke with training flags; refreshed input.md so Ralph can execute it.
Next: implement the training CLI execution-config flags, add the new pytest case, rerun the selectors + CLI smoke, and update the hub summaries/inventory.
Artifacts: plans/ptychodus_pytorch_integration_plan.md, docs/fix_plan.md, input.md, plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/green/pytest_backend_selector_cli.log

### Turn Summary
Exposed PyTorch execution config flags (--torch-accelerator, --torch-num-workers, --torch-inference-batch-size) in scripts/inference/inference.py and wired them through build_execution_config_from_args for validated configuration.
Extended test_inference_backend_selector.py with test_pytorch_execution_config_flags to verify flag propagation, reran the backend selector tests (2 PASSED), and executed the PyTorch CLI smoke with explicit execution config knobs (accelerator=cpu, batch_size=2, num_workers=0).
Inference succeeded with logged execution config values and generated amplitude/phase PNGs (15K each), confirming the new flags are functional and the PyTorch inference path is fully operational with runtime config control.
Artifacts: green/pytest_backend_selector_cli.log, cli/pytorch_cli_smoke/{train.log,inference.log,inference_outputs/{reconstructed_amplitude.png,reconstructed_phase.png}}, analysis/artifact_inventory.txt

### Turn Summary
Verified commit 12fa29dd delivered the PyTorch inference branch plus green pytest/CLI smoke evidence and documented the results in the active hub.
Updated plans/ptychodus_pytorch_integration_plan.md and docs/fix_plan.md with a new Do Now that exposes `--torch-*` execution flags in scripts/inference/inference.py via `build_execution_config_from_args`, plus refreshed pytest/CLI expectations.
Rewrote input.md so Ralph adds the new flags, extends backend-selector tests, reruns the PyTorch CLI smoke with explicit execution-config knobs, and updates the hub inventory/summaries (blockers → `$HUB/red/`).
Artifacts: docs/fix_plan.md, plans/ptychodus_pytorch_integration_plan.md, input.md, plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/analysis/artifact_inventory.txt

### Turn Summary
Implemented PyTorch inference execution branch in scripts/inference/inference.py so backend='pytorch' now calls ptycho_torch.inference._run_inference_and_reconstruct instead of TensorFlow's perform_inference().
Added test_pytorch_inference_execution_path to verify the new code path, reran all backend selector tests (3 PASSED), and executed full PyTorch CLI smoke (training + inference) with the minimal fixture.
Inference completed successfully and generated amplitude/phase PNGs (15K/19K) under inference_outputs/, confirming the PyTorch workflow is end-to-end functional from the canonical entry points.
Artifacts: green/pytest_backend_selector_cli.log, cli/pytorch_cli_smoke/{train.log,inference.log,inference_outputs/{reconstructed_amplitude.png,reconstructed_phase.png}}

### Turn Summary
Confirmed Ralph's inference CLI flag + smoke landed (commit c983bdc8) and captured the remaining PyTorch inference failure (`'probe'` KeyError), then reoriented the plan toward a backend-aware inference branch.
Updated plans/ptychodus_pytorch_integration_plan.md, docs/fix_plan.md, and input.md with the new Do Now covering the PyTorch branch, backend-selector test updates, and rerunning the minimal CLI smoke with refreshed hub evidence.
Next: implement the PyTorch inference execution path, extend backend selector tests, rerun the training/inference commands, and update the hub summaries/artifact inventory.
Artifacts: docs/fix_plan.md, plans/ptychodus_pytorch_integration_plan.md, input.md

### Turn Summary
Documented that backend-selector wiring (commit a53f897b + green backend-dispatch logs) is complete, marked the Phase R checklist done in the plan, and added the inference backend flag + PyTorch CLI smoke checklist.
Updated docs/fix_plan.md and input.md so Ralph now implements the new `--backend` inference option, reruns the backend selectors, and executes the minimal-dataset training/inference commands under `$HUB/cli/pytorch_cli_smoke`.
Next: implement the CLI flag/tests, run the PyTorch smoke commands, and publish the new pytest/CLI logs plus artifact inventory updates.
Artifacts: docs/fix_plan.md, input.md, plans/ptychodus_pytorch_integration_plan.md, plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/

### Turn Summary
Wired training and inference CLIs through backend_selector so --backend pytorch becomes reachable from canonical entry points (scripts/training/train.py, scripts/inference/inference.py).
Guarded TensorFlow-only persistence helpers (model_manager.save, save_outputs) with if config.backend == 'tensorflow' to avoid double-saving when PyTorch workflows emit their own bundles.
Added 5 unit tests verifying dispatch correctness and TensorFlow backward compatibility; all tests GREEN (2 training + 3 inference passed).
Artifacts: commit a53f897b, green/pytest_training_backend_dispatch.log, green/pytest_inference_backend_dispatch.log, analysis/artifact_inventory.txt

### Turn Summary
Verified the Phase R quick wins (config bridge invocation, persistence shim, and parity pytest) via the hub inventory + GREEN log and documented the completion inside the plan.
Rewrote docs/fix_plan.md and the plan/input brief so Ralph now targets wiring `scripts/training/train.py` / `scripts/inference/inference.py` through the backend selector while guarding TensorFlow-only persistence paths.
Next: implement the CLI routing plus new backend-dispatch tests, run the targeted pytest node, and refresh the hub artifact inventory + summaries.
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/

### Turn Summary
Reactivated the PyTorch parity initiative with a Phase R checklist (config bridge wiring, spec default backfill, persistence shim, regression guard) and recorded the new reports hub so evidence lands under a single path.
Resteered docs/fix_plan.md, plans/ptychodus_pytorch_integration_plan.md, input.md, and galph_memory.md so Ralph’s next loop targets update_legacy_dict wiring plus the config-bridge pytest gate while logging to the new hub.
Next: Ralph implements the Phase R Do Now (bridge wiring + persistence shim + pytest selector) and records artifacts under plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/.
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/
