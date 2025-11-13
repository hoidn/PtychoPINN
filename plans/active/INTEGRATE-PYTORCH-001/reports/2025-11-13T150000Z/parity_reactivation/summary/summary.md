### Turn Summary
Logged the supervised-loss fix plus the new EXEC-ACCUM-001 / DATA-SUP-001 blockers, updated the hub plan + docs/fix_plan/input.md so the next increment adds the gradient-accumulation guard, supervised data check, and PINN-mode CLI smoke rerun.
Documented the manual-optimization limitation and unlabeled-data dependency in docs/findings.md and refreshed the Reports Hub expectations (pytest selector with new guard test, training/inference CLI commands without `--torch-accumulate-grad-batches`).
Next: implement the guard + regression test, fail fast when supervised data lacks labels, rerun the selector, and redo the CLI smoke / hub summaries with updated logs.
Artifacts: plans/ptychodus_pytorch_integration_plan.md, docs/fix_plan.md (2025-11-13T185800Z Do Now), plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/{analysis/artifact_inventory.txt,red/blocked_*.md}

### Turn Summary
Confirmed commit b218696a delivered the inference CLI execution-config flags plus GREEN pytest + CLI smoke evidence (scripts/inference/inference.py, tests/scripts/test_inference_backend_selector.py, hub `green/pytest_backend_selector_cli.log`).
Pivoted plans/ptychodus_pytorch_integration_plan.md and docs/fix_plan.md to a new Do Now that surfaces the same execution-config knobs on scripts/training/train.py, adds backend-selector tests, and reruns the PyTorch CLI smoke with training flags; refreshed input.md so Ralph can execute it.
Next: implement the training CLI execution-config flags, add the new pytest case, rerun the selectors + CLI smoke, and update the hub summaries/inventory.
Artifacts: plans/ptychodus_pytorch_integration_plan.md, docs/fix_plan.md, input.md, plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/green/pytest_backend_selector_cli.log

### Turn Summary
Implemented PyTorch inference execution branch in scripts/inference/inference.py so backend='pytorch' now calls ptycho_torch.inference._run_inference_and_reconstruct instead of TensorFlow's perform_inference().
Added test_pytorch_inference_execution_path to verify the new code path, reran all backend selector tests (3 PASSED), and executed full PyTorch CLI smoke (training + inference) with the minimal fixture.
Inference completed successfully and generated amplitude/phase PNGs (15K/19K) under inference_outputs/, confirming the PyTorch workflow is end-to-end functional from the canonical entry points.
Artifacts: green/pytest_backend_selector_cli.log, cli/pytorch_cli_smoke/{train.log,inference.log,inference_outputs/{reconstructed_amplitude.png,reconstructed_phase.png}}
