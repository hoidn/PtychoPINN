Brief:
Work from repo root (`test "$(pwd -P)" = "/home/ollie/Documents/PtychoPINN"`), set `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity OUT=$PWD/outputs/torch_forward_parity_baseline`, and keep `$HUB` clean.
Add optional patch-stat instrumentation toggles to `ptycho_torch/model.py` + `ptycho_torch/inference.py` plus CLI plumbing (`--log-patch-stats`, `--patch-stats-limit`), writing JSON+PNG dumps (`torch_patch_stats.json`, `torch_patch_grid.png`) under `$HUB/analysis/`.
Create a targeted pytest in `tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump` that enables the new flags and proves the JSON/PNG files appear; run it and log to `$HUB/green/pytest_patch_stats.log`.
Rerun the short Torch baseline with instrumentation: the train command from `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/plan/plan.md` (10 epochs, 256 groups, `--log-patch-stats --patch-stats-limit 2`) and the paired inference command with `--debug-dump "$HUB"/analysis/forward_parity_debug`, capturing logs under `$HUB/cli/`.
Update `$HUB/analysis/artifact_inventory.txt` and `$HUB/summary.md` with the new artifacts, or drop `$HUB/red/blocked_<timestamp>.md` immediately if CUDA/memory constraints prevent the run.

Summary: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md
Plan: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md
Selector: tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump
