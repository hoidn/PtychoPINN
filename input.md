Brief:
Work from repo root, export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md plus HUB=$PWD/plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity, and keep HUB clean while wiring the new patch-stat toggles through ptycho_torch/model.py and ptycho_torch/inference.py so `--log-patch-stats --patch-stats-limit 2` emits torch_patch_stats.json + torch_patch_grid.png in $HUB/analysis/.
After the wiring, add/run pytest tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump so it exercises the CLI flags end-to-end and logs to $HUB/green/pytest_patch_stats.log.
Rerun the short 10-epoch train command and paired inference command from plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/plan/plan.md with `--log-patch-stats --patch-stats-limit 2`, capturing logs in $HUB/cli/{train_patch_stats.log,inference_patch_stats.log} and the debug dump under $HUB/analysis/forward_parity_debug.
Update $HUB/analysis/artifact_inventory.txt and $HUB/summary.md with the new JSON/PNG/log artifacts—or drop $HUB/red/blocked_<timestamp>.md immediately if CUDA/memory prevents the run, citing POLICY-001/CONFIG-001.

Summary: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md
Plan: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md
Selector: tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump
