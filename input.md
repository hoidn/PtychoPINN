Brief:
Set `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`, `HUB=$PWD/plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity`, and `OUT=outputs/torch_forward_parity_baseline`, and make sure the hub tree is clean so the new logs replace the Nov‑14 evidence called out in `analysis/artifact_inventory.txt`.
Re-run `pytest tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump -vv` tee’d to `$HUB/green/pytest_patch_stats_rerun.log` to prove the TrainingPayload-threaded flags still emit patch-stat artifacts (POLICY-001/CONFIG-001 compliance).
Run the documented 10-epoch training command with `--log-patch-stats --patch-stats-limit 2` (same datasets/gridsize/batch) and tee to `$HUB/cli/train_patch_stats_rerun.log`, then immediately execute the matching inference command with `--debug-dump "$HUB"/analysis/forward_parity_debug` tee’d to `$HUB/cli/inference_patch_stats_rerun.log`.
Copy the fresh `torch_patch_stats*.json`, `torch_patch_grid*.png`, and debug dump files from `outputs/torch_forward_parity_baseline/` into `$HUB/analysis/`, refresh `$HUB/analysis/artifact_inventory.txt`, `$HUB/summary.md`, and the initiative summary, and log `$HUB/red/blocked_<timestamp>.md` citing POLICY-001/CONFIG-001 right away if CUDA/memory blocks any command.

Summary: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md
Plan: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md
Selector: tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump
