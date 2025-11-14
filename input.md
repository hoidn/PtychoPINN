Brief:
From the repo root export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` and set `HUB=$PWD/plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity`, confirming the hub tree is clean before capturing new evidence.
Re-run `pytest tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump -vv` and tee it to `$HUB/green/pytest_patch_stats_rerun.log` so we have a post-dc5415ba proof the CLI emits torch_patch_stats.json/torch_patch_grid.png.
Execute the 10-epoch training command from `.../plan/plan.md` (OUT=outputs/torch_forward_parity_baseline) with `--log-patch-stats --patch-stats-limit 2`, teeing to `$HUB/cli/train_patch_stats_rerun.log`, then run the matching inference command with `--debug-dump "$HUB"/analysis/forward_parity_debug` teeing to `$HUB/cli/inference_patch_stats_rerun.log`.
Copy the fresh torch_patch_stats.json/torch_patch_grid.png (and any new debug-dump files) into `$HUB/analysis/`, overwrite `$HUB/analysis/artifact_inventory.txt` plus `$HUB/summary.md` and the initiative summary with the new stats, and drop `$HUB/red/blocked_<timestamp>.md` citing POLICY-001/CONFIG-001 if CUDA or memory errors prevent completion.

Summary: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md
Plan: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md
Selector: tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump
