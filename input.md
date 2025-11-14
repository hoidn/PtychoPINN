Brief:
Guard the repo root, export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md and HUB=$PWD/plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity, and leave the hub tree clean before recording fresh evidence.
Re-run `pytest tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump -vv`, teeing output to $HUB/green/pytest_patch_stats_rerun.log so we have a post-dc5415ba proof that the CLI now emits torch_patch_stats.json/torch_patch_grid.png.
Execute the 10-epoch training command from plan/plan.md with OUT=outputs/torch_forward_parity_baseline, `--log-patch-stats --patch-stats-limit 2`, and tee stdout/stderr to $HUB/cli/train_patch_stats_rerun.log; then run the matching inference command tee’d to $HUB/cli/inference_patch_stats_rerun.log with `--debug-dump "$HUB"/analysis/forward_parity_debug`.
Copy the new torch_patch_stats.json and torch_patch_grid.png from the run outputs into $HUB/analysis/, refresh $HUB/analysis/artifact_inventory.txt, $HUB/summary.md, and the initiative summary, and drop $HUB/red/blocked_<timestamp>.md citing POLICY-001/CONFIG-001 if CUDA or memory prevents completion.

Summary: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md
Plan: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md
Selector: tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump
