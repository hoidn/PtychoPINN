Brief:
Guard from the repo root by exporting `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`, setting `HUB=$PWD/plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity`, and ensuring the hub tree is clean before collecting post-dc5415ba evidence (POLICY-001).
Re-run `pytest tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump -vv` and tee it to `$HUB/green/pytest_patch_stats_rerun.log` so we have a fresh selector log proving the TrainingPayload path still emits torch_patch_stats.json/torch_patch_grid.png.
Using the exact commands in `.../plan/plan.md`, rerun the 10-epoch training and inference with `--log-patch-stats --patch-stats-limit 2`, tee stdout/stderr to `$HUB/cli/train_patch_stats_rerun.log` and `$HUB/cli/inference_patch_stats_rerun.log`, and copy the emitted torch_patch_stats.json / torch_patch_grid.png plus the debug dump into `$HUB/analysis/`.
Refresh `$HUB/analysis/artifact_inventory.txt`, `$HUB/summary.md`, and the initiative summary with the new stats (do not leave artifacts only under `outputs/`), and if CUDA/memory issues block any command drop `$HUB/red/blocked_<timestamp>.md` citing POLICY-001/CONFIG-001 before stopping.

Summary: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md
Plan: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md
Selector: tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump
