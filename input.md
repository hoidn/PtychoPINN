Brief:
Export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`, set `HUB=$PWD/plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity`, and confirm the hub tree is clean so the new logs replace the 2025‑11‑14 evidence called out in `analysis/artifact_inventory.txt`.
Re-run `pytest tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump -vv` and tee it to `$HUB/green/pytest_patch_stats_rerun.log` so POLICY-001/CONFIG-001 coverage shows the TrainingPayload-threaded flags still emit torch_patch_stats artifacts.
Using the exact commands documented in `.../plan/plan.md`, rerun the 10-epoch training job (`OUT=outputs/torch_forward_parity_baseline`) with `--log-patch-stats --patch-stats-limit 2` tee’d to `$HUB/cli/train_patch_stats_rerun.log`, then immediately run the matching inference command tee’d to `$HUB/cli/inference_patch_stats_rerun.log` with `--debug-dump "$HUB"/analysis/forward_parity_debug`.
Copy the newly emitted `torch_patch_stats*.json`, `torch_patch_grid*.png`, and debug dumps out of `outputs/torch_forward_parity_baseline/` into `$HUB/analysis/`, refresh `$HUB/analysis/artifact_inventory.txt` + `$HUB/summary.md` + the initiative summary, and if CUDA/memory blocks any command drop `$HUB/red/blocked_<timestamp>.md` citing POLICY-001/CONFIG-001 before stopping.

Summary: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md
Plan: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md
Selector: tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump
