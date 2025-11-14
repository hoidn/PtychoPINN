Brief:
Guard the repo plus Reports Hub by exporting `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`, `HUB=$PWD/plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity`, and `OUT=outputs/torch_forward_parity_baseline`, then capture the first few lines of `green/pytest_patch_stats_rerun_v2.log`, `cli/train_patch_stats_rerun_v2.log`, and `cli/inference_patch_stats_rerun_v2.log` to prove they still show the 2025-11-14 timestamps before clobbering.
Re-run `pytest tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump -vv`, teeing to `$HUB/green/pytest_patch_stats_rerun_v3.log` (or overwrite `_v2`) so the TrainingPayload-threaded selector stays GREEN.
Execute the documented 10-epoch training command against `datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_{train,test}.npz` with `--log-patch-stats --patch-stats-limit 2`, tee to `$HUB/cli/train_patch_stats_rerun_v3.log`, then immediately run the matching inference command with `--debug-dump "$HUB"/analysis/forward_parity_debug_v2` tee’d to `$HUB/cli/inference_patch_stats_rerun_v3.log`.
Copy the regenerated `torch_patch_stats*_v2.{json,png}` (rename to `_v3` if needed) plus the refreshed `forward_parity_debug_v2` bundle from `$OUT` into `$HUB/analysis/`, update `$HUB/analysis/artifact_inventory.txt`, `$HUB/summary.md`, and `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md`, and create `$HUB/red/blocked_<timestamp>.md` citing POLICY-001 / CONFIG-001 / ANTIPATTERN-001 immediately if CUDA/memory interrupts either CLI.

Summary: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md
Plan: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md
Selector: tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump
