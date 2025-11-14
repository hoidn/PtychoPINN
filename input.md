Brief:
Guard the repo plus Reports Hub, export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md, HUB=$PWD/plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity, and OUT=outputs/torch_forward_parity_baseline, then confirm the existing HUB logs still show their 2025-11-14 timestamps so the rerun will overwrite them.
Re-run pytest tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump -vv | tee "$HUB"/green/pytest_patch_stats_rerun_v2.log, execute the documented 10-epoch training command on datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_train.npz with --log-patch-stats --patch-stats-limit 2 | tee "$HUB"/cli/train_patch_stats_rerun_v2.log, and immediately follow with the matching inference command on the test NPZ using --debug-dump "$HUB"/analysis/forward_parity_debug_v2 | tee "$HUB"/cli/inference_patch_stats_rerun_v2.log.
Copy the refreshed torch_patch_stats*_v2.{json,png} outputs and the forward_parity_debug_v2 bundle from "$OUT"/analysis/ into "$HUB"/analysis/, then update "$HUB"/analysis/artifact_inventory.txt, "$HUB"/summary.md, and plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md so the `_v2` evidence is clearly recorded.
Log `$HUB`/red/blocked_<timestamp>.md citing KB POLICY-001 / CONFIG-001 / ANTIPATTERN-001 immediately if CUDA/memory interrupts any command; otherwise leave the hub clean for the next Phase B task.

Summary: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md
Plan: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md
Selector: tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump
