Brief:
Set AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md, HUB=$PWD/plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity, and OUT=outputs/torch_forward_parity_baseline, then confirm the hub tree is clean so the new evidence overwrites the Nov-14 logs called out in analysis/artifact_inventory.txt (KB: POLICY-001 / CONFIG-001 / ANTIPATTERN-001).
Re-run pytest tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump -vv | tee "$HUB"/green/pytest_patch_stats_rerun.log to prove the TrainingPayload-threaded patch-stat flags still emit JSON/PNG artifacts.
Execute the documented 10-epoch Torch training command against datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_train.npz (256 groups, gridsize=2) with --log-patch-stats --patch-stats-limit 2 tee'd to "$HUB"/cli/train_patch_stats_rerun.log, then immediately run the matching inference command on the test NPZ with --debug-dump "$HUB"/analysis/forward_parity_debug | tee "$HUB"/cli/inference_patch_stats_rerun.log.
Copy the refreshed torch_patch_stats*.json, torch_patch_grid*.png, and forward_parity_debug bundle from "$OUT"/analysis/ into "$HUB"/analysis/, update "$HUB"/analysis/artifact_inventory.txt, "$HUB"/summary.md, and the initiative summary, and drop "$HUB"/red/blocked_<timestamp>.md citing POLICY-001 / CONFIG-001 if CUDA/memory interrupts any step.

Summary: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md
Plan: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md
Selector: tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump
