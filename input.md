Brief:
Export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md and HUB=$PWD/plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity, keeping the hub tree clean while you gather fresh instrumentation evidence.
Rerun `pytest tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump -vv` and tee the log to $HUB/green/pytest_patch_stats_rerun.log so we have a post-fix proof that torch_patch_stats.json/torch_patch_grid.png are emitted.
Then execute the 10-epoch training command from plan/plan.md with `OUT=outputs/torch_forward_parity_baseline`, `--log-patch-stats --patch-stats-limit 2`, and tee stdout to $HUB/cli/train_patch_stats_rerun.log, followed immediately by the matching inference command tee’d to $HUB/cli/inference_patch_stats_rerun.log with `--debug-dump "$HUB"/analysis/forward_parity_debug`.
Copy the emitted torch_patch_stats.json and torch_patch_grid.png into $HUB/analysis/, refresh $HUB/analysis/artifact_inventory.txt plus $HUB/summary.md (and the initiative summary), and log $HUB/red/blocked_<timestamp>.md if CUDA/memory stops any command (cite POLICY-001/CONFIG-001 in the blocker).

Summary: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md
Plan: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md
Selector: tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump
