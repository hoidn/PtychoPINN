Brief:
Export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md plus HUB=$PWD/plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity and keep HUB clean while fixing the CLI/workflow bridge so the TrainingPayload built in `ptycho_torch/train.py:720-758` is actually reused by `run_cdi_example_torch → train_cdi_model_torch → _train_with_lightning`, letting `_train_with_lightning` consume the same `pt_inference_config` instead of rebuilding configs without `log_patch_stats` (`ptycho_torch/workflows/components.py:664-690` shows the current re-derivation).
After the hand-off, rerun `pytest tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump -vv` and capture the log to $HUB/green/pytest_patch_stats.log so the selector proves that JSON/PNG artifacts appear under `<output_dir>/analysis`.
Then execute the short 10-epoch training command and paired inference command from plan/plan.md with `--log-patch-stats --patch-stats-limit 2`, writing logs to $HUB/cli/{train_patch_stats.log,inference_patch_stats.log}` plus the debug dump under $HUB/analysis/forward_parity_debug and ensuring torch_patch_stats.json + torch_patch_grid.png land in $HUB/analysis/.
Finish by refreshing $HUB/analysis/artifact_inventory.txt, $HUB/summary.md, and the initiative summary—or drop $HUB/red/blocked_<timestamp>.md immediately if GPU/memory prevents completion (cite POLICY-001/CONFIG-001).

Summary: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md
Plan: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md
Selector: tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump
