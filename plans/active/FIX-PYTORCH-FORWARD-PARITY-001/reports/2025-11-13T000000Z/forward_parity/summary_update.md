### Turn Summary (2025-11-16T173000Z)
Wired --log-patch-stats and --patch-stats-limit flags through ptycho_torch/inference.py CLI and _run_inference_and_reconstruct helper.
Added TestPatchStatsCLI::test_patch_stats_dump selector to test_cli_train_torch.py proving flags are accepted and training completes, but artifacts are not emitted (blocker documented).
Root cause likely: PatchStatsLogger.stats empty or output_dir lazy init silently failing; finalize early-exits without logging.
Next: add debug logging to _log_patch_stats/finalize to diagnose why stats list stays empty, then rerun test and capture artifacts.
Artifacts: ptycho_torch/inference.py (wired), tests/torch/test_cli_train_torch.py (test added), red/blocked_*_patch_stats_not_emitted.md, green/pytest_patch_stats.log (FAILED - no artifacts)

Checklist:
- Files touched: ptycho_torch/inference.py, tests/torch/test_cli_train_torch.py
- Tests run: pytest tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump -xvs
- Artifacts updated: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/red/blocked_*_patch_stats_not_emitted.md, green/pytest_patch_stats.log
