Brief:
Extend `TestPatchStatsCLI::test_patch_stats_dump` so it runs the inference CLI after the existing training call (reuse the tmp outputs, pass `--log-patch-stats --patch-stats-limit 2 --accelerator cpu --quiet`) and generates `analysis/torch_patch_stats_inference.json` plus the inference patch-grid PNG.
Parse the first batch of that JSON and assert `patch_amplitude.var_zero_mean > 1e-6` and `abs(global_mean) > 1e-9`, citing `analysis/phase_c2_pytorch_only_metrics.txt` and `docs/specs/spec-ptycho-workflow.md` in the inline comment to show why gridsize≥2 must retain variance.
Re-run `pytest tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump -vv | tee "$HUB"/green/pytest_patch_variance_guard.log` (with `HUB=$PWD/plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity`), then record the pass/fail result in `$HUB/analysis/artifact_inventory.txt`, `$HUB/summary.md`, and the initiative summary—or drop `$HUB/red/blocked_<ts>_patch_variance_guard_inference.md` if the inference stats are missing or violate the thresholds.

Summary: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md
Plan: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md
Selector: tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump
