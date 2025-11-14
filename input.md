Brief:
Phase C3 now focuses on turning the existing Torch patch-stats selector into a regression guard: seed `TestPatchStatsCLI.minimal_train_args` so it writes a deterministic non-zero-variance NPZ, then update `test_patch_stats_dump` to parse `<output_dir>/analysis/torch_patch_stats.json` and assert `patch_amplitude.var_zero_mean > 1e-6` plus non-zero canvas means (cite POLICY-001/CONFIG-001 in a short comment).
Run `pytest tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump -vv | tee "$HUB"/green/pytest_patch_variance_guard.log` with `HUB=$PWD/plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity`, then update `$HUB/analysis/artifact_inventory.txt`, `$HUB/summary.md`, and the initiative summary with the guard result or drop `$HUB/red/blocked_<ts>_patch_variance_guard.md` if the selector fails/missing stats.
Make sure the threshold rationale references `analysis/phase_c2_pytorch_only_metrics.txt` so reviewers see why we only guard gridsize≥2, and leave docs/test registries untouched since the selector name stays the same.

Summary: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md
Plan: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md
Selector: tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump
