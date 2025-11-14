# Blocker: Patch Stats Artifacts Not Emitted (2025-11-14T020000Z)

## Symptom
Training completes successfully with `--log-patch-stats --patch-stats-limit 2` but no artifacts appear in `<output_dir>/analysis/`:
- Missing: `torch_patch_stats.json`
- Missing: `torch_patch_grid.png`

Test selector: `pytest tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump`

## Investigation
1. PatchStatsLogger initialized with `enabled=True, limit=2, output_dir=None` in model.__init__
2. `_log_patch_stats` called from `compute_loss` → sets output_dir on first call
3. `on_train_end` calls `self.patch_stats_logger.finalize()`
4. Finalize checks `if not self.enabled or not self.output_dir or not self.stats: return`

## Hypothesis
Either:
- `self.stats` is empty (log_batch never appended)
- `output_dir` is still None (lazy init failed)
- `finalize` exited early silently

## Next Steps
1. Add debug logging to `_log_patch_stats` showing when it's called and whether logger.should_log() returns True
2. Add debug logging to `finalize` showing early-exit conditions
3. Check if `batch_idx=0` is causing limit issues (maybe batch_count never increments)
4. Verify `self.patch_stats_logger.stats` is populated before finalize

## Command
```bash
pytest tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump -xvs
```

Training succeeded (exit_code=0), bundle saved, but no patch stats artifacts emitted.
