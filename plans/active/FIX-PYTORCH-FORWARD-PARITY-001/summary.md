### Turn Summary (2025-11-16T151500Z)
Re-read the PyTorch workflow/spec docs and verified the forward_parity hub still lacks torch_patch_stats.json / torch_patch_grid.png, so Phase A instrumentation + baseline reruns remain outstanding.
Refreshed the Phase A plan + fix_plan Do Now to emphasize the CLI flag plumbing, targeted pytest selector, and the short train/inference commands under POLICY-001 / CONFIG-001 constraints.
Next: wire the patch-stat logger through model/inference, run `tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump`, then rerun the short baseline + inference with instrumentation enabled and update the hub inventory.
Artifacts: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/{plan/plan.md,analysis/artifact_inventory.txt}

### Turn Summary (2025-11-14T014500Z)
Fixed the factory config propagation blocker by adding PTInferenceConfig to TrainingPayload and wiring it through config_factory→train.py→workflows/components.py.
Root cause was that create_training_payload() omitted PTInferenceConfig, so --log-patch-stats/--patch-stats-limit flags were lost and the workflow created a default disabled config instead.
Implemented the fix in 3 files (config_factory.py, train.py, workflows/components.py), added 2 new targeted tests proving factory creates the config correctly, and committed with 4/4 GREEN pytest results.
Next: Run end-to-end baseline training with --log-patch-stats --patch-stats-limit 2 to verify instrumentation produces JSON/PNG artifacts in analysis/.
Artifacts: ptycho_torch/{config_factory,train,workflows/components}.py diffs, tests/torch/test_patch_stats_cli.py (4/4 GREEN)

Checklist:
- Files touched: ptycho_torch/config_factory.py, ptycho_torch/train.py, ptycho_torch/workflows/components.py, tests/torch/test_patch_stats_cli.py
- Tests run: pytest tests/torch/test_patch_stats_cli.py -v; pytest tests/torch/test_config_factory.py -v
- Artifacts updated: none (no baseline run yet; config fix only)

### Turn Summary (2025-11-14T013800Z)
Wired PatchStatsLogger into PtychoPINN_Lightning model (__init__, _log_patch_stats, on_train_end) and ran training baseline successfully, but instrumentation didn't activate because config_factory doesn't create PTInferenceConfig for training workflows.
Root cause: create_training_payload() omits PT InferenceConfig entirely, so CLI overrides (log_patch_stats/patch_stats_limit) never reach the model instance.
Shipped the model integration (commit 34cbbe64), proved CLI flags parse correctly (test GREEN), and documented the factory blocker in red/blocked_*.md with exact fix steps.
Next: update config_factory to create/populate PTInferenceConfig with patch stats overrides, rerun training, and verify JSON/PNG artifacts land in analysis/.
Artifacts: ptycho_torch/model.py (wired), green/pytest_patch_stats_integration.log (PASSED), cli/train_patch_stats.log (completed but no stats), red/blocked_*_factory_inference_config.md

Checklist:
- Files touched: ptycho_torch/model.py
- Tests run: pytest tests/torch/test_patch_stats_cli.py::TestPatchStatsCLI::test_patch_stats_flags_accepted -v
- Artifacts updated: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/{green/pytest_patch_stats_integration.log, cli/train_patch_stats.log, red/blocked_*_factory_inference_config.md}

### Turn Summary (2025-11-16T131500Z)
Shipped Phase A CLI instrumentation nucleus: added `--log-patch-stats` and `--patch-stats-limit` flags to `ptycho_torch/train.py`, created `ptycho_torch/patch_stats_instrumentation.py` module with PatchStatsLogger class, extended InferenceConfig with new fields, and proved flags work via targeted pytest.
Test `tests/torch/test_patch_stats_cli.py::TestPatchStatsCLI::test_patch_stats_flags_accepted` passes (1/1 GREEN), confirming flags are parsed and forwarded through the config stack to factory overrides.
The instrumentation module is ready but not yet wired into model forward pass; that requires config_factory bridge updates (Phase B scope) to populate InferenceConfig from overrides and pass it to PtychoPINN_Lightning.
Next: update config_factory to accept log_patch_stats/patch_stats_limit overrides, instantiate PatchStatsLogger in model __init__, and invoke logger.log_batch() from training_step/validation_step before attempting the full baseline rerun.
Artifacts: green/pytest_patch_stats_flags.log (PASSED), analysis/artifact_inventory.txt, ptycho_torch/{patch_stats_instrumentation,config_params,train}.py diffs, tests/torch/test_patch_stats_cli.py

### Turn Summary
Bootstrapped FIX-PYTORCH-FORWARD-PARITY-001 now that the exporter initiative is complete and Tier‑3 dwell pressure forced the fly64 dense rerun focus to stay blocked.
Created the forward_parity hub at `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/` and staged the Phase A Do Now: add optional patch-stat instrumentation to `ptycho_torch/model.py` + `ptycho_torch/inference.py`, wire CLI flags (`--log-patch-stats/--patch-stats-limit`), and emit JSON + normalized PNG dumps under `$HUB/analysis/`.
Documented the short 10‑epoch baseline commands (train → inference) that must be rerun with instrumentation enabled plus the targeted pytest selector to keep the new flags covered.
Next: implement the instrumentation, capture the CLI + pytest logs under the new hub, and update the artifact inventory once the patch stats land.
Artifacts: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/{plan/plan.md}
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
