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
