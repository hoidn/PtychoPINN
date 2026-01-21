Summary: Carry raw diffraction intensity stats through loader/train so `calculate_intensity_scale()` finally records the dataset-derived gain (≈577) and prove it by rerunning gs1_ideal + gs2_ideal with updated analyzer outputs.
Focus: DEBUG-SIM-LINES-DOSE-001 — Phase D4f raw dataset-intensity scale bridging
Branch: paper
Mapped tests: pytest tests/test_loader_normalization.py::TestNormalizeData::test_dataset_stats_attachment -v; pytest tests/test_train_pinn.py::TestIntensityScale::test_uses_dataset_stats -v; pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T012500Z/
Do Now (DEBUG-SIM-LINES-DOSE-001/D4f — see implementation.md §Phase D4f):
- Implement: Update `ptycho/loader.py::load` (plus helpers) to compute raw `diffraction` sum-of-squares stats per split, attach them to `PtychoDataContainer`, and teach `ptycho/train_pinn.py::calculate_intensity_scale` to prefer those stats before touching `_X_np`; extend `tests/test_loader_normalization.py::TestNormalizeData::test_dataset_stats_attachment` and `tests/test_train_pinn.py::TestIntensityScale::test_uses_dataset_stats` so the regression is locked in; rerun `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py` for gs1_ideal + gs2_ideal, regenerate `bias_summary.*`, and capture logs so we can show `bundle_intensity_scale == dataset_scale`.
- Validate via: `pytest tests/test_loader_normalization.py::TestNormalizeData::test_dataset_stats_attachment -v`; `pytest tests/test_train_pinn.py::TestIntensityScale::test_uses_dataset_stats -v`; `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v` after the plan-local runner commands below (logs + analyzer output under the artifacts hub).
- Artifacts: Store all pytest logs plus `gs*_ideal_runner.log`, `gs*_ideal/intensity_stats.*`, updated `bias_summary.{json,md}`, and `analyze_intensity_bias.log` inside `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T012500Z/`.
How-To Map:
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` before running anything.
2. `pytest tests/test_loader_normalization.py::TestNormalizeData::test_dataset_stats_attachment -v | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T012500Z/logs/pytest_loader_dataset_stats.log` (ensures loader attaches pre-normalization stats per split).
3. `pytest tests/test_train_pinn.py::TestIntensityScale::test_uses_dataset_stats -v | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T012500Z/logs/pytest_train_pinn_dataset_stats.log` so calculate_intensity_scale prioritizes the new container attr.
4. `python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py --scenario gs1_ideal --output-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T012500Z/gs1_ideal --group-limit 64 --prediction-scale-source least_squares` (logs + stats under gs1_ideal/); repeat for `--scenario gs2_ideal --output-dir .../gs2_ideal`.
5. `python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py --scenario gs1_ideal=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T012500Z/gs1_ideal --scenario gs2_ideal=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T012500Z/gs2_ideal --output-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T012500Z | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T012500Z/analyze_intensity_bias.log` to refresh `bias_summary.*` with the new scales.
6. `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T012500Z/logs/pytest_cli_smoke.log` as the CLI guard.
Pitfalls To Avoid:
- Do not touch `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`; the fix lives in loader/train_pinn/tests only.
- Ensure raw `diffraction` stats are computed **before** calling normalize_data() and follow the same train/test split order; slicing the normalized tensors defeats the purpose.
- Keep `_tensor_cache` untouched in calculate_intensity_scale()—always read `_X_np` or the new stats dict first (PINN-CHUNKED-001).
- Normalize math in float64 so mean-of-squares does not overflow; cast back to float32 when storing stats.
- Preserve normalize_data() semantics (fixed `(N/2)^2` target) and don’t regress the D4e revert.
- Verify `bundle_intensity_scale` == `dataset_scale` in `run_metadata.json` for both gs1_ideal and gs2_ideal before wrapping up.
- Always run commands with `AUTHORITATIVE_CMDS_DOC` exported and log outputs under the artifacts hub (TEST-CLI-001).
- Keep plan-local scripts device/dtype neutral; no ad-hoc sampling or GPU-only assumptions.
If Blocked: Capture the blocker (stack trace, unexpected tensor stats, etc.) inside `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T012500Z/blocked.md`, update docs/fix_plan.md Attempts History, and ping me before attempting alternative pipelines.
Findings Applied (Mandatory):
- CONFIG-001 — keep `update_legacy_dict(params.cfg, config)` ahead of loader/training so intensity stats map to the right gridsize.
- NORMALIZATION-001 — respect the three-stage normalization architecture (physics vs statistical vs display) when wiring dataset stats.
- PINN-CHUNKED-001 — never materialize `_tensor_cache`; prefer NumPy data for reducers.
- TEST-CLI-001 — archive pytest + CLI logs per selector and keep selectors green.
Pointers:
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md (Phase D4f checklist)
- specs/spec-ptycho-core.md:86 (Normalization Invariants equations)
- docs/DEVELOPER_GUIDE.md:157 (three-tier normalization architecture)
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py:1179-1675 (telemetry + analyzer hooks)
Next Up (optional): If D4f lands quickly, start framing Phase D5 (loss wiring instrumentation) using the refreshed analyzer evidence.
