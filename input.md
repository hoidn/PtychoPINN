Summary: Build the probe normalization comparison CLI and capture stats for Phase B2.
Focus: DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy
Branch: paper (sync with origin/paper)
Mapped tests: pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T031500Z/

Do Now (hard validity contract)
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/probe_normalization_report.py::main — new CLI that loads the Phase A snapshot, reconstructs both probe normalization paths (legacy set_default_probe() vs sim_lines make_probe + normalize_probe_guess) for `gs1_custom`, `gs1_ideal`, `gs2_custom`, `gs2_ideal`, and emits JSON + Markdown summaries with amplitude min/max/mean, L2 norm, and ratio deltas; include scenario metadata and CLI log under the artifacts hub.
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/__init__.py (if needed) to keep plan-local scripts importable without polluting production modules; keep code under plans/active to honor non-production guardrails.
- Pytest: pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T031500Z/{probe_stats_gs1_custom.json, probe_stats_gs1_custom.md, ..., pytest_sim_lines_pipeline_import.log}

How-To Map
1. Run `python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/probe_normalization_report.py --snapshot plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/sim_lines_4x_params_snapshot.json --scenario <scenario> --output-json plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T031500Z/probe_stats_<scenario>.json --output-markdown plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T031500Z/probe_stats_<scenario>.md` for each scenario. The CLI should:
   - Read RunParams + ScenarioSpec from the snapshot (reuse helpers from grouping_summary where possible).
   - For the legacy branch: set `params.cfg['default_probe_scale']=scenario probe_scale`, call `probe.set_default_probe()` (idealized) or load the custom probe NPZ, then apply the legacy normalization (params-driven scale) to match dose_experiments behavior.
   - For the sim_lines branch: use `make_probe()` + `normalize_probe_guess()` with matching arguments (mode, scale, mask/big flags) to mirror the modern pipeline.
   - Compute stats: amp min/max/mean, std, L2 norm, normalization factor, difference ratios, optionally probe mask coverage; store them under clearly named JSON keys and human-friendly Markdown tables.
2. Capture a CLI log (text file) summarizing commands + warnings in the artifacts hub.
3. Run `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T031500Z/pytest_sim_lines_pipeline_import.log`.

Pitfalls To Avoid
- Do not touch production modules (ptycho/, scripts/) — keep all new code under plans/active.
- Respect CONFIG-001: call `update_legacy_dict(params.cfg, config)` before probe helpers that rely on params.cfg.
- Avoid mutating global params.cfg state permanently; use context managers or restore prior values after generating probes.
- Keep device/dtype neutral (numpy-based stats only); no TensorFlow-heavy deps.
- Ensure CLI accepts custom probe path (from snapshot) and fails fast if the NPZ is missing.
- Avoid ad-hoc scripts outside plans/active; reuse existing helper code through imports.
- Do not skip the pytest guard even though this is a plan-local CLI.

If Blocked
- If the snapshot file or custom probe NPZ is missing, document the missing asset in docs/fix_plan.md Attempts History and capture the traceback under the artifacts hub.
- If CLI raises import errors due to plan-local modules, re-export the needed helpers via a small shim (without touching production code) and note it in the plan before retrying.

Findings Applied (Mandatory)
- CONFIG-001 — call `update_legacy_dict(params.cfg, config)` before using legacy probe helpers (`set_default_probe`, `simulate_nongrid_raw_data`).
- MODULE-SINGLETON-001 — avoid relying on module-level singletons; construct probes explicitly per scenario and do not leak them into global state.
- NORMALIZATION-001 — keep physics/statistical/display normalization stages distinct; the CLI should report amplitude stats without reusing Yon mixing of scales.
- BUG-TF-001 — when gridsize differs between pipelines, ensure params.cfg is synchronized before any legacy grouping call to avoid shape mismatches.

Pointers
- docs/specs/spec-ptycho-workflow.md — grouping + normalization contract lines 1–40, 80–120.
- docs/DATA_GENERATION_GUIDE.md — grid vs nongrid simulation overview (sections “Grid-Based Pipeline” and “Nongrid Pipeline”).
- scripts/simulation/synthetic_helpers.py:40-130 — make_probe + normalize_probe_guess implementations to mirror in the CLI.
- ptycho/probe.py:94-150 — legacy `set_default_probe()` and normalization behavior.
- plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T020000Z/grouping_*.json — evidence showing current grouping behavior that motivates B2.

Next Up (optional)
- If time permits after completing B2, begin instrumenting `bin/grouping_summary.py` for Phase B3 neighbor-count sweeps (low priority until probe comparison lands).

Doc Sync Plan (not needed — no test additions).
Mapped Tests Guardrail satisfied via the CLI smoke test above.
Normative Math/Physics — reference `spec-ptycho-workflow.md` §Normalization and §Probe setup for any formula citations inside the CLI.
