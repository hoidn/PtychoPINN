## Summary
Add a dose_experiments-style normalization capture CLI so we can gather RawData→grouped→normalized→container stage stats plus dataset-vs-fallback intensity scales without rerunning the full sim_lines training loop.

## Focus
DEBUG-SIM-LINES-DOSE-001 — Phase D2 normalization parity instrumentation (D2b CLI)

## Branch
paper

## Mapped tests
- pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v

## Artifacts
plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T124212Z/

## Do Now — DEBUG-SIM-LINES-DOSE-001.D2b normalization parity
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/capture_dose_normalization.py::main — add a plan-local CLI that loads the legacy `dose_experiments_param_scan.md` defaults (gridsize=2, probe_scale=4, neighbor_count=5, etc.), simulates the nongrid dataset via `make_lines_object`/`simulate_nongrid_raw_data`, splits along `y`, and records stage telemetry using the existing `record_intensity_stage` helper so `write_intensity_stats_outputs` can emit both JSON + Markdown. The CLI must compute the dataset-derived intensity scale (per `specs/spec-ptycho-core.md §Normalization Invariants`) and the closed-form fallback `sqrt(nphotons)/(N/2)`, feed both into the payload, duplicate the outputs to `dose_normalization_stats.{json,md}`, and persist `capture_config.json` + `capture_summary.md` with the scenario parameters, normalization ratios, and spec citation. Support `--overwrite` to clear existing outputs safely.
- Implement: docs/fix_plan.md — log this D2b attempt (timestamp, CLI path, artifact hub) under DEBUG-SIM-LINES-DOSE-001, ensuring the Dependencies/Status fields reflect that normalization parity capture is underway and referencing the new `dose_normalization_stats` artifacts.
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md — update the Phase D checklist so D2b describes the new CLI entry point (and mark as completed once artifacts exist), plus refresh `plans/active/DEBUG-SIM-LINES-DOSE-001/summary.md` with a note that the normalization-only CLI now exists for quick parity captures.

## How-To Map
1. `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/capture_dose_normalization.py --scenario dose_legacy_gs2 --output-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T124212Z/dose_normalization --total-images 1024 --group-count 64 --neighbor-count 5 --nphotons 1e9 --buffer 10 --object-size 392 --sim-seed 42 --object-seed 42 --custom-probe-path ptycho/datasets/Run1084_recon3_postPC_shrunk_3.npz --probe-mode custom --probe-scale 4.0 --gridsize 2 --split-fraction 0.5 | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T124212Z/dose_normalization/capture.log`
2. `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T124212Z/pytest_cli_smoke.log`

## Pitfalls To Avoid
- Keep the new CLI plan-local; do not modify production modules under `ptycho/` or `scripts/studies/`.
- Reference `specs/spec-ptycho-core.md §Normalization Invariants` verbatim in the capture summary; avoid paraphrasing equations.
- Compute both dataset-derived and closed-form intensity scales and serialize them (JSON + Markdown) so analyzer diffs stay meaningful.
- Preserve the exact stage order (`raw_diffraction`, `grouped_diffraction`, `grouped_X_full`, `container_X`) used by `run_phase_c2_scenario` so comparisons are apples-to-apples.
- Default to fail-fast if the output directory already contains stats; only delete files when `--overwrite` is passed and log the action.
- Keep seeds and scenario counts deterministic (object seed, sim seed, total images) for reproducibility and caching.
- Capture all CLI stdout/stderr plus `capture_config.json` inside the artifacts hub before touching docs.
- Do not kick off training/inference workloads; the CLI should stop after loader-based normalization stats.
- Leave the memoized RawData cache enabled unless the CLI exposes a flag; avoid bespoke cache directories.
- When updating docs, describe the actual CLI behavior and artifact paths after verifying files exist.

## If Blocked
If the CLI fails (e.g., missing probe NPZ or loader import error), write the full traceback to `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T124212Z/dose_normalization/blocker.log`, add a DEBUG-SIM-LINES-DOSE-001 Attempts History note in docs/fix_plan.md summarizing the blocker, and stop rather than guessing at parameters.

## Findings Applied (Mandatory)
- CONFIG-001 — keep params.cfg synchronized via `update_legacy_dict` before calling loader/generator helpers.
- SIM-LINES-CONFIG-001 — plan-local runners must bridge legacy params prior to grouping/inference to avoid NaNs; document the bridge in the CLI summary.
- NORMALIZATION-001 — track all three normalization systems explicitly and cite the spec when presenting the dataset-scale vs fallback numbers.

## Pointers
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:330 — Phase D checklist + D2b requirements.
- docs/fix_plan.md:218 — D2 Attempts History describing the normalization parity plan.
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py:333 — intensity stage helpers to reuse.
- specs/spec-ptycho-core.md:86 — Normalization invariant clauses.
- docs/findings.md:22 — NORMALIZATION-001 guidance.
- docs/TESTING_GUIDE.md:1 — pytest command policy for plan-local work.

## Next Up (optional)
If the CLI + docs land quickly, start scoping D3 (hyperparameter deltas) by cataloging nepochs, batch sizes, and scheduler settings between dose_experiments and sim_lines.
