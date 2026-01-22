Summary: Ship the D0 parity logger CLI and tests so photon_grid_study_20250826_152459 parity evidence (raw/grouped/normalized stats + probe metrics) is reproducible.
Focus: seed — Inbox monitoring and response (checklist S3)
Branch: dose_experiments
Mapped tests: pytest tests/tools/test_d0_parity_logger.py::test_cli_emits_outputs -q
Artifacts: plans/active/seed/reports/2026-01-22T030216Z/

Do Now:
- seed S3 — D0 parity logger shipping
  - Implement: scripts/tools/d0_parity_logger.py::main (plus sha256/summarize_array/summarize_grouped/summarize_probe helpers) to emit JSON, Markdown, and probe_stats.csv exactly as defined in the D0 plan.
  - Implement: tests/tools/test_d0_parity_logger.py::test_cli_emits_outputs covering helper stats logic and CLI outputs using a synthetic NPZ + params fixture.
  - Validate: pytest tests/tools/test_d0_parity_logger.py::test_cli_emits_outputs -q | tee "$ARTIFACT_DIR"/pytest_d0_parity_logger.log
  - Artifacts: "$ARTIFACT_DIR"/{dose_parity_log.json,dose_parity_log.md,probe_stats.csv,pytest_d0_parity_logger.log,d0_parity_collect.log}

How-To Map:
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md, export ARTIFACT_DIR=plans/active/seed/reports/2026-01-22T030216Z, and mkdir -p "$ARTIFACT_DIR".
2. Re-read docs/fix_plan.md (Current Focus + Attempts, lines 1–54) plus plans/active/seed/reports/2026-01-22T042640Z/d0_parity_logger_plan.md (lines 1–59) to keep scope aligned with maintainer requirements from inbox/README_prepare_d0_response.md lines 1–41.
3. Implement scripts/tools/d0_parity_logger.py by lifting sha256/JSON conversion ideas from plans/active/seed/bin/dose_baseline_snapshot.py: load all data_p1e*.npz files under photon_grid_study_20250826_152459 (sorted), compute dataset metadata (photon_dose, size_bytes, shapes, sha256) plus stage-level stats (raw, normalized, grouped) exactly per the plan (normalized = diff3d / (diff3d.max() + 1e-12); grouped intensities = per-pattern mean aggregated via np.bincount on scan_index). Log git SHA, scenario_id, params.dill excerpt, probe amplitude/phase percentiles, intensity_scale, and metric tuples into dose_parity_log.json → mirrored into Markdown with readable tables, plus probe_stats.csv (amplitude/phase percentiles per column).
4. Add tests/tools/test_d0_parity_logger.py: build a tiny temporary NPZ (diff3d with 2 patterns × 2×2 pixels, scan_index with repeats, probeGuess) and dill params dict. Unit-test summarize_array p01/p99/nonzero_fraction math, summarize_grouped scan aggregation, and ensure main() writes JSON/MD/CSV under tmp_path (use monkeypatch/env overrides to aim outputs at tmp_path without touching production datasets).
5. Execute pytest tests/tools/test_d0_parity_logger.py::test_cli_emits_outputs -q | tee "$ARTIFACT_DIR"/pytest_d0_parity_logger.log; fix any failures until it passes.
6. Run the CLI on the real scenario: python scripts/tools/d0_parity_logger.py --dataset-root photon_grid_study_20250826_152459 --baseline-params photon_grid_study_20250826_152459/results_p1e5/train_1024/trial_1/baseline_run/08-26-2025-16.38.17_baseline_gs1/08-26-2025-16.38.17_baseline_gs1/params.dill --scenario-id PGRID-20250826-P1E5-T1024 --output "$ARTIFACT_DIR" and confirm dose_parity_log.{json,md} plus probe_stats.csv populate that directory.
7. Capture a collect-only log for the selector: pytest --collect-only tests/tools/test_d0_parity_logger.py::test_cli_emits_outputs -q | tee "$ARTIFACT_DIR"/d0_parity_collect.log.
8. Append a new Attempts entry in docs/fix_plan.md summarizing the shipped CLI/test + evidence, and cite the new artifacts.

Pitfalls To Avoid:
- Do not mutate or relocate photon_grid_study_* datasets; treat them as read-only evidence sources.
- Avoid holding multiple 4 GB NPZs in memory simultaneously—load, summarize, and release each file sequentially.
- Use np.abs / np.angle before computing probe stats so complex arrays serialize cleanly.
- Normalize using (max + 1e-12) to avoid division-by-zero for blank scans.
- Convert NumPy scalars/tuples/bool_ to native Python before JSON/Markdown/CSV emission.
- Keep dataset ordering deterministic (sorted filenames) for repeatable diffs.
- Limit stdout noise; rely on the generated files for detailed tables rather than printing large blobs.
- Tests must rely solely on synthetic tmp_path fixtures—not on real photon_grid datasets.
- Do not introduce new dependencies or adjust the environment; if dill/numpy imports fail, stop and log the exact traceback.
- Stick to ASCII text in Markdown/CSV so maintainers can diff easily.

If Blocked:
- If photon_grid_study_20250826_152459 or the params.dill path is missing, capture `ls -R photon_grid_study_20250826_152459 | head` plus the exception into "$ARTIFACT_DIR"/missing_data.log and pause; log the blocker in docs/fix_plan.md Attempts + galph_memory.
- If dill loading errors occur, write the traceback to "$ARTIFACT_DIR"/dill_error.log and halt rather than guessing parameter names.
- If pytest cannot import numpy/dill, capture the full traceback to "$ARTIFACT_DIR"/pytest_import_error.log and mark the attempt blocked for maintainer guidance.

Findings Applied (Mandatory): No relevant findings in the knowledge base.

Pointers:
- docs/fix_plan.md:7 — focus + Attempts history referencing S3 blueprint.
- plans/active/seed/reports/2026-01-22T042640Z/d0_parity_logger_plan.md:1 — canonical D0 parity logger requirements, stats definitions, and testing expectations.
- inbox/README_prepare_d0_response.md:1 — maintainer parity logging checklist (probe provenance + stage stats) to cite in outputs.
- plans/active/seed/bin/dose_baseline_snapshot.py:1 — reference helper implementations for hashing + JSON conversion to reuse.
- specs/data_contracts.md:5 — RawData NPZ key/shape contract to validate diff3d/probeGuess/scan_index handling.

Next Up (optional): Capture a gs2 scenario run with the new CLI once S3 lands so parity evidence spans multiple gridsizes.

Doc Sync Plan (Conditional): After pytest passes, run pytest --collect-only tests/tools/test_d0_parity_logger.py::test_cli_emits_outputs -q | tee "$ARTIFACT_DIR"/d0_parity_collect.log, then add the selector to docs/TESTING_GUIDE.md (§Running Tests) and docs/development/TEST_SUITE_INDEX.md with a one-line description referencing the parity logger tooling.

Mapped Tests Guardrail: pytest --collect-only tests/tools/test_d0_parity_logger.py::test_cli_emits_outputs -q must produce "$ARTIFACT_DIR"/d0_parity_collect.log (>0 tests collected) before finishing; fix the test module immediately if the count is zero.

Hard Gate: Do not close this loop until pytest tests/tools/test_d0_parity_logger.py::test_cli_emits_outputs -q passes and both JSON + MD outputs exist under "$ARTIFACT_DIR"; if the selector fails or collects zero tests, write the failure signature and keep the item open.

Normative Math/Physics: Use specs/data_contracts.md §RawData NPZ (lines 5–13) for the authoritative diff3d/probeGuess/scan_index structure and the stage-stat rules from plans/active/seed/reports/2026-01-22T042640Z/d0_parity_logger_plan.md §Stage-Level Stats — reference those sections directly instead of paraphrasing equations.
