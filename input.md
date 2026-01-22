Summary: Promote the D0 parity logger blueprint into a shipping CLI + unit test that captures raw/grouped/normalized stats for photon_grid_study_20250826_152459.
Focus: seed — Inbox monitoring and response (checklist S3)
Branch: dose_experiments
Mapped tests: pytest tests/tools/test_d0_parity_logger.py::test_cli_emits_outputs -q
Artifacts: plans/active/seed/reports/2026-01-22T042640Z/

Do Now:
- Implement: scripts/tools/d0_parity_logger.py::main (and helper functions) plus tests/tools/test_d0_parity_logger.py::test_cli_emits_outputs to promote the D0 parity logging CLI called for in the plan (raw/grouped/normalized stats, probe logging, JSON/MD/CSV output).
- Test: pytest tests/tools/test_d0_parity_logger.py::test_cli_emits_outputs -q | tee "$ARTIFACT_DIR"/pytest_d0_parity_logger.log
- Artifacts: "$ARTIFACT_DIR"/{dose_parity_log.json,dose_parity_log.md,probe_stats.csv,pytest_d0_parity_logger.log,d0_parity_collect.log}

How-To Map:
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md, export ARTIFACT_DIR=plans/active/seed/reports/2026-01-22T042640Z, and mkdir -p "$ARTIFACT_DIR".
2. Read docs/fix_plan.md (§Current Focus + Attempts, lines 1–54) and plans/active/seed/reports/2026-01-22T042640Z/d0_parity_logger_plan.md (lines 1–59) for scope; keep inbox/README_prepare_d0_response.md lines 1–41 handy for maintainer requirements.
3. Implement scripts/tools/d0_parity_logger.py: reuse sha256 logic from plans/active/seed/bin/dose_baseline_snapshot.py while adding helpers `summarize_array`, `summarize_grouped`, and `summarize_probe` exactly as defined in the blueprint (raw stats, scan-index grouped means via np.bincount, normalized diff3d = diff3d / (max + 1e-12)). Include metadata (scenario_id, git SHA, dataset root), baseline params fields, probe flags, and stage-level stats blocks. Emit JSON + Markdown + probe_stats.csv into $ARTIFACT_DIR. Guard main() with if __name__ == "__main__".
4. Add tests/tools/test_d0_parity_logger.py that builds a tiny synthetic NPZ + dill params file under tmp_path; unit-test summarize_array + summarize_grouped outputs and ensure cli `main` writes JSON/MD/CSV for the temporary dataset. Keep runtime small (<1s) by using 2 patterns × 2×2 pixels.
5. Run pytest tests/tools/test_d0_parity_logger.py::test_cli_emits_outputs -q | tee "$ARTIFACT_DIR"/pytest_d0_parity_logger.log. If the selector fails to collect, fix the test module before rerunning.
6. Execute the CLI against the real dataset: python scripts/tools/d0_parity_logger.py --dataset-root photon_grid_study_20250826_152459 --baseline-params photon_grid_study_20250826_152459/results_p1e5/train_1024/trial_1/baseline_run/08-26-2025-16.38.17_baseline_gs1/08-26-2025-16.38.17_baseline_gs1/params.dill --scenario-id PGRID-20250826-P1E5-T1024 --output "$ARTIFACT_DIR"; verify dose_parity_log.{json,md} + probe_stats.csv appear.
7. After tests pass, run pytest --collect-only tests/tools/test_d0_parity_logger.py::test_cli_emits_outputs -q | tee "$ARTIFACT_DIR"/d0_parity_collect.log as evidence for the guardrail and doc sync.
8. Update docs/TESTING_GUIDE.md §Running Tests and docs/development/TEST_SUITE_INDEX.md to list the new selector once code and collect-only succeed.

Pitfalls To Avoid:
- Do not move or modify photon_grid_study_* data; read-only access only.
- Keep summary calculations streaming-friendly (iterate over files, avoid loading all NPZs simultaneously).
- Handle complex arrays carefully: use np.abs / np.angle before computing stats to avoid ValueErrors.
- Ensure normalized stats divide by (max + 1e-12) to avoid zero-division when datasets are blank.
- Keep JSON serializable types (convert NumPy scalars, booleans, tuples) to plain Python before dumping.
- Maintain deterministic dataset ordering (sorted filenames) so future diffs stay stable.
- Limit stdout noise; capture CLI output to files under $ARTIFACT_DIR rather than printing large tables.
- Tests must create and clean up temporary NPZ/dill artifacts; never rely on production datasets during pytest.
- No environment or dependency changes; if imports fail, stop and surface the exact traceback.
- Write only ASCII/UTF-8 text; do not embed binary blobs in Markdown.

If Blocked:
- If the photon_grid_study_20250826_152459 data or params.dill is missing, capture `ls -R photon_grid_study_20250826_152459 | head` plus the exception into "$ARTIFACT_DIR"/missing_data.log and halt—note the block in docs/fix_plan.md Attempts and galph_memory.
- If dill loading fails, redirect the traceback to "$ARTIFACT_DIR"/dill_error.log and stop rather than guessing field names; log the blocker.
- If pytest cannot import numpy/dill, do not skip tests—record the failure signature and wait for maintainer guidance.

Findings Applied (Mandatory): No relevant findings in the knowledge base.

Pointers:
- docs/fix_plan.md:1 — focus/status + S3 blueprint next steps.
- plans/active/seed/reports/2026-01-22T042640Z/d0_parity_logger_plan.md:1 — detailed CLI/test requirements.
- inbox/README_prepare_d0_response.md:1 — maintainer parity logging checklist and stage-level expectations.
- plans/active/seed/bin/dose_baseline_snapshot.py:1 — reference implementation of baseline snapshot helpers to reuse.
- specs/data_contracts.md:1 — authoritative NPZ key/shape definitions referenced for dataset parsing.

Next Up (optional): Build gs2 dataset variant once the logger is merged so parity evidence covers multiple gridsizes.

Doc Sync Plan (Conditional): After pytest passes and collect-only log is captured, append the new selector to docs/TESTING_GUIDE.md (Running Tests list) and docs/development/TEST_SUITE_INDEX.md; include the collect-only log at "$ARTIFACT_DIR"/d0_parity_collect.log.

Mapped Tests Guardrail: Step 7 ensures `pytest --collect-only tests/tools/test_d0_parity_logger.py::test_cli_emits_outputs -q` produces "$ARTIFACT_DIR"/d0_parity_collect.log so the selector is proven collectible.
