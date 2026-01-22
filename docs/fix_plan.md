# Fix Plan

## Status
`in_progress`

## Current Focus
**seed — Inbox monitoring and response (checklist S4)**

Need to finish the D0 parity handoff by surfacing stage-level stats for every dataset in the Markdown log and syncing the new CLI/test selector across the testing docs so maintainers can run it without parsing JSON.

## Completed Items
- [x] S1: Check `inbox/` for new requests — found `README_prepare_d0_response.md`
- [x] S2: Document response plan — created `inbox/response_prepare_d0_response.md`
- [x] S3: Ship D0 parity logger CLI + tests, produce PGRID-20250826-P1E5-T1024 evidence

## Artifacts Produced
- `plans/active/seed/bin/dose_baseline_snapshot.py` — reusable script for baseline snapshots
- `plans/active/seed/reports/2026-01-22T024002Z/dose_baseline_summary.json` — machine-readable snapshot
- `plans/active/seed/reports/2026-01-22T024002Z/dose_baseline_summary.md` — human-readable summary
- `plans/active/seed/reports/2026-01-22T024002Z/pytest_seed.log` — test run log
- `inbox/response_prepare_d0_response.md` — maintainer reply covering sections 1–7
- `scripts/tools/d0_parity_logger.py` — shipping CLI for parity evidence capture
- `tests/tools/test_d0_parity_logger.py` — unit + integration tests for the CLI
- `plans/active/seed/reports/2026-01-22T030216Z/dose_parity_log.json` — structured parity evidence
- `plans/active/seed/reports/2026-01-22T030216Z/dose_parity_log.md` — human-readable parity summary
- `plans/active/seed/reports/2026-01-22T030216Z/probe_stats.csv` — probe amplitude/phase percentiles
- `plans/active/seed/reports/2026-01-22T030216Z/pytest_d0_parity_logger.log` — test execution log
- `plans/active/seed/reports/2026-01-22T030216Z/d0_parity_collect.log` — test collection verification

## Attempts History

### 2026-01-22T02:40Z — seed S1–S2 (initial)
**Action:** Created dose_baseline_snapshot.py, ran it to generate JSON/MD summaries, drafted maintainer response.

**Metrics:**
- Scenario ID: PGRID-20250826-P1E5-T1024
- Baseline ms_ssim: 0.925 (train) / 0.921 (test)
- Baseline psnr: 71.3 dB (amplitude)
- 7 datasets captured (1e3 through 1e9 photons)
- All SHA256 checksums recorded

**Test:** `pytest tests/test_generic_loader.py::test_generic_loader -q` — PASSED (1 passed, 5 warnings)

**Note:** The mapped test `tests/test_generate_data.py::test_placeholder` failed collection due to `ptycho/data_preprocessing.py:13` assertion (`data_source == 'generic'`). Used alternative test to validate environment.

**Artifacts:**
- `plans/active/seed/reports/2026-01-22T024002Z/`

**Next Actions:**
- Await maintainer review of `inbox/response_prepare_d0_response.md`
- If time permits, capture gs2 variant for future parity logging

### 2026-01-22T04:26Z — seed S3 (blueprint)
**Action:** Authored `plans/active/seed/reports/2026-01-22T042640Z/d0_parity_logger_plan.md` to scope the shipping CLI + tests for D0 parity logging. Captures dataset/probe metrics, raw/grouped/normalized stats, and testing expectations before promoting the script under `scripts/tools/`.

**Artifacts:**
- `plans/active/seed/reports/2026-01-22T042640Z/d0_parity_logger_plan.md`

**Next Actions:**
- Promote the plan into implementation: add `scripts/tools/d0_parity_logger.py` plus `tests/tools/test_d0_parity_logger.py`, then run the CLI over photon_grid_study_20250826_152459 to produce JSON/MD artifacts under a fresh timestamped reports directory.

### 2026-01-22T03:06Z — seed S3 (shipped)
**Action:** Verified existing implementation of `scripts/tools/d0_parity_logger.py` and `tests/tools/test_d0_parity_logger.py`. Ran tests successfully, then executed CLI on photon_grid_study_20250826_152459 to produce parity evidence.

**Test:** `pytest tests/tools/test_d0_parity_logger.py::test_cli_emits_outputs -q` — PASSED (1 passed in 0.11s)

**Collection:** `pytest --collect-only tests/tools/test_d0_parity_logger.py::test_cli_emits_outputs -q` — 1 test collected

**CLI Run:**
```
python scripts/tools/d0_parity_logger.py \
  --dataset-root photon_grid_study_20250826_152459 \
  --baseline-params "photon_grid_study.../params.dill" \
  --scenario-id PGRID-20250826-P1E5-T1024 \
  --output plans/active/seed/reports/2026-01-22T030216Z
```

**Metrics captured:**
- Scenario ID: PGRID-20250826-P1E5-T1024
- 7 datasets processed (1e3 through 1e9 photons)
- Baseline ms_ssim: 0.9248 (train) / 0.9206 (test)
- Baseline psnr: 71.32 dB (train) / 158.06 dB (test)
- intensity_scale_value: 988.21
- Probe stats: identical across all datasets (same probe, different photon counts)
- Raw diffraction stats: min=0, max varies by dose, nonzero_fraction ~0.21
- All SHA256 checksums recorded for reproducibility

**Artifacts:**
- `plans/active/seed/reports/2026-01-22T030216Z/`

**Next Actions:**
- Await maintainer review of parity evidence
- Optionally capture gs2 scenario for multi-gridsize comparison

### 2026-01-22T03:11Z — seed S4 (doc sync + Markdown parity tables)
**Action:** Reviewed the emitted Markdown parity log and found it only lists stage-level stats for the first dataset, forcing maintainers to open the JSON blob to compare other photon doses. Also confirmed the testing docs still omit the new `tests/tools/test_d0_parity_logger.py::test_cli_emits_outputs` selector that guards the CLI. Logged the gap and outlined the follow-up scope so the CLI deliverable fully matches the maintainer spec.

**Artifacts:**
- `plans/active/seed/reports/2026-01-22T031142Z/`

**Next Actions:**
- Extend `scripts/tools/d0_parity_logger.py::write_markdown` (and helpers if needed) to render stage-level stats tables for every processed dataset, including raw, normalized, and grouped values so reviewers can diff doses without JSON parsing.
- Update `tests/tools/test_d0_parity_logger.py` to assert the expanded Markdown content and add coverage for the `--limit-datasets` filter so maintainers can shorten runs.
- Sync `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md` with the new selector + CLI instructions, then re-run `pytest tests/tools/test_d0_parity_logger.py -q` with fresh logs.

## TODOs
- [ ] S4: Expand the D0 parity Markdown report to list stage-level stats for every dataset and document the new test selector (`tests/tools/test_d0_parity_logger.py`) inside `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md`.
- [x] S3: Promote D0 parity logger into `scripts/tools/` with stage-level stats + tests, then capture artifacts for photon_grid_study_20250826_152459
