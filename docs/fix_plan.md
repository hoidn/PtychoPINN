# Fix Plan

## Status
`in_progress`

## Current Focus
**seed — Inbox monitoring and response (checklist S1–S2)**

Captured photon_grid dose baseline facts and drafted maintainer-ready D0 parity response.

## Completed Items
- [x] S1: Check `inbox/` for new requests — found `README_prepare_d0_response.md`
- [x] S2: Document response plan — created `inbox/response_prepare_d0_response.md`

## Artifacts Produced
- `plans/active/seed/bin/dose_baseline_snapshot.py` — reusable script for baseline snapshots
- `plans/active/seed/reports/2026-01-22T024002Z/dose_baseline_summary.json` — machine-readable snapshot
- `plans/active/seed/reports/2026-01-22T024002Z/dose_baseline_summary.md` — human-readable summary
- `plans/active/seed/reports/2026-01-22T024002Z/pytest_seed.log` — test run log
- `inbox/response_prepare_d0_response.md` — maintainer reply covering sections 1–7

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

## TODOs
- [ ] S3: Promote D0 parity logger into `scripts/tools/` with stage-level stats + tests, then capture artifacts for photon_grid_study_20250826_152459
