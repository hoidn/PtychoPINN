# Phase F3 Sparse Metadata Recovery

## Context
- Initiative: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Focus: Restore GREEN state for Phase F3 sparse LSQML execution after metadata extraction remains RED.
- Dependencies:
  - Attempt #86 greedy downsampling (Phase D7) – sparse NPZs provide `_metadata` JSON.
  - Attempt #87 implementation — added `extract_phase_d_metadata` + CLI integration but pytest still RED.
  - Phase F plan (reports/2025-11-04T094500Z/phase_f_ptychi_baseline_plan/plan.md) F3.A–F3.E checklist.

## Tasks

| ID | Task | State | Notes |
| --- | --- | --- | --- |
| M1 | Diagnose why `extract_phase_d_metadata` returns `{}` for dense jobs in pytest harness | [ ] | Reproduce RED failure in `tests/study/test_dose_overlap_reconstruction.py::test_cli_executes_selected_jobs`; instrument loader to inspect `_metadata` tensor shape/type. |
| M2 | Patch metadata parsing to coerce NumPy scalars/arrays into JSON dict | [ ] | Treat `_metadata` stored as 0d array or bytes; use `.item()` / `.tolist()` to get string before json.loads; include schema validation + helpful logging. |
| M3 | Extend pytest assertions for sparse jobs & update fixtures if needed | [ ] | Ensure `mock_phase_d_datasets` writes metadata consistent with Phase D7 outputs (`selection_strategy`, acceptance metrics). Add guard ensuring baseline NPZs remain unaffected. |
| M4 | Re-run targeted selectors + collect proof | [ ] | `pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_executes_selected_jobs -vv` and `pytest tests/study/test_dose_overlap_reconstruction.py -k "ptychi" -vv`; archive logs under `.../green/`; capture collect-only log. |
| M5 | Execute sparse/train and sparse/test CLI runs with distinct manifest snapshots | [ ] | Run CLI twice with same artifact root; after each run copy manifest/skip summary to split-specific filenames to avoid overwrite. Capture stdout (train/test) under `.../cli/`. |
| M6 | Update docs and ledger once GREEN | [ ] | Refresh `docs/TESTING_GUIDE.md`, `docs/development/TEST_SUITE_INDEX.md`, implementation/test_strategy references, and docs/fix_plan.md Attempt #88 entry summarizing fix + artifacts. |

## Artifact Hub
`plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133218Z/phase_f_ptychi_baseline_f3_metadata_recovery/`

Subdirs: plan/, red/, green/, collect/, cli/, real_run/, docs/

## References
- `studies/fly64_dose_overlap/reconstruction.py`
- `tests/study/test_dose_overlap_reconstruction.py`
- `plans/active/.../reports/2025-11-05T050500Z/phase_f_ptychi_baseline_f3_sparse_runs/`
- `docs/findings.md` POLICY-001, CONFIG-001, DATA-001, OVERSAMPLING-001
