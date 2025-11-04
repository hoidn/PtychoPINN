# Phase E Training Bundle Real Runs (Retry) — Plan

**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001  
**Focus:** Phase G comparison & analysis — unblock Phase E6/E7 evidence  
**Timestamp:** 2025-11-06T050500Z  
**Mode:** TDD Implementation (engineer loop)

## Context
- Attempt #98 delivered the MemmapDatasetBridge fallback (`ptycho_torch/memmap_bridge.py`), clearing the KeyError seen in Attempt #96 real runs (`KeyError: 'diff3d'`).
- Phase E6 exit criteria (`test_strategy.md:268`) still require real bundle evidence (dense/baseline) with SHA256 / manifest records archived under initiative reports.
- Phase G comparisons remain blocked until Phase E bundles exist. 
- tmp artifacts currently available:
  - `tmp/phase_c_f2_cli/` — dose-swept patched datasets (legacy `diffraction` key)
  - `tmp/phase_d_f2_cli/` — overlap filtered views (dense/sparse)
  - `tmp/phase_e_training_gs2/` — previous manifests (failed due to KeyError)

## Objectives for Ralph
1. Re-run deterministic Phase E training CLI for dose=1000 (dense gs2 + baseline gs1) now that fallback exists.
2. Archive manifests, skip summaries, and generated bundles (`wts.h5.zip` + logs) into the new artifact hub.
3. Compute and verify SHA256 digests, ensuring they match manifest `bundle_sha256` entries.
4. Update initiative summary with execution outcomes + remaining gaps.
5. Prepare for subsequent doc/test registry sync once CLI evidence is GREEN.

## Guardrails
- Findings to enforce: POLICY-001, CONFIG-001, DATA-001, OVERSAMPLING-001.
- `AUTHORITATIVE_CMDS_DOC` must remain `./docs/TESTING_GUIDE.md` for all CLI/test invocations.
- Use deterministic flags (`--accelerator cpu --deterministic --num-workers 0`) to match prior reproducibility guarantees.
- Preserve tmp datasets; regenerate only if directories missing.
- All artifacts belong under `plans/active/.../reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/`.

## Tasks & Expected Evidence

| ID | Task | Evidence / Destination |
| --- | --- | --- |
| R1 | Rerun targeted bundle persistence tests to confirm SHA256 + CLI selectors still GREEN. | `pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle -vv` → `green/pytest_bundle_sha_green.log`; `pytest tests/study/test_dose_overlap_training.py -k training_cli -vv` → `green/pytest_training_cli_suite_green.log`; collect-only proof `collect/pytest_training_cli_collect.log`. |
| R2 | Execute Phase E training CLI (dense gs2 + baseline gs1) using fallback. | Logs under `cli/dose1000_dense_gs2.log` and `cli/dose1000_baseline_gs1.log`; ensure stdout/stderr captured with exit code 0. |
| R3 | Archive outputs: copy `training_manifest.json`, `skip_summary.json`, and resulting `wts.h5.zip` bundles to `data/`; capture SHA256 sums to `analysis/bundle_checksums.txt`; store manifest pretty-print at `analysis/training_manifest_pretty.json`. |
| R4 | Update initiative summary with CLI outcomes, bundle locations, SHA256 verification, and remaining work items. | Append section to `analysis/summary.md` summarizing dense/baseline runs, include manifest status + confirmation of checksum match, note pending sparse runs / doc sync. |
| R5 | Update ledger and memory after engineer loop, referencing new artifacts. | Supervisor action post-implementation (this plan). |

## Success Criteria
- Both targeted tests GREEN with logs placed in artifact hub.
- CLI commands exit 0 and produce `bundle_path` + `bundle_sha256` entries in manifest for dense + baseline jobs.
- `data/` contains manifests + skip summary + at least two `wts.h5.zip` bundles; SHA256 file matches manifest values.
- `analysis/summary.md` reflects updated status and points to bundle artifacts.
- `docs/fix_plan.md` Attempt #99 captures this attempt with artifact links and notes (to be authored after engineer loop).

## Fallback / Blockers
- If CLI still fails (e.g., missing Phase D data), capture log + stacktrace in `analysis/summary.md`, retain exit code output, and notify via ledger entry (mark attempt blocked).
- If SHA256 mismatch, halt comparisons; record diff in summary and investigate before proceeding.

