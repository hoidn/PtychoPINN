# Phase E Training Bundle Real Runs (Execution) — Plan

**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001  
**Focus:** Phase G comparison & analysis — Phase E6 bundle evidence  
**Timestamp:** 2025-11-06T090500Z  
**Mode:** TDD Implementation (engineer loop)

## Context
- Attempt #101 added on-disk SHA256 verification to `test_execute_training_job_persists_bundle`, satisfying specs/ptychodus_api_spec.md §4.6 integrity requirements.
- Real CLI evidence (dense gs2 + baseline gs1) remains outstanding; artifact hub 2025-11-06T050500Z lacks `cli/` logs, manifests, and bundle copies.
- Phase E6 exit criteria (plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md §268) require deterministic CLI execution, SHA256 proof, and archival before Phase G comparisons can proceed.

## Objectives for Ralph
1. Regenerate Phase C/D data only if directories missing; otherwise reuse existing tmp assets.
2. Execute deterministic Phase E training CLI for dose=1000 dense (gs2) and baseline (gs1) views.
3. Archive manifests, skip summaries, and `wts.h5.zip` bundles under the new artifact hub with SHA256 verification.
4. Update `analysis/summary.md` with outcomes, referencing log paths and checksum validation.

## Guardrails
- Findings: POLICY-001 (PyTorch dependency), CONFIG-001 (params.cfg bridge), DATA-001 (dataset contract), OVERSAMPLING-001 (gridsize constraints).
- `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` must prefix all pytest/CLI commands.
- Preserve tmp datasets (`tmp/phase_c_f2_cli`, `tmp/phase_d_f2_cli`, `tmp/phase_e_training_gs2`); regenerate only when absent.
- CLI runs must use deterministic flags `--accelerator cpu --deterministic --num-workers 0`.
- All artifacts for this loop belong under `plans/active/.../reports/2025-11-06T090500Z/phase_e_training_bundle_real_runs_exec/`.

## Tasks & Expected Evidence

| ID | Task | Evidence / Destination |
| --- | --- | --- |
| E1 | Run targeted pytest selectors to confirm SHA256 assertion + CLI harness still GREEN. | `pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle -vv` → `green/pytest_bundle_sha_green.log`; `pytest tests/study/test_dose_overlap_training.py -k training_cli -vv` → `green/pytest_training_cli_green.log`; collect-only proof `collect/pytest_training_cli_collect.log`. |
| E2 | Ensure datasets exist. Regenerate Phase C/D dose=1000 views only if missing. | Place regeneration logs (if run) under `prep/phase_c_generation.log` and `prep/phase_d_generation.log`. |
| E3 | Execute Phase E training CLI deterministic runs (dense gs2, baseline gs1). | Logs under `cli/dose1000_dense_gs2.log` and `cli/dose1000_baseline_gs1.log`; capture exit codes and key manifest lines. |
| E4 | Archive outputs: copy manifests, skip summaries, and bundles into hub; compute SHA256 digests. | `data/training_manifest.json`, `data/skip_summary.json`, `data/wts_dense_gs2.h5.zip`, `data/wts_baseline_gs1.h5.zip`; digests in `analysis/bundle_checksums.txt`; manifest pretty-print `analysis/training_manifest_pretty.json`. |
| E5 | Document outcomes + next steps. | Update `analysis/summary.md` with CLI success/failure, checksum results, bundle locations, and pending work (e.g., sparse view, doc sync). |

## Success Criteria
- Targeted pytest selectors PASS with artifact logs saved.
- CLI commands exit 0; logs include `bundle_path` + `bundle_sha256` entries.
- `analysis/bundle_checksums.txt` shows SHA256 values matching manifest entries.
- `analysis/summary.md` updated with bundle evidence and verification notes.
- docs/fix_plan.md records Attempt #102 referencing this hub, noting real-run status.

## Blockers / Fallbacks
- If Phase C/D regeneration fails, capture stderr into `prep/` logs and mark attempt BLOCKED in summary + ledger.
- If CLI raises, archive stacktrace in `cli/` and halt further steps pending supervisor guidance.
- If SHA mismatch occurs, document discrepancy in summary and stop Phase G progression until resolved.
