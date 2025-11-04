# Phase E Training Bundle Real Runs (Execution) — Plan Refresh

**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001  
**Focus:** Phase G comparison & analysis — Phase E6 bundle evidence  
**Timestamp:** 2025-11-06T110500Z  
**Mode:** TDD Implementation (engineer loop)

## Context
- Attempt #102 landed stdout emission of `bundle_path`/`bundle_sha256` in `studies.fly64_dose_overlap.training.main` and kept targeted selectors green.
- Phase E6 still lacks deterministic CLI evidence for dose=1000 dense (gs2) and baseline (gs1); no bundles or SHA proofs archived yet in the execution hub.
- Plan/test_strategy (§268) require CLI runs, manifest + checksum archival, and analysis summary before Phase G comparisons can start.

## Objectives for Ralph
1. Harden regression coverage so CLI tests assert bundle SHA output.
2. Execute deterministic training CLI runs for dense/baseline views (dose 1000) and capture stdout with SHA lines.
3. Archive manifests, skip summaries, bundles, and SHA evidence in the refreshed hub.
4. Update analysis notes summarizing CLI outcomes, digests, and remaining work (sparse view, doc sync).

## Guardrails
- Findings to honor: POLICY-001 (torch >=2.2 required), CONFIG-001 (bridge params.cfg via CONFIG-001 helpers), DATA-001 (dataset contract for generated NPZ), OVERSAMPLING-001 (keep gridsize/view pairing valid).
- Export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` for every pytest/CLI command.
- Preserve existing tmp assets (`tmp/phase_c_f2_cli`, `tmp/phase_d_f2_cli`, `tmp/phase_e_training_gs2`); regenerate only when missing.
- CLI runs must use deterministic knobs `--accelerator cpu --deterministic --num-workers 0` and reuse `--artifact-root tmp/phase_e_training_gs2`.
- All artifacts for this loop (logs, bundles, checksums, notes) belong under `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T110500Z/phase_e_training_bundle_real_runs_exec/`.

## Tasks & Expected Evidence

| ID | Task | Evidence / Destination |
| --- | --- | --- |
| E1 | Extend `tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path` to require bundle/SHA stdout lines include view/dose context (e.g., `baseline (dose=1e3) → Bundle: ...`). Capture RED failure before CLI update, then GREEN after updating `training.py::main`. | RED log → `red/pytest_training_cli_stdout_red.log`; GREEN log → `green/pytest_training_cli_stdout_green.log`. |
| E2 | Re-run integrity selectors after test update to ensure regression suite still green. | `pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle -vv` → `green/pytest_bundle_sha_green.log`; `pytest tests/study/test_dose_overlap_training.py -k training_cli -vv` → `green/pytest_training_cli_green.log`; collect proof → `collect/pytest_training_cli_collect.log`. |
| E3 | Ensure dataset prerequisites exist; regenerate Phase C/D only if directories missing, capturing logs when commands run. | Any regeneration stdout/stderr → `prep/phase_c_generation.log`, `prep/phase_d_generation.log`. |
| E4 | Execute deterministic Phase E CLI runs for dose=1000 dense (gs2) and baseline (gs1), capture stdout/exit codes. | Logs → `cli/dose1000_dense_gs2.log`, `cli/dose1000_baseline_gs1.log`; ensure stdout shows bundle/SHA lines. |
| E5 | Archive results from `tmp/phase_e_training_gs2`: copy manifests, skip summary, bundles (renaming to `wts_dense_gs2.h5.zip` & `wts_baseline_gs1.h5.zip`), compute SHA256 digest file, pretty-print manifest for review. | Files under `data/` (manifest, skip summary, bundles); `analysis/bundle_checksums.txt`; `analysis/training_manifest_pretty.json`. |
| E6 | Summarize outcomes, blockers, and next steps (sparse run + doc/test registry sync) in `analysis/summary.md`. | Updated summary stored in hub; reference in docs/fix_plan.md attempt and galph_memory.

## Success Criteria
- New stdout assertion test fails before code change (RED) and passes after update (GREEN).
- Targeted selectors and CLI suite pass; collection proof captured.
- CLI runs exit 0 with stdout lines showing bundle path + SHA digest for each job.
- Bundles and manifests archived with SHA256 proof; manifest digest matches CLI output.
- `analysis/summary.md` reflects new evidence and residual work.

## Blockers / Fallbacks
- If stdout capture fails due to logging route, document failure in RED log and coordinate follow-up change to CLI output.
- If CLI runs error, archive stacktrace in `cli/` folder and stop after updating summary + ledger with BLOCKED status.
- If SHA mismatch occurs, record discrepancy, retain failing artifacts, and halt Phase G progression pending investigation.
