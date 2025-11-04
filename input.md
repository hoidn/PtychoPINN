Summary: Fix Phase F3 sparse LSQML metadata extraction and recapture sparse train/test evidence without overwriting manifests.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.F3 — Sparse LSQML execution telemetry
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_executes_selected_jobs -vv; pytest tests/study/test_dose_overlap_reconstruction.py -k "ptychi" -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133218Z/phase_f_ptychi_baseline_f3_metadata_recovery/

Do Now:
- Implement: studies/fly64_dose_overlap/reconstruction.py::extract_phase_d_metadata — decode NumPy scalars/arrays into JSON and merge metadata; update tests/study/test_dose_overlap_reconstruction.py::test_cli_executes_selected_jobs fixture/assertions so RED log lands before patch. Capture failing run at plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133218Z/phase_f_ptychi_baseline_f3_metadata_recovery/red/pytest_phase_f_sparse_red.log.
- Validate: pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_executes_selected_jobs -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133218Z/phase_f_ptychi_baseline_f3_metadata_recovery/green/pytest_phase_f_sparse_green.log; pytest tests/study/test_dose_overlap_reconstruction.py -k "ptychi" -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133218Z/phase_f_ptychi_baseline_f3_metadata_recovery/green/pytest_phase_f_sparse_suite_green.log.
- Collect: pytest tests/study/test_dose_overlap_reconstruction.py --collect-only -k ptychi -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133218Z/phase_f_ptychi_baseline_f3_metadata_recovery/collect/pytest_phase_f_sparse_collect.log.
- CLI sparse/train: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.reconstruction --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133218Z/phase_f_ptychi_baseline_f3_metadata_recovery/real_run --dose 1000 --view sparse --split train --allow-missing-phase-d 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133218Z/phase_f_ptychi_baseline_f3_metadata_recovery/cli/sparse_train.log; cp plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133218Z/phase_f_ptychi_baseline_f3_metadata_recovery/real_run/reconstruction_manifest.json plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133218Z/phase_f_ptychi_baseline_f3_metadata_recovery/real_run/reconstruction_manifest_sparse_train.json; cp plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133218Z/phase_f_ptychi_baseline_f3_metadata_recovery/real_run/skip_summary.json plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133218Z/phase_f_ptychi_baseline_f3_metadata_recovery/real_run/skip_summary_sparse_train.json.
- CLI sparse/test: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.reconstruction --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133218Z/phase_f_ptychi_baseline_f3_metadata_recovery/real_run --dose 1000 --view sparse --split test --allow-missing-phase-d 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133218Z/phase_f_ptychi_baseline_f3_metadata_recovery/cli/sparse_test.log; cp plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133218Z/phase_f_ptychi_baseline_f3_metadata_recovery/real_run/reconstruction_manifest.json plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133218Z/phase_f_ptychi_baseline_f3_metadata_recovery/real_run/reconstruction_manifest_sparse_test.json; cp plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133218Z/phase_f_ptychi_baseline_f3_metadata_recovery/real_run/skip_summary.json plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133218Z/phase_f_ptychi_baseline_f3_metadata_recovery/real_run/skip_summary_sparse_test.json.
- Artifacts: Summarize metadata extraction fix + sparse run outcomes in plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133218Z/phase_f_ptychi_baseline_f3_metadata_recovery/docs/summary.md; update docs/TESTING_GUIDE.md and docs/development/TEST_SUITE_INDEX.md with sparse selectors, rerun evidence, and manifest snapshot notes; record Attempt #88 implementation results in docs/fix_plan.md and mark plan/test_strategy checkboxes when GREEN.

Priorities & Rationale:
- docs/fix_plan.md:31 — Status flags Phase F3 metadata as blocking sparse LSQML evidence; this loop clears the blocker.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline_plan/plan.md:48 — F3.1–F3.4 require metadata surfacing plus sparse train/test runs with docs sync.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133218Z/phase_f_ptychi_baseline_f3_metadata_recovery/plan/plan.md:1 — Checklist M1–M6 outlines this remediation scope.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:212 — Phase F selectors demand RED→GREEN proof for `-k "ptychi"` after metadata changes.
- studies/fly64_dose_overlap/reconstruction.py:274 — Metadata extraction helper currently returning `{}`; must decode JSON per DATA-001 findings.

How-To Map:
- export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
- Reproduce RED: pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_executes_selected_jobs -vv --maxfail=1 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133218Z/phase_f_ptychi_baseline_f3_metadata_recovery/red/pytest_phase_f_sparse_red.log
- Patch extract_phase_d_metadata to call `.item()` / `.tolist()` before json.loads and validate keys; adjust fixture metadata typing; rerun GREEN selector command (tee to green log) and full `-k "ptychi"` regression (tee to suite log)
- Collect-only proof: pytest tests/study/test_dose_overlap_reconstruction.py --collect-only -k ptychi -vv 2>&1 | tee plans/active/.../collect/pytest_phase_f_sparse_collect.log
- Sparse train run: ensure tmp/phase_c_f2_cli + tmp/phase_d_f2_cli exist; run train command (tee log); immediately copy manifest + skip summary to split-specific filenames to preserve evidence
- Sparse test run: same as train; copy manifest/skip summary to `_sparse_test` variants so both splits documented
- Verify manifest JSON includes `selection_strategy`, `acceptance_rate`, `spacing_threshold`, `n_accepted`, `n_rejected` for both splits; note acceptance rates in docs summary.md
- Update docs/TESTING_GUIDE.md (Phase F section) and TEST_SUITE_INDEX row with new selector outputs + manifest snapshot references; add summary of run outcomes and manifest filenames
- Append Attempt #88 results and artifact links in docs/fix_plan.md; mark plan M1–M6 progress and Phase F3 checkboxes when criteria satisfied

Pitfalls To Avoid:
- Do not mutate Phase D NPZs in place; read-only access preserves DATA-001 compliance.
- Keep reconstruction CLI pure (no params.cfg writes) per CONFIG-001; limit changes to metadata parsing.
- Ensure copied manifest filenames stay within artifact hub (no workspace root clutter).
- Preserve deterministic ordering of execution_results to avoid flaky assertions.
- After CLI runs, remove stale `reconstruction_manifest.json` if reusing artifact root would confuse later evidence — use copies instead of deleting.
- Avoid rerunning CLI with dry-run flag; need real execution telemetry even if returncode≠0.
- Do not install packages or modify environment; rely on existing torch/tike installation.
- Capture stderr in tee logs so singular matrix or warning details persist.
- Keep summary.md concise but include acceptance_rate + selection_strategy for both splits.
- Validate `--collect-only` output still lists ≥1 test; investigate immediately if zero collected.

If Blocked:
- If pytest still missing metadata after patch, print `type(data['_metadata'])` within helper (temporary) to inspect structure, capture snippet in summary.md, then remove debug before commit.
- If sparse CLI fails before emitting manifest, keep stdout/stderr logs, copy partial manifest if created, and record blocking error (e.g., torch.linalg.solve LinAlgError) in docs/summary.md plus docs/fix_plan Attempt log.
- If tmp/phase_c_f2_cli or tmp/phase_d_f2_cli missing, rerun Phase C/D generation commands documented in docs/TESTING_GUIDE.md:184 and archive regeneration log under this hub; if regeneration fails, mark attempt blocked with error signature.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch dependency already in place; keep CLI runs on existing torch stack without optional toggles.
- CONFIG-001 — Maintain pure config flow; metadata extraction must not touch global params.cfg.
- DATA-001 — Respect NPZ schema and treat `_metadata` as auxiliary JSON without altering diffraction arrays.
- OVERSAMPLING-001 — Confirm acceptance_rate + spacing_threshold reflect Phase D greedy selection guard (>10% acceptance); document values in summary.

Pointers:
- docs/fix_plan.md:31
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline_plan/plan.md:48
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133218Z/phase_f_ptychi_baseline_f3_metadata_recovery/plan/plan.md:1
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:212
- studies/fly64_dose_overlap/reconstruction.py:274
- tests/study/test_dose_overlap_reconstruction.py:440
- docs/TESTING_GUIDE.md:184

Next Up (optional): Phase F3 manifest docs sync follow-up or transition to Phase G comparisons once sparse runs are GREEN.
