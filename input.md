Summary: Persist Phase E training bundles so Phase G comparisons can load real PINN/baseline models.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.G2 — Deterministic comparison runs
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle -vv; pytest tests/study/test_dose_overlap_training.py -k training_cli -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T210500Z/phase_e_training_bundle/

Do Now:
- Implement: studies/fly64_dose_overlap/training.py::execute_training_job + tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle — add a RED→GREEN test proving we emit spec-compliant `wts.h5.zip` bundles (and report bundle paths) when training succeeds, wiring the runner to call `save_torch_bundle` and persist manifest fields for downstream consumers.
- Validate: pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T210500Z/phase_e_training_bundle/red/pytest_execute_training_job_bundle_red.log (expect failure), then rerun after implementation teeing GREEN output to plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T210500Z/phase_e_training_bundle/green/pytest_execute_training_job_bundle_green.log.
- Validate: pytest tests/study/test_dose_overlap_training.py -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T210500Z/phase_e_training_bundle/green/pytest_training_cli_suite_green.log to confirm no regressions in the CLI harness.
- Collect: pytest tests/study/test_dose_overlap_training.py --collect-only -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T210500Z/phase_e_training_bundle/collect/pytest_training_cli_collect.log (Doc Sync guardrail).
- Capture: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view dense --gridsize 2 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T210500Z/phase_e_training_bundle/cli/dose1000_dense_train.log (ensure resulting bundle + manifest copied into the timestamped artifact hub).
- Summarize: Update plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T210500Z/phase_e_training_bundle/analysis/summary.md with bundle paths, manifest excerpts, and outstanding gaps (e.g., missing sparse/test bundles).

Priorities & Rationale:
- docs/fix_plan.md:31 — Phase G comparisons remain blocked on real training bundles; this loop targets the gating prerequisite.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/plan/plan.md:32 — G2.1 expects dense runs with valid checkpoints before analysis.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T162500Z/phase_g_inventory/analysis/inventory.md:67 — Inventory notes `checkpoint.h5`/`wts.h5.zip` gaps for dose_1000; filling them unblocks comparisons.
- specs/ptychodus_api_spec.md:239 — Spec §4.6 mandates training emit `wts.h5.zip` archives for ModelManager consumers.
- docs/findings.md:8 (POLICY-001) — PyTorch tooling must be present; persistence path should surface actionable failures if torch unavailable.

How-To Map:
- export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
- mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T210500Z/phase_e_training_bundle/{red,green,collect,cli,analysis,docs}
- pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle -vv | tee .../red/pytest_execute_training_job_bundle_red.log
- Modify execute_training_job to call save_torch_bundle on success, set result['bundle_path'], and update manifest writer accordingly; add the new pytest covering bundle persistence + manifest fields.
- pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T210500Z/phase_e_training_bundle/green/pytest_execute_training_job_bundle_green.log
- pytest tests/study/test_dose_overlap_training.py -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T210500Z/phase_e_training_bundle/green/pytest_training_cli_suite_green.log
- pytest tests/study/test_dose_overlap_training.py --collect-only -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T210500Z/phase_e_training_bundle/collect/pytest_training_cli_collect.log
- AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view dense --gridsize 2 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T210500Z/phase_e_training_bundle/cli/dose1000_dense_train.log (if phase_c/phase_d roots missing, rerun the Phase C/D generation CLIs first and record commands in summary).
- cp -r tmp/phase_e_training_gs2/pinn tmp/phase_e_training_gs2/baseline plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T210500Z/phase_e_training_bundle/cli/ (include manifest + bundle for traceability) and document locations in analysis/summary.md.

Pitfalls To Avoid:
- Do not skip the RED test run; capture failing assertion before implementation.
- Avoid mutating Phase G executor this loop—keep scope limited to Phase E persistence.
- Ensure bundle files land inside artifact_root; no stray archives at repo root or /tmp beyond the canonical staging area.
- Preserve CONFIG-001 purity: execute_training_job must not mutate params.cfg beyond the existing bridge call.
- Do not delete/overwrite existing manifest fields; append new keys instead.
- Keep training CLI deterministic (set seed if needed) and note runtime; abort if execution exceeds budget without progress evidence.
- Do not assume tmp/phase_c_f2_cli paths exist—list directories and record outcome if regeneration required.
- Avoid recompressing artifacts inside artifact hub (wts.h5.zip should copy as-is).
- No environment/package changes; rely on existing torch install per POLICY-001.
- Capture stderr for CLI runs—tee ensures logs include failure context if persistence fails.

If Blocked:
- If save_torch_bundle raises (e.g., torch missing), keep RED logs, record the exception in docs/fix_plan.md, and mark the attempt blocked with artifact pointers so we can request environment fixes next loop.

Findings Applied (Mandatory):
- POLICY-001 — docs/findings.md:8; PyTorch dependency enforced, failures should raise visibly.
- CONFIG-001 — docs/findings.md:10; update_legacy_dict already executed before persistence, no extra global state edits.
- DATA-001 — docs/findings.md:14; source NPZ paths stay canonical (patched_{split}.npz, dense/sparse overlays).
- OVERSAMPLING-001 — docs/findings.md:17; document acceptance metadata when summarizing sparse runs.

Pointers:
- studies/fly64_dose_overlap/training.py:344 — Runner helper where bundle persistence belongs.
- tests/study/test_dose_overlap_training.py:987 — Existing CLI integration test to extend with bundle assertions.
- specs/ptychodus_api_spec.md:239 — Model persistence requirements (wts.h5.zip contract).
- docs/TESTING_GUIDE.md:111 — Phase E training selectors + CLI commands for authoritative execution references.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T190500Z/phase_g_execution_runs/analysis/summary.md:129 — Current failure mode citing missing bundles.

Next Up:
- After bundles exist, rerun Phase G comparisons (dose_1000 dense/test) to gather metrics.

Doc Sync Plan:
- Once GREEN, rerun `pytest tests/study/test_dose_overlap_training.py --collect-only -k training_cli -vv` (log captured above) and update docs/TESTING_GUIDE.md §Phase E plus docs/development/TEST_SUITE_INDEX.md with the new bundle persistence test and CLI command references before closing the loop.
