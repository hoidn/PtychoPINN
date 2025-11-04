Summary: Produce Phase E training bundles with manifest SHA256 metadata and archive real dose=1000 dense/baseline runs for Phase G.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (Phase E6 bundle evidence)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle -vv; pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv; pytest tests/study/test_dose_overlap_training.py -k training_cli -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/

Do Now:
- Implement: studies/fly64_dose_overlap/training.py::execute_training_job + tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle — add SHA256 computation for saved wts.h5.zip bundles, surface `bundle_sha256` alongside the normalized `bundle_path`, and extend the persistence test to assert a 64-character hex digest while keeping the relative-path invariant.
- Validate: pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/red/pytest_bundle_sha_red.log (run immediately after updating the test to capture the expected RED failure).
- Validate: pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/green/pytest_bundle_sha_green.log (rerun after implementation for GREEN evidence).
- Validate: pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/green/pytest_cli_bundlepath_green.log to confirm manifest normalization plus checksum propagation.
- Validate: pytest tests/study/test_dose_overlap_training.py -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/green/pytest_training_cli_suite_green.log for regression confidence.
- Collect: pytest tests/study/test_dose_overlap_training.py --collect-only -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/collect/pytest_training_cli_collect.log (guard mapped selectors).
- Prepare: if [ ! -d tmp/phase_c_f2_cli ]; then AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.generation --base-npz datasets/fly64/fly64_shuffled.npz --output-root tmp/phase_c_f2_cli | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/cli/phase_c_generation.log; fi
- Prepare: if [ ! -d tmp/phase_d_f2_cli ]; then AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.overlap --phase-c-root tmp/phase_c_f2_cli --output-root tmp/phase_d_f2_cli | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/cli/phase_d_overlap.log; fi
- Execute: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view dense --gridsize 2 --accelerator cpu --deterministic --num-workers 0 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/cli/dose1000_dense_gs2.log
- Execute: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view baseline --gridsize 1 --accelerator cpu --deterministic --num-workers 0 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/cli/dose1000_baseline_gs1.log
- Archive: cp tmp/phase_e_training_gs2/training_manifest.json plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/data/training_manifest.json && cp tmp/phase_e_training_gs2/skip_summary.json plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/data/skip_summary.json && find tmp/phase_e_training_gs2/dose_1000 -name 'wts.h5.zip' -exec cp {} plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/data/ \;
- Summarize: sha256sum plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/data/wts.h5.zip* | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/analysis/bundle_checksums.txt && python -m json.tool tmp/phase_e_training_gs2/training_manifest.json > plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/analysis/training_manifest_pretty.json && update plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/analysis/summary.md with CLI outcomes, manifest excerpts, checksum table, and outstanding gaps.
- Document: After GREEN, refresh docs/TESTING_GUIDE.md §Phase E and docs/development/TEST_SUITE_INDEX.md to mention the new `bundle_sha256` field plus real-run selectors, and log collect-only proof in summary before closing the attempt.

Priorities & Rationale:
- docs/fix_plan.md:53 — Attempt #27 closes manifest normalization but flags Phase E7 real runs as the next blocker for Phase G comparisons.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T230500Z/phase_e_training_bundle_real_run/analysis/summary.md:207 — Outstanding gap explicitly defers real CLI execution pending dataset regeneration and artifact capture.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:268 — Phase E6 checklist requires non-dry-run evidence for dense/sparse overlap conditions with deterministic seeds.
- specs/ptychodus_api_spec.md:239 — PyTorch backend must persist `wts.h5.zip` archives; manifest checksum guards ensure downstream loaders trust bundles.
- docs/TESTING_GUIDE.md:133 — Authoritative Phase E CLI commands; all runs must respect documented environment knobs and capture logs.

How-To Map:
- export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
- mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/{plan,red,green,collect,cli,data,docs,analysis}
- pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/red/pytest_bundle_sha_red.log
- Edit execute_training_job + test to compute and assert bundle SHA256
- pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/green/pytest_bundle_sha_green.log
- pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/green/pytest_cli_bundlepath_green.log
- pytest tests/study/test_dose_overlap_training.py -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/green/pytest_training_cli_suite_green.log
- pytest tests/study/test_dose_overlap_training.py --collect-only -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/collect/pytest_training_cli_collect.log
- if [ ! -d tmp/phase_c_f2_cli ]; then AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.generation --base-npz datasets/fly64/fly64_shuffled.npz --output-root tmp/phase_c_f2_cli | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/cli/phase_c_generation.log; fi
- if [ ! -d tmp/phase_d_f2_cli ]; then AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.overlap --phase-c-root tmp/phase_c_f2_cli --output-root tmp/phase_d_f2_cli | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/cli/phase_d_overlap.log; fi
- AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view dense --gridsize 2 --accelerator cpu --deterministic --num-workers 0 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/cli/dose1000_dense_gs2.log
- AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view baseline --gridsize 1 --accelerator cpu --deterministic --num-workers 0 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/cli/dose1000_baseline_gs1.log
- cp tmp/phase_e_training_gs2/training_manifest.json plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/data/training_manifest.json
- cp tmp/phase_e_training_gs2/skip_summary.json plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/data/skip_summary.json
- find tmp/phase_e_training_gs2/dose_1000 -name 'wts.h5.zip' -exec cp {} plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/data/ \;
- sha256sum plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/data/wts.h5.zip* | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/analysis/bundle_checksums.txt
- python -m json.tool tmp/phase_e_training_gs2/training_manifest.json > plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/analysis/training_manifest_pretty.json
- Update plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/analysis/summary.md with CLI logs, manifest excerpts, checksum verification, dataset provenance, and remaining deltas.
- After GREEN, edit docs/TESTING_GUIDE.md and docs/development/TEST_SUITE_INDEX.md to document checksum field plus real-run selectors; capture collect-only log reference in analysis summary.

Pitfalls To Avoid:
- Do not skip capturing the RED failure log; attach it before modifying production code.
- Ensure bundle SHA computation runs on the real archive file; avoid hashing empty temp files.
- Keep bundle paths relative in the manifest; checksum addition must not reintroduce absolute paths.
- Regenerate Phase C/D data only when directories are missing; avoid clobbering existing evidence without logging it.
- Run CLI commands with `--accelerator cpu` and deterministic flags to stay within Environment Freeze boundaries.
- Preserve CONFIG-001 discipline—no direct writes to `params.cfg` inside execute_training_job.
- Store CLI stdout/stderr via tee under the reserved artifact hub; do not leave logs in tmp/ or repo root.
- Copy only targeted outputs (manifest, skip summary, bundles); do not duplicate entire dataset trees into the artifact hub.
- Verify `bundle_sha256` values are 64-char lowercase hex before updating docs.
- If training fails, do not retry blindly—capture the failure log and analyze before re-running.

If Blocked:
- Capture the failing command output under plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/analysis/blocked_cli.log (or similar), note the error signature, and summarize the block in analysis/summary.md and docs/fix_plan.md Attempt update; mark the focus blocked pending dataset regeneration or runtime fix.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch backend must stay available; CLI runs with torch>=2.2 (docs/findings.md:8).
- CONFIG-001 — TrainingConfig bridges legacy params once; checksum logic must not mutate global state (docs/findings.md:10).
- DATA-001 — Phase C/D regeneration relies on canonical NPZ layouts; do not introduce ad-hoc schema changes (docs/findings.md:14).
- OVERSAMPLING-001 — Dense/gs2 runs still rely on neighbor_count=7; document sparse skip status if encountered (docs/findings.md:17).

Pointers:
- docs/fix_plan.md:53 — Phase E6 next-step directive for real training bundles.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T230500Z/phase_e_training_bundle_real_run/analysis/summary.md:207 — Deferred real-run rationale & follow-up scope.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:268 — Phase E6 evidence requirements.
- specs/ptychodus_api_spec.md:239 — Bundle persistence contract (`wts.h5.zip`).
- docs/TESTING_GUIDE.md:133 — Canonical Phase E training CLI invocation knobs.

Doc Sync Plan:
- After tests and real runs are GREEN, rerun `pytest tests/study/test_dose_overlap_training.py --collect-only -k training_cli -vv` (already captured above), update docs/TESTING_GUIDE.md §Phase E and docs/development/TEST_SUITE_INDEX.md to include the new checksum field and real-run commands, summarize changes plus collect-only evidence in analysis/summary.md, and store updated docs under plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/docs/.
