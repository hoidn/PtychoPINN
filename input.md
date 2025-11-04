Summary: Add bundle size tracking to the Phase E6 training CLI, then capture dense/baseline deterministic evidence with archival proof.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (Phase E real bundle evidence)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv; pytest tests/study/test_dose_overlap_training.py -k training_cli -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T190500Z/phase_e_training_bundle_real_runs_exec/

Do Now:
- STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase E6 dense/baseline bundle evidence
  - Implement: studies/fly64_dose_overlap/training.py::execute_training_job — compute `bundle_size_bytes` whenever a bundle is written, propagate through manifest serialization, and emit size information in stdout.
  - Implement: tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path — assert `bundle_size_bytes` exists, is an int > 0, and that stdout includes the new size line alongside bundle/SHA entries.
  - Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T190500Z/phase_e_training_bundle_real_runs_exec/red/pytest_training_cli_bundle_size_red.log
  - Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T190500Z/phase_e_training_bundle_real_runs_exec/green/pytest_training_cli_bundle_size_green.log
  - Collect: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py --collect-only -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T190500Z/phase_e_training_bundle_real_runs_exec/collect/pytest_training_cli_collect.log
  - Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T190500Z/phase_e_training_bundle_real_runs_exec/green/pytest_training_cli_suite_green.log
  - Run: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view dense --gridsize 2 --accelerator cpu --deterministic --num-workers 0 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T190500Z/phase_e_training_bundle_real_runs_exec/cli/dose1000_dense_gs2.log
  - Run: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view baseline --gridsize 1 --accelerator cpu --deterministic --num-workers 0 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T190500Z/phase_e_training_bundle_real_runs_exec/cli/dose1000_baseline_gs1.log
  - Archive: python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/archive_phase_e_outputs.py --phase-e-root tmp/phase_e_training_gs2 --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T190500Z/phase_e_training_bundle_real_runs_exec --dose 1000 --views dense baseline
  - Summarize: update plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T190500Z/phase_e_training_bundle_real_runs_exec/summary.md with manifest + checksum + size evidence; sync docs/fix_plan.md Attempts History with this hub.
  - Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T190500Z/phase_e_training_bundle_real_runs_exec/

How-To Map:
- export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
- if [ ! -d tmp/phase_c_f2_cli ]; then mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T190500Z/phase_e_training_bundle_real_runs_exec/prep && AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.generation --base-npz tike_outputs/fly001_reconstructed_final_prepared/fly001_reconstructed_interp_smooth_both.npz --output-root tmp/phase_c_f2_cli | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T190500Z/phase_e_training_bundle_real_runs_exec/prep/phase_c_generation.log; fi
- if [ ! -d tmp/phase_d_f2_cli ]; then mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T190500Z/phase_e_training_bundle_real_runs_exec/prep && AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.overlap --phase-c-root tmp/phase_c_f2_cli --output-root tmp/phase_d_f2_cli --doses 1000 --views dense sparse --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T190500Z/phase_e_training_bundle_real_runs_exec/prep | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T190500Z/phase_e_training_bundle_real_runs_exec/prep/phase_d_generation.log; fi
- pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv (RED then GREEN) with logs routed to `red/` and `green/`
- pytest tests/study/test_dose_overlap_training.py --collect-only -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T190500Z/phase_e_training_bundle_real_runs_exec/collect/pytest_training_cli_collect.log
- pytest tests/study/test_dose_overlap_training.py -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T190500Z/phase_e_training_bundle_real_runs_exec/green/pytest_training_cli_suite_green.log
- python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view dense --gridsize 2 --accelerator cpu --deterministic --num-workers 0 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T190500Z/phase_e_training_bundle_real_runs_exec/cli/dose1000_dense_gs2.log
- python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view baseline --gridsize 1 --accelerator cpu --deterministic --num-workers 0 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T190500Z/phase_e_training_bundle_real_runs_exec/cli/dose1000_baseline_gs1.log
- python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/archive_phase_e_outputs.py --phase-e-root tmp/phase_e_training_gs2 --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T190500Z/phase_e_training_bundle_real_runs_exec --dose 1000 --views dense baseline
- Update summary.md + docs/fix_plan.md with manifest, checksum, and size notes before closing the loop

Pitfalls To Avoid:
- Keep `update_legacy_dict` ordering intact when touching training helpers (CONFIG-001).
- Record `bundle_size_bytes` as an int in manifest; avoid casting to str when serializing.
- Ensure stdout size lines remain artifact-relative and appear only once per job.
- Preserve deterministic CLI flags (`--deterministic`, `--num-workers 0`) to keep evidence reproducible.
- Do not modify archive script to skip checksum comparisons; failures should surface as errors.
- Maintain findings ledger formatting (ASCII table, pipes aligned) if additional notes are added.
- Leave sparse view backlog untouched in this loop; note status in summary instead.

If Blocked:
- Capture failing command stderr/stdout into `analysis/` or `summary.md`, note dose/view affected, and log block in docs/fix_plan.md Attempts with reason.
- If bundle checksum mismatch occurs, stop further runs, keep artifacts, and flag the mismatch in summary + ledger before proceeding.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch backend must remain available; CLI runs target CPU accelerator for reproducibility.
- CONFIG-001 — Training helpers must continue to bridge params.cfg via `update_legacy_dict` before data/model access.
- DATA-001 — Phase C/D assets must stay contract-compliant; regenerate via validators if missing.
- OVERSAMPLING-001 — Dense gs2 runs must honor K≥C spacing; no manual overrides to grouping metadata.
- TYPE-PATH-001 — Path normalization fixes stay in place; new output must use Path-safe handling.

Pointers:
- docs/fix_plan.md:17 — Initiative status + attempt log.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:268 — Phase E6 evidence checklist.
- specs/ptychodus_api_spec.md:220 — Bundle persistence + SHA contract.
- docs/findings.md:8 — POLICY-001 reference.
- docs/findings.md:10 — CONFIG-001 reminder.

Next Up (optional):
- Phase E6 sparse gs2 deterministic run once dense/baseline artifacts archived.
