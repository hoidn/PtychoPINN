Summary: Tighten Phase E6 SHA parity test, capture dense/baseline deterministic runs, and record TYPE-PATH-001 evidence.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (Phase E real bundle evidence)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv; pytest tests/study/test_dose_overlap_training.py -k training_cli -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T170500Z/phase_e_training_bundle_real_runs_exec/

Do Now:
- STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase E6 dense/baseline real-run evidence
  - Implement: tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path — assert stdout SHA lines exactly match manifest `bundle_sha256`; adjust studies/fly64_dose_overlap/training.py::main output formatting only if RED exposes a mismatch.
  - Document: docs/findings.md — append TYPE-PATH-001 entry referencing this hub, summarizing the PyTorch Path normalization bug and mitigation.
  - Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T170500Z/phase_e_training_bundle_real_runs_exec/red/pytest_training_cli_sha_red.log
  - Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T170500Z/phase_e_training_bundle_real_runs_exec/green/pytest_training_cli_sha_green.log
  - Collect: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py --collect-only -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T170500Z/phase_e_training_bundle_real_runs_exec/collect/pytest_training_cli_collect.log
  - Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T170500Z/phase_e_training_bundle_real_runs_exec/green/pytest_training_cli_suite_green.log
  - Run: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view dense --gridsize 2 --accelerator cpu --deterministic --num-workers 0 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T170500Z/phase_e_training_bundle_real_runs_exec/cli/dose1000_dense_gs2.log
  - Run: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view baseline --gridsize 1 --accelerator cpu --deterministic --num-workers 0 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T170500Z/phase_e_training_bundle_real_runs_exec/cli/dose1000_baseline_gs1.log
  - Archive: python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/archive_phase_e_outputs.py --phase-e-root tmp/phase_e_training_gs2 --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T170500Z/phase_e_training_bundle_real_runs_exec --dose 1000 --views dense baseline
  - Summarize: update plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T170500Z/phase_e_training_bundle_real_runs_exec/summary.md with CLI + checksum evidence and note sparse backlog; log attempt + TYPE-PATH-001 in docs/fix_plan.md.
  - Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T170500Z/phase_e_training_bundle_real_runs_exec/

How-To Map:
- export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
- if [ ! -d tmp/phase_c_f2_cli ]; then mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T170500Z/phase_e_training_bundle_real_runs_exec/prep && AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.generation --base-npz tike_outputs/fly001_reconstructed_final_prepared/fly001_reconstructed_interp_smooth_both.npz --output-root tmp/phase_c_f2_cli | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T170500Z/phase_e_training_bundle_real_runs_exec/prep/phase_c_generation.log; fi
- if [ ! -d tmp/phase_d_f2_cli ]; then mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T170500Z/phase_e_training_bundle_real_runs_exec/prep && AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.overlap --phase-c-root tmp/phase_c_f2_cli --output-root tmp/phase_d_f2_cli --doses 1000 --views dense sparse --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T170500Z/phase_e_training_bundle_real_runs_exec/prep | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T170500Z/phase_e_training_bundle_real_runs_exec/prep/phase_d_generation.log; fi
- pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv (RED then GREEN) with logs under `red/` and `green/`
- pytest tests/study/test_dose_overlap_training.py --collect-only -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T170500Z/phase_e_training_bundle_real_runs_exec/collect/pytest_training_cli_collect.log
- pytest tests/study/test_dose_overlap_training.py -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T170500Z/phase_e_training_bundle_real_runs_exec/green/pytest_training_cli_suite_green.log
- python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view dense --gridsize 2 --accelerator cpu --deterministic --num-workers 0 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T170500Z/phase_e_training_bundle_real_runs_exec/cli/dose1000_dense_gs2.log
- python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view baseline --gridsize 1 --accelerator cpu --deterministic --num-workers 0 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T170500Z/phase_e_training_bundle_real_runs_exec/cli/dose1000_baseline_gs1.log
- python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/archive_phase_e_outputs.py --phase-e-root tmp/phase_e_training_gs2 --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T170500Z/phase_e_training_bundle_real_runs_exec --dose 1000 --views dense baseline
- Edit docs/findings.md to add TYPE-PATH-001 referencing this hub; ensure format matches existing table
- Update summary.md + docs/fix_plan.md attempt entry before finishing

Pitfalls To Avoid:
- Keep `update_legacy_dict` invocation ordering intact (CONFIG-001) when touching CLI code.
- Do not drop deterministic flags or change seeds when rerunning CLI commands.
- Avoid reintroducing absolute bundle paths or mismatched SHA strings in stdout.
- Use the archive script only; no manual `cp` into artifact hubs.
- Ensure findings table remains ASCII-aligned; preserve pipe separators.
- No environment modifications or package installs during this loop.

If Blocked:
- Capture failing command output in summary.md, note stack trace snippet, and log block in docs/fix_plan.md Attempts with reason (e.g., trainer crash, missing dataset).
- If PyTorch Path errors reoccur, stop runs, keep logs, and flag TYPE-PATH-001 regression in ledger before proceeding.

Findings Applied (Mandatory):
- POLICY-001 — Maintain PyTorch backend training with torch>=2.2 available; commands choose accelerator=cpu.
- CONFIG-001 — `update_legacy_dict` must precede runner invocation; verify tests keep bridge intact.
- DATA-001 — Phase C/D assets must stay DATA-001 compliant; reruns rely on validator pipeline.
- OVERSAMPLING-001 — Preserve K=7, gridsize handling in CLI runs; dense view must respect spacing filters.
- TYPE-PATH-001 — Normalize TrainingConfig path fields via Path(); ensure new assertions reinforce this behavior.

Pointers:
- docs/fix_plan.md:17 — Active initiative status + latest attempt details.
- specs/ptychodus_api_spec.md:220 — Backend checkpoint persistence + bundle requirements.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:268 — Phase E6 evidence definition.
- docs/findings.md:8 — POLICY-001 dependency policy reference.
- docs/findings.md:10 — CONFIG-001 legacy bridge rule.

Next Up (optional):
- Phase E6 sparse gs2 deterministic run once dense/baseline evidence lands.
