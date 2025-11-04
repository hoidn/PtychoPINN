Summary: Capture deterministic Phase E6 dense/baseline runs at dose=1000, extend the archive helper to enforce bundle size parity, and record SHA+size evidence in this hub.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (Phase E real bundle evidence)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T210500Z/phase_e_training_bundle_real_runs_exec/

Do Now:
- STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase E6 dense/baseline bundle evidence
  - Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/archive_phase_e_outputs.py::archive_bundles — compare manifest `bundle_size_bytes` against filesystem size, fail on mismatch, and emit `sha256  filename  size_bytes` rows to `analysis/bundle_checksums.txt`.
  - Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T210500Z/phase_e_training_bundle_real_runs_exec/green/pytest_training_cli_bundle_size_green.log
  - Run: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view dense --gridsize 2 --accelerator cpu --deterministic --num-workers 0 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T210500Z/phase_e_training_bundle_real_runs_exec/cli/dose1000_dense_gs2.log
  - Run: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view baseline --gridsize 1 --accelerator cpu --deterministic --num-workers 0 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T210500Z/phase_e_training_bundle_real_runs_exec/cli/dose1000_baseline_gs1.log
  - Archive: python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/archive_phase_e_outputs.py --phase-e-root tmp/phase_e_training_gs2 --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T210500Z/phase_e_training_bundle_real_runs_exec --dose 1000 --views dense baseline | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T210500Z/phase_e_training_bundle_real_runs_exec/analysis/archive_phase_e.log
  - Summarize: Update plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T210500Z/phase_e_training_bundle_real_runs_exec/summary.md with CLI commands, bundle SHA+size table, and any anomalies; sync docs/fix_plan.md Attempts History for this loop.
  - Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T210500Z/phase_e_training_bundle_real_runs_exec/

How-To Map:
- export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
- if [ ! -d tmp/phase_c_f2_cli ]; then mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T210500Z/phase_e_training_bundle_real_runs_exec/prep && AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.generation --base-npz tike_outputs/fly001_reconstructed_final_prepared/fly001_reconstructed_interp_smooth_both.npz --output-root tmp/phase_c_f2_cli | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T210500Z/phase_e_training_bundle_real_runs_exec/prep/phase_c_generation.log; fi
- if [ ! -d tmp/phase_d_f2_cli ]; then mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T210500Z/phase_e_training_bundle_real_runs_exec/prep && AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.overlap --phase-c-root tmp/phase_c_f2_cli --output-root tmp/phase_d_f2_cli --doses 1000 --views dense sparse --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T210500Z/phase_e_training_bundle_real_runs_exec/prep | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T210500Z/phase_e_training_bundle_real_runs_exec/prep/phase_d_generation.log; fi
- pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T210500Z/phase_e_training_bundle_real_runs_exec/green/pytest_training_cli_bundle_size_green.log
- python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view dense --gridsize 2 --accelerator cpu --deterministic --num-workers 0 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T210500Z/phase_e_training_bundle_real_runs_exec/cli/dose1000_dense_gs2.log
- python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view baseline --gridsize 1 --accelerator cpu --deterministic --num-workers 0 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T210500Z/phase_e_training_bundle_real_runs_exec/cli/dose1000_baseline_gs1.log
- python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/archive_phase_e_outputs.py --phase-e-root tmp/phase_e_training_gs2 --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T210500Z/phase_e_training_bundle_real_runs_exec --dose 1000 --views dense baseline | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T210500Z/phase_e_training_bundle_real_runs_exec/analysis/archive_phase_e.log
- Update summary.md and docs/fix_plan.md once evidence is captured.

Pitfalls To Avoid:
- Ensure `archive_phase_e_outputs.py` fails fast when manifest size mismatches filesystem size; do not log-and-continue.
- Preserve SHA validation logic; append size checks without muting checksum comparisons.
- Keep CLI runs deterministic (`--deterministic`, `--num-workers 0`) and CPU-bound for reproducibility.
- Maintain artifact-relative stdout paths; avoid absolute paths when printing bundle info.
- Do not overwrite prior hubs; keep all new outputs under 2025-11-06T210500Z.
- Leave sparse view backlog untouched in this iteration; document status instead.

If Blocked:
- Capture failing command output under `analysis/` (e.g., `analysis/archive_phase_e_failure.log`), mark Attempt as partial in docs/fix_plan.md, and flag the blocker in summary.md before exiting the loop.

Findings Applied (Mandatory):
- POLICY-001 — Maintain PyTorch availability while running training CLI on CPU; no backend downgrades.
- CONFIG-001 — Ensure CLI invocations continue bridging `params.cfg` via `update_legacy_dict` (confirm logs mention bridge success).
- DATA-001 — Verify regenerated Phase C/D assets with existing validators before reruns if directories are missing.
- OVERSAMPLING-001 — Leave grouping metadata untouched; dense gs2 run must retain K≥C spacing guarantees.
- TYPE-PATH-001 — Use `Path` when handling filesystem operations in the archive helper to avoid string path regressions.

Pointers:
- docs/fix_plan.md:27 — Current initiative attempts & notes.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:268 — Evidence checklist for Phase E6.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/archive_phase_e_outputs.py:56 — Helper function entry point to extend.
- specs/ptychodus_api_spec.md:220 — Bundle manifest requirements.
- docs/findings.md:21 — TYPE-PATH-001 context.

Next Up (optional):
- Sparse gs2 deterministic run and archival once dense/baseline evidence is complete.
