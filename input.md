Summary: Normalize Phase E CLI stdout bundle logging and capture deterministic dense/baseline training evidence with SHA256 proofs.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (Phase E real bundle evidence)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv; pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle -vv; pytest tests/study/test_dose_overlap_training.py -k training_cli -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T130500Z/phase_e_training_bundle_real_runs_exec/

Do Now:
- STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase E6 dense/baseline real-run evidence (relative bundle stdout + deterministic runs)
- Implement: tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path — assert stdout bundle lines use artifact-relative paths and SHA digests match manifest entries; adjust studies/fly64_dose_overlap/training.py::main to emit normalized paths while preserving CONFIG-001 guardrails.
- Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T130500Z/phase_e_training_bundle_real_runs_exec/red/pytest_training_cli_relative_red.log
- Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T130500Z/phase_e_training_bundle_real_runs_exec/green/pytest_training_cli_relative_green.log
- Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T130500Z/phase_e_training_bundle_real_runs_exec/green/pytest_bundle_sha_green.log
- Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T130500Z/phase_e_training_bundle_real_runs_exec/green/pytest_training_cli_suite_green.log
- Collect: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py --collect-only -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T130500Z/phase_e_training_bundle_real_runs_exec/collect/pytest_training_cli_collect.log
- Prep: if [ ! -d tmp/phase_c_f2_cli ]; then mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T130500Z/phase_e_training_bundle_real_runs_exec/prep && AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.generation --base-npz tike_outputs/fly001_reconstructed_final_prepared/fly001_reconstructed_interp_smooth_both.npz --output-root tmp/phase_c_f2_cli | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T130500Z/phase_e_training_bundle_real_runs_exec/prep/phase_c_generation.log; fi
- Prep: if [ ! -d tmp/phase_d_f2_cli ]; then mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T130500Z/phase_e_training_bundle_real_runs_exec/prep && AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.overlap --phase-c-root tmp/phase_c_f2_cli --output-root tmp/phase_d_f2_cli --doses 1000 --views dense sparse --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T130500Z/phase_e_training_bundle_real_runs_exec/prep | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T130500Z/phase_e_training_bundle_real_runs_exec/prep/phase_d_generation.log; fi
- Execute: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view dense --gridsize 2 --accelerator cpu --deterministic --num-workers 0 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T130500Z/phase_e_training_bundle_real_runs_exec/cli/dose1000_dense_gs2.log
- Execute: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view baseline --gridsize 1 --accelerator cpu --deterministic --num-workers 0 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T130500Z/phase_e_training_bundle_real_runs_exec/cli/dose1000_baseline_gs1.log
- Archive: python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/archive_phase_e_outputs.py --phase-e-root tmp/phase_e_training_gs2 --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T130500Z/phase_e_training_bundle_real_runs_exec --dose 1000 --views dense baseline
- Summarize: python - <<'PY'
from pathlib import Path

hub = Path("plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T130500Z/phase_e_training_bundle_real_runs_exec")
summary = hub / "analysis" / "summary.md"
summary.parent.mkdir(parents=True, exist_ok=True)
summary.write_text("# Phase E6 Dense/Baseline Evidence — Results\n\n"
                   "## CLI Outputs\n"
                   "- Dense log: cli/dose1000_dense_gs2.log\n"
                   "- Baseline log: cli/dose1000_baseline_gs1.log\n\n"
                   "## SHA256 Proof\n"
                   "- See analysis/bundle_checksums.txt\n\n"
                   "## Observations\n"
                   "- [ ] Replace with CLI digest observations\n"
                   "- [ ] Note any skips or regeneration steps\n\n"
                   "## Next Steps\n"
                   "- [ ] Sparse view real run\n"
                   "- [ ] Doc/test registry sync after green evidence\n")
PY

How-To Map:
- `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv` (run twice: once RED after test edit, once GREEN after CLI fix; logs saved as above).
- `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle -vv` to ensure bundle persistence contract remains intact.
- `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py -k training_cli -vv` plus collect-only variant for registry guardrail.
- Regenerate assets only if tmp roots missing (Phase C: `python -m studies.fly64_dose_overlap.generation ...`; Phase D: `python -m studies.fly64_dose_overlap.overlap ...` with `--views dense sparse`).
- Execute Phase E CLI twice (dense gs2 and baseline gs1) with deterministic knobs from test_strategy §268.
- Archive artefacts via `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/archive_phase_e_outputs.py --phase-e-root tmp/phase_e_training_gs2 --hub <hub> --dose 1000 --views dense baseline`.
- Update `analysis/summary.md` using the provided template and mark TODOs with concrete results before closing the loop.

Pitfalls To Avoid:
- Do not skip the RED run; ensure the updated test fails before adjusting CLI stdout.
- Keep `update_legacy_dict` invocation untouched in `execute_training_job`; changes belong only to stdout normalization.
- Ensure CLI commands run under CPU/deterministic settings to keep evidence reproducible.
- Do not overwrite Phase F artifacts or move bundles outside the designated hub.
- Avoid reusing stale tmp directories; regenerate Phase C/D if they are missing rather than pointing to prior report hubs.
- Maintain artifact-relative paths when copying manifests/bundles; the archive script enforces this.
- Capture all command output via `tee` into the hub; missing logs break audit trail.
- Keep PyTorch dependency assumptions intact (POLICY-001) — do not gate training behind torch-optional branches.
- If overlap CLI warns about missing sparse datasets, record the warning in prep logs and proceed.
- Do not edit `plans/.../inventory.md`; treat it as authoritative read-only evidence.

If Blocked:
- Capture failing CLI/test logs in the hub (use `red/` or `cli/failed_*.log`), update `analysis/summary.md` with the failure signature, and mark the attempt as blocked in docs/fix_plan.md with reproduction command.
- If datasets cannot regenerate (e.g., missing base NPZ), stop after logging the error, leave tmp roots untouched, and document the dependency gap.
- Should SHA mismatch arise, keep both manifest and computed digest files, flag in ledger, and halt before Phase G progression.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch runtime required; CLI runs rely on torch>=2.2 (docs/findings.md:8).
- CONFIG-001 — Preserve legacy bridge ordering in training execution helpers (docs/findings.md:10).
- DATA-001 — Regenerated NPZs must respect canonical contract (docs/findings.md:14).
- OVERSAMPLING-001 — Dense view expects K ≥ C; sparse skips should be recorded (docs/findings.md:17).

Pointers:
- docs/fix_plan.md:20 — Current status of Phase E6 evidence requirements.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:189 — Phase G readiness expectations.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:268 — Phase E6 evidence/test coverage.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T162500Z/phase_g_inventory/analysis/inventory.md:21 — Authoritative Phase C/D/E asset inventory.
- specs/ptychodus_api_spec.md:239 — wts.h5.zip persistence + SHA contract.
- docs/TESTING_GUIDE.md:101 — Training CLI selectors for Phase E evidence.

Next Up (optional):
- Sparse view Phase E run for dose=1000 once dense/baseline evidence is green.
