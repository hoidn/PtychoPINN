Summary: Capture Phase F dry-run and first LSQML execution with logging so real recon evidence is auditable.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.F2 — Phase F pty-chi baseline execution
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_executes_selected_jobs -vv; pytest tests/study/test_dose_overlap_reconstruction.py -k "ptychi" -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T180000Z/phase_f_ptychi_baseline_f2/

Do Now — STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.F2:
  - Implement: studies/fly64_dose_overlap/reconstruction.py::run_ptychi_job — extend the runner to accept a `log_root`/per-job log path, ensure non-dry-run executions stream stdout/stderr to `artifact_root/dose_{dose}/{view}/{split}/ptychi.log`, propagate non-zero return codes with actionable context, and update `main()` to pass the log root, persist execution telemetry (return codes, log paths) into the manifest, and keep dry-run behavior unchanged aside from recording the mock log location.
  - Tests: tests/study/test_dose_overlap_reconstruction.py::test_cli_executes_selected_jobs — add a RED test that patches `subprocess.run` to simulate a success + failure path, asserts the CLI (real-mode) writes per-job logs, surfaces the failing return code, and appends execution metadata to the manifest; include fixtures for temp artifact roots and verify skip summary remains untouched.
  - RED: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_executes_selected_jobs -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T180000Z/phase_f_ptychi_baseline_f2/red/pytest_phase_f_cli_exec_red.log
  - Validate: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_executes_selected_jobs -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T180000Z/phase_f_ptychi_baseline_f2/green/pytest_phase_f_cli_exec_green.log && pytest tests/study/test_dose_overlap_reconstruction.py -k "ptychi" -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T180000Z/phase_f_ptychi_baseline_f2/green/pytest_phase_f_cli_suite_green.log
  - Collect: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && pytest tests/study/test_dose_overlap_reconstruction.py --collect-only -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T180000Z/phase_f_ptychi_baseline_f2/collect/pytest_phase_f_cli_collect.log
  - CLI Evidence: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && python -m studies.fly64_dose_overlap.reconstruction --phase-c-root <phase_c_tmp> --phase-d-root <phase_d_tmp> --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T180000Z/phase_f_ptychi_baseline_f2/cli --allow-missing-phase-d --dry-run 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T180000Z/phase_f_ptychi_baseline_f2/cli/dry_run.log; export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && python -m studies.fly64_dose_overlap.reconstruction --phase-c-root <phase_c_tmp> --phase-d-root <phase_d_tmp> --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T180000Z/phase_f_ptychi_baseline_f2/real_run --dose 1000 --view dense --split train --allow-missing-phase-d 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T180000Z/phase_f_ptychi_baseline_f2/real_run/dose_1000/dense/train/run.log
  - Docs: Summarize dry-run + real-run evidence in plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T180000Z/phase_f_ptychi_baseline_f2/docs/summary.md, mark F2.1/F2.2/F2.3 progress in the plan/test_strategy, and sync `docs/TESTING_GUIDE.md` + `docs/development/TEST_SUITE_INDEX.md` with the new selector once GREEN.

Priorities & Rationale:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline_plan/plan.md:29-38 requires F2 dry-run + real LSQML execution with artifacts under the new timestamped hub.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:212-244 earmarks the non-dry-run selector and mandates capture of CLI transcripts before advancing to Phase G.
- docs/TESTING_GUIDE.md:100-142 outlines the authoritative CLI/test commands and `AUTHORITATIVE_CMDS_DOC` export we must honor.
- specs/data_contracts.md:120-214 define the NPZ layouts that the manifest validation and real run must respect.
- docs/findings.md:8-17 (POLICY-001, CONFIG-001, DATA-001, OVERSAMPLING-001) govern mandatory dependencies, params.cfg bridging boundaries, data contracts, and overlap coverage.

How-To Map:
- mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T180000Z/phase_f_ptychi_baseline_f2/{red,green,collect,cli,real_run,docs}
- Implement logging in `run_ptychi_job`: accept optional Path, ensure per-job directories exist, write stdout/stderr to `ptychi.log`, raise on non-zero codes, and return the CompletedProcess with log metadata for summary use.
- Update `main()` to compute log paths from `artifact_root`, pass them into `run_ptychi_job`, append execution records (dose/view/split/log_path/returncode) into the manifest JSON, and keep skip summary logic intact.
- Author `test_cli_executes_selected_jobs` using fixtures + `monkeypatch` for `subprocess.run`, verifying success/failure handling, log writing, manifest telemetry, and skip summary stability.
- Run RED/GREEN/collect commands with `AUTHORITATIVE_CMDS_DOC` exported; capture logs under the new hub.
- Prepare Phase C/D scratch inputs under tmp/ (reuse fixtures) before CLI invocations; preserve outputs under the initiative artifact tree.
- After GREEN, update docs/test registries per Doc Sync Plan and link artifacts inside summary.md.

Pitfalls To Avoid:
- Do not execute real LSQML runs inside pytest—keep real executions in the CLI evidence step only.
- Avoid mutating `params.cfg`; the reconstruction script handles CONFIG-001 bridging.
- Keep log files under the initiative hub; no stray outputs at repo root or /tmp after completion.
- Ensure manifest updates stay JSON-serializable (Path → str) and include log paths.
- Handle subprocess return codes defensively—surface non-zero exit codes with context and capture stderr in logs.
- Maintain deterministic job ordering when filtering so manifests remain diff-friendly.
- Respect `allow-missing-phase-d`; record skips instead of silently dropping jobs.
- Patch `subprocess.run` carefully in tests to avoid leaking mocks into other selectors.
- Ensure CLI invocations use relative tmp data that conform to DATA-001 (fixtures already provide canonical NPZs).
- Document hardware/runtime for the real run in summary.md (CPU vs GPU) without triggering package installs.

If Blocked:
- Capture failing pytest output to `.../red/pytest_phase_f_cli_exec_red.log`, stash CLI stderr in `cli/dry_run.log` or `real_run/.../run.log`, and record blocker details plus remediation plan in docs/fix_plan.md Attempts History before pausing.

Findings Applied (Mandatory):
- POLICY-001 — Assume torch>=2.2 present; no torch-optional fallbacks when invoking reconstruction.
- CONFIG-001 — Keep builder/CLI pure; note downstream bridge usage in summary.
- DATA-001 — Validate NPZ layout before dispatch; reflect any contract breakages in skip summary/logs.
- OVERSAMPLING-001 — Preserve dense/sparse coverage with neighbor_count=7; log if sparse data absent.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline_plan/plan.md:24-38
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:212-244
- docs/TESTING_GUIDE.md:100-142
- specs/data_contracts.md:120-214
- docs/findings.md:8-17

Next Up (optional):
- Once real run evidence is GREEN, expand to Phase F2.2 additional views or transition toward Phase F2.3 summary updates.

Doc Sync Plan:
- After tests pass, rerun `pytest tests/study/test_dose_overlap_reconstruction.py --collect-only -vv` (log already captured) and archive diffs under `.../docs/`.
- Update `docs/TESTING_GUIDE.md` §2 and `docs/development/TEST_SUITE_INDEX.md` with the new `test_cli_executes_selected_jobs` selector and artifact paths; reference the F2 hub.
