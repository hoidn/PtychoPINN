Summary: Turn the Phase F pty-chi manifest red test green by implementing the reconstruction job builder and runner harness.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.F1 — Phase F pty-chi job orchestrator
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_reconstruction.py::test_build_ptychi_jobs_manifest -vv; pytest tests/study/test_dose_overlap_reconstruction.py::test_run_ptychi_job_invokes_script -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T111500Z/phase_f_ptychi_baseline_f1/

Do Now — STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.F1:
  - Implement: studies/fly64_dose_overlap/reconstruction.py::build_ptychi_jobs — replace the Phase F0 stub with a ReconstructionJob dataclass, manifest builder, and run_ptychi_job helper that enumerate 3 doses × (baseline, dense, sparse) × train/test (21 jobs total), validate DATA-001 NPZ paths, and assemble CLI args for scripts/reconstruction/ptychi_reconstruct_tike.py.
  - Tests: tests/study/test_dose_overlap_reconstruction.py::test_build_ptychi_jobs_manifest — rewrite to assert manifest length, per-dose view/split coverage, artifact_dir layout, and CLI argument payload; add test_run_ptychi_job_invokes_script using unittest.mock to confirm run_ptychi_job dispatches subprocess with CONFIG-001-safe environment; capture the pre-implementation RED failure via `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` & pytest `-k ptychi` to plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T111500Z/phase_f_ptychi_baseline_f1/red/pytest_phase_f_red.log.
  - Validate: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && pytest tests/study/test_dose_overlap_reconstruction.py -k "ptychi" -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T111500Z/phase_f_ptychi_baseline_f1/green/pytest_phase_f_green.log
  - Collect: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && pytest tests/study/test_dose_overlap_reconstruction.py --collect-only -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T111500Z/phase_f_ptychi_baseline_f1/collect/pytest_phase_f_collect.log
  - Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T111500Z/phase_f_ptychi_baseline_f1/{red/,green/,collect/,docs/summary.md,cli/}

Priorities & Rationale:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline_plan/plan.md:18-27 mandates F1.1–F1.3 builder + CLI deliverables with GREEN pytest evidence.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:216-272 documents Phase F selectors, RED→GREEN expectations, and artifact requirements that must be satisfied before advancing to F2.
- docs/TESTING_GUIDE.md:101-140 provides the authoritative pytest/CLI invocation pattern we must mirror and cite in commands.
- specs/data_contracts.md:120-214 enumerates DATA-001 NPZ field/dtype requirements; builder assertions must respect these layouts.
- docs/findings.md:8-17 (POLICY-001, CONFIG-001, DATA-001, OVERSAMPLING-001) remain active guardrails for reconstruction workflows.

How-To Map:
- mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T111500Z/phase_f_ptychi_baseline_f1/{red,green,collect,cli,docs}
- Edit studies/fly64_dose_overlap/reconstruction.py to add @dataclass ReconstructionJob, helper enums, build_ptychi_jobs manifest logic (Path.exists validation, deterministic artifact dirs), and run_ptychi_job that shells out via subprocess.run with dry-run toggle.
- Update tests/study/test_dose_overlap_reconstruction.py fixtures to reuse Phase C/D builders, assert 21 ReconstructionJob entries, inspect CLI args (script path, --algorithm LSQML, --num-epochs 100, --input-npz/--output-dir), and mock subprocess for run_ptychi_job coverage.
- Export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md before every pytest invocation per governance.
- Capture RED failure (before implementation) and GREEN pass logs under the new artifact hub; regenerate collect-only proof post-implementation.
- Summarize outcomes and lingering gaps in plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T111500Z/phase_f_ptychi_baseline_f1/docs/summary.md.

Pitfalls To Avoid:
- Do not mutate params.cfg or call update_legacy_dict inside the builder; CONFIG-001 bridge stays in the eventual CLI runner.
- Keep ReconstructionJob fields serializable (Path/String) and avoid embedding numpy arrays in manifest entries.
- Ensure artifact directories stay inside artifact_root; no absolute tmp paths or user home leakage.
- Avoid importing heavyweight pty-chi dependencies in tests—use static CLI argument assertions instead.
- Preserve deterministic ordering: iterate doses ascending, views baseline→dense→sparse, splits train then test.
- Validate NPZ path existence using Path.exists; fail fast with clear FileNotFoundError when allow_missing is False.
- Mock subprocess.run in tests to avoid executing the real reconstruction script.
- Remember to seed numpy RNG in fixtures if randomness could make asserts unstable.
- Keep new CLI helper dry-run friendly; do not actually execute LSQML during tests.

If Blocked:
- Capture failing pytest output to plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T111500Z/phase_f_ptychi_baseline_f1/red/pytest_phase_f_red.log, document blocker + stack trace in docs/summary.md, and update docs/fix_plan.md Attempts History with the issue before pausing.

Findings Applied (Mandatory):
- POLICY-001 — Document PyTorch dependency in summary.md and ensure CLI args respect torch>=2.2 expectation.
- CONFIG-001 — Builder stays pure; run_ptychi_job defers bridge until actual execution layer.
- DATA-001 — Validate NPZ layout/paths match Phase C `patched_*` and Phase D `{view}_{split}` outputs.
- OVERSAMPLING-001 — Preserve gs2 jobs only for views with valid spacing; ensure manifest inherits neighbor_count assumptions.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline_plan/plan.md:18
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:216
- docs/findings.md:8
- docs/TESTING_GUIDE.md:101
- specs/data_contracts.md:120

Next Up (optional):
- Implement Phase F1.3 CLI entrypoint and manifest emission once builder + runner tests pass.

Doc Sync Plan (Conditional):
- After GREEN tests land, rerun `pytest tests/study/test_dose_overlap_reconstruction.py --collect-only -vv` (log to collect/).
- Update docs/TESTING_GUIDE.md §2 with new `pytest ... -k ptychi` selector and add Phase F row to docs/development/TEST_SUITE_INDEX.md once evidence captured; archive diffs under docs/ subdir in this loop’s artifact hub.
