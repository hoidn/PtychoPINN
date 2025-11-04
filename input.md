Summary: Persist skip summary artifacts and capture deterministic Phase E5 training evidence so we can close the runner integration.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.E5 — training runner integration
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_training.py::test_training_cli_manifest_and_bridging -vv; pytest tests/study/test_dose_overlap_training.py::test_build_training_jobs_skips_missing_view -vv; pytest tests/study/test_dose_overlap_training.py -k training_cli -vv; pytest tests/study/test_dose_overlap_training.py --collect-only -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/

Do Now — STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.E5:
  - Test: Update `tests/study/test_dose_overlap_training.py::test_training_cli_manifest_and_bridging` to expect a new `skip_summary.json` file under the CLI artifact root (delete any prior copy first), then run `pytest tests/study/test_dose_overlap_training.py::test_training_cli_manifest_and_bridging -vv` to capture the RED failure into `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/red/pytest_training_cli_manifest_red.log`.
  - Implement: `studies/fly64_dose_overlap/training.py::main` — write `skip_summary.json` beside `training_manifest.json`, populate it with `skipped_views`, and include the relative path in the manifest so downstream tooling can consume it.
  - Validate: Re-run `pytest tests/study/test_dose_overlap_training.py::test_training_cli_manifest_and_bridging -vv`, `pytest tests/study/test_dose_overlap_training.py::test_build_training_jobs_skips_missing_view -vv`, and `pytest tests/study/test_dose_overlap_training.py -k training_cli -vv`, teeing outputs into `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/green/` plus `pytest tests/study/test_dose_overlap_training.py --collect-only -vv` into `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/collect/pytest_collect.log`.
  - Run: Generate fresh Phase C/D artifacts if needed and execute `python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_training_evidence --phase-d-root tmp/phase_d_training_evidence --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/real_run --dose 1000 --view baseline --gridsize 1 --accelerator cpu --deterministic --num-workers 0 --logger csv`, archiving stdout, manifest, skip summary, and Lightning logs under the new hub.
  - Doc: Summarize outcomes in `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/docs/summary.md`, update Phase E5 rows in `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md` and `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md`, sync `docs/TESTING_GUIDE.md` §2 and `docs/development/TEST_SUITE_INDEX.md`, and record Attempt #24 with artifact links in `docs/fix_plan.md`.

Priorities & Rationale:
- `docs/fix_plan.md:31` keeps Phase E5 open until skip summary persistence and deterministic evidence exist.
- `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/plan.md:14` enumerates T1–T5 tasks you must execute this loop.
- `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:130` lists Phase E selectors and calls for deterministic CLI evidence before closure.
- `docs/TESTING_GUIDE.md:86` maps required study selectors; update keeps guidance aligned with new skip summary expectation.
- `specs/data_contracts.md:190` mandates canonical NPZ layout when regenerating Phase C/D inputs for the real run.

How-To Map:
- export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
- rm -f plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/real_run/skip_summary.json
- pytest tests/study/test_dose_overlap_training.py::test_training_cli_manifest_and_bridging -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/red/pytest_training_cli_manifest_red.log
- pytest tests/study/test_dose_overlap_training.py::test_training_cli_manifest_and_bridging -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/green/pytest_training_cli_manifest_green.log
- pytest tests/study/test_dose_overlap_training.py::test_build_training_jobs_skips_missing_view -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/green/pytest_training_cli_skips_green.log
- pytest tests/study/test_dose_overlap_training.py -k training_cli -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/green/pytest_training_cli_suite_green.log
- pytest tests/study/test_dose_overlap_training.py --collect-only -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/collect/pytest_collect.log
- python -m studies.fly64_dose_overlap.generation --base-npz datasets/fly/fly001_transposed.npz --output-root tmp/phase_c_training_evidence 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/real_run/phase_c_generation.log
- python -m studies.fly64_dose_overlap.overlap --phase-c-root tmp/phase_c_training_evidence --output-root tmp/phase_d_training_evidence --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/cli/overlap 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/real_run/phase_d_overlap.log
- python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_training_evidence --phase-d-root tmp/phase_d_training_evidence --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/real_run --dose 1000 --view baseline --gridsize 1 --accelerator cpu --deterministic --num-workers 0 --logger csv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/real_run/training_cli_real_run.log
- jq '.' plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/real_run/skip_summary.json > plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/docs/skip_summary_pretty.json
- rm -rf tmp/phase_c_training_evidence tmp/phase_d_training_evidence

Pitfalls To Avoid:
- Preserve CONFIG-001 ordering: builder stays pure; only `run_training_job` touches `params.cfg`.
- Ensure `skip_summary.json` content mirrors `manifest['skipped_views']` exactly; no divergent schemas.
- Keep artifact outputs inside the 2025-11-04T170500Z hub; do not leave logs in repo root or tmp/.
- Use deterministic CLI flags (`--accelerator cpu`, `--deterministic`, `--num-workers 0`) to avoid nondeterministic evidence.
- Maintain canonical NPZ keys/dtypes when regenerating Phase C/D data (DATA-001 compliance).
- Capture RED logs before implementing so the TDD trail is intact.
- Avoid relaxing spacing thresholds; sparse view absence is expected and must stay documented.
- Verify jq output after real run to catch malformed skip summary JSON early.
- Clean up tmp directories even on failure to prevent stale data leaks between runs.
- Do not modify core `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py` per stability directive.

If Blocked:
- Store failing pytest or CLI logs under the hub’s `red/` or `real_run/`, note the blocker in `docs/summary.md`, and record a Blocked Attempt #24 entry in `docs/fix_plan.md` before stopping.

Findings Applied (Mandatory):
- CONFIG-001 — Ensure legacy bridge remains in execution helpers; builder stays pure even while writing skip summaries.
- DATA-001 — Regenerated NPZ files must retain canonical schema to keep training reproducible.
- POLICY-001 — Keep PyTorch trainer wiring intact during CLI execution; do not fall back to TensorFlow runner.
- OVERSAMPLING-001 — Document that sparse view skips stem from spacing threshold enforcement; never reduce neighbor count to avoid skips.

Pointers:
- docs/fix_plan.md:31 (Phase E5 status & exit gap)
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/plan.md:14 (Task checklist T1–T5)
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:130 (Phase E selector expectations)
- docs/TESTING_GUIDE.md:86 (Study selector registry row)
- specs/data_contracts.md:190 (NPZ key/dtype contract for regeneration)

Next Up (optional):
- After baseline run lands, extend CLI execution to dense gs2 jobs for additional parity evidence.

Doc Sync Plan (Conditional):
- After GREEN tests pass, run `pytest tests/study/test_dose_overlap_training.py --collect-only -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/collect/pytest_collect_final.log`, then update `docs/TESTING_GUIDE.md` §2 and `docs/development/TEST_SUITE_INDEX.md` to mention the skip summary requirement, citing the new artifact hub.
