Summary: Capture deterministic Phase E5 training evidence with skip reporting and sync study documentation.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.E5 — training runner integration
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_training.py::test_training_cli_manifest_and_bridging -vv; pytest tests/study/test_dose_overlap_training.py::test_build_training_jobs_skips_missing_view -vv; pytest tests/study/test_dose_overlap_training.py -k training_cli -vv; pytest tests/study/test_dose_overlap_training.py --collect-only -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T161500Z/phase_e_training_e5_real_run/

Do Now — STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.E5:
  - Test: Expand `tests/study/test_dose_overlap_training.py::test_training_cli_manifest_and_bridging` to drop the sparse Phase D NPZ and assert the manifest exposes a `skipped_views` entry (dose/view/reason). Run `pytest tests/study/test_dose_overlap_training.py::test_training_cli_manifest_and_bridging -vv` before implementation; tee the expected failure to `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T161500Z/phase_e_training_e5_real_run/red/pytest_training_cli_manifest_red.log`.
  - Implement: `studies/fly64_dose_overlap/training.py::build_training_jobs` — accumulate skip metadata when `allow_missing_phase_d` bypasses an overlap job (append dicts to an optional `skip_events` list) and adjust `main()` to pass that list, print a summary, and embed `skipped_views` + `skipped_count` in `training_manifest.json` so CLI evidence captures missing sparse views.
  - Validate: After the code change, rerun `pytest tests/study/test_dose_overlap_training.py::test_training_cli_manifest_and_bridging -vv` and `pytest tests/study/test_dose_overlap_training.py::test_build_training_jobs_skips_missing_view -vv`, teeing outputs to `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T161500Z/phase_e_training_e5_real_run/green/`. Re-run `pytest tests/study/test_dose_overlap_training.py -k training_cli -vv` to exercise the CLI suite and collect proof with `pytest tests/study/test_dose_overlap_training.py --collect-only -vv`.
  - Run: Move any existing root-level `train_debug.log` into `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T161500Z/phase_e_training_e5_real_run/real_run/train_debug_prepath_fix.log`, regenerate Phase C (`python -m studies.fly64_dose_overlap.generation ...`) and Phase D (`python -m studies.fly64_dose_overlap.overlap ...`) artifacts, then execute `python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_training_evidence --phase-d-root tmp/phase_d_training_evidence --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T161500Z/phase_e_training_e5_real_run/real_run --dose 1000 --view baseline --gridsize 1 --accelerator cpu --deterministic --num-workers 0 --logger csv`; archive stdout, Lightning logs, and the new manifest (with `skipped_views`) under `real_run/`.
  - Doc: Update `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T161500Z/phase_e_training_e5_real_run/docs/summary.md` with skip-manifest results + real-run metrics, flip Phase E5 checklist/test_strategy rows, sync `docs/TESTING_GUIDE.md` §2 and `docs/development/TEST_SUITE_INDEX.md` with the new selector, and record Attempt #23 in `docs/fix_plan.md` referencing the 2025-11-04T161500Z artifact hub.

Priorities & Rationale:
- docs/fix_plan.md:31 keeps Phase E5 in-progress until real-run evidence and doc sync land.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T053500Z/phase_e_training_plan/plan.md:15 lists manifest/log deliverables for E5 completion.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:163 requires deterministic CLI execution evidence before closing the phase.
- docs/TESTING_GUIDE.md:86 highlights Phase E selectors that must reflect the new skip-aware test.
- specs/data_contracts.md:190 enforces canonical NPZ layout when regenerating datasets for the real run.

How-To Map:
- export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
- mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T161500Z/phase_e_training_e5_real_run/{red,green,collect,docs,real_run}
- pytest tests/study/test_dose_overlap_training.py::test_training_cli_manifest_and_bridging -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T161500Z/phase_e_training_e5_real_run/red/pytest_training_cli_manifest_red.log
- pytest tests/study/test_dose_overlap_training.py::test_training_cli_manifest_and_bridging -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T161500Z/phase_e_training_e5_real_run/green/pytest_training_cli_manifest_green.log
- pytest tests/study/test_dose_overlap_training.py::test_build_training_jobs_skips_missing_view -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T161500Z/phase_e_training_e5_real_run/green/pytest_training_cli_skips_green.log
- pytest tests/study/test_dose_overlap_training.py -k training_cli -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T161500Z/phase_e_training_e5_real_run/green/pytest_training_cli_suite_green.log
- pytest tests/study/test_dose_overlap_training.py --collect-only -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T161500Z/phase_e_training_e5_real_run/collect/pytest_collect.log
- if [ -f train_debug.log ]; then mv train_debug.log plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T161500Z/phase_e_training_e5_real_run/real_run/train_debug_prepath_fix.log; fi
- python -m studies.fly64_dose_overlap.generation --base-npz datasets/fly/fly001_transposed.npz --output-root tmp/phase_c_training_evidence 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T161500Z/phase_e_training_e5_real_run/real_run/phase_c_generation.log
- python -m studies.fly64_dose_overlap.overlap --phase-c-root tmp/phase_c_training_evidence --output-root tmp/phase_d_training_evidence --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T161500Z/phase_e_training_e5_real_run/real_run/overlap_cli 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T161500Z/phase_e_training_e5_real_run/real_run/phase_d_overlap.log
- python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_training_evidence --phase-d-root tmp/phase_d_training_evidence --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T161500Z/phase_e_training_e5_real_run/real_run --dose 1000 --view baseline --gridsize 1 --accelerator cpu --deterministic --num-workers 0 --logger csv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T161500Z/phase_e_training_e5_real_run/real_run/training_cli_real_run.log
- jq '.skipped_views' plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T161500Z/phase_e_training_e5_real_run/real_run/training_manifest.json > plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T161500Z/phase_e_training_e5_real_run/docs/skipped_views.jsonl
- rm -rf tmp/phase_c_training_evidence tmp/phase_d_training_evidence

Pitfalls To Avoid:
- Keep `allow_missing_phase_d` default strict so tests still catch missing overlaps unless explicitly enabled.
- Ensure new manifest fields are JSON-serializable (lists/dicts, not Paths or sets).
- Do not leave `train_debug.log` at repository root after relocating artifacts.
- Avoid mutating `params.cfg` in builder/manifest code; CONFIG-001 bridge belongs in the execution helper.
- Maintain deterministic CLI flags (`--accelerator cpu`, `--deterministic`, `--num-workers 0`) during the real run.
- Capture RED failure logs before implementation to preserve TDD ordering.
- Do not downgrade overlap spacing thresholds to force sparse view success; missing data must be skipped and logged.
- Keep artifact outputs inside the 2025-11-04T161500Z hub; clean `tmp/` directories after evidence is archived.

If Blocked:
- Archive failing pytest/CLI logs under `red/` or `real_run/`, summarize the blocker in `docs/summary.md`, and log a blocked Attempt #23 entry in docs/fix_plan.md before stopping.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch backend remains mandatory; ensure the CLI continues calling the PyTorch runner while emitting manifest metadata.
- CONFIG-001 — Leave dataset enumeration pure and rely on `run_training_job`/`execute_training_job` for the legacy bridge sequencing.
- DATA-001 — Phase C/D loaders and fixtures must keep canonical NPZ keys/dtypes when regenerating evidence.
- OVERSAMPLING-001 — Gridsize=2 overlap jobs must retain neighbor count assumptions even when skipped.

Pointers:
- docs/fix_plan.md:31
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T053500Z/phase_e_training_plan/plan.md:15
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:163
- docs/TESTING_GUIDE.md:86
- specs/data_contracts.md:190

Next Up (optional):
- Extend real-run evidence to the dense gs2 job once baseline passes and manifest skip reporting is validated.

Doc Sync Plan (Conditional):
- After GREEN tests, rerun `pytest tests/study/test_dose_overlap_training.py --collect-only -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T161500Z/phase_e_training_e5_real_run/collect/pytest_collect_final.log`, then update `docs/TESTING_GUIDE.md` §2 and `docs/development/TEST_SUITE_INDEX.md` with the final selector text once code passes.
