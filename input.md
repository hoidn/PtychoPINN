Summary: TDD the Phase E training CLI so job filtering, CONFIG-001 bridging, and manifest logging are verified before running real training.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.E4 — training CLI
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_training.py::test_training_cli_filters_jobs -vv; pytest tests/study/test_dose_overlap_training.py::test_training_cli_manifest_and_bridging -vv; pytest tests/study/test_dose_overlap_training.py -k training_cli -vv; pytest tests/study/test_dose_overlap_training.py --collect-only -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T081500Z/phase_e_training_cli/

Do Now — STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.E4:
  - Test: Extend `tests/study/test_dose_overlap_training.py` to tighten `test_run_training_job_invokes_runner` (assert `update_legacy_dict` receives a `TrainingConfig`) and add CLI coverage via `test_training_cli_filters_jobs` + `test_training_cli_manifest_and_bridging`; capture RED logs under `.../red/pytest_training_cli_filters_red.log` and `.../red/pytest_training_cli_manifest_red.log`.
  - Implement: studies/fly64_dose_overlap/training.py::main — parse CLI flags (`--phase-c-root`, `--phase-d-root`, `--artifact-root`, optional `--dose/--view/--gridsize`, `--dry-run`), build/ filter jobs with `build_training_jobs()`, upgrade `run_training_job()` to construct `TrainingConfig`/`ModelConfig`, call `update_legacy_dict(params.cfg, config)`, invoke the injected runner, and emit `training_manifest.json` plus CLI log under the artifact root.
  - Validate: Run `pytest tests/study/test_dose_overlap_training.py -k training_cli -vv` (tee to `.../green/pytest_training_cli_green.log`) and `pytest tests/study/test_dose_overlap_training.py --collect-only -vv` (tee to `.../collect/pytest_collect.log`); execute a CLI dry-run (`python -m studies.fly64_dose_overlap.training ... --dry-run`) writing stdout to `.../dry_run/training_cli_dry_run.txt` and ensure manifest/logs land under the targeted artifact hub.
  - Doc: Flip plan row E4 to `[x]`, mark new selectors Active in test_strategy.md, append selectors + evidence to docs/TESTING_GUIDE.md and docs/development/TEST_SUITE_INDEX.md, and summarize Attempt #15 in `.../docs/summary.md` with log/manifest pointers.

Priorities & Rationale:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T053500Z/phase_e_training_plan/plan.md:15-22 keeps E4 `[P]` pending and calls for CLI + CONFIG-001 bridge hardening.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:76-115 enumerates planned CLI selectors, bridging expectations, and artifact requirements.
- docs/DEVELOPER_GUIDE.md:68-104 mandates `update_legacy_dict(params.cfg, config)` before any legacy loader/model usage.
- specs/data_contracts.md:190-260 defines dataset structure, ensuring CLI/job wiring never violates DATA-001 when passing NPZ paths.
- docs/findings.md#CONFIG-001/#DATA-001/#OVERSAMPLING-001/#POLICY-001 capture the guardrails we must continue to satisfy while scripting the CLI.

How-To Map:
- export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
- mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T081500Z/phase_e_training_cli/{red,green,collect,docs,dry_run,cli,artifacts}
- pytest tests/study/test_dose_overlap_training.py::test_training_cli_filters_jobs -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T081500Z/phase_e_training_cli/red/pytest_training_cli_filters_red.log
- pytest tests/study/test_dose_overlap_training.py::test_training_cli_manifest_and_bridging -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T081500Z/phase_e_training_cli/red/pytest_training_cli_manifest_red.log
- pytest tests/study/test_dose_overlap_training.py -k training_cli -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T081500Z/phase_e_training_cli/green/pytest_training_cli_green.log
- pytest tests/study/test_dose_overlap_training.py --collect-only -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T081500Z/phase_e_training_cli/collect/pytest_collect.log
- python -m studies.fly64_dose_overlap.training --phase-c-root <phase_c_root> --phase-d-root <phase_d_root> --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T081500Z/phase_e_training_cli/artifacts --dose 1e3 --view baseline --gridsize 1 --dry-run 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T081500Z/phase_e_training_cli/dry_run/training_cli_dry_run.txt
- python -m studies.fly64_dose_overlap.training --phase-c-root <phase_c_root> --phase-d-root <phase_d_root> --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T081500Z/phase_e_training_cli/artifacts --dose 1e3 --view baseline --gridsize 1 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T081500Z/phase_e_training_cli/cli/training_cli_stdout.log

Pitfalls To Avoid:
- Do not regress CONFIG-001: ensure `update_legacy_dict` is called exactly once per job before any runner invocation.
- Keep CLI pure of real training side effects in tests; rely on injected stub runners and tmp_path fixtures.
- Ensure artifact directories/logs are created under the provided `--artifact-root` without writing elsewhere.
- Handle dose/view filters gracefully—fail fast with informative errors when filters match nothing.
- Maintain ASCII-only logs and manifests for diffability.
- Avoid global state leakage from tests (reset monkeypatches, params.cfg) to prevent cross-test flakiness.
- Honor POLICY-001: do not add torch-optional paths or downgrade dependency expectations.
- Ensure new tests collect (>0) and document selectors in registries after GREEN.
- Keep CLI arguments backward compatible with study scripts—no breaking renames without plan updates.
- Surface runner exceptions rather than swallowing them; tests should assert propagation.

If Blocked:
- Capture failing selector output in the artifact hub, note the blocker in summary.md, and log the condition as Attempt #15 in docs/fix_plan.md before halting.

Findings Applied (Mandatory):
- CONFIG-001 — enforce legacy bridge via `update_legacy_dict` prior to runner execution.
- DATA-001 — validate Phase C/D NPZ paths and mimic canonical filenames in tests.
- OVERSAMPLING-001 — preserve gridsize semantics when filtering jobs (baseline=1, dense/sparse=2).
- POLICY-001 — retain PyTorch dependency assumptions when wiring the CLI and tests.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T053500Z/phase_e_training_plan/plan.md:15
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:76
- studies/fly64_dose_overlap/training.py:1
- docs/DEVELOPER_GUIDE.md:68
- specs/data_contracts.md:190

Next Up (optional):
- STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.E5 — first deterministic training run once CLI ships and evidence is captured.

Doc Sync Plan:
- After GREEN, append new selectors/artifact paths to `docs/TESTING_GUIDE.md` §Study Tests and `docs/development/TEST_SUITE_INDEX.md`, and archive the updated `--collect-only` log under `.../collect/pytest_collect.log` per hard gate.
