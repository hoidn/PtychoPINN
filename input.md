Summary: Deliver Phase F1 CLI entry for pty-chi orchestrator with RED→GREEN TDD coverage and CLI dry-run evidence.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.F1 — Phase F pty-chi job orchestrator
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_reconstruction.py -k "ptychi" -vv; pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_filters_dry_run -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T130000Z/phase_f_ptychi_baseline_f1_cli/

Do Now — STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.F1:
  - Implement: studies/fly64_dose_overlap/reconstruction.py::main — add a click-style CLI (argparse is fine) that wires `build_ptychi_jobs` + `run_ptychi_job`, supports `--dose`, `--view`, `--split`, `--gridsize`, `--dry-run`, and `--allow-missing-phase-d`, and emits manifest + skip summary JSON to the artifact root (default `reports/.../cli/`); ensure CLI always calls `build_ptychi_jobs(..., allow_missing=not strict)` and preserves deterministic ordering.
  - Tests: tests/study/test_dose_overlap_reconstruction.py::test_cli_filters_dry_run — author RED test that constructs minimal Phase C/D fixtures, invokes CLI via `script_runner`/`subprocess` in `--dry-run` mode, asserts filtered manifest count (e.g., `--dose 1000 --view dense --split train` yields 1 job) and verifies output manifest + skip summary written beneath tmp artifact dir; add companion test ensuring CLI invokes `run_ptychi_job` once per selected job (mock subprocess + patch allow-missing) and logs skip metadata.
  - RED: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && pytest tests/study/test_dose_overlap_reconstruction.py -k "ptychi" -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T130000Z/phase_f_ptychi_baseline_f1_cli/red/pytest_phase_f_cli_red.log
  - Validate: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && pytest tests/study/test_dose_overlap_reconstruction.py -k "ptychi" -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T130000Z/phase_f_ptychi_baseline_f1_cli/green/pytest_phase_f_cli_green.log
  - Collect: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && pytest tests/study/test_dose_overlap_reconstruction.py --collect-only -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T130000Z/phase_f_ptychi_baseline_f1_cli/collect/pytest_phase_f_cli_collect.log
  - CLI Evidence: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && python -m studies.fly64_dose_overlap.reconstruction --phase-c-root <tmp_phase_c> --phase-d-root <tmp_phase_d> --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T130000Z/phase_f_ptychi_baseline_f1_cli/cli --dose 1000 --view dense --dry-run --allow-missing-phase-d 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T130000Z/phase_f_ptychi_baseline_f1_cli/cli/dry_run.log
  - Docs: Update plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T130000Z/phase_f_ptychi_baseline_f1_cli/docs/summary.md with manifest counts, CLI command, skip metadata, and link to RED/GREEN logs; refresh `test_strategy.md` selectors (mark CLI tests active) and `docs/development/TEST_SUITE_INDEX.md` + `docs/TESTING_GUIDE.md` entries per Doc Sync Plan.

Priorities & Rationale:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline_plan/plan.md:18-27 mandates F1.3 CLI entry, including filters, dry-run, and artifact capture under the new hub.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:212-243 marks CLI selectors as planned and requires GREEN evidence + dry-run transcript before Phase F2.
- docs/TESTING_GUIDE.md:101-140 prescribes PyTorch workflow CLI patterns and the `AUTHORITATIVE_CMDS_DOC` export we must honor for every invocation.
- specs/data_contracts.md:120-214 defines the NPZ structure (`patched_{split}.npz`, `{view}_{split}.npz`) that the CLI must validate when enumerating jobs.
- docs/findings.md:8-17 (POLICY-001, CONFIG-001, DATA-001, OVERSAMPLING-001) enforce dependency, bridge, and overlap spacing invariants the CLI must respect.

How-To Map:
- mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T130000Z/phase_f_ptychi_baseline_f1_cli/{red,green,collect,cli,docs}
- Implement CLI in `studies/fly64_dose_overlap/reconstruction.py::main`: parse options, hydrate `phase_c_root`, `phase_d_root`, `artifact_root`, call `build_ptychi_jobs`, filter by dose/view/split/gridsize, log skip metadata, and loop over jobs via `run_ptychi_job(dry_run=flag)`.
- Within tests, reuse Phase C/D fixtures to supply NPZs, invoke CLI via `subprocess.run([sys.executable, "-m", ...])` with temp artifact root, assert manifest JSON contents, skip summary, and `run_ptychi_job` call count using `patch`.
- Before every pytest/CLI invocation, `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`.
- Record failing RED run to red/pytest_phase_f_cli_red.log, then rerun after implementation for GREEN log.
- Run CLI dry-run command to populate `cli/dry_run.log`, manifest JSON, and skip summary under artifact hub; include path references in summary.md.
- Update `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md` Phase F section (mark CLI selectors active) and sync doc registries (Doc Sync Plan).

Pitfalls To Avoid:
- Do not mutate `params.cfg` inside CLI; keep CONFIG-001 bridge deferred to reconstruction script.
- Ensure CLI gracefully handles missing Phase D datasets when `--allow-missing-phase-d` is set; fail fast otherwise.
- Keep manifest outputs serializable (json) and avoid embedding numpy arrays.
- Preserve deterministic ordering when filters applied (dose asc, view baseline→dense→sparse, split train→test).
- Use tmp artifact directories inside the initiative hub—no stray files at repo root.
- Mock `run_ptychi_job` in tests to prevent real subprocess execution; assert call args for correctness.
- Respect Phase D sparse gaps by recording skip metadata instead of silently dropping jobs.
- Ensure new tests are pytest-style (no unittest classes) and integrate with existing fixtures.
- Avoid adding heavy optional dependencies; rely on existing numpy fixtures.
- Capture CLI command + outputs under `cli/` for audit—no missing logs.

If Blocked:
- Capture failing pytest output to `.../red/pytest_phase_f_cli_red.log`, record CLI/traceback snippet in docs/summary.md, and log blocker + remediation plan in docs/fix_plan.md Attempts History before pausing.

Findings Applied (Mandatory):
- POLICY-001 — CLI must assume torch>=2.2 availability; document in summary.md and avoid torch-optional branches.
- CONFIG-001 — Builder/CLI remain pure; note in summary that legacy bridge occurs downstream.
- DATA-001 — Validate NPZ path layout before invoking runner; reflect failed checks in skip summary.
- OVERSAMPLING-001 — Maintain dense/sparse job coverage with neighbor_count=7; log if sparse view missing.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline_plan/plan.md:25
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:212
- docs/TESTING_GUIDE.md:101
- specs/data_contracts.md:120
- docs/findings.md:8

Next Up (optional):
- Phase F2 dry-run + real LSQML execution once CLI is proven.

Doc Sync Plan:
- After GREEN tests, rerun `pytest tests/study/test_dose_overlap_reconstruction.py --collect-only -vv` (log already captured).
- Update docs/TESTING_GUIDE.md §2 with new Phase F CLI selector and add row under `docs/development/TEST_SUITE_INDEX.md` (Studies → fly64 overlap) pointing to CLI tests; archive diffs under `plans/.../docs/`.
