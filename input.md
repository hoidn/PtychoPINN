Summary: Drive Phase E forward by TDD'ing `run_training_job` so config bridging, runner injection, and dry-run logging are verified before wiring the CLI.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.E3 — run_training_job helper
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_training.py::test_run_training_job_invokes_runner -vv; pytest tests/study/test_dose_overlap_training.py::test_run_training_job_dry_run -vv; pytest tests/study/test_dose_overlap_training.py --collect-only -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T070000Z/phase_e_training_e2/

Do Now — STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.E3:
  - Test: Extend `tests/study/test_dose_overlap_training.py` with `test_run_training_job_invokes_runner` and `test_run_training_job_dry_run`; run each selector to capture RED evidence at `.../red/pytest_run_helper_invokes_runner_red.log` and `.../red/pytest_run_helper_dry_run_red.log`.
  - Implement: studies/fly64_dose_overlap/training.py::run_training_job — create artifact/log directories, call `update_legacy_dict(params.cfg, config)` before invoking the injected runner, touch/write `job.log_path`, and honor `dry_run` by summarizing the planned call without executing.
  - Validate: After implementation, run `pytest tests/study/test_dose_overlap_training.py -k run_training_job -vv` (tee to `.../green/pytest_run_helper_green.log`) and `pytest tests/study/test_dose_overlap_training.py --collect-only -vv` (tee to `.../collect/pytest_collect.log`); stash a dry-run preview under `.../dry_run/run_helper_dry_run_preview.txt` and the stub runner call transcript under `.../runner/run_helper_stub.log`.
  - Doc: Flip plan row E3 to `[x]`, promote new selectors to **Active** in test_strategy.md, append Attempt #14 evidence in docs/TESTING_GUIDE.md and docs/development/TEST_SUITE_INDEX.md, and summarize outcomes in `.../docs/summary.md` with log pointers.

Priorities & Rationale:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T053500Z/phase_e_training_plan/plan.md:15-18 sets E3 as in-progress with CONFIG-001 and logging requirements.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:84-123 defines the new RED selectors, bridge expectations, and execution proof artifacts for run helper coverage.
- docs/DEVELOPER_GUIDE.md:68-104 reiterates CONFIG-001 ordering and logging conventions we must satisfy before launching training.
- specs/data_contracts.md:190-260 ensures dataset paths passed into the runner remain DATA-001 compliant, backing the existence checks performed earlier.
- docs/findings.md#CONFIG-001/#DATA-001/#OVERSAMPLING-001/#POLICY-001 enforce the guardrails we must continue to honor while adding execution helpers.

How-To Map:
- export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
- mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T070000Z/phase_e_training_e2/{red,green,collect,docs,dry_run,runner}
- pytest tests/study/test_dose_overlap_training.py::test_run_training_job_invokes_runner -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T070000Z/phase_e_training_e2/red/pytest_run_helper_invokes_runner_red.log
- pytest tests/study/test_dose_overlap_training.py::test_run_training_job_dry_run -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T070000Z/phase_e_training_e2/red/pytest_run_helper_dry_run_red.log
- After implementation: pytest tests/study/test_dose_overlap_training.py -k run_training_job -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T070000Z/phase_e_training_e2/green/pytest_run_helper_green.log
- pytest tests/study/test_dose_overlap_training.py --collect-only -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T070000Z/phase_e_training_e2/collect/pytest_collect.log
- python - <<'PY' 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T070000Z/phase_e_training_e2/dry_run/run_helper_dry_run_preview.txt
from studies.fly64_dose_overlap.training import TrainingJob, run_training_job
from pathlib import Path
job = TrainingJob(dose=1e3, view='baseline', gridsize=1, train_data_path='train.npz', test_data_path='test.npz', artifact_dir=Path('artifacts'), log_path=Path('artifacts/train.log'))
preview = run_training_job(job, runner=lambda **_: None, dry_run=True)
print(preview)
PY
- python - <<'PY'
from studies.fly64_dose_overlap.training import TrainingJob, run_training_job
from pathlib import Path
class StubRunner:
    def __init__(self, sink):
        self.sink = sink
    def __call__(self, *, config, job, log_path):
        self.sink.write(f"RUN {job.view} dose={job.dose} log={log_path}\n")
with open('plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T070000Z/phase_e_training_e2/runner/run_helper_stub.log', 'w', encoding='utf-8') as sink:
    job = TrainingJob(dose=1e3, view='baseline', gridsize=1, train_data_path='train.npz', test_data_path='test.npz', artifact_dir=Path('artifacts'), log_path=Path('artifacts/train.log'))
    run_training_job(job, runner=StubRunner(sink), dry_run=False)
PY

Pitfalls To Avoid:
- Do not invoke real `ptycho_train`; rely on injected stubs so tests remain fast and deterministic.
- Keep CONFIG-001 compliance: bridge params.cfg via `update_legacy_dict` exactly once per call.
- Avoid mutating `job` objects or global state beyond logging; return useful metadata instead.
- Ensure log directories are created with `mkdir(parents=True, exist_ok=True)` to prevent race failures.
- Write ASCII logs; capture dose/view/gridsize metadata so downstream summaries stay informative.
- Honor dry-run semantics by skipping runner execution while still previewing the command.
- Do not swallow runner exceptions; allow them to surface after ensuring logs are persisted.
- Keep new tests isolated with tmp_path fixtures; no reliance on real datasets.
- Update docs only after GREEN to avoid documenting failing selectors.
- Remember mapped selectors must collect (>0); rerun collect-only after adding new tests.

If Blocked:
- Capture the failing selector output under the artifact hub, log the blocking condition in summary.md, and note the dependency/issue in docs/fix_plan.md Attempt #14 before stopping.

Findings Applied (Mandatory):
- CONFIG-001 — `run_training_job` must bridge params.cfg before any legacy loaders fire; verify via spy in tests.
- DATA-001 — Keep dataset paths pointed at Phase C/D NPZs created earlier; tests should stub but mirror real filenames.
- OVERSAMPLING-001 — Preserve gridsize semantics (baseline=1, overlap=2) when logging and invoking the runner.
- POLICY-001 — Maintain PyTorch-ready paths without downgrading dependency expectations in new helpers/tests.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T053500Z/phase_e_training_plan/plan.md:15-18
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:84-123
- studies/fly64_dose_overlap/training.py:1
- docs/DEVELOPER_GUIDE.md:68
- specs/data_contracts.md:190

Next Up (optional):
- STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.E4 — CLI entrypoint once run helper and tests are green.

Doc Sync Plan:
- After GREEN, append the new selectors and evidence paths to `docs/TESTING_GUIDE.md` §Study Tests and `docs/development/TEST_SUITE_INDEX.md`, and archive the `--collect-only` log in `.../collect/pytest_collect.log` per TESTING_GUIDE hard gate.
