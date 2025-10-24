Summary: Backfill EB3.B logger evidence and clean artifacts before advancing to documentation sync.
Mode: TDD
Focus: [ADR-003-BACKEND-API] Phase EB3 — Logger backend implementation
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_cli_train_torch.py -k logger -vv; pytest tests/torch/test_config_factory.py -k logger -vv; pytest tests/torch/test_workflows_components.py::TestLightningExecutionConfig::test_trainer_receives_logger -vv; pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/impl/2025-10-24T025339Z/{red/,green/,summary.md}
Do Now:
- [ADR-003-BACKEND-API] EB3.B1 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/plan.md — consolidate RED evidence: move `train_debug.log` into `.../green/`, relocate `logger_backend_investigation_report.md` into `.../impl/2025-10-24T025339Z/red/analysis.md`, and author `red/README.md` noting why live failing logs cannot be captured post-implementation; tests: none.
- [ADR-003-BACKEND-API] EB3.B3 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/plan.md — rerun mapped selectors under CPU, archive outputs as `green/pytest_cli_logger_green.log`, `green/pytest_factory_logger_green.log`, `green/pytest_workflows_logger_green.log`, `green/pytest_integration_logger_green.log`, and confirm no stray artifacts remain; tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py -k logger -vv && CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -k logger -vv && CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestLightningExecutionConfig::test_trainer_receives_logger -vv && CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv.
- [ADR-003-BACKEND-API] EB3.B close-out @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/plan.md — draft `summary.md` (test counts, warnings, artifact table), update plan rows B1/B3 to `[x]`, and note RED-log rationale in summary; tests: none.
If Blocked: Capture blockers in `summary.md`, leave plan rows `[P]`, and log context in red/README.md with actionable next steps before exiting.
Priorities & Rationale:
- plan.md B1–B3 — authoritative checklist; requires evidence + hygiene before Phase EB3.C.
- docs/fix_plan.md Attempt #68 — documents outstanding artifact work to unblock completion.
- specs/ptychodus_api_spec.md §4.9 / docs/workflows/pytorch.md §12 — pending documentation updates depend on clean EB3.B evidence.
- decision/approved.md — logger defaults must match approved governance decision (CSV default, tensorboard/mlflow opt-in).
- docs/TESTING_GUIDE.md — authoritative commands for targeted pytest runs.
How-To Map:
- Use `git mv train_debug.log plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/impl/2025-10-24T025339Z/green/train_debug.log` to relocate the log, then remove any empty directories left behind.
- Move the investigation report with `git mv logger_backend_investigation_report.md plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/impl/2025-10-24T025339Z/red/analysis.md` and create `red/README.md` summarizing the expected RED failures and why live logs are unavailable (reference commit 43ea2036 timeline).
- Run mapped selectors individually, piping output to files via `CUDA_VISIBLE_DEVICES="" pytest ... > green/<file>.log 2>&1`; ensure logs include command header and PASS/FAIL summary.
- Compose `summary.md` (ASCII) with sections: Implementation recap, Test matrix (command, pass counts), Evidence paths (red/green), outstanding issues (if any). Note explicitly that RED logs were captured via analysis.md + README due to simultaneous tests+fix commit.
- Update `plans/.../plan.md` B1/B3 states and add completion notes referencing artifact filenames; verify with `rg "B1" plan.md`.
- After file moves, confirm root is clean (`ls train_debug.log` should fail) before staging.
Pitfalls To Avoid:
- Do not delete logs; relocate into timestamped hub to keep provenance.
- Keep warning text in summary accurate (include DeprecationWarning strings if observed).
- Ensure pytest commands run in repository root and honour CPU-only (`CUDA_VISIBLE_DEVICES=""`).
- Avoid rewriting code files; this loop is evidence-only.
- Maintain ASCII filenames/content; no spaces in log filenames.
- Do not alter execution defaults (CSV) or revert implementation logic while capturing evidence.
- Confirm paths in plan.md and summary.md are relative and correct.
- Record reasons for missing RED logs; don’t leave placeholder TODOs.
- Run tests serially to avoid clobbering log files.
- Keep git status clean; stage relocations + new docs before exit.
Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/plan.md
- docs/fix_plan.md#ADR-003-BACKEND-API
- specs/ptychodus_api_spec.md:275
- docs/workflows/pytorch.md:320
- ptycho_torch/train.py:400
- tests/torch/test_cli_train_torch.py:621
Next Up: EB3.C documentation updates (spec/workflow/findings) once EB3.B evidence is archived.
