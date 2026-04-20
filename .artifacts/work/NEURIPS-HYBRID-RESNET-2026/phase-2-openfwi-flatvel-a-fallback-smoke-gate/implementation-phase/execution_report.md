## Completed In This Pass

- Aligned the OpenFWI smoke CLI guard with the approved tmux prelaunch layout: the selected run root may contain only `logs/smoke.run_id`, `logs/smoke.started_at_ns`, and the current launcher-owned `logs/smoke.pid` before the CLI writes artifacts. Other live or incomplete PID markers remain rejected.
- Updated the execution plan's real-launch and freshness steps to write selected-run start markers and call `validate_fresh_artifacts(...)`, so old artifacts are checked against the tracked start time before a selected run is treated as complete.
- Fixed stale data-access blocker handling. Successful data resolution clears obsolete `data_access_blocker.json`, and `collate_comparison()` now validates blocker `run_id` before allowing a blocker to dominate current metrics.
- Tightened real FlatVel-A shape validation to require full shard shapes `(500, 5, 1000, 70)` and `(500, 1, 70, 70)`, including leading sample count. Synthetic fixture support is explicit through `--allow-synthetic-shard-samples` and records `sample_count_contract: synthetic_fixture`.
- Added regressions covering the approved prelaunch output-root layout, stale blocker/current metrics collation, and non-500 real-shard sample-count rejection.
- Updated the durable OpenFWI smoke-gate summary to record the stricter shape contract, launcher guard behavior, and stale blocker handling.

## Completed Current-Scope Work

- Addressed all three high-severity implementation review findings.
- Preserved the approved OpenFWI FlatVel-A fallback smoke-gate layout and kept changes inside `scripts/studies/openfwi_flatvel_a/`, focused tests, the durable summary, and this execution report.
- Kept the current blocked gate decision unchanged because real FlatVel-A shards are still absent.

## Follow-Up Work

- Stage `data1.npy`, `model1.npy`, `data49.npy`, and `model49.npy` under an external or ignored FlatVel-A root before running the real smoke gate.
- Run the tmux/GPU smoke launch and freshness checks once real shards are available.
- Use a real external OpenFWI checkout to record official InversionNet compatibility; compatibility probing is not a full official baseline reproduction.

## Verification

- Red checks before implementation:
  - `pytest tests/studies/test_openfwi_flatvel_a_data.py::test_real_flatvel_shape_contract_requires_500_samples tests/studies/test_openfwi_flatvel_a_smoke_cli.py::test_plan_style_precreated_logs_and_launcher_pid_marker_runs_synthetic_smoke tests/studies/test_openfwi_flatvel_a_reporting.py::test_collator_ignores_stale_data_access_blocker_when_current_metrics_exist -v` -> failed as expected: default shape inspection accepted 4-sample shards, the parser/guard did not support the explicit synthetic/prelaunch path, and stale blockers still forced `block_data_access`.
- Green checks after implementation:
  - `pytest tests/studies/test_openfwi_flatvel_a_manifest.py tests/studies/test_openfwi_flatvel_a_data.py tests/studies/test_openfwi_flatvel_a_metrics.py tests/studies/test_openfwi_flatvel_a_models.py tests/studies/test_openfwi_flatvel_a_run_config.py tests/studies/test_openfwi_flatvel_a_smoke_cli.py tests/studies/test_openfwi_flatvel_a_reporting.py -v` -> 34 passed. Log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/review_fix_current/openfwi_focused_pytest.log`.
  - `pytest tests/studies/test_studies_index_entries.py -v` -> 5 passed. Log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/review_fix_current/studies_index_pytest.log`.
  - Structural summary check -> `OpenFWI smoke summary contains review-fix contract updates`.
  - Plan freshness check -> `execution plan records selected-run start markers and freshness helper check`.
  - Output contract check -> execution report target exists under `artifacts/work/`.
  - `git diff --check` on current-scope files -> clean.

## Residual Risks

- The fallback PDE pillar remains blocked on real OpenFWI shard access.
- The implemented official probe proves import and forward-shape compatibility only; it does not train or evaluate official InversionNet on OpenFWI data.
- Synthetic smoke tests prove code paths and artifact schemas only; they are not scientific viability evidence for FlatVel-A.
- The repository had substantial unrelated dirty state before this pass; only current-scope files should be staged.
