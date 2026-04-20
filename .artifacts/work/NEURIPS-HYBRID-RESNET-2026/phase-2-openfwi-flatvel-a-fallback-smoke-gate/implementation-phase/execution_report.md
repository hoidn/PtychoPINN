## Completed In This Pass

- Fixed the OpenFWI FlatVel-A smoke output-root guard so it rejects live `logs/smoke.pid` markers at the selected run root as well as nested `runs/*/logs/smoke.pid` markers.
- Added rejection for stale direct PID markers that lack `logs/smoke.exit_code` evidence, preserving the long-run guardrail before duplicate writes.
- Implemented the official InversionNet compatibility probe for supplied external checkouts: repo path, git commit, license path, controlled import, `InversionNet` resolution, and bounded CPU forward-shape probe.
- Added regression tests for the direct `$RUN_ROOT/logs/smoke.pid` layout and supplied-checkout official probe path.
- Updated the durable OpenFWI smoke-gate summary to describe the implemented probe and guard behavior.

## Completed Current-Scope Work

- Addressed both high-severity implementation review findings.
- Preserved the approved OpenFWI FlatVel-A fallback smoke-gate layout and kept changes inside `scripts/studies/openfwi_flatvel_a/`, focused tests, the durable summary, and this execution report.
- Kept the current blocked gate decision unchanged because real FlatVel-A shards are still absent.

## Follow-Up Work

- Stage `data1.npy`, `model1.npy`, `data49.npy`, and `model49.npy` under an external or ignored FlatVel-A root before running the real smoke gate.
- Run the tmux/GPU smoke launch and freshness checks once real shards are available.
- Use a real external OpenFWI checkout to record official InversionNet compatibility; compatibility probing is not a full official baseline reproduction.

## Verification

- Red checks before implementation:
  - `pytest tests/studies/test_openfwi_flatvel_a_smoke_cli.py::test_direct_output_root_live_pid_marker_is_rejected_with_allow_existing tests/studies/test_openfwi_flatvel_a_smoke_cli.py::test_direct_output_root_pid_without_exit_code_is_rejected -v` -> failed as expected.
  - `pytest tests/studies/test_openfwi_flatvel_a_models.py::test_official_inversionnet_probe_imports_existing_checkout_and_runs_forward -v` -> failed as expected.
- Green checks after implementation:
  - `pytest tests/studies/test_openfwi_flatvel_a_manifest.py tests/studies/test_openfwi_flatvel_a_data.py tests/studies/test_openfwi_flatvel_a_metrics.py tests/studies/test_openfwi_flatvel_a_models.py tests/studies/test_openfwi_flatvel_a_run_config.py tests/studies/test_openfwi_flatvel_a_smoke_cli.py tests/studies/test_openfwi_flatvel_a_reporting.py -v` -> 31 passed. Log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/review_fix/openfwi_focused_pytest.log`.
  - `pytest tests/studies/test_studies_index_entries.py -v` -> 5 passed. Log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/review_fix/studies_index_pytest.log`.
  - Structural summary check -> `OpenFWI smoke summary is structurally valid`.
  - Plan pointer check -> `plan pointer is valid`.
  - Output contract check -> execution report target exists under `artifacts/work/`.
  - Summary decision check -> `OpenFWI fallback smoke gate summary decision is valid`.
  - Forbidden later-phase summary check -> `no forbidden later-phase tracked summaries found`.
  - `git diff --check` on current-scope files -> clean.
  - Official-source sanity check: downloaded LANL OpenFWI `network.py` and `LICENSE` to a temporary directory, then ran `probe_official_inversionnet(...)` -> `compatible`, forward shape `[1, 5, 1000, 70]` to `[1, 1, 70, 70]`.

## Residual Risks

- The fallback PDE pillar remains blocked on real OpenFWI shard access.
- The implemented official probe proves import and forward-shape compatibility only; it does not train or evaluate official InversionNet on OpenFWI data.
- The repository had substantial unrelated dirty state before this pass; only current-scope files should be staged.
