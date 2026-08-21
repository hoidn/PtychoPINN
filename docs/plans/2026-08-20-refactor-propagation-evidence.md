# Refactor-branch propagation evidence

Companion to `docs/plans/2026-08-20-refactor-consolidation-propagation.md`.

## Entry state

- `refactor` tip: `2948e52e1ce52bb7e5d9ca6dcc55b66347caa983` (pull --ff-only: already up to date, 2026-08-20)
- Baseline suite (`bash ci/run_ci_tests.sh`, ptycho311 PATH python, 2m17s):
  `1438 passed, 6 skipped, 56 deselected, 11 xfailed, 1 xpassed` — **zero failed**.
  Note: the plan's pre-measured baseline (`1 failed, 1482 passed, 4 skipped,
  13 deselected`) was taken under different collection conditions; this
  session's measured line above is the comparison baseline for every task.
- Known-red baseline fact:
  `tests/torch/test_workflows_components.py::TestLightningCheckpointCallbacks::test_model_checkpoint_callback_configured`
  PASSES inside the CI suite run but FAILS when run in isolation
  (test-ordering dependence). Task 3 re-checks it at the facade split.
- Caller inventory (git grep counts at tip; listings in /tmp/refactor-*.txt):
  - TF facade importers (`ptycho.workflows.components`): **29**
  - torch facade importers (`ptycho_torch.workflows.components`): **134**
  - dill/weights_only sites (`manifest.dill|params.dill|weights_only`): **113**
  - old-name sites (`n_groups|n_subsample|.K|C_model|C_forward|grid_size`
    in `ptycho_torch/*.py` + `ptycho/config/*.py`): **255**

## Phase closeouts

| Phase | Commit | Suite line | Notes |
| --- | --- | --- | --- |
| 0 | `9cc9a811a` | n/a — no full-suite CI (per-task constraint); targeted tests green | Verified-dead barycentric route deleted; loud oversampling guard placed in the previously-silent `else` branch |
| 1 | `20b09fab6` | 1441 passed / 0 failed | Behavior change: TF training honors `subsample_seed`; torch oversampling request now reaches the Phase-0 guard |
| 2 | `0baf9eeb9` | 1443 passed / 0 failed | Facades pure; 63+16 names re-exported; moves AST-verified byte-identical; known-red baseline unchanged |
| 3 | `af354ba44` | 1441 passed / 0 failed (baseline 1438/0) | dill/weights_only load flips + JSON manifest; artifact v1/v2 frozen literals |
| 4 | `845670972` + `e09e08712` | 1481 passed / 0 failed | `torch-artifact-portable-v3` + `torch-model-spec-portable-v3`; `gridsize`/`n_raw_frames_selected` honest fields; C-twins deleted |
| closeout | `78fb2d263` + `4157cf5da` + `c5831c9b2` | 1488 passed / 0 failed | `torch-artifact-portable-v4` (v1–v3 frozen); one name per quantity; omitted ratchets: module-size gates, coordinator size gate, D2 tuple seam (task-6-report.md); model-spec era stays portable-v3 (no model-spec wire change) |

## Spec-drift finding (cross-branch)

`fno-stable`'s `specs/ptychodus_api_spec.md` §4.6 declares, for newly written
PyTorch archives, `artifact_schema_version='torch-artifact-v2'` and a nested
`torch-model-spec-v2` (lines 303–305), while
`ptycho_torch/artifact_schema.py:40` stamps
`CURRENT_ARTIFACT_SCHEMA_VERSION = ARTIFACT_SCHEMA_V4_VERSION`
(`torch-artifact-v4`; v1–v4 literals at lines 36–39) and
`ptycho_torch/model_spec.py:32` stamps `torch-model-spec-v3`. The spec lags the
code by two archive eras and one model-spec era; Phase 4 and the closeout each
changed the archive contract without the same-commit spec amendment that
Decision 1 requires. Back-fix belongs on `fno-stable` (its canonical
`docs/findings.md`); out of scope for this propagation, which fixes the pattern
on `refactor` rather than inheriting it.
