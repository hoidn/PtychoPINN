# Execution Report

## Completed In This Pass

- added a corrected BRDT FFNO rerun wrapper at
  `scripts/studies/born_rytov_dt/run_corrected_ffno_rerun.py` so the new
  append-only root emits truthful `2026-05-06` backlog metadata while leaving
  the historical `2026-05-04` extension path stable
- added focused regression coverage for the corrected dry-run/live metadata path
  and the required row-local `model_profile.json`
- ran the corrected FFNO rerun to completion in `tmux` under `ptycho311`; the
  tracked tmux shell reported `EXIT:0`
- backfilled the missing row-local `model_profile.json` and wrote the required
  same-contract / no-refiner audit artifacts under the corrected root
- wrote the corrected BRDT summary authority and updated BRDT discovery/index
  surfaces to point pure-FFNO readers at the corrected root while preserving
  historical proxy lineage

## Completed Plan Tasks

- Task 1: baseline inputs, adapter contract, and runner truthfulness audit
- Task 2: narrow corrected runner/tests fix
- Task 3: corrected same-contract FFNO rerun launch and completion
- Task 4: same-contract fairness and no-refiner purity audit
- Task 5: durable summary and discoverability refresh

## Remaining Required Plan Tasks

- none

## Verification

- preflight input check: baseline manifest, metrics, and dataset manifest present
- adapter contract check: `build_neural_adapter("ffno", in_channels=1)` reports
  `parameter_count=27394` and rejects `cnn_blocks`
- test gate after final fix:
  `pytest -q tests/studies/test_born_rytov_dt_adapters.py tests/studies/test_born_rytov_dt_preflight.py`
  -> `129 passed in 361.69s (0:06:01)`
- compile gate:
  `python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch`
- long-run gate:
  tmux-tracked PID exited `0`
- artifact gate:
  corrected root contains `preflight_manifest.json`, `metrics.json`,
  `metrics.csv`, `metric_schema.json`, `visual_manifest.json`,
  `combined_metrics.json`, `combined_metrics.csv`, `combined_manifest.json`,
  `rows/ffno/row_summary.json`, `rows/ffno/model_profile.json`,
  `rows/ffno/model_state.pt`, and invocation artifacts
- same-contract audit standard:
  exact field equality across dataset id, split counts, normalization, input
  mode, operator pointer/geometry, fixed sample ids, and training contract

## Residual Risks

- the corrected pure-FFNO row is materially weaker than the historical
  local-refiner proxy, so any downstream BRDT narrative must preserve that
  distinction explicitly
- the BRDT `40`-epoch secondary bundle still contains only the historical
  local-refiner FFNO proxy; this task did not regenerate the `40`-epoch FFNO
  lane
- BRDT remains `decision_support_append_only` candidate context and should not
  be promoted into the required CDI/CNS paper pillars without a later approved
  scope change
