# Execution Report

## Completed In This Pass

- Ran the required blocking preflight checks:
  - `pytest -q tests/studies/test_pdebench_image128_rollout_video.py`
  - `pytest -q tests/studies/test_pdebench_image128_runner.py -k "cfd_cns or model_state or matched_condition"`
  - `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py`
- Ran the exact-contract inspect pass at
  `tmp/cns_rollout_checkpoint_refresh_inspect/`, confirming the locked
  matched-condition contract and exposing the rollout-length limit
  (`time_steps=21` with `history_len=5`, so the maximal valid shared rollout is
  `steps=16` from `start_time=5`).
- Recovered from one narrow launcher mistake on the first author-FFNO attempt
  (precreated directory made the output root non-empty) by relaunching on a
  fresh timestamped root with launcher logs outside the run root.
- Produced fresh checkpoint-bearing same-contract reruns for:
  - `author_ffno_cns_base`
  - `spectral_resnet_bottleneck_base`
  - `fno_base`
- Exported deterministic density rollout GIFs plus manifests for all three rows
  on the shared fixed window:
  - `split=test`
  - `sample_id=0`
  - `trajectory_id=7823`
  - `start_time=5`
  - `steps=16`
- Wrote the durable study summary at
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cns_rollout_checkpoint_refresh_summary.md`
  and updated the required discovery indexes:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`

## Completed Plan Tasks

- Task 1: lock the contract and clear the preflight gate
- Task 2: produce one checkpoint-bearing rerun per required row
- Task 3: audit rerun metrics without replacing the current table authority
- Task 4: export deterministic rollout GIFs from the trained checkpoints
- Task 5: publish the durable summary and required discovery updates

## Remaining Required Plan Tasks

- None

## Verification

- Preflight checks:
  - `pytest -q tests/studies/test_pdebench_image128_rollout_video.py` -> `7 passed`
  - `pytest -q tests/studies/test_pdebench_image128_runner.py -k "cfd_cns or model_state or matched_condition"` -> `19 passed, 38 deselected`
  - `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py` -> success
- Inspect evidence:
  - `tmp/cns_rollout_checkpoint_refresh_inspect/split_manifest.json` confirms
    `history_len=5`, split counts `512 / 64 / 64`, `max_windows_per_trajectory=8`,
    and `time_steps=21`
- Row launcher exits:
  - `author_ffno_cns_base` -> `0`
  - `spectral_resnet_bottleneck_base` -> `0`
  - `fno_base` -> `0`
- Required per-row checkpoint artifacts exist for all three rows:
  - `model_state_<row_id>.pt`
  - `model_state_<row_id>.json`
  - `model_profile_<row_id>.json`
  - `metrics_<row_id>.json`
  - `split_manifest.json`
  - `normalization_stats_state.json`
  - `invocation.json`
  - `invocation.sh`
- Rollout validation for all three rows:
  - GIF exists
  - frame count `16`
  - first frame nonblank
  - rollout manifest names the correct run root, checkpoint, row id, split,
    sample id, trajectory id, start time, and step count
- Audit outcomes:
  - `author_ffno_cns_base` -> `metric_compatible_with_current_authority`
    with exact zero deltas across all tracked metrics
  - `spectral_resnet_bottleneck_base` -> `visualization_lineage_only`
    with same-contract inputs but metric drift (`err_nRMSE +0.0155590586`)
  - `fno_base` -> `visualization_lineage_only`
    with same-contract inputs but rounded-table drift (`err_nRMSE +0.0021856390`)

## Residual Risks

- Two of the three fresh same-contract reruns (`spectral_resnet_bottleneck_base`
  and `fno_base`) drifted enough from the frozen authority metrics that they
  should be treated as checkpoint/rollout lineage only. This backlog item does
  not reopen the matched-condition headline table.
- The rollout exporter required module-style invocation with explicit
  `PYTHONPATH=/home/ollie/Documents/PtychoPINN` in this environment because
  direct script execution did not resolve the `scripts.studies...` package
  imports.
