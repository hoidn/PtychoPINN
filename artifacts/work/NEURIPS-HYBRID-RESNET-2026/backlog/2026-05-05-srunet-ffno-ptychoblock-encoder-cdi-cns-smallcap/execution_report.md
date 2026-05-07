## Completed In This Pass

- Completed the corrected `24`-block `20`-epoch CDI rerun at
  `cdi_ffno_ptychoblock_encoder_20260507T073814Z`, validated the emitted
  metrics and visuals, and ensured the run root carries
  `exit_code_proof.json` plus `artifact_freshness_check.json`.
- Completed the corrected `24`-block `20`-epoch matched-condition CNS rerun at
  `cns_ffno_ptychoblock_encoder_20260507T082701Z`, validated the emitted
  metrics and visuals, and repaired the root-level proof artifacts so the run
  root carries `exit_code_proof.json` plus `artifact_freshness_check.json`.
- Rebuilt the item-local `comparison_bundle.json` from the corrected CDI and
  CNS roots only.
- Rewrote the durable summary and discovery surfaces so the corrected
  `24`-block roots are current authority and the historical `2`-block roots are
  explicit superseded lineage only.

## Completed Plan Tasks

- Task 1: Freeze Authorities And Baseline Lineage.
- Task 2: Implement The Shared FFNO Encoder Variant.
- Task 3: Wire The CDI Grid-Lines Row.
- Task 4: Wire The Manual-Only CNS Profile.
- Task 5: Re-Run Deterministic Code Gates Before Expensive Launches.
- Task 6: Complete the corrected CDI rerun and validate the required artifact
  set.
- Task 7: Complete the corrected CNS rerun and validate the required artifact
  set.
- Task 8: Rebuild the comparison bundle and downstream evidence/index surfaces
  from the corrected completed run roots.

## Remaining Required Plan Tasks

- None.

## Verification

- Prerequisite presence gate: pass.
- `pytest -q tests/torch/test_fno_generators.py -k "PtychoBlock or hybrid_resnet or ffno"`:
  `61 passed, 62 deselected`
- `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or ffno"`:
  `12 passed, 143 deselected`
- `pytest -q tests/studies/test_pdebench_image128_models.py -k "hybrid_resnet or ffno"`:
  `21 passed, 27 deselected`
- `pytest -q tests/test_grid_lines_compare_wrapper.py -k "hybrid_resnet or ffno"`:
  `16 passed, 67 deselected`
- `pytest -q tests/torch/test_generator_registry.py`:
  `11 passed`
- `pytest -q tests/torch/test_lightning_checkpoint.py -k "hybrid_resnet or ffno"`:
  `4 passed, 11 deselected`
- `python -m compileall -q ptycho_torch scripts/studies`: exit `0`
- `pytest -v -m integration`:
  `5 passed, 4 skipped, 2291 deselected`
- Comparison standard for this pass:
  no parity `atol` / `rtol` check applied; interpretation used exact scalar
  delta comparisons against reused authority metrics.
- Corrected CDI tracked PID `3215263` exited `0`; the root
  `cdi_ffno_ptychoblock_encoder_20260507T073814Z` contains invocation/config,
  history, metrics, visuals, `exit_code_proof.json`, and
  `artifact_freshness_check.json`.
- Corrected CNS tracked PID `3220360` exited `0`; the root
  `cns_ffno_ptychoblock_encoder_20260507T082701Z` contains invocation/config,
  metrics, comparison visuals, `exit_code_proof.json`, and
  `artifact_freshness_check.json`.

## Residual Risks

- The corrected results are `20`-epoch mechanism probes only. They must not be
  promoted into the `40`-epoch CDI or CNS headline authorities.
- Cross-epoch comparisons against the current `40`-epoch CNS headline rows are
  directionally useful but are not same-budget performance claims.
- Historical `2`-block roots remain on disk and rely on the updated discovery
  surfaces to stay clearly labeled as superseded lineage only.
