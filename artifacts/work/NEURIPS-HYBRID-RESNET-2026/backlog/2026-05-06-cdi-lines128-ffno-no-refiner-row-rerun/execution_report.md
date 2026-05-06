# Execution Report

## Completed In This Pass

- Ran the required deterministic preflight gates and archived logs under:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun/verification/`
- Executed the single tmux-owned compare-wrapper rerun for `pinn_ffno` into:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun/runs/ffno_no_refiner_20260506T223454Z`
- Verified tracked completion with wrapper shell marker `__EXIT_CODE__=0`.
- Repaired the missing required row-local `launcher_completion.json` from fresh
  wrapper stderr markers.
- Wrote the durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_no_refiner_row_rerun_summary.md`
- Updated required discovery surfaces:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`

## Completed Plan Tasks

- Tranche 1: froze the authoritative `lines128` contract from the complete-table
  bundle and historical FFNO proxy root.
- Tranche 2: completed no-refiner preflight, targeted selectors, and compile
  gate without production-code edits.
- Tranche 3: launched exactly one append-only wrapper rerun under tmux,
  preserved the historical proxy roots read-only, and waited for clean exit.
- Tranche 4: produced machine-readable contract and no-refiner audits proving
  only `fno_cnn_blocks` changed and the executed model path contains zero
  local-refiner keys.
- Tranche 5: published the summary and updated the required discovery surfaces.

## Remaining Required Plan Tasks

- None within this execution plan.
- Downstream table promotion remains out of scope here and is still owned by
  `2026-05-06-cdi-lines128-no-refiner-ffno-table-refresh`.

## Verification

- Deterministic model-construction proof:
  `python - <<'PY' ... FfnoGeneratorModule(cnn_blocks=0) ...`
  Log:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun/verification/model_instantiation_no_refiner.log`
- Deterministic runner selector:
  `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "ffno"`
  Log:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun/verification/pytest_grid_lines_torch_runner_ffno.log`
- Deterministic wrapper selector:
  `pytest -q tests/test_grid_lines_compare_wrapper.py -k "ffno"`
  Log:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun/verification/pytest_grid_lines_compare_wrapper_ffno.log`
- Deterministic syntax gate:
  `python -m compileall -q ptycho_torch scripts/studies`
  Log:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun/verification/compileall.log`
- Contract audit:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun/verification/contract_diff.json`
- No-refiner inspection:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun/verification/no_refiner_inspection.json`
- Dataset comparison standard:
  exact array equality against the authoritative complete-table datasets after
  normalizing `_metadata.creation_info.timestamp`; no `atol` / `rtol` was
  needed because all numeric arrays matched bitwise and only metadata
  timestamps differed.

## Residual Risks

- The corrected pure-FFNO row is materially weaker than the historical
  local-refiner proxy on the main CDI headline metrics, so this item resolves
  architecture correctness rather than performance promotion.
- The row-local `launcher_completion.json` had to be emitted manually from
  fresh wrapper logs because the shared helper currently gates ordinary fresh
  compare-wrapper runs.
- Canonical table refresh and manuscript-facing FFNO table changes remain a
  separate downstream task.
