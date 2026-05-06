## Completed In This Pass

- Corrected the approved encoder recipe from the mistakenly committed
  `ffno_encoder_blocks=2` back to the plan-required `ffno_encoder_blocks=24`
  across the shared SRU-Net generator, CDI runner payload, and PDEBench CNS
  model/profile surfaces.
- Updated focused regression coverage first, then verified the corrected recipe
  passes the required deterministic selectors plus the repo integration marker.
- Invalidated the stale `2`-block summary/index wording so the repo no longer
  presents the superseded CDI/CNS results as authoritative for this backlog
  item.
- Neutralized the machine-readable ablation/model-variant entries for this
  backlog item so they now mark the historical `2`-block rows as superseded
  rerun-pending context instead of current live evidence.
- Started a fresh corrected CDI rerun, confirmed the `24`-block recipe trains
  normally for multiple epochs under tmux-backed PID ownership, then terminated
  the incomplete rerun and removed its partial output root when it became clear
  the corrected CDI rerun plus the still-required corrected CNS rerun were not
  practical to finish inside this pass.

## Completed Plan Tasks

- Task 1: Freeze Authorities And Baseline Lineage.
- Task 2: Implement The Shared FFNO Encoder Variant.
- Task 3: Wire The CDI Grid-Lines Row.
- Task 4: Wire The Manual-Only CNS Profile.
- Task 5: Re-Run Deterministic Code Gates Before Expensive Launches.
- Task 8: Partial durability hygiene only.
  The stale `2`-block summary and user-facing index wording were corrected to
  mark the old evidence as superseded and rerun-pending, and the corresponding
  ablation/model-variant entries were downgraded to historical context.

## Remaining Required Plan Tasks

- Task 6: Complete a fresh CDI rerun under the corrected `24`-block recipe and
  validate the required run-root and row-local artifacts.
- Task 7: Complete a fresh matched-condition CNS rerun under the corrected
  `24`-block recipe and validate the required artifacts.
- Task 8: Rebuild the item-local comparison bundle and any downstream
  evidence/index surfaces from the corrected completed run roots.

## Verification

- Prerequisite presence gate: pass.
- `pytest -q tests/torch/test_fno_generators.py -k "PtychoBlock or hybrid_resnet or ffno"`:
  `61 passed, 62 deselected`
- `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or ffno"`:
  `12 passed, 143 deselected`
- `pytest -q tests/studies/test_pdebench_image128_models.py -k "hybrid_resnet or ffno"`:
  `21 passed, 25 deselected`
- `pytest -q tests/test_grid_lines_compare_wrapper.py -k "hybrid_resnet or ffno"`:
  `13 passed, 66 deselected`
- `pytest -q tests/torch/test_generator_registry.py`:
  `11 passed`
- `pytest -q tests/torch/test_lightning_checkpoint.py -k "hybrid_resnet or ffno"`:
  `4 passed, 11 deselected`
- `python -m compileall -q ptycho_torch scripts/studies`: exit `0`
- `pytest -v -m integration`:
  `5 passed, 4 skipped, 2256 deselected`
- Comparison standard for this pass:
  no parity `atol` / `rtol` check applied because this pass stopped before any
  fresh corrected CDI/CNS comparison rows completed.
- Corrected CDI launch sanity:
  tmux-backed shell PID `3023672` with child Python PID `3023677` trained
  cleanly through multiple epochs before the run was intentionally terminated;
  the incomplete output root
  `cdi_ffno_ptychoblock_encoder_20260506T203040Z` was removed afterward so it
  cannot be mistaken for completed evidence.

## Residual Risks

- There is currently no completed corrected `24`-block CDI or CNS evidence for
  this backlog item. Historical `2`-block roots remain available only as
  superseded implementation history.
- Machine-readable downstream indexes that embed the old metrics were not fully
  regenerated with corrected `24`-block metrics in this pass. They now mark
  this backlog item's old `2`-block rows as historical superseded context, but
  they still need a clean regeneration from corrected completed run roots.
