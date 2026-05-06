# SRU-Net FFNO-To-PtychoBlock Encoder CDI/CNS Small-Cap Summary

- Date: `2026-05-06`
- Backlog item: `2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap`
- State: `implementation_corrected_rerun_pending`
- Governing plan:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap/execution_plan.md`
- Item artifact root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap/`

## Status

The current checkout was corrected to the approved encoder recipe after finding
that the previously committed implementation and its emitted evidence used
`ffno_encoder_blocks=2` instead of the plan-required `ffno_encoder_blocks=24`.

As a result:

- the earlier CDI/CNS evidence for this backlog item is now historical
  `2`-block context only and is **not authoritative** for the approved plan
- no fresh `24`-block CDI or CNS row completed in this pass
- the corrected rerun target has been narrowed to `20` epochs for this
  mechanism-probe item; these rows must not replace existing `40`-epoch
  headline authorities
- the backlog item remains open until both corrected long runs finish and the
  comparison bundle is rebuilt from the corrected outputs

## Approved Fixed Recipe

- architecture id: `hybrid_resnet_ffno_ptychoblock_encoder`
- `encoder_variant=ffno_ptychoblock_encoder`
- `ffno_encoder_blocks=24`
- `ffno_encoder_modes=12`
- `ffno_encoder_share_weights=true`
- `ffno_encoder_gate_init=0.1`
- `ffno_encoder_norm=instance`
- `ffno_encoder_mlp_ratio=2.0`
- `ptychoblock_stage_count=2`
- `downsample_steps=2`
- `downsample_op=stride_conv`
- corrected rerun epoch budget: `20` epochs for both CDI and CNS mechanism
  rows

## Historical Superseded Evidence

The following roots were produced before the contract fix and therefore reflect
the superseded `ffno_encoder_blocks=2` recipe:

- CDI historical root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap/runs/cdi_ffno_ptychoblock_encoder_20260506T183959Z/`
- CNS historical root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap/runs/cns_ffno_ptychoblock_encoder_20260506T190421Z/`

These roots remain useful for debugging and implementation history only. They
must not be cited as the completed evidence for this backlog item.

## Current Pass

Completed in this pass:

- restored the code-level encoder recipe from `2` to `24` FFNO blocks in:
  `ptycho_torch/generators/hybrid_resnet.py`,
  `scripts/studies/grid_lines_torch_runner.py`,
  `scripts/studies/pdebench_image128/models.py`, and
  `scripts/studies/pdebench_image128/run_config.py`
- updated focused regression tests so the approved `24`-block recipe is encoded
  explicitly
- re-ran the required deterministic gates and the repo integration marker under
  the corrected contract
- started a fresh corrected CDI rerun, confirmed it trained normally under the
  heavier `24`-block recipe, then stopped and removed the incomplete run root
  when it became clear that the corrected CDI rerun plus the required corrected
  CNS rerun were not practical to finish inside this pass

## Verification

Deterministic verification completed against the corrected `24`-block recipe:

- prerequisite presence gate: pass
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

Corrected CDI rerun sanity check:

- fresh run launched under tmux-backed PID ownership and advanced through
  multiple epochs with the corrected `24`-block recipe
- the in-flight rerun was intentionally terminated and its incomplete root was
  removed, so there is still **no completed corrected CDI evidence root** for
  this backlog item

## Remaining Required Work

- run the fresh corrected `20`-epoch CDI row to tracked exit `0` and validate
  the full artifact set under the approved `24`-block recipe
- run the fresh corrected `20`-epoch CNS row to tracked exit `0` and validate
  the full artifact set under the approved `24`-block recipe
- rebuild the item-local comparison bundle and any downstream evidence/index
  surfaces from those corrected run roots

## Claim Boundary

Until the corrected `20`-epoch reruns complete, this backlog item has no current
valid cross-pillar result claim. The historical `2`-block rows are superseded,
and the approved `24`-block rows remain rerun-pending. Completed corrected rows
should be interpreted as mechanism-probe evidence, not as replacements for the
existing `40`-epoch CDI or CNS headline authorities.
