# NeurIPS Lines128 U-NO Preflight Summary

- Date: `2026-04-30`
- Backlog item: `2026-04-30-cdi-lines128-uno-design-preflight`
- Final status: `ready_for_uno_generator_integration`
- Artifact root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-design-preflight/`

## Completed In This Pass

- added the bounded preflight helper `scripts/studies/lines128_uno_preflight.py`
  and focused regression coverage in
  `tests/studies/test_lines128_uno_preflight.py`
- captured the live `ptycho311` environment, `pip show neuraloperator`
  provenance, observed `UNO` constructor signature, and a tiny dummy forward
  probe under the locked `lines128` CDI contract
- resolved two design-to-runtime mismatches before declaring readiness:
  `UNO` requires explicit `uno_out_channels` and `uno_scalings` despite
  signature defaults of `None`, and the accepted `uno_n_modes` form is a
  nested per-layer `2D` sequence rather than a flat four-entry list
- froze the exact pre-metric U-NO settings for the later generator-integration
  item and recorded that the raw `UNO` output already matches the required
  `B x 2 x 128 x 128` real/imag channel layout

## Authoritative Inputs

- selected item context:
  `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/5/items/2026-04-30-cdi-lines128-uno-design-preflight/selected-item-context.md`
- extension design:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_design.md`
- completed base-table authority:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
- supervised-adjacent authority:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_supervised_equivalent_rows_summary.md`
- checked-in implementation plan:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-design-preflight/execution_plan.md`

## Runtime And Package Provenance

- Python executable: `/home/ollie/miniconda3/envs/ptycho311/bin/python3.11`
- Python version: `3.11.13`
- Torch / CUDA: `torch 2.9.1+cu128`, CUDA `12.8`
- GPU: `NVIDIA GeForce RTX 3090`
- distribution package: `neuraloperator==2.0.0`
- import module: `neuralop==2.0.0`
- module file:
  `/home/ollie/miniconda3/envs/ptycho311/lib/python3.11/site-packages/neuralop/__init__.py`
- git commit during preflight: `2a49b4e170a5ed05e3cdafe01d73e1601c3f4050`
- git dirty at probe time: `true`

Primary artifact payloads:

- `environment_probe.json`
- `pip_show_neuraloperator.txt`
- `uno_signature.json`
- `uno_shape_probe.json`
- `preflight_decision.json`

## Observed UNO Signature

Observed constructor surface in `ptycho311`:

```text
UNO(
    in_channels,
    out_channels,
    hidden_channels,
    lifting_channels=256,
    projection_channels=256,
    positional_embedding='grid',
    n_layers=4,
    uno_out_channels=None,
    uno_n_modes=None,
    uno_scalings=None,
    horizontal_skips_map=None,
    channel_mlp_dropout=0,
    channel_mlp_expansion=0.5,
    non_linearity=gelu,
    norm=None,
    preactivation=False,
    fno_skip='linear',
    horizontal_skip='linear',
    channel_mlp_skip='soft-gating',
    ...
)
```

Critical verified behavior beyond the raw signature text:

- `uno_out_channels` cannot remain `None` at runtime
- `uno_n_modes` cannot remain `None` and must be a per-layer nested 2D list
- `uno_scalings` cannot remain `None`
- the chosen asymmetric four-layer channel plan required
  `channel_mlp_skip="linear"` during preflight

## Frozen U-NO Settings

These settings are the current pre-metric authority for the later
`neuralop_uno` integration item:

- `in_channels=1`
- `out_channels=2`
- `hidden_channels=32`
- `lifting_channels=128`
- `projection_channels=128`
- `n_layers=4`
- `uno_out_channels=[32, 64, 64, 32]`
- `uno_n_modes=[[12, 12], [12, 12], [12, 12], [12, 12]]`
- `uno_scalings=[[1.0, 1.0], [0.5, 0.5], [1, 1], [2, 2]]`
- `positional_embedding="grid"`
- `channel_mlp_skip="linear"`
- `generator_output_mode="real_imag"`

Interpretation note:

- the accepted `uno_n_modes` form is a nested four-entry per-layer 2D sequence
  aligned to the locked `fno_modes=12` contract

## Shape Probe Outcome

- dummy input shape: `B=2, C=1, H=128, W=128`
- raw `UNO` output shape observed: `B x 2 x 128 x 128`
- wrapper mapping decision: direct `real_imag` channel acceptance
- adapter requirement: `false`
- implication for the next item: the later `neuralop_uno` wrapper needs normal
  generator integration and repo output-layout plumbing, not a deeper semantic
  conversion from the probed raw UNO output

## Scope Boundary Preserved

- no generator registry support was added
- no `architecture` enum or runtime launcher routing was changed
- no training, benchmark rows, table extension, or promoted bundle execution
  ran in this pass
- the authoritative six-row `lines128` base table remained untouched

## Next Handoff

- next backlog target:
  `2026-04-30-cdi-lines128-uno-generator-integration`
- handoff requirement:
  implement `neuralop_uno` as a generator wrapper using the frozen settings
  above, keep the locked `lines128` CDI contract unchanged, and add focused
  supervised/PINN construction plus shape/provenance tests before any row runs

## Verification

- required deterministic input check:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-design-preflight/verification/required_inputs_check.log`
- initial red selector:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-design-preflight/verification/pytest_red.log`
- constructor regression red:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-design-preflight/verification/pytest_constructor_red.log`
- nested-mode regression red:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-design-preflight/verification/pytest_modes_red.log`
- focused green selector:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-design-preflight/verification/pytest_green.log`
- constructor regression green:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-design-preflight/verification/pytest_constructor_green.log`
- nested-mode regression green:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-design-preflight/verification/pytest_modes_green.log`
- compile gate:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-design-preflight/verification/compileall.log`
- live import gate:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-design-preflight/verification/neuralop_import_check.log`
- live helper run:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-design-preflight/verification/live_preflight.log`
