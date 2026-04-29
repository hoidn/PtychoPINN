# NeurIPS Lines128 Paper Benchmark Harness Preflight

- Date: `2026-04-29`
- Backlog item: `2026-04-29-cdi-lines128-paper-benchmark-harness`
- Governing plan:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/execution_plan.md`
- Design authorities:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_design.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_preflight.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_ffno_generator_lines_best_config_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/spectral_resnet_bottleneck_n128_integration_summary.md`
- Machine-readable decision artifact:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/preflight/benchmark_decisions.json`

## Fixed Contract

The harness preserves the approved recovered `lines128` CDI paper contract:

- `N=128`, `gridsize=1`
- synthetic grid-lines with `set_phi=True`
- custom Run1084 probe
- `probe_scale_mode=pad_extrapolate`
- `probe_smoothing_sigma=0.5`
- `nimgs_train=2`, `nimgs_test=2`
- `nphotons=1e9`
- `seed=3`
- `torch_epochs=40`
- `torch_learning_rate=2e-4`
- `torch_scheduler=ReduceLROnPlateau`
- `torch_plateau_factor=0.5`
- `torch_plateau_patience=2`
- `torch_plateau_min_lr=1e-4`
- `torch_plateau_threshold=0.0`
- `torch_loss_mode=mae`
- `torch_mae_pred_l2_match_target=off`
- `torch_output_mode=real_imag`
- `probe_mask=off`
- `fno_modes=12`, `fno_width=32`, `fno_blocks=4`, `fno_cnn_blocks=2`

## Minimum Harness Row Roster

The minimum draftable harness subset for this item is:

- `pinn_hybrid_resnet`
- `pinn`
- `pinn_fno_vanilla`

The FNO comparator is fixed to `fno_vanilla`.

Rationale:

- the paper benchmark design required choosing `fno` versus `fno_vanilla`
  before launch
- `fno_vanilla` is the less Hybrid-adjacent Fourier baseline
- the Torch runner and compare wrapper already expose `fno_vanilla`, so this
  choice does not require new model-family plumbing beyond the harness surface

## Spectral And FFNO Status

- `pinn_spectral_resnet_bottleneck_net`: `supported_for_harness`
  - basis: the Torch runner already accepts
    `spectral_resnet_bottleneck_net`, and the existing integration summary
    proves the model family traverses the `N=128` grid-lines path
- `pinn_ffno`: `supported_for_harness`
  - basis: the fixed-contract prerequisite compare completed with merged
    wrapper artifacts, row-local invocation provenance, and stable
    `metrics/recons` outputs

Neither row is treated as paper-complete evidence in this item. This item only
proves the harness can represent those rows under the fixed contract.

## Seed Policy

- fixed paper-contract seed: `3`
- no multi-seed expansion is authorized in this harness item

## Go / No-Go

- Decision: `GO_FOR_HARNESS_PREFLIGHT_ONLY`
- Authorized now:
  - checked-in contract freeze
  - machine-readable decision manifest
  - bounded readiness-only harness validation artifacts
  - required deterministic verification gates
- Not authorized now:
  - launching the full multi-row paper benchmark
  - promoting readiness-only validation tables to benchmark-performance evidence
  - changing the fixed CDI contract to accommodate a row family

## Expected Validation Root

- readiness-only harness root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/validation/readiness_only_preflight`

Acceptance rule for this item:

- the harness may emit `metrics.json`, `metrics_table.csv`,
  `metrics_table.tex`, `metrics_table_best.tex`, and `metric_schema.json`
  under the readiness root
- the merged result must remain `benchmark_incomplete` until the later full
  benchmark execution item produces real row metadata and fresh multi-row
  artifacts
