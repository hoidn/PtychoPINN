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

## Historical Sources Used For Reconstruction

- Study-contract anchor:
  `docs/studies/index.md#grid-lines-n128-hybrid-resnet-legacy-best-e40-seed3`
- Fixed-contract prerequisite evidence:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_preflight.md`
- Stable FFNO-versus-Hybrid slice:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_ffno_generator_lines_best_config_summary.md`
- Live command/provenance source for recovered flags:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet/invocation.json`
- Spectral routing evidence:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/spectral_resnet_bottleneck_n128_integration_summary.md`

The JSON decision artifact is now the harness source of truth. The harness must
consume its fixed contract, seed policy, and go/no-go state directly and fail
closed if the delegated compare-wrapper preflight drifts from those fields.

For the later backlog item that launches the minimum draftable CDI subset under
the same frozen contract, see
`docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_minimum_paper_table_execution_authority.md`.
That separate note supersedes launch authority only for the minimum-subset
execution item; this preflight note remains readiness-only authority for the
harness prerequisite item.

## Fixed Contract Reconstruction

The recovered paper-harness contract stays aligned with the approved
`grid-lines-n128-hybrid-resnet-legacy-best-e40-seed3` slice. The machine
readable fields live in `benchmark_decisions.json::fixed_contract`; the table
below mirrors those fields with per-field provenance/confidence.

| Field | Value | Confidence | Primary source(s) |
| --- | --- | --- | --- |
| `N` | `128` | high | `lines128_paper_benchmark_design.md`, `lines128_paper_benchmark_preflight.md` |
| `gridsize` | `1` | high | `lines128_paper_benchmark_design.md`, `lines128_paper_benchmark_preflight.md` |
| `dataset_source` | `synthetic_lines` | high | `lines128_paper_benchmark_design.md` |
| `set_phi` | `true` | high | `lines128_paper_benchmark_design.md`, `lines128_paper_benchmark_preflight.md` |
| `probe_source` | `custom` | high | `lines128_paper_benchmark_preflight.md`, prerequisite `invocation.json` |
| `probe_npz` | `datasets/Run1084_recon3_postPC_shrunk_3.npz` | high | `lines128_paper_benchmark_preflight.md`, prerequisite `invocation.json` |
| `probe_scale_mode` | `pad_extrapolate` | high | `lines128_paper_benchmark_preflight.md`, prerequisite `invocation.json` |
| `probe_smoothing_sigma` | `0.5` | high | `lines128_paper_benchmark_preflight.md`, prerequisite `invocation.json` |
| `probe_mask_diameter` | `null` (`probe_mask=off`) | high | `lines128_paper_benchmark_design.md`, `cdi_ffno_generator_lines_best_config_summary.md` |
| `nimgs_train` | `2` | high | `lines128_paper_benchmark_preflight.md`, prerequisite `invocation.json` |
| `nimgs_test` | `2` | high | `lines128_paper_benchmark_preflight.md`, prerequisite `invocation.json` |
| `nphotons` | `1e9` | high | `lines128_paper_benchmark_preflight.md`, prerequisite `invocation.json` |
| `seed` | `3` | high | `lines128_paper_benchmark_preflight.md`, prerequisite `invocation.json` |
| `torch_epochs` | `40` | high | `lines128_paper_benchmark_design.md`, prerequisite `invocation.json` |
| `torch_learning_rate` | `2e-4` | high | `lines128_paper_benchmark_design.md`, prerequisite `invocation.json` |
| `torch_scheduler` | `ReduceLROnPlateau` | high | `lines128_paper_benchmark_design.md`, prerequisite `invocation.json` |
| `torch_plateau_factor` | `0.5` | high | `lines128_paper_benchmark_design.md`, prerequisite `invocation.json` |
| `torch_plateau_patience` | `2` | high | `lines128_paper_benchmark_design.md`, prerequisite `invocation.json` |
| `torch_plateau_min_lr` | `1e-4` | high | `lines128_paper_benchmark_design.md`, prerequisite `invocation.json` |
| `torch_plateau_threshold` | `0.0` | high | `lines128_paper_benchmark_design.md`, prerequisite `invocation.json` |
| `torch_loss_mode` | `mae` | high | `lines128_paper_benchmark_design.md`, prerequisite `invocation.json` |
| `torch_mae_pred_l2_match_target` | `false` (`off`) | high | `lines128_paper_benchmark_design.md`, prerequisite `invocation.json` |
| `torch_output_mode` | `real_imag` | high | `lines128_paper_benchmark_design.md`, prerequisite `invocation.json` |
| `fno_modes` | `12` | high | `lines128_paper_benchmark_design.md`, prerequisite `invocation.json` |
| `fno_width` | `32` | high | `lines128_paper_benchmark_design.md`, prerequisite `invocation.json` |
| `fno_blocks` | `4` | high | `lines128_paper_benchmark_design.md`, prerequisite `invocation.json` |
| `fno_cnn_blocks` | `2` | high | `lines128_paper_benchmark_design.md`, prerequisite `invocation.json` |

Approved deviations:

- none

Absence of a deviation record means the harness must preserve every field above
exactly. The harness now validates that the delegated compare-wrapper preflight
returns the same machine-readable contract before emitting readiness artifacts.

## Minimum Harness Row Roster

Minimum draftable subset for this backlog item:

- `pinn_hybrid_resnet`
- `pinn`
- `pinn_fno_vanilla`

Selected FNO comparator:

- `fno_vanilla`
- rationale: use the less Hybrid-adjacent Fourier baseline and freeze that
  choice before benchmark execution so later rows cannot drift between `fno`
  and `fno_vanilla`

Additional row states:

- `pinn_spectral_resnet_bottleneck_net`: `supported_for_harness`
  - basis: the Torch runner already accepts
    `spectral_resnet_bottleneck_net`, and the existing integration summary
    proves the family traverses the `N=128` grid-lines path
- `pinn_ffno`: `supported_for_harness`
  - basis: the fixed-contract prerequisite compare completed with merged
    wrapper artifacts, row-local invocation provenance, and stable
    `metrics/recons` outputs

Neither additional row is treated as paper-complete evidence in this item. The
current task only proves the harness can represent them under the frozen CDI
contract.

## Seed Policy

- policy: fixed seed
- value: `3`
- confidence: high
- source(s):
  - `lines128_paper_benchmark_design.md`
  - `lines128_paper_benchmark_preflight.md`
  - prerequisite compare `invocation.json`

No multi-seed expansion is authorized in this harness item.

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
