# PDEBench 128x128 Image-Suite Plan

## Status

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Scope owner: Roadmap Phase 2
- Status: active plan; SWE/Darcy readiness implemented, full benchmarks and 2D Compressible Navier-Stokes support pending
- Date: 2026-04-20
- Experiment root: `/home/ollie/Documents/PtychoPINN/`
- Manuscript artifact root: `/home/ollie/Documents/neurips/` (future Phase 5 root; this plan must not create it)
- Stage A preflight summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_preflight.md`
- Darcy static-operator execution plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-darcy-static-operator-benchmark/execution_plan.md`
- 2D Compressible Navier-Stokes design: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_design.md`
- Spectral ResNet bottleneck variant design: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_resnet_bottleneck_design.md`

This plan updates the PDE pillar from a single SWE-first path to one compact native `128x128` PDEBench image suite:

1. SWE (`swe`)
2. Darcy Flow (`darcy`)
3. 2D Compressible Navier-Stokes (`2d_cfd_cns`)

No manuscript prose and no `/home/ollie/Documents/neurips/` table or figure artifacts are created by this plan. Paper-facing artifacts are still deferred until the roadmap evidence-bundle phase.

## Inputs Read

- `AGENTS.md`
- `docs/index.md`
- `docs/findings.md`
- `docs/model_baselines.md`
- `docs/studies/index.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_benchmark_selection.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_execution_summary.md`
- `scripts/studies/pdebench_swe/`
- Official PDEBench data download README, accessed 2026-04-20

## Why This Suite

The suite is small enough to be one coherent PDEBench image contribution rather than a broad PDEBench survey. All three selected tasks are PDEBench 2D image/grid tasks that can share dataset inspection, split, normalization, model-profile, metric, provenance, and reporting machinery.

- SWE supplies a dynamic wave/flow-like task with local structures and larger-scale field evolution.
- Darcy supplies a static elliptic operator-map task, useful for testing whether the hybrid spectral/local representation helps global coefficient-to-solution mapping.
- 2D Compressible Navier-Stokes supplies a native `128x128` four-field time-dependent fluid benchmark, useful for testing coupled thermodynamic/kinematic dynamics and sharper structures than SWE or Darcy.

OpenFWI FlatVel-A remains useful as an adjacent inverse-wave fallback or extension, but it is `70x70` and should not displace this native `128x128` PDEBench image suite while the suite is feasible.

## Official Data Targets

The exact local files and schemas must be confirmed in the first implementation tranche. Expected PDEBench download identifiers are:

| Task | PDEBench `pde_name` | Expected file from PDEBench docs | Listed size | Local status |
| --- | --- | --- | ---: | --- |
| SWE | `swe` | `2D_rdb_NA_NA.h5` | 6.2 GB | staged and preflight-ready under `/home/ollie/Documents/pdebench-data/swe/` |
| Darcy Flow | `darcy` | `2D_DarcyFlow_beta1.0_Train.hdf5` | 6.2 GB | staged and preflight-ready under `/home/ollie/Documents/pdebench-data/darcy/` |
| 2D Compressible Navier-Stokes | `2d_cfd_cns` | `2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5` | `55,050,245,208` bytes | staging in progress under `/home/ollie/Documents/pdebench-data/2d_cfd_cns/` after local storage gate passed |

If local disk is too tight for all files and generated outputs, use an approved external data root. Dataset files must stay outside git and should not be copied into `/home/ollie/Documents/neurips/`.

## Claims and Evidence Boundaries

Allowed evidence after successful execution:

- Same-protocol local comparison of Hybrid ResNet, FNO, and U-Net on SWE, Darcy, and 2D Compressible Navier-Stokes where each task completes.
- Suite-level evidence that the spectral/local hybrid architecture transfers beyond CDI to native `128x128` PDEBench image problems.
- Focused ablations showing whether spectral capacity and local ResNet capacity matter for the completed tasks.

Not allowed:

- Any model ranking or competitiveness claim from smoke metrics.
- Any meaningful benchmark-performance claim from capped, subsampled, smoke, or pilot-only training data.
- Broad PDEBench SOTA claims.
- Same-protocol claims against published rows unless the same task, split, preprocessing, model code or accepted reimplementation, metric, and horizon are reproduced.
- Promotion of OpenFWI smoke metrics as performance evidence.

## Architecture

Prefer a shared image-suite adapter over three unrelated study stacks.

Hybrid ResNet adapter/channel contract:

- The PDEBench image-suite path must use a supervised real-channel adapter, not the CDI `PtychoPINN_Lightning` physics wrapper.
- The adapter must keep the full Hybrid ResNet encoder-bottleneck-decoder body: `SpatialLifter`, Hybrid ResNet encoder blocks, downsampling, `ResnetBottleneck`, `CycleGanUpsampler`, and a final real-channel `Conv2d` projection.
- Optional future architecture extension: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_resnet_bottleneck_design.md` defines a separate `spectral_resnet_bottleneck_net` family that keeps the current encoder/downsample/upsample shell but replaces only the constant-resolution `ResnetBottleneck` with a ResNet-local bottleneck plus a shared factorized spectral residual branch. It is intentionally exposed outside the `hybrid_resnet_*` namespace to avoid benchmark-table confusion.
- Channel counts are ordinary supervised tensor dimensions in `(B,C,H,W)`. They come from each task spec's input and target fields and are passed as `in_channels` and `out_channels`.
- Do not reuse `data_config.C` or any ptychographic `C` grouping for PDE/OpenFWI channels. In the registered CDI builder, `C` means object-patch grouping and feeds the real/imag complex output contract; that is the wrong semantic contract for real PDE fields.
- Same-field dynamic tasks such as SWE may use `in_channels == out_channels`. Static or inverse-map tasks such as Darcy or OpenFWI-style data may use different values, for example `5 -> 1`.

PDEBench dataset-family contract:

- Dynamic state datasets, currently SWE and 2D Compressible Navier-Stokes, provide one or more state fields with trajectory/time axes. The supervised unit is `(trajectory_id, time_index)`: input is a state history window, target is the next state. Splits are trajectory-level before any one-step pair expansion, and rollout claims are out of scope until one-step evidence is stable.
- Static operator-map datasets, currently Darcy Flow, provide paired input and target fields with a shared sample axis and no time axis. The supervised unit is one sample index: input field `x_i` maps to target field `y_i`. Splits are sample-level, there is no one-step pair construction, and there is no rollout interpretation.
- All task loaders must convert HDF5 layouts to channel-first `(C,H,W)` tensors before batching. `NHW` fields get an explicit singleton channel, `NCHW` fields are already channel-first, and `NHWC` fields move the last axis to channel position.
- Input and target channels are derived separately. For example, the current Darcy preflight identifies input dataset `nu` with axis `NHW`, shape `[10000,128,128]`, which becomes `in_channels=1`, and target dataset `tensor` with axis `NCHW`, shape `[10000,1,128,128]`, which becomes `out_channels=1`. Future PDEBench static-operator tasks may have `in_channels != out_channels`.
- Normalization must be fit on the train split only. For same-field dynamic tasks, one recorded per-channel state normalization may be applied consistently to both input and target states. For static operator maps, input and target statistics must be recorded separately because coefficient/permeability fields and solution fields are different physical quantities.
- Metrics are computed on denormalized target-space predictions. Dynamic tasks report the one-step target-state metric. Darcy reports target solution-field metrics such as relative L2 or the fixed PDEBench-equivalent nRMSE; it should not report a trajectory or rollout metric.

Target module shape:

- `scripts/studies/pdebench_image128/task_specs.py`
  - Task specs for `swe`, `darcy`, and `2d_cfd_cns`.
  - Dataset identity, expected HDF5 layout, state/input/target fields, axis policy, task type, metric contract, and horizon policy.
- `scripts/studies/pdebench_image128/data.py`
  - Lazy HDF5 datasets for grouped trajectory layouts, monolithic trajectory arrays, and static operator maps.
  - Channel-first tensor output and optional padding/cropping compatible with the existing SWE code.
- `scripts/studies/pdebench_image128/splits.py`
  - Deterministic trajectory-level splits for dynamic tasks and sample-level splits for Darcy.
  - Seed `20260420` unless a plan records an explicit override.
- `scripts/studies/pdebench_image128/metrics.py`
  - `err_nRMSE` and `err_RMSE` for all tasks where meaningful.
  - Relative L2 or documented PDEBench-equivalent operator metric for Darcy.
- `scripts/studies/pdebench_image128/models.py`
  - Reuse the existing SWE model wrappers where sensible: full-body supervised Hybrid ResNet adapter, FNO, and U-Net.
  - Keep task-specific channel counts and spatial dimensions explicit.
- `scripts/studies/pdebench_image128/run_config.py`
  - Shared profile definitions and run-budget validation.
- `scripts/studies/pdebench_image128/reporting.py`
  - Per-task comparison summaries and suite-level collation.
- `scripts/studies/run_pdebench_image128_suite.py`
  - CLI entrypoint for preflight, smoke, pilot, and longer runs.

Do not create this whole shape mechanically if the existing `scripts/studies/pdebench_swe/` module can be generalized in smaller steps. The first implementation plan should decide the minimal safe migration path.

## Required Model Profiles

Competitiveness runs must inherit the current grid-lines Hybrid ResNet recipe from `docs/model_baselines.md` unless a later plan records a justified override.

Required primary profiles:

- `hybrid_resnet_base`: hidden width `32`, `fno_modes=12`, `fno_blocks=4`, downsample steps `2`, Hybrid ResNet blocks `6`.
- `fno_base`: hidden width `32`, `fno_modes=12`, `fno_blocks=4`.
- Task-specific strong U-Net profile: for Darcy this is `unet_strong`; other tasks may use `unet_base` only if the implementation records a non-toy parameter count and marks it as a strong baseline.

Task-local CNS override:

- The canonical CNS Hybrid row is `hybrid_resnet_cns`, which inherits the `hybrid_resnet_base` shell, enables `hybrid_skip_connections=on` with `hybrid_skip_style=add`, and now defaults to `hybrid_upsampler=pixelshuffle`.
- This is a task-local override for `2d_cfd_cns`, not a global rename of `hybrid_resnet_base`.

Darcy-specific strong baseline rule:

- `unet_base` is not automatically a strong Darcy baseline if it resolves to the existing tiny smoke-scale `SmallUNet`. Darcy benchmark rows must use `unet_strong`, preferably an official-style PDEBench `UNet2d(init_features=32)` or an explicitly documented equivalent with recorded parameter count.
- The Darcy beta `1.0` literature calibration from the PDEBench supplement is U-Net RMSE/nRMSE about `6.4e-3`/`3.3e-2` and FNO RMSE/nRMSE about `1.2e-2`/`6.4e-2`. The HAMLET paper reports later nRMSE context around `2.05e-2` for OFormer and `1.40e-2` for HAMLET. These are protocol-dependent calibration values, not same-protocol local reproduction unless the spatial downsampling, split, training recipe, and metrics are matched.

Required training recipe for benchmark-performance runs:

- Loss: MAE for training unless the task-specific metric contract justifies a documented override.
- Optimizer: Adam.
- Learning rate: `2e-4`.
- Scheduler: `ReduceLROnPlateau`.
- Scheduler parameters: factor `0.5`, patience `2`, min LR no higher than `1e-5` (default `1e-5` for PDE studies), threshold `0.0`.
- Training seed: `20260420` unless explicitly overridden and recorded.
- Any PDE competitiveness row with a scheduler floor higher than `1e-5` is underconfigured for this campaign unless a later plan records a justified, pre-run override and the summary labels it as such.

Full available training-set rule:

- A meaningful benchmark-performance row must train on the full available training split for the selected official file after deterministic validation/test holdout.
- Example: if a selected task has 10,000 total samples and the approved split holds out 1,000 validation and 1,000 test samples, the benchmark training split is the remaining 8,000 samples. A run trained on 512 or 1,024 samples is a pilot or triage result, not a meaningful benchmark row.
- For dynamic tasks, "samples" means the selected unit of supervision after the task contract is fixed. If the task contract is one-step prediction over all available trajectories, benchmark runs should use all training trajectories and all eligible one-step pairs unless a later plan documents a scientifically defensible horizon/pair subsampling policy before any model is trained.
- For Darcy, benchmark runs should use all training samples in the selected official file after validation/test holdout.
- If the full available training split cannot fit storage, runtime, or GPU constraints, the correct action is to record a feasibility blocker, reduce the paper claim, or choose a new plan scope. Do not relabel a capped pilot as a benchmark.

Smoke runs may use tiny data and short budgets for readiness only. They still should use the same model shape where feasible, but no smoke metric can rank models or support a pivot.

## Task Contracts

### SWE

- Task type: dynamic one-step next-state prediction.
- Expected file: `2D_rdb_NA_NA.h5`.
- Local verified shape from current staged file: grouped trajectory datasets with per-trajectory `data` shape `(101, 128, 128, 1)`.
- Initial adapter source: existing `scripts/studies/pdebench_swe/`.
- Primary metric: `err_nRMSE`.
- Secondary metrics: `err_RMSE`, runtime, peak GPU memory, and optional rollout metrics only after one-step evidence is stable.

### Darcy Flow

- Task type: static operator map.
- Expected file: `2D_DarcyFlow_beta1.0_Train.hdf5`.
- Current preflight schema: input dataset `nu`, shape `[10000,128,128]`, axis `NHW`; target dataset `tensor`, shape `[10000,1,128,128]`, axis `NCHW`; native spatial shape `128x128`; available supervision units `10000`.
- Supervised sample contract: `x = nu[i]` as `(1,128,128)` coefficient/permeability input, `y = tensor[i]` as `(1,128,128)` solution target. Do not synthesize a time axis or one-step pairs.
- Split policy: deterministic sample-level train/validation/test manifest over the shared sample index. The same index must select both input and target arrays.
- Normalization policy: train-split input statistics and train-split target statistics are separate records. Apply input stats to `nu`, target stats to `tensor`, and denormalize predictions with target stats before metrics.
- Primary metric: relative L2 or documented PDEBench-equivalent nRMSE, chosen and fixed before the first longer run.
- Secondary metrics: RMSE, runtime, peak GPU memory.
- Dedicated plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-darcy-static-operator-benchmark/execution_plan.md`.
- Published-context caveat: PDEBench's published Darcy beta `1.0` U-Net/FNO values and later HAMLET/OFormer values are calibration targets for sanity and paper context. They should be reported separately from local same-protocol rows and must carry resolution/split/training caveats.

### 2D Compressible Navier-Stokes

- Task type: dynamic one-step next-state prediction.
- Active design: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_design.md`.
- Preferred first file: `2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`, DaRUS datafile `164690`, official MD5 `21969082d0e9524bcc4708e216148e60`.
- Storage note: the full `2d_cfd` family is listed as `551 GB`; do not download the family to the root filesystem. Individual official `128x128` random-viscous files are `55,050,245,208` bytes each. After deleting the abandoned diffusion-reaction partial download and later local cleanup, the root filesystem had `75,243,819,008` bytes available before CNS staging, clearing the `60 GB` local gate for one direct-file download.
- Expected schema from official PDEBench code, not yet locally verified: separate datasets for `density`, `Vx`, `Vy`, and `pressure`, plus coordinate arrays. The adapter must stack fields in that order.
- Primary adapter contract after staging: `history_len=2`, flattening two prior four-field states into `input=(8,128,128)` and predicting `target=(4,128,128)`.
- Required ablation if budget permits: `history_len=1`, with `input=(4,128,128)` and `target=(4,128,128)`.
- Primary metrics: denormalized aggregate and per-field `err_nRMSE`/`err_RMSE`.
- Required shock-capture diagnostic: denormalized Fourier-space RMSE bands `fRMSE_low`, `fRMSE_mid`, and `fRMSE_high`, with `fRMSE_high` treated as the primary small-scale/shock-capture metric for CNS. Report low/mid bands alongside high so a model cannot improve high-frequency content while losing large-scale accuracy.
- Secondary diagnostics: runtime, peak GPU memory, optional gradient/front-weighted error, and conservation-drift checks if low-cost.
- Split policy: deterministic trajectory/sample-level train/validation/test manifest before history-window expansion.

## Run Strategy

The plan uses staged evidence so one RTX 3090 for several days is not consumed before data/schema risk is reduced.

### Stage A: Data and Schema Preflight

- Confirm files, source URLs or DaRUS IDs, sizes, checksums or size/mtime manifests, license/access terms, and HDF5 keys.
- Record native spatial shape, time axis, channel count, dtype, value ranges, and whether the layout is grouped or monolithic.
- Write a durable preflight summary under `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_preflight.md`.
- Gate: suite-level benchmark claims remain blocked until all selected tasks are staged or explicitly marked infeasible/pivoted. Task-local implementation and full-training runs may proceed for tasks already confirmed loadable and native `128x128`, provided each task records its own data/schema gate before training.

Execution note, 2026-04-20: the first preflight pass found SWE ready and Darcy/2D diffusion-reaction missing under `/home/ollie/Documents/pdebench-data`. After staging Darcy, the user replaced diffusion-reaction with 2D Compressible Navier-Stokes and deleted the partial diffusion-reaction download. The local storage gate later passed with `75,243,819,008` bytes free before download start, so the preferred CNS file staging began under `/home/ollie/Documents/pdebench-data/2d_cfd_cns/`. Task-local Darcy full benchmark scheduling is allowed from the Darcy gate, but full three-task suite claims remain blocked until the selected official `128x128` CNS file is fully downloaded, checksum-verified, and inspected. The old diffusion-reaction missing-file row is historical and must not block the active CNS suite.

### Stage B: Shared Adapter and Tests

- Generalize the existing SWE data/model/metric/reporting surface only as far as needed for these three tasks.
- Add tests for task specs, HDF5 schema inspection, deterministic splits, channel-first datasets, normalization stats, metrics, run-budget validation, and reporting.
- CNS metric tests must include the frequency-band fRMSE contract before any benchmark run that interprets shock capture or high-frequency fidelity.
- Preserve existing SWE tests or migrate them with compatibility coverage.
- Gate: `pytest` collects and passes for existing SWE tests plus new image-suite tests.

### Stage C: Smoke Readiness for Each Task

- Run one tiny load/train/eval/write pass per task with Hybrid ResNet, FNO, and U-Net where buildable.
- Emit smoke summaries with:
  - `evidence_scope: smoke_feasibility_only`
  - `metric_interpretation: sanity_only_not_benchmark_performance`
  - `performance_assessment_complete: false`
- Gate: smoke confirms data loading, model execution, metric plumbing, and provenance writing. It cannot rank models or decide competitiveness.

### Stage D: Capped Pilot and Triage Runs

- Run same-protocol Hybrid ResNet, FNO, and U-Net on all three tasks with fixed splits, normalization, metrics, seed, and training recipe.
- Use a bounded pilot budget selected after Stage A reports dataset sizes and pair counts only to debug learning curves, runtime, memory, and logging.
- Dynamic tasks may cap trajectories and one-step pairs per trajectory initially; Darcy may cap train/validation/test samples initially.
- Escalate only tasks with plausible learning curves and clean runtime.
- Gate: capped pilot runs are decision-support evidence only. They do not count as meaningful benchmarks, cannot rank models for the paper, and cannot be used to reject Hybrid ResNet for competitiveness.

### Stage E: Full-Training Benchmark Runs

- Run same-protocol Hybrid ResNet, FNO, and U-Net on the full available training split for each task after validation/test holdout.
- Use the same deterministic splits, normalization, metrics, seed, training recipe, and evaluation horizon established before the pilot.
- For a task with `N` total examples, train on `N - N_val - N_test` examples. For trajectory datasets, apply this rule to the predeclared supervision unit: training trajectories and eligible one-step pairs for one-step tasks.
- Gate: at least two tasks must complete full-training primary profiles before the suite can support a meaningful PDEBench benchmark claim. The strongest outcome is all three tasks.
- If full-training execution is infeasible, document the blocker and route back to selector planning rather than promoting capped results.

### Stage F: Focused Ablations

Run after full-training primary profiles complete and only within remaining budget:

- Spectral-reduced Hybrid ResNet: `fno_modes=6`, all else fixed.
- Local-reduced Hybrid ResNet: `hybrid_resnet_blocks=2`, all else fixed.

If budget is tight, run ablations on the task where Hybrid ResNet is strongest and one task where it is weakest. Do not spend ablation budget on a task that failed data or baseline gates.

Optional model-family extension after the primary suite rows are stable:

- `spectral_resnet_bottleneck_base` as defined in `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_resnet_bottleneck_design.md`. This row is an additive architecture experiment, not a replacement for `hybrid_resnet_base`.

### Stage G: Suite Summary and Selector Update

- Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_summary.md`.
- Update `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_execution_summary.md` to point to the suite summary or mark the older SWE-only summary as superseded.
- Update `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` with the next selector-relevant decision.
- Do not write `/home/ollie/Documents/neurips/` artifacts until the roadmap artifact phase.

## Recommended Tranche Segmentation

The selector should decide exact next-plan scope from the roadmap, design, this plan, previous plans, and the live progress ledger. These are recommended coherent scopes, not a fixed manifest:

1. Data/schema preflight for Darcy and 2D Compressible Navier-Stokes plus confirmation that the existing SWE staged file still matches the plan.
2. Shared image-suite adapter design and SWE migration with tests.
3. Darcy support with static operator-map data tests, strong U-Net/FNO baseline gates, literature-calibrated reporting, and smoke readiness. Implemented on 2026-04-20; summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_darcy_static_operator_summary.md`. Full Darcy benchmark training remains pending and must use `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark/run_budget.json`.
4. 2D Compressible Navier-Stokes support with separate-field dynamic data tests, frequency-band fRMSE metric tests, and smoke readiness, using `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_design.md` as the design source of truth.
5. Capped pilot/triage run plan and execution, if needed to de-risk full training.
6. Full available training-set benchmark run plan and execution.
7. Ablation plan and execution for selected completed tasks.
8. Suite summary, roadmap interpretation, and selector ledger update.

The first next tranche should not launch expensive training. It should unblock exact data access, HDF5 schemas, and the minimal adapter design. After Darcy support implementation and the CNS pivot on 2026-04-20, a reasonable next selector action is either to stage/implement 2D CNS support or to schedule the Darcy full available training-split benchmark using the written budget; the broader three-task suite remains data-blocked until the selected 2D CNS file is staged.

## Verification Commands

Initial doc/plan validation:

```bash
python - <<'PY'
from pathlib import Path
path = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md")
text = path.read_text(encoding="utf-8")
required = ["swe", "darcy", "2d_cfd_cns", "128x128", "smoke_feasibility_only", "fno_modes=12"]
missing = [term for term in required if term.lower() not in text.lower()]
if missing:
    raise SystemExit(f"missing required plan terms: {missing}")
print("PDEBench image-suite plan contains required terms")
PY
```

Current SWE regression selectors before adapter migration:

```bash
pytest tests/studies/test_pdebench_swe_metrics.py tests/studies/test_pdebench_swe_splits_data.py tests/studies/test_pdebench_swe_run_config.py -v
```

After new image-suite tests exist:

```bash
pytest --collect-only tests/studies/test_pdebench_image128_*.py tests/studies/test_pdebench_darcy_*.py
pytest tests/studies/test_pdebench_image128_*.py tests/studies/test_pdebench_darcy_*.py -v
```

Long-running runs must use tmux and `ptycho311`, for example:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ptycho311
python scripts/studies/run_pdebench_image128_suite.py --task swe --smoke
```

## Blockers and Pivot Rules

Block the suite and return to selector planning if:

- Darcy or 2D Compressible Navier-Stokes cannot be staged locally or on an approved external data root.
- Either added task is not actually native `128x128` after schema inspection and no defensible resizing policy is approved.
- The shared adapter would require a broad rewrite beyond the current Phase 2 risk budget.
- FNO or U-Net cannot run for at least two suite tasks and no acceptable local-baseline substitute is approved.
- Full-training benchmark runs show Hybrid ResNet is clearly noncompetitive after the inherited recipe is used and obvious configuration defects are ruled out. Capped pilot runs may motivate debugging or budget changes, but they cannot by themselves establish noncompetitiveness.

Potential pivots:

- Complete a narrower SWE-only PDEBench result if SWE remains strong and the other two tasks fail operationally.
- Return to OpenFWI FlatVel-A as an adjacent inverse-wave extension if the user approves and storage/runtime are acceptable.
- Narrow the paper to CDI plus limited PDE feasibility only if no same-protocol PDE result is credible.
