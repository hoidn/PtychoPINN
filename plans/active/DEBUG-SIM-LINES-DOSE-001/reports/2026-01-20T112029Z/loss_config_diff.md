# SIM-LINES-4X vs dose_experiments Parameter Diff

- Snapshot source: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/sim_lines_4x_params_snapshot.json`
- Legacy defaults source: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/dose_experiments_param_scan.md` (init() assignments)
- `—` indicates the parameter was not defined in that pipeline.

## Scenario: gs1_custom

| Parameter | dose_experiments | sim_lines | Δ / note |
|-----------|------------------|-----------|----------|
| gridsize | 2 | 1 | -1 |
| probe_big | — | True | not set in dose_experiments |
| probe_mask | — | False | not set in dose_experiments |
| probe_scale | — | 4.0 | not set in dose_experiments |
| offset | 4 | — | not set in sim_lines |
| outer_offset_train | 8 | — | not set in sim_lines |
| outer_offset_test | 20 | — | not set in sim_lines |
| nimgs_train | 2 | 1000 | +998 |
| nimgs_test | 2 | 1000 | +998 |
| nphotons | nphotons | 1000000000.0 | differs |
| group_count | — | 1000 | not set in dose_experiments |
| neighbor_count | — | 4 | not set in dose_experiments |
| reassemble_M | — | 20 | not set in dose_experiments |
| intensity_scale.trainable | True | — | not set in sim_lines |
| total_images | — | 2000 | not set in dose_experiments |
| mae_weight | — | 0.0 | not set in dose_experiments |
| nll_weight | — | 1.0 | not set in dose_experiments |
| realspace_weight | — | 0.0 | not set in dose_experiments |
| realspace_mae_weight | — | 0.0 | not set in dose_experiments |

## Scenario: gs1_ideal

| Parameter | dose_experiments | sim_lines | Δ / note |
|-----------|------------------|-----------|----------|
| gridsize | 2 | 1 | -1 |
| probe_big | — | False | not set in dose_experiments |
| probe_mask | — | True | not set in dose_experiments |
| probe_scale | — | 10.0 | not set in dose_experiments |
| offset | 4 | — | not set in sim_lines |
| outer_offset_train | 8 | — | not set in sim_lines |
| outer_offset_test | 20 | — | not set in sim_lines |
| nimgs_train | 2 | 1000 | +998 |
| nimgs_test | 2 | 1000 | +998 |
| nphotons | nphotons | 1000000000.0 | differs |
| group_count | — | 1000 | not set in dose_experiments |
| neighbor_count | — | 4 | not set in dose_experiments |
| reassemble_M | — | 20 | not set in dose_experiments |
| intensity_scale.trainable | True | — | not set in sim_lines |
| total_images | — | 2000 | not set in dose_experiments |
| mae_weight | — | 0.0 | not set in dose_experiments |
| nll_weight | — | 1.0 | not set in dose_experiments |
| realspace_weight | — | 0.0 | not set in dose_experiments |
| realspace_mae_weight | — | 0.0 | not set in dose_experiments |

## Scenario: gs2_custom

| Parameter | dose_experiments | sim_lines | Δ / note |
|-----------|------------------|-----------|----------|
| gridsize | 2 | 2 | match |
| probe_big | — | True | not set in dose_experiments |
| probe_mask | — | False | not set in dose_experiments |
| probe_scale | — | 4.0 | not set in dose_experiments |
| offset | 4 | — | not set in sim_lines |
| outer_offset_train | 8 | — | not set in sim_lines |
| outer_offset_test | 20 | — | not set in sim_lines |
| nimgs_train | 2 | 4000 | +3998 |
| nimgs_test | 2 | 4000 | +3998 |
| nphotons | nphotons | 1000000000.0 | differs |
| group_count | — | 1000 | not set in dose_experiments |
| neighbor_count | — | 4 | not set in dose_experiments |
| reassemble_M | — | 20 | not set in dose_experiments |
| intensity_scale.trainable | True | — | not set in sim_lines |
| total_images | — | 8000 | not set in dose_experiments |
| mae_weight | — | 0.0 | not set in dose_experiments |
| nll_weight | — | 1.0 | not set in dose_experiments |
| realspace_weight | — | 0.0 | not set in dose_experiments |
| realspace_mae_weight | — | 0.0 | not set in dose_experiments |

## Scenario: gs2_ideal

| Parameter | dose_experiments | sim_lines | Δ / note |
|-----------|------------------|-----------|----------|
| gridsize | 2 | 2 | match |
| probe_big | — | False | not set in dose_experiments |
| probe_mask | — | True | not set in dose_experiments |
| probe_scale | — | 10.0 | not set in dose_experiments |
| offset | 4 | — | not set in sim_lines |
| outer_offset_train | 8 | — | not set in sim_lines |
| outer_offset_test | 20 | — | not set in sim_lines |
| nimgs_train | 2 | 4000 | +3998 |
| nimgs_test | 2 | 4000 | +3998 |
| nphotons | nphotons | 1000000000.0 | differs |
| group_count | — | 1000 | not set in dose_experiments |
| neighbor_count | — | 4 | not set in dose_experiments |
| reassemble_M | — | 20 | not set in dose_experiments |
| intensity_scale.trainable | True | — | not set in sim_lines |
| total_images | — | 8000 | not set in dose_experiments |
| mae_weight | — | 0.0 | not set in dose_experiments |
| nll_weight | — | 1.0 | not set in dose_experiments |
| realspace_weight | — | 0.0 | not set in dose_experiments |
| realspace_mae_weight | — | 0.0 | not set in dose_experiments |

## Legacy dose_experiments Loss Configuration (Runtime Captured)

The following weights were captured by executing the legacy `init()` function
with a stubbed `ptycho.params.cfg` for each `loss_fn` mode.

| Parameter | loss_fn='nll' (default) | loss_fn='mae' (conditional) |
|-----------|-------------------------|----------------------------|
| mae_weight | — | 1.0 |
| nll_weight | — | 0.0 |
| realspace_weight | — | — |
| realspace_mae_weight | — | — |

**Key insight:** When `loss_fn='nll'` (the default), the legacy script does **not**
set `mae_weight` or `nll_weight` explicitly — it relies on the underlying framework
defaults. The `mae_weight=1.0, nll_weight=0.0` values only apply when `loss_fn='mae'`.
