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
