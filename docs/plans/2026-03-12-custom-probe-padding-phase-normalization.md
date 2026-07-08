## Goal

Add an explicit preserved-padding mode for custom probes, keep `pad_extrapolate` available as a separate option, then regenerate the `lines_256` dataset under the architecture-loop output tree and repoint the thin wrapper/docs to the regenerated pair.

## Why

The current `pad_extrapolate` path extrapolates phase across the larger padded canvas. That makes the phase image expand with `N` even though the padded probe support is meant to stay fixed at the source pixel scale inside a larger field for the fixed `lines_256` dataset. The intended preserved-padding behavior is:

- amplitude stays padded, so the probe support becomes relatively smaller as `N` grows
- phase stays attached to that same centered complex probe, so its visible support also becomes relatively smaller with the padded probe

In short: when the probe becomes tiny relative to `N`, both amplitude and phase should become tiny in the same way.

## Design

Split the behavior into two explicit modes:

1. `pad_preserve`
   - optionally smooth the source complex probe at source resolution
   - center-pad the complex probe into the target canvas with zeros
   - do not rescale the phase as `N` changes
2. `pad_extrapolate`
   - retain the amplitude-padding plus quadratic phase extrapolation option as a separate explicit mode

This keeps the extrapolation option available while fixing the scaling rule used for the regenerated `lines_256` dataset: `phase(n - center_N)` stays the same function regardless of `N`.

## Files

- Modify: `ptycho/workflows/grid_lines_workflow.py`
- Modify: `tests/test_grid_lines_workflow.py`
- Modify: `scripts/studies/run_lines_256_arch_experiment.py`
- Modify: `docs/studies/lines_256_dataset.md`

If the public CLI or runbook text still describes the old behavior after the implementation change, update the minimal affected docs in the same pass.

## Dataset regeneration target

Write the regenerated `lines_256` pair under:

- `outputs/lines_256_arch_improvement/datasets/N256/gs1/train.npz`
- `outputs/lines_256_arch_improvement/datasets/N256/gs1/test.npz`

Use the same invariant study settings already documented for `lines_256`:

- `N=256`
- `gridsize=1`
- `probe_source=custom`
- `probe_scale_mode=pad_preserve`
- `probe_smoothing_sigma=0.5`
- `nimgs_train=2`
- `nimgs_test=1`
- `nphotons=1e9`
- `size=392`
- `offset=4`
- `outer_offset_train=8`
- `outer_offset_test=20`

## Verification

1. Add a regression test that fails under the current implementation and passes only when the centered complex probe is preserved under padding.
2. Run the targeted grid-lines workflow test file.
3. Run the `lines_256` thin-wrapper tests.
4. Regenerate the dataset and inspect its probe visually.
5. Confirm the thin wrapper now points to the regenerated pair.
