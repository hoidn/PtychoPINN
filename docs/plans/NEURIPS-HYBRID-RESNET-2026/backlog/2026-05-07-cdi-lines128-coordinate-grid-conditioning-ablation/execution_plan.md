# Execution Plan: Lines128 CDI Coordinate-Grid Conditioning Ablation

## Goal

Implement and run an append-only Lines128 CDI ablation that concatenates fixed
unit-grid coordinate channels to SRU-Net and corrected pure FFNO inputs.

## Required Contract

- Use the completed Lines128 paper benchmark as the baseline authority.
- Launch only:
  - `pinn_hybrid_resnet_grid_channels`;
  - `pinn_ffno_grid_channels`.
- Append exactly two input channels: `y` and `x` unit coordinates on `[0, 1]`.
- Do not combine grid conditioning with probe conditioning in these rows.
- Hold dataset, split, probe preprocessing, seed, epochs, scheduler, loss,
  output mode, metrics, visual sample ids, and visual scale policy fixed.
- Compare against completed unconditioned SRU-Net and corrected pure FFNO rows
  by lineage; do not rerun those rows unless an audit proves they are unusable.

## Implementation Steps

1. Add explicit runner/config support for deterministic CDI coordinate-grid
   input channels. Keep the default path unchanged.
2. Reuse the same unit-grid convention as the PDEBench authored-FFNO adapter
   unless a reviewed plan amendment records a different convention.
3. Add focused tests that prove row ids enable the extra channels, record the
   grid-channel config, and keep unconditioned rows unchanged.
4. Run the two fresh rows under the fixed Lines128 contract.
5. Write the durable summary and update evidence indexes.

## Verification

Run the backlog `check_commands`, plus any narrower row-id tests added during
implementation. The final summary must include metric deltas versus the
unconditioned baselines and paths to row-local invocation/config artifacts.
