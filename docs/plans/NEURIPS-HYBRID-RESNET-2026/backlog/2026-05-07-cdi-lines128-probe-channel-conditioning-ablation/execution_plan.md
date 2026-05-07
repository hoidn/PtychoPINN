# Execution Plan: Lines128 CDI Probe-Channel Conditioning Ablation

## Goal

Implement and run an append-only Lines128 CDI ablation that concatenates the
fixed complex probe to SRU-Net and corrected pure FFNO inputs.

## Required Contract

- Use the completed Lines128 paper benchmark as the baseline authority.
- Launch only:
  - `pinn_hybrid_resnet_probe_channels`;
  - `pinn_ffno_probe_channels`.
- Append exactly two input channels: preprocessed probe real and imaginary
  parts, broadcast per sample.
- Hold dataset, split, probe preprocessing, seed, epochs, scheduler, loss,
  output mode, metrics, visual sample ids, and visual scale policy fixed.
- Compare against completed unconditioned SRU-Net and corrected pure FFNO rows
  by lineage; do not rerun those rows unless an audit proves they are unusable.

## Implementation Steps

1. Add explicit runner/config support for CDI input-conditioning channels.
   Keep the default path unchanged.
2. Route probe-channel conditioning through the data/model boundary explicitly,
   not through hidden module globals.
3. Add focused tests that prove row ids enable the extra channels, record the
   probe-channel config, and keep unconditioned rows unchanged.
4. Run the two fresh rows under the fixed Lines128 contract.
5. Write the durable summary and update evidence indexes.

## Verification

Run the backlog `check_commands`, plus any narrower row-id tests added during
implementation. The final summary must include metric deltas versus the
unconditioned baselines and paths to row-local invocation/config artifacts.
