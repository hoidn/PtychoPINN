Proposal mode: `explore`

In explore mode, the recent local neighborhood looks saturated.

- Prefer one credible hypothesis from a different family than the recent saturated cluster rather than another nearby perturbation of the champion.
- Good exploration families include stage-specific spectral schedules, encoder hidden-scale redistribution, encoder-vs-decoder capacity rebalancing, skip-projection or skip-fusion changes, and optimizer or scheduler adjustments that still respect the fixed dataset and `20`-epoch contract.
- Broader ideas are allowed here, but they must still be coherent, mechanistically justified, and expressible as one candidate rather than a bundle.
- Do not make a candidate “exploratory” by making it sloppy. Keep the same smoke-green and metadata-quality bar as exploit mode.
