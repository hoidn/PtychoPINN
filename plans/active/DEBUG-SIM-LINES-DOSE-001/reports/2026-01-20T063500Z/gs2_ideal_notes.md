# gs2_ideal Visual Notes
- amplitude.png (magma, vmax≈2.07) shows the expected SIM-LINES lattice with bright overlapping stripes; `stats.json` reports max≈2.07, mean≈0.21, and `nan_count=0`.
- phase.png (twilight, ±π) follows the amplitude geometry with smooth gradients and no obvious ringing; stats confirm finite values across the canvas.
- `run_metadata.json` records the baked profile (`stable_profile_gs2_ideal`: base_total_images=256, group_count=128, neighbor_count=4, batch_size=4) proving the reduced loads are now hard-coded rather than enforced via manual CLI flags.
- Reassembly telemetry (`reassembly_gs2_ideal.{json,md}`) verifies `padded_size=826` vs `required_canvas=826` with |offset|max≈381 px and zero loss ratio, so the jitter-driven padded-size helper continues to satisfy the spec under the new profile.
