# gs2_ideal Visual Notes
- amplitude.png (magma, vmin=0, vmax≈2.28) shows the same coherent multi-line structure as gs1 but with the denser gs2 tiling; no saturated tiles or blank gutters appeared despite the reduced sample count.
- phase.png (twilight) retains smooth alternating bands aligned with the amplitude stripes, and stats again show `nan_count=0`.
- The stats file records padded_size=822 vs required_canvas=817 with |dx|max≈376 px, so the updated jitter logic still keeps `fits_canvas=true` for the ideal gs2 offsets.
- To avoid GPU OOM while honoring the “never train on CPU” rule, this run used a reduced workload (base_total_images=256, group_count=128, batch_size=4) before the `--group-limit 64` inference slice; those overrides are documented in `gs2_ideal_runner.log` and `run_metadata.json`.
