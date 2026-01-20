# gs1_ideal Visual Notes
- amplitude.png (magma, vmin=0, vmax≈2.08) shows the expected bright SIM-LINES lattice with continuous stripes radiating across the canvas; no clipping or blank regions were visible after limiting the workload.
- phase.png (twilight, ±π) exhibits smooth ring-like gradients that align with the amplitude structure; stats confirm `nan_count=0` for both tensors.
- Stats JSON now reports padded_size=828 and required_canvas=818 with |dx|max≈377 px (`fits_canvas=true`), so the jitter-driven padding still covers the observed offsets.
- To keep training stable (avoid the NaNs seen in the full 1000-group run) this pass used the reduced configuration documented in `run_metadata.json` (`--base-total-images 512 --group-count 256 --batch-size 8 --group-limit 64`); inference artifacts (npy, png, stats, run_metadata.json) live under `gs1_ideal/inference_outputs/`.
