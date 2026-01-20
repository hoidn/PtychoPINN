# gs1_ideal Visual Notes
- amplitude.png is effectively blank (vmax locked to 1e-9) and stats report `nan_count=23065`, so this rerun reproduced the NaN collapse seen in prior high-workload attempts even though the stable profile is now baked into the runner; the PNG and `.npy` dumps remain useful for documenting the failure state.
- phase.png also renders as zeros because inference emitted NaNs/zeros, matching the stats block (phase `nan_count=23065`). Train logs captured in `gs1_ideal_runner.log` show Lightning reporting `nan` losses almost immediately despite the reduced profile.
- The baked profile overrides are now captured directly in `run_metadata.json` (`stable_profile_gs1_ideal` with base_total_images=512, group_count=256, batch_size=8) instead of manual CLI overrides.
- `stats.json` still confirms `padded_size=828` and `required_canvas=828` with |offset|max≈382 px and `fits_canvas=true`, so the jitter fix is behaving even though the reconstruction failed numerically.
