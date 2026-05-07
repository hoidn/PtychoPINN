# Execution Plan: BRDT Sinogram-Input 40-Epoch Paper Evidence

## Prerequisite

Complete `2026-05-07-brdt-sinogram-input-adapter-contract`.

## Tasks

- Run `python -m scripts.studies.born_rytov_dt.run_sinogram_input_40ep` in the
  `ptycho311` environment, preferably in tmux for live output.
- Verify both learned rows complete 40 epochs and write loss history.
- Verify the non-learned Born inverse row is present only as a reference.
- Refresh BRDT manuscript assets from the new artifact root.
- Update evidence indexes and contract-supersession notes so the 2026-05-06
  Born-image-input bundles remain discoverable but no longer source
  sinogram-input claims.
- Rebuild the manuscript PDF and package zip.

## Verification

- `python -m scripts.studies.born_rytov_dt.run_sinogram_input_40ep --dry-run`
- `python -m scripts.studies.born_rytov_dt.run_sinogram_input_smoke`
- Full 40-epoch run completion with `input_mode=sinogram` in the config
  snapshot.
- Paper build has no missing BRDT tables or figures.
