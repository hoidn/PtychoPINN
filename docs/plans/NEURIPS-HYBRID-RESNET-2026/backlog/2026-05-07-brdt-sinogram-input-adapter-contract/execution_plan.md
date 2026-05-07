# Execution Plan: BRDT Sinogram-Input Adapter Contract

## Tasks

- Update BRDT row-schema support for `input_mode="sinogram"` while preserving
  historical `born_init_image` support.
- Add channels-first sinogram tensor conversion.
- Add sinogram-input wrappers for FFNO and SRU-Net.
- Update training/evaluation/preflight input preparation so sinogram rows use
  measured sinograms directly.
- Add tests for input-mode validation, adapter forward shapes, and runner
  dry-run contract output.
- Run focused verification commands from the backlog item.

## Non-Goals

- Do not remove historical Born-image-input runners.
- Do not promote old Born-image-input artifacts to sinogram-input evidence.
- Do not run the 40-epoch experiment in this adapter-contract item.
