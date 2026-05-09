## Completed In This Pass

- Staged the selected WaveBench member
  `wavebench_dataset/time_varying/is/thick_lines_gaussian_lens.beton`
  under the stable follow-up target
  `tmp/wavebench_repo/wavebench_dataset/time_varying/is/`, with durable size,
  mtime, SHA-256, and source-range provenance captured in
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-wavebench-provisioning-decision/dataset_manifest.json`.
- Normalized the durable path contract to the singular code path
  `wavebench_dataset/time_varying/is/` while explicitly recording the upstream
  README drift to `wavebench_datasets/`.
- Probed the available local Python environments and recorded that none of the
  existing local envs imported `ffcv`, `jax`, `jwave`, and `ml_collections`
  together; captured the recommended follow-up environment contract in
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-wavebench-provisioning-decision/environment_probe.json`.
- Recovered exact public Google Drive checkpoint identifiers for the native
  `is_gaussian_lens` FNO and U-Net baselines, then ran representative local
  load smokes:
  native U-Net `unet-ch-32` loaded successfully, while native FNO
  `fno-depth-4` failed with a Fourier-weight shape mismatch against the current
  upstream code.
- Published the durable checked-in decision summary, machine-readable
  provisioning decision, validator, and regression test surface for later
  WaveBench planners.

## Completed Plan Tasks

- Task 1 complete: authoritative inputs were re-read, the selected `.beton`
  member was staged locally, and the stable-target/path-normalization contract
  was written durably.
- Task 2 complete: required import surfaces were probed in the active PATH
  python and nearby local conda envs, a recommended follow-up environment was
  identified, and native baseline provenance was resolved into reusable U-Net
  versus retrain-required FNO.
- Task 3 complete: the checked-in summary, provisioning decision JSON,
  preflight forward link, docs index update, validator, and test were added.

## Remaining Required Plan Tasks

- None. Later WaveBench execution items remain gated or unblocked according to
  the published route matrix, but this provisioning-decision item has completed
  its scoped decision work.

## Verification

- Required preflight checks passed:
  `python - <<'PY' ...`, `python scripts/studies/validate_wavebench_preflight_contract.py`
- Dataset staging completed with the expected member name, stable target, and
  SHA-256 provenance.
- Representative checkpoint probes produced durable evidence in
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-wavebench-provisioning-decision/checkpoint_probe.json`.
- `python scripts/studies/validate_wavebench_provisioning_decision.py`
  passed.
- `pytest -q tests/studies/test_wavebench_preflight_contract.py tests/studies/test_wavebench_provisioning_decision_contract.py`
  passed with `6 passed in 0.95s`.
- JSON structure checks passed for `dataset_manifest.json`,
  `environment_probe.json`, `native_baseline_provenance.json`, and
  `provisioning_decision.json` via `python -m json.tool`.

## Residual Risks

- No local environment imported `ffcv`, `jax`, `jwave`, and `ml_collections`
  together in this pass, so the forward-model physics validation item remains
  blocked until a WaveBench-capable environment is actually provisioned.
- The public native FNO checkpoint is now identifiable and downloadable, but
  it is not currently drop-in reusable against the current upstream WaveBench
  code because of the observed Fourier-weight shape mismatch.
