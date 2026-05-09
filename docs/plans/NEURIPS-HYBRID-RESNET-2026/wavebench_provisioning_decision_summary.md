# NeurIPS WaveBench Provisioning Decision Summary

## Decision

- Selected variant: `time_varying/is/thick_lines_gaussian_lens`
- Stable dataset target: `<wavebench repo>/wavebench_dataset/time_varying/is/`
- Staged dataset member: `wavebench_dataset/time_varying/is/thick_lines_gaussian_lens.beton`
- Dataset staging result: `staged_locally`
- Environment recommendation: upstream `wavebench` conda recipe, extended with explicit `jax` and `jwave` installation before any physics-loop run
- Native FNO decision: `retrain_required`
- Native U-Net decision: `checkpoint_reusable`

This pass resolves the preflight outcome `needs_dataset_or_checkpoint_decision`
into a durable provisioning contract. The selected `.beton` member is now staged
locally under the stable singular code path
`wavebench_dataset/time_varying/is/`, the README-versus-code naming drift is
made explicit (`wavebench_datasets/` in the README versus `wavebench_dataset/`
in code), exact public Google Drive checkpoint IDs were recovered, and the
native baseline split is now concrete rather than speculative: the
representative official U-Net checkpoint loads on the current upstream
WaveBench checkout, while the representative official FNO checkpoint downloads
but does not load against the current FNO layer contract.

## Dataset Staging

- WaveBench checkout reused: `tmp/wavebench_repo` at revision
  `2bea258d9f05ec7182741293be11be1e545576ae`
- Staged path:
  `tmp/wavebench_repo/wavebench_dataset/time_varying/is/thick_lines_gaussian_lens.beton`
- Size: `1376211712` bytes
- SHA-256:
  `ca1b2b707a25e7549303f31d1fc2c32a442d14fded63f61dd6b20908235ac37f`
- Source: Zenodo record `https://zenodo.org/records/8015145`, extracted from
  `wavebench_dataset.zip` by remote ZIP-member staging rather than full-archive
  download

Durable path normalization for later items:

- Upstream README still documents `wavebench_datasets/`.
- Current code resolves `wavebench_dataset/` through `wavebench.__init__.py`.
- Later WaveBench commands in this repo should therefore target
  `wavebench_dataset/time_varying/is/`.

## Environment And Imports

Observed local status in this pass:

- Active PATH `python`:
  `/home/ollie/miniconda3/envs/ptycho311/bin/python` (`Python 3.11.13`)
- Existing local env probes (`ptycho311`, `ptycho311_2`, `ptycho2`) all failed
  to import `ffcv`, `jax`, `jwave`, and `ml_collections` together
- Local supervised-loader readiness: `false`
- Local physics readiness: `false`

Recommended follow-up environment contract:

- Start from the upstream README `wavebench` conda recipe (`python=3.9`,
  PyTorch, OpenCV, `ffcv`, `ml-collections`, Lightning)
- Add explicit `jax` and `jwave` installation before any
  `2026-04-29-wavebench-forward-model-physics-validation` work

Interpretation:

- This item now records a usable environment recommendation, but it does not
  claim that a local WaveBench-capable environment was provisioned in this pass.
- That remaining gap blocks physics validation, not the provisioning decision
  itself.

## Native Baselines

Recovered official public checkpoint structure for `is_gaussian_lens`:

- FNO folders:
  `fno-depth-4` (`1Id2-BrE9md3ypqHysTITYVGMfQQ3qsbJ`),
  `fno-depth-8` (`1awl1usPolKY_PgNWs3CQowI18pzNK2n3`)
- U-Net folders:
  `unet-ch-32` (`17FXa31RSMA-7vwRB_492Ex8AY-2YgSdl`),
  `unet-ch-64` (`1HtrvoR5GqDWiLZWmt3KNK65PCv2Lj5uF`)

Representative load-smoke results:

- Native FNO `fno-depth-4`: `retrain_required`
  because `LitModel.load_from_checkpoint(...)` failed on the current upstream
  checkout with Fourier-weight shape mismatch
  (`[64, 64, 16, 16]` in the checkpoint versus
  `[64, 64, 16, 16, 2]` expected locally)
- Native U-Net `unet-ch-32`: `checkpoint_reusable`
  because the public checkpoint loaded successfully on CPU through the current
  upstream `LitModel`

Durable execution routes:

- Native FNO retraining surface:
  `tmp/wavebench_repo/src/train_time_varying/is/train_fno_is.py --medium_type gaussian_lens --num_layers 4`
- Native U-Net reuse surface:
  public `unet-ch-32` checkpoint or a fresh local rerun via
  `tmp/wavebench_repo/src/train_time_varying/is/train_unet_is.py --medium_type gaussian_lens --channel_reduction_factor 2`

## Downstream Route Matrix

- `2026-04-29-wavebench-native-baseline-reproduction`: `unblocked`
  because the dataset is staged, the checkpoint IDs are explicit, and the split
  between reusable U-Net versus retrain-required FNO is durable.
- `2026-04-29-wavebench-shared-encoder-supervised-benchmark`: `unblocked`
  because the stable dataset target, path normalization decision, environment
  recommendation, and native-baseline status are now explicit inputs.
- `2026-04-29-wavebench-forward-model-physics-validation`: `still_blocked`
  because no local environment imported `ffcv`, `jax`, `jwave`, and
  `ml_collections` together in this pass, and the upstream environment recipe
  still needs an explicit `jax`/`jwave` extension.
- `2026-04-29-wavebench-hybrid-physics-rows`: `still_blocked`
  because it still depends on both completed supervised WaveBench results and a
  passed forward-model validation report.
- `2026-04-29-wavebench-paper-table-figure-bundle`: `still_blocked`
  because WaveBench remains an additive candidate lane with no claim-bearing
  evidence bundle yet.

## Evidence Pointers

- Dataset manifest:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-wavebench-provisioning-decision/dataset_manifest.json`
- Environment probe:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-wavebench-provisioning-decision/environment_probe.json`
- Native baseline provenance:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-wavebench-provisioning-decision/native_baseline_provenance.json`
- Provisioning decision JSON:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-wavebench-provisioning-decision/provisioning_decision.json`
- Representative checkpoint load probe:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-wavebench-provisioning-decision/checkpoint_probe.json`
