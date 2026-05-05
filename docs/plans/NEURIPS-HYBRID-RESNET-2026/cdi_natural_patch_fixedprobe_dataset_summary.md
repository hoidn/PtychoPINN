# CDI Natural-Patch Fixed-Probe Dataset Summary

- Date: `2026-05-05`
- Backlog item: `2026-05-04-cdi-natural-patch-fixedprobe-dataset`
- Plan path: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-natural-patch-fixedprobe-dataset/execution_plan.md`
- Dataset id: `natural_patches128_fixedprobe_v1`
- Dataset root: `.artifacts/data/NEURIPS-HYBRID-RESNET-2026/natural_patches128_fixedprobe_v1/`
- State: `dataset_locked`

## Claim Boundary

- This item creates the frozen expanded-object CDI dataset prerequisite only.
- It does not train any model row, update manuscript result tables, or replace
  the existing `lines128` paper table authority
  (`lines128_paper_benchmark_summary.md`).
- Downstream consumers must reference this summary and the manifests under the
  dataset root; they must not reuse the lines128 grid-lines workflow path for
  expanded-object benchmarking without going through `adapter_contract.json`.

## Frozen Contract

- Resolution: `N=128`
- Total object cap: `<= 10_000`
- Locked split: `8_000 / 1_000 / 1_000` train/validation/test
- Split invariant: source-image partition before patch sampling, no source
  overlap across train/validation/test
- Probe lineage: `datasets/Run1084_recon3_postPC_shrunk_3.npz` →
  `pad_extrapolate_complex(target_N=128)` → `smooth_complex(sigma=0.5)`
  (default preprocessing inherited from the lines128 CDI authority via
  `ptycho.workflows.grid_lines_workflow.normalize_probe_transform_pipeline`
  with `scale_mode="pad_extrapolate"`, `probe_smoothing_sigma=0.5`)
  - Canonical pipeline string: `pad_extrapolate:128|smooth:0.5`
- Object encoding contract:
  - RGB → grayscale via Rec. 709 luminance
  - Normalize to `[0, 1]` over uint8 source range
  - Complex-object mapping: `amplitude = 0.5 + 0.5 * x`, `phase = pi * (x - 0.5)`
  - Object = `amplitude * exp(1j * phase)` with `dtype=complex64`
- Simulation contract: single-shot CDI exit-wave forward model
  - `exit_wave = probe * object`
  - `diffraction = |fftshift(fft2(exit_wave)) / sqrt(N**2)|` (`dtype=float32`)

## Source Corpus

- Source identity: scikit-image bundled natural-image data,
  `scikit-image==0.25.2` at generation time.
- Curated set (alphabetically sorted, providing stable source ordering):
  `astronaut`, `brick`, `camera`, `cat`, `chelsea`, `clock`, `coffee`,
  `coins`, `eagle`, `grass`, `gravel`, `hubble_deep_field`, `moon`, `page`,
  `retina`, `rocket`.
- Source-image partition (split seed = `1337`):
  - train (12 images): `camera`, `cat`, `clock`, `coffee`, `coins`, `eagle`,
    `grass`, `gravel`, `hubble_deep_field`, `moon`, `page`, `retina`
  - val (2 images): `chelsea`, `rocket`
  - test (2 images): `astronaut`, `brick`
- Crop seed: `4242` (deterministic top-left positions per source image).
- License: scikit-image data license (records preserved in
  `source_manifest.json`).

## Generated Output Inventory

Under `.artifacts/data/NEURIPS-HYBRID-RESNET-2026/natural_patches128_fixedprobe_v1/`:

- `dataset_manifest.json` (schema_version `natural_patches_v1`,
  `total_patches=10000`)
- `source_manifest.json`
- `split_manifest.json` (asserts `no_source_overlap = true`)
- `probe_manifest.json`
- `simulation_manifest.json`
- `adapter_contract.json`
- `train.npz` (8000 entries; keys: `objects`, `diffraction`, `crop_coords`,
  `source_ids`, `patch_ids`)
- `val.npz` (1000 entries)
- `test.npz` (1000 entries)
- `probe.npz` (single key `probeGuess`, complex64 128x128)
- `contact_sheet.png` (3 splits x 3 sample columns x object/diffraction rows)
- `samples/` (sample NPZs referenced by the contact sheet)
- `verification/post_audit.json`

## Downstream Consumer Note

- Consumer for this dataset is the deferred backlog item
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog-gaps/2026-05-04-cdi-natural-patch-expanded-benchmark.md`.
- Consumption must follow `adapter_contract.json`. Concretely, expanded-object
  benchmark adapters MUST construct one scan group per object patch with a
  single zero-coordinate, set `probeGuess` to `probe.npz['probeGuess']`, and
  derive `Y_I` / `Y_phi` from `objects` rather than regenerating diffraction
  with a different forward operator.

## Verification

- Builder tests: `pytest -q tests/studies/test_cdi_natural_patch_dataset.py`
  (12 tests, all pass; covers determinism, cap enforcement, manifest
  completeness, probe lineage, adapter-contract consumability, contract
  defaults).
- Compile gate: `python -m compileall -q scripts/studies ptycho_torch`
  (passes).
- Live generation: `python scripts/studies/run_cdi_natural_patch_dataset.py
  --dataset-root .artifacts/data/NEURIPS-HYBRID-RESNET-2026/natural_patches128_fixedprobe_v1`
  (runtime ~13 s on the local machine; emits `verification/post_audit.json`
  with `total_objects=10000`, `no_source_overlap=true`,
  `manifests_present=true`).

## Residual Risks

- scikit-image bundle size is small; cap is enforced inside `build_dataset` so
  the dataset cannot grow beyond `10_000` patches without a roadmap amendment.
- Adapter contract proves consumer compatibility; later expanded-object
  benchmark work must consume the NPZs through the documented contract rather
  than reaching into raw arrays directly.
- Provenance for the source corpus depends on the `scikit-image` package
  version captured in `source_manifest.json["package_version"]`; later
  regeneration with a different `scikit-image` release should record any
  bundled-image hash drift as a separate provenance note rather than silently
  overwriting the locked dataset.
