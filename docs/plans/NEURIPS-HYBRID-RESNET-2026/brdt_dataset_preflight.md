# BRDT Dataset Preflight Summary

## Identity

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-29-brdt-dataset-preflight`
- Plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-dataset-preflight/execution_plan.md`
- Tier: `feasibility` (additive candidate work; not manuscript evidence)
- Generator: `scripts/studies/born_rytov_dt/generate_brdt_dataset.py`
- Artifact root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-dataset-preflight/`
- Machine-readable artifacts:
  - `dataset_manifest.json`
  - `dry_run_manifest.json`
  - `dry_run_summary.json`
  - `dataset/brdt128_sparse_fullview_preflight_train.h5`
  - `dataset/brdt128_sparse_fullview_preflight_val.h5`
  - `dataset/brdt128_sparse_fullview_preflight_test.h5`

## Prerequisite Status

The operator-validation gate
(`docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_operator_validation_report.md`,
verdict `pass_with_documented_limits`) is the binding authority for the
operator contract this dataset consumes. The corresponding machine-
readable artifact
`.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-operator-validation/operator_validation.json`
is consumed directly by the dry-run and live generator paths; the
generator refuses to proceed if its `verdict` is not pass-equivalent or
if the recorded operator geometry mismatches the locked smoke geometry.

## Locked Physical Target And Normalization Rule

The dataset stores both the physical scattering potential

```math
q(x,z) = k_m^2 \left(\left(\frac{n(x,z)}{n_m}\right)^2 - 1\right)
```

and a normalized convenience tensor `q_true_norm`. The forward operator
always consumes physical `q`. Any future physics loss must obey

```math
L_{\mathrm{phys}} = \left\| A(\mathrm{unnormalize}(\hat q_{\mathrm{norm}})) - y \right\|.
```

The manifest carries this rule explicitly under
`physical_target.physics_loss_rule` and the dataset-contract module
exposes `dataset_contract.reject_normalized_q_to_operator(...)` as a
hard guard so a downstream call site that tries to feed normalized `q`
to the operator surfaces a `ValueError` rather than silently drifting
units.

## Locked Smoke Geometry

Copied verbatim from the operator validation authority:

| Field | Value |
| --- | --- |
| `mode` | `born` |
| `normalize` | `unitary_fft` |
| `grid_size` | `128` |
| `detector_size` | `128` |
| `angle_count` | `64` (full view, `np.linspace(0, 2*pi, 64, endpoint=False)`) |
| `wavelength_px` | `8.0` |
| `medium_ri` | `1.333` |
| Output layout | `(B, A, D, 2)` real/imag |

The dry-run mode validates these fields against the operator authority
before any arrays are generated and emits `verdict:
ready_for_smoke_generation` only when the comparison succeeds.

## Split, Phantom, And Noise Contract

- Split counts: `16 train / 4 val / 4 test` (locked via
  `dataset_contract.SplitCounts`).
- Split seed: `42` by default; reproduced via
  `dataset_contract.deterministic_object_seeds`.
- Object seeds: drawn from a deterministic disjoint pool so train, val,
  and test object-seed sets are disjoint by construction; verified by
  the test suite.
- Phantom families (non-CDI by design):
  - `overlapping_ellipses`
  - `soft_blobs`
  - `sparse_inclusions`
- Refractive-index contrast envelope: `delta_n in [0.002, 0.03]` with
  `n_m = 1.333`.
- Noise model: additive complex Gaussian with `noise_sigma = 1e-3` in
  physical sinogram units (recorded in the manifest); the generator
  seeds the per-split noise deterministically and records the measured
  per-split SNR in dB.

For the default seed (42) the recorded SNR is approximately
`train ~18.5 dB`, `val ~16.4 dB`, `test ~19.3 dB`.

## HDF5 Schema

Each split file exposes:

| Dataset | Shape | Dtype |
| --- | --- | --- |
| `q_true_physical` | `(N, 128, 128)` | `float32` |
| `q_true_norm` | `(N, 128, 128)` | `float32` |
| `sinogram_real` | `(N, 64, 128)` | `float32` |
| `sinogram_imag` | `(N, 64, 128)` | `float32` |
| `sinogram_clean_real` | `(N, 64, 128)` | `float32` |
| `sinogram_clean_imag` | `(N, 64, 128)` | `float32` |
| `angle_mask` | `(64,)` | `float32` |
| `angles_rad` | `(64,)` | `float64` |
| `sample_seed` | `(N,)` | `int64` |
| `phantom_family` | `(N,)` | bytes (S32) |

File-level attributes carry the dataset name, split, sample count, the
physics-loss rule string, and the `forward_input_is_physical_q` /
`model_output_space` flags. The `sinogram_real` / `sinogram_imag`
arrays carry the noisy sinograms; the clean copies are kept separately
for diagnostics and noiseless operator tests only.

## Dry-Run Outcome

Running

```bash
python -m scripts.studies.born_rytov_dt.generate_brdt_dataset --dry-run-manifest
```

with the default split seed produces both `dry_run_summary.json` and
`dry_run_manifest.json`. The summary reports
`verdict: ready_for_smoke_generation`, zero geometry mismatches against
the operator authority, the estimated artifact paths, the exact
generation command, and the requested `noise_sigma` in physical
sinogram units. The manifest skeleton mirrors the live manifest
schema with `normalization: null`, the same requested
`noise_sigma_physical_units`, `measured_snr: null`, and
`extra.generation_mode: dry_run_manifest` so downstream tooling can
consume a concrete dry-run contract rather than inferring one.

## Reproducing The Smoke Dataset

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ptycho311
python -m scripts.studies.born_rytov_dt.generate_brdt_dataset --dry-run-manifest
python -m scripts.studies.born_rytov_dt.generate_brdt_dataset
```

Generation runtime is short (well under a minute on CUDA). For a fully
fresh artifact root, pass `--output-root <path>`; the generator creates
`logs/`, `dataset/`, `dataset_manifest.json`, `dry_run_manifest.json`,
and `dry_run_summary.json` under that root.

Validation gates required by the plan:

```bash
pytest -q tests/studies/test_born_rytov_dt_dataset.py
python -m compileall -q scripts/studies/born_rytov_dt
```

Both must succeed.

## Claim Boundary

This item is **feasibility-only**. It does not authorize BRDT manuscript
evidence, does not register BRDT as a normal CDI generator, does not
add Lightning modules / dataloaders / metric wrappers, does not run the
four-row preflight, and does not generate the later larger
decision-support split. Those remain follow-up work in
`2026-04-29-brdt-task-adapters` and `2026-04-29-brdt-four-row-preflight`.

CDI `lines128` and PDEBench CNS remain the required manuscript pillars;
BRDT cannot be promoted into manuscript evidence without a separately
checked-in roadmap or evidence-package amendment.

## Handoff

Downstream consumers should:

- consume the operator contract from `operator_validation.json`
  (`operator` block) — not from this summary or the candidate-lane
  design — to avoid contract drift;
- consume the dataset contract via the `dataset_contract` module helpers
  (`build_manifest`, `compute_train_normalization`, `normalize_q`,
  `unnormalize_q`, `reject_normalized_q_to_operator`) so any future
  schema or normalization change is caught at one location;
- treat the smoke dataset as preflight-only and re-run the generator
  against a larger split before any decision-support row.
