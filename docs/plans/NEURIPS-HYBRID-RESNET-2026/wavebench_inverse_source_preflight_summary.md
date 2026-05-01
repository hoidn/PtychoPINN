# NeurIPS WaveBench Inverse Source Preflight Summary

## Decision

- Final status: `needs_dataset_or_checkpoint_decision`
- Selected variant: `time_varying/is/thick_lines_gaussian_lens`
- Selected split: seed-42 `train/val/test` permutation from
  `wavebench.dataloaders.is_loader.get_dataloaders_is_thick_lines`
- train/validation/test counts: `9000 / 500 / 500`

This preflight stayed inside the approved low-expense scope. It audited the
upstream WaveBench contract, selected the first runnable inverse-source
variant, recovered direct distributed-dataset evidence from the public archive,
separated supervised readiness from physics-readiness, and closed with a
follow-up decision instead of launching training. No roadmap amendment,
paper-row promotion, or benchmark execution was performed.

## Upstream Contract

- Repository URL: `https://github.com/wavebench/wavebench.git`
- Repository revision used for this preflight:
  `2bea258d9f05ec7182741293be11be1e545576ae`
- Dataset source: Zenodo record `https://zenodo.org/records/8015145`
  with DOI `10.5281/zenodo.8015145`
- Dataset archive observed from Zenodo: `wavebench_dataset.zip`,
  `75,625,669,111` bytes (`75.6 GB`), md5
  `1519b4f1ce1b9f594460ab27580809d5`, license `CC BY 4.0`
- Alternate dataset host named by upstream README:
  `https://drive.switch.ch/index.php/s/L7LeTyb9B9HMuG6`
- Native checkpoint host named by upstream README:
  `https://drive.google.com/drive/folders/11xLtRWP3q31ki_r4vsV09yM84BPffcJU?usp=sharing`
- Native checkpoint load path:
  `wavebench.nn.pl_model_wrapper.LitModel.load_from_checkpoint(...)`

Observed setup-risk mismatch:

- The README says to unzip `wavebench_datasets.zip` into a folder named
  `wavebench_datasets/`.
- The current Zenodo archive is named `wavebench_dataset.zip`.
- The code resolves `wavebench_dataset_path` as `<repo>/wavebench_dataset`
  (singular).

The preflight therefore treats `<wavebench repo>/wavebench_dataset/` as the
authoritative local staging path for any later smoke or training run, and it
records the README naming drift as an upstream contract risk that should be
normalized before follow-up execution.

## Selected Variant And Tensor Contract

The first runnable supervised target should be
`time_varying/is/thick_lines_gaussian_lens` because it keeps the fixed-medium
case procedural and avoids dependence on the external GRF wavespeed files for
the first local bring-up.

Important contract correction from this pass:

- The public Zenodo bundle distributes the selected variant as a single FFCV
  member:
  `wavebench_dataset/time_varying/is/thick_lines_gaussian_lens.beton`
- The generator-stage memmaps referenced by upstream conversion code,
  `thick_lines_gaussian_lens_initial_pressure_dataset.npy` and
  `thick_lines_gaussian_lens_boundary_measurement_dataset.npy`, are not
  distributed in the public archive and therefore cannot be treated as the
  authoritative staged-file contract for follow-up execution.

Observed tensor contract from the distributed archive member plus the upstream
loader/generation code:

- Raw observed boundary-time measurements `y`:
  `(334, 128)` `float32` per sample
- Distributed archive `input` tensor:
  `(1, 128, 128)` `float32`
  after nearest-neighbor interpolation from `(334, 128)` to `128 x 128`
- Raw target `q_0`:
  `(128, 128)` `float32` per sample
- Distributed archive `target` tensor:
  `(1, 128, 128)` `float32`
- Wavespeed field `c(x)` for the selected variant:
  fixed `gaussian_lens` `128 x 128` field synthesized in code and then scaled
  to `[1400, 4000]`
- Receiver layout:
  one top-boundary sensor trace taken from `p.on_grid[medium.pml_size, ...]`,
  producing `128` receiver positions over `334` time steps

Normalization and value-contract notes:

- Source images are loaded from PNG and divided by `255`.
- The distributed `target` field remains the original
  `128 x 128` image-space source field.
- The actual simulated source used for forward propagation is resized to
  `64 x 64` and inserted into the top-center region of a zero canvas before
  simulation.
- No additional measurement normalization is applied in the loader beyond the
  nearest-neighbor reshape to `128 x 128`.

Direct distributed-archive inspection was recovered in this pass by extracting
the public `.beton` member remotely, parsing the FFCV header locally, and
sampling one train, one validation, and one test example from the seed-42
split:

- Selected split sample ids:
  train `7542`, validation `59`, test `5238`
- Observed `input` value ranges:
  train `[-0.1639, 0.3550]`,
  validation `[-0.3119, 0.7100]`,
  test `[-0.0624, 0.1376]`
- Observed `target` value ranges:
  train `[0.0000, 1.0000]`,
  validation `[0.0000, 1.0000]`,
  test `[0.0000, 0.7804]`

This satisfies the missing dataset-slice inspection requirement for the
distributed archive contract while keeping the raw `334 x 128` shape explicit
as a generator/loader fact rather than silently pretending it was observed on
disk.

## Native Baseline Surface

The upstream training surface for the selected inverse-source lane is present:

- FNO: `src/train_time_varying/is/train_fno_is.py`
- U-Net: `src/train_time_varying/is/train_unet_is.py`
- UNO: `src/train_time_varying/is/train_uno_is.py`

The native baseline scripts use the thick-lines inverse-source loader, seed the
same `9000 / 500 / 500` split, and train through the shared Lightning wrapper.
However, the current checkout does not contain the upstream checkpoint files,
and the public Google Drive folder did not expose per-file identifiers through
the non-interactive probe used in this pass. As a result:

- FNO checkpoint status: `retrain_required`
- U-Net checkpoint status: `retrain_required`

This keeps the reuse path conceptually visible without overstating that exact
selected-variant checkpoints are already recoverable.

## Supervised Follow-Up Readiness

The supervised readiness question is now separated explicitly from the
physics-readiness question.

- `y(t,b)` can be treated as a stable 2D measurement image for the first
  follow-up row because the distributed archive already stores the model input
  as a deterministic `1 x 128 x 128` float32 image.
- In-repo precedent for the required real-channel adapter boundary already
  exists in
  `scripts/studies/pdebench_swe/models.py::HybridResnetSweModel`,
  `scripts/studies/pdebench_swe/models.py::SmallUNet`,
  `scripts/studies/openfwi_flatvel_a/models.py::HybridResnetSmoke`, and
  `scripts/studies/openfwi_flatvel_a/models.py::SmallUNet`.
- The missing work for a later supervised item is narrow:
  add a WaveBench-specific dataloader wrapper plus a boundary-measurement
  encoder that projects the archived `128 x 128` input image to the chosen
  latent width `C`, without reusing the CDI physics wrapper.
- Recommended first follow-up widths: `C=32` and `C=64`, with `C=32` as the
  minimum first serious row and `C=64` as the immediate sensitivity follow-up.

This is enough to authorize later supervised planning once the dataset and
native-baseline provisioning decision is resolved, but not enough to promote
the overall preflight to `ready_for_supervised_plan` yet.

## Local Compatibility

Current local status:

- No persistent local `wavebench_dataset/` tree was found under `/home/ollie`.
- The selected distributed archive member was nevertheless inspected directly
  from the public Zenodo ZIP in this pass; no long-lived local dataset copy was
  retained in the repo checkout.
- No local WaveBench baseline checkpoint files were found.
- The active PATH `python` environment is
  `/home/ollie/miniconda3/envs/ptycho311/bin/python` (`Python 3.11.13`).
- That environment currently imports `pytorch_lightning` and `cv2`, but not
  `ffcv`, `jwave`, `jax`, or `ml_collections`.
- GPU availability is not the blocker: `NVIDIA GeForce RTX 3090` with
  `24576 MiB` was visible via `nvidia-smi`.

Interpretation:

- Supervised follow-up is structurally plausible once the dataset is staged
  under the expected local path and the exact baseline-checkpoint decision is
  made.
- Physics-informed follow-up is not yet authorized because the forward-model
  reproduction check was not runnable in the current environment and no
  measured residuals were produced.

## Physics-Loop Readiness

Forward-model availability: `portable_with_narrow_work`

Evidence:

- The inverse-source generation path is checked into
  `wavebench/generate_data/time_varying/generate_data_is.py`.
- The forward simulation uses JAX plus J-Wave and defines the exact local
  sensor operator for the selected inverse-source task.
- The upstream test fixture
  `wavebench/generate_data/test/test_generate_is.py` already compares the
  J-Wave measurements against a MATLAB/k-Wave reference.

Physics-readiness classification: `physics_loop_deferred`

Observed reproduction fields for this pass:

- Reproduction-check sample count: `0`
- Waveform MAE: not measured
- Waveform RMSE: not measured
- relative L2: not measured
- normalized residual: not measured

Accepted thresholds carried forward from the approved design because no
override is justified yet:

- median normalized residual `<= 0.02`
- no more than `5%` of checked examples above `0.05`

The lack of a measured metric bundle is the reason this preflight does not
authorize any physics-informed benchmark row.

## Follow-Up Directions

Required next decisions before any later WaveBench implementation item should
start:

1. Stage the distributed `.beton` member under
   `<wavebench repo>/wavebench_dataset/time_varying/is/` and normalize the
   singular/plural path mismatch explicitly.
2. Resolve native baseline provenance by either:
   `a)` recovering the exact FNO and U-Net checkpoints for
   `thick_lines_gaussian_lens`, or
   `b)` choosing a from-scratch native baseline reproduction path.
3. Provision a dedicated WaveBench-capable environment with
   `ffcv`, `jax`, `jwave`, and `ml_collections` before any loader or
   forward-model smoke.
4. Only after those inputs exist:
   run the shared-encoder supervised benchmark with `C=32` and `C=64`, then
   the native-baseline reproduction item, then the forward-model validation
   item.
5. Do not start any WaveBench paper-bundle assembly item until the supervised,
   native-baseline, and physics-validation follow-ups produce claim-bearing
   evidence.

## Evidence-Index Note

No update was made to `evidence_matrix.md`, `model_variant_index.json`, or
`ablation_index.json` because this item produced no benchmark row, no paper
artifact, and no claim-bearing metric bundle. `docs/index.md` already points to
this durable preflight summary, and no broader paper-facing evidence index was
touched because Phase 5 evidence-bundle work remains out of scope.
