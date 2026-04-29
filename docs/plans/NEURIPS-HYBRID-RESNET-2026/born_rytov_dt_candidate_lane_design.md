# Born/Rytov Diffraction Tomography Candidate Lane Design

## Context And Status

- Status: draft design
- Date: 2026-04-29
- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Scope: candidate third evidence lane only
- Governing roadmap:
  - `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/inverse_wave_benchmark_rationale.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/wavebench_inverse_source_benchmark_design.md`

This document designs a Born/Rytov 2D diffraction tomography lane as an
additional inverse-wave benchmark candidate. It does not replace CDI
`lines128` or PDEBench CNS. CDI remains the primary inverse-imaging pillar and
CNS remains the current cross-domain dynamics pillar. Any Born/Rytov diffraction
tomography result can enter the paper only after a later checked-in roadmap or
evidence-package amendment explicitly authorizes it.

The intended role is narrow:

> Born/Rytov diffraction tomography is a cheap-forward, PINN-compatible inverse
> scattering preflight. It may become a third paper table only if it passes
> strict operator, dataset, adapter, baseline, and provenance gates.

## Decision Summary

Add Born/Rytov 2D diffraction tomography as a gated candidate lane because it
has the right computational and scientific shape for this manuscript:

- it is a genuine inverse-scattering task;
- its forward operator is FFT-heavy and differentiable;
- its target is image-like and compatible with SRU-Net/Hybrid-family models;
- it is much cheaper than differentiating through full-wave time-domain FWI;
- it is closer to CDI than WaveBench inverse source in measurement geometry and
  inverse-wave interpretation.

The first milestone is not a full benchmark. It is an operator/data/model
preflight with four rows:

- `classical_born_backprop`
- `unet`, supervised plus Born consistency
- `fno_vanilla`, supervised plus Born consistency
- `sru_or_hybrid`, supervised plus Born consistency

All other rows, Rytov mode, limited-angle stress tests, FFNO, physics-only
training, multi-seed robustness, and external FDTD mismatch checks are optional
promotion work after the preflight passes.

## External Anchor

ODTbrain is the external reference for the physical model and classical
reconstruction side. Its documentation describes 2D and 3D optical diffraction
tomography reconstruction under Born and Rytov approximations, including the
Fourier diffraction theorem, `backpropagate_2d`, `fourier_map_2d`, and the
object function

```math
f(x,z) = k_m^2 \left(\left(\frac{n(x,z)}{n_m}\right)^2 - 1\right).
```

ODTbrain also warns that the detector-distance parameter `lD` should not be
used directly with the Rytov approximation; Rytov data must be numerically
refocused before conversion. This design therefore starts with Born mode and
allows Rytov only after a separate preprocessing gate.

External sources:

- ODTbrain introduction:
  <https://odtbrain.readthedocs.io/en/stable/sec_introduction.html>
- ODTbrain 2D inversion:
  <https://odtbrain.readthedocs.io/en/stable/recon_2d.html>
- ODTbrain 3D inversion note for object-function and Rytov/refocus semantics:
  <https://odtbrain.readthedocs.io/en/stable/recon_3d.html>

## Physical Model And Units

The first implementation is a scalar, monochromatic, weak-scattering 2D
diffraction tomography model. It is not a full Maxwell, acoustic, or elastic
wave solver.

Coordinate convention:

- reconstruction plane: `(x, z)`;
- object grid: square `N x N`, default `N=128`;
- detector coordinate: `x_D`, sampled with `D=128` points;
- illumination angle: `theta`;
- surrounding-medium refractive index: `n_m`;
- vacuum wavelength: `lambda_0`, if physical units are recorded;
- wavelength in medium: `lambda_m = lambda_0 / n_m`;
- medium wave number: `k_m = 2*pi / lambda_m`;
- implementation parameter: `wavelength_px`, the wavelength expressed in grid
  pixels so the synthetic first contract can stay unitless and reproducible.

The physical object is the real scattering potential:

```math
q(x,z) = k_m^2 \left(\left(\frac{n(x,z)}{n_m}\right)^2 - 1\right).
```

For weak refractive-index contrast:

```math
q(x,z) \approx 2 k_m^2 \frac{n(x,z)-n_m}{n_m}.
```

The dataset may store refractive index `n(x,z)`, but the canonical target for
training and metrics is `q`. Any conversion from `q` back to refractive index
must use the recorded `k_m` and `n_m`:

```math
n(x,z) = n_m \sqrt{1 + q(x,z)/k_m^2}.
```

Rows with negative values inside the square root because of model error must
record the clipping or invalid-pixel policy before reporting refractive-index
metrics. `q` metrics remain the default because they avoid that ambiguity.

Forward-model assumptions:

- scalar field;
- single scattering under Born mode;
- homogeneous surrounding medium;
- no multiple scattering;
- no absorption in the first contract;
- no detector-distance propagation unless explicitly modeled;
- periodic or padded FFT convention recorded in the operator manifest;
- all geometry and normalization choices fixed before dataset generation.

The first synthetic contract should use unitless pixel coordinates. If SI units
are later introduced, the manifest must record pixel size, wavelength, and
refractive-index scale so `q` remains dimensionally consistent.

## Roadmap Boundary

This design does not authorize training runs that compete for CDI/CNS deadline
budget. The only work it authorizes before a roadmap amendment is:

- implement or prototype operator correctness checks;
- inspect dataset feasibility for synthetic Born/Rytov data generation;
- draft a minimal adapter plan;
- record whether the lane is worth scheduling later.

Promotion requires a checked-in amendment that states:

- why CDI and CNS gates are secure enough to spend effort here;
- whether Born/Rytov DT is included in the manuscript, supplement, or backlog
  only;
- the exact rows, datasets, metrics, and claim boundaries being promoted.

## Task Definition

Recover a 2D weak-scattering object from multi-angle complex scattered-field
measurements.

For each sample, the physical target is the real scattering potential

```math
q(x,z) = k_m^2 \left(\left(\frac{n(x,z)}{n_m}\right)^2 - 1\right),
```

where `n(x,z)` is refractive index, `n_m` is the surrounding-medium refractive
index, and `k_m` is the wave number in the medium.

For each observed angle `theta`, the measurement is a complex detector-line
field:

```math
y_\theta(x_D) = A_\theta(q)(x_D) + \epsilon.
```

The model receives a fixed representation derived from the measurements and
predicts:

```math
\hat q(x,z) \in R^{1 \times 128 \times 128}.
```

The physics loss compares the differentiable Born/Rytov forward prediction to
the observed complex sinogram.

## Primary Benchmark Contract

The first serious contract is sparse full-view Born diffraction tomography:

- name: `brdt128_sparse_fullview`
- grid: `128 x 128`
- detector samples: `128`
- observed angles: `64`
- angle coverage: full `0` to `2*pi`
- target: real scattering potential `q(x,z)`
- measurement: complex linearized sinogram, stored as real/imag channels
- noise: fixed complex Gaussian noise level, recorded in manifest
- input mode: classical initialization image plus optional confidence/mask
  channel
- output mode: real `q` map
- forward mode: `born`
- default medium refractive index: `n_m = 1.333`
- default refractive-index contrast: `delta_n in [0.002, 0.03]`
- default wavelength: recorded as `wavelength_px`; preflight should choose a
  value that gives valid Ewald-arc coverage on the `128 x 128` grid
- stored dtype: `float32`
- operator-validation dtype: `float64` on CPU for numerical checks and
  `float32` on CUDA for training smoke

Limited-angle data is a later stress contract, not the first benchmark. The
operator, normalization, and adapter must be proven on full-view data before
missing-wedge behavior is introduced.

## Forward Operator

Add a task-local differentiable operator:

```text
ptycho_torch/physics/born_rytov_dt.py
```

Primary class:

```python
class BornRytovForward2D(torch.nn.Module):
    def __init__(
        self,
        grid_size: int,
        detector_size: int,
        angles: torch.Tensor,
        wavelength_px: float,
        medium_ri: float,
        mode: Literal["born", "rytov_linearized"] = "born",
        normalize: Literal["unitary_fft", "odtbrain_compatible"] = "unitary_fft",
        device: torch.device | None = None,
    ): ...

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        q: (B, 1, N, N), physical scattering potential
        returns: (B, A, D, 2), complex sinogram as real/imag
        """
```

The implementation should use the Fourier diffraction theorem:

1. Compute `Q = fft2(q)`.
2. For each angle, sample `Q` on the angle-dependent Ewald arc.
3. Apply the selected normalization and geometric weights.
4. Inverse FFT along detector frequency to produce a complex detector-line
   measurement.
5. Return real/imag channels, not Python complex objects.

Implementation requirements:

- use `torch.fft.fft2` and `torch.fft.ifft`;
- use `torch.nn.functional.grid_sample` for differentiable spectral sampling;
- register angle grids and Fourier coordinates as buffers;
- support deterministic CPU/GPU execution under fixed dtype/device;
- document the coordinate convention and FFT normalization;
- avoid Python-side loops over batch dimension.

## Operator Validation Gate

The operator cannot be trusted just because training losses decrease. Synthetic
data generated by the same wrong operator would be self-consistent but
physically meaningless. Before any neural row is trained, write an operator
validation report with:

- analytic point-phantom or Gaussian-blob checks where spectral support is easy
  to inspect;
- tiny-grid direct Born integral comparison against the Fourier-theorem
  implementation;
- ODTbrain inverse consistency checks using `backpropagate_2d` or
  `fourier_map_2d` on generated sinograms;
- finite-difference or `gradcheck` validation for the PyTorch operator;
- dtype/device reproducibility checks for CPU and CUDA where available;
- explicit tolerances, sample counts, and failure behavior.

Minimum pass criteria:

- direct tiny-grid comparison has relative error below a predeclared tolerance;
- ODTbrain reconstruction from generated sinograms recovers the low-frequency
  structure of `q` on at least 16 held-out phantoms;
- finite-difference and autograd gradients agree within tolerance for a small
  differentiable test case;
- the validation report records any known scale or phase convention offset.

If this gate fails, no Born/Rytov row may be used as benchmark evidence.

## Normalization Contract

The physical operator consumes physical scattering potential, not arbitrary
standardized values. Dataset normalization must therefore be explicit.

Preferred contract:

- store physical `q_true_physical`;
- store train-only normalization statistics;
- train the model to predict normalized `q_pred_norm`;
- unnormalize before physics loss:

```math
L_{\mathrm{phys}} =
\left\| A(\mathrm{unnormalize}(\hat q_{\mathrm{norm}})) - y \right\|.
```

Image-space losses may be computed in normalized or physical units, but the
metric schema must state which. Paper tables should report image-space metrics
on physical `q` and optionally converted refractive index.

Dataset manifest fields:

- `q_units`
- `q_mean_train`
- `q_std_train`
- `q_min_train`
- `q_max_train`
- `forward_input_is_physical_q: true`
- `model_output_space: normalized_q` or `physical_q`

Rows that violate this contract are invalid.

## Rytov Mode Boundary

The default mode is Born. Rytov mode is a second-stage option:

```text
mode = "rytov_linearized"
```

For Rytov:

- preprocess measured complex fields into the Rytov/log-domain sinogram outside
  the training graph;
- do not put phase unwrapping inside the first differentiable training loop;
- use `lD=0` synthetic data unless a numerical refocus step is implemented and
  recorded;
- record the refocus method, phase-unwrapping method, and failure cases.

No Rytov row should appear in a paper table until the Born operator and dataset
gates have already passed.

## Dataset Design

Add the generator script:

```text
scripts/studies/born_rytov_dt/generate_brdt_dataset.py
```

Output layout:

```text
datasets/brdt128/
  brdt128_sparse_fullview_train.h5
  brdt128_sparse_fullview_val.h5
  brdt128_sparse_fullview_test.h5
  brdt128_sparse_fullview_manifest.json
```

Each HDF5 sample contains:

```text
q_true_physical       float32  (1, 128, 128)
q_true_norm           float32  (1, 128, 128)
ri_true               float32  (1, 128, 128), optional
sinogram_real         float32  (A, 128)
sinogram_imag         float32  (A, 128)
init_born_real        float32  (1, 128, 128)
init_born_imag        float32  (1, 128, 128), optional
angle_mask            float32  (A,)
support_mask          float32  (1, 128, 128), optional
sample_seed           int
```

Dataset-level manifest contains:

- grid size and detector size;
- angle list and angle coverage;
- wavelength in pixels;
- optional physical wavelength and pixel size if SI units are introduced;
- medium refractive index;
- object-function formula;
- refractive-index contrast distribution;
- object-family probabilities and per-sample generator family;
- noise model and noise sigma;
- signal-to-noise summary measured on generated sinograms;
- operator version;
- validation report path;
- generation git SHA and dirty-state note;
- exact generation command;
- split seed and split counts;
- train-only normalization stats;
- license/access note for generated data.

The dataset generator must be deterministic under the recorded split seed and
sample seeds. It should support a `--dry-run-manifest` or equivalent preflight
mode that writes only metadata and validates geometry before generating large
arrays.

Storage format:

- HDF5 is preferred for chunked row-wise access by Lightning dataloaders.
- If HDF5 dependencies are not accepted for the first preflight, `.npz` shards
  are acceptable only for the `N=64` smoke and must be replaced before any
  decision-support row.
- HDF5 chunks should be sample-major. A reasonable default is one sample per
  chunk for arrays with mixed access patterns.
- Compression may be enabled for generated datasets, but the manifest must
  record compression settings because they affect data-loading throughput.

Split policy:

- train/validation/test are disjoint at the object seed level;
- generator-family proportions are fixed across splits;
- normalization statistics are fit on train only;
- validation/test objects are never used for selecting normalization, color
  scales, or thresholds except through predeclared visualization sample IDs.

Noise policy:

- first contract: additive complex Gaussian noise on real and imaginary
  sinogram channels;
- `noise_sigma` is recorded in physical sinogram units before any measurement
  normalization;
- the manifest records measured SNR distribution over train/val/test;
- noiseless sinograms may be stored for diagnostics but are not used as observed
  data unless the row is explicitly a noiseless operator test.

## Object Distribution

Use weak-scattering phantoms that are not merely CDI line-pattern objects.
Recommended families:

- random overlapping ellipses;
- soft cell-like blobs;
- annular or nucleus-like inclusions;
- smooth random Gaussian fields thresholded into weak refractive-index regions;
- sparse fine inclusions for local-feature stress.

Keep refractive-index contrast within a weak-scattering regime:

```text
n_m = 1.333
delta_n approximately 0.002 to 0.03
```

The manifest must record the exact generator family and contrast distribution
for every sample.

Recommended generator controls:

- object support margin so objects do not touch boundaries in the first
  full-view contract;
- smoothness range for Gaussian blobs and random fields;
- ellipse count, size, eccentricity, and contrast distributions;
- inclusion count and radius distribution for sparse fine features;
- optional support mask derived from the known generator support.

The first contract should avoid objects whose contrast or high-frequency
content violates the weak-scattering approximation enough to make Born
reconstruction fail as a baseline. Harder phantoms should be introduced only as
stress rows after the operator and classical baseline are stable.

## Package Dependencies And Environment

The repo metadata currently provides the main numerical stack through
`pyproject.toml` and `setup.py`:

- Python `>=3.10,<3.12`;
- NumPy;
- SciPy;
- PyTorch;
- scikit-image;
- scikit-learn;
- matplotlib;
- pandas;
- neuraloperator;
- tqdm;
- optional `lightning`, `mlflow`, and `tensordict` under the Torch extra in
  `setup.py`.

Born/Rytov DT adds likely dependencies that are not currently guaranteed by the
base package metadata:

- `h5py` for HDF5 dataset storage and chunked loading;
- `odtbrain` for classical reconstruction and external validation;
- `nrefocus` only if Rytov detector-distance refocusing is implemented;
- optional `torchmetrics` only if reused elsewhere; otherwise implement metrics
  locally to avoid expanding dependency surface.

Dependency policy:

- do not make ODTbrain a hard runtime dependency of training unless the training
  loop calls it;
- keep ODTbrain and nrefocus in a preflight/validation extra if packaging is
  updated;
- fail closed with a clear message when optional validation dependencies are
  missing;
- record exact package versions in every operator validation and training
  artifact.

Recommended environment for local runs:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ptycho311
python - <<'PY'
import sys, torch, numpy, scipy
print(sys.version)
print(torch.__version__)
print(numpy.__version__)
print(scipy.__version__)
PY
```

Long-running generation or training should use `tmux` in the `ptycho311`
environment, following the repo workflow convention. GPU runs should record:

- GPU model;
- CUDA availability;
- PyTorch CUDA version;
- dtype;
- deterministic settings;
- batch size;
- peak VRAM if measured.

Minimum implementation tests should not require CUDA. CUDA tests should be
separate smoke checks guarded by availability.

## Dataset Sizes

Preflight:

- `N = 64`
- angles: `32`
- train/val/test: `256 / 64 / 64`
- epochs: `2` to `5`
- purpose: operator, data, adapter, and artifact smoke only

Capped decision-support:

- `N = 128`
- angles: `64`
- train/val/test: `2048 / 256 / 256`
- epochs: `20` to `40`
- purpose: decide whether BRDT deserves paper-grade promotion

Paper-grade candidate:

- `N = 128`
- angles: `64`
- train/val/test: `8192 / 1024 / 1024`, or the largest predeclared feasible
  split
- epochs and seeds fixed before launch
- at least three seeds only if robustness claims are planned

No row can be described as paper-grade without a locked dataset/split manifest
and complete provenance under the same standards as CDI/CNS.

## Input Representation

Primary mode: `born_init_image`

```math
x = \mathrm{concat}(
  \mathrm{init\_born\_real},
  \mathrm{init\_born\_imag\_or\_zero},
  \mathrm{optional\_confidence\_mask}
).
```

Shape:

```text
x: (B, C_in, 128, 128)
target: (B, 1, 128, 128)
```

The physics loss still uses the raw measured sinogram:

```math
L_{\mathrm{phys}} = \| A(q_{\mathrm{pred,physical}}) - y_{\mathrm{obs}} \|.
```

Direct sinogram input is optional later work. It changes the meaning of the two
spatial axes and can unfairly penalize image decoders. It should not be mixed
with init-image rows in one table.

## Model Integration

Do not register Born/Rytov DT as a normal PtychoPINN CDI generator. The
PyTorch generator registry is tied to the CDI/PtychoPINN output and stitching
contracts. Born/Rytov DT should reuse architecture bodies through task-specific
adapters.

Add task-local modules:

```text
scripts/studies/born_rytov_dt/models.py
scripts/studies/born_rytov_dt/lightning_module.py
scripts/studies/born_rytov_dt/train.py
scripts/studies/born_rytov_dt/evaluate.py
```

Adapter contract:

```python
class BRDTModelAdapter(torch.nn.Module):
    def __init__(
        self,
        architecture: Literal[
            "unet",
            "fno_vanilla",
            "fno",
            "ffno",
            "spectral_resnet_bottleneck",
            "hybrid_resnet",
            "sru_net",
        ],
        in_channels: int,
        out_channels: int = 1,
        grid_size: int = 128,
        **arch_kwargs,
    ): ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, N, N)
        returns q_pred_norm or q_pred_physical according to the run config
        """
```

The visible paper label should be SRU-Net if that is the manuscript name for
the selected Hybrid-family architecture. Internal architecture IDs must still
be recorded in manifests.

## Training Objective

The first neural rows use supervised reconstruction plus Born consistency:

```math
L =
\lambda_{\mathrm{img}} \|\hat q - q\|_1
+ \lambda_{\mathrm{phys}} \|A(q_{\mathrm{phys}}) - y\|_1
+ \lambda_{\mathrm{rel}}
   \frac{\|A(q_{\mathrm{phys}}) - y\|_2}{\|y\|_2 + \epsilon}
+ \lambda_{\mathrm{tv}} TV(\hat q)
+ \lambda_{\mathrm{pos}} \|\min(q_{\mathrm{phys}}, 0)\|_2^2.
```

Default preflight weights:

```text
lambda_img  = 1.0
lambda_phys = 0.1
lambda_rel  = 0.1
lambda_tv   = 1e-5
lambda_pos  = 1e-4
```

Training labels should distinguish model from training procedure:

```text
Model: U-Net
Training: supervised + Born consistency
```

Do not call these default rows `PINN` or `physics-only`. Reserve those labels
for rows with matching training semantics.

Optional later regimes:

- supervised only: `lambda_phys = 0`
- physics regularized: `lambda_img > 0`, `lambda_phys > 0`
- physics only: `lambda_img = 0`, with priors and support constraints

Physics-only is not a first milestone row because failure could reflect
ill-posedness or priors rather than architecture quality.

## Minimum Rows

Preflight rows:

| Model | Training | Purpose |
| --- | --- | --- |
| Classical Born backprop | none | sanity/reference baseline |
| U-Net | supervised + Born consistency | local CNN baseline |
| FNO vanilla | supervised + Born consistency | spectral baseline |
| SRU-Net or Hybrid-family row | supervised + Born consistency | target architecture |

Candidate promotion rows:

- classical Rytov backprop, only if Rytov mode is active;
- cascaded FNO;
- spectral ResNet bottleneck;
- FFNO, only if the adapter is stable and protocol-compatible;
- physics-only variants, only after supervised-plus-physics rows are stable.

Every row in a single table must share the same input representation, split,
normalization, loss family, angle mask, and metric schema.

## Metrics

Image-space metrics on physical `q`:

- MAE;
- RMSE;
- relative L2;
- PSNR;
- SSIM;
- MS-SSIM;
- FRC50.

Optional refractive-index metrics may be reported after a verified conversion
from `q` to `n`.

Measurement-space metrics:

- complex sinogram MAE;
- complex sinogram RMSE;
- relative measurement L2;
- amplitude residual;
- phase residual if phase is meaningful for the selected representation;
- per-angle residual curve;
- per-frequency residual curve;
- held-out-angle residual if full simulated sinograms are available.

Runtime/provenance metrics:

- parameter count;
- training wall time;
- inference wall time;
- GPU model;
- peak VRAM if available;
- operator forward time per batch;
- operator backward time per batch.

## Visual Bundle

Create fixed-sample comparison figures mirroring the CDI/CNS evidence style:

```text
visuals/brdt_compare_q.png
visuals/brdt_error_q.png
visuals/brdt_sinogram_residual.png
visuals/brdt_frc_curves.png
visuals/brdt_per_angle_residual.png
visuals/brdt_compare_ri.png, optional
```

For each fixed test sample ID, include:

- ground-truth `q`;
- classical Born reconstruction;
- U-Net reconstruction;
- FNO reconstruction;
- SRU-Net/Hybrid-family reconstruction;
- absolute error panels;
- sinogram residual panels.

Save source arrays:

```text
figures/source_arrays/sample_<id>_q_true.npy
figures/source_arrays/sample_<id>_<model>_q_pred.npy
figures/source_arrays/sample_<id>_<model>_sinogram_pred.npy
figures/source_arrays/sample_<id>_sinogram_obs.npy
```

No visual is accepted unless it can be regenerated from saved arrays and a
manifest.

## External FDTD Context

The public ODTbrain/Figshare FDTD dataset may be useful as an external realism
check, but it is not the first training benchmark. If used, it should be
labeled:

```text
external_fDTD_model_mismatch_context
```

It must not be mixed into the synthetic Born/Rytov table because the forward
model differs.

## Artifacts

Minimum preflight artifacts:

```text
docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_operator_validation_report.md
docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_preflight_summary.md
.artifacts/NEURIPS-HYBRID-RESNET-2026/brdt_preflight/
  operator_validation.json
  dataset_manifest.json
  metrics.csv
  metrics.json
  visual_manifest.json
```

Any promoted paper table must additionally emit:

- CSV/JSON/TeX table sources;
- metric schema JSON;
- figure manifests;
- source arrays;
- command/config/git/environment provenance;
- row status: `paper_grade`, `decision_support`, `blocked`, or
  `not_protocol_compatible`.

## Go / No-Go Gates

The lane can move from design to implementation planning only if:

- CDI and CNS remain the active required pillars;
- the roadmap or backlog explicitly allows a BRDT preflight;
- the first implementation scope is limited to operator/data/adapters and four
  rows;
- operator validation has independent oracle checks;
- normalization cannot bypass physical units in the forward loss;
- row labels separate model from training procedure.

The lane can move from preflight to candidate paper evidence only if:

- Born operator validation passes;
- the dataset manifest is locked;
- the four-row decision-support table exists under one contract;
- fixed-sample visuals and source arrays exist;
- the result adds useful inverse-wave evidence beyond CDI and CNS;
- a checked-in roadmap/evidence amendment authorizes the promotion.

## Non-Goals

- Replacing CDI or CNS.
- Launching a broad benchmark suite before the operator is validated.
- Using limited-angle data as the first test.
- Comparing init-image rows against direct-sinogram rows in the same table.
- Calling supervised-plus-physics rows `PINN-only`.
- Claiming full-wave validity from a Born/Rytov synthetic benchmark.
- Using external FDTD mismatch rows as same-protocol benchmark evidence.
