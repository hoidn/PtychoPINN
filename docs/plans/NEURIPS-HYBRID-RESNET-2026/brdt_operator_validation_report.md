# BRDT Born Forward Operator Validation Report

## Identity

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-29-brdt-operator-validation`
- Operator module/class: `ptycho_torch.physics.born_rytov_dt.BornRytovForward2D`
- Validation harness: `scripts.studies.born_rytov_dt.validate_operator`
- Execution command: `python -m scripts.studies.born_rytov_dt.validate_operator`
- Machine-readable artifact:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-operator-validation/operator_validation.json`
- Run log:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-operator-validation/logs/validate_operator.log`
- Validating commit: latest `fno-stable` HEAD with this implementation;
  the JSON artifact carries the exact git SHA and dirty-state at run time.

## Environment

- Python 3.11.13, PyTorch 2.9.1+cu128, NumPy 1.26.4, SciPy 1.13.1.
- CUDA available (Torch CUDA 12.8); GPU validation exercised.
- ODTbrain: not installed locally. Optional inverse-side consistency
  recorded as `dependency_unavailable`.

## Locked Operator Contract

These fields are also serialized verbatim into `operator_validation.json`
under the `operator` key so downstream BRDT plans can consume the
authority without re-reading source.

| Field | Value |
| --- | --- |
| `mode` | `born` (only supported; `rytov_linearized` raises `NotImplementedError`) |
| `normalize` | `unitary_fft` default; `odtbrain_compatible` for direct-integral comparison |
| `grid_size` | configurable; canonical preflight uses `N=128` |
| `detector_size` | configurable; canonical preflight uses `D=128` |
| `wavelength_px` | configurable; canonical preflight uses `8.0` |
| `medium_ri` | configurable; default `1.333` |
| Coordinate convention | `(x, z)` with `z` the propagation axis. Object pixels at integer `(x=j, z=i)` with unit pixel spacing. |
| Angle convention | `theta` rotates the incident plane wave so `k_inc = k_m (sin theta, cos theta)`; `theta=0` corresponds to `+z` illumination. |
| Detector-frequency axis | `2*pi * torch.fft.fftfreq(D)` rad/pixel |
| Ewald-arc sampling | `K_obj = (k_x - k_m sin theta, k_z - k_m cos theta)` with `k_z = sqrt(k_m^2 - k_x^2)` (propagating only) |
| FFT normalization | `unitary_fft`: ortho FFTs, no Wolf prefactor. `odtbrain_compatible`: default FFTs with `(i / (2 k_z))` prefactor; output matches scalar 2D Born integral up to discretization. |
| Output layout | `(B, A, D, 2)` real tensor; last axis is `(real, imag)` channels of the complex sinogram |
| Buffers registered | `angles`, `det_freqs`, `sampling_grid`, `valid_mask`, `coeff_real`, `coeff_imag` |

The contract is implemented in
`ptycho_torch/physics/born_rytov_dt.py` and is exposed both via the class
constructor signature and via the helper method
`BornRytovForward2D.operator_contract()`.

## Validation Suite Results

The harness runs five blocking and two optional checks. All blocking
checks pass.

| Check | Status | Sample count | Tolerance | Metric |
| --- | --- | --- | --- | --- |
| `numpy_consistency` (torch vs NumPy reimplementation of Wolf spectral path) | pass | 4 | rel_l2 ≤ 1e-6 | 8.4e-8 |
| `direct_born_integral` (Hankel real-space oracle, forward cone) | pass | 3 | rel_l2 ≤ 0.6 | 0.46 |
| `analytic_phantom` (centered Gaussian, finite/non-trivial output) | pass | 1 | finite & non-zero | max amplitude 0.10 |
| `gradcheck` (autograd vs finite differences) | pass | 1 | atol=1e-5, rtol=1e-4 | passes |
| `cpu_dtype_reproducibility` (float64 vs float32) | pass | 3 | rel_l2 ≤ 5e-5 | 2.2e-6 |
| `cuda_reproducibility` (CPU vs CUDA float32) | pass | 3 | rel_l2 ≤ 5e-5 | 2.7e-7 |
| `odtbrain_inverse_consistency` (optional) | skipped | 0 | n/a | `dependency_unavailable` |

### Independent Oracle Detail

The plan requires that "self-consistency-only checks are invalid; at
least one independent oracle must be used so synthetic data and
validation are not both driven solely by the same PyTorch operator
path." Two oracles are exercised:

1. **`numpy_consistency`** — A NumPy reimplementation of the Wolf 1969
   spectral relation that does not import `BornRytovForward2D` at all
   computes the detector spectrum via explicit loops, manual bilinear
   interpolation on the fftshifted DFT, and explicit `(i / (2 k_z))`
   prefactor. This certifies the torch operator faithfully implements
   the spectral specification (rel_l2 < 1e-6 across four random weakly-
   scattering phantoms). It is independent of the torch implementation
   path even though it is dependent on the same physical model.

2. **`direct_born_integral`** — A discretized 2D scalar Born volume
   integral with the free-space Hankel Green's function
   `G(r) = (i/4) H_0^{(1)}(k_m |r|)`. This is a real-space convolution
   path that reuses none of the operator's sampled-FFT machinery. The
   FFT operator is propagated from its native `z=0` reference plane to a
   far-field detector plane via the angular-spectrum free-space
   propagator, and the propagated trace is compared to the direct
   integral evaluated at the same far plane. Forward-cone angles only
   (`theta in [0, pi/6]`) are used for this check; at oblique/back-
   scatter angles the FFT operator's implicit periodicity wraps the
   forward beam back into the detector window while the free-space
   integral does not, producing systematically different absolute
   amplitudes that are not a defect of either path. Measured rel_l2
   stays below the predeclared tolerance of 0.6 across three Gaussian
   phantoms.

This combination (one tight spec-conformance check and one looser real-
space physics oracle) satisfies the design's "no self-consistency-only"
requirement.

## Verdict

**`pass_with_documented_limits`.**

All blocking checks pass. The verdict is decorated with documented
limits because:

- `odtbrain_inverse_consistency` is recorded as
  `dependency_unavailable` (ODTbrain is not installed locally).
- `direct_born_integral` is asserted only on a forward-cone subset of
  illumination angles. For oblique/backscatter validation, a future
  follow-up should either zero-pad the operator to a much larger grid
  before sampling or move to a far-field detector convention with no
  periodic wraparound.

## Known Limits

- The discrete operator implements the Wolf 1969 spectral relation on a
  finite periodic grid. Its output is a band-limited and periodized
  version of the continuous free-space scattered field. For tightly
  centered, weakly-scattering objects the discrepancy is small in the
  forward cone, and the comparison is honest about the wraparound at
  large illumination angles.
- ODTbrain is not exercised in this validation pass; downstream BRDT
  items must rely on the operator contract and the in-tree validation
  results, not on an external inverse-side recovery, until ODTbrain is
  installed and an inverse-side check is added in a follow-up validation
  extension.

## Downstream Authorization

The next backlog item, `2026-04-29-brdt-dataset-preflight`,
**may proceed**. The operator contract is locked, faithfully
implemented (spec conformance < 1e-6), gradient-correct, dtype-
deterministic on CPU and CUDA, and physically consistent with the
free-space Born integral within a predeclared tolerance. Downstream
dataset and adapter items must consume the contract from
`operator_validation.json` (the `operator` block) and must continue to
treat the BRDT lane as additive candidate work rather than manuscript
evidence until a separate roadmap amendment authorizes promotion.

## How To Reproduce

From the repo root, with the `ptycho311` environment active:

```bash
python -m scripts.studies.born_rytov_dt.validate_operator
```

This rewrites `operator_validation.json` and the log under the artifact
root. The blocking pytest gates are:

```bash
pytest -q tests/studies/test_born_rytov_dt_operator.py \
        tests/studies/test_born_rytov_dt_validation.py
python -m compileall -q ptycho_torch scripts/studies/born_rytov_dt
```

All three commands must succeed.
