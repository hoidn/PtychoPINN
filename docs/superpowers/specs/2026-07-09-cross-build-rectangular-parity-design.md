# Cross-Build Rectangular Parity Design

**Date:** 2026-07-09
**Status:** Approved

## Problem

`tests/torch/test_cross_branch_rectangular_parity.py` compares frozen floating-
point tensors with `torch.equal` and zero-tolerance scalar checks. The fixtures
were generated on one Torch/CPU build, while CI runs an unpinned current CPU
build on heterogeneous GitHub runners. The fixture generator already documents
that replay is only bit-exact within one Torch build.

Identical code and Torch 2.13 environments have alternated between pass and
failure on GitHub runners, while the CI software stack passes locally. The
failure is backend-level FFT/reduction rounding, not a physics divergence.

The triggering incident was GitHub Actions run `29052341823` at main commit
`f3be50eb504738397b1573c61702762b73d949e8`. Its `pytest-cpu` job used Python
3.11, Torch `2.13.0+cpu`, and NumPy `2.4.6` and failed these five nodes:

- `test_c1_bigF_rectangular_bit_exact_under_real_padding`;
- `test_bigT_residual_is_padding_only_and_vanishes_under_matched_padding`
  for `c1_bigT_probe`, `c1_bigT_uniform`, `c4_bigT_probe`, and
  `c4_bigT_uniform`.

The same parity module passes locally with Torch `2.13.0+cpu` and NumPy `2.4.6`.
The repository's earlier incident record also documents alternating pass/fail
outcomes for identical trees and Torch 2.13 software on different hosted
runners.

## Decision

Cross-build fixture replay will use `rtol=1e-5` and `atol=1e-6`, matching the
authoritative rectangular acceptance suite. A shared test helper will apply the
same contract to forward tensors and loss scalars.

The regression suite will prove both sides of the boundary:

- representative backend-sized floating-point drift is accepted; and
- a material perturbation beyond the tolerance is rejected.

`test_cross_build_tolerance_accepts_roundoff_and_rejects_material_drift` will
load the `c1_bigF` fixture and calculate its real forward tensor and loss
scalar. It will pass both through the shared assertion helper, then apply
controlled perturbations. A perturbation smaller than
`atol + rtol * abs(expected)` must pass. A perturbation at least 100 times that
bound must raise `AssertionError`. Both the forward tensor and loss scalar are
covered so relaxing the frozen-output contract cannot silently remove either
physics or loss sensitivity.

Mode forcing, stored weighting, padding isolation, tensor shapes, and finite-
residual checks remain unchanged. The matched-padding claim is corrected from
"exactly zero" to "within the registered cross-build tolerance" because it too
compares against a fixture produced by another Torch/CPU build. The 100-times-
bound rejection test preserves sensitivity to material padding or physics
drift. Frozen fixture inputs and outputs are not regenerated.

## Rejected Alternatives

- Pinning Torch alone does not control CPU microarchitecture or SIMD kernel
  dispatch and therefore cannot guarantee bitwise replay.
- Deselecting the tests would remove useful physics and configuration coverage.
- Re-freezing on one CI runner would merely anchor the fixtures to a different
  non-portable backend.

## Verification

Run the targeted parity module under the repository environment and the CI-
matched Torch 2.13 CPU environment, then run `bash ci/run_ci_tests.sh`. The
GitHub `pytest-cpu` job must complete successfully after publication.
