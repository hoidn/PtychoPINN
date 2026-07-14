# Task 27 Report - Reference Re-Qualification And Floor Re-Pinning Closeout

**Date:** 2026-07-12
**Branch:** `feature/ci-compatibility-ablation`
**Inclusive commits:** `46cc4d5bf` through `7c221d7e1`
**Harness/preparation commits:** `46cc4d5bf`, `c5dced526`, `a7e0dfa8b`,
`8aa239881`, `7f0e3dde5`
**Pin promotion/hardening commits:** `532761c59`, `a2bc9634d`, `7c221d7e1`
**Authoritative final evidence roots:**
`.artifacts/reference_qualification/task27_gain16_hybrid_final_commit-7c221d7`
and
`.artifacts/reference_qualification/task27_gain16_cnn_final_commit-7c221d7`

## Requirements

Task 27 re-qualified the Hybrid ResNet five-epoch and CNN twenty-epoch
references under the corrected probe-rank physics contract and the Task 26
gain-16 calibration. It required mandatory visual approval before pin
promotion, one atomic re-pin of both reference contracts and all dependent
test controls, preservation of superseded evidence, and fresh post-pin
controller runs before final completion.

The implementation did not change the gain default or rectangular/CI
contract. `ModelConfig.amplitude_physics_gain` still defaults to `1.0`, and
rectangular/CI scaling still rejects non-1.0 gain. The old and new pin values
and immutable artifact movement remain recorded in the plan ledger.

## Final Evidence

| Arm | Verdict | Amp / phase SSIM | Amp / phase MAE | Evidence SHA-256 | Visual SHA-256 |
|---|---|---|---|---|---|
| Hybrid ResNet | PASS | `0.8858652644013688` / `0.9618665959387648` | `0.08168590068817139` / `0.12818376669684495` | `2d297b391101909ebe9757359e28506a8130471a9ceade82a7a511a8e3527866` | `11280df5ee0e8f90d9abe4e6e4b4b2afd4bae9d658d8ba131fb5e178ffc785d5` |
| CNN | PASS | `0.8846891595066123` / `0.9150671199457723` | `0.08112537115812302` / `0.18809200960630929` | `80035e854686343ff5ec5ed9b0d712efc4ac23b0012a33c012c5bb295779ce4a` | `e5fcce8be6f83742079cac3fe5db7918abdda72f857ec4e2499f1c16468bfb70` |

The final evidence JSON files and visual PNGs match those recorded SHA-256
identities. For each arm, the final historical canvas, generic canvas, and
pre-stitch patch hashes exactly match the approved pre-pin run. The visual PNG
hashes also match the pre-pin reviews exactly.

Manual visual review passed for both arms. Hybrid showed recognizable line
morphology with no blank or malformed panels; CNN likewise passed manual
review. The final visual manifests record six populated truth,
reconstruction, and absolute-error panels with no resize.

## Reviews And Verification

Spec compliance was APPROVED, and code quality was APPROVED after fixes. The
recorded focused suite result was `495 passed`. The focused GPU integration
result was `1 passed in 212.89s`.

Closeout validation independently:

- recomputed all four final evidence/visual SHA-256 values;
- parsed both qualification reports and confirmed `verdict=pass`, exact
  metrics, evidence paths, locked floors, and gain 16;
- compared final and approved pre-pin visual/canvas identities;
- swept live Markdown status and routing surfaces for stale Task 27
  `pending`/`in_progress` language while preserving historical execution
  records;
- checked local Markdown links, the focused diff, and whitespace validity.

## Caveats And Follow-On

The existing aligned-ablation `rectangular_scaled` plus MAE test is stale and
is correctly rejected by the CI/NLL contract. It is an out-of-scope follow-on;
Task 27 does not weaken that contract to admit the obsolete combination.

Task 28 is next but has not started. It remains pending implementation of the
`absolute_ssim_delta_v1` gate and materialization of a fresh exact mmap twin
before dictionary-to-mmap convergence can be evaluated. No Task 28 result or
completion claim is made here.

## Final Status

Task 27 is `complete_final`. Both final post-pin reference qualifications and
manual visual reviews passed, the promoted pins are covered by the focused and
GPU integration verification, and the implementation reviews are approved.
