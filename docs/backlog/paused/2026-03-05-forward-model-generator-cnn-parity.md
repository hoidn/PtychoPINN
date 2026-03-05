# Backlog: Normalize Generator Forward-Model Semantics to CNN Path

**Created:** 2026-03-05
**Status:** Open
**Priority:** High
**Related:** `ptycho_torch/model.py`, `ptycho_torch/workflows/components.py`, `scripts/studies/grid_lines_torch_runner.py`, `ptycho_torch/generators/hybrid_resnet.py`
**Impacts:** cross-architecture physics parity, interpretation of C-channel predictions, fairness of grid-lines comparisons

## Summary
Generator architectures (for example `hybrid_resnet`) can currently run with `object_big=False` while still producing multi-channel object predictions (`C>1`). In this mode, the forward model skips position-aware union/re-extract and applies probe illumination channel-wise without coordinate-conditioned forward mapping.

The CNN path behaves differently under `object_big=False` (single-channel object semantics), so model families are not using equivalent physics in the same study setting.

Define and enforce a single forward-model contract so generator paths and CNN paths are physically comparable, with TensorFlow CNN behavior as the normalization target.

Current wrong behavior to fix: in generator paths with `object_big=False` and `C>1`, each channel gets the same unshifted probe application (broadcast-style), not a position-specific forward map.

## Current Incorrect Behavior (Must Fix)
1. `cnn` + `object_big=False`:
   - object prediction semantics are effectively single-channel.
2. generator (`hybrid_resnet`/FNO-family) + `object_big=False` + `C>1`:
   - object prediction can remain multi-channel;
   - probe is applied identically across channels;
   - no per-channel coordinate shift/re-extract is applied in forward physics.
3. This creates an architecture-dependent meaning of channel axis `C`, which is not acceptable for fair compare/study runs.

## Target Reference Behavior (TF Contract)
Normalize PyTorch generator behavior to the established TensorFlow CNN semantics:
1. With `object.big=False`, object prediction is single-channel.
2. Forward physics still uses scan positions by extracting per-position patches from the padded object before illumination/diffraction.
3. Channel axis in predicted diffraction corresponds to position-indexed extracted patches, not independent generator channels with identical unshifted probe application.

## Why
- Current behavior allows architecture-dependent meaning of channel axis `C` when `object_big=False`.
- Cross-model comparisons can unintentionally mix different physics assumptions.
- Debugging quality regressions is harder when architecture choice changes forward-map semantics.

## Historical Context
1. Why `object_big=False` was pinned in the grid-lines Torch runner:
   - It was introduced as a parity fix for the grid-lines workflow, not as a complete semantic alignment between CNN and generator paths.
   - The TF grid-lines workflow config already set `object_big=False`, and Torch was updated to match that contract for integration stability.
2. Why it still changed behavior at `gridsize=1` (unexpectedly):
   - In Torch, `object_big=True` routes through reassembly/extraction codepaths that were not a no-op even for `C=1`.
   - Historical bisect/debug notes showed that bypassing object-big reassembly at `C=1` recovered expected integration metrics.
3. Why this backlog remains necessary after that fix:
   - The `object_big=False` pin restored workflow parity at a high level, but did not resolve architecture-specific forward semantics in generator models.
   - We still need an explicit contract so generator and CNN paths mean the same thing under identical study settings.

## Proposed Contract
1. `object_big=False` must imply the same forward-physics semantics across architectures:
   - no coordinate-conditioned union/re-extract;
   - explicit, documented channel behavior (not implicit/broadcast side effects).
   - behavior must match TF reference semantics for this mode.
2. For generator architectures, enforce CNN-equivalent behavior in this mode by one of:
   - hard-enforce single-channel object forward path (`C=1` physics path), or
   - canonical, explicit channel reduction before physics (with runtime manifest fields).
   - disallow the current implicit behavior where `C>1` channels each receive identical unshifted probe application.
3. If multi-position physics is desired (`C>1` with position meaning), require `object_big=True` and the union/re-extract path for all architectures.
4. Runtime contract artifacts must record:
   - effective forward mode,
   - effective channel policy,
   - whether coordinates were used inside forward physics.

## Acceptance Criteria
1. For identical config in grid-lines studies, CNN and generator arms report the same effective forward contract fields.
2. `object_big=False` no longer permits silent architecture-specific channel semantics.
3. Generator runs with `object_big=False` and `C>1` must either:
   - fail fast with an actionable contract error, or
   - execute an explicit canonical channel-reduction policy before physics.
4. Integration tests cover both cases:
   - `object_big=False` parity across `cnn` and `hybrid_resnet`;
   - `object_big=True` coordinate-conditioned union/re-extract parity.
5. Study manifests/metrics metadata expose forward-mode fields so comparisons are auditable.

## Risks / Open Questions
1. Backward compatibility for existing generator runs that implicitly relied on `C>1` with `object_big=False`.
2. Choice of canonical reduction policy (if reduction is selected) and its metric impact.
3. Whether to gate incompatible configs at CLI parse time or fail at runtime with actionable errors.

## Suggested Next Step
Prototype strict contract validation first (warn/fail on ambiguous configs), then implement the selected canonical channel policy with regression tests and a small parity study comparing `cnn` vs `hybrid_resnet` under identical settings.
