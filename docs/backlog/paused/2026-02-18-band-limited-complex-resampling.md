# Backlog: Band-Limited Complex Resampling for Probe/Object Downsampling

**Created:** 2026-02-18
**Status:** Open
**Priority:** High
**Related:** `scripts/studies/prepare_nersc_hybrid_dataset.py`, `tests/studies/test_prepare_nersc_hybrid_dataset.py`, `scripts/studies/nersc_orchestration.py`
**Impacts:** external NERSC `256->128` prep semantics for `probeGuess`/`objectGuess`, cross-dataset hybrid reconstruction quality

## Summary
Add a physically grounded real-space downsampling option for complex fields (`probeGuess`, `objectGuess`) based on anti-alias filtering in Fourier space plus decimation, instead of complex block-mean binning.

Current diffraction downsampling is already detector-consistent (intensity-domain binning). The gap is only the complex-field path used by `crop-bin` policy, where block averaging can introduce phase cancellation and aliasing artifacts.

## Why
- Detector binning is naturally defined in intensity/count space, not complex object/probe space.
- Complex block-mean downsampling is a heuristic and can distort high-frequency phase/amplitude structure.
- A band-limited decimation path is the standard signal-processing way to reduce grid resolution while controlling aliasing.

## Proposed Contract
1. Keep diffraction downsampling unchanged:
   - Bin in intensity domain and map back to amplitude when needed.
2. Add a new helper for complex fields:
   - `resample_complex_bandlimited(array_2d, factor, window)`
   - Steps: center-crop to divisible shape -> FFT -> low-pass mask/window -> decimate spectrum -> IFFT.
3. Coordinate semantics must be explicit for this mode:
   - If downsampling keeps the same physical FOV with fewer pixels, divide pixel-coordinate offsets by `factor`.
   - If downsampling is interpreted as support crop (smaller FOV at same pixel size), do not scale coordinates.
4. Keep existing policies for compatibility; add the new path as opt-in until validated.

## Validation Requirements Before Defaulting
1. Unit tests for numerical behavior:
   - Constant/plane-wave fields preserve phase and amplitude within tolerance.
   - Above-Nyquist synthetic patterns are attenuated (not aliased).
   - Output shape/dtype/centering behavior is deterministic.
2. Parity checks versus current `bin-crop` baseline on known-good runs.
3. A/B comparison against current complex block-mean path on representative NERSC data:
   - patch-level metrics,
   - stitched-object metrics,
   - visual artifact checks (checkerboard/ringing).
4. Archive a compact evidence bundle under `tmp/debug/` before any production default change.

## Risks / Open Questions
1. Low-pass mask choice (hard crop vs tapered window) and ringing tradeoffs.
2. Exact coordinate-frame contract for external NPZ (`pixel` vs `meter`) across orchestration stages.
3. Interaction with reassembly backend behavior; avoid conflating stitching regressions with resampling changes.

## Suggested Direction
Start with non-invasive evidence work only (scripts/artifacts under `tmp/`) to compare current complex binning versus FFT low-pass decimation on the same checkpoint/data. Promote to `scripts/studies/prepare_nersc_hybrid_dataset.py` only after acceptance criteria pass.
