# Torch Loader Regression Hardening Design

**Date:** 2026-07-09
**Branch:** `fno-stable`
**Status:** Implemented and verified

## Purpose

Close the five defects and coverage gaps found while auditing the restored
`PtychoDataset` loader behavior:

1. integrated legacy `(H, W, N)` loading is broken by write-time coordinate
   alignment using the raw array's first dimension;
2. `PtychoDataset.from_np` fails for grouped `normalize='Group'` inputs;
3. `calculate_length` can skip a file while `memory_map_data` still iterates it,
   desynchronizing per-file bookkeeping;
4. the TF-compatible negative coordinate convention is implemented but not
   pinned through the mmap loader; and
5. scalar `dataset[i]` access fails even though `PtychoDataset` implements the
   standard PyTorch `Dataset` interface.

This design also preserves the recent multi-mode probe restoration and the
memory-map group-count fix.

Final review extended the hardening to cover coordinate-informed legacy layout
detection, exact NPZ diffraction-key selection, and finite `data_scaling='Max'`
factors. These are loader integrity cases under the same governing contracts,
not changes to the physics model.

## Governing Contracts

- Standalone NPZ diffraction uses canonical `(N, H, W)` layout. The loader may
  accept legacy `(H, W, N)` arrays, but every downstream count and index must
  use the canonicalized layout.
- Every diffraction pattern needs exactly one aligned scan position after the
  documented trailing-coordinate reconciliation. Fewer positions are invalid.
- A directory build is atomic with respect to its input file set. Invalid files
  are rejected with their path and reason; they are not silently omitted.
- Grouped local coordinates use the TF convention
  `coords_relative = -(coords_global - coords_center)`.
- Probe tensors retain incoherent mode shape. Batched access returns
  `(B, C, P, N, N)`; scalar access returns the corresponding sample shape
  `(C, P, N, N)`.
- Probe scaling remains `batch[2]`. Batched scaling has shape `(B, 1, 1, 1)`;
  scalar scaling has sample shape `(1, 1, 1)`.

## Design

### 1. Canonical Diffraction Alignment

`memory_map_data` will load and canonicalize each diffraction stack before
coordinate alignment. The canonical stack's leading dimension is the only
source of `n_diff`, and the same stack is reused later for normalization and
memory-map writes. This removes the current disagreement between
`npz_headers`, `_get_diffraction_stack`, and the write pass without adding a
second full-stack load.

`_align_coords_to_diffraction` will validate both coordinate arrays. Unequal
`xcoords`/`ycoords` lengths raise `ValueError`. Matched counts pass through;
trailing coordinates are dropped with the existing warning; fewer positions
than diffraction patterns raise.

### 2. Fail-Fast File Validation

`calculate_length` will stop catching and skipping malformed files. Header,
coordinate, diffraction-shape, and cross-file image-shape failures will raise
immediately with the offending path. Consequently `file_list`,
`valid_indices_per_file`, `grouping_per_file`, and `cum_length` always describe
the same ordered file set.

This deliberately rejects partial directory builds. Supporting partial builds
would require an explicit accepted-file manifest and remapped experiment IDs;
that is unnecessary for this repair and would make silent data omission easier.

### 3. `from_np` Normalization Parity

`from_np` will construct grouped or ungrouped `images` before calculating
normalization factors.

- Batch normalization uses the source diffraction stack, matching the file
  loader's per-experiment factor.
- The existing C=1 unsupervised compatibility rule continues to select Batch
  normalization even if the configured mode is Group.
- Group normalization uses grouped 4D `images`, producing one RMS and physics
  factor per group.
- The existing string mode `'None'` produces unit factors.
- An explicit `scaling_constant` overrides only the RMS factor and is expanded
  to all groups; physics factors still follow the selected normalization mode.

The in-memory TensorDict receives factors with the same per-sample shapes as
the mmap TensorDict. The legacy `data_dict['scaling_constant']` remains a
single per-experiment value only where such a scalar is defined: Batch and
`'None'` modes always populate it, and Group mode populates it only when the
caller supplies an explicit scalar override. Group mode without an override
omits the key rather than flattening, averaging, or otherwise misrepresenting
per-group factors as one scalar. `get_experiment_dataset` must therefore copy
this legacy key only when it is present.

### 4. Coordinate-Sign Regression Pins

Regression tests will build real grouped datasets through both the mmap and
`from_np` paths and assert:

```python
coords_relative == -(coords_global - coords_center)
```

The assertion must use nonzero, asymmetric offsets so the opposite sign cannot
pass. Existing bridge and reassembly tests remain useful but are not substitutes
for loader-ingestion coverage.

### 5. Scalar and Batched Indexing

`__getitem__` will normalize scalar and batched experiment IDs through one
internal indexing path. It will build batched probe/scaling outputs first, then
remove the leading batch dimension for scalar input. This keeps mode and channel
expansion identical in both cases:

- scalar: probe `(C, P, N, N)`, scaling `(1, 1, 1)`;
- batched: probe `(B, C, P, N, N)`, scaling `(B, 1, 1, 1)`.

The returned TensorDict keeps TensorDict's normal scalar or batched indexing
semantics. Existing custom `TensorDictDataLoader` behavior is unchanged.

## Error Handling

Errors must identify the input file and violated contract. No error path may
continue with shorter per-file metadata arrays. Existing trailing-coordinate
warnings remain warnings because that reconciliation is an established loader
behavior; all other count and shape disagreements fail before memory-map
allocation or writing.

## Test Strategy

Implementation follows red-green TDD. Required regression cases are:

- a complete `PtychoDataset` build from legacy `(H, W, N)` diffraction;
- unequal x/y coordinate lengths;
- fewer positions than patterns through the public dataset constructor;
- a multi-file directory containing a spatial-shape mismatch, with a direct
  actionable `ValueError` rather than a later bookkeeping `IndexError`;
- `from_np` with `object_big=True`, `C=4`, and Group normalization, compared
  tensor-for-tensor with the file pipeline;
- Group normalization with an explicit RMS override, asserting that the scalar
  is expanded across all TensorDict RMS rows, physics factors remain the
  mode-derived per-group values, and `data_dict['scaling_constant']` contains
  exactly the supplied scalar;
- Group normalization without an override, asserting that the TensorDict has
  per-group RMS/physics factors and `data_dict` omits the legacy scalar key;
- mmap and `from_np` coordinate-sign equations with nonzero offsets;
- scalar indexing for single- and multi-mode probes, plus unchanged batched
  indexing; and
- the existing multi-mode, probe-normalization, group-count, batch-scale,
  coordinate, and reassembly suites.

Final verification runs the focused loader/probe suite followed serially by
`python -m pytest tests/torch -m "not slow" -q`. The known fixture metadata
checksum mismatch must be reported separately if it remains the only failure.

## Files

- Modify `ptycho_torch/dataloader.py` for canonical alignment, fail-fast file
  validation, `from_np` normalization, and scalar indexing.
- Modify `ptycho_torch/helper.py` narrowly so Max normalization returns the
  documented Batch/Group factor shapes and rejects invalid denominators.
- Extend `tests/torch/test_loader_length_guards.py` for alignment, file-set, and
  mmap coordinate-sign coverage.
- Extend `tests/torch/test_multimode_probe_and_from_np.py` for Group parity,
  `from_np` sign coverage, and scalar indexing.
- Do not change physics/model implementations or the legacy
  `dset_loader_pt_mmap.py` module.

## Non-Goals

- Refactoring `PtychoDataset` into new metadata or pipeline classes.
- Changing `group_coords` policy or random neighbor selection.
- Changing probe normalization mathematics or multi-mode mode summation.
- Repairing the separate grid-lines `ProbeIllumination` collation defect.
- Changing `object_big` configuration derivation.
- Repairing the unrelated fixture checksum mismatch.

## Documents Read

- `docs/index.md`
- `docs/findings.md`
- `docs/specs/spec-ptycho-core.md`
- `specs/data_contracts.md`
- `docs/TESTING_GUIDE.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/workflows/pytorch.md`
- `prompts/git_hygiene.md`
- `tmp/2026-07-09-session-report.md`
- `tmp/multimode_probe_from_np_restoration.md`
