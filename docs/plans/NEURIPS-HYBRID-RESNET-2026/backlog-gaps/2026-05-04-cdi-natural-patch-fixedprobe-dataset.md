# Seed Plan: CDI Natural-Patch Fixed-Probe Dataset

## Objective

Create and lock a synthetic CDI benchmark dataset with a fixed probe and a
larger variety of object patches derived from natural images. This is the
prerequisite for any expanded-object CDI benchmark rows.

## Scope

- Dataset id: `natural_patches128_fixedprobe_v1`.
- Resolution: `N=128`, matching the current `lines128` CDI benchmark
  resolution.
- Probe: use the same fixed Run1084 probe lineage as the `lines128` paper
  benchmark unless the implementation records a reviewed reason to choose a
  different fixed probe. The probe preprocessing pipeline must be recorded.
- Object source: use a local natural-image corpus with clear provenance.
  ImageNet-style images are acceptable, but the exact source is less important
  than recording source identity, license/access notes, checksums or a
  size/mtime manifest, and deterministic patch selection.
- Dataset size cap: no more than `10_000` object images total. A recommended
  first split is `8_000 / 1_000 / 1_000` train/validation/test objects, but a
  smaller split is acceptable if source availability or runtime requires it.
- Patch policy:
  - deterministic source-image ordering and patch sampling;
  - centered or randomly cropped `128x128` patches with predeclared seeds;
  - grayscale or luminance conversion recorded;
  - normalization from natural-image intensity to complex CDI object amplitude
    and/or phase recorded before data generation;
  - no train/validation/test source-image overlap.
- Simulate diffraction with the fixed probe and the same CDI data schema needed
  by the existing Torch runner/wrapper path, or write an adapter contract that
  the later benchmark item can consume without changing model semantics.

## Non-Goals

- Do not train model rows in this item.
- Do not tune object selection after seeing model metrics.
- Do not create a broad data-corpus search. Pick the first source that can be
  made reproducible and legally/documentably usable.
- Do not commit bulky dataset artifacts to git. Store generated arrays under a
  git-ignored artifact/data root and commit only manifests, summaries, and
  lightweight visual/contact-sheet evidence.
- Do not change the existing `lines128` table authority; this dataset defines a
  new expanded-object benchmark scope.

## Expected Artifacts

- Dataset root under a git-ignored path, for example:
  `.artifacts/data/NEURIPS-HYBRID-RESNET-2026/natural_patches128_fixedprobe_v1/`
- A durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_natural_patch_fixedprobe_dataset_summary.md`
- Machine-readable manifests:
  - source-image manifest with source path, split, patch coordinates, and
    checksum or size/mtime entries;
  - probe manifest;
  - dataset manifest with array paths, shapes, dtypes, split counts,
    simulation parameters, seeds, and schema version;
  - small visual/contact-sheet artifact for source patches and simulated
    diffraction samples.
- Updates to:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/studies/index.md`

## Verification Commands

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("datasets/Run1084_recon3_postPC_shrunk_3.npz"),
    Path("scripts/studies/grid_lines_torch_runner.py"),
    Path("scripts/studies/grid_lines_compare_wrapper.py"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing natural-patch dataset prerequisites: {missing}")
print("natural-patch fixed-probe dataset prerequisites present")
PY
pytest -q tests/studies/test_cdi_natural_patch_dataset.py
python -m compileall -q scripts/studies ptycho_torch
```
