---
priority: 35
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog-gaps/2026-05-04-cdi-natural-patch-fixedprobe-dataset.md
check_commands:
  - |
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
  - pytest -q tests/studies/test_cdi_natural_patch_dataset.py
  - python -m compileall -q scripts/studies ptycho_torch
prerequisites:
  - 2026-04-29-cdi-lines128-paper-benchmark-execution
related_roadmap_phases:
  - phase-3-cdi-anchor-regeneration
signals_for_selection:
  - Reviewer-style feedback identified the tiny two-object CDI benchmark as a major acceptance risk.
  - This item creates the fixed dataset prerequisite for expanded-object CDI evidence without mixing dataset generation and model ranking.
  - The dataset is capped at no more than 10000 object images so later benchmark rows remain tractable.
---

# Backlog Item: CDI Natural-Patch Fixed-Probe Dataset

## Objective

- Generate and lock `natural_patches128_fixedprobe_v1`, a synthetic CDI dataset
  using natural-image-derived object patches and a fixed probe.

## Scope

- Use `N=128` object patches.
- Use the Run1084 fixed-probe lineage from the current `lines128` paper
  benchmark unless a reviewed implementation note records a better fixed probe.
- Use a local natural-image source such as ImageNet-style files when available.
  The exact corpus is secondary; source identity, split determinism, and
  provenance are mandatory.
- Limit the generated dataset to at most `10_000` object images total.
- Recommended first split: `8_000 / 1_000 / 1_000` train/validation/test
  objects.
- Emit source, probe, split, patch, simulation, and array manifests.
- Emit a small contact sheet for source patches and diffraction samples.

## Required Interpretation

- This item only creates the dataset prerequisite for expanded-object CDI
  benchmarking.
- It does not train models, update manuscript result tables, or replace the
  current `lines128` table.
- If no usable natural-image source is locally available, record a blocker with
  the required source contract rather than substituting a synthetic line/shapes
  dataset.

## Outputs

- Git-ignored dataset artifacts under:
  `.artifacts/data/NEURIPS-HYBRID-RESNET-2026/natural_patches128_fixedprobe_v1/`
- A durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_natural_patch_fixedprobe_dataset_summary.md`
- Source/probe/split/dataset manifests and a small visual contact sheet.
- Updates to the evidence matrix and study index.

## Notes For Reviewer

- Reject implementations that train models inside this item.
- Reject datasets with train/test source-image overlap.
- Reject datasets larger than `10_000` object images unless a later roadmap
  amendment explicitly raises the cap.
- Reject dataset roots committed into git.
