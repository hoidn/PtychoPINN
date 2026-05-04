# Seed Plan: CDI Natural-Patch Expanded Benchmark

## Objective

Run the first expanded-object CDI benchmark on the locked
`natural_patches128_fixedprobe_v1` dataset, without rerunning the existing
`lines128` table or changing the dataset after seeing metrics.

## Scope

- Consume the completed natural-patch fixed-probe dataset summary and manifests.
- Use the same model-row family as the current CDI headline table where
  supported by the adapter:
  - SRU-Net / `pinn_hybrid_resnet`;
  - paired CDI `cnn` U-Net-class supervised and PINN rows;
  - `pinn_fno_vanilla`;
  - `pinn_ffno`;
  - `pinn_neuralop_uno`.
- Use the locked dataset split, fixed probe, metric schema, visual policy, and
  training budget recorded by the dataset item or a checked-in preflight.
- Prefer one seed for the initial expanded-object table unless a later
  multi-seed expansion is explicitly authorized.
- Produce a standalone natural-patch benchmark table and visual bundle. The
  `lines128` table remains the original synthetic-line authority.

## Non-Goals

- Do not regenerate or mutate `natural_patches128_fixedprobe_v1`.
- Do not rerun completed `lines128` rows.
- Do not broaden into hyperparameter search, object-source search, or
  multi-seed robustness.
- Do not claim ImageNet-scale generalization if the locked source is a smaller
  natural-image corpus.

## Expected Artifacts

- Fresh row roots under:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-natural-patch-expanded-benchmark/`
- A durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_natural_patch_expanded_benchmark_summary.md`
- JSON, CSV, and TeX table payloads.
- Visual comparison bundle using fixed held-out object IDs.
- Updates to:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - `docs/studies/index.md`

## Verification Commands

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_natural_patch_fixedprobe_dataset_summary.md"),
    Path("scripts/studies/grid_lines_torch_runner.py"),
    Path("scripts/studies/grid_lines_compare_wrapper.py"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing natural-patch expanded benchmark inputs: {missing}")
print("natural-patch expanded benchmark inputs present")
PY
pytest -q tests/studies/test_cdi_natural_patch_dataset.py tests/studies/test_cdi_natural_patch_benchmark.py
python -m compileall -q scripts/studies ptycho_torch
```
