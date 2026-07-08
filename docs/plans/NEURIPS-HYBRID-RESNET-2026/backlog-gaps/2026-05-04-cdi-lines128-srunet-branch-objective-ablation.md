# Seed Plan: Lines128 SRU-Net Branch And Objective Ablation

## Objective

Run a same-contract `lines128` CDI ablation that supports the paper's SRU-Net
mechanism story by testing:

- local spatial encoder features versus global spectral encoder coupling;
- SRU-Net trained with the CDI physics-consistency objective versus SRU-Net
  trained with the supervised CDI objective.

## Scope

- Start from the completed `lines128` CDI paper benchmark root and reuse the
  existing `pinn_hybrid_resnet` row by lineage.
- Add only missing rows:
  - `pinn_hybrid_resnet_encoder_conv_only`;
  - `pinn_hybrid_resnet_encoder_spectral_only`;
  - `supervised_hybrid_resnet` or an equivalent explicit `SRU-Net +
    supervised` row id.
- Keep the dataset, split, probe preprocessing, seed, epoch count, scheduler,
  output mode, bottleneck, decoder, skip policy, metric schema, fixed visual
  samples, and shared visual scales fixed.
- Implement branch disablement through narrow, testable model/config plumbing.
  The conv-only row must preserve the local `3x3` spatial branch and disable
  only the spectral branch. The spectral-only row must preserve the spectral
  branch and disable only the local spatial branch.
- Use the completed encoder branch-gating / LayerScale ablation as context only;
  this item tests branch necessity, not learned branch weighting.

## Non-Goals

- Do not rerun completed FNO, FFNO, U-NO, CNN/U-Net-class, spectral bottleneck,
  or SRU-Net PINN rows unless an audit proves the existing row is unusable.
- Do not change the fixed `lines128` contract after seeing metrics.
- Do not broaden into skip-style, residual-scale, decoder-family, bottleneck,
  `256x256`, CNS, BRDT, or WaveBench work.
- Use "local spatial features" or "local spatial branch" for the local encoder
  contribution.

## Expected Artifacts

- Fresh row roots under:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-lines128-srunet-branch-objective-ablation/`
- A durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_srunet_branch_objective_ablation_summary.md`
- Updated discoverability surfaces:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/studies/index.md`
- Optional manuscript/table refresh only after row provenance and metrics are
  complete, with the base `lines128` bundle preserved by lineage.

## Verification Commands

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/backlog/done/2026-04-29-cdi-lines128-paper-benchmark-execution.md"),
    Path("docs/backlog/done/2026-04-29-cdi-lines128-supervised-equivalent-rows.md"),
    Path("docs/backlog/done/2026-04-30-cdi-lines128-uno-table-extension.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_hybrid_resnet_encoder_fusion_variants_summary.md"),
    Path("scripts/studies/grid_lines_compare_wrapper.py"),
    Path("scripts/studies/grid_lines_torch_runner.py"),
    Path("ptycho_torch/generators/hybrid_resnet.py"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing SRU-Net branch/objective ablation inputs: {missing}")
print("SRU-Net branch/objective ablation inputs present")
PY
pytest -q tests/torch/test_fno_generators.py -k "hybrid_resnet or hybrid_encoder"
pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or hybrid_encoder or supervised"
pytest -q tests/test_grid_lines_compare_wrapper.py
python -m compileall -q scripts/studies ptycho_torch
```
