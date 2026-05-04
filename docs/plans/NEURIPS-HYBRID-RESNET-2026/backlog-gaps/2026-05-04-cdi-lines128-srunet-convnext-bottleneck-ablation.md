# Seed Plan: Lines128 SRU-Net ConvNeXt Bottleneck Ablation

## Objective

Test whether replacing the current SRU-Net/Hybrid ResNet bottleneck ResNet
block stack with a ConvNeXt-style bottleneck improves or degrades the fixed
`lines128` CDI benchmark while keeping the rest of the SRU-Net shell unchanged.

## Scope

- Add a narrow `hybrid_resnet_convnext_bottleneck` generator variant or
  equivalent explicitly labelled architecture.
- Keep fixed relative to the completed `pinn_hybrid_resnet` / SRU-Net Lines128
  row:
  - dataset, split, probe preprocessing, seed policy, epoch budget, scheduler,
    output mode, loss, fixed visual samples, and shared visual scales;
  - encoder architecture, downsampling, decoder, skip policy, bottleneck width,
    and block count.
- Replace only the bottleneck body:
  - baseline: current `ResnetBottleneck` stack;
  - candidate: ConvNeXt-style constant-resolution block stack using depthwise
    spatial convolution, channel expansion/projection, GELU, normalization, and
    residual LayerScale.
- Start with the current SRU-Net LayerScale convention (`0.1`) so the first row
  isolates block family rather than initialization policy. Record canonical
  ConvNeXt-style tiny LayerScale initialization as a possible follow-up, not as
  part of the first row.
- Launch only the missing `pinn_hybrid_resnet_convnext_bottleneck` row. Reuse
  the completed `pinn_hybrid_resnet` row by lineage; do not rerun existing
  FNO, FFNO, CNN/U-Net-class, U-NO, spectral bottleneck, or SRU-Net rows.
- Emit row-local invocation/config/history/metrics/reconstruction artifacts,
  then publish an append-only summary comparing ConvNeXt bottleneck versus the
  existing SRU-Net bottleneck.

## Non-Goals

- Do not modify the completed Lines128 base bundle or claim that ConvNeXt is a
  new default model family from one row.
- Do not combine this with the SRU-Net encoder branch/objective ablation. That
  item tests encoder branch necessity and supervised objective controls; this
  item tests bottleneck block family.
- Do not change branch gates, skip connections, residual-scale policy, decoder,
  probe, loss, schedule, data contract, or visual scales in the same row.
- Do not run the canonical tiny LayerScale variant unless a later item or
  reviewed plan explicitly adds it.

## Expected Artifacts

- Fresh row root under:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-lines128-srunet-convnext-bottleneck-ablation/`
- A durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_srunet_convnext_bottleneck_ablation_summary.md`
- Updates to discoverability surfaces:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/studies/index.md`
- Optional manuscript/table refresh only after the row has complete metrics,
  provenance, and visuals; preserve the completed base Lines128 bundle by
  lineage.

## Verification Commands

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/backlog/done/2026-04-29-cdi-lines128-paper-benchmark-execution.md"),
    Path("docs/backlog/done/2026-04-30-cdi-lines128-hybrid-resnet-skip-residual-ablation.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md"),
    Path("scripts/studies/grid_lines_compare_wrapper.py"),
    Path("scripts/studies/grid_lines_torch_runner.py"),
    Path("ptycho_torch/generators/hybrid_resnet.py"),
    Path("ptycho_torch/generators/resnet_components.py"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing SRU-Net ConvNeXt bottleneck ablation inputs: {missing}")
print("SRU-Net ConvNeXt bottleneck ablation inputs present")
PY
pytest -q tests/torch/test_fno_generators.py -k "hybrid_resnet or convnext"
pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or convnext"
pytest -q tests/test_grid_lines_compare_wrapper.py -k "hybrid_resnet or convnext"
python -m compileall -q scripts/studies ptycho_torch
```
