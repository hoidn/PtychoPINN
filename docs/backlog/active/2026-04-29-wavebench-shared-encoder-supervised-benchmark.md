---
priority: 33
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-shared-encoder-supervised-benchmark/execution_plan.md
check_commands:
  - |
    python - <<'PY'
    from pathlib import Path
    required = [
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/wavebench_inverse_source_preflight_summary.md"),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"missing WaveBench preflight summary: {missing}")
    print("wavebench preflight summary present")
    PY
prerequisites:
  - 2026-04-29-wavebench-inverse-source-preflight
  - 2026-04-29-wavebench-native-baseline-reproduction
related_roadmap_phases:
  - wavebench-additional-inverse-wave-extension
signals_for_selection:
  - Select only after preflight confirms supervised shared-encoder readiness and native WaveBench baseline status is known.
  - This item is the first actual repo-local SRU-Net/hybrid-spectral WaveBench architecture comparison.
  - Steering on 2026-04-30 moved this WaveBench follow-up ahead of remaining optional U-NO table-extension work, subject to the preflight and native-baseline outcomes.
---

# Backlog Item: Run WaveBench Shared-Encoder Supervised Benchmark

## Objective

- Run the first supervised shared-encoder architecture comparison on the
  selected WaveBench inverse-source variant.

## Scope

- Implement or configure the shared boundary-measurement encoder from the
  WaveBench design:
  `y(t,b) -> h in R^{128 x 128 x C}`.
- Use the preflight-selected model IDs for:
  - U-Net/local convolutional row
  - SRU-Net / current `hybrid_resnet` row if protocol-compatible
  - hybrid-spectral / `spectral_resnet_bottleneck_net` or selected spectral row
  - FNO
  - FFNO
- Start with `C=32`; run `C=64` only if preflight or early smoke indicates the
  smaller latent is inadequate.
- Train supervised rows with the same dataset split, encoder architecture,
  loss, optimizer policy, and metric schema.
- Report encoder parameters, body parameters, total parameters, runtime, and
  memory separately where possible.
- Emit metrics, fixed-sample reconstruction figures, error maps, and a summary.

## Notes for Reviewer

- This compares `encoder + body`, not body-only. Do not describe it as a pure
  reconstruction-body ablation unless a frozen-encoder follow-up is run.
- Do not change WaveBench variant, split, normalization, or target resolution
  after seeing model metrics.
- Keep native WaveBench reference baselines separate from shared-encoder rows.
- Do not run physics-informed rows in this item.
- Do not add manuscript claims until an evidence-package amendment approves
  WaveBench as an additional lane.
