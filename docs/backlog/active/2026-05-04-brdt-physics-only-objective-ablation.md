---
priority: 34
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_physics_only_objective_ablation_plan.md
check_commands:
  - |
    python - <<'PY'
    from pathlib import Path
    required = [
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/born_rytov_dt_candidate_lane_design.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_task_adapters.md"),
        Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/preflight_manifest.json"),
        Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/metrics.json"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing BRDT physics-only ablation inputs: {missing}")
    print("brdt physics-only ablation inputs present")
    PY
  - pytest -q tests/studies/test_born_rytov_dt_adapters.py tests/studies/test_born_rytov_dt_preflight.py
  - python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch
prerequisites:
  - 2026-04-29-brdt-four-row-preflight
related_roadmap_phases:
  - candidate-brdt-preflight
signals_for_selection:
  - The completed BRDT four-row preflight showed FNO and U-Net collapsing toward the sparse zero-physical-q solution under supervised image L1 plus weak Born consistency.
  - A same-contract physics-only objective ablation can test whether the collapse is caused by the sparse supervised term or by architecture/optimization alone.
  - This is a high-priority candidate diagnostic that remains below required Phase 2 evidence work and must not promote BRDT into paper claims.
---

# Backlog Item: BRDT Physics-Only Objective Ablation

## Objective

- Run an append-only BRDT objective ablation for U-Net, FNO vanilla, and Hybrid
  ResNet using only the relative Born measurement residual:
  `relative_physics_L2(A(q_pred_phys), observed_sinogram)`.

## Scope

- Reuse the completed BRDT four-row preflight operator, decision-support
  dataset, `born_init_image` input mode, physical `q` target convention, split,
  train-only normalization, metric schema, and fixed-sample visualization set.
- Add objective/config support only as needed to set:
  - `image = 0`;
  - `physics = 0`;
  - `relative_physics = 1`;
  - `tv = 0`;
  - `positivity = 0`.
- Train append-only rows for:
  - `unet`;
  - `fno_vanilla`;
  - `hybrid_resnet`.
- Report image-space physical-`q` metrics, measurement-space residual metrics,
  parameter counts, runtime, final loss breakdown, output dynamic-range
  statistics, and fixed-sample source arrays.
- Compare the physics-only rows against the existing supervised-plus-Born rows
  without rerunning or overwriting the completed rows.
- Emit a concise summary that answers whether FNO/U-Net collapse is primarily
  objective-induced or persists under physics-only training.

## Notes for Reviewer

- Do not call these rows paper-grade evidence or manuscript-ready BRDT support.
- Do not mix direct-sinogram input, Rytov mode, limited-angle data, FFNO, or
  multi-seed robustness into this item.
- Do not replace the completed supervised-plus-Born bundle; publish the
  physics-only rows as an append-only ablation with explicit lineage to
  `2026-04-29-brdt-four-row-preflight`.
