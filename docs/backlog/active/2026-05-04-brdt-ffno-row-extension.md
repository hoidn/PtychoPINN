---
priority: 36
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_ffno_row_extension_plan.md
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
        raise SystemExit(f"missing BRDT FFNO extension inputs: {missing}")
    print("brdt ffno extension inputs present")
    PY
  - pytest -q tests/studies/test_born_rytov_dt_adapters.py tests/studies/test_born_rytov_dt_preflight.py
  - python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch
prerequisites:
  - 2026-04-29-brdt-four-row-preflight
related_roadmap_phases:
  - candidate-brdt-preflight
signals_for_selection:
  - The completed BRDT four-row preflight showed a large Hybrid ResNet advantage over U-Net and FNO vanilla under the same capped decision-support contract.
  - A single append-only FFNO row can test whether factorized Fourier structure closes the gap without rerunning existing BRDT rows.
  - This remains candidate-lane decision support and must not preempt required CDI/CNS evidence or promote BRDT into paper claims.
---

# Backlog Item: Add BRDT FFNO Row Extension

## Objective

- Add one append-only FFNO row to the completed BRDT decision-support preflight
  and compare it against the already generated U-Net, FNO vanilla, and Hybrid
  ResNet/SRU-Net rows under the same BRDT contract.

## Scope

- Reuse the existing BRDT operator, decision-support dataset, `born_init_image`
  input mode, physical `q` target, split, normalization, fixed sample IDs,
  metric schema, and supervised plus Born-consistency training procedure.
- Add only the task-local FFNO row and any minimal adapter/config support needed
  to run or explicitly block that row.
- Write FFNO row provenance, parameter count, runtime, row status, image-space
  physical-`q` metrics, measurement-space residual metrics, and fixed-sample
  source arrays.
- Emit an append-only combined metrics view or extension manifest that clearly
  references the original four-row BRDT bundle.
- Update durable evidence/model indexes only after the FFNO row is completed or
  blocked with a structured reason.

## Notes for Reviewer

- Do not rerun or overwrite existing classical, U-Net, FNO vanilla, or Hybrid
  ResNet rows.
- Do not mix direct-sinogram input, Rytov mode, limited-angle data, or
  multi-seed robustness into this item.
- Do not call the result paper-grade evidence or manuscript-ready BRDT support.
