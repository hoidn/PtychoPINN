---
priority: 2
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-row-rerun/execution_plan.md
check_commands:
  - |
    python - <<'PY'
    from pathlib import Path
    required = [
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_ffno_row_extension_summary.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-row-rerun/execution_plan.md"),
        Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/preflight_manifest.json"),
        Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/metrics.json"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing corrected BRDT FFNO rerun inputs: {missing}")
    print("corrected BRDT FFNO rerun inputs present")
    PY
  - pytest -q tests/studies/test_born_rytov_dt_adapters.py tests/studies/test_born_rytov_dt_preflight.py
  - python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch
prerequisites:
  - 2026-04-29-brdt-four-row-preflight
related_roadmap_phases:
  - candidate-brdt-preflight
signals_for_selection:
  - The completed BRDT FFNO row was generated before the task-local FFNO adapter was corrected to remove post-bottleneck CNN refiners.
  - The current BRDT FFNO adapter now rejects `cnn_blocks` and uses only `SpatialLifter -> SharedFactorizedFfnoBottleneck -> 1x1`.
  - A corrected 20-epoch no-refiner FFNO row is required before any BRDT 20->40 FFNO convergence delta or pure-FFNO manuscript comparison is valid.
---

# Backlog Item: Corrected BRDT FFNO 20-Epoch Rerun

## Objective

- Rerun exactly the BRDT FFNO row under the original 20-epoch same-contract
  supervised+Born setup, using the corrected no-refiner BRDT FFNO adapter.

## Scope

- Reuse the completed BRDT four-row preflight as read-only lineage.
- Keep the same dataset, operator, `born_init_image` input, split counts,
  fixed-sample IDs, normalization, loss weights, batch size, seed, and 20-epoch
  budget as the historical FFNO row extension.
- Use current code where BRDT FFNO has no `refiners`, no `cnn_blocks`, and a
  minimal 1x1 output projection after the shared factorized FFNO stack.
- Publish a new append-only corrected extension root. Do not overwrite the
  historical `2026-05-04-brdt-ffno-row-extension` root.
- Update durable summaries and indexes so the historical row remains labeled as
  a legacy FFNO-local-refiner proxy and the corrected row is discoverable as the
  pure-BRDT-FFNO 20-epoch authority.

## Notes for Reviewer

- Do not rerun U-Net, FNO vanilla, Hybrid ResNet, or the model-based Born row.
- Do not use the historical `36,674` parameter count for the corrected row; the
  current no-refiner adapter should report `27,394` trainable parameters unless
  a code change intentionally alters the corrected adapter.
- If the corrected row fails, record a row-level blocker rather than relaxing the
  no-refiner FFNO contract.
