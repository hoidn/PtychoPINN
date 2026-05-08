---
priority: 2
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-40ep-rerun/execution_plan.md
check_commands:
  - |
    python - <<'PY'
    from pathlib import Path
    required = [
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-40ep-rerun/execution_plan.md"),
        Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/preflight_manifest.json"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing corrected BRDT 40-epoch inputs: {missing}")
    print("corrected BRDT 40-epoch inputs present")
    PY
  - pytest -q tests/studies/test_born_rytov_dt_adapters.py tests/studies/test_born_rytov_dt_preflight.py
  - python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch
prerequisites:
  - 2026-05-06-brdt-corrected-ffno-row-rerun
related_roadmap_phases:
  - candidate-brdt-preflight
signals_for_selection:
  - The previous 40-epoch BRDT bundle used the historical FFNO-local-refiner proxy and also failed provenance gates.
  - A corrected no-refiner FFNO 40-epoch row is required before BRDT FFNO can be cited as pure FFNO-paper-stack evidence.
  - A clean rerun should either produce corrected secondary BRDT evidence or preserve BRDT as candidate context with explicit gate failures.
---

# Backlog Item: Corrected BRDT FFNO 40-Epoch Rerun

## Objective

- Produce a corrected 40-epoch BRDT bundle using SRU-Net / Hybrid ResNet and
  the no-refiner BRDT FFNO adapter, with full runtime provenance, per-epoch
  history, convergence audit, and sample-255 visuals.

## Scope

- Depend on the corrected 20-epoch no-refiner FFNO row so 20->40 FFNO deltas do
  not compare against the legacy local-refiner proxy.
- Rerun `ffno` under corrected no-refiner semantics. Rerun `hybrid_resnet` as
  well if the item targets a clean same-run paper-evidence gate; otherwise keep
  any reused Hybrid/SRU-Net row explicitly labeled by lineage and do not claim a
  fully fresh two-row bundle.
- Keep the BRDT dataset/operator/input/split/normalization/loss/sample policy
  fixed to the completed BRDT contract.
- Regenerate metrics, model profiles, history, convergence audit, efficiency
  fields, sample-255 source arrays, and manuscript-facing figures/tables that
  include the corrected FFNO row.
- Replace or supersede the old BRDT manuscript/table authority only after the
  corrected artifact root passes its evidence gate.

## Notes for Reviewer

- Do not consume the old `2026-05-05-brdt-supervised-born-40ep-paper-evidence`
  FFNO row as pure FFNO evidence.
- Do not promote BRDT over CDI `lines128` or PDEBench CNS.
- If clean provenance cannot be captured, keep the corrected artifact out of
  paper-evidence status and record the exact failed gate.
