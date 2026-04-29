---
priority: 120
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/born_rytov_dt_candidate_lane_design.md
check_commands:
  - |
    python - <<'PY'
    from pathlib import Path
    required = [
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/born_rytov_dt_candidate_lane_design.md"),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"missing BRDT candidate design: {missing}")
    print("brdt candidate design present")
    PY
prerequisites: []
related_roadmap_phases:
  - candidate-brdt-preflight
signals_for_selection:
  - Born/Rytov diffraction tomography is a cheap-forward inverse-scattering candidate that may be scientifically closer to CDI than WaveBench while remaining much cheaper than full-wave FWI.
  - The candidate design requires independent operator validation, physical-unit-safe normalization, and a four-row preflight before any paper-table promotion.
  - This preflight may execute concurrently with CDI/CNS work, but it must not replace or narrow those required evidence lanes.
---

# Backlog Item: Preflight Born/Rytov Diffraction Tomography Candidate Lane

## Objective

- Decide whether Born/Rytov 2D diffraction tomography is a practical additional
  inverse-scattering evidence lane for the SRU-Net manuscript.

## Scope

- Consume
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/born_rytov_dt_candidate_lane_design.md`.
- Validate the physical target, normalization contract, package dependencies,
  and environment requirements.
- Implement or prototype only the operator/data/adapters needed for an
  operator/data/model preflight.
- Require independent operator checks before any generated synthetic dataset is
  used as benchmark evidence.
- Limit the first runnable decision-support roster to:
  - classical Born backpropagation;
  - U-Net with supervised plus Born consistency training;
  - FNO vanilla with supervised plus Born consistency training;
  - SRU-Net or Hybrid-family row with supervised plus Born consistency
    training.
- Do not add BRDT rows to manuscript tables without a later checked-in roadmap
  or evidence-package amendment.

## Notes for Reviewer

- Keep BRDT additional to CDI `lines128` and PDEBench CNS. It is not a
  replacement pillar.
- Do not call supervised-plus-physics rows `PINN-only`.
- Do not run limited-angle, Rytov, FFNO, physics-only, external FDTD mismatch,
  or multi-seed rows in the first preflight.
- This item is allowed by the active roadmap gate as concurrent candidate
  preflight work, but paper-table promotion still requires a later checked-in
  roadmap or evidence-package amendment.
