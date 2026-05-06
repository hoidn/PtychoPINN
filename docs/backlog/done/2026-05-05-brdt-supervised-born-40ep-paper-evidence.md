---
priority: 29
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-brdt-supervised-born-40ep-paper-evidence/execution_plan.md
check_commands:
  - |
    python - <<'PY'
    from pathlib import Path
    required = [
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/born_rytov_dt_candidate_lane_design.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_preflight_summary.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_ffno_row_extension_summary.md"),
        Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/preflight_manifest.json"),
        Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/metrics.json"),
        Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-brdt-ffno-row-extension/combined_metrics.json"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing BRDT 40-epoch paper-evidence inputs: {missing}")
    print("brdt 40-epoch paper-evidence inputs present")
    PY
  - pytest -q tests/studies/test_born_rytov_dt_adapters.py tests/studies/test_born_rytov_dt_preflight.py
  - python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch
prerequisites:
  - 2026-04-29-brdt-four-row-preflight
  - 2026-05-04-brdt-ffno-row-extension
related_roadmap_phases:
  - candidate-brdt-preflight
signals_for_selection:
  - The BRDT 20-epoch decision-support rows showed Hybrid ResNet and FFNO work under supervised plus Born consistency, but neither row has per-epoch loss history or convergence evidence.
  - A same-contract 40-epoch rerun with ReduceLROnPlateau and per-epoch loss logging can decide whether BRDT should be promoted from candidate context into additive paper evidence.
  - The paper package currently requires a checked-in evidence amendment before candidate lanes become manuscript evidence; this item owns that amendment gate and must not relabel old 20-epoch rows as paper evidence by fiat.
---

# Backlog Item: BRDT Supervised+Born 40-Epoch Paper Evidence Run

> **Post-hoc status (2026-05-06):** completed historical artifact only for the
> FFNO row. The `ffno` row was produced by the old BRDT FFNO-local-refiner proxy,
> not by the corrected no-refiner FFNO-paper-stack adapter. The bundle also
> failed provenance gates. It remains discoverable as historical context, but it
> must not be consumed as pure-FFNO manuscript evidence. Reactivated replacement:
> `docs/backlog/active/2026-05-06-brdt-corrected-ffno-40ep-rerun.md`.

## Objective

- Rerun the BRDT Hybrid ResNet and FFNO rows for 40 epochs under the supervised
  plus Born-consistency loss that worked at 20 epochs, add per-epoch loss
  logging and `ReduceLROnPlateau`, and promote the resulting BRDT comparison
  into additive paper evidence only if the paper-evidence amendment and
  provenance gates pass.

## Scope

- Reuse the locked BRDT operator, decision-support dataset,
  `born_init_image` input mode, physical `q` target, train-only normalization,
  split counts, fixed sample IDs, metric schema, and seed lineage from the
  completed four-row preflight and FFNO row extension.
- Rerun only:
  - `hybrid_resnet`;
  - `ffno`.
- Use exactly the supervised plus Born-consistency loss:
  - `image = 1.0`;
  - `physics = 0.1`;
  - `relative_physics = 0.1`;
  - `tv = 1e-5`;
  - `positivity = 1e-4`.
- Change the training budget and scheduler contract to:
  - `40` epochs;
  - Adam, initial `lr = 2e-4`;
  - `ReduceLROnPlateau(factor=0.5, patience=2, threshold=0.0, min_lr=1e-5)`;
  - batch size `16`;
  - seed `42`;
  - one logged training-loss record per epoch at minimum.
- Emit `history.json` / `history.csv` per row with epoch, train loss,
  component losses, current learning rate, scheduler step metric, and any
  validation or test-side loss added by the plan.
- Produce a convergence audit comparing the new 40-epoch rows against the
  frozen 20-epoch Hybrid and FFNO rows, including late-window loss movement and
  metric deltas.
- Emit a paper-facing visual comparison on BRDT test sample `255`, using shared
  scales and source arrays, comparing:
  - ground truth;
  - physics-based Born inverse;
  - SRU-Net / Hybrid ResNet;
  - FFNO.
- Update paper-evidence surfaces only after the new artifact root proves the
  paper evidence gate:
  - `paper_evidence_package_design.md` or a checked-in amendment;
  - `paper_evidence_manifest.json`;
  - `paper_evidence_index.md`;
  - `evidence_matrix.md`;
  - `model_variant_index.json`;
  - `ablation_index.json`;
  - `docs/index.md` if a new durable summary is created.

## Notes for Reviewer

- This item intentionally changes the BRDT claim target: the new 40-epoch
  artifact is meant to become paper evidence, not just local decision support.
- Do not promote the old 20-epoch BRDT rows by label change. Paper evidence
  requires the fresh 40-epoch artifact, per-epoch training history, scheduler
  provenance, convergence audit, and explicit evidence-package amendment.
- Do not rerun U-Net, FNO vanilla, classical Born inverse, Rytov mode,
  direct-sinogram input, limited-angle data, or multi-seed robustness in this
  item.
- The physics-based visual panel should reuse or deterministically regenerate
  the existing model-based Born inverse / classical physical-`q` prediction
  under the locked operator contract; it must not introduce a separate
  classical-training study.
- The required paper-facing visual sample is `255`; do not substitute a smoother
  default fixed sample unless the summary records an unrecoverable sample-level
  blocker and keeps the result out of paper evidence.
- If the 40-epoch run does not satisfy the paper-evidence gate, leave the
  result as bounded decision support and record the failed gate explicitly.
