---
priority: 31
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-wavebench-provisioning-decision/execution_plan.md
check_commands:
  - |
    python - <<'PY'
    import json
    from pathlib import Path

    summary = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/wavebench_inverse_source_preflight_summary.md")
    metadata = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-inverse-source-preflight/preflight_metadata.json")
    missing = [str(p) for p in [summary, metadata] if not p.exists()]
    if missing:
        raise SystemExit(f"missing WaveBench preflight authority: {missing}")
    data = json.loads(metadata.read_text())
    if data.get("final_status") != "needs_dataset_or_checkpoint_decision":
        raise SystemExit(f"unexpected preflight status: {data.get('final_status')}")
    if data.get("selected_variant", {}).get("dataset_name") != "thick_lines":
        raise SystemExit("preflight metadata does not identify the selected thick-lines variant")
    print("wavebench provisioning inputs present")
    PY
prerequisites:
  - 2026-04-29-wavebench-inverse-source-preflight
related_roadmap_phases:
  - candidate-wavebench-inverse-source-extension
signals_for_selection:
  - Select this before WaveBench training or paper-bundle items because the completed preflight ended in `needs_dataset_or_checkpoint_decision`.
  - This item clears the concrete blockers: dataset staging path, singular/plural path normalization, WaveBench-capable environment, and native checkpoint recovery versus retraining decision.
  - It remains candidate work and does not authorize manuscript claims or replace CDI/CNS evidence pillars.
---

# Backlog Item: Resolve WaveBench Provisioning And Baseline Decision

## Objective

- Convert the completed WaveBench inverse-source preflight from
  `needs_dataset_or_checkpoint_decision` into explicit follow-up readiness
  decisions before any WaveBench training item runs.

## Scope

- Consume the durable preflight summary and metadata.
- Stage or verify the selected distributed `.beton` dataset member under the
  stable target:

  `<wavebench repo>/wavebench_dataset/time_varying/is/`

- Normalize the upstream singular/plural path mismatch in the follow-up run
  notes so later commands do not rediscover it.
- Provision or document a WaveBench-capable environment for the required loader
  and physics surfaces: `ffcv`, `jax`, `jwave`, and `ml_collections`.
- Resolve native baseline provenance by either:
  - recovering exact FNO and U-Net checkpoint identifiers for
    `thick_lines_gaussian_lens`; or
  - recording a from-scratch native retraining path from the upstream
    `train_fno_is.py` and `train_unet_is.py` surfaces.
- Emit a concise provisioning decision summary with:
  - selected variant and dataset path
  - dataset size/checksum or size/mtime manifest
  - environment/import status
  - native FNO/U-Net checkpoint or retraining decision
  - which WaveBench follow-up items are now unblocked, still blocked, or need
    narrowing

## Notes for Reviewer

- This is not a model-training item.
- Do not launch shared-encoder rows, native retraining, or physics-informed
  rows here except for tiny import/path/data-access smokes needed to prove the
  provisioning decision.
- If an external dataset, checkpoint, or environment dependency is unavailable,
  record the exact blocker and leave downstream items gated rather than
  inventing a weak substitute.
- Do not add WaveBench rows to manuscript tables from this item.
