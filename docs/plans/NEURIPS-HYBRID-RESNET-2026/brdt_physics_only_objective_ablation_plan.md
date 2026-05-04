# BRDT Physics-Only Objective Ablation Plan

## Purpose

Run an append-only BRDT candidate-lane ablation that isolates the training
objective. The completed four-row preflight showed that U-Net and FNO vanilla
collapsed toward the sparse zero-physical-`q` solution under supervised image
L1 plus weak Born consistency. This ablation keeps the BRDT data/operator/input
contract fixed and changes only the neural-row loss weights to a pure relative
measurement residual.

## Fixed Inputs

- Backlog item:
  `docs/backlog/active/2026-05-04-brdt-physics-only-objective-ablation.md`
- Completed source bundle:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/`
- Dataset manifest:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/decision_support_dataset/dataset_manifest.json`
- Governing design:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/born_rytov_dt_candidate_lane_design.md`
- Adapter contract:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_task_adapters.md`

## Objective Contract

Use only:

```text
relative_physics_L2(A(q_pred_phys), observed_sinogram)
```

with loss weights:

```json
{
  "image": 0.0,
  "physics": 0.0,
  "relative_physics": 1.0,
  "tv": 0.0,
  "positivity": 0.0
}
```

`q_pred` may still be emitted in normalized-`q` model-output space, but the
Born operator must receive physical `q` through the existing
`BRDTTrainingModule.to_physical_q` path.

## Rows

- `unet`
- `fno_vanilla`
- `hybrid_resnet`

Use the same BRDT decision-support dataset, `born_init_image` input,
train/val/test split, normalization stats, metric schema, and fixed sample IDs
as the completed four-row preflight.

## Implementation Steps

1. Add the smallest run-config or CLI surface needed to pass explicit
   `LossWeights` into the BRDT preflight/training path.
2. Add tests proving the physics-only objective sets image, raw physics, TV,
   and positivity weights to zero while preserving unnormalize-before-operator
   routing.
3. Launch the three append-only rows under a new output root, for example:
   `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-brdt-physics-only-objective-ablation/`.
4. Emit metrics JSON/CSV, row summaries, invocations, source arrays, and
   fixed-sample visuals for the physics-only rows.
5. Emit a combined comparison summary that references the completed
   supervised-plus-Born rows without rerunning or overwriting them.
6. Update durable evidence/model indexes only after rows finish or block with
   structured reasons.

## Claim Boundary

This item is candidate-lane decision support. It diagnoses objective-induced
collapse versus architecture/optimization failure for BRDT. It does not promote
BRDT into CDI/CNS evidence, does not replace the existing four-row preflight,
and does not authorize manuscript claims without a later evidence-package
amendment.
