# Lines128 Minimum Paper Table Execution Authority

- Date: `2026-04-29`
- Backlog item: `2026-04-29-cdi-lines128-minimum-paper-table`
- Supersedes only the launch authority for this backlog item. The prerequisite harness note at
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_harness_preflight.md`
  remains `GO_FOR_HARNESS_PREFLIGHT_ONLY` for the readiness item itself.
- Inputs:
  - checked-in note:
    `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_harness_preflight.md`
  - machine-readable prerequisite artifact:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/preflight/benchmark_decisions.json`

This authority freezes the minimum draftable CDI subset only. It does not authorize the complete
`lines128` benchmark. Launch authority is revoked if Task 4 deterministic gates are red or if the
derived execution manifest drifts from the JSON payload embedded below.

Authorized four-row minimum subset:

- `baseline`: CDI `cnn` supervised row
- `pinn`: CDI `cnn` PINN row
- `pinn_hybrid_resnet`
- `pinn_fno_vanilla`

Still required later for the complete `lines128` benchmark but intentionally out of scope here:

- `pinn_spectral_resnet_bottleneck_net`
- `pinn_ffno`

<!-- lines128_execution_authority_json:start -->
{
  "state": "go_for_minimum_subset_execution",
  "prerequisite_preflight_decision_artifact": ".artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/preflight/benchmark_decisions.json",
  "prerequisite_preflight_note": "docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_harness_preflight.md",
  "selected_fno_comparator": "fno_vanilla",
  "seed_policy": {
    "type": "fixed",
    "seed": 3
  },
  "fixed_contract": {
    "N": 128,
    "gridsize": 1,
    "dataset_source": "synthetic_lines",
    "set_phi": true,
    "probe_source": "custom",
    "probe_npz": "datasets/Run1084_recon3_postPC_shrunk_3.npz",
    "probe_scale_mode": "pad_extrapolate",
    "probe_smoothing_sigma": 0.5,
    "probe_mask_diameter": null,
    "nimgs_train": 2,
    "nimgs_test": 2,
    "nphotons": 1000000000.0,
    "seed": 3,
    "torch_epochs": 40,
    "torch_learning_rate": 0.0002,
    "torch_scheduler": "ReduceLROnPlateau",
    "torch_plateau_factor": 0.5,
    "torch_plateau_patience": 2,
    "torch_plateau_min_lr": 0.0001,
    "torch_plateau_threshold": 0.0,
    "torch_loss_mode": "mae",
    "torch_mae_pred_l2_match_target": false,
    "torch_output_mode": "real_imag",
    "fno_modes": 12,
    "fno_width": 32,
    "fno_blocks": 4,
    "fno_cnn_blocks": 2
  },
  "fixed_sample_ids": [
    0,
    1
  ],
  "shared_visual_scales": {
    "amp": {
      "strategy": "shared_across_rows",
      "source": "stitched_numeric_arrays"
    },
    "phase": {
      "strategy": "shared_across_rows",
      "source": "stitched_numeric_arrays"
    },
    "amp_abs_error": {
      "strategy": "shared_across_rows",
      "source": "derived_from_stitched_numeric_arrays"
    },
    "phase_abs_error": {
      "strategy": "shared_across_rows",
      "source": "derived_from_stitched_numeric_arrays"
    }
  },
  "rows": [
    {
      "model_id": "baseline",
      "model_label": "CDI CNN + supervised",
      "architecture_id": "cnn",
      "training_procedure": "supervised",
      "required_for_minimum_subset": true
    },
    {
      "model_id": "pinn",
      "model_label": "CDI CNN + PINN",
      "architecture_id": "cnn",
      "training_procedure": "pinn",
      "required_for_minimum_subset": true
    },
    {
      "model_id": "pinn_hybrid_resnet",
      "model_label": "Hybrid ResNet + PINN",
      "architecture_id": "hybrid_resnet",
      "training_procedure": "pinn",
      "required_for_minimum_subset": true
    },
    {
      "model_id": "pinn_fno_vanilla",
      "model_label": "FNO Vanilla + PINN",
      "architecture_id": "fno_vanilla",
      "training_procedure": "pinn",
      "required_for_minimum_subset": true
    }
  ],
  "later_complete_table_rows": [
    "pinn_spectral_resnet_bottleneck_net",
    "pinn_ffno"
  ]
}
<!-- lines128_execution_authority_json:end -->
