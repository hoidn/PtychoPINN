# Lines128 Supervised-Equivalent Rows Execution Authority

- Date: `2026-04-30`
- Backlog item: `2026-04-29-cdi-lines128-supervised-equivalent-rows`
- Scope: one adjacent supervised-extension bundle for the locked `lines128` CDI contract.
- Primary claim boundary: `lines128_supervised_ffno_extension`

This authority freezes the supervised-equivalent extension contract without reopening the
authoritative six-row CDI paper bundle. The existing six-row `paper_complete` bundle remains the
main CDI claim authority; this extension only adds the same-architecture `FFNO + supervised`
control row and reuses preserved same-contract comparator evidence.

Preserved prerequisite roots:

- minimum-subset supervised reference root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260430T084339Z`
- authoritative complete-table root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`
- preserved prerequisite FFNO-vs-Hybrid pair root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet`

The supervised extension may promote preserved rows by reference, but it must not rewrite or
relaunch those accepted roots.

<!-- lines128_supervised_equivalent_authority_json:start -->
{
  "state": "go_for_supervised_equivalent_execution",
  "claim_boundary": "lines128_supervised_ffno_extension",
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
  "fixed_sample_ids": [0, 1],
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
  "preserved_roots": {
    "minimum_subset_root": ".artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260430T084339Z",
    "complete_table_root": ".artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux",
    "ffno_pair_root": ".artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet"
  },
  "rows": [
    {
      "model_id": "baseline",
      "model_label": "CDI CNN + supervised",
      "architecture_id": "cnn",
      "training_procedure": "supervised",
      "decision": "promote_existing",
      "source_root": ".artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260430T084339Z"
    },
    {
      "model_id": "pinn_ffno",
      "model_label": "FFNO + PINN",
      "architecture_id": "ffno",
      "training_procedure": "pinn",
      "decision": "promote_existing",
      "source_root": ".artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux"
    },
    {
      "model_id": "supervised_ffno",
      "model_label": "FFNO + supervised",
      "architecture_id": "ffno",
      "training_procedure": "supervised",
      "decision": "rerun_required"
    }
  ]
}
<!-- lines128_supervised_equivalent_authority_json:end -->
