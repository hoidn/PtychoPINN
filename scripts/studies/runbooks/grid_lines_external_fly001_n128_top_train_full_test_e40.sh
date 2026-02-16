#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import json
from pathlib import Path

from ptycho.workflows.grid_lines_workflow import GridLinesConfig
from scripts.studies.grid_study_dataset_builder import build_datasets
from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig, run_grid_lines_torch
from scripts.studies.grid_lines_compare_wrapper import evaluate_selected_models, _finalize_compare_outputs

OUT = Path("outputs/grid_lines_external_fly001_n128_top_train_full_test_e40_seed3_cnn_hybrid_resnet")
TRAIN_RAW = Path("datasets/fly001_128/fly001_128_top_half_converted.npz")
TEST_RAW = Path("datasets/fly001_128/fly001_128_full_test_converted.npz")

OUT.mkdir(parents=True, exist_ok=True)

cfg = GridLinesConfig(
    N=128,
    gridsize=1,
    output_dir=OUT,
    probe_npz=Path("datasets/Run1084_recon3_postPC_shrunk_3.npz"),
    nimgs_train=1,
    nimgs_test=1,
    nphotons=1e9,
    nepochs=40,
    batch_size=8,
    nll_weight=0.0,
    mae_weight=1.0,
    realspace_weight=0.0,
    probe_smoothing_sigma=0.5,
    probe_source="custom",
    probe_scale_mode="pad_extrapolate",
    set_phi=False,
)

bundles = build_datasets(
    dataset_source="external_raw_npz",
    cfg=cfg,
    required_ns=[128],
    train_data=TRAIN_RAW,
    test_data=TEST_RAW,
    n_groups=None,
    n_subsample=None,
    neighbor_count=7,
    subsample_seed=3,
)

train_npz = Path(bundles[128]["train_npz"])
test_npz = Path(bundles[128]["test_npz"])

common = dict(
    train_npz=train_npz,
    test_npz=test_npz,
    output_dir=OUT,
    seed=3,
    epochs=40,
    batch_size=8,
    learning_rate=2e-4,
    infer_batch_size=128,
    gradient_clip_val=0.0,
    gradient_clip_algorithm="norm",
    generator_output_mode="real_imag",
    N=128,
    gridsize=1,
    probe_source="custom",
    torch_loss_mode="mae",
    fno_modes=12,
    fno_width=32,
    fno_blocks=4,
    fno_cnn_blocks=2,
    optimizer="adam",
    weight_decay=0.0,
    adam_beta1=0.9,
    adam_beta2=0.999,
    scheduler="ReduceLROnPlateau",
    plateau_factor=0.5,
    plateau_patience=2,
    plateau_min_lr=5e-5,
    plateau_threshold=0.0,
    enable_checkpointing=False,
    reassembly_mode="position",
)

print("[study] running hybrid_resnet")
_ = run_grid_lines_torch(TorchRunnerConfig(architecture="hybrid_resnet", **common))

print("[study] running cnn")
_ = run_grid_lines_torch(TorchRunnerConfig(architecture="cnn", **common))

recon_paths = {
    "pinn_cnn": OUT / "recons" / "pinn_cnn" / "recon.npz",
    "pinn_hybrid_resnet": OUT / "recons" / "pinn_hybrid_resnet" / "recon.npz",
}
gt_path = OUT / "recons" / "gt" / "recon.npz"

metrics_by_model = evaluate_selected_models(recon_paths, gt_path)
(OUT / "metrics_by_model.json").write_text(json.dumps(metrics_by_model, indent=2, default=str))
legacy_metrics = {model_id: payload["metrics"] for model_id, payload in metrics_by_model.items()}

_finalize_compare_outputs(
    output_dir=OUT,
    merged_metrics=legacy_metrics,
    visual_order=("gt", "pinn_cnn", "pinn_hybrid_resnet"),
    model_ns={"pinn_cnn": 128, "pinn_hybrid_resnet": 128},
)

print("[study] complete")
print("output_dir", OUT)
print("metrics", OUT / "metrics.json")
PY

echo "__STUDY_DONE__:0"
