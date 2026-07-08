from pathlib import Path
import json

import numpy as np


def _write_recon(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    yy = np.ones((32, 32), dtype=np.complex64)
    np.savez(path, YY_pred=yy, amp=np.abs(yy), phase=np.angle(yy))


def test_metrics_visual_stage_writes_metrics_json_metrics_by_model_and_tables(
    monkeypatch, tmp_path
):
    from scripts.studies.nersc_orchestration import aggregate_metrics_visuals_stage

    gt = tmp_path / "recons" / "gt" / "recon.npz"
    pvit = tmp_path / "recons" / "pinn_ptychovit" / "recon.npz"
    hybrid = tmp_path / "recons" / "pinn_hybrid_resnet" / "recon.npz"
    _write_recon(gt)
    _write_recon(pvit)
    _write_recon(hybrid)

    monkeypatch.setattr(
        "scripts.studies.nersc_orchestration.evaluate_selected_models",
        lambda recon_paths, gt_path: {
            "pinn_ptychovit": {"metrics": {"mse": [0.1, 0.2]}},
            "pinn_hybrid_resnet": {"metrics": {"mse": [0.3, 0.4]}},
        },
    )

    def fake_finalize(output_dir, merged_metrics, visual_order, model_ns):
        _ = (merged_metrics, visual_order, model_ns)
        (output_dir / "metrics.json").write_text(json.dumps({"ok": True}))
        (output_dir / "metrics_table.md").write_text("table")
        (output_dir / "metrics_table.tex").write_text("table")
        return {
            "metrics_path": str(output_dir / "metrics.json"),
            "metrics_table_md": str(output_dir / "metrics_table.md"),
            "metrics_table_tex": str(output_dir / "metrics_table.tex"),
        }

    monkeypatch.setattr("scripts.studies.nersc_orchestration._finalize_compare_outputs", fake_finalize)

    aggregate_metrics_visuals_stage(
        dataset_output_dir=tmp_path,
        recon_paths={
            "pinn_ptychovit": pvit,
            "pinn_hybrid_resnet": hybrid,
        },
        gt_recon_path=gt,
    )

    assert (tmp_path / "metrics_by_model.json").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "metrics_table.md").exists()
    assert (tmp_path / "metrics_table.tex").exists()


def test_metrics_visual_stage_renders_compare_amp_phase_png(monkeypatch, tmp_path):
    from scripts.studies.nersc_orchestration import aggregate_metrics_visuals_stage

    gt = tmp_path / "recons" / "gt" / "recon.npz"
    pvit = tmp_path / "recons" / "pinn_ptychovit" / "recon.npz"
    hybrid = tmp_path / "recons" / "pinn_hybrid_resnet" / "recon.npz"
    _write_recon(gt)
    _write_recon(pvit)
    _write_recon(hybrid)

    monkeypatch.setattr(
        "scripts.studies.nersc_orchestration.evaluate_selected_models",
        lambda recon_paths, gt_path: {
            "pinn_ptychovit": {"metrics": {"mse": [0.1, 0.2]}},
            "pinn_hybrid_resnet": {"metrics": {"mse": [0.3, 0.4]}},
        },
    )

    def fake_finalize(output_dir, merged_metrics, visual_order, model_ns):
        _ = (merged_metrics, visual_order, model_ns)
        visuals = output_dir / "visuals"
        visuals.mkdir(parents=True, exist_ok=True)
        (visuals / "compare_amp_phase.png").write_bytes(b"png")
        return {"metrics_path": str(output_dir / "metrics.json")}

    monkeypatch.setattr("scripts.studies.nersc_orchestration._finalize_compare_outputs", fake_finalize)

    aggregate_metrics_visuals_stage(
        dataset_output_dir=tmp_path,
        recon_paths={
            "pinn_ptychovit": pvit,
            "pinn_hybrid_resnet": hybrid,
        },
        gt_recon_path=gt,
    )

    assert (tmp_path / "visuals" / "compare_amp_phase.png").exists()
