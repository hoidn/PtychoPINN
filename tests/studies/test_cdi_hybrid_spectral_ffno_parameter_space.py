"""Tests for the CDI hybrid-spectral to FFNO parameter-space study harness."""

from __future__ import annotations

import json
from pathlib import Path
import shutil


FIXED_CONTRACT = {
    "N": 128,
    "gridsize": 1,
    "dataset_source": "synthetic_lines",
    "set_phi": True,
    "probe_source": "custom",
    "probe_npz": "datasets/Run1084_recon3_postPC_shrunk_3.npz",
    "probe_scale_mode": "pad_extrapolate",
    "probe_smoothing_sigma": 0.5,
    "probe_mask_diameter": None,
    "nimgs_train": 2,
    "nimgs_test": 2,
    "nphotons": 1e9,
    "seed": 3,
    "torch_epochs": 40,
    "torch_learning_rate": 2e-4,
    "torch_scheduler": "ReduceLROnPlateau",
    "torch_plateau_factor": 0.5,
    "torch_plateau_patience": 2,
    "torch_plateau_min_lr": 1e-4,
    "torch_plateau_threshold": 0.0,
    "torch_loss_mode": "mae",
    "torch_mae_pred_l2_match_target": False,
    "torch_output_mode": "real_imag",
    "fno_modes": 12,
    "fno_width": 32,
    "fno_blocks": 4,
    "fno_cnn_blocks": 2,
}


AUTHORITATIVE_ROOT = (
    Path(".artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog")
    / "2026-04-29-cdi-lines128-paper-benchmark-execution/runs"
    / "complete_table_20260430T150757Z_repair_tmux"
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fake_row_invocation(*, architecture: str, overrides: dict[str, object] | None = None) -> dict[str, object]:
    parsed_args = {
        "architecture": architecture,
        "seed": 3,
        "epochs": 40,
        "learning_rate": 2e-4,
        "generator_output_mode": "real_imag",
        "N": 128,
        "gridsize": 1,
        "probe_source": "custom",
        "torch_loss_mode": "mae",
        "torch_mae_pred_l2_match_target": False,
        "probe_mask": False,
        "fno_modes": 12,
        "fno_width": 32,
        "fno_blocks": 4,
        "fno_cnn_blocks": 2,
        "hybrid_downsample_steps": 2,
        "hybrid_downsample_op": "stride_conv",
        "spectral_bottleneck_blocks": 6,
        "spectral_bottleneck_modes": 12,
        "spectral_bottleneck_share_weights": True,
        "spectral_bottleneck_gate_init": 0.1,
        "spectral_bottleneck_gate_mode": "shared",
        "scheduler": "ReduceLROnPlateau",
        "plateau_factor": 0.5,
        "plateau_patience": 2,
        "plateau_min_lr": 1e-4,
        "plateau_threshold": 0.0,
    }
    if overrides:
        parsed_args.update(overrides)
    return {
        "script": "scripts/studies/grid_lines_torch_runner.py",
        "status": "completed",
        "exit_code": 0,
        "parsed_args": parsed_args,
        "extra": {
            "git_commit": "deadbeef",
            "runtime_provenance": {
                "python_executable": "/tmp/python",
                "cwd": "/tmp/repo",
                "pythonpath": "/tmp/repo",
                "ptycho_torch_file": "/tmp/repo/ptycho_torch/__init__.py",
            },
            "invocation_mode": "library",
        },
    }


def _build_fake_authoritative_root(tmp_path: Path, *, bad_row_overrides: dict[str, dict[str, object]] | None = None) -> Path:
    root = tmp_path / "authoritative"
    bad_row_overrides = bad_row_overrides or {}
    _write_json(
        root / "paper_benchmark_manifest.json",
        {
            "benchmark_status": "paper_complete",
            "claim_boundary": "complete_lines128_cdi_benchmark",
            "fixed_contract": dict(FIXED_CONTRACT),
            "dataset": {
                "train_npz": str(root / "datasets" / "N128" / "gs1" / "train.npz"),
                "test_npz": str(root / "datasets" / "N128" / "gs1" / "test.npz"),
                "gt_recon": str(root / "recons" / "gt" / "recon.npz"),
                "manifest_json": "dataset_identity_manifest.json",
            },
        },
    )
    _write_json(
        root / "dataset_identity_manifest.json",
        {
            "dataset_source": "synthetic_lines",
            "probe_source": "custom",
            "probe_scale_mode": "pad_extrapolate",
            "probe_npz": {
                "path": str(root / "datasets" / "Run1084_recon3_postPC_shrunk_3.npz"),
                "sha256": "abc123",
            },
            "train_npz": {"path": str(root / "datasets" / "N128" / "gs1" / "train.npz"), "sha256": "train"},
            "test_npz": {"path": str(root / "datasets" / "N128" / "gs1" / "test.npz"), "sha256": "test"},
        },
    )
    _write_json(
        root / "split_manifest.json",
        {
            "train_npz": str(root / "datasets" / "N128" / "gs1" / "train.npz"),
            "test_npz": str(root / "datasets" / "N128" / "gs1" / "test.npz"),
            "seed": 3,
            "nimgs_train": 2,
            "nimgs_test": 2,
            "gridsize": 1,
            "set_phi": True,
        },
    )
    _write_json(root / "invocation.json", {"status": "completed", "exit_code": 0})
    _write_json(root / "metrics.json", {"benchmark_status": "paper_complete"})
    _write_json(root / "model_manifest.json", {"rows": []})
    datasets_dir = root / "datasets" / "N128" / "gs1"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    (datasets_dir / "train.npz").write_bytes(b"train")
    (datasets_dir / "test.npz").write_bytes(b"test")
    (root / "datasets" / "Run1084_recon3_postPC_shrunk_3.npz").write_bytes(b"probe")
    (root / "recons" / "gt").mkdir(parents=True, exist_ok=True)
    (root / "recons" / "gt" / "recon.npz").write_bytes(b"gt")

    row_architectures = {
        "pinn_hybrid_resnet": "hybrid_resnet",
        "pinn_spectral_resnet_bottleneck_net": "spectral_resnet_bottleneck_net",
        "pinn_ffno": "ffno",
    }
    for model_id, architecture in row_architectures.items():
        run_dir = root / "runs" / model_id
        recon_dir = root / "recons" / model_id
        run_dir.mkdir(parents=True, exist_ok=True)
        recon_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            run_dir / "invocation.json",
            _fake_row_invocation(
                architecture=architecture,
                overrides=bad_row_overrides.get(model_id),
            ),
        )
        _write_json(run_dir / "config.json", {"model_id": model_id})
        _write_json(run_dir / "history.json", {"train_loss": [0.1], "val_loss": [0.2]})
        _write_json(run_dir / "metrics.json", {"mae": [0.1, 0.2]})
        _write_json(run_dir / "exit_code_proof.json", {"exit_code": 0})
        (recon_dir / "recon.npz").write_bytes(model_id.encode("utf-8"))
    return root


def test_preflight_artifacts_freeze_exact_six_row_matrix(tmp_path):
    from scripts.studies.cdi_hybrid_spectral_ffno_parameter_space import (
        build_preflight_artifacts,
    )

    note_path = tmp_path / "cdi_preflight.md"
    matrix_path = tmp_path / "preflight" / "study_matrix.json"
    reference_path = tmp_path / "preflight" / "reference_runs.json"

    result = build_preflight_artifacts(
        authoritative_root=AUTHORITATIVE_ROOT,
        artifact_root=tmp_path,
        note_path=note_path,
        matrix_path=matrix_path,
        reference_runs_path=reference_path,
    )

    assert result["study_matrix_path"] == matrix_path
    assert result["reference_runs_path"] == reference_path
    assert result["note_path"] == note_path
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert [row["model_id"] for row in matrix["rows"]] == [
        "pinn_hybrid_resnet",
        "pinn_spectral_resnet_bottleneck_net",
        "pinn_ffno",
        "pinn_spectral_resnet_bottleneck_ds1",
        "pinn_spectral_resnet_bottleneck_linear_decoder",
        "pinn_hybrid_resnet_ffno_bottleneck",
    ]


def test_preflight_note_repeats_row_details_and_output_layout(tmp_path):
    from scripts.studies.cdi_hybrid_spectral_ffno_parameter_space import (
        build_preflight_artifacts,
    )

    result = build_preflight_artifacts(
        authoritative_root=AUTHORITATIVE_ROOT,
        artifact_root=tmp_path / "study",
        note_path=tmp_path / "cdi_preflight.md",
        matrix_path=tmp_path / "preflight" / "study_matrix.json",
        reference_runs_path=tmp_path / "preflight" / "reference_runs.json",
    )

    note = Path(result["note_path"]).read_text(encoding="utf-8")
    assert "Output-root layout" in note
    assert "`pinn_spectral_resnet_bottleneck_ds1`" in note
    assert "Nearest anchor: `pinn_spectral_resnet_bottleneck_net`" in note
    assert "Expression path: runner override on `spectral_resnet_bottleneck_net`" in note
    assert "Display label: `Spectral ResNet Bottleneck DS1 + PINN`" in note
    assert "Reuse acceptability" in note


def test_reference_runs_manifest_maps_reused_rows_to_authoritative_root(tmp_path):
    from scripts.studies.cdi_hybrid_spectral_ffno_parameter_space import (
        build_preflight_artifacts,
    )

    result = build_preflight_artifacts(
        authoritative_root=AUTHORITATIVE_ROOT,
        artifact_root=tmp_path,
        note_path=tmp_path / "cdi_preflight.md",
        matrix_path=tmp_path / "preflight" / "study_matrix.json",
        reference_runs_path=tmp_path / "preflight" / "reference_runs.json",
    )

    payload = json.loads(Path(result["reference_runs_path"]).read_text(encoding="utf-8"))

    assert payload["authoritative_root"] == str(AUTHORITATIVE_ROOT)
    assert [row["model_id"] for row in payload["reused_rows"]] == [
        "pinn_hybrid_resnet",
        "pinn_spectral_resnet_bottleneck_net",
        "pinn_ffno",
    ]
    assert payload["reused_rows"][0]["run_dir"].endswith("runs/pinn_hybrid_resnet")
    assert payload["fixed_contract"]["probe_scale_mode"] == "pad_extrapolate"
    assert payload["reused_rows"][0]["validation"]["status"] == "accepted"


def test_runbook_preflight_only_validates_fresh_rows_without_training(monkeypatch, tmp_path):
    from scripts.studies.runbooks.run_cdi_hybrid_spectral_ffno_parameter_space import (
        run_cdi_parameter_space_study,
    )

    captured = {}

    def fake_compare(**kwargs):
        captured["models"] = kwargs["models"]
        captured["preflight_only"] = kwargs["preflight_only"]
        captured["row_specs"] = kwargs["row_specs"]
        return {
            "mode": "preflight_only",
            "selected_models": list(kwargs["models"]),
        }

    monkeypatch.setattr(
        "scripts.studies.runbooks.run_cdi_hybrid_spectral_ffno_parameter_space.run_grid_lines_compare",
        fake_compare,
    )

    result = run_cdi_parameter_space_study(
        authoritative_root=AUTHORITATIVE_ROOT,
        output_root=tmp_path / "study",
        preflight_root=tmp_path / "preflight",
        note_path=tmp_path / "cdi_preflight.md",
        preflight_only=True,
    )

    assert captured["preflight_only"] is True
    assert captured["models"] == (
        "pinn_spectral_resnet_bottleneck_ds1",
        "pinn_spectral_resnet_bottleneck_linear_decoder",
        "pinn_hybrid_resnet_ffno_bottleneck",
    )
    assert [spec["model_id"] for spec in captured["row_specs"]] == list(captured["models"])
    assert result["preflight_validation"]["selected_models"] == list(captured["models"])


def test_materialize_reused_rows_copies_without_mutating_authoritative_root(tmp_path):
    from scripts.studies.runbooks.run_cdi_hybrid_spectral_ffno_parameter_space import (
        _materialize_reused_rows,
    )

    authoritative_root = _build_fake_authoritative_root(tmp_path)
    output_root = tmp_path / "study"
    _materialize_reused_rows(authoritative_root=authoritative_root, output_root=output_root)

    copied_run_dir = output_root / "runs" / "pinn_hybrid_resnet"
    copied_recon = output_root / "recons" / "pinn_hybrid_resnet" / "recon.npz"
    source_recon = authoritative_root / "recons" / "pinn_hybrid_resnet" / "recon.npz"

    assert copied_run_dir.exists()
    assert copied_run_dir.is_symlink() is False
    assert copied_recon.is_symlink() is False

    copied_recon.write_bytes(b"mutated-study-copy")
    assert source_recon.read_bytes() == b"pinn_hybrid_resnet"


def test_build_preflight_artifacts_fails_closed_on_contract_mismatch(tmp_path):
    from scripts.studies.cdi_hybrid_spectral_ffno_parameter_space import (
        build_preflight_artifacts,
    )

    authoritative_root = _build_fake_authoritative_root(
        tmp_path,
        bad_row_overrides={"pinn_ffno": {"epochs": 10}},
    )

    try:
        build_preflight_artifacts(
            authoritative_root=authoritative_root,
            artifact_root=tmp_path / "study",
            note_path=tmp_path / "cdi_preflight.md",
            matrix_path=tmp_path / "preflight" / "study_matrix.json",
            reference_runs_path=tmp_path / "preflight" / "reference_runs.json",
        )
    except ValueError as exc:
        assert "pinn_ffno" in str(exc)
        assert "epochs" in str(exc)
    else:
        raise AssertionError("expected contract validation failure")


def test_validate_bundle_reports_missing_frozen_rows(tmp_path):
    from scripts.studies.runbooks.run_cdi_hybrid_spectral_ffno_parameter_space import (
        validate_cdi_parameter_space_bundle,
    )
    from scripts.studies.cdi_hybrid_spectral_ffno_parameter_space import build_preflight_artifacts

    paths = build_preflight_artifacts(
        authoritative_root=AUTHORITATIVE_ROOT,
        artifact_root=tmp_path / "study",
        note_path=tmp_path / "cdi_preflight.md",
        matrix_path=tmp_path / "preflight" / "study_matrix.json",
        reference_runs_path=tmp_path / "preflight" / "reference_runs.json",
    )

    report = validate_cdi_parameter_space_bundle(
        output_root=tmp_path / "study",
        study_matrix_path=paths["study_matrix_path"],
    )

    assert report["ok"] is False
    assert "pinn_hybrid_resnet" in report["missing_rows"]


def test_validate_bundle_rejects_reused_symlink_materialization(tmp_path):
    from scripts.studies.runbooks.run_cdi_hybrid_spectral_ffno_parameter_space import (
        validate_cdi_parameter_space_bundle,
    )
    from scripts.studies.cdi_hybrid_spectral_ffno_parameter_space import build_preflight_artifacts

    authoritative_root = _build_fake_authoritative_root(tmp_path)
    paths = build_preflight_artifacts(
        authoritative_root=authoritative_root,
        artifact_root=tmp_path / "study",
        note_path=tmp_path / "cdi_preflight.md",
        matrix_path=tmp_path / "preflight" / "study_matrix.json",
        reference_runs_path=tmp_path / "preflight" / "reference_runs.json",
    )

    study_root = tmp_path / "study"
    (study_root / "runs").mkdir(parents=True, exist_ok=True)
    (study_root / "recons").mkdir(parents=True, exist_ok=True)
    shutil.copytree(authoritative_root / "runs" / "pinn_hybrid_resnet", study_root / "runs" / "pinn_hybrid_resnet")
    shutil.copytree(authoritative_root / "runs" / "pinn_ffno", study_root / "runs" / "pinn_ffno")
    shutil.copytree(authoritative_root / "runs" / "pinn_spectral_resnet_bottleneck_net", study_root / "runs" / "pinn_spectral_resnet_bottleneck_net")
    shutil.copytree(authoritative_root / "recons" / "pinn_hybrid_resnet", study_root / "recons" / "pinn_hybrid_resnet")
    shutil.copytree(authoritative_root / "recons" / "pinn_ffno", study_root / "recons" / "pinn_ffno")
    shutil.copytree(authoritative_root / "recons" / "pinn_spectral_resnet_bottleneck_net", study_root / "recons" / "pinn_spectral_resnet_bottleneck_net")
    shutil.copytree(authoritative_root / "recons" / "gt", study_root / "recons" / "gt")
    for model_id in [
        "pinn_spectral_resnet_bottleneck_ds1",
        "pinn_spectral_resnet_bottleneck_linear_decoder",
        "pinn_hybrid_resnet_ffno_bottleneck",
    ]:
        run_dir = study_root / "runs" / model_id
        recon_dir = study_root / "recons" / model_id
        run_dir.mkdir(parents=True, exist_ok=True)
        recon_dir.mkdir(parents=True, exist_ok=True)
        _write_json(run_dir / "invocation.json", {"status": "completed"})
        _write_json(run_dir / "config.json", {})
        _write_json(run_dir / "history.json", {})
        _write_json(run_dir / "metrics.json", {})
        _write_json(run_dir / "exit_code_proof.json", {"exit_code": 0})
        (recon_dir / "recon.npz").write_bytes(model_id.encode("utf-8"))

    shutil.rmtree(study_root / "recons" / "pinn_ffno")
    (study_root / "recons" / "pinn_ffno").symlink_to(authoritative_root / "recons" / "pinn_ffno", target_is_directory=True)

    report = validate_cdi_parameter_space_bundle(
        output_root=study_root,
        study_matrix_path=paths["study_matrix_path"],
        reference_runs_path=paths["reference_runs_path"],
    )

    assert report["ok"] is False
    assert "pinn_ffno" in report["reused_root_drift"]


def _write_completed_fresh_row(study_root: Path, model_id: str) -> None:
    run_dir = study_root / "runs" / model_id
    recon_dir = study_root / "recons" / model_id
    run_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "invocation.json", {"status": "completed", "exit_code": 0})
    _write_json(run_dir / "config.json", {"model_id": model_id})
    _write_json(run_dir / "history.json", {"train_loss": [0.1], "val_loss": [0.2]})
    _write_json(run_dir / "metrics.json", {"mae": [0.1, 0.2]})
    _write_json(run_dir / "exit_code_proof.json", {"exit_code": 0})
    (recon_dir / "recon.npz").write_bytes(model_id.encode("utf-8"))


def _write_collated_outputs(study_root: Path, row_ids: list[str]) -> None:
    metrics_by_model = {
        model_id: {
            "reference_shape": [128, 128],
            "metrics": {"amp_mae": 0.1, "phase_mae": 0.2},
        }
        for model_id in row_ids
    }
    _write_json(study_root / "metrics_by_model.json", metrics_by_model)
    _write_json(
        study_root / "metrics.json",
        {model_id: payload["metrics"] for model_id, payload in metrics_by_model.items()},
    )
    (study_root / "metrics_table.csv").write_text(
        "model_id,model_label,N,amp_mae,phase_mae\n"
        + "\n".join(f"{model_id},{model_id},128,0.1,0.2" for model_id in row_ids)
        + "\n",
        encoding="utf-8",
    )
    (study_root / "metrics_table.tex").write_text(
        "\\begin{tabular}{lll}\nModel & N & amp\\_mae \\\\\n"
        + "\n".join(f"{model_id} & 128 & 0.1 \\\\" for model_id in row_ids)
        + "\n\\end{tabular}\n",
        encoding="utf-8",
    )


def _build_completed_study_root(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    from scripts.studies.cdi_hybrid_spectral_ffno_parameter_space import build_preflight_artifacts
    from scripts.studies.runbooks.run_cdi_hybrid_spectral_ffno_parameter_space import (
        _all_row_ids,
        _materialize_reused_rows,
    )

    authoritative_root = _build_fake_authoritative_root(tmp_path)
    paths = build_preflight_artifacts(
        authoritative_root=authoritative_root,
        artifact_root=tmp_path / "study",
        note_path=tmp_path / "cdi_preflight.md",
        matrix_path=tmp_path / "preflight" / "study_matrix.json",
        reference_runs_path=tmp_path / "preflight" / "reference_runs.json",
    )
    study_root = tmp_path / "study"
    _materialize_reused_rows(authoritative_root=authoritative_root, output_root=study_root)
    for model_id in [
        "pinn_spectral_resnet_bottleneck_ds1",
        "pinn_spectral_resnet_bottleneck_linear_decoder",
        "pinn_hybrid_resnet_ffno_bottleneck",
    ]:
        _write_completed_fresh_row(study_root, model_id)
    _write_collated_outputs(study_root, list(_all_row_ids()))
    return study_root, paths


def test_validate_bundle_rejects_failed_fresh_row_completion_proof(tmp_path):
    from scripts.studies.runbooks.run_cdi_hybrid_spectral_ffno_parameter_space import (
        validate_cdi_parameter_space_bundle,
    )

    study_root, paths = _build_completed_study_root(tmp_path)
    failed_model_id = "pinn_spectral_resnet_bottleneck_ds1"
    _write_json(
        study_root / "runs" / failed_model_id / "invocation.json",
        {"status": "failed", "exit_code": 1},
    )
    _write_json(
        study_root / "runs" / failed_model_id / "exit_code_proof.json",
        {"exit_code": 1},
    )

    report = validate_cdi_parameter_space_bundle(
        output_root=study_root,
        study_matrix_path=paths["study_matrix_path"],
        reference_runs_path=paths["reference_runs_path"],
    )

    assert report["ok"] is False
    assert failed_model_id in report["fresh_row_completion_failures"]
    assert any("status='failed'" in issue for issue in report["fresh_row_completion_failures"][failed_model_id])
    assert any("exit_code=1" in issue for issue in report["fresh_row_completion_failures"][failed_model_id])


def test_validate_bundle_rejects_missing_merged_outputs(tmp_path):
    from scripts.studies.runbooks.run_cdi_hybrid_spectral_ffno_parameter_space import (
        validate_cdi_parameter_space_bundle,
    )

    study_root, paths = _build_completed_study_root(tmp_path)
    for path in [
        study_root / "metrics_by_model.json",
        study_root / "metrics.json",
        study_root / "metrics_table.csv",
        study_root / "metrics_table.tex",
    ]:
        path.unlink()

    report = validate_cdi_parameter_space_bundle(
        output_root=study_root,
        study_matrix_path=paths["study_matrix_path"],
        reference_runs_path=paths["reference_runs_path"],
    )

    assert report["ok"] is False
    assert sorted(report["missing_merged_outputs"]) == [
        "metrics.json",
        "metrics_by_model.json",
        "metrics_table.csv",
        "metrics_table.tex",
    ]


def test_runbook_reruns_failed_fresh_row_instead_of_reusing_stale_recon(monkeypatch, tmp_path):
    from scripts.studies.runbooks.run_cdi_hybrid_spectral_ffno_parameter_space import (
        run_cdi_parameter_space_study,
    )

    authoritative_root = _build_fake_authoritative_root(tmp_path)
    output_root = tmp_path / "study"
    stale_model_id = "pinn_spectral_resnet_bottleneck_ds1"
    stale_run_dir = output_root / "runs" / stale_model_id
    stale_recon_dir = output_root / "recons" / stale_model_id
    stale_run_dir.mkdir(parents=True, exist_ok=True)
    stale_recon_dir.mkdir(parents=True, exist_ok=True)
    _write_json(stale_run_dir / "invocation.json", {"status": "failed", "exit_code": 1})
    (stale_recon_dir / "recon.npz").write_bytes(b"stale-recon")

    compare_calls: list[tuple[tuple[str, ...], bool]] = []

    def fake_compare(**kwargs):
        models = tuple(kwargs["models"])
        compare_calls.append((models, bool(kwargs.get("reuse_existing_recons", False))))
        if models == (stale_model_id,):
            assert not (output_root / "recons" / stale_model_id / "recon.npz").exists()
            assert not (output_root / "runs" / stale_model_id / "invocation.json").exists()
        if len(models) == 1:
            _write_completed_fresh_row(output_root, models[0])
        else:
            _write_collated_outputs(output_root, list(models))
        return {"selected_models": list(models)}

    monkeypatch.setattr(
        "scripts.studies.runbooks.run_cdi_hybrid_spectral_ffno_parameter_space.run_grid_lines_compare",
        fake_compare,
    )

    run_cdi_parameter_space_study(
        authoritative_root=authoritative_root,
        output_root=output_root,
        preflight_root=tmp_path / "preflight",
        note_path=tmp_path / "cdi_preflight.md",
    )

    assert compare_calls[0] == ((stale_model_id,), False)
    assert any(models == ("pinn_spectral_resnet_bottleneck_linear_decoder",) for models, _ in compare_calls)
    assert any(models == ("pinn_hybrid_resnet_ffno_bottleneck",) for models, _ in compare_calls)
    assert compare_calls[-1][1] is True
    assert json.loads((output_root / "runs" / stale_model_id / "invocation.json").read_text(encoding="utf-8"))[
        "status"
    ] == "completed"


def test_runbook_fails_closed_before_fresh_launch_when_reused_copy_drifted(monkeypatch, tmp_path):
    from scripts.studies.runbooks.run_cdi_hybrid_spectral_ffno_parameter_space import (
        _materialize_reused_rows,
        run_cdi_parameter_space_study,
    )

    authoritative_root = _build_fake_authoritative_root(tmp_path)
    output_root = tmp_path / "study"
    _materialize_reused_rows(authoritative_root=authoritative_root, output_root=output_root)
    (output_root / "recons" / "pinn_ffno" / "recon.npz").write_bytes(b"drifted-local-copy")

    compare_calls: list[tuple[str, ...]] = []

    def fake_compare(**kwargs):
        compare_calls.append(tuple(kwargs["models"]))
        return {"selected_models": list(kwargs["models"])}

    monkeypatch.setattr(
        "scripts.studies.runbooks.run_cdi_hybrid_spectral_ffno_parameter_space.run_grid_lines_compare",
        fake_compare,
    )

    try:
        run_cdi_parameter_space_study(
            authoritative_root=authoritative_root,
            output_root=output_root,
            preflight_root=tmp_path / "preflight",
            note_path=tmp_path / "cdi_preflight.md",
        )
    except RuntimeError as exc:
        assert "reused" in str(exc).lower()
        assert "pinn_ffno" in str(exc)
    else:
        raise AssertionError("expected reused-root drift failure before fresh launches")

    assert compare_calls == []


def test_runbook_raises_when_final_bundle_validation_fails(monkeypatch, tmp_path):
    from scripts.studies.runbooks.run_cdi_hybrid_spectral_ffno_parameter_space import (
        run_cdi_parameter_space_study,
    )

    authoritative_root = _build_fake_authoritative_root(tmp_path)
    output_root = tmp_path / "study"

    def fake_compare(**kwargs):
        models = tuple(kwargs["models"])
        if len(models) == 1:
            _write_completed_fresh_row(output_root, models[0])
        return {"selected_models": list(models)}

    monkeypatch.setattr(
        "scripts.studies.runbooks.run_cdi_hybrid_spectral_ffno_parameter_space.run_grid_lines_compare",
        fake_compare,
    )
    monkeypatch.setattr(
        "scripts.studies.runbooks.run_cdi_hybrid_spectral_ffno_parameter_space.validate_cdi_parameter_space_bundle",
        lambda **_: {"ok": False, "missing_rows": [], "missing_artifacts": {}, "reused_root_drift": {"pinn_ffno": ["sha drift"]}},
    )

    try:
        run_cdi_parameter_space_study(
            authoritative_root=authoritative_root,
            output_root=output_root,
            preflight_root=tmp_path / "preflight",
            note_path=tmp_path / "cdi_preflight.md",
        )
    except RuntimeError as exc:
        assert "bundle validation failed" in str(exc).lower()
        assert "pinn_ffno" in str(exc)
    else:
        raise AssertionError("expected final bundle validation failure")
