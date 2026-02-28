import argparse
import csv
import json
from pathlib import Path
import sys
import types

from scripts.studies.runbooks import run_hybrid_resnet_mode_skip_sweep as sweep


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_source_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "summary_schema_version",
        "run_id",
        "stage_id",
        "substage_id",
        "modes",
        "skip",
        "width",
        "amp_mae",
        "amp_mse",
        "phase_ssim_drop_vs_baseline",
        "train_wall_time_sec",
        "inference_time_s",
        "model_params",
        "pareto_rank_macro",
        "is_feasible",
        "is_stage_anchor",
    ]
    _write_csv(path, fieldnames, rows)


def _write_seed_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "summary_schema_version",
        "run_id",
        "seed",
        "modes",
        "skip",
        "width",
        "amp_mae",
        "amp_mse",
        "train_wall_time_sec",
        "inference_time_s",
        "model_params",
        "phase_ssim_drop_vs_baseline",
        "is_feasible",
    ]
    _write_csv(path, fieldnames, rows)


def test_stage_id_guardrail_rejects_missing_stage_c_substage(tmp_path, capsys):
    output_root = tmp_path / "out"
    rc = sweep.main([
        "--stage-id",
        "C",
        "--ns",
        "128",
        "--output-root",
        str(output_root),
    ])
    assert rc == 1
    assert "Stage C requires substage_id" in capsys.readouterr().err


def test_matrix_guardrail_rejects_non_active_axis_multivalue_stage_b(tmp_path, capsys):
    source_summary = tmp_path / "source.csv"
    _write_source_summary(
        source_summary,
        [
            {
                "summary_schema_version": "v1",
                "run_id": "anchor",
                "stage_id": "A",
                "substage_id": "none",
                "modes": "12",
                "skip": "off",
                "width": "32",
                "amp_mae": 0.08,
                "amp_mse": 0.01,
                "phase_ssim_drop_vs_baseline": 0.0,
                "train_wall_time_sec": 100,
                "inference_time_s": 1.0,
                "model_params": 1000,
                "pareto_rank_macro": 1,
                "is_feasible": True,
                "is_stage_anchor": True,
            }
        ],
    )

    rc = sweep.main([
        "--stage-id",
        "B",
        "--ns",
        "128",
        "--promotion-source-summary",
        str(source_summary),
        "--modes",
        "12,16",
        "--fno-blocks-values",
        "4,5",
        "--output-root",
        str(tmp_path / "out"),
    ])

    assert rc == 1
    assert "non-active" in capsys.readouterr().err.lower()


def test_stage_a_guardrail_rejects_non_default_scalar_structural_axis(tmp_path, capsys):
    rc = sweep.main([
        "--stage-id",
        "A",
        "--ns",
        "128",
        "--fno-blocks-values",
        "5",
        "--output-root",
        str(tmp_path / "out"),
    ])

    assert rc == 1
    stderr = capsys.readouterr().err
    assert "Stage A can only vary {modes, skip-values, widths}" in stderr
    assert "fno_blocks=5" in stderr


def test_guardrail_rejects_missing_summary_schema_version(tmp_path, capsys):
    source_summary = tmp_path / "source_missing_schema.csv"
    _write_csv(
        source_summary,
        ["run_id", "modes", "skip", "width", "is_stage_anchor"],
        [{"run_id": "anchor", "modes": 12, "skip": "off", "width": 32, "is_stage_anchor": True}],
    )

    rc = sweep.main([
        "--stage-id",
        "B",
        "--ns",
        "128",
        "--promotion-source-summary",
        str(source_summary),
        "--output-root",
        str(tmp_path / "out"),
    ])

    assert rc == 1
    assert "summary_schema_version" in capsys.readouterr().err


def test_guardrail_rejects_unknown_summary_schema_version(tmp_path, capsys):
    source_summary = tmp_path / "source_unknown_schema.csv"
    _write_csv(
        source_summary,
        [
            "summary_schema_version",
            "run_id",
            "modes",
            "skip",
            "width",
            "amp_mae",
            "amp_mse",
            "phase_ssim_drop_vs_baseline",
            "train_wall_time_sec",
            "inference_time_s",
            "model_params",
            "is_stage_anchor",
        ],
        [
            {
                "summary_schema_version": "unknown.v99",
                "run_id": "anchor",
                "modes": 12,
                "skip": "off",
                "width": 32,
                "amp_mae": 0.08,
                "amp_mse": 0.01,
                "phase_ssim_drop_vs_baseline": 0.0,
                "train_wall_time_sec": 100,
                "inference_time_s": 1.0,
                "model_params": 1000,
                "is_stage_anchor": True,
            }
        ],
    )

    rc = sweep.main([
        "--stage-id",
        "B",
        "--ns",
        "128",
        "--promotion-source-summary",
        str(source_summary),
        "--output-root",
        str(tmp_path / "out"),
    ])

    assert rc == 1
    assert "unsupported summary_schema_version" in capsys.readouterr().err


def test_guardrail_rejects_missing_required_summary_columns(tmp_path, capsys):
    source_summary = tmp_path / "source_missing_columns.csv"
    _write_csv(
        source_summary,
        ["summary_schema_version", "run_id", "modes", "skip", "width", "is_stage_anchor"],
        [
            {
                "summary_schema_version": "v1",
                "run_id": "anchor",
                "modes": 12,
                "skip": "off",
                "width": 32,
                "is_stage_anchor": True,
            }
        ],
    )

    rc = sweep.main([
        "--stage-id",
        "B",
        "--ns",
        "128",
        "--promotion-source-summary",
        str(source_summary),
        "--output-root",
        str(tmp_path / "out"),
    ])

    assert rc == 1
    assert "missing required columns" in capsys.readouterr().err.lower()


def test_guardrail_rejects_invalid_resnet_width_value(tmp_path, capsys):
    rc = sweep.main([
        "--stage-id",
        "D",
        "--substage-id",
        "D3",
        "--ns",
        "128",
        "--promotion-source-summary",
        str(tmp_path / "source.csv"),
        "--resnet-width-values",
        "30",
        "--output-root",
        str(tmp_path / "out"),
    ])

    assert rc == 1
    assert "resnet_width must be positive and divisible by 4" in capsys.readouterr().err


def test_invocation_and_cleanup_artifacts_are_written(tmp_path, monkeypatch):
    train_npz = tmp_path / "train.npz"
    test_npz = tmp_path / "test.npz"

    def _fake_runner(*, args, candidate, run_dir, train_npz, test_npz):
        _ = (args, candidate, train_npz, test_npz)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "lightning_logs" / "version_0").mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints" / "epoch=0-step=1.ckpt").write_bytes(b"r" * 128)
        (run_dir / "lightning_logs" / "version_0" / "events.out.tfevents").write_bytes(b"s" * 96)
        runtime_root = run_dir / "runs" / "pinn_hybrid_resnet"
        (runtime_root / "checkpoints").mkdir(parents=True, exist_ok=True)
        (runtime_root / "lightning_logs" / "version_0").mkdir(parents=True, exist_ok=True)
        (runtime_root / "checkpoints" / "epoch=0-step=1.ckpt").write_bytes(b"c" * 128)
        (runtime_root / "lightning_logs" / "version_0" / "events.out.tfevents").write_bytes(b"l" * 96)
        (runtime_root / "model.pt").write_bytes(b"m" * 64)
        return {
            "amp_mae": 0.08 if candidate["skip"] == "off" else 0.09,
            "amp_mse": 0.01,
            "phase_ssim": 0.9,
            "phase_ssim_drop_vs_baseline": 0.0,
            "model_params": 1234,
            "train_wall_time_sec": 12.0,
            "inference_time_s": 1.0,
        }

    monkeypatch.setattr(sweep, "_run_candidate_with_runner", _fake_runner)

    output_root = tmp_path / "stage_a"
    rc = sweep.main([
        "--stage-id",
        "A",
        "--ns",
        "128",
        "--dataset-profiles-n128",
        "custom_npz_pair_n128",
        "--custom-n128-train-npz",
        str(train_npz),
        "--custom-n128-test-npz",
        str(test_npz),
        "--modes",
        "12",
        "--skip-values",
        "off,on",
        "--widths",
        "32",
        "--probe-mask",
        "--torch-mae-pred-l2-match-target",
        "--output-root",
        str(output_root),
    ])
    assert rc == 0

    assert (output_root / "invocation.json").exists()
    assert (output_root / "invocation.sh").exists()

    summary_rows = list(csv.DictReader((output_root / "summary.csv").open()))
    assert len(summary_rows) == 2
    assert "train_wall_time_sec" in summary_rows[0]
    assert "phase_ssim_drop_vs_baseline" in summary_rows[0]
    assert "is_feasible" in summary_rows[0]
    assert "pareto_rank_macro" in summary_rows[0]
    assert "probe_mask_enabled" in summary_rows[0]
    assert "torch_mae_pred_l2_match_target" in summary_rows[0]
    assert summary_rows[0]["probe_mask_enabled"] == "True"
    assert summary_rows[0]["torch_mae_pred_l2_match_target"] == "True"
    assert summary_rows[0]["retention_tier"] == "full_anchor"
    assert summary_rows[1]["retention_tier"] == "pruned"

    manifest = json.loads((output_root / "sweep_manifest.json").read_text())
    assert manifest["probe_mask_enabled"] is True
    assert manifest["torch_mae_pred_l2_match_target"] is True

    for row in summary_rows:
        run_dir = output_root / "runs" / row["run_id"]
        cleanup_path = run_dir / "cleanup_report.json"
        assert cleanup_path.exists()
        payload = json.loads(cleanup_path.read_text())
        assert payload["retention_tier"] in {"full_anchor", "pruned"}

    full_anchor_dir = output_root / "runs" / summary_rows[0]["run_id"] / "runs" / "pinn_hybrid_resnet"
    full_anchor_root = output_root / "runs" / summary_rows[0]["run_id"]
    pruned_dir = output_root / "runs" / summary_rows[1]["run_id"] / "runs" / "pinn_hybrid_resnet"
    pruned_root = output_root / "runs" / summary_rows[1]["run_id"]
    assert (full_anchor_root / "checkpoints").exists()
    assert (full_anchor_root / "lightning_logs").exists()
    assert (full_anchor_dir / "model.pt").exists()
    assert (full_anchor_dir / "checkpoints").exists()
    assert (pruned_root / "checkpoints").exists() is False
    assert (pruned_root / "lightning_logs").exists() is False
    assert (pruned_dir / "model.pt").exists() is False
    assert (pruned_dir / "checkpoints").exists() is False
    assert (pruned_dir / "lightning_logs").exists() is False

    pruned_report = json.loads(
        (output_root / "runs" / summary_rows[1]["run_id"] / "cleanup_report.json").read_text()
    )
    assert pruned_report["bytes_reclaimed"] > 0
    assert any(path.endswith("model.pt") for path in pruned_report["deleted_paths"])
    assert any(
        path == "checkpoints" or path.startswith("checkpoints/")
        for path in pruned_report["deleted_paths"]
    )
    assert any("checkpoints" in path for path in pruned_report["deleted_paths"])


def test_stage_c_run_sweep_emits_default_stage_anchor_summary(tmp_path, monkeypatch):
    source_summary = tmp_path / "source.csv"
    _write_source_summary(
        source_summary,
        [
            {
                "summary_schema_version": "v1",
                "run_id": "stage_b_anchor",
                "stage_id": "B",
                "substage_id": "none",
                "modes": "12",
                "skip": "off",
                "width": "32",
                "amp_mae": 0.08,
                "amp_mse": 0.01,
                "phase_ssim_drop_vs_baseline": 0.0,
                "train_wall_time_sec": 100,
                "inference_time_s": 1.0,
                "model_params": 1000,
                "pareto_rank_macro": 1,
                "is_feasible": True,
                "is_stage_anchor": True,
            }
        ],
    )

    train_npz = tmp_path / "train.npz"
    test_npz = tmp_path / "test.npz"
    train_npz.touch()
    test_npz.touch()

    def _fake_runner(*, args, candidate, run_dir, train_npz, test_npz):
        _ = (args, train_npz, test_npz)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "lightning_logs" / "version_0").mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints" / "epoch=0-step=1.ckpt").write_bytes(b"r" * 128)
        (run_dir / "lightning_logs" / "version_0" / "events.out.tfevents").write_bytes(b"s" * 96)
        runtime_root = run_dir / "runs" / "pinn_hybrid_resnet"
        (runtime_root / "checkpoints").mkdir(parents=True, exist_ok=True)
        (runtime_root / "lightning_logs" / "version_0").mkdir(parents=True, exist_ok=True)
        (runtime_root / "checkpoints" / "epoch=0-step=1.ckpt").write_bytes(b"c" * 128)
        (runtime_root / "lightning_logs" / "version_0" / "events.out.tfevents").write_bytes(b"l" * 96)
        (runtime_root / "model.pt").write_bytes(b"m" * 64)
        amp_mae = 0.07 if int(candidate["downsample_schedule"]) == 2 else 0.08
        return {
            "amp_mae": amp_mae,
            "amp_mse": 0.01,
            "phase_ssim": 0.9,
            "phase_ssim_drop_vs_baseline": 0.0,
            "model_params": 1234,
            "train_wall_time_sec": 12.0,
            "inference_time_s": 1.0,
        }

    monkeypatch.setattr(sweep, "_run_candidate_with_runner", _fake_runner)

    output_root = tmp_path / "stage_c1"
    rc = sweep.main(
        [
            "--stage-id",
            "C",
            "--substage-id",
            "C1",
            "--ns",
            "128",
            "--promotion-source-summary",
            str(source_summary),
            "--dataset-profiles-n128",
            "custom_npz_pair_n128",
            "--custom-n128-train-npz",
            str(train_npz),
            "--custom-n128-test-npz",
            str(test_npz),
            "--downsample-schedule-values",
            "1,2",
            "--top-k-n256",
            "0",
            "--output-root",
            str(output_root),
        ]
    )

    assert rc == 0
    anchor_summary = output_root / "promotion" / "stage_anchor_summary.csv"
    assert anchor_summary.exists()

    summary_rows = list(csv.DictReader((output_root / "summary.csv").open()))
    summary_anchor_rows = [row for row in summary_rows if row["is_stage_anchor"].lower() == "true"]
    assert len(summary_anchor_rows) == 1
    assert summary_anchor_rows[0]["retention_tier"] == "full_anchor"

    non_anchor_rows = [row for row in summary_rows if row["is_stage_anchor"].lower() != "true"]
    assert non_anchor_rows
    for row in non_anchor_rows:
        assert row["retention_tier"] == "pruned"
        run_dir = output_root / "runs" / row["run_id"]
        runtime_root = run_dir / "runs" / "pinn_hybrid_resnet"
        assert (run_dir / "checkpoints").exists() is False
        assert (run_dir / "lightning_logs").exists() is False
        assert (runtime_root / "model.pt").exists() is False
        cleanup_payload = json.loads((run_dir / "cleanup_report.json").read_text())
        assert cleanup_payload["retention_tier"] == "pruned"
        assert cleanup_payload["bytes_reclaimed"] > 0
        assert any(path.endswith("model.pt") for path in cleanup_payload["deleted_paths"])

    anchor_rows = list(csv.DictReader(anchor_summary.open()))
    assert len(anchor_rows) == 1
    assert anchor_rows[0]["is_stage_anchor"].lower() == "true"
    assert anchor_rows[0]["run_id"] == summary_anchor_rows[0]["run_id"]


def test_manifest_hashes_include_generated_profile_npz(tmp_path, monkeypatch):
    def _fake_resolve_profile_npz_inputs(args, profile, cache):
        cached = cache.get(profile)
        if cached:
            return cached
        profile_dir = args.output_root / "datasets" / profile
        profile_dir.mkdir(parents=True, exist_ok=True)
        train_npz = profile_dir / "train.npz"
        test_npz = profile_dir / "test.npz"
        train_npz.write_bytes(b"train-bytes")
        test_npz.write_bytes(b"test-bytes")
        cache[profile] = (train_npz, test_npz)
        return cache[profile]

    def _fake_runner(*, args, candidate, run_dir, train_npz, test_npz):
        _ = (args, candidate, run_dir, train_npz, test_npz)
        return {
            "amp_mae": 0.08,
            "amp_mse": 0.01,
            "phase_ssim": 0.9,
            "phase_ssim_drop_vs_baseline": 0.0,
            "model_params": 1234,
            "train_wall_time_sec": 12.0,
            "inference_time_s": 1.0,
        }

    monkeypatch.setattr(sweep, "_resolve_profile_npz_inputs", _fake_resolve_profile_npz_inputs)
    monkeypatch.setattr(sweep, "_run_candidate_with_runner", _fake_runner)

    output_root = tmp_path / "stage_a_hashes"
    rc = sweep.main([
        "--stage-id",
        "A",
        "--ns",
        "128",
        "--dataset-profiles-n128",
        "integration_grid_lines_n128_v1",
        "--output-root",
        str(output_root),
    ])

    assert rc == 0
    manifest = json.loads((output_root / "sweep_manifest.json").read_text())
    resolved_payload = next(
        payload
        for payload in manifest["dataset_profiles"]["resolved"]
        if payload["profile"] == "integration_grid_lines_n128_v1"
    )
    train_path = output_root / "datasets" / "integration_grid_lines_n128_v1" / "train.npz"
    test_path = output_root / "datasets" / "integration_grid_lines_n128_v1" / "test.npz"
    assert resolved_payload["sha256"]["train_npz"] == sweep._compute_file_sha256(train_path)
    assert resolved_payload["sha256"]["test_npz"] == sweep._compute_file_sha256(test_path)


def test_summary_markdown_includes_stage_metadata_and_fno_blocks_column(tmp_path, monkeypatch):
    source_summary = tmp_path / "source.csv"
    _write_source_summary(
        source_summary,
        [
            {
                "summary_schema_version": "v1",
                "run_id": "anchor",
                "stage_id": "A",
                "substage_id": "none",
                "modes": "12",
                "skip": "off",
                "width": "32",
                "amp_mae": 0.08,
                "amp_mse": 0.01,
                "phase_ssim_drop_vs_baseline": 0.0,
                "train_wall_time_sec": 100,
                "inference_time_s": 1.0,
                "model_params": 1000,
                "pareto_rank_macro": 1,
                "is_feasible": True,
                "is_stage_anchor": True,
            }
        ],
    )

    train_npz = tmp_path / "train.npz"
    test_npz = tmp_path / "test.npz"
    train_npz.touch()
    test_npz.touch()

    def _fake_runner(*, args, candidate, run_dir, train_npz, test_npz):
        _ = (args, candidate, run_dir, train_npz, test_npz)
        return {
            "amp_mae": 0.08,
            "amp_mse": 0.01,
            "phase_ssim": 0.9,
            "phase_ssim_drop_vs_baseline": 0.0,
            "model_params": 1234,
            "train_wall_time_sec": 12.0,
            "inference_time_s": 1.0,
        }

    monkeypatch.setattr(sweep, "_run_candidate_with_runner", _fake_runner)

    output_root = tmp_path / "stage_b"
    rc = sweep.main([
        "--stage-id",
        "B",
        "--ns",
        "128",
        "--promotion-source-summary",
        str(source_summary),
        "--dataset-profiles-n128",
        "custom_npz_pair_n128",
        "--custom-n128-train-npz",
        str(train_npz),
        "--custom-n128-test-npz",
        str(test_npz),
        "--modes",
        "12",
        "--skip-values",
        "off",
        "--widths",
        "32",
        "--fno-blocks-values",
        "4,6",
        "--output-root",
        str(output_root),
    ])

    assert rc == 0
    summary_md = (output_root / "summary.md").read_text()
    assert "- stage_id: B" in summary_md
    assert "- substage_id: none" in summary_md
    assert "| fno_blocks |" in summary_md


def test_stage_id_aggregation_mode_parse_without_ns_or_output_root(tmp_path):
    source_summary = tmp_path / "source.csv"
    _write_source_summary(
        source_summary,
        [
            {
                "summary_schema_version": "v1",
                "run_id": "r1",
                "stage_id": "A",
                "substage_id": "none",
                "modes": "12",
                "skip": "off",
                "width": "32",
                "amp_mae": 0.08,
                "amp_mse": 0.01,
                "phase_ssim_drop_vs_baseline": 0.0,
                "train_wall_time_sec": 100,
                "inference_time_s": 1.0,
                "model_params": 1000,
                "pareto_rank_macro": 1,
                "is_feasible": True,
                "is_stage_anchor": True,
            }
        ],
    )

    args = sweep.parse_args([
        "--stage-id",
        "A",
        "--aggregate-seed-rerank-root",
        str(tmp_path / "seed_rerank"),
        "--source-summary",
        str(source_summary),
        "--emit-stage-anchor-summary",
        str(tmp_path / "promotion" / "stage_anchor_summary.csv"),
        "--emit-robust-promotion-summary",
        str(tmp_path / "promotion" / "summary_seed_robust.csv"),
    ])

    assert args.aggregate_seed_rerank_root == tmp_path / "seed_rerank"


def test_guardrail_seed_rerank_requires_complete_boundary_seed_coverage(tmp_path, capsys):
    source_summary = tmp_path / "source.csv"
    _write_source_summary(
        source_summary,
        [
            {
                "summary_schema_version": "v1",
                "run_id": "c1",
                "stage_id": "A",
                "substage_id": "none",
                "modes": "12",
                "skip": "off",
                "width": "32",
                "amp_mae": 0.08,
                "amp_mse": 0.01,
                "phase_ssim_drop_vs_baseline": 0.01,
                "train_wall_time_sec": 100,
                "inference_time_s": 2.0,
                "model_params": 1000,
                "pareto_rank_macro": 1,
                "is_feasible": True,
                "is_stage_anchor": True,
            },
            {
                "summary_schema_version": "v1",
                "run_id": "c2",
                "stage_id": "A",
                "substage_id": "none",
                "modes": "16",
                "skip": "on",
                "width": "32",
                "amp_mae": 0.09,
                "amp_mse": 0.011,
                "phase_ssim_drop_vs_baseline": 0.01,
                "train_wall_time_sec": 105,
                "inference_time_s": 2.1,
                "model_params": 1000,
                "pareto_rank_macro": 2,
                "is_feasible": True,
                "is_stage_anchor": False,
            },
            {
                "summary_schema_version": "v1",
                "run_id": "c3",
                "stage_id": "A",
                "substage_id": "none",
                "modes": "24",
                "skip": "off",
                "width": "48",
                "amp_mae": 0.10,
                "amp_mse": 0.012,
                "phase_ssim_drop_vs_baseline": 0.01,
                "train_wall_time_sec": 110,
                "inference_time_s": 2.2,
                "model_params": 1000,
                "pareto_rank_macro": 3,
                "is_feasible": True,
                "is_stage_anchor": False,
            },
        ],
    )

    rerank_root = tmp_path / "seed_rerank"
    _write_seed_summary(
        rerank_root / "c1_seed11" / "summary.csv",
        [
            {
                "summary_schema_version": "v1",
                "run_id": "c1",
                "seed": 11,
                "modes": "12",
                "skip": "off",
                "width": "32",
                "amp_mae": 0.081,
                "amp_mse": 0.011,
                "train_wall_time_sec": 101,
                "inference_time_s": 2.0,
                "model_params": 1000,
                "phase_ssim_drop_vs_baseline": 0.01,
                "is_feasible": True,
            }
        ],
    )

    rc = sweep.main([
        "--stage-id",
        "A",
        "--aggregate-seed-rerank-root",
        str(rerank_root),
        "--source-summary",
        str(source_summary),
        "--emit-stage-anchor-summary",
        str(tmp_path / "promotion" / "stage_anchor_summary.csv"),
        "--emit-robust-promotion-summary",
        str(tmp_path / "promotion" / "summary_seed_robust.csv"),
        "--output-root",
        str(tmp_path / "unused"),
    ])

    assert rc == 1
    assert "missing required seeds" in capsys.readouterr().err.lower()


def test_guardrail_seed_rerank_writes_median_pareto_summary_and_anchor(tmp_path):
    source_summary = tmp_path / "source.csv"
    _write_source_summary(
        source_summary,
        [
            {
                "summary_schema_version": "v1",
                "run_id": "c1",
                "stage_id": "A",
                "substage_id": "none",
                "modes": "12",
                "skip": "off",
                "width": "32",
                "amp_mae": 0.080,
                "amp_mse": 0.010,
                "phase_ssim_drop_vs_baseline": 0.01,
                "train_wall_time_sec": 100,
                "inference_time_s": 2.0,
                "model_params": 1000,
                "pareto_rank_macro": 1,
                "is_feasible": True,
                "is_stage_anchor": True,
            },
            {
                "summary_schema_version": "v1",
                "run_id": "c2",
                "stage_id": "A",
                "substage_id": "none",
                "modes": "16",
                "skip": "on",
                "width": "32",
                "amp_mae": 0.090,
                "amp_mse": 0.011,
                "phase_ssim_drop_vs_baseline": 0.01,
                "train_wall_time_sec": 110,
                "inference_time_s": 2.2,
                "model_params": 1000,
                "pareto_rank_macro": 2,
                "is_feasible": True,
                "is_stage_anchor": False,
            },
            {
                "summary_schema_version": "v1",
                "run_id": "c3",
                "stage_id": "A",
                "substage_id": "none",
                "modes": "24",
                "skip": "off",
                "width": "48",
                "amp_mae": 0.100,
                "amp_mse": 0.012,
                "phase_ssim_drop_vs_baseline": 0.01,
                "train_wall_time_sec": 120,
                "inference_time_s": 2.4,
                "model_params": 1000,
                "pareto_rank_macro": 3,
                "is_feasible": True,
                "is_stage_anchor": False,
            },
        ],
    )

    rerank_root = tmp_path / "seed_rerank"
    # Make c2 the best robust candidate so anchor selection cannot rely on robust_rows[0].
    for run_id, seed3_amp, seed11_amp, seed17_amp in [
        ("c1", 0.090, 0.091, 0.092),
        ("c2", 0.080, 0.079, 0.081),
        ("c3", 0.100, 0.101, 0.099),
    ]:
        _write_seed_summary(
            rerank_root / f"{run_id}_seed11" / "summary.csv",
            [
                {
                    "summary_schema_version": "v1",
                    "run_id": run_id,
                    "seed": 3,
                    "modes": "12",
                    "skip": "off",
                    "width": "32",
                    "amp_mae": seed3_amp,
                    "amp_mse": 0.01,
                    "train_wall_time_sec": 100,
                    "inference_time_s": 2.0,
                    "model_params": 1000,
                    "phase_ssim_drop_vs_baseline": 0.01,
                    "is_feasible": True,
                },
                {
                    "summary_schema_version": "v1",
                    "run_id": run_id,
                    "seed": 11,
                    "modes": "12",
                    "skip": "off",
                    "width": "32",
                    "amp_mae": seed11_amp,
                    "amp_mse": 0.01,
                    "train_wall_time_sec": 100,
                    "inference_time_s": 2.0,
                    "model_params": 1000,
                    "phase_ssim_drop_vs_baseline": 0.01,
                    "is_feasible": True,
                },
                {
                    "summary_schema_version": "v1",
                    "run_id": run_id,
                    "seed": 17,
                    "modes": "12",
                    "skip": "off",
                    "width": "32",
                    "amp_mae": seed17_amp,
                    "amp_mse": 0.01,
                    "train_wall_time_sec": 100,
                    "inference_time_s": 2.0,
                    "model_params": 1000,
                    "phase_ssim_drop_vs_baseline": 0.01,
                    "is_feasible": True,
                },
            ],
        )

    robust_path = tmp_path / "promotion" / "summary_seed_robust.csv"
    anchor_path = tmp_path / "promotion" / "stage_anchor_summary.csv"
    rc = sweep.main([
        "--stage-id",
        "A",
        "--aggregate-seed-rerank-root",
        str(rerank_root),
        "--source-summary",
        str(source_summary),
        "--emit-stage-anchor-summary",
        str(anchor_path),
        "--emit-robust-promotion-summary",
        str(robust_path),
        "--top-k-n256",
        "2",
        "--output-root",
        str(tmp_path / "unused"),
    ])

    assert rc == 0
    robust_rows = list(csv.DictReader(robust_path.open()))
    assert len(robust_rows) == 3
    assert "pareto_rank_median" in robust_rows[0]
    assert "pareto_rank_seed11" in robust_rows[0]

    anchor_rows = list(csv.DictReader(anchor_path.open()))
    assert len(anchor_rows) == 1
    assert anchor_rows[0]["is_stage_anchor"].lower() == "true"
    # Stage-A control anchor must stay at true defaults for downstream N=128 stage isolation.
    assert anchor_rows[0]["modes"] == "12"
    assert anchor_rows[0]["skip"] == "off"
    assert anchor_rows[0]["width"] == "32"
    assert anchor_rows[0]["fno_blocks"] == "4"
    assert anchor_rows[0]["downsample_schedule"] == "2"
    assert anchor_rows[0]["downsample_op"] == "stride_conv"
    assert anchor_rows[0]["encoder_conv_hidden"] == "none"
    assert anchor_rows[0]["encoder_spectral_hidden"] == "none"
    assert anchor_rows[0]["max_hidden"] == "none"
    assert anchor_rows[0]["resnet_width"] == "none"
    assert anchor_rows[0]["resnet_blocks"] == "6"
    assert anchor_rows[0]["skip_style"] == "add"


def test_profile_resolution_cameraman_uses_train_test_data_args(tmp_path):
    args = sweep.parse_args(
        [
            "--ns",
            "256",
            "--output-root",
            str(tmp_path / "out"),
            "--cameraman-dp",
            str(tmp_path / "cameraman_dp.hdf5"),
            "--cameraman-para",
            str(tmp_path / "cameraman_para.hdf5"),
        ]
    )
    payload = sweep._profile_resolution(args, "cameraman256_halfsplit_v1")
    assert payload["emitted_dataset_args"] == ["--train-data", "--test-data"]


def test_resolve_profile_npz_inputs_cameraman_builds_canonical_cached_npz(tmp_path, monkeypatch):
    output_root = tmp_path / "out"
    args = sweep.parse_args(
        [
            "--ns",
            "256",
            "--output-root",
            str(output_root),
            "--cameraman-dp",
            str(tmp_path / "cameraman_dp.hdf5"),
            "--cameraman-para",
            str(tmp_path / "cameraman_para.hdf5"),
        ]
    )

    profile_dir = output_root / "datasets" / "cameraman256_halfsplit_v1"
    raw_train = profile_dir / "raw_train.npz"
    raw_test = profile_dir / "raw_test.npz"
    canonical_train = profile_dir / "canonical_cache" / "datasets" / "N256" / "gs1" / "train.npz"
    canonical_test = profile_dir / "canonical_cache" / "datasets" / "N256" / "gs1" / "test.npz"

    def _fake_prepare_hybrid_dataset(**kwargs):
        _ = kwargs
        profile_dir.mkdir(parents=True, exist_ok=True)
        raw_train.write_bytes(b"raw-train")
        raw_test.write_bytes(b"raw-test")
        return {
            "train_npz": str(raw_train),
            "test_npz": str(raw_test),
        }

    def _fake_build_datasets(
        *,
        dataset_source,
        cfg,
        required_ns,
        train_data,
        test_data,
        n_groups=None,
        n_subsample=None,
        neighbor_count=7,
        subsample_seed=None,
    ):
        _ = (cfg, required_ns, n_groups, n_subsample, neighbor_count, subsample_seed)
        assert dataset_source == "external_raw_npz"
        assert Path(train_data) == raw_train
        assert Path(test_data) == raw_test
        canonical_train.parent.mkdir(parents=True, exist_ok=True)
        canonical_train.write_bytes(b"canonical-train")
        canonical_test.write_bytes(b"canonical-test")
        return {
            256: {
                "train_npz": str(canonical_train),
                "test_npz": str(canonical_test),
                "gt_recon": str(profile_dir / "canonical_cache" / "recons" / "gt" / "recon.npz"),
                "tag": "N256",
            }
        }

    fake_prep_module = types.SimpleNamespace(prepare_hybrid_dataset=_fake_prepare_hybrid_dataset)
    fake_builder_module = types.SimpleNamespace(build_datasets=_fake_build_datasets)
    fake_grid_workflow_module = types.SimpleNamespace(
        GridLinesConfig=lambda **kwargs: types.SimpleNamespace(**kwargs)
    )
    monkeypatch.setitem(sys.modules, "scripts.studies.prepare_nersc_hybrid_dataset", fake_prep_module)
    monkeypatch.setitem(sys.modules, "scripts.studies.grid_study_dataset_builder", fake_builder_module)
    monkeypatch.setitem(sys.modules, "ptycho.workflows.grid_lines_workflow", fake_grid_workflow_module)

    cache: dict[str, tuple[Path, Path]] = {}
    train_npz, test_npz = sweep._resolve_profile_npz_inputs(args, "cameraman256_halfsplit_v1", cache)

    assert train_npz == canonical_train
    assert test_npz == canonical_test
    assert cache["cameraman256_halfsplit_v1"] == (canonical_train, canonical_test)


def test_n256_promotion_deduplicates_rows_by_config(tmp_path, monkeypatch):
    source_path = tmp_path / "promotion_source.csv"
    fieldnames = [
        "summary_schema_version",
        "run_id",
        "stage_id",
        "substage_id",
        "modes",
        "skip",
        "width",
        "fno_blocks",
        "downsample_schedule",
        "downsample_op",
        "encoder_conv_hidden",
        "encoder_spectral_hidden",
        "max_hidden",
        "resnet_width",
        "resnet_blocks",
        "skip_style",
        "amp_mae",
        "amp_mse",
        "train_wall_time_sec",
        "inference_time_s",
        "model_params",
        "phase_ssim_drop_vs_baseline",
        "pareto_rank_seed3",
        "pareto_rank_seed11",
        "pareto_rank_seed17",
        "pareto_rank_median",
        "pareto_rank_macro",
        "is_feasible",
        "is_stage_anchor",
    ]
    _write_csv(
        source_path,
        fieldnames,
        [
            {
                "summary_schema_version": "v1",
                "run_id": "dup_config_row_1",
                "stage_id": "A",
                "substage_id": "none",
                "modes": 12,
                "skip": "off",
                "width": 32,
                "fno_blocks": 4,
                "downsample_schedule": 2,
                "downsample_op": "stride_conv",
                "encoder_conv_hidden": "none",
                "encoder_spectral_hidden": "none",
                "max_hidden": "none",
                "resnet_width": "none",
                "resnet_blocks": 6,
                "skip_style": "add",
                "amp_mae": 0.08,
                "amp_mse": 0.01,
                "train_wall_time_sec": 100,
                "inference_time_s": 2.0,
                "model_params": 1000,
                "phase_ssim_drop_vs_baseline": 0.0,
                "pareto_rank_seed3": 1,
                "pareto_rank_seed11": 1,
                "pareto_rank_seed17": 1,
                "pareto_rank_median": 1,
                "pareto_rank_macro": 1,
                "is_feasible": True,
                "is_stage_anchor": False,
            },
            {
                "summary_schema_version": "v1",
                "run_id": "dup_config_row_2",
                "stage_id": "A",
                "substage_id": "none",
                "modes": 12,
                "skip": "off",
                "width": 32,
                "fno_blocks": 4,
                "downsample_schedule": 2,
                "downsample_op": "stride_conv",
                "encoder_conv_hidden": "none",
                "encoder_spectral_hidden": "none",
                "max_hidden": "none",
                "resnet_width": "none",
                "resnet_blocks": 6,
                "skip_style": "add",
                "amp_mae": 0.081,
                "amp_mse": 0.011,
                "train_wall_time_sec": 101,
                "inference_time_s": 2.1,
                "model_params": 1001,
                "phase_ssim_drop_vs_baseline": 0.0,
                "pareto_rank_seed3": 2,
                "pareto_rank_seed11": 2,
                "pareto_rank_seed17": 2,
                "pareto_rank_median": 2,
                "pareto_rank_macro": 2,
                "is_feasible": True,
                "is_stage_anchor": False,
            },
            {
                "summary_schema_version": "v1",
                "run_id": "unique_config",
                "stage_id": "A",
                "substage_id": "none",
                "modes": 16,
                "skip": "on",
                "width": 32,
                "fno_blocks": 4,
                "downsample_schedule": 2,
                "downsample_op": "stride_conv",
                "encoder_conv_hidden": "none",
                "encoder_spectral_hidden": "none",
                "max_hidden": "none",
                "resnet_width": "none",
                "resnet_blocks": 6,
                "skip_style": "add",
                "amp_mae": 0.082,
                "amp_mse": 0.012,
                "train_wall_time_sec": 102,
                "inference_time_s": 2.2,
                "model_params": 1002,
                "phase_ssim_drop_vs_baseline": 0.0,
                "pareto_rank_seed3": 3,
                "pareto_rank_seed11": 3,
                "pareto_rank_seed17": 3,
                "pareto_rank_median": 3,
                "pareto_rank_macro": 3,
                "is_feasible": True,
                "is_stage_anchor": False,
            },
        ],
    )

    def _fake_runner(*, args, candidate, run_dir, train_npz, test_npz):
        _ = (args, run_dir, train_npz, test_npz)
        return {
            "amp_mae": 0.09 + (0.001 * int(candidate["modes"])),
            "amp_mse": 0.01,
            "phase_ssim": 0.9,
            "phase_ssim_drop_vs_baseline": 0.0,
            "model_params": 1111,
            "train_wall_time_sec": 22.0,
            "inference_time_s": 1.2,
        }

    monkeypatch.setattr(sweep, "_run_candidate_with_runner", _fake_runner)

    rc = sweep.main(
        [
            "--stage-id",
            "A",
            "--ns",
            "256",
            "--promotion-source-summary",
            str(source_path),
            "--top-k-n256",
            "2",
            "--dataset-profiles-n256",
            "custom_npz_pair_n256",
            "--custom-n256-train-npz",
            str(tmp_path / "train.npz"),
            "--custom-n256-test-npz",
            str(tmp_path / "test.npz"),
            "--output-root",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 0

    rows = list(csv.DictReader((tmp_path / "out" / "summary.csv").open()))
    assert len(rows) == 2
    unique_configs = {
        (row["modes"], row["skip"], row["width"], row["fno_blocks"]) for row in rows
    }
    assert len(unique_configs) == 2


def test_seed_rerank_aggregation_deduplicates_boundary_configs(tmp_path):
    source_summary = tmp_path / "source.csv"
    _write_source_summary(
        source_summary,
        [
            {
                "summary_schema_version": "v1",
                "run_id": "dup_row_1",
                "stage_id": "A",
                "substage_id": "none",
                "modes": "12",
                "skip": "off",
                "width": "32",
                "amp_mae": 0.08,
                "amp_mse": 0.01,
                "phase_ssim_drop_vs_baseline": 0.0,
                "train_wall_time_sec": 100,
                "inference_time_s": 2.0,
                "model_params": 1000,
                "pareto_rank_macro": 1,
                "is_feasible": True,
                "is_stage_anchor": False,
            },
            {
                "summary_schema_version": "v1",
                "run_id": "dup_row_2",
                "stage_id": "A",
                "substage_id": "none",
                "modes": "12",
                "skip": "off",
                "width": "32",
                "amp_mae": 0.081,
                "amp_mse": 0.011,
                "phase_ssim_drop_vs_baseline": 0.0,
                "train_wall_time_sec": 101,
                "inference_time_s": 2.1,
                "model_params": 1000,
                "pareto_rank_macro": 2,
                "is_feasible": True,
                "is_stage_anchor": False,
            },
            {
                "summary_schema_version": "v1",
                "run_id": "unique_row",
                "stage_id": "A",
                "substage_id": "none",
                "modes": "16",
                "skip": "on",
                "width": "32",
                "amp_mae": 0.09,
                "amp_mse": 0.012,
                "phase_ssim_drop_vs_baseline": 0.0,
                "train_wall_time_sec": 102,
                "inference_time_s": 2.2,
                "model_params": 1000,
                "pareto_rank_macro": 3,
                "is_feasible": True,
                "is_stage_anchor": False,
            },
        ],
    )

    rerank_root = tmp_path / "seed_rerank"
    for run_id in ("dup_row_1", "unique_row"):
        _write_seed_summary(
            rerank_root / f"{run_id}_seed11" / "summary.csv",
            [
                {
                    "summary_schema_version": "v1",
                    "run_id": run_id,
                    "seed": 11,
                    "modes": "12",
                    "skip": "off",
                    "width": "32",
                    "amp_mae": 0.082,
                    "amp_mse": 0.011,
                    "train_wall_time_sec": 100,
                    "inference_time_s": 2.0,
                    "model_params": 1000,
                    "phase_ssim_drop_vs_baseline": 0.0,
                    "is_feasible": True,
                },
                {
                    "summary_schema_version": "v1",
                    "run_id": run_id,
                    "seed": 17,
                    "modes": "12",
                    "skip": "off",
                    "width": "32",
                    "amp_mae": 0.083,
                    "amp_mse": 0.011,
                    "train_wall_time_sec": 100,
                    "inference_time_s": 2.0,
                    "model_params": 1000,
                    "phase_ssim_drop_vs_baseline": 0.0,
                    "is_feasible": True,
                },
            ],
        )

    robust_path = tmp_path / "promotion" / "summary_seed_robust.csv"
    anchor_path = tmp_path / "promotion" / "stage_anchor_summary.csv"
    rc = sweep.main(
        [
            "--stage-id",
            "A",
            "--aggregate-seed-rerank-root",
            str(rerank_root),
            "--source-summary",
            str(source_summary),
            "--emit-stage-anchor-summary",
            str(anchor_path),
            "--emit-robust-promotion-summary",
            str(robust_path),
            "--top-k-n256",
            "2",
            "--output-root",
            str(tmp_path / "unused"),
        ]
    )
    assert rc == 0
    robust_rows = list(csv.DictReader(robust_path.open()))
    assert len(robust_rows) == 2
    unique_configs = {(row["modes"], row["skip"], row["width"]) for row in robust_rows}
    assert len(unique_configs) == 2


def test_run_candidate_with_runner_forwards_structural_knobs(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    train_npz = tmp_path / "train.npz"
    test_npz = tmp_path / "test.npz"
    train_npz.touch()
    test_npz.touch()

    args = argparse.Namespace(
        ns=128,
        epochs_n128=5,
        epochs_n256=9,
        seed=7,
        probe_mask=True,
        torch_mae_pred_l2_match_target=True,
    )
    candidate = {
        "run_key": "candidate-1",
        "modes": 16,
        "width": 48,
        "fno_blocks": 5,
        "skip": "on",
        "downsample_schedule": 1,
        "downsample_op": "avgpool_conv",
        "encoder_conv_hidden": "48",
        "encoder_spectral_hidden": "64",
        "resnet_width": "256",
        "resnet_blocks": 8,
        "skip_style": "gated_add",
    }
    captured: dict[str, object] = {}

    def _fake_run(cmd, check, stdout, stderr, text):
        _ = (check, stderr, text)
        captured["cmd"] = cmd
        metrics_path = run_dir / "runs" / "pinn_hybrid_resnet" / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps({"mae": [0.1, 0.2], "mse": [0.01, 0.02], "ssim": [0.9, 0.8]}))
        stdout.write('{"inference_time_s": 1.5, "model_params": 4321}\n')
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(sweep.subprocess, "run", _fake_run)

    sweep._run_candidate_with_runner(
        args=args,
        candidate=candidate,
        run_dir=run_dir,
        train_npz=train_npz,
        test_npz=test_npz,
    )

    cmd = captured["cmd"]
    assert "--hybrid-skip-connections" in cmd
    assert "--hybrid-downsample-steps" in cmd
    assert cmd[cmd.index("--hybrid-downsample-steps") + 1] == "1"
    assert "--hybrid-downsample-op" in cmd
    assert cmd[cmd.index("--hybrid-downsample-op") + 1] == "avgpool_conv"
    assert "--hybrid-encoder-conv-hidden" in cmd
    assert cmd[cmd.index("--hybrid-encoder-conv-hidden") + 1] == "48"
    assert "--hybrid-encoder-spectral-hidden" in cmd
    assert cmd[cmd.index("--hybrid-encoder-spectral-hidden") + 1] == "64"
    assert "--hybrid-resnet-blocks" in cmd
    assert cmd[cmd.index("--hybrid-resnet-blocks") + 1] == "8"
    assert "--hybrid-skip-style" in cmd
    assert cmd[cmd.index("--hybrid-skip-style") + 1] == "gated_add"
    assert "--torch-resnet-width" in cmd
    assert cmd[cmd.index("--torch-resnet-width") + 1] == "256"


def test_run_candidate_with_runner_omits_optional_hidden_widths_when_none(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    train_npz = tmp_path / "train.npz"
    test_npz = tmp_path / "test.npz"
    train_npz.touch()
    test_npz.touch()

    args = argparse.Namespace(
        ns=256,
        epochs_n128=5,
        epochs_n256=9,
        seed=11,
        probe_mask=False,
        torch_mae_pred_l2_match_target=False,
    )
    candidate = {
        "run_key": "candidate-2",
        "modes": 12,
        "width": 32,
        "fno_blocks": 4,
        "skip": "off",
        "downsample_schedule": 2,
        "downsample_op": "stride_conv",
        "encoder_conv_hidden": "none",
        "encoder_spectral_hidden": "none",
        "resnet_width": "none",
        "resnet_blocks": 6,
        "skip_style": "add",
    }
    captured: dict[str, object] = {}

    def _fake_run(cmd, check, stdout, stderr, text):
        _ = (check, stderr, text)
        captured["cmd"] = cmd
        metrics_path = run_dir / "runs" / "pinn_hybrid_resnet" / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps({"mae": [0.1, 0.2], "mse": [0.01, 0.02], "ssim": [0.9, 0.8]}))
        stdout.write('{"inference_time_s": 1.5, "model_params": 4321}\n')
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(sweep.subprocess, "run", _fake_run)

    sweep._run_candidate_with_runner(
        args=args,
        candidate=candidate,
        run_dir=run_dir,
        train_npz=train_npz,
        test_npz=test_npz,
    )

    cmd = captured["cmd"]
    assert "--no-hybrid-skip-connections" in cmd
    assert "--hybrid-downsample-steps" in cmd
    assert "--hybrid-downsample-op" in cmd
    assert "--hybrid-resnet-blocks" in cmd
    assert "--hybrid-skip-style" in cmd
    assert "--hybrid-encoder-conv-hidden" not in cmd
    assert "--hybrid-encoder-spectral-hidden" not in cmd
    assert "--torch-resnet-width" not in cmd
    assert "--no-probe-mask" in cmd
    assert "--no-torch-mae-pred-l2-match-target" in cmd
