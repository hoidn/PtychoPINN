import csv
import json
from pathlib import Path

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


def test_invocation_and_cleanup_artifacts_are_written(tmp_path, monkeypatch):
    train_npz = tmp_path / "train.npz"
    test_npz = tmp_path / "test.npz"

    def _fake_runner(*, args, candidate, run_dir, train_npz, test_npz):
        _ = (args, run_dir, train_npz, test_npz)
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
        "--output-root",
        str(output_root),
    ])
    assert rc == 0

    assert (output_root / "invocation.json").exists()
    assert (output_root / "invocation.sh").exists()

    summary_rows = list(csv.DictReader((output_root / "summary.csv").open()))
    assert len(summary_rows) == 2
    assert summary_rows[0]["retention_tier"] == "full_anchor"
    assert summary_rows[1]["retention_tier"] == "pruned"

    for row in summary_rows:
        run_dir = output_root / "runs" / row["run_id"]
        cleanup_path = run_dir / "cleanup_report.json"
        assert cleanup_path.exists()
        payload = json.loads(cleanup_path.read_text())
        assert payload["retention_tier"] in {"full_anchor", "pruned"}


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
    for run_id, seed3_amp, seed11_amp, seed17_amp in [
        ("c1", 0.080, 0.081, 0.079),
        ("c2", 0.090, 0.088, 0.091),
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
