import json
import subprocess
import types
from pathlib import Path

import numpy as np
import pytest


def _import_study():
    from scripts.studies import ood_fig5_metrics as study

    return study


def _write_minimal_default_inputs(repo_root: Path) -> None:
    study = _import_study()
    for row in study.build_default_rows(repo_root):
        row.reconstruction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            row.reconstruction_path,
            reconstructed_object=np.ones((68, 68), dtype=np.complex64),
        )
        if row.panel_reference_path is not None:
            row.panel_reference_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                row.panel_reference_path,
                ground_truth_complex=np.ones((68, 68), dtype=np.complex64),
            )
    coordinate_source = repo_root / "datasets/Run1084_recon3_postPC_shrunk_3.npz"
    coordinate_source.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        coordinate_source,
        xcoords=np.array([10, 70], dtype=np.float32),
        ycoords=np.array([10, 70], dtype=np.float32),
        objectGuess=np.ones((80, 80), dtype=np.complex64),
    )
    full_reference = repo_root / "experiment_outputs/ground_truth/run1084_ground_truth.npz"
    full_reference.parent.mkdir(parents=True, exist_ok=True)
    np.savez(full_reference, ground_truth_complex=np.ones((80, 80), dtype=np.complex64))


def test_default_rows_are_pinned_to_fig5_recon_on_run1084_paths():
    study = _import_study()

    rows = study.build_default_rows(Path("/repo"))
    by_id = {row.row_id: row for row in rows}

    assert list(by_id) == [
        "id_ptychopinn",
        "id_supervised_baseline",
        "ood_ptychopinn",
        "ood_supervised_baseline",
    ]
    assert by_id["id_ptychopinn"].reconstruction_path == Path(
        "/repo/experiment_outputs/run1084_trained_models/recon_on_run1084_pinn/reconstruction.npz"
    )
    assert by_id["id_supervised_baseline"].reconstruction_path == Path(
        "/repo/experiment_outputs/run1084_trained_models/recon_on_run1084_baseline/baseline_reconstruction.npz"
    )
    assert by_id["ood_ptychopinn"].reconstruction_path == Path(
        "/repo/experiment_outputs/fly64_trained_models/recon_on_run1084_pinn/reconstruction.npz"
    )
    assert by_id["ood_supervised_baseline"].reconstruction_path == Path(
        "/repo/experiment_outputs/fly64_trained_models/recon_on_run1084_baseline/baseline_reconstruction.npz"
    )
    assert by_id["id_supervised_baseline"].panel_reference_path is None
    assert "ground_truth_run1084_for_fly64trained.npz" in str(
        by_id["ood_ptychopinn"].panel_reference_path
    )


def test_npz_inventory_records_size_mtime_sha256_keys_shapes_and_dtypes(tmp_path):
    study = _import_study()
    path = tmp_path / "sample.npz"
    np.savez(
        path,
        complex_value=np.ones((2, 3), dtype=np.complex64),
        metadata=np.array({"source": "fixture"}, dtype=object),
    )

    inventory = study.inventory_npz(path)

    assert inventory["path"] == str(path)
    assert inventory["exists"] is True
    assert inventory["size_bytes"] == path.stat().st_size
    assert inventory["mtime_ns"] == path.stat().st_mtime_ns
    assert inventory["sha256"] == study.sha256_file(path)
    assert inventory["keys"] == ["complex_value", "metadata"]
    assert inventory["arrays"]["complex_value"]["shape"] == [2, 3]
    assert inventory["arrays"]["complex_value"]["dtype"] == "complex64"
    assert inventory["arrays"]["metadata"]["dtype"] == "object"


def test_load_complex_reconstruction_prefers_reconstructed_object(tmp_path):
    study = _import_study()
    path = tmp_path / "reconstruction.npz"
    preferred = np.full((3, 4), 2.0 + 1.0j, dtype=np.complex64)
    fallback_amp = np.full((3, 4), 9.0, dtype=np.float32)
    fallback_phase = np.full((3, 4), 0.0, dtype=np.float32)
    np.savez(
        path,
        reconstructed_object=preferred,
        reconstructed_amplitude=fallback_amp,
        reconstructed_phase=fallback_phase,
    )

    loaded = study.load_complex_array(
        path,
        complex_keys=("reconstructed_object",),
        amp_key="reconstructed_amplitude",
        phase_key="reconstructed_phase",
    )

    np.testing.assert_allclose(loaded, preferred)


def test_load_complex_reconstruction_accepts_singleton_final_dimension(tmp_path):
    study = _import_study()
    path = tmp_path / "baseline_reconstruction.npz"
    expected = np.full((3, 4, 1), 1.0 + 2.0j, dtype=np.complex64)
    np.savez(path, reconstructed_object=expected)

    loaded = study.load_complex_array(
        path,
        complex_keys=("reconstructed_object",),
        amp_key="reconstructed_amplitude",
        phase_key="reconstructed_phase",
    )

    assert loaded.shape == (3, 4)
    np.testing.assert_allclose(loaded, expected[:, :, 0])


def test_load_complex_reconstruction_rejects_ambiguous_3d_shape(tmp_path):
    study = _import_study()
    path = tmp_path / "bad_reconstruction.npz"
    np.savez(path, reconstructed_object=np.ones((3, 4, 2), dtype=np.complex64))

    with pytest.raises(ValueError, match="2D complex reconstruction"):
        study.load_complex_array(
            path,
            complex_keys=("reconstructed_object",),
            amp_key="reconstructed_amplitude",
            phase_key="reconstructed_phase",
        )


def test_coordinate_source_requires_matching_xcoords_ycoords(tmp_path):
    study = _import_study()
    path = tmp_path / "coords.npz"
    np.savez(
        path,
        xcoords=np.array([10, 11, 12], dtype=np.float32),
        ycoords=np.array([20, 21], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="xcoords.*ycoords"):
        study.load_scan_coords_yx(path)


def test_full_reference_accepts_ground_truth_complex_or_object_guess(tmp_path):
    study = _import_study()
    primary = tmp_path / "ground_truth.npz"
    coordinate_source = tmp_path / "coords.npz"
    ground_truth = np.full((5, 6), 3.0 + 4.0j, dtype=np.complex64)
    object_guess = np.full((5, 6), 1.0 + 1.0j, dtype=np.complex64)
    np.savez(primary, ground_truth_complex=ground_truth)
    np.savez(coordinate_source, objectGuess=object_guess, xcoords=[1], ycoords=[2])

    loaded, provenance = study.load_full_reference(primary, coordinate_source)

    np.testing.assert_allclose(loaded, ground_truth)
    assert provenance["source_path"] == str(primary)
    assert provenance["source_key"] == "ground_truth_complex"

    primary.unlink()
    loaded, provenance = study.load_full_reference(primary, coordinate_source)

    np.testing.assert_allclose(loaded, object_guess)
    assert provenance["source_path"] == str(coordinate_source)
    assert provenance["source_key"] == "objectGuess"


def test_missing_or_nonfinite_arrays_are_rejected_before_metrics(tmp_path):
    study = _import_study()
    missing = tmp_path / "missing.npz"
    np.savez(missing, reconstructed_amplitude=np.ones((2, 2), dtype=np.float32))

    with pytest.raises(KeyError, match="reconstructed_object"):
        study.load_complex_array(
            missing,
            complex_keys=("reconstructed_object",),
            amp_key="reconstructed_amplitude",
            phase_key="reconstructed_phase",
        )

    nonfinite = tmp_path / "nonfinite.npz"
    bad = np.ones((2, 2), dtype=np.complex64)
    bad[0, 0] = np.nan + 0j
    np.savez(nonfinite, reconstructed_object=bad)

    with pytest.raises(ValueError, match="non-finite"):
        study.load_complex_array(
            nonfinite,
            complex_keys=("reconstructed_object",),
            amp_key="reconstructed_amplitude",
            phase_key="reconstructed_phase",
        )


def test_metrics_table_bolds_best_values_within_each_condition():
    study = _import_study()

    table = study.render_metrics_table(
        [
            {
                "condition": "ID",
                "model": "PtychoPINN",
                "amplitude_mse": 0.01504,
                "amplitude_mse_unscaled": 0.015008098945888785,
                "amplitude_psnr": 66.358,
                "amplitude_psnr_unscaled": 17.53946844649455,
                "amplitude_ssim": 0.4955,
                "phase_mse": 0.1553,
                "phase_psnr": 56.219,
                "phase_ssim": 0.7687,
            },
            {
                "condition": "ID",
                "model": "Supervised baseline",
                "amplitude_mse": 0.01246,
                "amplitude_mse_unscaled": 0.013047583730996593,
                "amplitude_psnr": 67.176,
                "amplitude_psnr_unscaled": 18.147424363658715,
                "amplitude_ssim": 0.538,
                "phase_mse": 0.07434,
                "phase_psnr": 59.418,
                "phase_ssim": 0.8119,
            },
            {
                "condition": "OOD",
                "model": "PtychoPINN",
                "amplitude_mse": 0.02412,
                "amplitude_mse_unscaled": 0.03223916468129783,
                "amplitude_psnr": 64.307,
                "amplitude_psnr_unscaled": 14.218887482422423,
                "amplitude_ssim": 0.1757,
                "phase_mse": 0.295,
                "phase_psnr": 53.432,
                "phase_ssim": 0.5137,
            },
            {
                "condition": "OOD",
                "model": "Supervised baseline",
                "amplitude_mse": 0.02182,
                "amplitude_mse_unscaled": 0.24756047207007909,
                "amplitude_psnr": 64.742,
                "amplitude_psnr_unscaled": 5.365912267523627,
                "amplitude_ssim": 0.07236,
                "phase_mse": 0.3981,
                "phase_psnr": 52.131,
                "phase_ssim": 0.3805,
            },
        ]
    )

    assert (
        r"ID & Supervised baseline & \textbf{0.01305} & \textbf{18.147} & \textbf{0.538} & \textbf{0.07434} & \textbf{59.418} & \textbf{0.8119} \\"
        in table
    )
    assert (
        r"OOD & PtychoPINN & \textbf{0.03224} & \textbf{14.219} & \textbf{0.1757} & \textbf{0.295} & \textbf{53.432} & \textbf{0.5137} \\"
        in table
    )
    assert (
        r"OOD & Supervised baseline & 0.2476 & 5.366 & 0.07236 & 0.3981 & 52.131 & 0.3805 \\"
        in table
    )


def test_output_root_lock_is_acquired_before_invocation_artifacts_are_written(tmp_path):
    study = _import_study()
    output_root = tmp_path / "run"
    touched = []

    with study.acquire_output_lock(output_root, force_stale_lock=False) as lock:
        assert (output_root / "run.lock").read_text().strip() == str(lock.pid)
        assert not (output_root / "invocation.json").exists()
        touched.append("locked")
        (output_root / "invocation.json").write_text("{}\n")

    assert touched == ["locked"]
    assert not (output_root / "run.lock").exists()
    assert json.loads((output_root / "invocation.json").read_text()) == {}


def test_existing_live_run_lock_blocks_without_overwriting_invocation_artifacts(tmp_path):
    study = _import_study()
    output_root = tmp_path / "run"
    output_root.mkdir()
    (output_root / "run.lock").write_text(f"{study.os.getpid()}\n")
    invocation = output_root / "invocation.json"
    invocation.write_text('{"preserve": true}\n')

    with pytest.raises(study.OutputLockError, match="active"):
        study.acquire_output_lock(output_root, force_stale_lock=False)

    assert json.loads(invocation.read_text()) == {"preserve": True}


def test_stale_lock_requires_force_and_records_stale_lock_content(tmp_path):
    study = _import_study()
    output_root = tmp_path / "run"
    output_root.mkdir()
    stale_lock = output_root / "run.lock"
    stale_lock.write_text("999999999\n")

    with pytest.raises(study.OutputLockError, match="stale"):
        study.acquire_output_lock(output_root, force_stale_lock=False)

    with study.acquire_output_lock(output_root, force_stale_lock=True) as lock:
        assert lock.replaced_stale_lock is True
        assert lock.stale_lock_content == "999999999"
        assert stale_lock.read_text().strip() == str(lock.pid)


def test_script_file_execution_inventory_only_uses_repo_root_imports(tmp_path):
    fixture_repo = tmp_path / "fixture_repo"
    _write_minimal_default_inputs(fixture_repo)
    output_root = tmp_path / "inventory"

    result = subprocess.run(
        [
            "python",
            "scripts/studies/ood_fig5_metrics.py",
            "--repo-root",
            str(fixture_repo),
            "--output-root",
            str(output_root),
            "--inventory-only",
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (output_root / "invocation.json").exists()
    assert (output_root / "fig5_source_inventory.json").exists()
    assert not (output_root / "fig5_reference_validation.json").exists()
    assert not (output_root / "fig5_ood_metrics.json").exists()


def test_default_alignment_calls_align_for_evaluation_with_yx_coords_and_stitch_size(monkeypatch):
    study = _import_study()
    row = study.StudyRow(
        row_id="toy",
        condition="ID",
        model="PtychoPINN",
        reconstruction_path=Path("recon.npz"),
        panel_reference_path=None,
    )
    recon = np.ones((4, 4), dtype=np.complex64)
    reference = np.ones((8, 8), dtype=np.complex64) * 2
    scan_coords_yx = np.array([[3, 4], [5, 7]], dtype=np.float32)
    calls = []

    def sentinel(reconstruction_image, ground_truth_image, coords, stitch_patch_size):
        calls.append((reconstruction_image, ground_truth_image, coords, stitch_patch_size))
        return reconstruction_image[:3, :3], ground_truth_image[:3, :3]

    monkeypatch.setattr(study, "align_for_evaluation", sentinel)

    aligned_recon, aligned_gt, manifest = study.align_row_for_metrics(
        row=row,
        reconstruction=recon,
        full_reference=reference,
        scan_coords_yx=scan_coords_yx,
        stitch_patch_size=20,
        panel_reference=None,
        panel_reference_inventory=None,
        allow_panel_artifact_exception=False,
        panel_exception_reason="",
    )

    assert calls == [(recon, reference, scan_coords_yx, 20)]
    assert aligned_recon.shape == (3, 3)
    assert aligned_gt.shape == (3, 3)
    assert manifest["alignment_mode"] == "coordinate_align_for_evaluation"
    assert manifest["coordinate_count"] == 2
    assert manifest["coordinate_ranges_yx"] == {"y": [3.0, 5.0], "x": [4.0, 7.0]}
    assert manifest["stitch_patch_size"] == 20


def test_paper_mode_requires_heldout_test_half_evaluation(tmp_path):
    study = _import_study()
    args = study.parse_args(
        [
            "--output-root",
            str(tmp_path),
            "--evaluation-region",
            "all_scan",
            "--require-heldout-eval",
        ]
    )

    with pytest.raises(study.StopCondition, match="held-out"):
        study.validate_heldout_evaluation_requirement(args)


def test_heldout_sorted_y_split_assigns_odd_count_deterministically():
    study = _import_study()
    scan_coords_yx = np.array(
        [
            [50.0, 0.0],
            [10.0, 0.0],
            [30.0, 0.0],
            [20.0, 0.0],
            [40.0, 0.0],
        ],
        dtype=np.float32,
    )

    split = study.select_spatial_holdout_split(
        scan_coords_yx,
        policy="bottom_half_by_sorted_y",
        split_source=None,
    )

    assert split["total_coordinate_count"] == 5
    assert split["train_indices"] == [1, 3]
    assert split["eval_indices"] == [2, 4, 0]
    assert split["train_half"] == "top"
    assert split["eval_half"] == "bottom"
    assert split["eval_coordinate_count"] == 3
    assert split["axis"] == "y"


def test_paper_top_train_bottom_test_split_uses_high_y_for_training():
    study = _import_study()
    scan_coords_yx = np.array(
        [
            [50.0, 0.0],
            [10.0, 0.0],
            [30.0, 0.0],
            [20.0, 0.0],
            [40.0, 0.0],
        ],
        dtype=np.float32,
    )

    split = study.select_spatial_holdout_split(
        scan_coords_yx,
        policy="paper_top_train_bottom_test_by_high_y",
        split_source=None,
    )

    assert split["total_coordinate_count"] == 5
    assert split["train_indices"] == [2, 4, 0]
    assert split["eval_indices"] == [1, 3]
    assert split["train_half"] == "paper_top_high_y"
    assert split["eval_half"] == "paper_bottom_low_y"
    assert "ycoords >= threshold" in split["axis_direction_evidence"]


def test_split_contract_rejects_coordinate_index_overlap():
    study = _import_study()
    scan_coords_yx = np.array([[5, 5], [5, 6], [6, 5]], dtype=np.float32)

    with pytest.raises(study.StopCondition, match="coordinate indices overlap"):
        study.audit_split_contract(
            scan_coords_yx,
            train_indices=[0, 1],
            eval_indices=[1, 2],
            stitch_patch_size=2,
            nonoverlap_level="coordinate_indices",
        )


def test_split_contract_rejects_noncontiguous_eval_subset():
    study = _import_study()
    scan_coords_yx = np.array(
        [
            [5, 5],
            [5, 6],
            [6, 5],
            [6, 6],
            [30, 30],
            [30, 31],
            [31, 30],
            [31, 31],
            [70, 70],
            [70, 71],
            [71, 70],
            [71, 71],
        ],
        dtype=np.float32,
    )

    with pytest.raises(study.StopCondition, match="eval.*contiguous"):
        study.audit_split_contract(
            scan_coords_yx,
            train_indices=[0, 1, 2, 3],
            eval_indices=[4, 5, 6, 7, 8, 9, 10, 11],
            stitch_patch_size=2,
            nonoverlap_level="coordinate_indices",
            full_reference_shape=(96, 96),
        )


def test_split_contract_records_panel_crop_mismatch_as_internal_provenance():
    study = _import_study()
    scan_coords_yx = np.array(
        [
            [5, 5],
            [5, 6],
            [6, 5],
            [6, 6],
            [30, 30],
            [30, 31],
            [31, 30],
            [31, 31],
        ],
        dtype=np.float32,
    )

    contract = study.audit_split_contract(
        scan_coords_yx,
        train_indices=[0, 1, 2, 3],
        eval_indices=[4, 5, 6, 7],
        stitch_patch_size=2,
        nonoverlap_level="coordinate_indices",
        full_reference_shape=(64, 64),
        metric_region_matches_displayed_panel=False,
        allow_panel_metric_region_mismatch=True,
    )

    assert contract["status"] == "ok"
    assert contract["metric_region_matches_displayed_panel"] is False
    assert contract["panel_metric_region_mismatch_policy"] == "internal_provenance_only"


def test_object_footprint_overlap_requires_guard_band_when_strict():
    study = _import_study()
    scan_coords_yx = np.array([[40, 40], [45, 40]], dtype=np.float32)

    with pytest.raises(study.StopCondition, match="object footprint overlap"):
        study.audit_split_contract(
            scan_coords_yx,
            train_indices=[0],
            eval_indices=[1],
            stitch_patch_size=20,
            nonoverlap_level="object_footprint",
            full_reference_shape=(96, 96),
        )


def test_strict_split_contract_drops_boundary_eval_coordinates_before_crop(monkeypatch):
    study = _import_study()
    row = study.StudyRow("toy", "OOD", "PtychoPINN", Path("recon.npz"), None)
    scan_coords_yx = np.array(
        [
            [10, 10],
            [10, 11],
            [11, 10],
            [11, 11],
            [20, 10],
            [20, 11],
            [30, 10],
            [30, 11],
        ],
        dtype=np.float32,
    )
    full_aligned = np.ones((32, 13), dtype=np.complex64)
    monkeypatch.setattr(study, "align_for_evaluation", lambda *_args: (full_aligned, full_aligned.copy()))

    _, _, manifest = study.align_row_for_metrics(
        row=row,
        reconstruction=np.zeros((40, 30), dtype=np.complex64),
        full_reference=np.zeros((50, 40), dtype=np.complex64),
        scan_coords_yx=scan_coords_yx,
        stitch_patch_size=12,
        panel_reference=None,
        panel_reference_inventory=None,
        allow_panel_artifact_exception=False,
        panel_exception_reason="",
        evaluation_region="heldout_test_half",
        heldout_split_policy="bottom_half_by_sorted_y",
        heldout_split_source=None,
        split_nonoverlap_level="object_footprint",
        allow_panel_metric_region_mismatch=True,
    )

    assert manifest["eval_coordinate_count"] == 2
    assert manifest["guard_band_eval_original_coordinate_count"] == 4
    assert manifest["guard_band_eval_dropped_indices"] == [4, 5]
    assert manifest["split_contract"]["status"] == "ok"
    assert manifest["split_contract"]["nonoverlap_level"] == "object_footprint"
    assert manifest["split_contract"]["object_footprint_overlap_pixel_count"] == 0
    assert manifest["split_contract"]["object_footprint_overlap_policy"] == "enforced_no_overlap"


def test_heldout_crop_uses_relative_scan_bbox_not_center_crop(monkeypatch):
    study = _import_study()
    row = study.StudyRow(
        row_id="toy",
        condition="ID",
        model="PtychoPINN",
        reconstruction_path=Path("recon.npz"),
        panel_reference_path=None,
    )
    scan_coords_yx = np.array(
        [[y, x] for y in (5, 6) for x in range(5, 15)]
        + [[y, x] for y in (13, 14) for x in range(5, 15)],
        dtype=np.float32,
    )
    full_aligned_recon = np.repeat(np.arange(11, dtype=np.float32)[:, None], 11, axis=1).astype(np.complex64)
    full_aligned_gt = (100 + full_aligned_recon).astype(np.complex64)

    def sentinel(reconstruction_image, ground_truth_image, coords, stitch_patch_size):
        assert coords is scan_coords_yx
        assert stitch_patch_size == 2
        return full_aligned_recon, full_aligned_gt

    monkeypatch.setattr(study, "align_for_evaluation", sentinel)

    aligned_recon, aligned_gt, manifest = study.align_row_for_metrics(
        row=row,
        reconstruction=np.zeros((11, 11), dtype=np.complex64),
        full_reference=np.zeros((20, 20), dtype=np.complex64),
        scan_coords_yx=scan_coords_yx,
        stitch_patch_size=2,
        panel_reference=None,
        panel_reference_inventory=None,
        allow_panel_artifact_exception=False,
        panel_exception_reason="",
        evaluation_region="heldout_test_half",
        heldout_split_policy="bottom_half_by_sorted_y",
        heldout_split_source=None,
    )

    np.testing.assert_allclose(aligned_recon[:, 0].real, np.array([8, 9, 10], dtype=np.float32))
    np.testing.assert_allclose(aligned_gt[:, 0].real, np.array([108, 109, 110], dtype=np.float32))
    assert manifest["evaluation_region"] == "heldout_test_half"
    assert manifest["heldout_only"] is True
    assert manifest["total_coordinate_count"] == 40
    assert manifest["eval_coordinate_count"] == 20
    assert manifest["heldout_relative_slice_rows_cols"] == [8, 11, 0, 11]
    assert manifest["heldout_crop_shape"] == [3, 11]
    assert manifest["split_contract"]["status"] == "ok"


def test_alignment_manifest_records_heldout_bbox_and_coordinate_counts(monkeypatch):
    study = _import_study()
    row = study.StudyRow("toy", "OOD", "Supervised baseline", Path("recon.npz"), None)
    scan_coords_yx = np.array(
        [[5, x] for x in range(5, 15)] + [[14, x] for x in range(5, 15)],
        dtype=np.float32,
    )
    monkeypatch.setattr(
        study,
        "align_for_evaluation",
        lambda *_args, **_kwargs: (
            np.ones((11, 11), dtype=np.complex64),
            np.ones((11, 11), dtype=np.complex64),
        ),
    )

    _, _, manifest = study.align_row_for_metrics(
        row=row,
        reconstruction=np.ones((11, 11), dtype=np.complex64),
        full_reference=np.ones((20, 20), dtype=np.complex64),
        scan_coords_yx=scan_coords_yx,
        stitch_patch_size=2,
        panel_reference=None,
        panel_reference_inventory=None,
        allow_panel_artifact_exception=False,
        panel_exception_reason="",
        evaluation_region="heldout_test_half",
        heldout_split_policy="bottom_half_by_sorted_y",
        heldout_split_source=None,
    )

    assert manifest["all_scan_bbox_rows_cols"] == [4, 15, 4, 15]
    assert manifest["aligned_full_bbox_rows_cols"] == [4, 15, 4, 15]
    assert manifest["heldout_bbox_rows_cols"] == [13, 15, 4, 15]
    assert manifest["total_coordinate_count"] == 20
    assert manifest["eval_coordinate_count"] == 10
    assert manifest["heldout_split_policy"] == "bottom_half_by_sorted_y"
    assert manifest["split_contract"]["status"] == "ok"


def test_panel_reference_validation_records_shape_and_max_abs_difference():
    study = _import_study()
    row = study.StudyRow(
        row_id="toy",
        condition="OOD",
        model="PtychoPINN",
        reconstruction_path=Path("recon.npz"),
        panel_reference_path=Path("panel.npz"),
    )
    aligned_gt = np.array([[1 + 0j, 2 + 0j], [3 + 0j, 4 + 0j]], dtype=np.complex64)
    panel_reference = aligned_gt + np.array([[0, 0], [0, 0.5]], dtype=np.complex64)
    inventory = {
        "path": "panel.npz",
        "sha256": "abc",
        "keys": ["ground_truth_complex"],
        "arrays": {"ground_truth_complex": {"shape": [2, 2], "dtype": "complex64"}},
    }

    validation = study.validate_panel_reference(
        row=row,
        aligned_gt=aligned_gt,
        panel_reference=panel_reference,
        panel_reference_inventory=inventory,
        reference_validation_scope="row_specific_panel_reference",
    )

    assert validation["reference_validation_status"] == "failed"
    assert validation["reference_validation_mode"] == "numeric_mismatch"
    assert validation["panel_reference_shape"] == [2, 2]
    assert validation["aligned_ground_truth_shape"] == [2, 2]
    assert validation["panel_reference_sha256"] == "abc"
    assert validation["panel_reference_keys"] == ["ground_truth_complex"]
    assert validation["max_abs_diff"] == pytest.approx(0.5)
    assert validation["relative_l2_diff"] > 0
    assert validation["reference_validation_scope"] == "row_specific_panel_reference"


def test_reference_validation_accepts_exact_complex128_digest_match():
    study = _import_study()
    row = study.StudyRow("toy", "OOD", "PtychoPINN", Path("recon.npz"), Path("panel.npz"))
    aligned_gt = np.array([[1 + 2j, 3 + 4j]], dtype=np.complex64)
    panel_reference = aligned_gt.astype(np.complex128)

    validation = study.validate_panel_reference(
        row=row,
        aligned_gt=aligned_gt,
        panel_reference=panel_reference,
        panel_reference_inventory={"sha256": "sha", "keys": ["ground_truth_complex"], "arrays": {}},
        reference_validation_scope="row_specific_panel_reference",
    )

    assert validation["reference_validation_status"] == "passed"
    assert validation["reference_validation_mode"] == "exact_digest_match"
    assert validation["canonical_cropped_reference_digest"] == validation["panel_reference_digest"]


def test_reference_validation_accepts_numeric_identity_with_design_tolerances():
    study = _import_study()
    row = study.StudyRow("toy", "OOD", "PtychoPINN", Path("recon.npz"), Path("panel.npz"))
    aligned_gt = np.ones((4, 4), dtype=np.complex128)
    panel_reference = aligned_gt.copy()
    panel_reference[0, 0] += 1e-12

    validation = study.validate_panel_reference(
        row=row,
        aligned_gt=aligned_gt,
        panel_reference=panel_reference,
        panel_reference_inventory={"sha256": "sha", "keys": ["ground_truth_complex"], "arrays": {}},
        reference_validation_scope="row_specific_panel_reference",
    )

    assert validation["reference_validation_status"] == "passed"
    assert validation["reference_validation_mode"] == "numeric_identity_tolerance"
    assert validation["max_abs_diff"] <= validation["max_abs_diff_tolerance"]
    assert validation["relative_l2_diff"] <= validation["relative_l2_diff_tolerance"]


def test_reference_validation_rejects_shape_or_numeric_mismatch():
    study = _import_study()
    row = study.StudyRow("toy", "OOD", "PtychoPINN", Path("recon.npz"), Path("panel.npz"))

    shape_validation = study.validate_panel_reference(
        row=row,
        aligned_gt=np.ones((4, 4), dtype=np.complex64),
        panel_reference=np.ones((3, 4), dtype=np.complex64),
        panel_reference_inventory={"sha256": "sha", "keys": ["ground_truth_complex"], "arrays": {}},
        reference_validation_scope="row_specific_panel_reference",
    )
    assert shape_validation["reference_validation_status"] == "failed"
    assert shape_validation["reference_validation_mode"] == "shape_mismatch"

    numeric_validation = study.validate_panel_reference(
        row=row,
        aligned_gt=np.ones((4, 4), dtype=np.complex64),
        panel_reference=np.ones((4, 4), dtype=np.complex64) * 2,
        panel_reference_inventory={"sha256": "sha", "keys": ["ground_truth_complex"], "arrays": {}},
        reference_validation_scope="row_specific_panel_reference",
    )
    assert numeric_validation["reference_validation_status"] == "failed"
    assert numeric_validation["reference_validation_mode"] == "numeric_mismatch"


def test_panel_artifact_exception_requires_flag_and_reason(monkeypatch):
    study = _import_study()
    row = study.StudyRow(
        row_id="toy",
        condition="OOD",
        model="PtychoPINN",
        reconstruction_path=Path("recon.npz"),
        panel_reference_path=Path("panel.npz"),
    )

    def fail_align(*_args, **_kwargs):
        raise ValueError("coordinate source mismatch")

    monkeypatch.setattr(study, "align_for_evaluation", fail_align)

    with pytest.raises(ValueError, match="panel-exception-reason"):
        study.align_row_for_metrics(
            row=row,
            reconstruction=np.ones((4, 4), dtype=np.complex64),
            full_reference=np.ones((8, 8), dtype=np.complex64),
            scan_coords_yx=np.array([[1, 2]], dtype=np.float32),
            stitch_patch_size=20,
            panel_reference=np.ones((4, 4), dtype=np.complex64),
            panel_reference_inventory={"sha256": "abc", "keys": [], "arrays": {}},
            allow_panel_artifact_exception=True,
            panel_exception_reason="",
        )


def test_panel_artifact_exception_is_never_used_when_coordinate_validation_succeeds(monkeypatch):
    study = _import_study()
    row = study.StudyRow(
        row_id="toy",
        condition="OOD",
        model="PtychoPINN",
        reconstruction_path=Path("recon.npz"),
        panel_reference_path=Path("panel.npz"),
    )

    def sentinel(reconstruction_image, ground_truth_image, _coords, _stitch_patch_size):
        return reconstruction_image, ground_truth_image[:4, :4]

    monkeypatch.setattr(study, "align_for_evaluation", sentinel)

    _, _, manifest = study.align_row_for_metrics(
        row=row,
        reconstruction=np.ones((4, 4), dtype=np.complex64),
        full_reference=np.ones((8, 8), dtype=np.complex64),
        scan_coords_yx=np.array([[1, 2]], dtype=np.float32),
        stitch_patch_size=20,
        panel_reference=np.ones((4, 4), dtype=np.complex64),
        panel_reference_inventory={"sha256": "abc", "keys": [], "arrays": {}},
        allow_panel_artifact_exception=True,
        panel_exception_reason="preapproved fallback if coordinate alignment fails",
    )

    assert manifest["alignment_mode"] == "coordinate_align_for_evaluation"
    assert manifest["panel_exception_available"] is True
    assert "panel_artifact_exception" not in manifest["alignment_mode"]


def test_alignment_manifest_records_coordinate_crop_final_shape_and_mode_per_row(monkeypatch):
    study = _import_study()
    row = study.StudyRow(
        row_id="toy",
        condition="ID",
        model="Supervised baseline",
        reconstruction_path=Path("recon.npz"),
        panel_reference_path=None,
    )

    def sentinel(reconstruction_image, ground_truth_image, _coords, _stitch_patch_size):
        return reconstruction_image[:3, :4], ground_truth_image[:3, :4]

    monkeypatch.setattr(study, "align_for_evaluation", sentinel)

    _, _, manifest = study.align_row_for_metrics(
        row=row,
        reconstruction=np.ones((5, 6), dtype=np.complex64),
        full_reference=np.ones((8, 8), dtype=np.complex64),
        scan_coords_yx=np.array([[2, 3], [4, 6]], dtype=np.float32),
        stitch_patch_size=20,
        panel_reference=None,
        panel_reference_inventory=None,
        allow_panel_artifact_exception=False,
        panel_exception_reason="",
    )

    assert manifest["row_id"] == "toy"
    assert manifest["alignment_mode"] == "coordinate_align_for_evaluation"
    assert manifest["input_reconstruction_shape"] == [5, 6]
    assert manifest["coordinate_aligned_reconstruction_shape"] == [3, 4]
    assert manifest["coordinate_aligned_ground_truth_shape"] == [3, 4]
    assert manifest["panel_validation_status"] == "missing_panel_reference"
    assert manifest["reference_validation_status"] == "failed"


def test_fine_registration_uses_find_translation_offset_then_apply_shift_and_crop(monkeypatch):
    study = _import_study()
    aligned_recon = np.ones((6, 6), dtype=np.complex64)
    aligned_gt = np.ones((6, 6), dtype=np.complex64) * 2
    calls = []

    def fake_find(image, reference, upsample_factor):
        calls.append(("find", image, reference, upsample_factor))
        return (0.25, -0.5)

    def fake_apply(image, reference, offset, border_crop):
        calls.append(("apply", image, reference, offset, border_crop))
        return image[1:-1, 1:-1], reference[1:-1, 1:-1]

    monkeypatch.setattr(study, "find_translation_offset", fake_find)
    monkeypatch.setattr(study, "apply_shift_and_crop", fake_apply)

    registered_recon, registered_gt, manifest = study.register_aligned_row(
        aligned_recon,
        aligned_gt,
        upsample_factor=50,
        border_crop=2,
        fail_on_error=True,
    )

    assert calls == [
        ("find", aligned_recon, aligned_gt, 50),
        ("apply", aligned_recon, aligned_gt, (0.25, -0.5), 2),
    ]
    assert registered_recon.shape == (4, 4)
    assert registered_gt.shape == (4, 4)
    assert manifest["registration_status"] == "ok"
    assert manifest["registration_offset_yx"] == [0.25, -0.5]
    assert manifest["upsample_factor"] == 50
    assert manifest["border_crop"] == 2


def test_registration_failure_is_manifest_recorded_and_not_silent(monkeypatch):
    study = _import_study()

    def fail_find(*_args, **_kwargs):
        raise RuntimeError("phase correlation failed")

    monkeypatch.setattr(study, "find_translation_offset", fail_find)

    registered_recon, registered_gt, manifest = study.register_aligned_row(
        np.ones((6, 6), dtype=np.complex64),
        np.ones((6, 6), dtype=np.complex64),
        upsample_factor=50,
        border_crop=2,
        fail_on_error=False,
    )

    assert registered_recon is None
    assert registered_gt is None
    assert manifest["registration_status"] == "failed"
    assert "phase correlation failed" in manifest["registration_error"]


def test_fine_registration_offsets_are_recorded_after_primary_metrics_and_large_offsets_are_not_scored(monkeypatch):
    study = _import_study()
    calls = []

    def fake_find(image, reference, upsample_factor):
        calls.append(("find", upsample_factor))
        return (3.0, 0.25) if upsample_factor == 50 else (3.1, 0.25)

    def fake_apply(*_args, **_kwargs):
        calls.append(("apply",))
        raise AssertionError("crop-unsafe diagnostics must not be shifted for scoring")

    monkeypatch.setattr(study, "find_translation_offset", fake_find)
    monkeypatch.setattr(study, "apply_shift_and_crop", fake_apply)
    row = study.StudyRow("toy", "OOD", "PtychoPINN", Path("recon.npz"), None)

    manifest = study.compute_fine_registration_sensitivity(
        row=row,
        aligned_recon=np.ones((64, 64), dtype=np.complex64),
        aligned_gt=np.ones((64, 64), dtype=np.complex64),
        args=types.SimpleNamespace(
            upsample_factor=50,
            registration_stability_factor=10,
            border_crop=2,
            eval_offset=4,
            phase_align_method="plane",
            frc_sigma=0.0,
            ms_ssim_sigma=1.0,
        ),
    )

    assert calls == [("find", 50), ("find", 10)]
    assert manifest["registration_offset_yx_upsample50"] == [3.0, 0.25]
    assert manifest["registration_offset_yx_upsample10"] == [3.1, 0.25]
    assert manifest["registration_offset_max_abs_component_upsample50"] == pytest.approx(3.0)
    assert manifest["diagnostic_crop_safety_status"] == "offset_exceeds_border_crop"
    assert manifest["fine_registration_sensitivity_status"] == "not_scored_offset_exceeds_border_crop"
    assert "fine_registered_amplitude_mse" not in manifest


def test_crop_safe_fine_registration_metrics_are_labeled_sensitivity_only(monkeypatch):
    study = _import_study()

    monkeypatch.setattr(study, "find_translation_offset", lambda *_args, **kwargs: (0.5, -0.25))
    monkeypatch.setattr(
        study,
        "apply_shift_and_crop",
        lambda image, reference, offset, border_crop: (image[border_crop:-border_crop, border_crop:-border_crop], reference[border_crop:-border_crop, border_crop:-border_crop]),
    )
    monkeypatch.setattr(study, "eval_reconstruction", lambda *args, **kwargs: _dummy_eval_metrics())
    row = study.StudyRow("toy", "ID", "PtychoPINN", Path("recon.npz"), None)

    manifest = study.compute_fine_registration_sensitivity(
        row=row,
        aligned_recon=np.ones((68, 68), dtype=np.complex64),
        aligned_gt=np.ones((68, 68), dtype=np.complex64),
        args=types.SimpleNamespace(
            upsample_factor=50,
            registration_stability_factor=10,
            border_crop=2,
            eval_offset=4,
            phase_align_method="plane",
            frc_sigma=0.0,
            ms_ssim_sigma=1.0,
        ),
    )

    assert manifest["fine_registration_sensitivity_status"] == "scored_crop_safe_sensitivity_only"
    assert manifest["fine_registered_amplitude_mse"] == pytest.approx(0.01)
    assert manifest["fine_registered_primary_fields_promoted"] is False


def _metric_args(**overrides):
    values = {
        "eval_offset": 4,
        "phase_align_method": "plane",
        "frc_sigma": 0.0,
        "ms_ssim_sigma": 1.0,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _dummy_eval_metrics():
    return {
        "mae": (0.1, 0.2),
        "mse": (0.01, 0.02),
        "psnr": (30.0, 20.0),
        "ssim": (0.9, 0.8),
        "ms_ssim": (0.91, 0.81),
        "frc50": (12.0, 10.0),
        "frc1over7": (18.0, 16.0),
        "frc": ([1.0, 0.4], [1.0, 0.3]),
    }


def test_metrics_call_eval_reconstruction_with_plane_phase_alignment_by_default(monkeypatch):
    study = _import_study()
    calls = []

    def fake_eval(recon, gt, label, phase_align_method, frc_sigma, ms_ssim_sigma):
        calls.append(
            {
                "recon_shape": recon.shape,
                "gt_shape": gt.shape,
                "label": label,
                "phase_align_method": phase_align_method,
                "frc_sigma": frc_sigma,
                "ms_ssim_sigma": ms_ssim_sigma,
                "offset": study.legacy_params.cfg["offset"],
            }
        )
        return _dummy_eval_metrics()

    monkeypatch.setattr(study, "eval_reconstruction", fake_eval)
    row = study.StudyRow("toy", "ID", "PtychoPINN", Path("recon.npz"), None)

    payload, manifest = study.compute_row_metrics(
        row=row,
        recon_registered=np.ones((64, 64), dtype=np.complex64),
        gt_registered=np.ones((64, 64), dtype=np.complex64) * 2,
        args=_metric_args(),
    )

    assert calls == [
        {
            "recon_shape": (1, 64, 64, 1),
            "gt_shape": (64, 64, 1),
            "label": "toy",
            "phase_align_method": "plane",
            "frc_sigma": 0.0,
            "ms_ssim_sigma": 1.0,
            "offset": 4,
        }
    ]
    assert payload["row_id"] == "toy"
    assert manifest["metric_status"] == "ok"
    assert manifest["pre_eval_adapter_reconstruction_shape"] == [64, 64]
    assert manifest["eval_stitched_obj_shape"] == [1, 64, 64, 1]
    assert manifest["eval_ground_truth_obj_shape"] == [64, 64, 1]
    assert manifest["eval_shape_adapter_no_semantic_change"] is True


def test_metrics_records_amplitude_scale_factor_phase_policy_background_policy_offsets_and_pre_post_eval_shapes(monkeypatch):
    study = _import_study()
    monkeypatch.setattr(study, "eval_reconstruction", lambda *args, **kwargs: _dummy_eval_metrics())
    row = study.StudyRow("toy", "OOD", "Supervised baseline", Path("recon.npz"), None)
    recon = np.ones((64, 66), dtype=np.complex64) * 2
    gt = np.ones((64, 66), dtype=np.complex64) * 8

    _, manifest = study.compute_row_metrics(
        row=row,
        recon_registered=recon,
        gt_registered=gt,
        args=_metric_args(),
    )

    assert manifest["metric_trim_offset"] == 4
    assert manifest["metric_trim_pixels_per_edge"] == 2
    assert manifest["pre_eval_shape"] == [64, 66]
    assert manifest["post_eval_trim_shape"] == [60, 62]
    assert manifest["amplitude_scale_factor"] == pytest.approx(4.0)
    assert manifest["amplitude_mean_reference"] == pytest.approx(8.0)
    assert manifest["amplitude_mean_reconstruction"] == pytest.approx(2.0)
    assert manifest["amplitude_mean_ratio_unscaled"] == pytest.approx(0.25)
    assert manifest["amplitude_mse_unscaled"] == pytest.approx(36.0)
    assert manifest["amplitude_mae_unscaled"] == pytest.approx(6.0)
    assert manifest["amplitude_scaled_unscaled_conclusion_conflict"] is False
    assert manifest["amplitude_scale_factor_source"] == "post_eval_trim"
    assert manifest["phase_align_method"] == "plane"
    assert manifest["background_policy"] == "no_support_or_background_mask"
    assert "eval_reconstruction_legacy_trim" in manifest["metric_contract"]


def test_metrics_sets_and_restores_params_cfg_offset_around_eval_reconstruction(monkeypatch):
    study = _import_study()
    before = study.legacy_params.cfg.get("offset")
    study.legacy_params.cfg["offset"] = 98
    observed = []

    def fake_eval(*_args, **_kwargs):
        observed.append(study.legacy_params.cfg["offset"])
        return _dummy_eval_metrics()

    monkeypatch.setattr(study, "eval_reconstruction", fake_eval)
    try:
        study.compute_row_metrics(
            row=study.StudyRow("toy", "ID", "PtychoPINN", Path("recon.npz"), None),
            recon_registered=np.ones((64, 64), dtype=np.complex64),
            gt_registered=np.ones((64, 64), dtype=np.complex64),
            args=_metric_args(eval_offset=6),
        )
        assert observed == [6]
        assert study.legacy_params.cfg["offset"] == 98
    finally:
        if before is None:
            study.legacy_params.cfg.pop("offset", None)
        else:
            study.legacy_params.cfg["offset"] = before


def test_eval_offset_must_be_positive_even_and_smaller_than_registered_shape():
    study = _import_study()

    with pytest.raises(ValueError, match="positive even"):
        study.validate_eval_offset(0, (8, 8))
    with pytest.raises(ValueError, match="positive even"):
        study.validate_eval_offset(3, (8, 8))
    with pytest.raises(ValueError, match="smaller"):
        study.validate_eval_offset(8, (8, 8))

    assert study.validate_eval_offset(4, (8, 8)) == 2


def test_eval_offset_4_requires_post_eval_shape_at_least_56_by_56(monkeypatch):
    study = _import_study()
    monkeypatch.setattr(study, "eval_reconstruction", lambda *args, **kwargs: _dummy_eval_metrics())

    with pytest.raises(ValueError, match="post-eval-trim.*56"):
        study.compute_row_metrics(
            row=study.StudyRow("tiny", "ID", "PtychoPINN", Path("recon.npz"), None),
            recon_registered=np.ones((59, 64), dtype=np.complex64),
            gt_registered=np.ones((59, 64), dtype=np.complex64),
            args=_metric_args(),
        )


def test_min_post_eval_trim_dim_can_be_lowered_for_heldout_metrics(monkeypatch):
    study = _import_study()
    monkeypatch.setattr(study, "eval_reconstruction", lambda *args, **kwargs: _dummy_eval_metrics())

    payload, manifest = study.compute_row_metrics(
        row=study.StudyRow("heldout", "ID", "PtychoPINN", Path("recon.npz"), None),
        recon_registered=np.ones((42, 65), dtype=np.complex64),
        gt_registered=np.ones((42, 65), dtype=np.complex64),
        args=_metric_args(min_post_eval_trim_dim=32),
    )

    assert payload["post_eval_trim_shape"] == [38, 61]
    assert manifest["min_post_eval_trim_dim"] == 32


def test_frc_square_crop_is_recorded_when_core_metric_shape_is_non_square():
    study = _import_study()

    fov = study.describe_frc_field_of_view((4, 6))

    assert fov["frc_input_shape"] == [4, 6]
    assert fov["frc_square_crop_applied"] is True
    assert fov["frc_shape"] == [4, 4]
    assert fov["frc_crop_slices_yx"] == {"y": [0, 4], "x": [1, 5]}
    assert fov["frc_field_of_view_matches_core_metrics"] is False
    assert fov["frc_field_of_view_policy"] == "post_eval_metric_shape -> frc_cutoffs_square_center_crop"


def test_frc_is_artifact_only_when_square_crop_differs_from_core_metric_fov(monkeypatch):
    study = _import_study()
    monkeypatch.setattr(study, "eval_reconstruction", lambda *args, **kwargs: _dummy_eval_metrics())

    payload, manifest = study.compute_row_metrics(
        row=study.StudyRow("toy", "OOD", "PtychoPINN", Path("recon.npz"), None),
        recon_registered=np.ones((64, 66), dtype=np.complex64),
        gt_registered=np.ones((64, 66), dtype=np.complex64),
        args=_metric_args(),
    )

    assert manifest["frc_square_crop_applied"] is True
    assert manifest["frc_field_of_view_matches_core_metrics"] is False
    assert manifest["frc_paper_table_status"] == "artifact_only_due_to_frc_square_crop"
    assert payload["frc_paper_table_status"] == "artifact_only_due_to_frc_square_crop"


def test_reference_based_frc_failure_omits_frc_from_paper_table_but_keeps_core_metrics(monkeypatch):
    study = _import_study()
    metrics = _dummy_eval_metrics()
    metrics["frc50"] = (np.nan, np.nan)
    metrics["frc1over7"] = (np.nan, np.nan)
    monkeypatch.setattr(study, "eval_reconstruction", lambda *args, **kwargs: metrics)

    payload, manifest = study.compute_row_metrics(
        row=study.StudyRow("toy", "ID", "PtychoPINN", Path("recon.npz"), None),
        recon_registered=np.ones((64, 64), dtype=np.complex64),
        gt_registered=np.ones((64, 64), dtype=np.complex64),
        args=_metric_args(),
    )

    assert payload["amplitude_mse"] == pytest.approx(0.01)
    assert payload["phase_ssim"] == pytest.approx(0.8)
    assert manifest["frc_paper_table_status"] == "omitted_due_to_frc_failure"
    assert payload["frc_paper_table_status"] == "omitted_due_to_frc_failure"


def test_metrics_json_and_csv_contain_four_rows(tmp_path):
    study = _import_study()
    rows = []
    for index in range(4):
        row = {
            "row_id": f"row_{index}",
            "condition": "ID" if index < 2 else "OOD",
            "model": "PtychoPINN" if index % 2 == 0 else "Supervised baseline",
            "amplitude_mse": 0.1 + index,
            "amplitude_psnr": 30.0 - index,
            "amplitude_ssim": 0.9,
            "phase_mse": 0.2 + index,
            "phase_psnr": 20.0 - index,
            "phase_ssim": 0.8,
            "frc_square_crop_applied": False,
            "frc_paper_table_status": "paper_candidate_same_fov",
        }
        rows.append(row)

    study.write_metrics_artifacts(
        output_root=tmp_path,
        rows=rows,
        manifest={"status": "metrics_complete", "rows": rows},
        metric_policy={"alignment": "coordinate"},
    )

    payload = json.loads((tmp_path / "fig5_ood_metrics.json").read_text())
    assert len(payload["rows"]) == 4
    csv_text = (tmp_path / "fig5_ood_metrics.csv").read_text()
    assert csv_text.count("\n") == 5
    assert "row_3" in csv_text


def test_frc_json_csv_include_field_of_view_status_when_available(tmp_path):
    study = _import_study()
    rows = [
        {
            "row_id": "toy",
            "condition": "OOD",
            "model": "PtychoPINN",
            "amplitude_mse": 0.1,
            "amplitude_psnr": 30.0,
            "amplitude_ssim": 0.9,
            "phase_mse": 0.2,
            "phase_psnr": 20.0,
            "phase_ssim": 0.8,
            "frc_input_shape": [4, 6],
            "frc_square_crop_applied": True,
            "frc_shape": [4, 4],
            "frc_crop_slices_yx": {"y": [0, 4], "x": [1, 5]},
            "frc_field_of_view_matches_core_metrics": False,
            "frc_field_of_view_policy": "post_eval_metric_shape -> frc_cutoffs_square_center_crop",
            "frc_paper_table_status": "artifact_only_due_to_frc_square_crop",
        }
    ]

    study.write_metrics_artifacts(
        output_root=tmp_path,
        rows=rows,
        manifest={"status": "metrics_complete", "rows": rows},
        metric_policy={"alignment": "coordinate"},
    )

    json_text = (tmp_path / "fig5_ood_metrics.json").read_text()
    csv_text = (tmp_path / "fig5_ood_metrics.csv").read_text()
    assert "frc_square_crop_applied" in json_text
    assert "frc_square_crop_applied" in csv_text
    assert "artifact_only_due_to_frc_square_crop" in csv_text


def test_metric_run_writes_required_artifacts_and_four_rows(tmp_path, monkeypatch):
    study = _import_study()
    fixture_repo = tmp_path / "fixture_repo"
    _write_minimal_default_inputs(fixture_repo)
    output_root = tmp_path / "run"

    def fake_align(reconstruction_image, ground_truth_image, _coords, _stitch_patch_size):
        return reconstruction_image, ground_truth_image[: reconstruction_image.shape[0], : reconstruction_image.shape[1]]

    def fake_fine(row, aligned_recon, aligned_gt, args):
        return {
            "registration_offset_yx_upsample50": [0.0, 0.0],
            "registration_offset_yx_upsample10": [0.0, 0.0],
            "registration_offset_norm_upsample50": 0.0,
            "registration_offset_norm_upsample10": 0.0,
            "registration_offset_max_abs_component_upsample50": 0.0,
            "registration_offset_max_abs_component_upsample10": 0.0,
            "diagnostic_crop_safety_status": "crop_safe",
            "offset_stability_status": "stable",
            "fine_registration_sensitivity_status": "not_scored_test_stub",
        }

    monkeypatch.setattr(study, "align_for_evaluation", fake_align)
    monkeypatch.setattr(study, "compute_fine_registration_sensitivity", fake_fine)
    monkeypatch.setattr(study, "eval_reconstruction", lambda *args, **kwargs: _dummy_eval_metrics())
    args = study.parse_args(
        [
            "--repo-root",
            str(fixture_repo),
            "--output-root",
            str(output_root),
            "--eval-offset",
            "2",
        ]
    )

    with study.acquire_output_lock(output_root, force_stale_lock=False) as lock:
        study.run_metrics(args, lock)

    for name in [
        "fig5_reference_validation.json",
        "fig5_ood_metrics_manifest.json",
        "fig5_ood_metrics.json",
        "fig5_ood_metrics.csv",
        "fig5_ood_metrics_table.tex",
        "fig5_ood_metrics_summary.md",
    ]:
        assert (output_root / name).exists()
    payload = json.loads((output_root / "fig5_ood_metrics.json").read_text())
    assert len(payload["rows"]) == 4
    manifest = json.loads((output_root / "fig5_ood_metrics_manifest.json").read_text())
    assert manifest["status"] == "metrics_complete"
    assert manifest["reference_validation_completed_before_metrics"] is True
    assert manifest["metric_evaluation_started"] is True
    assert manifest["artifact_acceptance"]["accepted_metrics_artifacts"] is True
    assert manifest["artifact_acceptance"]["paper_claims_allowed"] is True
    assert manifest["metric_policy"]["primary_registration_mode"] == "none"
    for name in [
        "metrics_json",
        "metrics_csv",
        "metrics_table_tex",
        "metrics_summary_md",
    ]:
        assert manifest["output_artifact_status"][name]["exists"] is True
        assert manifest["output_artifact_status"][name]["accepted_for_paper_claims"] is True
    for row in payload["rows"]:
        assert row["primary_registration_mode"] == "none"
        assert row["alignment_mode"] == "coordinate_align_for_evaluation"
        assert row["reference_validation_status"] == "passed"
        assert row["reference_validation_completed_before_metrics"] is True
        assert row["fine_registration_sensitivity_status"] == "not_scored_test_stub"


def test_primary_metrics_are_computed_for_all_rows_before_fine_registration_diagnostics(monkeypatch, tmp_path):
    study = _import_study()
    rows = [
        study.StudyRow(f"row_{idx}", "ID" if idx < 2 else "OOD", f"model_{idx}", Path(f"r{idx}.npz"), None)
        for idx in range(4)
    ]
    manifest_rows = [
        {
            "row_id": row.row_id,
            "alignment_mode": "coordinate_align_for_evaluation",
            "reference_validation_status": "passed",
            "reference_validation_completed_before_metrics": True,
        }
        for row in rows
    ]
    records = [
        {
            "row": row,
            "aligned_recon": np.ones((64, 64), dtype=np.complex64),
            "aligned_gt": np.ones((64, 64), dtype=np.complex64),
            "manifest_entry": manifest_entry,
        }
        for row, manifest_entry in zip(rows, manifest_rows)
    ]
    events = []

    def fake_compute(row, recon_registered, gt_registered, args):
        events.append(("primary", row.row_id))
        return {
            "row_id": row.row_id,
            "condition": row.condition,
            "model": row.model,
            "primary_registration_mode": "none",
            "alignment_mode": "coordinate_align_for_evaluation",
            "reference_validation_status": "passed",
            "reference_validation_completed_before_metrics": True,
            "amplitude_mse": 1.0,
            "amplitude_mse_unscaled": 1.0,
        }, {"metric_status": "ok"}

    def fake_fine(row, aligned_recon, aligned_gt, args):
        events.append(("fine", row.row_id))
        return {"fine_registration_sensitivity_status": "not_scored_test_stub"}

    monkeypatch.setattr(study, "_load_alignment_records", lambda args: ({"rows": manifest_rows}, records))
    monkeypatch.setattr(study, "compute_row_metrics", fake_compute)
    monkeypatch.setattr(study, "compute_fine_registration_sensitivity", fake_fine)
    monkeypatch.setattr(study, "_git_summary", lambda path: {"path": str(path), "commit": "abc123", "dirty_status": []})
    monkeypatch.setattr(study, "_environment_summary", lambda: {"python_version": "test", "packages": {}})

    args = types.SimpleNamespace(
        output_root=tmp_path,
        repo_root=Path("."),
        paper_root=Path("."),
        stitch_patch_size=20,
        upsample_factor=50,
        registration_stability_factor=10,
        border_crop=2,
        eval_offset=4,
        phase_align_method="plane",
        frc_sigma=0.0,
        ms_ssim_sigma=1.0,
    )
    lock = types.SimpleNamespace(pid=123, replaced_stale_lock=False, stale_lock_content=None)

    study.run_metrics(args, lock)

    assert events[:4] == [("primary", f"row_{idx}") for idx in range(4)]
    assert events[4:] == [("fine", f"row_{idx}") for idx in range(4)]


def test_validate_references_only_runs_alignment_without_metric_artifacts(tmp_path, monkeypatch):
    study = _import_study()
    fixture_repo = tmp_path / "fixture_repo"
    _write_minimal_default_inputs(fixture_repo)
    output_root = tmp_path / "validate"
    metric_called = False

    def fake_align(reconstruction_image, ground_truth_image, _coords, _stitch_patch_size):
        return reconstruction_image, ground_truth_image[: reconstruction_image.shape[0], : reconstruction_image.shape[1]]

    def fake_eval(*_args, **_kwargs):
        nonlocal metric_called
        metric_called = True
        return _dummy_eval_metrics()

    monkeypatch.setattr(study, "align_for_evaluation", fake_align)
    monkeypatch.setattr(study, "eval_reconstruction", fake_eval)
    args = study.parse_args(
        [
            "--repo-root",
            str(fixture_repo),
            "--output-root",
            str(output_root),
            "--validate-references-only",
        ]
    )

    with study.acquire_output_lock(output_root, force_stale_lock=False) as lock:
        exit_code = study.run_validate_references_only(args, lock)

    assert exit_code == 0
    assert metric_called is False
    assert (output_root / "fig5_reference_validation.json").exists()
    assert not (output_root / "fig5_ood_metrics.json").exists()
    assert not (output_root / "fig5_ood_metrics.csv").exists()
    manifest = json.loads((output_root / "fig5_ood_metrics_manifest.json").read_text())
    assert manifest["status"] == "reference_validation_only"
    assert manifest["metric_evaluation_started"] is False
    assert manifest["reference_validation_completed_before_metrics"] is True


def test_metric_run_archives_prior_validation_only_gate_before_invocation_overwrite(tmp_path, monkeypatch):
    study = _import_study()
    output_root = tmp_path / "run"
    output_root.mkdir()
    (output_root / "fig5_ood_metrics_manifest.json").write_text(
        json.dumps(
            {
                "status": "reference_validation_only",
                "metric_evaluation_started": False,
                "reference_validation_completed_before_metrics": True,
            }
        )
    )
    (output_root / "fig5_reference_validation.json").write_text(
        json.dumps({"status": "reference_validation_only", "metric_evaluation_started": False})
    )
    (output_root / "fig5_source_inventory.json").write_text(json.dumps({"status": "reference_validation_only"}))
    (output_root / "invocation.json").write_text('{"validate_references_only": true}\n')
    (output_root / "invocation.sh").write_text("python validate-only\n")
    captured = {}

    def fake_run_metrics(args, lock):
        captured["gate"] = args._reference_validation_only_gate

    monkeypatch.setattr(study, "run_metrics", fake_run_metrics)

    result = study.main(["--output-root", str(output_root)])

    assert result == 0
    gate_dir = output_root / "reference_validation_only_gate"
    assert captured["gate"]["preserved_before_metric_invocation"] is True
    assert Path(captured["gate"]["manifest"]).parent == gate_dir
    assert (gate_dir / "invocation.json").read_text() == '{"validate_references_only": true}\n'
    assert (gate_dir / "invocation.sh").read_text() == "python validate-only\n"
    assert json.loads((gate_dir / "fig5_reference_validation.json").read_text())[
        "metric_evaluation_started"
    ] is False


def test_full_metric_command_aborts_before_metrics_when_reference_validation_fails(tmp_path, monkeypatch):
    study = _import_study()
    fixture_repo = tmp_path / "fixture_repo"
    _write_minimal_default_inputs(fixture_repo)
    output_root = tmp_path / "run"
    metric_called = False

    def fake_align(reconstruction_image, ground_truth_image, _coords, _stitch_patch_size):
        return reconstruction_image, ground_truth_image[: reconstruction_image.shape[0], : reconstruction_image.shape[1]]

    def fake_eval(*_args, **_kwargs):
        nonlocal metric_called
        metric_called = True
        return _dummy_eval_metrics()

    # Break one panel reference while keeping shapes valid.
    panel = fixture_repo / "experiment_outputs/fly64_trained_models/recon_on_run1084_pinn/ground_truth_run1084_for_fly64trained.npz"
    np.savez(panel, ground_truth_complex=np.ones((68, 68), dtype=np.complex64) * 2)

    monkeypatch.setattr(study, "align_for_evaluation", fake_align)
    monkeypatch.setattr(study, "eval_reconstruction", fake_eval)
    args = study.parse_args(
        [
            "--repo-root",
            str(fixture_repo),
            "--output-root",
            str(output_root),
        ]
    )

    with study.acquire_output_lock(output_root, force_stale_lock=False) as lock:
        with pytest.raises(study.StopCondition, match="reference validation"):
            study.run_metrics(args, lock)

    assert metric_called is False
    assert (output_root / "fig5_reference_validation.json").exists()
    assert not (output_root / "fig5_ood_metrics.json").exists()
    manifest = json.loads((output_root / "fig5_ood_metrics_manifest.json").read_text())
    assert manifest["status"] == "reference_validation_failed"
    assert manifest["metric_evaluation_started"] is False
    assert manifest["reference_validation_completed_before_metrics"] is True


def test_attach_run_provenance_records_command_git_environment_invocation_and_outputs(tmp_path, monkeypatch):
    study = _import_study()
    output_root = tmp_path / "run"
    output_root.mkdir()
    invocation_json = output_root / "invocation.json"
    invocation_sh = output_root / "invocation.sh"
    invocation_json.write_text("{}\n")
    invocation_sh.write_text("python script\n")
    args = study.parse_args(["--output-root", str(output_root), "--paper-root", str(tmp_path / "paper")])
    raw_argv = ["--output-root", str(output_root), "--paper-root", str(tmp_path / "paper")]
    monkeypatch.setattr(study, "_git_summary", lambda path: {"path": str(path), "commit": "abc123", "dirty": [" M file.py"]})
    monkeypatch.setattr(
        study,
        "_environment_summary",
        lambda: {
            "python_executable": "/usr/bin/python",
            "python_version": "3.x",
            "packages": {"numpy": "1.0"},
        },
    )

    manifest = {"status": "stopped", "rows": []}
    updated = study.attach_run_provenance(
        manifest,
        args=args,
        raw_argv=raw_argv,
        invocation_json_path=invocation_json,
        invocation_sh_path=invocation_sh,
        invocation_artifact_kind="metric_run",
        include_paper_git=False,
    )

    assert updated is manifest
    assert updated["command_invocation"]["command"].startswith("python scripts/studies/ood_fig5_metrics.py")
    assert updated["invocation_artifacts"] == {
        "kind": "metric_run",
        "invocation_json": str(invocation_json),
        "invocation_sh": str(invocation_sh),
    }
    assert updated["source_repo_git"]["commit"] == "abc123"
    assert updated["environment"]["python_executable"] == "/usr/bin/python"
    assert updated["output_artifacts"]["manifest"] == str(output_root / "fig5_ood_metrics_manifest.json")
    assert updated["metric_policy"]["eval_offset"] == 4


def test_pivot_summary_reports_metric_evaluated_rows_and_paper_recommendation(tmp_path):
    study = _import_study()
    manifest = {
        "rows": [
            {"row_id": "id_ptychopinn", "metric_status": "ok", "registration_offset_norm": 0.1},
            {"row_id": "id_supervised_baseline", "metric_status": "ok", "registration_offset_norm": 0.7},
            {"row_id": "ood_ptychopinn", "registration_offset_norm": 26.9},
            {"row_id": "ood_supervised_baseline"},
        ]
    }

    study._write_pivot_summary(tmp_path, "registration offset too large", manifest)

    text = (tmp_path / "pivot_summary.md").read_text()
    assert "Rows inventoried/aligned: 4" in text
    assert "Rows metric-evaluated: 2" in text
    assert "id_ptychopinn" in text
    assert "ood_supervised_baseline" in text
    assert "Do not update the manuscript with Fig. 5 metric claims" in text
    assert "Rows processed: 4" not in text


def test_large_registration_offsets_suppress_only_fine_registration_not_primary_metrics(monkeypatch, tmp_path):
    study = _import_study()
    rows = [
        study.StudyRow("id_ptychopinn", "ID", "PtychoPINN", Path("id_pinn.npz"), None),
        study.StudyRow("id_supervised_baseline", "ID", "Supervised baseline", Path("id_base.npz"), None),
        study.StudyRow("ood_ptychopinn", "OOD", "PtychoPINN", Path("ood_pinn.npz"), None),
        study.StudyRow("ood_supervised_baseline", "OOD", "Supervised baseline", Path("ood_base.npz"), None),
    ]
    manifest_rows = [
        {
            "row_id": row.row_id,
            "alignment_mode": "coordinate_align_for_evaluation",
            "reference_validation_status": "passed",
            "reference_validation_completed_before_metrics": True,
        }
        for row in rows
    ]
    records = [
        {
            "row": row,
            "aligned_recon": np.ones((64, 64), dtype=np.complex64),
            "aligned_gt": np.ones((64, 64), dtype=np.complex64),
            "manifest_entry": manifest_entry,
        }
        for row, manifest_entry in zip(rows, manifest_rows)
    ]
    monkeypatch.setattr(study, "_load_alignment_records", lambda args: ({"rows": manifest_rows}, records))
    monkeypatch.setattr(study, "_git_summary", lambda path: {"path": str(path), "commit": "abc123", "dirty_status": []})
    monkeypatch.setattr(study, "_environment_summary", lambda: {"python_version": "test", "packages": {}})

    offsets = iter([(0.0, 0.0), (0.0, 0.0), (6.0, 0.0), (7.0, 0.0)])
    diagnostics_seen = []

    def fake_fine(row, aligned_recon, aligned_gt, args):
        offset_yx = next(offsets)
        diagnostics_seen.append((row.row_id, offset_yx))
        status = "scored_crop_safe_sensitivity_only" if max(map(abs, offset_yx)) <= 2.0 else "not_scored_offset_exceeds_border_crop"
        return {
            "registration_offset_yx_upsample50": [float(offset_yx[0]), float(offset_yx[1])],
            "registration_offset_yx_upsample10": [float(offset_yx[0]), float(offset_yx[1])],
            "registration_offset_norm_upsample50": float(np.linalg.norm(offset_yx)),
            "registration_offset_max_abs_component_upsample50": float(max(map(abs, offset_yx))),
            "diagnostic_crop_safety_status": "crop_safe" if status.startswith("scored") else "offset_exceeds_border_crop",
            "offset_stability_status": "stable",
            "fine_registration_sensitivity_status": status,
        }

    def fake_compute(row, recon_registered, gt_registered, args):
        return {
            "row_id": row.row_id,
            "condition": row.condition,
            "model": row.model,
            "amplitude_mse": 1.0,
            "amplitude_mse_unscaled": 1.0,
            "primary_registration_mode": "none",
            "alignment_mode": "coordinate_align_for_evaluation",
            "reference_validation_status": "passed",
            "reference_validation_completed_before_metrics": True,
        }, {"metric_status": "ok", "primary_registration_mode": "none"}

    monkeypatch.setattr(study, "compute_fine_registration_sensitivity", fake_fine)
    monkeypatch.setattr(study, "compute_row_metrics", fake_compute)
    args = types.SimpleNamespace(
        output_root=tmp_path,
        repo_root=Path("."),
        paper_root=Path("."),
        stitch_patch_size=20,
        upsample_factor=50,
        registration_stability_factor=10,
        border_crop=2,
        eval_offset=4,
        phase_align_method="plane",
        frc_sigma=0.0,
        ms_ssim_sigma=1.0,
    )
    lock = types.SimpleNamespace(pid=123, replaced_stale_lock=False, stale_lock_content=None)

    study.run_metrics(args, lock)

    assert diagnostics_seen == [
        ("id_ptychopinn", (0.0, 0.0)),
        ("id_supervised_baseline", (0.0, 0.0)),
        ("ood_ptychopinn", (6.0, 0.0)),
        ("ood_supervised_baseline", (7.0, 0.0)),
    ]
    persisted = json.loads((tmp_path / "fig5_ood_metrics_manifest.json").read_text())
    assert persisted["status"] == "metrics_complete"
    assert persisted["rows"][2]["metric_status"] == "ok"
    assert persisted["rows"][2]["fine_registration_sensitivity_status"] == "not_scored_offset_exceeds_border_crop"
    assert persisted["rows"][3]["metric_status"] == "ok"
    assert persisted["rows"][3]["registration_offset_max_abs_component_upsample50"] == 7.0
    payload = json.loads((tmp_path / "fig5_ood_metrics.json").read_text())
    assert len(payload["rows"]) == 4
    assert all(row["primary_registration_mode"] == "none" for row in payload["rows"])


def test_emit_paper_assets_preserves_metric_run_invocation_artifacts_and_writes_paper_assets(tmp_path):
    study = _import_study()
    output_root = tmp_path / "run"
    output_root.mkdir()
    paper_root = tmp_path / "paper"
    (paper_root / "data").mkdir(parents=True)
    (paper_root / "tables").mkdir(parents=True)
    original_invocation = '{"metric_run": true}\n'
    original_shell = "python metric-run\n"
    (output_root / "invocation.json").write_text(original_invocation)
    (output_root / "invocation.sh").write_text(original_shell)
    metrics_payload = {
        "metric_policy": {
            "primary_registration_mode": "none",
            "evaluation_region": "heldout_test_half",
            "eval": "offset",
            "phase": "plane",
            "background": "none",
            "unscaled": True,
        },
        "rows": [
            {
                "row_id": "id_ptychopinn",
                "condition": "ID",
                "model": "PtychoPINN",
                "evaluation_region": "heldout_test_half",
                "heldout_only": True,
                "split_contract": {
                    "status": "ok",
                    "policy": "paper_top_train_bottom_test_by_high_y",
                    "nonoverlap_level": "coordinate_indices",
                    "train_coordinate_count": 512,
                    "eval_coordinate_count": 543,
                    "train_contiguous": True,
                    "eval_contiguous": True,
                    "coordinate_index_overlap_count": 0,
                    "object_footprint_overlap_pixel_count": 10,
                    "object_footprint_overlap_policy": "recorded_not_enforced_coordinate_split_only",
                    "metric_region_matches_displayed_panel": False,
                    "panel_metric_region_mismatch_policy": "internal_provenance_only",
                },
                "primary_registration_mode": "none",
                "amplitude_mse": 0.1,
                "phase_ssim": 0.8,
            }
        ],
    }
    (output_root / "fig5_ood_metrics.json").write_text(json.dumps(metrics_payload))
    (output_root / "fig5_ood_metrics.csv").write_text("row_id,amplitude_mse\nid_ptychopinn,0.1\n")
    (output_root / "fig5_ood_metrics_table.tex").write_text("% table\n")
    (output_root / "fig5_ood_metrics_summary.md").write_text("# summary\n")
    (output_root / "fig5_ood_metrics_manifest.json").write_text(
        json.dumps(
            {
                "status": "metrics_complete",
                "evaluation_region": "heldout_test_half",
                "artifact_acceptance": {"accepted_metrics_artifacts": True},
                "output_artifacts": {"metrics_json": str(output_root / "fig5_ood_metrics.json")},
                "rows": [
                    {
                        "row_id": "id_ptychopinn",
                        "evaluation_region": "heldout_test_half",
                        "heldout_only": True,
                        "split_contract": {
                            "status": "ok",
                            "policy": "paper_top_train_bottom_test_by_high_y",
                            "nonoverlap_level": "coordinate_indices",
                            "train_coordinate_count": 512,
                            "eval_coordinate_count": 543,
                            "train_contiguous": True,
                            "eval_contiguous": True,
                            "coordinate_index_overlap_count": 0,
                            "object_footprint_overlap_pixel_count": 10,
                            "object_footprint_overlap_policy": "recorded_not_enforced_coordinate_split_only",
                            "metric_region_matches_displayed_panel": False,
                            "panel_metric_region_mismatch_policy": "internal_provenance_only",
                        },
                    }
                ],
            }
        )
    )

    result = study.main(
        [
            "--output-root",
            str(output_root),
            "--paper-root",
            str(paper_root),
            "--emit-paper-assets",
        ]
    )

    assert result == 0
    assert (output_root / "invocation.json").read_text() == original_invocation
    assert (output_root / "invocation.sh").read_text() == original_shell
    assert (output_root / "paper_asset_emission_invocation" / "invocation.json").exists()
    assert (output_root / "paper_asset_emission_invocation" / "invocation.sh").exists()
    assert (paper_root / "data" / "fig5_ood_metrics.json").exists()
    assert (paper_root / "tables" / "fig5_ood_metrics.tex").read_text() == "% table\n"
    paper_payload = json.loads((paper_root / "data" / "fig5_ood_metrics.json").read_text())
    assert paper_payload["source_run_root"] == str(output_root)
    assert paper_payload["metric_policy"]["primary_registration_mode"] == "none"
    assert paper_payload["metric_policy"]["evaluation_region"] == "heldout_test_half"
    assert paper_payload["rows"][0]["split_contract"]["status"] == "ok"


def test_emit_paper_assets_requires_split_contract_ok(tmp_path):
    study = _import_study()
    output_root = tmp_path / "run"
    output_root.mkdir()
    paper_root = tmp_path / "paper"
    metrics_payload = {
        "metric_policy": {"primary_registration_mode": "none", "evaluation_region": "heldout_test_half"},
        "rows": [
            {
                "row_id": "id_ptychopinn",
                "evaluation_region": "heldout_test_half",
                "heldout_only": True,
            }
        ],
    }
    (output_root / "fig5_ood_metrics.json").write_text(json.dumps(metrics_payload))
    (output_root / "fig5_ood_metrics_table.tex").write_text("% table\n")
    (output_root / "fig5_ood_metrics_manifest.json").write_text(
        json.dumps(
            {
                "status": "metrics_complete",
                "evaluation_region": "heldout_test_half",
                "artifact_acceptance": {"accepted_metrics_artifacts": True},
                "rows": [
                    {
                        "row_id": "id_ptychopinn",
                        "evaluation_region": "heldout_test_half",
                        "heldout_only": True,
                    }
                ],
            }
        )
    )
    args = types.SimpleNamespace(output_root=output_root, paper_root=paper_root)
    lock = types.SimpleNamespace(pid=123)

    with pytest.raises(study.StopCondition, match="split contract"):
        study.emit_paper_assets(args, lock)


def test_emit_paper_assets_refuses_all_scan_metrics(tmp_path):
    study = _import_study()
    output_root = tmp_path / "run"
    output_root.mkdir()
    paper_root = tmp_path / "paper"
    (output_root / "fig5_ood_metrics.json").write_text(
        json.dumps(
            {
                "metric_policy": {"primary_registration_mode": "none", "evaluation_region": "all_scan"},
                "rows": [{"row_id": "id_ptychopinn", "evaluation_region": "all_scan"}],
            }
        )
    )
    (output_root / "fig5_ood_metrics_table.tex").write_text("% table\n")
    (output_root / "fig5_ood_metrics_manifest.json").write_text(
        json.dumps(
            {
                "status": "metrics_complete",
                "evaluation_region": "all_scan",
                "artifact_acceptance": {"accepted_metrics_artifacts": True},
                "rows": [{"row_id": "id_ptychopinn", "evaluation_region": "all_scan"}],
            }
        )
    )
    args = types.SimpleNamespace(output_root=output_root, paper_root=paper_root)
    lock = types.SimpleNamespace(pid=123)

    with pytest.raises(study.StopCondition, match="held-out"):
        study.emit_paper_assets(args, lock)
