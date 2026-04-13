import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest


def _toy_probe():
    yy, xx = np.indices((8, 8))
    rr = np.sqrt((yy - 3.5) ** 2 + (xx - 3.5) ** 2)
    amp = np.clip(1.0 - 0.23 * rr, 0.0, None)
    phase = 0.05 * (yy - 3.5) + 0.03 * (xx - 3.5)
    return (amp * np.exp(1j * phase)).astype(np.complex64)


def _toy_complex(size=8):
    yy, xx = np.indices((size, size))
    amp = 1.0 + 0.05 * yy + 0.03 * xx
    phase = 0.1 * yy - 0.07 * xx
    return (amp * np.exp(1j * phase)).astype(np.complex64)


def _toy_split_bundle():
    probe = _toy_probe()
    x = np.ones((2, 8, 8, 1), dtype=np.float32)
    y_i = (np.abs(probe)[None, ..., None] * np.ones((2, 1, 1, 1), dtype=np.float32)).astype(np.float32)
    y_phi = np.zeros((2, 8, 8, 1), dtype=np.float32)
    coords = np.zeros((2, 1, 2, 1), dtype=np.float32)
    yy_full = np.ones((8, 8), dtype=np.complex64)
    split = {
        "X": x,
        "Y_I": y_i,
        "Y_phi": y_phi,
        "coords_nominal": coords,
        "coords_true": coords + 1,
        "coords_offsets": coords + 2,
        "YY_full": yy_full,
    }
    return {
        "train": dict(split),
        "test": {
            **split,
            "YY_ground_truth": yy_full,
            "norm_Y_I": 3.0,
        },
    }


def test_make_probe_support_rejects_empty_and_full_masks():
    from scripts.reconstruction.hio_cdi_benchmark import make_probe_support

    zero_probe = np.zeros((8, 8), dtype=np.complex64)
    with pytest.raises(ValueError, match="zero-amplitude"):
        make_probe_support(zero_probe, threshold=0.05)

    full_probe = np.ones((8, 8), dtype=np.complex64)
    with pytest.raises(ValueError, match="full-frame"):
        make_probe_support(full_probe, threshold=0.0)


def test_make_probe_support_records_primary_threshold_and_grid():
    from scripts.reconstruction.hio_cdi_benchmark import make_probe_support

    support, record = make_probe_support(
        _toy_probe(),
        threshold=0.05,
        threshold_grid=[0.01, 0.05, 0.10],
    )

    assert support.dtype == np.bool_
    assert support.shape == (8, 8)
    assert 0 < record["support_pixel_count"] < 64
    assert record["support_fraction"] == pytest.approx(record["support_pixel_count"] / 64)
    assert record["support_threshold"] == 0.05
    assert record["threshold_grid"] == [0.01, 0.05, 0.1]
    assert record["selection_policy"] == "pre_registered_primary_not_metric_selected"


def test_forward_amplitude_uses_normalized_fftshift_convention():
    from scripts.reconstruction.hio_cdi_benchmark import forward_amplitude

    psi = _toy_complex()
    expected = np.abs(np.fft.fftshift(np.fft.fft2(psi)) / np.sqrt(psi.shape[0] * psi.shape[1]))

    assert np.allclose(forward_amplitude(psi), expected)


def test_project_fourier_magnitude_preserves_target_amplitude_and_phase():
    from scripts.reconstruction.hio_cdi_benchmark import (
        forward_amplitude,
        project_fourier_magnitude,
    )

    psi = _toy_complex()
    target = np.ones((8, 8), dtype=np.float32) * 3.0
    projected = project_fourier_magnitude(psi, target)

    assert np.allclose(forward_amplitude(projected), target, atol=1e-6)

    original_phase = np.angle(np.fft.fftshift(np.fft.fft2(psi)))
    projected_phase = np.angle(np.fft.fftshift(np.fft.fft2(projected)))
    assert np.allclose(projected_phase, original_phase, atol=1e-6)


def test_project_fourier_magnitude_rejects_shape_mismatch_and_nonfinite():
    from scripts.reconstruction.hio_cdi_benchmark import project_fourier_magnitude

    psi = _toy_complex()
    with pytest.raises(ValueError, match="shape"):
        project_fourier_magnitude(psi, np.ones((4, 4), dtype=np.float32))
    target = np.ones((8, 8), dtype=np.float32)
    target[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        project_fourier_magnitude(psi, target)


def test_hio_and_er_updates_use_support_without_ground_truth():
    from scripts.reconstruction.hio_cdi_benchmark import (
        er_cleanup,
        hio_update,
        project_fourier_magnitude,
    )

    previous = _toy_complex()
    target = np.abs(np.fft.fftshift(np.fft.fft2(previous)) / 8.0) + 0.1
    support = np.zeros((8, 8), dtype=bool)
    support[2:6, 2:6] = True

    projected = project_fourier_magnitude(previous, target)
    hio = hio_update(previous, target, support, beta=0.9)
    er = er_cleanup(previous, target, support)

    assert np.allclose(hio[support], projected[support])
    assert np.allclose(hio[~support], previous[~support] - 0.9 * projected[~support])
    assert np.allclose(er[support], projected[support])
    assert np.all(er[~support] == 0)


def test_residual_and_restart_selection_are_ground_truth_free():
    from scripts.reconstruction.hio_cdi_benchmark import (
        RestartResult,
        fourier_residual,
        select_restart_by_residual,
    )

    psi = _toy_complex()
    target = np.ones((8, 8), dtype=np.float32)
    residual = fourier_residual(psi, target)
    expected = np.linalg.norm(
        np.abs(np.fft.fftshift(np.fft.fft2(psi)) / 8.0) - target
    ) / np.linalg.norm(target)
    assert residual == pytest.approx(expected)

    results = [
        RestartResult(seed=7, psi=np.ones((2, 2)), final_residual=0.2, residual_curve=[0.3, 0.2]),
        RestartResult(seed=3, psi=np.ones((2, 2)), final_residual=0.1, residual_curve=[0.2, 0.1]),
        RestartResult(seed=2, psi=np.ones((2, 2)), final_residual=0.1, residual_curve=[0.4, 0.1]),
    ]
    selected = select_restart_by_residual(results)

    assert selected.seed == 2
    assert selected.residual_curve == [0.4, 0.1]


def test_fourier_residual_uses_contract_denominator_floor_for_zero_target():
    from scripts.reconstruction.hio_cdi_benchmark import fourier_residual, forward_amplitude

    psi = np.ones((2, 2), dtype=np.complex64)
    target = np.zeros((2, 2), dtype=np.float32)

    residual = fourier_residual(psi, target)
    expected = np.linalg.norm(forward_amplitude(psi) - target) / 1e-12

    assert residual == pytest.approx(expected)


def test_run_restarts_retains_curves_and_recovers_object_patch():
    from scripts.reconstruction.hio_cdi_benchmark import (
        recover_object_patch,
        run_restarts,
    )

    probe = _toy_probe()
    support = np.abs(probe) >= 0.05 * np.abs(probe).max()
    target = np.ones((8, 8), dtype=np.float32)

    result = run_restarts(
        target,
        support,
        seeds=[11, 12, 13],
        beta=0.9,
        hio_iters=2,
        er_iters=2,
        residual_period=1,
    )

    assert result.selected.seed in {11, 12, 13}
    assert len(result.restarts) == 3
    assert all(restart.residual_curve for restart in result.restarts)

    patch = recover_object_patch(result.selected.psi, probe, support, epsilon_ratio=1e-6)
    assert patch.shape == probe.shape
    assert np.all(np.isfinite(patch))
    assert np.all(patch[~support] == 0)


def test_run_restarts_derives_patch_specific_seeds_and_uses_default_residual_cadence():
    from scripts.reconstruction.hio_cdi_benchmark import run_restarts

    support = np.zeros((8, 8), dtype=bool)
    support[2:6, 2:6] = True
    target = np.ones((8, 8), dtype=np.float32)

    patch_three = run_restarts(
        target,
        support,
        seeds=[2026041201, 2026041202],
        beta=0.9,
        hio_iters=20,
        er_iters=10,
        condition_id="gs1_custom",
        patch_index=3,
    )
    patch_four = run_restarts(
        target,
        support,
        seeds=[2026041201, 2026041202],
        beta=0.9,
        hio_iters=20,
        er_iters=10,
        condition_id="gs1_custom",
        patch_index=4,
    )

    seeds_three = [restart.seed for restart in patch_three.restarts]
    seeds_four = [restart.seed for restart in patch_four.restarts]
    assert seeds_three != [2026041201, 2026041202]
    assert seeds_three != seeds_four
    assert [restart.base_seed for restart in patch_three.restarts] == [2026041201, 2026041202]
    assert [restart.restart_index for restart in patch_three.restarts] == [0, 1]
    assert all(len(restart.residual_curve) == 4 for restart in patch_three.restarts)


def test_restart_payload_records_all_curves_not_only_selected():
    from scripts.reconstruction.hio_cdi_benchmark import RestartResult, RestartRun, _restart_records

    run = RestartRun(
        restarts=[
            RestartResult(
                seed=11,
                psi=np.ones((2, 2)),
                final_residual=0.2,
                residual_curve=[0.3, 0.2],
                base_seed=101,
                restart_index=0,
            ),
            RestartResult(
                seed=12,
                psi=np.ones((2, 2)),
                final_residual=0.1,
                residual_curve=[0.2, 0.1],
                base_seed=102,
                restart_index=1,
            ),
        ],
        selected=RestartResult(
            seed=12,
            psi=np.ones((2, 2)),
            final_residual=0.1,
            residual_curve=[0.2, 0.1],
            base_seed=102,
            restart_index=1,
        ),
    )

    records = _restart_records(run)

    assert [record["seed"] for record in records] == [11, 12]
    assert all("residual_curve" in record for record in records)
    assert all("metrics" in record for record in records)
    assert records[0]["selected"] is False
    assert records[1]["selected"] is True


def test_forward_self_consistency_uses_known_exit_wave_not_residual_bookkeeping():
    from scripts.reconstruction.hio_cdi_benchmark import (
        check_forward_amplitude_self_consistency,
        forward_amplitude,
    )

    probe = _toy_probe()
    yy, xx = np.indices(probe.shape)
    object_amp = 1.0 + 0.01 * yy + 0.02 * xx
    object_phase = 0.03 * yy - 0.02 * xx
    stored_label_amp = object_amp * np.abs(probe)
    target = forward_amplitude(stored_label_amp * np.exp(1j * object_phase) * np.exp(1j * np.angle(probe)))

    check = check_forward_amplitude_self_consistency(target, stored_label_amp, object_phase, probe)

    assert check["status"] == "ok"
    assert check["normalized_residual"] == pytest.approx(0.0, abs=1e-7)
    assert check["exit_wave_source"] == "stored_label_amplitude_plus_object_phase_plus_probe_phase"


def test_ground_truth_object_patch_removes_probe_amplitude_without_probe_phase():
    from scripts.reconstruction.hio_cdi_benchmark import object_patch_from_simulated_labels

    probe = _toy_probe()
    yy, xx = np.indices(probe.shape)
    object_amp = 1.0 + 0.01 * yy + 0.02 * xx
    object_phase = 0.03 * yy - 0.02 * xx
    stored_label_amp = object_amp * np.abs(probe)

    recovered = object_patch_from_simulated_labels(stored_label_amp, object_phase, probe)

    valid = np.abs(probe) >= 1e-6 * np.abs(probe).max()
    assert np.allclose(np.abs(recovered)[valid], object_amp[valid], atol=1e-5)
    assert np.allclose(np.angle(recovered)[valid], object_phase[valid], atol=1e-6)
    assert np.all(recovered[~valid] == 0)


def test_ambiguity_policy_forbids_oracle_alignment_for_main_row():
    from scripts.reconstruction.hio_cdi_benchmark import build_ambiguity_policy

    policy = build_ambiguity_policy(oracle_diagnostic=False)

    assert policy["row_type"] == "main"
    assert policy["ground_truth_shift_alignment"] is False
    assert policy["twin_selection_by_metric"] is False
    assert policy["phase_sign_selection_by_metric"] is False

    with pytest.raises(ValueError, match="separate output label"):
        build_ambiguity_policy(oracle_diagnostic=True, output_label="primary")

    oracle = build_ambiguity_policy(oracle_diagnostic=True, output_label="oracle_shift")
    assert oracle["row_type"] == "oracle_diagnostic"


def test_manifest_writers_and_duplicate_output_root_refusal(tmp_path):
    from scripts.reconstruction.hio_cdi_benchmark import (
        assert_metric_gates_allow_metrics,
        refuse_duplicate_output_root,
        write_benchmark_manifest,
        write_data_identity_manifest,
        write_metric_contract_manifest,
        write_solver_manifest,
    )

    out = tmp_path / "run"
    out.mkdir()
    (out / "manifest.json").write_text("{}")
    with pytest.raises(FileExistsError, match="already contains benchmark artifacts"):
        refuse_duplicate_output_root(out, force=False)

    solver = write_solver_manifest(out, run_id="unit", selected_solver="study_local_hio_er")
    data = write_data_identity_manifest(out, branch="frozen-artifact", artifact_paths=[])
    metric = write_metric_contract_manifest(out, mode="direct-stitch")
    with pytest.raises(RuntimeError, match="Data identity gate blocked"):
        assert_metric_gates_allow_metrics(data, metric)

    manifest = write_benchmark_manifest(
        out,
        run_id="unit",
        solver_manifest=solver,
        data_identity_manifest=data,
        metric_contract_manifest=metric,
        preflight_only=True,
    )

    for path in [solver, data, metric, manifest]:
        assert path.exists()
        payload = json.loads(path.read_text())
        assert isinstance(payload, dict)
    assert json.loads(manifest.read_text())["preflight_only"] is True


def test_same_split_branch_blocks_metrics_until_generated_bundle_is_attached(tmp_path):
    from scripts.reconstruction.hio_cdi_benchmark import (
        assert_metric_gates_allow_metrics,
        write_data_identity_manifest,
        write_metric_contract_manifest,
    )

    out = tmp_path / "same_split"
    data = write_data_identity_manifest(out, branch="same-split-rerun", artifact_paths=[])
    metric = write_metric_contract_manifest(out, mode="direct-stitch")

    data_payload = json.loads(data.read_text())

    with pytest.raises(RuntimeError, match="bundle_not_frozen"):
        assert_metric_gates_allow_metrics(data, metric)

    assert data_payload["metric_inspection_allowed"] is False
    assert data_payload["old_table2_same_data_comparator_allowed"] is False
    assert data_payload["old_table2_value_policy"] == "historical_context_only"
    assert data_payload["decision"] == "same_split_rerun_bundle_not_frozen"


def test_study_local_seeded_same_split_metric_run_is_blocked(tmp_path):
    from scripts.reconstruction.hio_cdi_benchmark import (
        assert_metric_gates_allow_metrics,
        write_data_identity_manifest,
        write_metric_contract_manifest,
    )

    data = write_data_identity_manifest(
        tmp_path,
        branch="same-split-rerun",
        artifact_paths=[],
        data_generation_control="study-local-seeded",
    )
    metric = write_metric_contract_manifest(tmp_path, mode="direct-stitch")

    with pytest.raises(RuntimeError, match="study-local-seeded"):
        assert_metric_gates_allow_metrics(data, metric)

    payload = json.loads(data.read_text())
    assert payload["metric_inspection_allowed"] is False
    assert payload["data_generation_control_status"] == "unimplemented_for_metric_runs"


def test_same_split_bundle_persistence_records_npzs_and_key_checksums(monkeypatch, tmp_path):
    from ptycho.config.config import ModelConfig, TrainingConfig
    from ptycho.workflows.grid_lines_workflow import GridLinesConfig
    from scripts.reconstruction.hio_cdi_benchmark import (
        persist_same_split_data_bundle,
        update_data_identity_manifest_with_generated_bundle,
        write_data_identity_manifest,
    )

    monkeypatch.setenv("PTYCHO_DISABLE_MEMOIZE", "1")
    cfg = GridLinesConfig(
        N=8,
        gridsize=1,
        output_dir=tmp_path,
        probe_npz=tmp_path / "probe.npz",
        probe_source="custom",
    )
    config = TrainingConfig(
        model=ModelConfig(N=8, gridsize=1, object_big=False),
        nphotons=1e9,
        nepochs=1,
        batch_size=1,
        nll_weight=0.0,
        mae_weight=1.0,
        realspace_weight=0.0,
    )
    data_identity = write_data_identity_manifest(
        tmp_path,
        branch="same-split-rerun",
        artifact_paths=[],
        data_generation_control="loader-compatible",
    )

    bundle = persist_same_split_data_bundle(
        output_root=tmp_path,
        run_id="unit",
        cfg=cfg,
        data=_toy_split_bundle(),
        config=config,
        probe=_toy_probe(),
        probe_transform_pipeline="smooth:0.5|pad:8",
        probe_transform_steps=[{"op": "smooth_complex", "sigma": 0.5}, {"op": "pad_complex", "target_N": 8}],
        data_generation_control="loader-compatible",
    )
    update_data_identity_manifest_with_generated_bundle(data_identity, bundle)

    assert Path(bundle["train_npz"]).exists()
    assert Path(bundle["test_npz"]).exists()
    assert Path(bundle["data_generation_manifest"]).exists()
    assert Path(bundle["data_bundle_manifest"]).exists()
    assert Path(bundle["probe_transform_manifest"]).exists()

    bundle_payload = json.loads(Path(bundle["data_bundle_manifest"]).read_text())
    canonical = {
        (record["split"], record["canonical_key"])
        for record in bundle_payload["key_records"]
        if record["exists"]
    }
    assert ("train", "X") in canonical
    assert ("train", "probeGuess") in canonical
    assert ("test", "YY_ground_truth") in canonical
    assert ("test", "norm_Y_I") in canonical
    assert all(record["sha256"] for record in bundle_payload["key_records"] if record["exists"])

    identity_payload = json.loads(data_identity.read_text())
    assert identity_payload["generated_data_bundle"]["train_npz"] == bundle["train_npz"]
    assert identity_payload["generated_data_bundle"]["memoization"]["cache_mode"] == "disabled_fresh_generation"
    assert identity_payload["metric_inspection_allowed"] is True


def test_pinn_randomness_manifest_records_seed_policy_and_metric_contract_checksum(tmp_path):
    from scripts.reconstruction.hio_cdi_benchmark import (
        write_metric_contract_manifest,
        write_pinn_randomness_manifest,
    )

    metric = write_metric_contract_manifest(tmp_path, mode="direct-stitch", run_id="unit")
    path = write_pinn_randomness_manifest(
        tmp_path,
        run_id="unit",
        data_generation_control="loader-compatible",
        metric_contract_manifest=metric,
        data_bundle_manifest=tmp_path / "data_bundle_manifest.json",
    )

    payload = json.loads(path.read_text())
    assert payload["primary_training_seed"] == 2026041211
    assert payload["stochastic_fallback_seeds"] == [2026041211, 2026041212, 2026041213]
    assert payload["model_construction_started"] is False
    assert payload["training_started"] is False
    assert payload["metric_contract_sha256"]
    assert payload["loader_compatible_data_seeds"] == {"train": 1, "test": 2}


def test_pinn_comparator_manifest_update_records_training_evidence(tmp_path):
    from scripts.reconstruction.hio_cdi_benchmark import (
        _write_json,
        update_pinn_randomness_manifest_with_comparator,
        write_metric_contract_manifest,
        write_pinn_randomness_manifest,
    )

    metric_contract = write_metric_contract_manifest(tmp_path, mode="direct-stitch", run_id="unit")
    randomness_manifest = write_pinn_randomness_manifest(
        tmp_path,
        run_id="unit",
        data_generation_control="loader-compatible",
        metric_contract_manifest=metric_contract,
    )
    metrics_path = _write_json(
        tmp_path / "pinn_comparator_metrics.json",
        {
            "metrics": {
                "ssim": [0.9, 0.1],
                "psnr": [68.0, 12.0],
            }
        },
    )

    update_pinn_randomness_manifest_with_comparator(
        randomness_manifest,
        training_seed=2026041211,
        initial_weight_checksums={"aggregate_sha256": "initial", "n_weight_arrays": 2},
        final_weight_checksums={"aggregate_sha256": "final", "n_weight_arrays": 2},
        training_history={"loss": [1.0, 0.5]},
        seed_api_status={"tf_set_random_seed": "ok"},
        determinism_status="validated_single_seed_controlled_shuffle",
        model_artifacts={"model_archive": {"path": "pinn/wts.h5.zip", "exists": True}},
        metrics_path=metrics_path,
        comparison_summary={
            "historical_amp_ssim": 0.9044216561120993,
            "amp_ssim_delta": -0.004421656112099313,
            "within_table2_tolerance": True,
        },
    )

    payload = json.loads(randomness_manifest.read_text())
    assert payload["pinn_comparator_status"] == "completed_primary_seed"
    assert payload["model_construction_started"] is True
    assert payload["training_started"] is True
    assert payload["training_completed"] is True
    assert payload["determinism_status"] == "validated_single_seed_controlled_shuffle"
    assert payload["initial_model_weight_checksums"]["aggregate_sha256"] == "initial"
    assert payload["final_model_or_checkpoint_checksums"]["aggregate_sha256"] == "final"
    assert payload["training_history"] == {"loss": [1.0, 0.5]}
    assert payload["metric_output"]["path"] == str(metrics_path)
    assert payload["same_split_historical_comparison"]["within_table2_tolerance"] is True


def test_pinn_comparator_metrics_payload_records_historical_tolerance(tmp_path):
    from scripts.reconstruction.hio_cdi_benchmark import build_pinn_comparator_metrics_payload

    payload = build_pinn_comparator_metrics_payload(
        run_id="unit",
        row_label="pinn_same_split_gs1_custom",
        metrics_json={"ssim": [0.9, 0.0], "psnr": [67.0, 0.0]},
        data_bundle_manifest=tmp_path / "data_bundle_manifest.json",
        randomness_manifest=tmp_path / "pinn_randomness_manifest.json",
        recon_path=tmp_path / "recons" / "pinn_same_split_gs1_custom" / "recon.npz",
        training_seed=2026041211,
        training_history={"loss": [1.0]},
        timing_seconds=12.5,
    )

    assert payload["comparator"] == "ptychopinn_same_split"
    assert payload["row_label"] == "pinn_same_split_gs1_custom"
    assert payload["training_seed"] == 2026041211
    assert payload["historical_table2_context"]["gs1_custom_amp_ssim"] == 0.9044216561120993
    assert payload["historical_table2_comparison"]["amp_ssim_delta"] == pytest.approx(
        0.9 - 0.9044216561120993
    )
    assert payload["historical_table2_comparison"]["amp_psnr_delta"] == pytest.approx(67.0 - 68.8864772792175)
    assert payload["historical_table2_comparison"]["within_table2_tolerance"] is True
    assert payload["metric_policy"] == "fresh_same_split_rerun_historical_context_only"


def test_hio_metric_evaluation_uses_stored_test_split_ground_truth(monkeypatch, tmp_path):
    import ptycho.evaluation as evaluation
    import ptycho.workflows.grid_lines_workflow as grid_lines
    import scripts.reconstruction.hio_cdi_benchmark as hio

    stored_ground_truth = (np.ones((8, 8, 1), dtype=np.complex64) * (7 + 0j))
    data = _toy_split_bundle()
    data["test"]["YY_ground_truth"] = stored_ground_truth
    data["test"]["X"] = np.ones((2, 8, 8, 1), dtype=np.float32)

    restart = hio.RestartResult(
        seed=11,
        psi=np.ones((8, 8), dtype=np.complex64),
        final_residual=0.1,
        residual_curve=[0.2, 0.1],
        base_seed=2026041201,
        restart_index=0,
    )
    restart_run = hio.RestartRun(restarts=[restart], selected=restart)
    monkeypatch.setattr(hio, "run_restarts", lambda *args, **kwargs: restart_run)
    monkeypatch.setattr(
        hio,
        "recover_object_patch",
        lambda psi, probe, support, epsilon_ratio: np.ones((8, 8), dtype=np.complex64),
    )
    monkeypatch.setattr(
        hio,
        "check_forward_amplitude_self_consistency",
        lambda *args, **kwargs: {
            "status": "ok",
            "normalized_residual": 0.0,
            "tolerance": 5e-3,
        },
    )
    monkeypatch.setattr(grid_lines, "stitch_predictions", lambda predictions, norm_y_i, part: predictions)
    monkeypatch.setattr(grid_lines, "save_recon_artifact", lambda output_root, label, stitched: tmp_path / "recon.npz")

    calls = {}

    def fake_eval_reconstruction(stitched_obj, ground_truth_obj, **kwargs):
        calls["ground_truth_obj"] = ground_truth_obj
        return {"ssim": (1.0, 1.0), "psnr": (1.0, 1.0), "frc": ([], [])}

    monkeypatch.setattr(evaluation, "eval_reconstruction", fake_eval_reconstruction)

    hio.run_smoke_benchmark(
        tmp_path,
        SimpleNamespace(
            run_id="unit",
            probe_source="custom",
            primary_support_threshold=0.05,
            restart_seeds=[2026041201],
            beta=0.9,
            hio_iters=2,
            er_iters=1,
            residual_period=1,
            max_test_frames=0,
            smoke=False,
        ),
        _toy_probe(),
        np.ones((8, 8), dtype=bool),
        {"support_threshold": 0.05},
        cfg=SimpleNamespace(nimgs_test=2),
        data=data,
    )

    assert np.array_equal(calls["ground_truth_obj"], stored_ground_truth)


def test_cli_parses_pinn_comparator_flags():
    from scripts.reconstruction.hio_cdi_benchmark import parse_args

    args = parse_args(
        [
            "--output-root",
            "out",
            "--run-id",
            "unit",
            "--probe-npz",
            "datasets/Run1084_recon3_postPC_shrunk_3.npz",
            "--probe-source",
            "custom",
            "--probe-scale-mode",
            "pad_preserve",
            "--probe-smoothing-sigma",
            "0.5",
            "--support-thresholds",
            "0.05",
            "--primary-support-threshold",
            "0.05",
            "--restart-seeds",
            "2026041201",
            "2026041202",
            "2026041203",
            "--data-identity-branch",
            "same-split-rerun",
            "--metric-contract-mode",
            "direct-stitch",
            "--run-pinn-comparator",
            "--pinn-comparator-only",
            "--pinn-training-seed",
            "2026041211",
            "--pinn-comparator-epochs",
            "60",
        ]
    )

    assert args.run_pinn_comparator is True
    assert args.pinn_comparator_only is True
    assert args.pinn_training_seed == 2026041211
    assert args.pinn_comparator_epochs == 60


def test_seed_hook_runs_before_same_split_data_generation_when_tf_deterministic(monkeypatch):
    import scripts.reconstruction.hio_cdi_benchmark as hio

    calls = []
    monkeypatch.setenv("TF_DETERMINISTIC_OPS", "1")
    monkeypatch.setattr(
        hio,
        "_configure_pinn_seed",
        lambda seed: calls.append(seed) or {"training_seed": seed},
    )

    status = hio.configure_seed_before_generated_data(
        SimpleNamespace(data_identity_branch="same-split-rerun", pinn_training_seed=2026041211)
    )

    assert calls == [2026041211]
    assert status == {"training_seed": 2026041211}


def test_metric_contract_manifest_records_note_decisions_evidence_and_deviation(tmp_path):
    from scripts.reconstruction.hio_cdi_benchmark import write_metric_contract_manifest

    path = write_metric_contract_manifest(tmp_path, mode="direct-stitch", run_id="unit")
    payload = json.loads(path.read_text())

    assert payload["table2_compatibility"] == "fresh_same_split_direct_stitch_not_historical_table2"
    assert payload["deviation"]["table2_compatible"] is False
    assert payload["deviation"]["result_role"] == "fresh_same_split_exploratory_or_rerun_comparator"
    decisions = {item["id"]: item for item in payload["paper_side_note_decisions"]}
    assert decisions["paper_json_align_for_evaluation"]["decision"] == "unresolved"
    assert decisions["table_script_subsample_1000_seed_7"]["decision"] == "unresolved"
    assert all(item["evidence"]["path"] for item in decisions.values())
    assert all(item["evidence"]["line_reference"] for item in decisions.values())
    assert payload["metric_subset_policy"]["selected_indices_checksum"] is None
    assert payload["metric_subset_policy"]["reason"] == "full generated smoke/test subset, not paper nsamples=1000 subsample"


def test_solver_manifest_has_a2_provenance_fields(tmp_path):
    from scripts.reconstruction.hio_cdi_benchmark import write_solver_manifest

    path = write_solver_manifest(tmp_path, run_id="unit", selected_solver="study_local_hio_er")
    payload = json.loads(path.read_text())

    assert payload["search_date"]
    assert payload["searched_sources"] == ["repo", "environment", "PyPI", "GitHub", "web"]
    assert payload["selected_solver"] == "study_local_hio_er"
    for candidate in payload["candidates"]:
        assert {
            "name",
            "source_url",
            "package_version",
            "license",
            "install_command",
            "api_entry_point",
            "accepted",
            "reason",
        } <= set(candidate)


def test_metric_json_sanitizes_nonfinite_smoke_values(tmp_path):
    from scripts.reconstruction.hio_cdi_benchmark import _metrics_jsonable, _write_json

    payload, annotations = _metrics_jsonable(
        {
            "mae": (1.0, np.nan),
            "frc50": (np.inf, 2.0),
            "frc": ("curve-array-placeholder",),
        }
    )

    assert payload["mae"] == [1.0, None]
    assert payload["frc50"] == [None, 2.0]
    assert annotations == [
        {"metric": "mae.1", "value": "nan", "stored_as": None},
        {"metric": "frc50.0", "value": "inf", "stored_as": None},
    ]

    path = _write_json(tmp_path / "metrics.json", {"metrics": payload})
    text = path.read_text()
    assert "NaN" not in text
    assert "Infinity" not in text


def test_metrics_artifact_context_embeds_contract_and_residual_content(tmp_path):
    from scripts.reconstruction.hio_cdi_benchmark import (
        _write_json,
        build_metrics_artifact_context,
        write_data_identity_manifest,
        write_metric_contract_manifest,
    )

    data_identity = write_data_identity_manifest(
        tmp_path,
        branch="same-split-rerun",
        artifact_paths=[],
        run_id="unit",
        data_generation_control="loader-compatible",
    )
    metric_contract = write_metric_contract_manifest(tmp_path, mode="direct-stitch", run_id="unit")
    support_manifest = _write_json(
        tmp_path / "support_manifest.json",
        {
            "run_id": "unit",
            "support_records": [
                {"support_threshold": 0.01, "status": "valid_sensitivity"},
                {"support_threshold": 0.05, "status": "selected_primary"},
                {"support_threshold": 0.10, "status": "invalid", "reason": "empty support mask"},
            ],
        },
    )
    residual_payload = {
        "run_id": "unit",
        "condition": "gs1_custom",
        "patches": [
            {
                "patch_index": 0,
                "selected_seed": 123,
                "selected_base_seed": 2026041201,
                "selected_restart_index": 0,
                "selected_final_residual": 0.25,
                "restart_final_residuals": [
                    {"restart_index": 0, "base_seed": 2026041201, "seed": 123, "final_residual": 0.25}
                ],
                "selected_residual_curve": [0.6, 0.25],
            }
        ],
    }
    residual_path = _write_json(tmp_path / "residuals_gs1_custom_support_0p05.json", residual_payload)

    context = build_metrics_artifact_context(
        row_label="gs1_custom_support_0p05",
        data_identity_manifest=data_identity,
        metric_contract_manifest=metric_contract,
        support_manifest=support_manifest,
        residual_payload=residual_payload,
        residual_path=residual_path,
    )

    assert context["data_identity"]["branch"] == "same-split-rerun"
    assert context["data_identity"]["decision"] == "same_split_rerun_bundle_not_frozen"
    assert context["metric_contract"]["mode"] == "direct-stitch"
    assert context["metric_contract"]["table2_compatibility"] == "fresh_same_split_direct_stitch_not_historical_table2"
    assert context["support_threshold_grid_status"] == [
        {"support_threshold": 0.01, "status": "valid_sensitivity"},
        {"support_threshold": 0.05, "status": "selected_primary"},
        {"support_threshold": 0.10, "status": "invalid", "reason": "empty support mask"},
    ]
    assert context["selected_restart"]["per_patch"] == [
        {
            "patch_index": 0,
            "selected_seed": 123,
            "selected_base_seed": 2026041201,
            "selected_restart_index": 0,
            "selected_final_residual": 0.25,
        }
    ]
    assert context["residuals"]["path"] == str(residual_path)
    assert context["residuals"]["sha256"]
    assert context["residuals"]["selected_residual_curves"] == [
        {"patch_index": 0, "selected_residual_curve": [0.6, 0.25]}
    ]
    assert context["residuals"]["restart_final_residuals_by_patch"][0]["restart_final_residuals"][0]["seed"] == 123


def test_cli_preflight_writes_required_manifests_without_metrics(tmp_path):
    script = Path("scripts/reconstruction/hio_cdi_benchmark.py")
    out = tmp_path / "preflight"
    cmd = [
        sys.executable,
        str(script),
        "--output-root",
        str(out),
        "--run-id",
        "unit_preflight",
        "--probe-npz",
        "datasets/Run1084_recon3_postPC_shrunk_3.npz",
        "--probe-source",
        "custom",
        "--probe-scale-mode",
        "pad_preserve",
        "--probe-smoothing-sigma",
        "0.5",
        "--support-thresholds",
        "0.01",
        "0.05",
        "0.10",
        "--primary-support-threshold",
        "0.05",
        "--restart-seeds",
        "2026041201",
        "2026041202",
        "2026041203",
        "--data-identity-branch",
        "frozen-artifact",
        "--metric-contract-mode",
        "unresolved",
        "--preflight-only",
    ]
    import subprocess

    completed = subprocess.run(cmd, check=True, text=True, capture_output=True)

    assert "preflight complete" in completed.stdout
    assert (out / "solver_manifest.json").exists()
    assert (out / "data_identity_manifest.json").exists()
    assert (out / "metric_contract_manifest.json").exists()
    assert (out / "runtime_provenance.json").exists()
    assert (out / "manifest.json").exists()
    assert (out / "invocation.json").exists()
    assert (out / "invocation.sh").exists()
    assert not list(out.glob("metrics*.json"))
