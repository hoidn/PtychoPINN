import json
import sys
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
