"""Train-only legacy gain calibration preparation and selection tests."""

from __future__ import annotations

import hashlib
import json
import tomllib
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from scripts.studies.ablation import legacy_gain_calibration as calibration
from scripts.studies.ablation import reporting
from scripts.studies.ablation.datasets import load_checked_dataset_bundle
from scripts.studies.ablation.manifest import load_manifest, resolve_manifest
from scripts.studies.ablation.metrics import (
    build_image_metric_record,
    build_metric_record,
)
from scripts.studies.ablation.verdicts import (
    AttemptRow,
    AttemptStatus,
    CompletionState,
)
from scripts.studies import torch_ablation_driver


@pytest.fixture(autouse=True)
def _isolated_repository_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(calibration, "_REPO_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source_train(path: Path) -> None:
    side = 5
    scan_index = np.arange(side * side, dtype=np.int64)
    xcoords, ycoords = np.meshgrid(
        np.arange(side, dtype=np.float32),
        np.arange(side, dtype=np.float32),
    )
    # Repeat one complete identity to prove an identity cannot cross the split.
    scan_index = np.concatenate((scan_index, scan_index[:1]))
    xcoords = np.concatenate((xcoords.ravel(), xcoords.ravel()[:1]))
    ycoords = np.concatenate((ycoords.ravel(), ycoords.ravel()[:1]))
    samples = scan_index.size
    rng = np.random.default_rng(41)
    diffraction = rng.uniform(0.1, 1.0, size=(samples, 64, 64)).astype(np.float32)
    probe = np.ones((64, 64), dtype=np.complex64)
    truth = np.ones((96, 96), dtype=np.complex64)
    truth_patches = np.ones((samples, 64, 64, 1), dtype=np.complex64)
    np.savez(
        path,
        diff3d=diffraction,
        probeGuess=probe,
        probeGeometry=probe[None],
        objectGuess=truth,
        xcoords=xcoords,
        ycoords=ycoords,
        xcoords_start=xcoords.copy(),
        ycoords_start=ycoords.copy(),
        scan_index=scan_index,
        ground_truth_patches=truth_patches,
        _metadata=np.asarray("source-train-only"),
    )


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(b"|")
    digest.update(str(value.shape).encode())
    digest.update(b"|")
    digest.update(value.tobytes())
    return digest.hexdigest()


def _write_base_spec(path: Path, source_train: Path) -> None:
    with np.load(source_train, allow_pickle=False) as archive:
        probe_hash = _array_sha256(np.asarray(archive["probeGuess"])[None])
    path.write_text(
        f"""
[schema]
version = 1

[study]
id = "base-compatibility"
seeds = [3, 17, 29]
output_root = ".artifacts/not-for-calibration"
require_all_explicit = false

[base.overrides]
"dataset.id" = "lines_legacy_amp"
"data.N" = 64
"data.C" = 4
"data.n_subsample" = 1
"data.grid_size" = [2, 2]
"model.mode" = "Unsupervised"
"model.C_model" = 4
"model.C_forward" = 4
"model.object_big" = true
"model.probe_big" = false
"model.offset" = 6
"model.training_patch_weighting" = "probe"
"training.batch_size" = 4
"inference.middle_trim" = 32
"execution.accelerator" = "cpu"
"execution.devices" = 1

[datasets.lines_legacy_amp]
kind = "synthetic"
format = "npz_mmap"
scale_contract_version = "legacy_v1"
measurement_domain = "normalized_amplitude"
truth = "object_truth"
truth_location = "embedded_test"
truth_key = "objectGuess"
measurement_key = "diff3d"
probe_key = "probeGuess"
x_key = "xcoords"
y_key = "ycoords"
coords_convention = "xy_pixels"
detector_shape = [64, 64]
grouping_max_C = 4
probe_modes = 1
train = {json.dumps(str(source_train))}
test = "DO-NOT-COPY-heldout-test.npz"
provenance = "DO-NOT-COPY-heldout-provenance.json"
train_sha256 = "{_sha256(source_train)}"
test_sha256 = "{"a" * 64}"
provenance_sha256 = "{"b" * 64}"

[datasets.lines_legacy_amp.probe]
source = "fixture-probe"
calibration = "legacy_normalized"
gauge = "legacy_normalized"
mask_policy = "model_config"
sha256 = "{probe_hash}"

[[matrix.dimensions]]
name = "architecture"

[[matrix.dimensions.values]]
id = "cnn"
[matrix.dimensions.values.overrides]
"model.architecture" = "cnn"
"model.cnn_output_mode" = "real_imag"
"model.use_shared_decoder" = false
"model.n_filters_scale" = 2
"model.batch_norm" = false
"model.cbam_encoder" = true
"model.cbam_bottleneck" = false
"model.decoder_last_c_outer_fraction" = 0.125
"model.decoder_last_amp_channels" = 4
"training.epochs_fine_tune" = 0

[[matrix.dimensions.values]]
id = "hybrid_resnet"
[matrix.dimensions.values.overrides]
"model.architecture" = "hybrid_resnet"
"model.generator_output_mode" = "real_imag"
"model.fno_modes" = 12
"model.fno_width" = 32
"model.fno_blocks" = 4
"model.fno_input_transform" = "none"
"model.learned_input_channels" = 1
"model.max_hidden_channels" = "auto"
"model.resnet_width" = "auto"
"model.hybrid_skip_connections" = false
"model.hybrid_downsample_steps" = 2
"model.hybrid_downsample_op" = "stride_conv"
"model.hybrid_encoder_conv_hidden_scale" = 1.0
"model.hybrid_encoder_spectral_hidden_scale" = 1.0
"model.hybrid_resnet_blocks" = 6

[[matrix.dimensions]]
name = "physics_profile"

[[matrix.dimensions.values]]
id = "ci_nll"
[matrix.dimensions.values.overrides]
"data.scale_contract_version" = "ci_intensity_v2"
"data.measurement_domain" = "count_intensity"
"model.physics_forward_mode" = "rectangular_scaled"
"model.loss_function" = "Poisson"
"training.torch_loss_mode" = "poisson"
"inference.varpro_scaling" = true

[[matrix.dimensions.values]]
id = "legacy_mae"
[matrix.dimensions.values.overrides]
"data.scale_contract_version" = "legacy_v1"
"data.measurement_domain" = "normalized_amplitude"
"data.normalize" = "Batch"
"data.data_scaling" = "Parseval"
"model.physics_forward_mode" = "amplitude"
"model.rect_s1s2_trainable" = false
"model.loss_function" = "MAE"
"training.torch_loss_mode" = "mae"
"training.torch_mae_pred_l2_match_target" = false
"inference.varpro_scaling" = false

[[matrix.dimensions.values]]
id = "legacy_nll"
[matrix.dimensions.values.overrides]
"data.scale_contract_version" = "legacy_v1"
"data.measurement_domain" = "normalized_amplitude"
"data.normalize" = "Batch"
"data.data_scaling" = "Parseval"
"model.physics_forward_mode" = "amplitude"
"model.rect_s1s2_trainable" = false
"model.loss_function" = "Poisson"
"training.torch_loss_mode" = "poisson"
"inference.varpro_scaling" = false

[[matrix.dimensions.values]]
id = "legacy_shadow"
[matrix.dimensions.values.overrides]
"data.scale_contract_version" = "ci_intensity_v2"
"data.measurement_domain" = "count_intensity"
"model.physics_forward_mode" = "rectangular_scaled"
"model.rect_s1s2_trainable" = true
"model.loss_function" = "Poisson"
"training.torch_loss_mode" = "poisson"
"inference.varpro_scaling" = true
""".strip()
        + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def source_train(tmp_path: Path) -> Path:
    path = tmp_path / "lines_legacy_amp_train.npz"
    _write_source_train(path)
    return path


def _request(tmp_path: Path, source_train: Path, **overrides):
    base_spec = tmp_path / "base.toml"
    _write_base_spec(base_spec, source_train)
    values = dict(
        source_train=source_train,
        base_spec=base_spec,
        output_root=tmp_path / "calibration",
        architectures=("cnn", "hybrid_resnet"),
        loss_profiles=("legacy_mae", "legacy_nll"),
        gains=(1.0, 4.0, 16.0, 64.0),
        seed=7,
        epochs=5,
        calibration_fraction=0.24,
    )
    values.update(overrides)
    return calibration.CalibrationRequest(**values)


def test_split_is_deterministic_disjoint_and_preserves_invariants(
    tmp_path: Path, source_train: Path
) -> None:
    prepared = calibration.prepare_split(
        calibration.SplitRequest(source_train, tmp_path / "split", 7, 0.24)
    )

    evidence = json.loads(prepared.evidence_path.read_text())
    assert evidence["source_train_sha256"] == _sha256(source_train)
    assert evidence["disjoint"] is True
    assert evidence["coverage"] is True
    assert evidence["optimization_scan_ids"]
    assert evidence["calibration_scan_ids"]
    assert set(evidence["optimization_scan_ids"]).isdisjoint(
        evidence["calibration_scan_ids"]
    )
    assert len(evidence["optimization_scan_ids_sha256"]) == 64
    assert len(evidence["calibration_scan_ids_sha256"]) == 64
    assert evidence["output_sha256"] == {
        "optimization": _sha256(prepared.optimization_npz),
        "calibration": _sha256(prepared.calibration_npz),
    }

    with (
        np.load(source_train, allow_pickle=False) as source,
        np.load(prepared.optimization_npz, allow_pickle=False) as optimization,
        np.load(prepared.calibration_npz, allow_pickle=False) as held,
    ):
        for key in ("probeGuess", "probeGeometry", "objectGuess", "_metadata"):
            np.testing.assert_array_equal(optimization[key], source[key])
            np.testing.assert_array_equal(held[key], source[key])
        optimization_rows = set(optimization["scan_index"].tolist())
        calibration_rows = set(held["scan_index"].tolist())
        assert optimization_rows.isdisjoint(calibration_rows)
        assert optimization_rows | calibration_rows == set(
            source["scan_index"].tolist()
        )
        assert (
            np.count_nonzero(optimization["scan_index"] == 0),
            np.count_nonzero(held["scan_index"] == 0),
        ) in {(2, 0), (0, 2)}

    before = prepared.evidence_path.stat().st_mtime_ns
    repeated = calibration.prepare_split(
        calibration.SplitRequest(source_train, tmp_path / "split", 7, 0.24)
    )
    assert repeated == prepared
    assert repeated.evidence_path.stat().st_mtime_ns == before


def test_split_refuses_overwrite_mismatch(tmp_path: Path, source_train: Path) -> None:
    request = calibration.SplitRequest(source_train, tmp_path / "split", 7, 0.24)
    prepared = calibration.prepare_split(request)
    prepared.optimization_npz.write_bytes(b"tampered")

    with pytest.raises(calibration.CalibrationError, match="overwrite mismatch"):
        calibration.prepare_split(request)


def test_split_refuses_tampered_identity_evidence(
    tmp_path: Path, source_train: Path
) -> None:
    request = calibration.SplitRequest(source_train, tmp_path / "split", 7, 0.24)
    prepared = calibration.prepare_split(request)
    evidence = json.loads(prepared.evidence_path.read_text())
    evidence["calibration_scan_ids"] = ["tampered"]
    prepared.evidence_path.write_text(json.dumps(evidence))

    with pytest.raises(calibration.CalibrationError, match="overwrite mismatch"):
        calibration.prepare_split(request)


def test_split_requires_scan_index(tmp_path: Path, source_train: Path) -> None:
    with np.load(source_train, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files if key != "scan_index"}
    without_scan_index = tmp_path / "without_scan_index.npz"
    np.savez(without_scan_index, **payload)

    with pytest.raises(calibration.CalibrationError, match="scan_index"):
        calibration.prepare_split(
            calibration.SplitRequest(without_scan_index, tmp_path / "split", 7, 0.24)
        )


def test_split_rejects_ambiguous_sample_shaped_extra(
    tmp_path: Path, source_train: Path
) -> None:
    with np.load(source_train, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["unexpected_per_sample"] = np.arange(
        payload["scan_index"].shape[0], dtype=np.int32
    )
    ambiguous = tmp_path / "ambiguous.npz"
    np.savez(ambiguous, **payload)

    with pytest.raises(calibration.CalibrationError, match="ambiguous.*per-sample"):
        calibration.prepare_split(
            calibration.SplitRequest(ambiguous, tmp_path / "split", 7, 0.24)
        )


def test_prepare_generates_generic_legacy_only_matrix_without_test_provenance(
    tmp_path: Path, source_train: Path
) -> None:
    prepared = calibration.prepare_calibration(_request(tmp_path, source_train))

    raw = tomllib.loads(prepared.spec_path.read_text())
    assert raw["study"]["seeds"] == [7]
    assert raw["base"]["overrides"]["training.epochs"] == 5
    assert set(raw["datasets"]) == {"legacy_gain_calibration"}
    dataset = raw["datasets"]["legacy_gain_calibration"]
    assert Path(dataset["train"]).resolve() == prepared.split.optimization_npz
    assert Path(dataset["test"]).resolve() == prepared.split.calibration_npz
    assert dataset["measurement_domain"] == "normalized_amplitude"
    dimensions = {item["name"]: item["values"] for item in raw["matrix"]["dimensions"]}
    assert [item["id"] for item in dimensions["architecture"]] == [
        "cnn",
        "hybrid_resnet",
    ]
    assert [item["id"] for item in dimensions["loss_profile"]] == [
        "legacy_mae",
        "legacy_nll",
    ]
    assert [
        item["overrides"]["model.amplitude_physics_gain"] for item in dimensions["gain"]
    ] == [
        1.0,
        4.0,
        16.0,
        64.0,
    ]
    study = resolve_manifest(load_manifest(prepared.spec_path))
    assert len(study.runs) == 16
    assert all("ci" not in run.dimensions["loss_profile"] for run in study.runs)
    persisted_text = "\n".join(
        path.read_text(errors="ignore")
        for path in prepared.output_root.rglob("*")
        if path.is_file() and path.suffix != ".npz"
    )
    assert "DO-NOT-COPY-heldout" not in persisted_text


def test_prepare_rejects_external_output_root_before_writing(
    tmp_path: Path, source_train: Path
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside" / "calibration"
    request = _request(tmp_path, source_train, output_root=outside)

    with pytest.raises(calibration.CalibrationError, match="inside repository root"):
        calibration.prepare_calibration(request)

    assert not outside.exists()


def test_prepare_rejects_hash_matched_non_lines_dataset(
    tmp_path: Path, source_train: Path
) -> None:
    request = _request(tmp_path, source_train)
    text = request.base_spec.read_text().replace(
        "lines_legacy_amp", "deadleaves_legacy_amp"
    )
    request.base_spec.write_text(text)

    with pytest.raises(calibration.CalibrationError, match="lines_legacy_amp"):
        calibration.prepare_calibration(request)


def test_prepare_rejects_disguised_ci_profile(
    tmp_path: Path, source_train: Path
) -> None:
    request = _request(
        tmp_path,
        source_train,
        loss_profiles=("legacy_shadow",),
        gains=(1.0,),
    )

    with pytest.raises(calibration.CalibrationError, match="legacy semantics"):
        calibration.prepare_calibration(request)


def test_driver_command_uses_path_python_and_existing_driver(
    tmp_path: Path, source_train: Path
) -> None:
    prepared = calibration.prepare_calibration(_request(tmp_path, source_train))

    command = calibration.driver_command(prepared, dry_run=True)

    assert command[0] == "python"
    assert command[1:3] == ("-m", "scripts.studies.torch_ablation_driver")
    assert "--dry-run" in command
    assert "--spec" in command
    assert "--output-root" in command


def test_generated_dataset_passes_generic_driver_preflight(
    tmp_path: Path, source_train: Path
) -> None:
    prepared = calibration.prepare_calibration(_request(tmp_path, source_train))
    raw = tomllib.loads(prepared.spec_path.read_text())

    validated = load_checked_dataset_bundle(raw["datasets"], repo_root=tmp_path)

    assert set(validated) == {"legacy_gain_calibration"}


def test_prepare_runs_generated_dataset_preflight(
    tmp_path: Path,
    source_train: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[set[str]] = []
    real_preflight = calibration.load_checked_dataset_bundle

    def recording_preflight(values, *, repo_root):
        calls.append(set(values))
        return real_preflight(values, repo_root=repo_root)

    monkeypatch.setattr(calibration, "load_checked_dataset_bundle", recording_preflight)

    calibration.prepare_calibration(_request(tmp_path, source_train))

    assert calls == [{"legacy_gain_calibration"}]


def test_existing_generic_driver_accepts_generated_dry_run(
    tmp_path: Path,
    source_train: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = calibration.prepare_calibration(_request(tmp_path, source_train))

    code = torch_ablation_driver.main(
        [
            "--spec",
            str(prepared.spec_path),
            "--output-root",
            str(prepared.driver_output_root),
            "--dry-run",
        ]
    )

    output = capsys.readouterr()
    assert code == 0, output.err
    assert "runs 16" in output.out
    assert "legacy_mae" in output.out
    assert "legacy_nll" in output.out


def test_cli_has_no_held_out_test_path_surface() -> None:
    parser = calibration.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "prepare",
                "--train-npz",
                "train.npz",
                "--base-spec",
                "base.toml",
                "--output-root",
                "out",
                "--test-npz",
                "forbidden.npz",
            ]
        )


def _row(
    architecture: str,
    profile: str,
    gain: float,
    amp: float,
    phase: float,
    *,
    collapsed: bool = False,
) -> dict[str, object]:
    return {
        "architecture": architecture,
        "loss_profile": profile,
        "gain": gain,
        "amplitude_ssim": amp,
        "phase_ssim": phase,
        "collapsed": collapsed,
        "status": "success",
    }


def test_selection_is_profile_wide_independent_and_uses_tie_breaks() -> None:
    architectures = ("cnn", "hybrid_resnet")
    profiles = ("legacy_mae", "legacy_nll")
    gains = (1.0, 4.0, 16.0)
    amp = {
        "legacy_mae": {1.0: (0.5, 0.6), 4.0: (0.8, 0.9), 16.0: (0.8, 0.9)},
        "legacy_nll": {1.0: (0.4, 0.5), 4.0: (0.6, 0.7), 16.0: (0.9, 0.8)},
    }
    phase = {
        "legacy_mae": {1.0: (0.5, 0.5), 4.0: (0.7, 0.7), 16.0: (0.7, 0.7)},
        "legacy_nll": {1.0: (0.5, 0.5), 4.0: (0.6, 0.6), 16.0: (0.8, 0.8)},
    }
    rows = [
        _row(
            architecture,
            profile,
            gain,
            amp[profile][gain][index],
            phase[profile][gain][index],
        )
        for profile, gain, (index, architecture) in product(
            profiles, gains, enumerate(architectures)
        )
    ]

    selected = calibration.select_profile_gains(
        rows,
        architectures=architectures,
        loss_profiles=profiles,
        gains=gains,
    )

    assert selected["legacy_mae"]["selected_gain"] == 4.0
    assert selected["legacy_mae"]["status"] == "selected"
    assert selected["legacy_nll"]["selected_gain"] == 16.0
    assert selected["legacy_nll"]["status"] == "unbracketed"
    assert selected["legacy_nll"]["boundary"] == "upper"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing", "complete row"),
        ("nonfinite", "finite"),
        ("collapsed", "collapsed"),
    ],
)
def test_selection_rejects_incomplete_nonfinite_or_collapsed_rows(
    mutation: str, match: str
) -> None:
    rows = [
        _row(architecture, "legacy_mae", gain, 0.7, 0.6)
        for architecture, gain in product(("cnn", "hybrid_resnet"), (1.0, 4.0))
    ]
    if mutation == "missing":
        rows.pop()
    elif mutation == "nonfinite":
        rows[0]["amplitude_ssim"] = float("nan")
    else:
        rows[0]["collapsed"] = True

    with pytest.raises(calibration.CalibrationError, match=match):
        calibration.select_profile_gains(
            rows,
            architectures=("cnn", "hybrid_resnet"),
            loss_profiles=("legacy_mae",),
            gains=(1.0, 4.0),
        )


def _write_driver_report(
    prepared,
    scores: dict[tuple[str, str, float], tuple[float, float]],
    *,
    manifest_sha256: str | None = None,
    study_id: str | None = None,
    matrix_dataset_id: str | None = None,
    include_unrelated_curve: bool = False,
) -> None:
    manifest = load_manifest(prepared.spec_path)
    study = resolve_manifest(manifest)
    canonical_sha256 = hashlib.sha256(manifest.canonical_json.encode()).hexdigest()
    selected_runs = []
    rows = []
    identities = []
    for run in study.runs:
        architecture = str(run.dimensions["architecture"])
        profile = str(run.dimensions["loss_profile"])
        gain = float(run.overrides["model.amplitude_physics_gain"])
        amp, phase = scores[(architecture, profile, gain)]
        selected_runs.append(
            {
                "id": run.id,
                "arm_id": run.arm_id,
                "dataset_id": matrix_dataset_id or run.dataset_id,
                "seed": run.seed,
                "dimensions": dict(run.dimensions),
                "overrides": dict(run.overrides),
            }
        )
        unrelated_records = (
            (
                build_metric_record(
                    "truth_quality.amp_frc_curve",
                    (1.0, 0.7, 0.2),
                    basis="calibration_truth",
                    alignment="centering",
                    truth_role="object_truth",
                ),
            )
            if include_unrelated_curve
            else ()
        )
        metric_records = (
            build_image_metric_record(
                "amp_ssim",
                amp,
                truth_role="object_truth",
                basis="calibration_truth",
                alignment="centering",
            ),
            build_image_metric_record(
                "phase_ssim",
                phase,
                truth_role="object_truth",
                basis="calibration_truth",
                alignment="centering",
            ),
            *(
                build_metric_record(
                    path,
                    value,
                    basis="canonical_reassembly",
                    alignment="none",
                )
                for path, value in (
                    ("stability.finite", 1.0),
                    ("stability.amp_variance", 0.1),
                    ("stability.amp_dynamic_range", 0.2),
                    ("stability.phase_variance", 0.1),
                    ("stability.phase_dynamic_range", 0.2),
                )
            ),
            *unrelated_records,
        )
        attempt = AttemptRow(
            run_id=run.id,
            arm_id=run.arm_id,
            dataset_id=run.dataset_id,
            seed=run.seed,
            status=AttemptStatus.SUCCESS,
            completion=CompletionState.TERMINAL,
            metrics={record.path: record.value for record in metric_records},
        )
        reconstruction = np.asarray([[1 + 0j, 2 + 0j], [3 + 0j, 4 + 0j]])
        target = reconstruction.copy()
        rows.append(
            reporting.ReportRow(
                attempt=attempt,
                truth_role="object_truth",
                reconstruction=reconstruction,
                target=target,
                error=np.zeros_like(target),
                common_valid_mask=np.ones(target.shape, dtype=np.bool_),
                training_loss=(1.0,),
                gradient_norm=(0.1,),
                metric_records=metric_records,
            )
        )
        identities.append(
            reporting.RunIdentity(
                run.id,
                run.arm_id,
                run.dataset_id,
                run.seed,
                truth_role="object_truth",
                capabilities=frozenset({"has_object_truth"}),
                contract_declared=True,
                object_family="lines",
            )
        )
    expansion = {
        "schema_version": "ablation_expansion_v1",
        "study_id": study_id or manifest.study_id,
        "manifest_sha256": manifest_sha256 or canonical_sha256,
        "requested_seeds": list(manifest.seeds),
        "dataset_materialization_profiles": {
            "legacy_gain_calibration": None,
        },
        "selected_runs": selected_runs,
        "excludes": [],
        "gates": [],
        "comparisons": [],
    }
    root = prepared.driver_output_root
    reporting.write_report(
        reporting.ReportInput(
            study_id=manifest.study_id,
            rows=tuple(rows),
            requested_runs=tuple(identities),
            gate_results=(),
            source_manifest=prepared.spec_path.read_bytes(),
            source_config={},
            invocation={},
            expansion=expansion,
        ),
        root,
    )


def _rewrite_sealed_report_artifact(
    prepared, filename: str, mutate
) -> None:
    artifact_path = prepared.driver_output_root / filename
    payload = json.loads(artifact_path.read_text())
    mutate(payload)
    artifact_path.write_text(json.dumps(payload))
    completion_path = prepared.driver_output_root / "report_completion.json"
    completion = json.loads(completion_path.read_text())
    artifact = next(
        item for item in completion["artifacts"] if item["path"] == filename
    )
    artifact["sha256"] = _sha256(artifact_path)
    completion_path.write_text(json.dumps(completion))


def _selection_request(tmp_path: Path, source_train: Path):
    request = _request(
        tmp_path,
        source_train,
        gains=(1.0, 4.0),
        loss_profiles=("legacy_mae", "legacy_nll"),
    )
    prepared = calibration.prepare_calibration(request)
    scores = {
        (architecture, profile, gain): (
            0.9 if gain == 4.0 else 0.5,
            0.8 if gain == 4.0 else 0.4,
        )
        for architecture, profile, gain in product(
            request.architectures, request.loss_profiles, request.gains
        )
    }
    return request, prepared, scores


def test_finalize_persists_candidate_rows_and_selection_without_test_provenance(
    tmp_path: Path, source_train: Path
) -> None:
    request, prepared, scores = _selection_request(tmp_path, source_train)
    _write_driver_report(prepared, scores)

    result = calibration.finalize_selection(request.output_root)

    assert len(result["candidate_rows"]) == 8
    assert set(result["selected_gains"]) == {"legacy_mae", "legacy_nll"}
    assert result["selectors"] == {
        "aggregate": "median_across_architectures",
        "primary": "amplitude_ssim",
        "tie_break": ["phase_ssim", "smaller_gain"],
    }
    assert result["architectures"] == ["cnn", "hybrid_resnet"]
    assert result["seed"] == 7
    assert result["epochs"] == 5
    assert "test" not in json.dumps(result).lower()
    assert prepared.candidate_rows_path.is_file()
    assert prepared.selection_path.is_file()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("source_train_sha256", "f" * 64, "source"),
        ("selectors", {"aggregate": "mean"}, "selectors"),
    ],
)
def test_finalize_rejects_tampered_request_provenance(
    tmp_path: Path,
    source_train: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    _, prepared, scores = _selection_request(tmp_path, source_train)
    _write_driver_report(prepared, scores)
    request_payload = json.loads(prepared.request_path.read_text())
    request_payload[field] = value
    prepared.request_path.write_text(json.dumps(request_payload))

    with pytest.raises(calibration.CalibrationError, match=match):
        calibration.finalize_selection(prepared.output_root)


def test_finalize_ignores_unrelated_curve_metrics(
    tmp_path: Path, source_train: Path
) -> None:
    _, prepared, scores = _selection_request(tmp_path, source_train)
    _write_driver_report(prepared, scores, include_unrelated_curve=True)

    result = calibration.finalize_selection(prepared.output_root)

    assert len(result["candidate_rows"]) == 8


@pytest.mark.parametrize(
    ("filename", "field", "wrong_value"),
    [
        ("arm_seed_status.json", "arm_id", "wrong-arm"),
        ("arm_seed_status.json", "dataset_id", "wrong-dataset"),
        ("arm_seed_status.json", "seed", 999),
        ("aggregate_metrics.json", "arm_id", "wrong-arm"),
        ("aggregate_metrics.json", "dataset_id", "wrong-dataset"),
        ("aggregate_metrics.json", "seed", 999),
    ],
)
def test_finalize_rejects_wrong_status_or_selection_metric_identity(
    tmp_path: Path,
    source_train: Path,
    filename: str,
    field: str,
    wrong_value: object,
) -> None:
    _, prepared, scores = _selection_request(tmp_path, source_train)
    _write_driver_report(prepared, scores)

    def mutate(payload: dict[str, object]) -> None:
        rows = payload["rows"]
        assert isinstance(rows, list)
        rows[0][field] = wrong_value

    _rewrite_sealed_report_artifact(prepared, filename, mutate)

    with pytest.raises(calibration.CalibrationError, match="identity"):
        calibration.finalize_selection(prepared.output_root)


def test_finalize_requires_completed_report(tmp_path: Path, source_train: Path) -> None:
    _, prepared, scores = _selection_request(tmp_path, source_train)
    _write_driver_report(prepared, scores)
    (prepared.driver_output_root / "report_completion.json").unlink()

    with pytest.raises(calibration.CalibrationError, match="completion"):
        calibration.finalize_selection(prepared.output_root)


def test_finalize_rejects_tampered_report_artifact(
    tmp_path: Path, source_train: Path
) -> None:
    _, prepared, scores = _selection_request(tmp_path, source_train)
    _write_driver_report(prepared, scores)
    (prepared.driver_output_root / "report.md").write_text("tampered\n")

    with pytest.raises(calibration.CalibrationError, match="report"):
        calibration.finalize_selection(prepared.output_root)


def test_finalize_rejects_wrong_manifest_binding(
    tmp_path: Path, source_train: Path
) -> None:
    _, prepared, scores = _selection_request(tmp_path, source_train)
    _write_driver_report(prepared, scores, manifest_sha256="f" * 64)

    with pytest.raises(calibration.CalibrationError, match="manifest"):
        calibration.finalize_selection(prepared.output_root)


def test_finalize_rejects_wrong_run_matrix_binding(
    tmp_path: Path, source_train: Path
) -> None:
    _, prepared, scores = _selection_request(tmp_path, source_train)
    _write_driver_report(prepared, scores, matrix_dataset_id="wrong_dataset")

    with pytest.raises(calibration.CalibrationError, match="run matrix"):
        calibration.finalize_selection(prepared.output_root)
