import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _toy_probe():
    yy, xx = np.indices((8, 8))
    amp = 1.0 + 0.05 * yy + 0.03 * xx
    phase = 0.1 * (yy - 3.5) ** 2 + 0.04 * (xx - 3.5)
    return (amp * np.exp(1j * phase)).astype(np.complex64)


def test_array_sha256_is_stable_for_complex_arrays():
    from scripts.studies.probe_mischaracterization_stress_test import array_sha256

    probe = _toy_probe()
    assert array_sha256(probe) == array_sha256(np.array(probe, copy=True))
    assert array_sha256(probe) != array_sha256(probe.astype(np.complex128))


def test_energy_renormalization_preserves_reference_energy():
    from scripts.studies.probe_mischaracterization_stress_test import (
        probe_energy,
        renormalize_probe_energy,
    )

    probe = _toy_probe()
    perturbed = 2.5 * probe
    normalized = renormalize_probe_energy(perturbed, probe)
    assert probe_energy(normalized) == pytest.approx(probe_energy(probe), rel=1e-6)


def test_build_perturbation_grid_matches_approved_design():
    from scripts.studies.probe_mischaracterization_stress_test import build_perturbation_grid

    conditions = build_perturbation_grid()
    condition_ids = [condition.condition_id for condition in conditions]

    assert condition_ids == [
        "baseline",
        "phase_curvature_scale_0p75",
        "phase_curvature_scale_0p50",
        "phase_curvature_scale_0p25",
        "amplitude_blur_sigma_px_0p5",
        "amplitude_blur_sigma_px_1p0",
        "amplitude_blur_sigma_px_2p0",
        "phase_noise_sigma_rad_0p1pi_seed11",
        "phase_noise_sigma_rad_0p2pi_seed17",
        "phase_noise_sigma_rad_0p4pi_seed23",
    ]
    assert all(condition.reviewer_facing for condition in conditions)
    assert "amplitude_scale" not in {condition.perturbation_type for condition in conditions}


def test_probe_perturbations_are_deterministic_and_energy_preserving():
    from scripts.studies.probe_mischaracterization_stress_test import (
        PerturbationCondition,
        apply_probe_perturbation,
        array_sha256,
        probe_energy,
    )

    probe = _toy_probe()
    condition = PerturbationCondition(
        condition_id="phase_noise_sigma_rad_0p2pi_seed17",
        perturbation_type="phase_noise_sigma_rad",
        value=0.2 * np.pi,
        seed=17,
    )

    first, first_meta = apply_probe_perturbation(probe, condition)
    second, second_meta = apply_probe_perturbation(probe, condition)

    assert first.shape == probe.shape
    assert array_sha256(first) == array_sha256(second)
    assert array_sha256(first) != array_sha256(probe)
    assert probe_energy(first) == pytest.approx(probe_energy(probe), rel=1e-6)
    assert first_meta == second_meta
    assert first_meta["normalization_policy"] == "renormalize_total_energy"


def test_manifest_condition_entry_contains_probe_provenance():
    from scripts.studies.probe_mischaracterization_stress_test import (
        PerturbationCondition,
        build_condition_manifest_entry,
        array_sha256,
    )

    probe = _toy_probe()
    assumed = probe * np.exp(0.25j)
    condition = PerturbationCondition(
        condition_id="phase_test",
        perturbation_type="phase_noise_sigma_rad",
        value=0.25,
        seed=11,
    )

    entry = build_condition_manifest_entry(
        condition=condition,
        source_probe=probe,
        true_probe=probe,
        assumed_probe=assumed,
        perturbation_metadata={"normalization_policy": "renormalize_total_energy"},
    )

    assert entry["condition_id"] == "phase_test"
    assert entry["source_probe_sha256"] == array_sha256(probe)
    assert entry["true_probe_sha256"] == array_sha256(probe)
    assert entry["assumed_probe_sha256"] == array_sha256(assumed)
    assert entry["perturbation"]["type"] == "phase_noise_sigma_rad"
    assert entry["perturbation"]["value"] == 0.25
    assert entry["perturbation"]["seed"] == 11
    assert entry["normalization_policy"] == "renormalize_total_energy"


class _ToyContainer:
    def __init__(
        self,
        *,
        X,
        coords_nominal,
        coords_true,
        probe,
        norm_Y_I=1.0,
        YY_full=None,
        nn_indices=None,
        global_offsets=None,
        local_offsets=None,
    ):
        self._X_np = np.array(X, copy=True)
        self._Y_I_np = np.ones_like(self._X_np, dtype=np.float32) * np.float32(norm_Y_I)
        self._Y_phi_np = np.zeros_like(self._X_np, dtype=np.float32) + np.float32(0.25)
        self._coords_nominal_np = np.array(coords_nominal, copy=True)
        self._coords_true_np = np.array(coords_true, copy=True)
        self._probe_np = np.array(probe, copy=True)
        self.norm_Y_I = np.array(norm_Y_I, dtype=np.float32)
        self.YY_full = YY_full
        self.nn_indices = nn_indices
        self.global_offsets = global_offsets
        self.local_offsets = local_offsets


def _toy_sim(probe):
    train_X = np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1)
    test_X = train_X + 100
    train_coords = np.array([[[[1.0], [2.0]]]], dtype=np.float32)
    test_coords = np.array([[[[3.0], [4.0]]]], dtype=np.float32)
    train_YY_full = (np.ones((4, 4, 1), dtype=np.float32) * 3).astype(np.complex64)
    test_YY_full = (np.ones((4, 4, 1), dtype=np.float32) * 5).astype(np.complex64)
    return {
        "train": {
            "X": train_X,
            "coords_nominal": train_coords,
            "coords_true": train_coords + 0.5,
            "coords_offsets": np.array([[[[10.0], [20.0]]]], dtype=np.float32),
            "YY_full": train_YY_full,
            "container": _ToyContainer(
                X=train_X,
                coords_nominal=train_coords,
                coords_true=train_coords + 0.5,
                probe=probe,
                norm_Y_I=7.0,
                YY_full=train_YY_full,
                nn_indices=np.array([[0]], dtype=np.int64),
                global_offsets=np.array([[[[10.0], [20.0]]]], dtype=np.float32),
                local_offsets=np.array([[[[1.0], [2.0]]]], dtype=np.float32),
            ),
        },
        "test": {
            "X": test_X,
            "coords_nominal": test_coords,
            "coords_true": test_coords + 0.5,
            "YY_ground_truth": np.ones((4, 4, 1), dtype=np.complex64),
            "norm_Y_I": np.array(2.0, dtype=np.float32),
            "coords_offsets": np.array([[[[30.0], [40.0]]]], dtype=np.float32),
            "YY_full": test_YY_full,
            "intensity_scale": np.array(11.0, dtype=np.float32),
            "container": _ToyContainer(
                X=test_X,
                coords_nominal=test_coords,
                coords_true=test_coords + 0.5,
                probe=probe,
                norm_Y_I=13.0,
                YY_full=test_YY_full,
                nn_indices=np.array([[0]], dtype=np.int64),
                global_offsets=np.array([[[[30.0], [40.0]]]], dtype=np.float32),
                local_offsets=np.array([[[[3.0], [4.0]]]], dtype=np.float32),
            ),
        },
        "intensity_scale": np.array(17.0, dtype=np.float32),
    }


def test_write_canonical_condition_bundle_preserves_distinct_normalization_fields(tmp_path):
    from scripts.studies import probe_mischaracterization_stress_test as study

    true_probe = _toy_probe()
    sim = _toy_sim(true_probe)

    bundle = study.write_canonical_condition_bundle(tmp_path, sim, true_probe, cfg=types.SimpleNamespace())

    assert bundle["npz_path"] == tmp_path / "canonical_condition_inputs.npz"
    assert bundle["manifest_path"] == tmp_path / "canonical_condition_inputs_manifest.json"
    with np.load(bundle["npz_path"]) as data:
        keys = set(data.files)
        assert keys >= {
            "X_train",
            "X_test",
            "Y_I_train",
            "Y_I_test",
            "Y_phi_train",
            "Y_phi_test",
            "coords_nominal_train",
            "coords_nominal_test",
            "coords_true_train",
            "coords_true_test",
            "YY_full_train",
            "YY_full_test",
            "YY_ground_truth_test",
            "probe_true",
            "norm_Y_I_train_container",
            "norm_Y_I_test_container",
            "norm_Y_I_test_stitch",
            "intensity_scale_model",
        }
        assert "norm_Y_I" not in keys
        assert data["norm_Y_I_train_container"] != data["norm_Y_I_test_container"]
        assert data["norm_Y_I_test_container"] != data["norm_Y_I_test_stitch"]
        assert data["norm_Y_I_test_stitch"] != data["intensity_scale_model"]


def test_canonical_condition_bundle_manifest_records_presence_absence_and_constructor_mapping(tmp_path):
    from scripts.studies import probe_mischaracterization_stress_test as study

    true_probe = _toy_probe()
    sim = _toy_sim(true_probe)
    sim["train"]["container"].local_offsets = None
    sim["test"]["container"].local_offsets = None

    bundle = study.write_canonical_condition_bundle(tmp_path, sim, true_probe, cfg=types.SimpleNamespace())
    manifest = json.loads(bundle["manifest_path"].read_text())

    field = manifest["fields"]["X_train"]
    assert field["dtype"] == "float32"
    assert field["shape"] == [1, 4, 4, 1]
    assert field["checksum"]
    assert field["source_split"] == "train"
    assert field["required"] is True
    assert field["status"] == "present"

    assert manifest["fields"]["global_offsets_train"]["status"] == "present"
    assert manifest["fields"]["global_offsets_train"]["checksum"]
    assert manifest["fields"]["local_offsets_train"]["status"] == "absent"
    assert manifest["fields"]["local_offsets_train"]["absent_reason"] == "source container local_offsets is absent"
    assert manifest["constructor_mapping"]["train"]["norm_Y_I"] == "norm_Y_I_train_container"
    assert manifest["constructor_mapping"]["test"]["norm_Y_I"] == "norm_Y_I_test_container"
    assert manifest["constructor_mapping"]["train"]["probeGuess"] == "condition assumed_probe"
    assert "normalization_aliases" in manifest
    assert set(manifest["normalization_fields"]) == {
        "norm_Y_I_train_container",
        "norm_Y_I_test_container",
        "norm_Y_I_test_stitch",
        "intensity_scale_model",
    }


def test_validate_canonical_condition_bundle_rejects_bare_norm_or_missing_normalization(tmp_path):
    from scripts.studies import probe_mischaracterization_stress_test as study

    true_probe = _toy_probe()
    bundle = study.write_canonical_condition_bundle(tmp_path, _toy_sim(true_probe), true_probe, cfg=types.SimpleNamespace())
    study.validate_canonical_condition_bundle(bundle["npz_path"], bundle["manifest_path"])

    with np.load(bundle["npz_path"]) as data:
        payload = {key: data[key] for key in data.files}
    payload["norm_Y_I"] = np.array(1.0, dtype=np.float32)
    bad_npz = tmp_path / "bad_norm.npz"
    np.savez(bad_npz, **payload)

    with pytest.raises(ValueError, match="bare norm_Y_I"):
        study.validate_canonical_condition_bundle(bad_npz, bundle["manifest_path"])

    bad_manifest = json.loads(bundle["manifest_path"].read_text())
    bad_manifest["fields"].pop("norm_Y_I_test_stitch")
    bad_manifest_path = tmp_path / "bad_manifest.json"
    bad_manifest_path.write_text(json.dumps(bad_manifest))
    with pytest.raises(ValueError, match="norm_Y_I_test_stitch"):
        study.validate_canonical_condition_bundle(bundle["npz_path"], bad_manifest_path)


def test_load_condition_inputs_for_child_reconstructs_containers_and_preflight(tmp_path):
    from scripts.studies import probe_mischaracterization_stress_test as study

    true_probe = _toy_probe()
    assumed_probe = (true_probe * np.exp(0.2j)).astype(np.complex64)
    bundle = study.write_canonical_condition_bundle(tmp_path, _toy_sim(true_probe), true_probe, cfg=types.SimpleNamespace())
    assumed_probe_path = tmp_path / "assumed_probe.npz"
    np.savez(assumed_probe_path, probe=assumed_probe)

    loaded = study.load_condition_inputs_for_child(
        bundle["npz_path"],
        bundle["manifest_path"],
        assumed_probe_path,
    )

    assert np.array_equal(loaded["train_container"]._X_np, _toy_sim(true_probe)["train"]["container"]._X_np)
    assert np.array_equal(loaded["test_container"]._X_np, _toy_sim(true_probe)["test"]["container"]._X_np)
    assert loaded["train_container"].norm_Y_I == pytest.approx(np.array(7.0, dtype=np.float32))
    assert loaded["test_container"].norm_Y_I == pytest.approx(np.array(13.0, dtype=np.float32))
    assert study.array_sha256(loaded["train_container"]._probe_np) == study.array_sha256(assumed_probe)
    assert loaded["preflight"]["assumed_probe_checksums"]["expected_assumed_probe"] == study.array_sha256(assumed_probe)
    assert loaded["preflight"]["normalization_field_checksums"]["norm_Y_I_test_stitch"]


def test_child_mode_parse_and_dispatch_does_not_require_parent_output_root(monkeypatch, tmp_path):
    from scripts.studies import probe_mischaracterization_stress_test as study

    request = tmp_path / "child_request.json"
    request.write_text(json.dumps({"run_root": str(tmp_path), "child_invocation_path": str(tmp_path / "child_invocation.json")}))

    args = study.parse_args(["--child-condition-runner", "--child-request-json", str(request)])
    assert args.output_root is None
    assert args.child_condition_runner is True

    called = {}

    def fake_condition_child(path):
        called["path"] = path
        return 0

    monkeypatch.setattr(study, "run_condition_child_from_request", fake_condition_child)
    monkeypatch.setattr(study, "select_conditions", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("parent selector called")))
    monkeypatch.setattr(study, "prepare_output_root", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("parent output root prepared")))
    monkeypatch.setattr(study, "write_invocation_artifacts", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("parent invocation written")))

    assert study.main(["--child-condition-runner", "--child-request-json", str(request)]) == 0
    assert called["path"] == request


def test_child_smoke_mode_parse_and_dispatch_does_not_require_parent_output_root(monkeypatch, tmp_path):
    from scripts.studies import probe_mischaracterization_stress_test as study

    (tmp_path / "existing_child_output.txt").write_text("child run roots may already contain artifacts")
    request = tmp_path / "child_request.json"
    request.write_text(
        json.dumps(
            {
                "run_root": str(tmp_path),
                "child_invocation_path": str(tmp_path / "child_invocation.json"),
            }
        )
    )

    args = study.parse_args(["--child-smoke-runner", "--child-request-json", str(request)])
    assert args.output_root is None
    assert args.child_smoke_runner is True

    called = {}

    def fake_smoke_child(path):
        called["path"] = path
        return 0

    monkeypatch.setattr(study, "run_smoke_child_from_request", fake_smoke_child)
    monkeypatch.setattr(study, "select_conditions", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("parent selector called")))
    monkeypatch.setattr(study, "prepare_output_root", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("parent output root prepared")))
    monkeypatch.setattr(study, "write_invocation_artifacts", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("parent invocation written")))

    assert study.main(["--child-smoke-runner", "--child-request-json", str(request)]) == 0
    assert called["path"] == request


def test_child_requests_do_not_include_source_provenance_requirements(tmp_path):
    from scripts.studies import probe_mischaracterization_stress_test as study

    true_probe = _toy_probe()
    true_probe_path = tmp_path / "true_probe.npz"
    np.savez(true_probe_path, probe=true_probe)
    bundle_paths = {
        "npz_path": tmp_path / "canonical_condition_inputs.npz",
        "manifest_path": tmp_path / "canonical_condition_inputs_manifest.json",
    }
    cfg = study.grid_config_from_args(study.parse_args(["--output-root", str(tmp_path / "run")]))
    condition = study.PerturbationCondition("baseline", "baseline")
    assumed_probe = true_probe.copy()
    perturbation_metadata = {"normalization_policy": "renormalize_total_energy"}

    smoke_request = study.build_smoke_child_request(
        output_root=tmp_path,
        bundle_paths=bundle_paths,
        true_probe_path=true_probe_path,
        assumed_probe=assumed_probe,
        smoke_condition=condition,
        perturbation_metadata=perturbation_metadata,
        cfg=cfg,
    )["request"]
    condition_request = study.build_condition_child_request(
        output_root=tmp_path,
        bundle_paths=bundle_paths,
        condition=condition,
        assumed_probe=assumed_probe,
        perturbation_metadata=perturbation_metadata,
        branch_decision="fixed_wrong_probe_training",
        cfg=cfg,
    )["request"]

    assert "expected_source_provenance" not in smoke_request
    assert "expected_source_provenance" not in condition_request


def test_child_invocation_records_runtime_without_source_provenance_gate(tmp_path):
    from scripts.studies import probe_mischaracterization_stress_test as study

    request_path = tmp_path / "child_request.json"
    request = {
        "child_invocation_path": str(tmp_path / "child_invocation.json"),
    }
    request_path.write_text(json.dumps(request))

    invocation_path = study.write_child_invocation_artifact(
        request_path,
        request,
        "--child-condition-runner",
    )
    payload = json.loads(invocation_path.read_text())

    assert "runtime_provenance" in payload
    assert "source_provenance" not in payload
    assert "expected_source_provenance" not in payload
    assert "source_provenance_matches_expected" not in payload


def test_launch_child_process_uses_path_python_and_records_pid(monkeypatch, tmp_path):
    from scripts.studies import probe_mischaracterization_stress_test as study

    request_path = tmp_path / "conditions" / "baseline" / "child_request.json"
    request_path.parent.mkdir(parents=True)
    stdout_path = request_path.parent / "child_stdout.log"
    stderr_path = request_path.parent / "child_stderr.log"
    request_path.write_text(
        json.dumps(
            {
                "condition_id": "baseline",
                "stdout_log_path": str(stdout_path),
                "stderr_log_path": str(stderr_path),
            }
        )
    )

    class _FakePopen:
        def __init__(self, command, cwd, text, stdout, stderr, env):
            self.command = command
            self.cwd = cwd
            self.text = text
            self.stdout = stdout
            self.stderr = stderr
            self.env = env
            self.pid = 4321
            self.returncode = 0

        def communicate(self):
            return "child out", "child err"

        def wait(self):
            return self.returncode

    launched = {}

    def fake_popen(*args, **kwargs):
        proc = _FakePopen(*args, **kwargs)
        launched["proc"] = proc
        return proc

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    record = study.launch_child_process(
        mode_flag="--child-condition-runner",
        request_path=request_path,
    )

    assert launched["proc"].command[:2] == ["python", "scripts/studies/probe_mischaracterization_stress_test.py"]
    assert launched["proc"].env["PYTHONPATH"] == str(study.REPO_ROOT)
    assert record["pid"] == 4321
    assert record["return_code"] == 0
    assert stdout_path.read_text() == "child out"
    assert stderr_path.read_text() == "child err"


def test_condition_preflight_records_true_vs_assumed_contract(tmp_path):
    from scripts.studies import probe_mischaracterization_stress_test as study

    true_probe = _toy_probe()
    assumed_probe = true_probe * np.exp(0.1j)
    sim = _toy_sim(true_probe)
    canonical = study.build_canonical_data_checksums(sim)
    assumed_path = tmp_path / "assumed_probe.npz"
    np.savez(assumed_path, probe=assumed_probe)
    train_container = _ToyContainer(
        X=sim["train"]["X"],
        coords_nominal=sim["train"]["coords_nominal"],
        coords_true=sim["train"]["coords_true"],
        probe=assumed_probe,
    )
    test_container = _ToyContainer(
        X=sim["test"]["X"],
        coords_nominal=sim["test"]["coords_nominal"],
        coords_true=sim["test"]["coords_true"],
        probe=assumed_probe,
    )

    record = study.assert_condition_preflight(
        sim=sim,
        train_container=train_container,
        test_container=test_container,
        assumed_probe=assumed_probe,
        assumed_probe_path=assumed_path,
        canonical_data_checksums=canonical,
    )

    assert record["condition_probe_policy"] == "assumed_probe_replaces_container_probe"
    assert record["canonical_data_checksums"] == canonical
    assert record["condition_data_checksums"] == canonical
    assumed_probe_checksum = study.array_sha256(np.asarray(assumed_probe, dtype=np.complex64))
    assert record["assumed_probe_checksums"] == {
        "expected_assumed_probe": assumed_probe_checksum,
        "train_data_probe": assumed_probe_checksum,
        "test_data_probe": assumed_probe_checksum,
        "persisted_condition_probe": assumed_probe_checksum,
    }

    bad_train = _ToyContainer(
        X=sim["train"]["X"] + 1.0,
        coords_nominal=sim["train"]["coords_nominal"],
        coords_true=sim["train"]["coords_true"],
        probe=assumed_probe,
    )
    with pytest.raises(AssertionError, match="canonical data checksums"):
        study.assert_condition_preflight(
            sim=sim,
            train_container=bad_train,
            test_container=test_container,
            assumed_probe=assumed_probe,
            assumed_probe_path=assumed_path,
            canonical_data_checksums=canonical,
        )


def test_preflight_uses_training_container_arrays_as_canonical(tmp_path):
    from scripts.studies import probe_mischaracterization_stress_test as study

    true_probe = _toy_probe()
    assumed_probe = true_probe * np.exp(0.1j)
    sim = _toy_sim(true_probe)
    sim["train"]["X"] = sim["train"]["X"] + np.float32(1e-6)
    sim["test"]["X"] = sim["test"]["X"] + np.float32(1e-6)
    assumed_path = tmp_path / "assumed_probe.npz"
    np.savez(assumed_path, probe=assumed_probe)
    train_container = _ToyContainer(
        X=sim["train"]["container"]._X_np,
        coords_nominal=sim["train"]["container"]._coords_nominal_np,
        coords_true=sim["train"]["container"]._coords_true_np,
        probe=assumed_probe,
    )
    test_container = _ToyContainer(
        X=sim["test"]["container"]._X_np,
        coords_nominal=sim["test"]["container"]._coords_nominal_np,
        coords_true=sim["test"]["container"]._coords_true_np,
        probe=assumed_probe,
    )

    canonical = study.build_canonical_data_checksums(sim)
    record = study.assert_condition_preflight(
        sim=sim,
        train_container=train_container,
        test_container=test_container,
        assumed_probe=assumed_probe,
        assumed_probe_path=assumed_path,
        canonical_data_checksums=canonical,
    )

    assert record["condition_data_checksums"] == canonical


def test_persist_true_measurements_uses_training_container_arrays(tmp_path):
    from scripts.studies import probe_mischaracterization_stress_test as study

    true_probe = _toy_probe()
    sim = _toy_sim(true_probe)
    sim["train"]["X"] = sim["train"]["X"] + np.float32(1e-6)
    sim["test"]["X"] = sim["test"]["X"] + np.float32(1e-6)

    paths = study.persist_true_measurements(tmp_path, sim, true_probe)
    with np.load(paths["true_measurement_data"]) as data:
        assert np.array_equal(data["X_train"], sim["train"]["container"]._X_np)
        assert np.array_equal(data["X_test"], sim["test"]["container"]._X_np)


def test_prepare_output_root_refuses_existing_nonempty_root(tmp_path):
    from scripts.studies.probe_mischaracterization_stress_test import prepare_output_root

    output_root = tmp_path / "run"
    output_root.mkdir()
    (output_root / "existing.txt").write_text("do not clobber")

    with pytest.raises(FileExistsError):
        prepare_output_root(output_root, force=False)
    with pytest.raises(FileExistsError):
        prepare_output_root(output_root, force=True)


def test_dry_run_writes_required_artifacts(tmp_path):
    from scripts.studies import probe_mischaracterization_stress_test as study

    probe_npz = tmp_path / "probe.npz"
    np.savez(probe_npz, probeGuess=_toy_probe())
    output_root = tmp_path / "dry_run"

    status = study.main(
        [
            "--output-root",
            str(output_root),
            "--probe-npz",
            str(probe_npz),
            "--dry-run",
        ]
    )

    assert status == 0
    for name in [
        "invocation.json",
        "invocation.sh",
        "manifest.json",
        "provenance_discovery.json",
        "artifact_manifest.json",
    ]:
        assert (output_root / name).exists()

    manifest = json.loads((output_root / "manifest.json").read_text())
    assert manifest["true_probe_policy"] == "canonical_true_probe_fixed"
    assert manifest["measurement_arrays_fixed_across_conditions"] is True
    assert manifest["scope"]["trainable_probe_variants"] is False
    assert manifest["config"]["N"] == 64
    assert manifest["config"]["set_phi"] is False
    artifact_manifest = json.loads((output_root / "artifact_manifest.json").read_text())
    artifact_names = {Path(item["path"]).name for item in artifact_manifest["artifacts"]}
    assert artifact_names >= {
        "invocation.json",
        "invocation.sh",
        "manifest.json",
        "provenance_discovery.json",
        "artifact_manifest.json",
    }
    artifact_manifest_entry = next(
        item for item in artifact_manifest["artifacts"] if Path(item["path"]).name == "artifact_manifest.json"
    )
    assert artifact_manifest_entry["role"] == "artifact_manifest"
    assert artifact_manifest_entry["sha256"] is None


def test_dry_run_records_runtime_dependency_provenance(tmp_path):
    from scripts.studies import probe_mischaracterization_stress_test as study

    probe_npz = tmp_path / "probe.npz"
    np.savez(probe_npz, probeGuess=_toy_probe())
    output_root = tmp_path / "dry_run"

    status = study.main(
        [
            "--output-root",
            str(output_root),
            "--probe-npz",
            str(probe_npz),
            "--dry-run",
        ]
    )

    assert status == 0
    invocation = json.loads((output_root / "invocation.json").read_text())
    manifest = json.loads((output_root / "manifest.json").read_text())
    provenance = json.loads((output_root / "provenance_discovery.json").read_text())

    for payload in (
        invocation["extra"]["runtime_provenance"],
        manifest["runtime_provenance"],
        provenance["runtime_provenance"],
    ):
        assert payload["python_version"]
        assert "python_version_command" in payload
        assert "git_commit" in payload
        assert payload["package_versions"]["numpy"]
        assert "tensorflow" in payload["package_versions"]
        assert "scikit-image" in payload["package_versions"]


def test_smoke_only_uses_bounded_execution_config(tmp_path):
    from scripts.studies import probe_mischaracterization_stress_test as study

    base_args = study.parse_args(["--output-root", str(tmp_path / "full")])
    base_cfg, base_meta = study.execution_config_from_args(base_args)
    assert base_cfg.nepochs == 60
    assert base_cfg.nimgs_train == 2
    assert base_cfg.nimgs_test == 2
    assert base_meta["smoke_only_reduced_workload"] is False

    smoke_args = study.parse_args(["--output-root", str(tmp_path / "smoke"), "--smoke-only"])
    smoke_cfg, smoke_meta = study.execution_config_from_args(smoke_args)
    assert smoke_cfg.nepochs == 1
    assert smoke_cfg.nimgs_train == 1
    assert smoke_cfg.nimgs_test == 1
    assert smoke_cfg.size <= 128
    assert smoke_meta["smoke_only_reduced_workload"] is True
    assert smoke_meta["reviewer_facing_metrics"] is False
    assert smoke_meta["effective_config"]["nepochs"] == 1
    assert smoke_meta["effective_config"]["size"] == 96


def test_baseline_comparability_gate_blocks_unsafe_table2_comparison():
    from scripts.studies import probe_mischaracterization_stress_test as study

    gate = study.evaluate_baseline_comparability(
        {"baseline": {"status": "ok", "amp_ssim": 0.50, "amp_psnr": 10.0}},
        adopt_rerun_baseline=False,
    )

    assert gate["status"] == "pivot_no_numeric_stress_table"
    assert gate["claim_safe"] is False
    assert gate["baseline_policy"] == "rerun_baseline_not_comparable"


def test_baseline_comparability_gate_accepts_either_ssim_or_psnr_tolerance():
    from scripts.studies import probe_mischaracterization_stress_test as study

    gate = study.evaluate_baseline_comparability(
        {
            "baseline": {
                "status": "ok",
                "amp_ssim": study.TABLE2_AMP_SSIM + 0.01,
                "amp_psnr": study.TABLE2_AMP_PSNR - 10.0,
            }
        },
        adopt_rerun_baseline=False,
    )

    assert gate["status"] == "comparable_to_table2"
    assert gate["claim_safe"] is True
    assert gate["table2_comparable"] is True


def test_baseline_comparability_gate_allows_explicit_rerun_baseline_adoption():
    from scripts.studies import probe_mischaracterization_stress_test as study

    gate = study.evaluate_baseline_comparability(
        {"baseline": {"status": "ok", "amp_ssim": 0.50, "amp_psnr": 10.0}},
        adopt_rerun_baseline=True,
    )

    assert gate["status"] == "adopt_rerun_baseline"
    assert gate["claim_safe"] is True
    assert gate["baseline_policy"] == "adopt_rerun_baseline_no_old_numeric_comparison"


def test_infrastructure_failure_gate_counts_repeated_nonbaseline_failures():
    from scripts.studies import probe_mischaracterization_stress_test as study

    gate = study.evaluate_infrastructure_failure_gate(
        {
            "baseline": {"status": "ok"},
            "a": {"status": "failed", "error": "RuntimeError('xla failed')"},
            "b": {"status": "failed", "error": "RuntimeError('xla failed')"},
            "c": {"status": "failed", "error": "RuntimeError('xla failed')"},
        }
    )

    assert gate["status"] == "stop_full_grid"
    assert gate["claim_safe"] is False
    assert gate["failed_condition_count"] == 3


def test_mild_perturbation_gate_records_sensitivity_claim_boundary():
    from scripts.studies import probe_mischaracterization_stress_test as study

    gate = study.evaluate_mild_perturbation_gate(
        {
            "baseline": {"status": "ok", "amp_ssim": 0.90, "amp_psnr": 70.0},
            "phase_curvature_scale_0p75": {
                "status": "ok",
                "amp_ssim": 0.75,
                "amp_psnr": 69.0,
            },
            "amplitude_blur_sigma_px_0p5": {
                "status": "ok",
                "amp_ssim": 0.89,
                "amp_psnr": 69.5,
            },
            "phase_noise_sigma_rad_0p1pi_seed11": {
                "status": "ok",
                "amp_ssim": 0.88,
                "amp_psnr": 69.0,
            },
        }
    )

    assert gate["status"] == "sensitivity_language_required"
    assert gate["export_allowed"] is True
    assert gate["robustness_claim_safe"] is False
    assert gate["mild_conditions"]["phase_curvature_scale_0p75"]["amp_ssim_drop"] == pytest.approx(0.15)


def test_export_paper_assets_requires_gates_and_carries_baseline_policy(tmp_path):
    from scripts.studies import probe_mischaracterization_stress_test as study

    output_root = tmp_path / "run"
    output_root.mkdir()
    (output_root / "figures").mkdir()
    (output_root / "figures" / "probe_mischaracterization_stress.png").write_bytes(b"png")
    args = study.parse_args(
        [
            "--output-root",
            str(output_root / "unused"),
            "--paper-root",
            str(tmp_path / "paper"),
        ]
    )
    smoke = {"decision": "fixed_wrong_probe_training"}
    condition_results = {"baseline": {"status": "ok", "amp_ssim": 0.91, "amp_psnr": 68.0}}
    baseline_gate = {
        "claim_safe": True,
        "baseline_policy": "adopt_rerun_baseline_no_old_numeric_comparison",
    }

    with pytest.raises(RuntimeError, match="mild perturbation"):
        study.export_paper_assets(
            args,
            output_root,
            smoke,
            condition_results,
            baseline_gate=baseline_gate,
            mild_perturbation_gate={"export_allowed": False, "status": "blocked_missing_mild_conditions"},
        )

    study.export_paper_assets(
        args,
        output_root,
        smoke,
        condition_results,
        baseline_gate=baseline_gate,
        mild_perturbation_gate={"export_allowed": True, "robustness_claim_safe": False},
    )
    payload = json.loads(
        (Path(args.paper_root) / "data" / "probe_mischaracterization_metrics.json").read_text()
    )
    assert payload["baseline_policy"] == "adopt_rerun_baseline_no_old_numeric_comparison"
    assert payload["mild_perturbation_gate"]["robustness_claim_safe"] is False


def test_export_paper_assets_requires_existing_stress_figure(tmp_path):
    from scripts.studies import probe_mischaracterization_stress_test as study

    output_root = tmp_path / "run"
    output_root.mkdir()
    args = study.parse_args(
        [
            "--output-root",
            str(output_root / "unused"),
            "--paper-root",
            str(tmp_path / "paper"),
        ]
    )

    with pytest.raises(RuntimeError, match="stress figure"):
        study.export_paper_assets(
            args,
            output_root,
            {"decision": "fixed_wrong_probe_training"},
            {"baseline": {"status": "ok", "amp_ssim": 0.91, "amp_psnr": 68.0}},
            baseline_gate={
                "claim_safe": True,
                "baseline_policy": "rerun_baseline_table2_comparable",
            },
            mild_perturbation_gate={"export_allowed": True, "robustness_claim_safe": False},
        )


def test_export_paper_assets_ignores_child_source_provenance_manifest(tmp_path, monkeypatch):
    from scripts.studies import probe_mischaracterization_stress_test as study

    output_root = tmp_path / "run"
    output_root.mkdir()
    (output_root / "figures").mkdir()
    (output_root / "figures" / "probe_mischaracterization_stress.png").write_bytes(b"png")
    stale_child_source_provenance = {
        "all_match": False,
        "parent_fingerprint_sha256": "old-fingerprint",
        "children": {
            "baseline": {
                "fingerprint_sha256": "different-fingerprint",
                "matches_parent": False,
            }
        },
    }
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "child_source_provenance": stale_child_source_provenance,
            }
        )
    )
    monkeypatch.setattr(study, "_git_commit", lambda: "runabc1")
    args = study.parse_args(
        [
            "--output-root",
            str(output_root / "unused"),
            "--paper-root",
            str(tmp_path / "paper"),
        ]
    )

    study.export_paper_assets(
        args,
        output_root,
        {"decision": "fixed_wrong_probe_training"},
        {"baseline": {"status": "ok", "amp_ssim": 0.91, "amp_psnr": 68.0}},
        baseline_gate={
            "claim_safe": True,
            "baseline_policy": "rerun_baseline_table2_comparable",
        },
        mild_perturbation_gate={"export_allowed": True, "robustness_claim_safe": False},
    )

    payload = json.loads(
        (Path(args.paper_root) / "data" / "probe_mischaracterization_metrics.json").read_text()
    )
    assert payload["git_commit"] == "runabc1"
    assert "source_provenance" not in payload
    assert "child_source_provenance" not in payload
    assert "source_provenance_policy" not in payload


def test_reset_tf_state_deletes_repeated_training_model_singletons(monkeypatch):
    from scripts.studies import probe_mischaracterization_stress_test as study

    fake_model = types.SimpleNamespace(
        _lazy_cache={"model": object()},
        _model_construction_done=True,
        autoencoder=object(),
        diffraction_to_obj=object(),
        autoencoder_no_nll=object(),
    )
    import ptycho

    monkeypatch.setattr(ptycho, "model", fake_model, raising=False)
    monkeypatch.setitem(sys.modules, "ptycho.model", fake_model)

    study.reset_tf_state(clear_model_cache=True)

    assert fake_model._lazy_cache == {}
    assert fake_model._model_construction_done is False
    assert not hasattr(fake_model, "autoencoder")
    assert not hasattr(fake_model, "diffraction_to_obj")
    assert not hasattr(fake_model, "autoencoder_no_nll")


def test_run_condition_failure_payload_preserves_preflight(monkeypatch, tmp_path):
    from scripts.studies import probe_mischaracterization_stress_test as study
    from ptycho.workflows import grid_lines_workflow

    true_probe = _toy_probe()
    assumed_probe = true_probe * np.exp(0.1j)
    sim = _toy_sim(true_probe)

    def clone_for_test(container, assumed):
        return _ToyContainer(
            X=container._X_np,
            coords_nominal=container._coords_nominal_np,
            coords_true=container._coords_true_np,
            probe=assumed,
        )

    def fail_after_preflight(*_args, **_kwargs):
        raise RuntimeError("simulated oom after preflight")

    monkeypatch.setattr(study, "reset_tf_state", lambda clear_model_cache=True: None)
    monkeypatch.setattr(study, "clone_container_with_probe", clone_for_test)
    monkeypatch.setattr(grid_lines_workflow, "configure_legacy_params", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(grid_lines_workflow, "train_pinn_model", fail_after_preflight)

    result = study.run_condition(
        condition=study.PerturbationCondition("phase_test", "phase_curvature_scale", value=0.75),
        sim=sim,
        true_probe=true_probe,
        assumed_probe=assumed_probe,
        branch_decision="fixed_wrong_probe_training",
        cfg=types.SimpleNamespace(),
        output_root=tmp_path,
        canonical_data_checksums=study.build_canonical_data_checksums(sim),
    )

    assert result["status"] == "failed"
    assert result["condition_probe_policy"] == "assumed_probe_replaces_container_probe"
    assert result["preflight"]["condition_probe_policy"] == "assumed_probe_replaces_container_probe"
    assert result["canonical_data_checksums"] == study.build_canonical_data_checksums(sim)
    persisted = json.loads((tmp_path / "conditions" / "phase_test" / "metrics.json").read_text())
    assert persisted["preflight"]["condition_probe_policy"] == "assumed_probe_replaces_container_probe"


def test_script_path_execution_uses_repo_import_root(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    probe_npz = tmp_path / "probe.npz"
    np.savez(probe_npz, probeGuess=_toy_probe())
    output_root = tmp_path / "script_path_dry_run"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/studies/probe_mischaracterization_stress_test.py",
            "--output-root",
            str(output_root),
            "--probe-npz",
            str(probe_npz),
            "--dry-run",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert (output_root / "invocation.json").exists()
