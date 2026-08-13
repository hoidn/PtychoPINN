"""Explicit amplitude physics gain (PROBE-RANK-001, design 2026-07-12 §3.3).

The banned flat (B, H, W) probe layout used to multiply the predicted
amplitude by the batch size — an accidental training gain that demonstrably
conditioned amplitude-mode training (amp SSIM 0.486 vs 0.896 at B=16). The
gain survives as ``ModelConfig.amplitude_physics_gain``: an explicit,
batch-size-independent, provenance-carrying constant, plumbed through
``create_training_payload`` -> Lightning hparams -> the amplitude-mode
forward, validated by the scaling contract (finite, > 0; exactly 1.0 for
rectangular_scaled/CI modes, fail-closed).

Contract: docs/specs/spec-ptycho-torch-probe-layout.md.
"""


from dataclasses import replace

import numpy as np
import pytest
import torch

from ptycho_torch.config_params import (
    DataConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
)

N = 16


@pytest.fixture
def mock_train_npz(tmp_path):
    """Minimal DATA-001-compliant NPZ for create_training_payload."""
    n_images = 8
    npz_path = tmp_path / "mock_train.npz"
    np.savez(
        npz_path,
        diffraction=np.random.rand(n_images, 64, 64).astype(np.float32),
        probeGuess=np.random.rand(64, 64).astype(np.complex64),
        objectGuess=np.random.rand(128, 128).astype(np.complex64),
        xcoords=np.linspace(0, 1, n_images).astype(np.float64),
        ycoords=np.linspace(0, 1, n_images).astype(np.float64),
        scan_index=np.arange(n_images).astype(np.int32),
    )
    return npz_path


@pytest.mark.torch
class TestConfigPlumbing:
    def test_model_config_defaults_to_unit_gain(self):
        assert ModelConfig().amplitude_physics_gain == 1.0

    def test_training_payload_plumbs_gain_and_audits_it(self, mock_train_npz, tmp_path):
        from ptycho_torch.config_factory import create_training_payload

        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=tmp_path / "out",
            overrides={"n_groups": 8, "amplitude_physics_gain": 16.0},
        )
        assert payload.pt_model_config.amplitude_physics_gain == 16.0
        assert payload.overrides_applied["amplitude_physics_gain"] == 16.0

    def test_training_payload_audits_default_gain(self, mock_train_npz, tmp_path):
        from ptycho_torch.config_factory import create_training_payload

        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=tmp_path / "out",
            overrides={"n_groups": 8},
        )
        assert payload.pt_model_config.amplitude_physics_gain == 1.0
        assert payload.overrides_applied["amplitude_physics_gain"] == 1.0

    def test_gain_lands_in_lightning_hparams(self):
        from ptycho_torch.model import PtychoPINN_Lightning

        module = PtychoPINN_Lightning(
            ModelConfig(
                object_big=False,
                probe_big=False,
                C_model=1,
                C_forward=1,
                amplitude_physics_gain=16.0,
            ),
            DataConfig(N=64, C=1, grid_size=(1, 1)),  # CNN autoencoder needs N>=64
            TrainingConfig(device="cpu", torch_loss_mode="mae"),
            InferenceConfig(),
        )
        assert module.hparams["model_config"]["amplitude_physics_gain"] == 16.0


@pytest.mark.torch
class TestScalingContractValidation:
    """Design §3.3 + §8 case 4: finite, > 0 everywhere; exactly 1.0 for
    rectangular_scaled/CI modes (fail-closed)."""

    @staticmethod
    def _configs(gain, physics_forward_mode="amplitude", **data_overrides):
        data_cfg = DataConfig(N=N, C=1, grid_size=(1, 1), **data_overrides)
        model_cfg = ModelConfig(
            physics_forward_mode=physics_forward_mode,
            amplitude_physics_gain=gain,
        )
        train_cfg = TrainingConfig(torch_loss_mode="poisson")
        return data_cfg, model_cfg, train_cfg

    def test_amplitude_mode_accepts_non_unit_gain(self):
        from ptycho_torch.scaling_contract import validate_scale_contract

        validate_scale_contract(*self._configs(16.0))  # must not raise

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
    def test_rejects_nonpositive_or_nonfinite_gain_in_amplitude_mode(self, bad):
        from ptycho_torch.scaling_contract import validate_scale_contract

        with pytest.raises(ValueError, match="amplitude_physics_gain"):
            validate_scale_contract(*self._configs(bad))

    def test_rejects_non_unit_gain_for_rectangular_ci_mode(self):
        from ptycho_torch.scaling_contract import validate_scale_contract

        with pytest.raises(ValueError, match="amplitude_physics_gain"):
            validate_scale_contract(
                *self._configs(16.0, physics_forward_mode="rectangular_scaled")
            )

    def test_rejects_non_unit_gain_for_rectangular_legacy_profile(self):
        from ptycho_torch.scaling_contract import validate_scale_contract

        with pytest.raises(ValueError, match="amplitude_physics_gain"):
            validate_scale_contract(
                *self._configs(
                    16.0,
                    physics_forward_mode="rectangular_scaled",
                    scale_contract_version="legacy_v1",
                    measurement_domain="normalized_amplitude",
                )
            )

    def test_unit_gain_passes_rectangular_mode(self):
        from ptycho_torch.scaling_contract import validate_scale_contract

        resolved = validate_scale_contract(
            *self._configs(1.0, physics_forward_mode="rectangular_scaled")
        )
        assert resolved is not None

    def test_missing_attribute_treated_as_unit_gain(self):
        """Duck-typed configs without the field (pre-fix checkpoints, test
        stand-ins) must resolve to the 1.0 default, not crash."""
        from types import SimpleNamespace

        from ptycho_torch.scaling_contract import validate_amplitude_physics_gain

        legacy = SimpleNamespace(physics_forward_mode="amplitude")
        assert validate_amplitude_physics_gain(legacy) == 1.0


@pytest.mark.torch
class TestDerivedAmplitudePhysicsGain:
    @staticmethod
    def _inputs():
        measured = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 1.0], [4.0, 3.0]],
            ],
            dtype=np.float32,
        )
        objects = np.array(
            [
                [[1.0 + 0.0j, 0.5 + 0.5j], [0.25 + 0.0j, 0.75 - 0.25j]],
                [[0.5 + 0.0j, 1.0 - 0.5j], [0.75 + 0.25j, 0.25 + 0.0j]],
            ],
            dtype=np.complex64,
        )
        probe = np.array(
            [[1.0 + 0.0j, 0.5 + 0.25j], [0.25 - 0.5j, 0.75 + 0.0j]],
            dtype=np.complex64,
        )
        return measured, objects, probe

    def test_derived_gain_pins_documented_formula_and_metadata(self):
        from ptycho_torch.helper import normalize_probe_like_tf
        from ptycho_torch.scaling_contract import (
            derive_legacy_amplitude_physics_gain,
        )

        measured, objects, probe = self._inputs()
        probe_scale = 4.0
        normalized_probe, _ = normalize_probe_like_tf(probe, probe_scale)
        effective_probe = normalized_probe.astype(np.complex128) / probe_scale
        reference = np.fft.fftshift(
            np.abs(
                np.fft.fft2(
                    objects.astype(np.complex128) * effective_probe[None, ...]
                )
            ),
            axes=(-2, -1),
        ) / 2
        measured_energy = np.square(measured, dtype=np.float64)
        r = np.sqrt(4 / np.mean(np.sum(measured_energy, axis=(-2, -1))))
        expected = r * np.sqrt(
            np.sum(measured_energy) / np.sum(np.square(reference, dtype=np.float64))
        )

        record = derive_legacy_amplitude_physics_gain(
            measured,
            objects,
            probe,
            probe_scale=probe_scale,
        )
        repeated = derive_legacy_amplitude_physics_gain(
            measured.copy(),
            objects.copy(),
            probe.copy(),
            probe_scale=probe_scale,
        )

        assert record.value == pytest.approx(expected, rel=1e-12)
        assert np.isfinite(record.value) and record.value > 0
        assert record == repeated
        assert record.provenance == "derived"
        assert record.method == "normalized_amplitude_physical_gain"
        assert record.version == "legacy-amplitude-physics-gain-v1"
        assert record.input_statistics["sample_count"] == 2
        assert record.input_statistics["N"] == 2
        assert record.input_statistics["probe_scale"] == 4.0
        assert record.factory_overrides() == {
            "amplitude_physics_gain": record.value
        }

    def test_gain_record_serialization_is_plain_and_digest_free(self):
        import json

        from ptycho_torch.scaling_contract import (
            amplitude_physics_gain_record_to_json,
            derive_legacy_amplitude_physics_gain,
        )

        measured, objects, probe = self._inputs()
        record = derive_legacy_amplitude_physics_gain(
            measured,
            objects,
            probe,
            probe_scale=4.0,
        )

        metadata = record.to_metadata()
        assert set(metadata) == {
            "value",
            "provenance",
            "method",
            "version",
            "input_statistics",
        }
        assert "sha256" not in json.dumps(metadata, sort_keys=True)
        assert json.loads(amplitude_physics_gain_record_to_json(record)) == metadata

    @pytest.mark.parametrize(
        "bad_probe_scale",
        [True, 0.0, -1.0, float("nan"), float("inf"), "4.0"],
    )
    def test_derived_gain_rejects_invalid_probe_scale(self, bad_probe_scale):
        from ptycho_torch.scaling_contract import (
            derive_legacy_amplitude_physics_gain,
        )

        measured, objects, probe = self._inputs()
        with pytest.raises((TypeError, ValueError), match="probe_scale"):
            derive_legacy_amplitude_physics_gain(
                measured,
                objects,
                probe,
                probe_scale=bad_probe_scale,
            )

    def test_explicit_advanced_override_is_never_relabelled_derived(self):
        from ptycho_torch.scaling_contract import resolve_amplitude_physics_gain

        measured, objects, probe = self._inputs()
        record = resolve_amplitude_physics_gain(
            measured,
            objects,
            probe,
            probe_scale=4.0,
            override=2.75,
            physics_forward_mode="amplitude",
        )

        assert record.value == 2.75
        assert record.provenance == "override"
        assert record.method == "advanced_model_override"
        assert record.input_statistics == {}

    def test_record_propagates_through_factory_model_spec_and_artifact_identity(
        self, mock_train_npz, tmp_path
    ):
        from ptycho_torch.artifact_schema import encode_artifact_identity
        from ptycho_torch.config_factory import create_training_payload
        from ptycho_torch.scaling_contract import (
            derive_legacy_amplitude_physics_gain,
        )

        measured, objects, probe = self._inputs()
        record = derive_legacy_amplitude_physics_gain(
            measured, objects, probe, probe_scale=4.0
        )
        derived_payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=tmp_path / "derived",
            overrides={"n_groups": 8, **record.factory_overrides()},
        )
        unit_payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=tmp_path / "unit",
            overrides={"n_groups": 8, "amplitude_physics_gain": 1.0},
        )

        assert derived_payload.pt_model_config.amplitude_physics_gain == record.value
        assert (
            derived_payload.model_spec.to_model_config().amplitude_physics_gain
            == record.value
        )
        assert (
            derived_payload.model_spec.to_payload()["model_config"]
            ["amplitude_physics_gain"]
            == record.value
        )
        derived_identity = encode_artifact_identity(
            derived_payload.model_spec,
            derived_payload.pt_data_config,
            derived_payload.pt_training_config,
            derived_payload.pt_inference_config,
        )
        unit_identity = encode_artifact_identity(
            unit_payload.model_spec,
            unit_payload.pt_data_config,
            unit_payload.pt_training_config,
            unit_payload.pt_inference_config,
        )
        assert derived_identity != unit_identity
        assert (
            derived_identity["model_spec"]["model_config"]
            ["amplitude_physics_gain"]
            == record.value
        )

    def test_grouped_samples_and_masked_multimode_probe_match_flat_derivation(self):
        from ptycho_torch.helper import normalize_probe_like_tf
        from ptycho_torch.probe_mask import resolve_probe_mask_np
        from ptycho_torch.scaling_contract import (
            derive_legacy_amplitude_physics_gain,
        )

        rng = np.random.default_rng(11)
        measured = rng.uniform(0.1, 2.0, size=(4, 4, 4)).astype(np.float32)
        objects = (
            rng.uniform(0.2, 1.0, size=(4, 4, 4))
            * np.exp(1j * rng.uniform(-1.0, 1.0, size=(4, 4, 4)))
        ).astype(np.complex64)
        probe = (
            rng.uniform(0.25, 1.0, size=(2, 4, 4))
            * np.exp(1j * rng.uniform(-0.5, 0.5, size=(2, 4, 4)))
        ).astype(np.complex64)
        measured_grouped = measured.reshape(2, 2, 4, 4).transpose(0, 2, 3, 1)
        objects_grouped = objects.reshape(2, 2, 4, 4).transpose(0, 2, 3, 1)

        flat = derive_legacy_amplitude_physics_gain(
            measured,
            objects,
            probe,
            probe_scale=4.0,
            probe_mask=True,
            probe_mask_sigma=0.75,
            probe_mask_diameter=3.0,
        )
        grouped = derive_legacy_amplitude_physics_gain(
            measured_grouped,
            objects_grouped,
            probe,
            probe_scale=4.0,
            probe_mask=True,
            probe_mask_sigma=0.75,
            probe_mask_diameter=3.0,
        )

        assert grouped.value == flat.value
        assert grouped.input_statistics == flat.input_statistics
        assert grouped.input_statistics["sample_count"] == 4
        assert grouped.input_statistics["probe_mode_count"] == 2
        assert grouped.input_statistics["probe_mask_settings"] == {
            "probe_mask": True,
            "probe_mask_tensor": None,
            "probe_mask_sigma": 0.75,
            "probe_mask_diameter": 3.0,
        }
        normalized_probe, _ = normalize_probe_like_tf(
            probe,
            4.0,
            probe_mask=True,
            probe_mask_sigma=0.75,
            probe_mask_diameter=3.0,
        )
        mask = resolve_probe_mask_np(
            4,
            probe_mask=True,
            probe_mask_sigma=0.75,
            probe_mask_diameter=3.0,
        )
        effective_probe = (
            normalized_probe.astype(np.complex128) * mask[None, ...]
        ) / 4.0
        forward = np.fft.fftshift(
            np.abs(
                np.fft.fft2(
                    objects.astype(np.complex128)[:, None, ...]
                    * effective_probe[None, ...],
                    axes=(-2, -1),
                ).sum(axis=1)
            ),
            axes=(-2, -1),
        ) / 4.0
        measured_squared = np.square(measured, dtype=np.float64)
        r = np.sqrt(
            16.0 / np.mean(np.sum(measured_squared, axis=(-2, -1)))
        )
        expected = r * np.sqrt(
            np.sum(measured_squared)
            / np.sum(np.square(forward, dtype=np.float64))
        )
        assert flat.value == pytest.approx(expected, rel=1e-12)

    def test_gain_record_metadata_is_a_detached_plain_copy(self):
        from ptycho_torch.scaling_contract import derive_legacy_amplitude_physics_gain

        measured, objects, probe = self._inputs()
        record = derive_legacy_amplitude_physics_gain(
            measured,
            objects,
            probe,
            probe_scale=4.0,
            probe_mask=False,
        )

        metadata = record.to_metadata()
        metadata["input_statistics"]["probe_mask_settings"]["probe_mask_sigma"] = 9
        assert (
            record.to_metadata()["input_statistics"]["probe_mask_settings"]
            ["probe_mask_sigma"]
            == 1.0
        )

    @pytest.mark.parametrize(
        ("value", "provenance", "method", "version", "message"),
        [
            (
                2.0,
                "scale_contract_fixed",
                "rectangular_scale_contract_fixed",
                "amplitude-physics-gain-resolution-v1",
                "exactly 1.0",
            ),
            (
                2.0,
                "override",
                "wrong_override_method",
                "amplitude-physics-gain-resolution-v1",
                "requires method/version",
            ),
            (
                2.0,
                "invented",
                "invented",
                "invented-v1",
                "provenance",
            ),
        ],
    )
    def test_gain_record_rejects_invalid_provenance_semantics(
        self, value, provenance, method, version, message
    ):
        from ptycho_torch.scaling_contract import AmplitudePhysicsGainRecord

        with pytest.raises(ValueError, match=message):
            AmplitudePhysicsGainRecord(
                value=value,
                provenance=provenance,
                method=method,
                version=version,
                input_statistics={},
            )

    def test_ci_rectangular_contract_stays_fixed_at_unit_gain(self):
        from ptycho_torch.scaling_contract import resolve_amplitude_physics_gain

        measured, objects, probe = self._inputs()
        record = resolve_amplitude_physics_gain(
            measured,
            objects,
            probe,
            probe_scale=4.0,
            physics_forward_mode="rectangular_scaled",
        )

        assert record.value == 1.0
        assert record.provenance == "scale_contract_fixed"
        assert record.factory_overrides() == {"amplitude_physics_gain": 1.0}

        unit_override = resolve_amplitude_physics_gain(
            measured,
            objects,
            probe,
            probe_scale=4.0,
            override=1.0,
            physics_forward_mode="rectangular_scaled",
        )
        assert unit_override.provenance == "scale_contract_fixed"
        assert unit_override.method == "rectangular_scale_contract_fixed"
        assert unit_override.value == 1.0

        with pytest.raises(ValueError, match="amplitude_physics_gain"):
            resolve_amplitude_physics_gain(
                measured,
                objects,
                probe,
                probe_scale=4.0,
                override=2.0,
                physics_forward_mode="rectangular_scaled",
            )

    @pytest.mark.parametrize(
        ("bad_measured", "message"),
        [
            (np.zeros((1, 2, 2), dtype=np.float32), "measured_amplitude"),
            (np.full((1, 2, 2), np.nan, dtype=np.float32), "finite"),
        ],
    )
    def test_derived_gain_rejects_degenerate_training_arrays(
        self, bad_measured, message
    ):
        from ptycho_torch.scaling_contract import (
            derive_legacy_amplitude_physics_gain,
        )

        _, objects, probe = self._inputs()
        with pytest.raises(ValueError, match=message):
            derive_legacy_amplitude_physics_gain(
                bad_measured,
                objects[:1],
                probe,
                probe_scale=4.0,
            )


@pytest.mark.torch
class TestAmplitudePhysicsGainBundleRecord:
    _REPRESENTATIVE_ARCHIVE_MEMBERS = {
        "autoencoder/model.pth": b"\x00\x01torch-autoencoder-weights\xff",
        "diffraction_to_obj/model.pth": bytes(range(64)),
    }

    @staticmethod
    def _record(value=2.5):
        from ptycho_torch.scaling_contract import resolve_amplitude_physics_gain

        measured, objects, probe = TestDerivedAmplitudePhysicsGain._inputs()
        return resolve_amplitude_physics_gain(
            measured,
            objects,
            probe,
            probe_scale=4.0,
            override=value,
        )

    @staticmethod
    def _payload(mock_train_npz, output_dir, gain):
        from ptycho_torch.config_factory import create_training_payload

        return create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=output_dir,
            overrides={"n_groups": 8, "amplitude_physics_gain": gain},
        )

    @staticmethod
    def _model_for_payload(payload, *, model_config=None):
        from types import SimpleNamespace

        return SimpleNamespace(
            _model_spec=payload.model_spec,
            data_config=payload.pt_data_config,
            model_config=model_config or payload.pt_model_config,
            training_config=payload.pt_training_config,
            inference_config=payload.pt_inference_config,
            hparams={},
            get_ci_statistics=lambda: None,
        )

    @staticmethod
    def _write_minimal_bundle(
        bundle_path,
        payload,
        *,
        representative_archive_members=None,
    ):
        import dill
        import zipfile

        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "models": ["autoencoder", "diffraction_to_obj"],
            "version": "2.0-pytorch",
            "backend": "pytorch",
        }
        params_payload = {"_version": "2.0-pytorch"}
        with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.dill", dill.dumps(manifest))
            for model_name in manifest["models"]:
                archive.writestr(
                    f"{model_name}/params.dill",
                    dill.dumps(params_payload),
                )
            members = (
                TestAmplitudePhysicsGainBundleRecord
                ._REPRESENTATIVE_ARCHIVE_MEMBERS
                if representative_archive_members is None
                else representative_archive_members
            )
            for name, content in members.items():
                archive.writestr(name, content)

    @staticmethod
    def _read_unmanaged_archive_members(bundle_path):
        import zipfile

        from ptycho_torch.workflows import components

        rewritten_members = {
            "manifest.dill",
            components._BUNDLE_SCALING_METADATA,
            components._BUNDLE_AMPLITUDE_PHYSICS_GAIN_RECORD,
        }
        with zipfile.ZipFile(bundle_path, "r") as archive:
            names = archive.namelist()
            assert len(names) == len(set(names))
            return {
                name: archive.read(name)
                for name in names
                if name not in rewritten_members
            }

    def test_json_sidecar_roundtrip_is_plain_and_deterministic(self):
        import json

        from ptycho_torch.scaling_contract import (
            amplitude_physics_gain_record_from_json,
            amplitude_physics_gain_record_to_json,
        )

        record = self._record()
        encoded = amplitude_physics_gain_record_to_json(record)
        expected = record.to_metadata()

        assert json.loads(encoded) == expected
        assert encoded == json.dumps(
            expected,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        decoded = amplitude_physics_gain_record_from_json(encoded)
        assert decoded == record
        assert amplitude_physics_gain_record_to_json(decoded) == encoded

    def test_json_sidecar_rejects_a_missing_plain_record_field(self):
        import json

        from ptycho_torch.scaling_contract import (
            amplitude_physics_gain_record_from_json,
        )

        payload = self._record().to_metadata()
        del payload["version"]
        with pytest.raises(ValueError, match="missing 'version'"):
            amplitude_physics_gain_record_from_json(json.dumps(payload))

    def test_persisted_sidecar_roundtrips_complete_record(
        self,
        mock_train_npz,
        tmp_path,
    ):
        import zipfile

        from ptycho_torch.scaling_contract import (
            amplitude_physics_gain_record_from_json,
        )
        from ptycho_torch.workflows import components

        record = self._record()
        payload = self._payload(mock_train_npz, tmp_path / "out", gain=record.value)
        bundle_path = tmp_path / "bundle" / "wts.h5.zip"
        self._write_minimal_bundle(bundle_path, payload)
        unmanaged_before = self._read_unmanaged_archive_members(bundle_path)

        components._persist_bundle_scaling_metadata(
            bundle_path,
            self._model_for_payload(payload),
            amplitude_physics_gain_record=record,
        )

        with zipfile.ZipFile(bundle_path, "r") as archive:
            assert components._BUNDLE_AMPLITUDE_PHYSICS_GAIN_RECORD in archive.namelist()
            encoded = archive.read(
                components._BUNDLE_AMPLITUDE_PHYSICS_GAIN_RECORD
            )
        assert self._read_unmanaged_archive_members(bundle_path) == unmanaged_before
        assert amplitude_physics_gain_record_from_json(encoded) == record

    def test_metadata_rewrite_without_current_record_removes_stale_sidecar(
        self,
        mock_train_npz,
        tmp_path,
    ):
        import zipfile

        from ptycho_torch.workflows import components

        record = self._record()
        payload = self._payload(mock_train_npz, tmp_path / "out", gain=record.value)
        bundle_path = tmp_path / "bundle" / "wts.h5.zip"
        self._write_minimal_bundle(bundle_path, payload)
        unmanaged_before = self._read_unmanaged_archive_members(bundle_path)
        model = self._model_for_payload(payload)
        components._persist_bundle_scaling_metadata(
            bundle_path,
            model,
            amplitude_physics_gain_record=record,
        )
        assert self._read_unmanaged_archive_members(bundle_path) == unmanaged_before

        components._persist_bundle_scaling_metadata(bundle_path, model)

        with zipfile.ZipFile(bundle_path, "r") as archive:
            assert (
                components._BUNDLE_AMPLITUDE_PHYSICS_GAIN_RECORD
                not in archive.namelist()
            )
        assert self._read_unmanaged_archive_members(bundle_path) == unmanaged_before

    def test_loader_pins_one_archive_generation_across_atomic_replacement(
        self,
        mock_train_npz,
        monkeypatch,
        tmp_path,
    ):
        import os
        import zipfile

        from ptycho_torch.workflows import components

        live_path = tmp_path / "live" / "wts.h5.zip"
        replacement_path = tmp_path / "replacement" / "wts.h5.zip"
        old_weights = {
            "autoencoder/model.pth": b"generation-a-autoencoder",
            "diffraction_to_obj/model.pth": b"generation-a-reconstruction",
        }
        new_weights = {
            "autoencoder/model.pth": b"generation-b-autoencoder",
            "diffraction_to_obj/model.pth": b"generation-b-reconstruction",
        }
        old_payload = self._payload(
            mock_train_npz,
            tmp_path / "old-output",
            gain=1.0,
        )
        new_payload = self._payload(
            mock_train_npz,
            tmp_path / "new-output",
            gain=2.0,
        )
        self._write_minimal_bundle(
            live_path,
            old_payload,
            representative_archive_members=old_weights,
        )
        self._write_minimal_bundle(
            replacement_path,
            new_payload,
            representative_archive_members=new_weights,
        )
        components._persist_bundle_scaling_metadata(
            live_path,
            self._model_for_payload(old_payload),
        )
        components._persist_bundle_scaling_metadata(
            replacement_path,
            self._model_for_payload(new_payload),
        )

        read_metadata = components._read_bundle_scaling_metadata
        supplied_snapshot_paths = []

        def read_metadata_then_replace(path):
            metadata = read_metadata(path)
            os.replace(replacement_path, live_path)
            return metadata

        def reconstruct_from_supplied_generation(
            _archive_path,
            zip_path,
            **kwargs,
        ):
            supplied_snapshot_paths.append(zip_path)
            gain = (
                kwargs["identity"]
                .model_spec.to_model_config()
                .amplitude_physics_gain
            )
            with zipfile.ZipFile(zip_path, "r") as archive:
                models = {
                    name: (gain, archive.read(f"{name}/model.pth"))
                    for name in kwargs["manifest"]["models"]
                }
            return models, dict(kwargs["params_dict"]), kwargs["identity"]

        monkeypatch.setattr(
            components,
            "_read_bundle_scaling_metadata",
            read_metadata_then_replace,
        )
        monkeypatch.setattr(
            components,
            "_reconstruct_inference_bundle_explicit",
            reconstruct_from_supplied_generation,
        )

        models, _ = components.load_inference_bundle_torch(live_path.parent)

        assert models == {
            "autoencoder": (1.0, old_weights["autoencoder/model.pth"]),
            "diffraction_to_obj": (
                1.0,
                old_weights["diffraction_to_obj/model.pth"],
            ),
        }
        with zipfile.ZipFile(live_path, "r") as archive:
            assert archive.read("diffraction_to_obj/model.pth") == (
                new_weights["diffraction_to_obj/model.pth"]
            )
        assert len(supplied_snapshot_paths) == 1
        assert not supplied_snapshot_paths[0].exists()

    def test_loader_rejects_duplicate_members_without_mutating_legacy_params(
        self,
        mock_train_npz,
        tmp_path,
    ):
        import dill
        import zipfile

        from ptycho import params
        from ptycho_torch.workflows import components

        payload = self._payload(mock_train_npz, tmp_path / "out", gain=1.0)
        bundle_path = tmp_path / "bundle" / "wts.h5.zip"
        self._write_minimal_bundle(bundle_path, payload)
        components._persist_bundle_scaling_metadata(
            bundle_path,
            self._model_for_payload(payload),
        )
        with pytest.warns(UserWarning, match="Duplicate name"):
            with zipfile.ZipFile(bundle_path, "a") as archive:
                archive.writestr(
                    "manifest.dill",
                    dill.dumps({"models": ["forged"]}),
                )
        params_before = dict(params.cfg)

        with pytest.raises(ValueError, match="duplicate.*manifest.dill"):
            components.load_inference_bundle_torch(bundle_path.parent)

        assert params.cfg == params_before

    def test_loader_rejects_sidecar_value_that_disagrees_with_modelspec(
        self,
        mock_train_npz,
        monkeypatch,
        tmp_path,
    ):
        import zipfile

        from ptycho_torch.scaling_contract import (
            amplitude_physics_gain_record_to_json,
        )
        from ptycho_torch.workflows import components

        payload = self._payload(mock_train_npz, tmp_path / "out", gain=1.0)
        bundle_path = tmp_path / "bundle" / "wts.h5.zip"
        self._write_minimal_bundle(bundle_path, payload)
        components._persist_bundle_scaling_metadata(
            bundle_path,
            self._model_for_payload(payload),
        )
        with zipfile.ZipFile(bundle_path, "a", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                components._BUNDLE_AMPLITUDE_PHYSICS_GAIN_RECORD,
                amplitude_physics_gain_record_to_json(self._record()),
            )
        monkeypatch.setattr(
            components,
            "_reconstruct_inference_bundle_explicit",
            lambda *_args, **_kwargs: pytest.fail(
                "record/ModelSpec mismatch reached model reconstruction"
            ),
        )

        with pytest.raises(ValueError, match="ModelSpec"):
            components.load_inference_bundle_torch(bundle_path.parent)

    def test_loader_returns_strictly_matched_record_and_allows_absent_legacy_sidecar(
        self,
        mock_train_npz,
        monkeypatch,
        tmp_path,
    ):
        from ptycho import params
        from ptycho_torch.workflows import components

        record = self._record()
        # Reuse the fixture bytes so both bundles resolve identical current metadata.
        (tmp_path / "copy.npz").write_bytes(mock_train_npz.read_bytes())
        payload = self._payload(tmp_path / "copy.npz", tmp_path / "out", gain=record.value)

        def fake_reconstruct(_archive_path, _zip_path, **kwargs):
            return {"diffraction_to_obj": object()}, dict(kwargs["params_dict"]), kwargs["identity"]

        monkeypatch.setattr(
            components,
            "_reconstruct_inference_bundle_explicit",
            fake_reconstruct,
        )
        params_before = dict(params.cfg)
        try:
            strict_path = tmp_path / "strict" / "wts.h5.zip"
            self._write_minimal_bundle(strict_path, payload)
            components._persist_bundle_scaling_metadata(
                strict_path,
                self._model_for_payload(payload),
                amplitude_physics_gain_record=record,
            )
            _, strict_params = components.load_inference_bundle_torch(
                strict_path.parent,
            )
            assert strict_params["amplitude_physics_gain_record"] == record
            assert "amplitude_physics_gain_record" not in params.cfg

            legacy_path = tmp_path / "legacy" / "wts.h5.zip"
            self._write_minimal_bundle(legacy_path, payload)
            components._persist_bundle_scaling_metadata(
                legacy_path,
                self._model_for_payload(payload),
            )
            _, legacy_params = components.load_inference_bundle_torch(
                legacy_path.parent
            )
            assert "amplitude_physics_gain_record" not in legacy_params
        finally:
            params.cfg.clear()
            params.cfg.update(params_before)

    def test_run_results_expose_record_metadata_and_written_bundle_path(
        self,
        mock_train_npz,
        monkeypatch,
        tmp_path,
    ):
        from ptycho_torch.workflows import components

        record = self._record()
        output_dir = tmp_path / "out"
        payload = self._payload(mock_train_npz, output_dir, gain=record.value)
        model = self._model_for_payload(payload)
        captured = {}

        monkeypatch.setattr(
            components,
            "_ensure_container",
            lambda *_args, **_kwargs: object(),
        )

        def fake_train(*_args, **kwargs):
            captured.update(kwargs)
            return {
                "models": {
                    "autoencoder": model,
                    "diffraction_to_obj": model,
                },
                "bundle_path": output_dir / "wts.h5.zip",
            }

        monkeypatch.setattr(
            components,
            "_train_with_lightning",
            fake_train,
        )

        _, _, results = components.run_cdi_example_torch(
            object(),
            None,
            payload.tf_training_config,
            resolved_payload=payload,
            amplitude_physics_gain_record=record,
        )

        assert captured["persist_bundle"] is True
        assert captured["amplitude_physics_gain_record"] is record
        assert results["amplitude_physics_gain_record"] is record
        assert results["amplitude_physics_gain_metadata"] == record.to_metadata()
        assert results["bundle_path"] == output_dir / "wts.h5.zip"


@pytest.mark.torch
class TestForwardApplication:
    """Design §3.3: applied ONCE, multiplicatively, to the predicted
    amplitude inside the amplitude-mode forward; rectangular_scaled path
    untouched; inference (forward_predict) never applies it."""

    @staticmethod
    def _forward_model(gain, **model_overrides):
        from ptycho_torch.model import ForwardModel

        model_cfg = ModelConfig(
            object_big=False,
            C_model=1,
            C_forward=1,
            amplitude_physics_gain=gain,
            **model_overrides,
        )
        data_cfg = DataConfig(N=N, C=1, grid_size=(1, 1))
        return ForwardModel(model_cfg, data_cfg)

    @staticmethod
    def _inputs(batch=2):
        torch.manual_seed(0)
        x = (
            torch.randn(batch, 1, N, N) + 1j * torch.randn(batch, 1, N, N)
        ).to(torch.complex64)
        probe = (
            torch.randn(1, 1, 1, N, N) + 1j * torch.randn(1, 1, 1, N, N)
        ).to(torch.complex64)
        ones = torch.ones(batch, 1, 1, 1)
        eids = torch.zeros(batch, dtype=torch.long)
        return x, probe, ones, eids

    def test_gain_multiplies_amplitude_prediction_exactly_once(self):
        x, probe, ones, eids = self._inputs()
        with torch.no_grad():
            base = self._forward_model(1.0).forward(x, None, None, probe, ones, eids)
            gained = self._forward_model(16.0).forward(x, None, None, probe, ones, eids)
        # 16 is a power of two: the multiply is exact, so bit-equality holds.
        assert torch.equal(gained, 16.0 * base)

    def test_unit_gain_is_bit_identical_noop(self):
        x, probe, ones, eids = self._inputs()
        fm = self._forward_model(1.0)
        with torch.no_grad():
            out = fm.forward(x, None, None, probe, ones, eids)
            # Manual reference chain (probe mask disabled -> ones).
            illuminated = x.unsqueeze(2) * probe
            import ptycho_torch.helper as hh

            ref, _ = hh.pad_and_diffract(illuminated, pad=False)
            ref = fm.scaler.inv_scale(ref, ones)
        assert torch.equal(out, ref)

    def test_gain_is_read_live_from_model_config(self):
        """The sealed-checkpoint tie-back sets the gain on the loaded
        module's (shared) model_config; the forward must honor it without
        reconstruction."""
        x, probe, ones, eids = self._inputs()
        fm = self._forward_model(1.0)
        with torch.no_grad():
            base = fm.forward(x, None, None, probe, ones, eids)
            fm.model_config.amplitude_physics_gain = 4.0
            gained = fm.forward(x, None, None, probe, ones, eids)
        assert torch.equal(gained, 4.0 * base)

    def test_rectangular_scaled_forward_ignores_gain(self):
        """The gain application site must leave the rectangular_scaled chain
        untouched (its contract validator separately rejects non-1.0; this
        pins the forward-site isolation itself)."""
        x, probe, ones, eids = self._inputs()
        rect_kwargs = dict(physics_forward_mode="rectangular_scaled")
        with torch.no_grad():
            base = self._forward_model(1.0, **rect_kwargs).forward(
                x, None, None, probe, ones, eids
            )
            torch.manual_seed(0)  # rect_scaler params are deterministic anyway
            gained = self._forward_model(16.0, **rect_kwargs).forward(
                x, None, None, probe, ones, eids
            )
        assert torch.equal(gained, base)

    def test_inference_forward_predict_never_applies_gain(self):
        from ptycho_torch.model import PtychoPINN

        n_model = 64  # CNN autoencoder needs N>=64
        data_cfg = DataConfig(N=n_model, C=1, grid_size=(1, 1))
        train_cfg = TrainingConfig(device="cpu", torch_loss_mode="mae")

        def build(gain):
            torch.manual_seed(7)
            return PtychoPINN(
                ModelConfig(
                    object_big=False,
                    probe_big=False,
                    C_model=1,
                    C_forward=1,
                    amplitude_physics_gain=gain,
                ),
                data_cfg,
                train_cfg,
            ).eval()

        x = torch.rand(2, 1, n_model, n_model)
        positions = torch.zeros(2, 1, 1, 2)
        probe = torch.ones(1, 1, 1, n_model, n_model, dtype=torch.complex64)
        ones = torch.ones(2, 1, 1, 1)
        with torch.no_grad():
            pred_1 = build(1.0).forward_predict(x, positions, probe, ones)
            pred_16 = build(16.0).forward_predict(x, positions, probe, ones)
        assert torch.equal(pred_1, pred_16)
