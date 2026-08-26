"""
Tests for ptycho_torch/model_manager.py — PyTorch model persistence layer.

This module validates that the PyTorch persistence functions (`save_torch_bundle`,
the JSON manifest reader, and the legacy migration script) satisfy the
reconstructor persistence contract defined in specs/ptychodus_api_spec.md §4.6
and maintain archival format parity with the ptycho/model_manager.py TensorFlow
implementation.

Critical Behavioral Requirements (from spec §4.6 + Phase D3 callchain):
1. save_torch_bundle MUST produce wts.h5.zip-compatible archives with dual-model structure
2. Archive MUST contain manifest.json, per-model subdirectories with params.json snapshots
3. params.json MUST capture the config-derived legacy projection (CONFIG-001)
4. Archives identify their backend; unsupported cross-backend loads fail descriptively
5. All persistence functions must be torch-optional (importable when PyTorch unavailable)

Test Strategy:
- Red-phase: document required archive structure via failing tests using zip inspection
- Green-phase: implement save_torch_bundle producing spec-compliant archives
- torch-optional: module structure follows test_config_bridge.py pattern (guarded imports)

Artifacts (Phase D3.B):
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T110500Z/phase_d3_writer.md
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T110500Z/pytest_red.log
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T110500Z/pytest_green.log
"""

import copy
import json
from pathlib import Path
import zipfile

import pytest

# Add to conftest.py TORCH_OPTIONAL_MODULES if not already present
# This test must run without torch runtime


class TestSaveTorchBundle:
    """
    Phase D3.B archive writer tests — validate wts.h5.zip structure compliance.

    These tests validate that save_torch_bundle produces archives matching the
    TensorFlow ModelManager.save_multiple_models format documented in
    ptycho/model_manager.py:346-378 and required by specs/ptychodus_api_spec.md:192-202.
    """

    @pytest.fixture
    def params_cfg_snapshot(self):
        """Snapshot and restore params.cfg state across tests."""
        from ptycho import params
        original = params.cfg.copy()
        yield params.cfg
        params.cfg.clear()
        params.cfg.update(original)

    @pytest.fixture
    def minimal_training_config(self):
        """Create minimal TrainingConfig fixture with canonical params."""
        from ptycho.config.config import TrainingConfig, ModelConfig

        model_config = ModelConfig(
            N=64,
            gridsize=2,
            model_type='pinn',
            n_filters_scale=1.0,
            amp_activation='sigmoid',
            object_big=False,
            probe_big=False,
            pad_object=False,
        )

        training_config = TrainingConfig(
            model=model_config,
            train_data_file=Path("/tmp/dummy_train.npz"),
            test_data_file=Path("/tmp/dummy_test.npz"),
            training_groups=10,
            neighbor_count=4,
            nphotons=1e9,
            nepochs=5,
            batch_size=16,
        )

        return training_config

    @pytest.fixture
    def dummy_torch_models(self):
        """Create minimal PyTorch model stubs for persistence testing.

        Returns dictionary with 'autoencoder' and 'diffraction_to_obj' keys
        matching the dual-model bundle requirement from spec §4.6.

        Note: Models are sentinel dicts when torch unavailable; real nn.Module
        instances when torch is available.
        """
        try:
            import torch
            import torch.nn as nn

            class DummyModel(nn.Module):
                """Minimal PyTorch model for testing persistence."""
                def __init__(self, name):
                    super().__init__()
                    self.name = name
                    self.conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)
                    self.fc = nn.Linear(16 * 64 * 64, 128)

                def forward(self, x):
                    x = self.conv(x)
                    x = x.view(x.size(0), -1)
                    return self.fc(x)

            return {
                'autoencoder': DummyModel('autoencoder'),
                'diffraction_to_obj': DummyModel('diffraction_to_obj'),
            }
        except ImportError:
            # Torch unavailable — return sentinel dicts for structure testing
            return {
                'autoencoder': {'_sentinel': 'torch_unavailable', 'name': 'autoencoder'},
                'diffraction_to_obj': {'_sentinel': 'torch_unavailable', 'name': 'diffraction_to_obj'},
            }

    def test_archive_structure(
        self,
        tmp_path,
        minimal_training_config,
        dummy_torch_models
    ):
        """
        CRITICAL PARITY TEST: save_torch_bundle must produce spec-compliant zip archives.

        Requirement: specs/ptychodus_api_spec.md:192-202 mandates wts.h5.zip format
        with manifest.json + per-model subdirectories containing params.json, model weights.

        TensorFlow baseline: ptycho/model_manager.py:346-378 (ModelManager.save_multiple_models)
        Archive schema (PyTorch JSON-manifest era):
        ```
        wts.h5.zip/
        ├── manifest.json  # {'models': [...], 'version': '2.0-pytorch', 'manifest_version': ...}
        ├── autoencoder/
        │   ├── model.pth  # PyTorch state_dict (replaces model.keras)
        │   └── params.json  # Config-derived legacy projection (CONFIG-001)
        └── diffraction_to_obj/
            ├── model.pth
            └── params.json
        ```

        Red-phase contract (Phase D3.B):
        - Function signature: save_torch_bundle(models_dict, base_path, config)
        - MUST create {base_path}.zip archive
        - MUST include manifest.json at root with 'models' and 'version' keys
        - MUST create subdirectory per model in models_dict
        - Each model dir MUST contain the config-derived params.json projection

        Test mechanism:
        - Call save_torch_bundle with dummy models + minimal config
        - Extract and inspect zip contents using zipfile.ZipFile.namelist()
        - Load manifest.json and validate structure
        - Load params.json from each model dir and validate CONFIG-001 fields
        """
        # This test will initially FAIL because ptycho_torch.model_manager doesn't exist yet
        pytest.importorskip("ptycho_torch.model_manager", reason="model_manager module not yet implemented")

        from ptycho_torch.model_manager import save_torch_bundle

        # Define output path
        base_path = tmp_path / "test_bundle"

        # Call save_torch_bundle
        save_torch_bundle(
            models_dict=dummy_torch_models,
            base_path=str(base_path),
            config=minimal_training_config
        )

        # Validate zip archive was created
        zip_path = Path(f"{base_path}.zip")
        assert zip_path.exists(), (
            f"save_torch_bundle MUST create {zip_path.name} archive "
            "(spec §4.6 requirement)"
        )

        # Inspect archive structure
        with zipfile.ZipFile(zip_path, 'r') as zf:
            archive_files = set(zf.namelist())

            # Validate manifest.json exists
            assert 'manifest.json' in archive_files, (
                "Archive MUST contain manifest.json at root "
                "(spec §4.6 versioned JSON manifest era)"
            )

            # Validate per-model subdirectories exist
            assert any('autoencoder/' in f for f in archive_files), (
                "Archive MUST contain autoencoder/ subdirectory "
                "(dual-model bundle requirement per spec §4.6)"
            )
            assert any('diffraction_to_obj/' in f for f in archive_files), (
                "Archive MUST contain diffraction_to_obj/ subdirectory "
                "(dual-model bundle requirement per spec §4.6)"
            )

            # Validate params.json exists in each model directory
            assert 'autoencoder/params.json' in archive_files, (
                "Each model directory MUST contain params.json "
                "(CONFIG-001 requirement: params.cfg snapshot for load-time restoration)"
            )
            assert 'diffraction_to_obj/params.json' in archive_files, (
                "Each model directory MUST contain params.json "
                "(CONFIG-001 requirement)"
            )

            # Validate model weights exist (format: model.pth for PyTorch)
            assert 'autoencoder/model.pth' in archive_files, (
                "Each model directory MUST contain model.pth "
                "(PyTorch state_dict; replaces TensorFlow model.keras)"
            )
            assert 'diffraction_to_obj/model.pth' in archive_files, (
                "Each model directory MUST contain model.pth"
            )

            # Load and validate manifest structure
            with zf.open('manifest.json') as manifest_file:
                manifest = json.load(manifest_file)

            assert 'models' in manifest, "manifest.json MUST contain 'models' key"
            assert 'version' in manifest, "manifest.json MUST contain 'version' key"
            assert set(manifest['models']) == {'autoencoder', 'diffraction_to_obj'}, (
                "manifest['models'] MUST list both model names"
            )
            assert manifest['version'] == '2.0-pytorch', (
                "manifest['version'] MUST be '2.0-pytorch' for format detection "
                "(enables cross-backend compatibility checks)"
            )
            assert manifest['backend'] == 'pytorch'

    def test_params_snapshot(
        self,
        tmp_path,
        minimal_training_config,
        dummy_torch_models
    ):
        """
        CRITICAL CONFIG-001 TEST: params.dill captures the config projection.

        The archive must carry the flat compatibility projection needed by the
        public wrapper's post-validation CONFIG-001 restoration. The projection
        itself comes only from ``dataclass_to_legacy_dict``.

        Red-phase contract (Phase D3.B):
        - params.dill MUST be valid dill-serialized dictionary
        - MUST contain all CONFIG-001 critical fields: N, gridsize, model_type, nphotons
        - MUST contain intensity_scale (inference requirement per spec §4.4)
        - Values MUST match minimal_training_config after dataclass_to_legacy_dict translation

        Test mechanism:
        - Call save_torch_bundle and extract params.dill from archive
        - Load params.dill and validate presence of critical fields
        - Assert values match expected config bridge output
        """
        # This test will initially FAIL because ptycho_torch.model_manager doesn't exist yet
        pytest.importorskip("ptycho_torch.model_manager", reason="model_manager module not yet implemented")

        from ptycho_torch.model_manager import save_torch_bundle
        from ptycho.config.config import dataclass_to_legacy_dict

        # Capture expected params snapshot for comparison
        expected_params = dataclass_to_legacy_dict(minimal_training_config)

        # Define output path
        base_path = tmp_path / "test_params"

        # Call save_torch_bundle
        save_torch_bundle(
            models_dict=dummy_torch_models,
            base_path=str(base_path),
            config=minimal_training_config
        )

        # Extract and validate params.dill from autoencoder directory
        zip_path = Path(f"{base_path}.zip")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            with zf.open('autoencoder/params.json') as params_file:
                loaded_params = json.load(params_file)

        # Validate params.dill is a dictionary
        assert isinstance(loaded_params, dict), (
            "params.dill MUST contain a dictionary (TensorFlow baseline format)"
        )

        # Validate CONFIG-001 critical fields
        assert 'N' in loaded_params, "params.dill MUST contain 'N' (model input size)"
        assert 'gridsize' in loaded_params, "params.dill MUST contain 'gridsize' (grouping parameter)"
        assert 'model_type' in loaded_params, "params.dill MUST contain 'model_type' (pinn/supervised)"
        assert 'nphotons' in loaded_params, "params.dill MUST contain 'nphotons' (physics scaling)"

        # Validate values match expected config bridge output
        assert loaded_params['N'] == 64, f"Expected N=64, got {loaded_params['N']}"
        assert loaded_params['gridsize'] == 2, f"Expected gridsize=2, got {loaded_params['gridsize']}"
        assert loaded_params['model_type'] == 'pinn', f"Expected model_type='pinn', got {loaded_params['model_type']}"
        assert loaded_params['nphotons'] == 1e9, f"Expected nphotons=1e9, got {loaded_params['nphotons']}"

        # Validate version tag for format detection
        assert '_version' in loaded_params, "params.dill MUST contain '_version' tag"
        assert loaded_params['_version'] == '2.0-pytorch', (
            "params.dill['_version'] MUST be '2.0-pytorch' for backend identification"
        )

        # Validate intensity_scale presence (inference requirement)
        # Note: intensity_scale may be added during training; for now, ensure field is documented
        # Full validation deferred to Phase D3.C (loader implementation)
        if 'intensity_scale' in loaded_params:
            assert isinstance(loaded_params['intensity_scale'], (int, float)), (
                "intensity_scale MUST be numeric when present"
            )

    def test_save_bundle_with_intensity_scale(
        self,
        tmp_path,
        minimal_training_config,
        dummy_torch_models
    ):
        """
        Phase B2: Validate that save_torch_bundle persists non-default intensity_scale.

        This test ensures that when an explicit intensity_scale is provided to
        save_torch_bundle, it is correctly stored in params.dill and will be available
        during inference loading. This satisfies the Phase B2 requirement to persist
        the learned scale from training.

        Test mechanism:
        - Call save_torch_bundle with explicit intensity_scale=2.5
        - Extract params.dill from the bundle
        - Verify intensity_scale field equals the provided value
        """
        from ptycho_torch.model_manager import save_torch_bundle

        # Define output path
        base_path = tmp_path / "test_intensity_scale"

        # Call save_torch_bundle with explicit intensity_scale
        test_intensity_scale = 2.5
        save_torch_bundle(
            models_dict=dummy_torch_models,
            base_path=str(base_path),
            config=minimal_training_config,
            intensity_scale=test_intensity_scale
        )

        # Extract and validate params.dill contains the intensity_scale
        zip_path = Path(f"{base_path}.zip")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            with zf.open('diffraction_to_obj/params.json') as params_file:
                loaded_params = json.load(params_file)

        # Validate intensity_scale was persisted
        assert 'intensity_scale' in loaded_params, (
            "params.dill MUST contain 'intensity_scale' when provided to save_torch_bundle"
        )
        assert loaded_params['intensity_scale'] == test_intensity_scale, (
            f"Expected intensity_scale={test_intensity_scale}, "
            f"got {loaded_params['intensity_scale']}"
        )

    def test_save_snapshot_ignores_poisoned_global_state(
        self,
        tmp_path,
        params_cfg_snapshot,
        minimal_training_config,
        dummy_torch_models,
    ):
        """The save codec is derived only from its config and explicit scale."""
        from ptycho.config.config import dataclass_to_legacy_dict
        from ptycho_torch.model_manager import save_torch_bundle

        config_before = copy.deepcopy(minimal_training_config)
        params_cfg_snapshot.clear()
        params_cfg_snapshot.update(
            {
                "N": 999,
                "intensity_scale": 137.0,
                "poisoned_global_only": True,
            }
        )
        base_path = tmp_path / "poisoned_global"

        save_torch_bundle(
            models_dict=dummy_torch_models,
            base_path=str(base_path),
            config=minimal_training_config,
        )

        expected = dataclass_to_legacy_dict(minimal_training_config)
        expected["intensity_scale"] = 1.0
        expected["_version"] = "2.0-pytorch"
        with zipfile.ZipFile(f"{base_path}.zip", "r") as archive:
            autoencoder_bytes = archive.read("autoencoder/params.json")
            reconstruction_bytes = archive.read(
                "diffraction_to_obj/params.json"
            )

        assert json.loads(autoencoder_bytes) == expected
        assert autoencoder_bytes == reconstruction_bytes
        assert autoencoder_bytes == json.dumps(expected).encode("utf-8")
        assert minimal_training_config == config_before

    def test_save_rejects_sentinel_models(self, tmp_path, minimal_training_config):
        from ptycho_torch.model_manager import save_torch_bundle

        with pytest.raises(RuntimeError, match="trained nn.Module"):
            save_torch_bundle(
                {
                    "autoencoder": {"_sentinel": True},
                    "diffraction_to_obj": {"_sentinel": True},
                },
                str(tmp_path / "sentinel"),
                minimal_training_config,
            )



class TestMigrateLegacyBundle:
    """Migration script round-trip: dill metadata-free bundle -> strict load."""

    def test_migrates_metadata_free_dill_bundle(self, tmp_path):
        import dill
        import io
        import subprocess
        import sys
        import torch

        from ptycho.config.config import (
            TrainingConfig,
            ModelConfig,
            dataclass_to_legacy_dict,
        )
        from scripts.migrate_legacy_bundle import (
            create_torch_model_with_gridsize,
            _seal_identity,
        )

        repo_root = Path(__file__).resolve().parents[2]

        config = TrainingConfig(model=ModelConfig(N=64, gridsize=1), nphotons=1e9)
        archived = dataclass_to_legacy_dict(config)
        archived["intensity_scale"] = 123.0
        archived["_version"] = "2.0-pytorch"
        model = create_torch_model_with_gridsize(1, 64, archived)

        source = tmp_path / "legacy"
        source.mkdir()
        with zipfile.ZipFile(source / "wts.h5.zip", "w") as zf:
            zf.writestr(
                "manifest.dill",
                dill.dumps(
                    {
                        "models": ["autoencoder", "diffraction_to_obj"],
                        "version": "2.0-pytorch",
                    }
                ),
            )
            for model_name in ("autoencoder", "diffraction_to_obj"):
                zf.writestr(
                    f"{model_name}/params.dill",
                    dill.dumps(archived),
                )
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                zf.writestr(f"{model_name}/model.pth", buffer.getvalue())

        out = tmp_path / "migrated"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "ptycho_torch.migrate_bundle",
                str(source),
                str(out),
            ],
            cwd=repo_root,
            text=True,
            capture_output=True,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr

        # Decoded-identity equality: the migrated sealed payload matches the
        # identity a fresh legacy reconstruction would seal.
        expected = _seal_identity(model)
        with zipfile.ZipFile(out / "wts.h5.zip", "r") as zf:
            assert "manifest.dill" not in zf.namelist()
            payload = torch.load(
                io.BytesIO(zf.read("torch_scaling_metadata.pt")),
                map_location="cpu",
                weights_only=True,
            )
        assert payload == expected

        from ptycho_torch.workflows.components import load_inference_bundle_torch

        models, params = load_inference_bundle_torch(out)
        assert params["intensity_scale"] == 123.0
        assert params["scale_contract_version"] == "legacy_v1"
        assert params["measurement_domain"] == "normalized_amplitude"
        for name in ("autoencoder", "diffraction_to_obj"):
            loaded = models[name]
            assert loaded.data_config.scale_contract_version == "legacy_v1"
            assert loaded.data_config.measurement_domain == "normalized_amplitude"
            assert loaded.model_config.intensity_scale == 123.0

    def test_dill_era_bundle_routes_to_migration_script(self, tmp_path):
        """A pre-JSON dill manifest names the migration script, not a raw KeyError."""
        import dill

        from ptycho_torch.workflows.components import load_inference_bundle_torch

        bundle_dir = tmp_path / "legacy"
        bundle_dir.mkdir()
        with zipfile.ZipFile(bundle_dir / "wts.h5.zip", "w") as zf:
            zf.writestr(
                "manifest.dill",
                dill.dumps(
                    {
                        "models": ["autoencoder", "diffraction_to_obj"],
                        "version": "2.0-pytorch",
                    }
                ),
            )
            zf.writestr(
                "autoencoder/params.dill",
                dill.dumps({"_version": "2.0-pytorch", "N": 64, "gridsize": 1}),
            )

        with pytest.raises(ValueError, match=r"migrate_bundle"):
            load_inference_bundle_torch(bundle_dir)
