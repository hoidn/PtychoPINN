"""
Tests for ptycho_torch/model_manager.py — PyTorch model persistence layer.

This module validates that the PyTorch persistence functions (`save_torch_bundle`,
`load_torch_bundle`) satisfy the reconstructor persistence contract defined in
specs/ptychodus_api_spec.md §4.6 and maintain archival format parity with
ptycho/model_manager.py TensorFlow implementation.

Critical Behavioral Requirements (from spec §4.6 + Phase D3 callchain):
1. save_torch_bundle MUST produce wts.h5.zip-compatible archives with dual-model structure
2. Archive MUST contain manifest.dill, per-model subdirectories with params.dill snapshots
3. params.dill MUST capture full params.cfg state via dataclass_to_legacy_dict (CONFIG-001)
4. Archive format MUST enable cross-backend loading (PyTorch archives loadable by TF loaders)
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

import pytest
from pathlib import Path
import tempfile
import zipfile
import dill
import numpy as np

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
            n_groups=10,
            neighbor_count=4,
            nphotons=1e9,
            nepochs=5,
            batch_size=4,
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
        params_cfg_snapshot,
        minimal_training_config,
        dummy_torch_models
    ):
        """
        CRITICAL PARITY TEST: save_torch_bundle must produce spec-compliant zip archives.

        Requirement: specs/ptychodus_api_spec.md:192-202 mandates wts.h5.zip format
        with manifest.dill + per-model subdirectories containing params.dill, model weights.

        TensorFlow baseline: ptycho/model_manager.py:346-378 (ModelManager.save_multiple_models)
        Archive schema from Phase D3.A callchain (static.md):
        ```
        wts.h5.zip/
        ├── manifest.dill  # {'models': ['autoencoder', 'diffraction_to_obj'], 'version': '2.0-pytorch'}
        ├── autoencoder/
        │   ├── model.pth  # PyTorch state_dict (replaces model.keras)
        │   └── params.dill  # Full params.cfg snapshot (CONFIG-001 critical)
        └── diffraction_to_obj/
            ├── model.pth
            └── params.dill
        ```

        Red-phase contract (Phase D3.B):
        - Function signature: save_torch_bundle(models_dict, base_path, config)
        - MUST create {base_path}.zip archive
        - MUST include manifest.dill at root with 'models' and 'version' keys
        - MUST create subdirectory per model in models_dict
        - Each model dir MUST contain params.dill with params.cfg snapshot

        Test mechanism:
        - Call save_torch_bundle with dummy models + minimal config
        - Extract and inspect zip contents using zipfile.ZipFile.namelist()
        - Load manifest.dill and validate structure
        - Load params.dill from each model dir and validate CONFIG-001 fields
        """
        # This test will initially FAIL because ptycho_torch.model_manager doesn't exist yet
        pytest.importorskip("ptycho_torch.model_manager", reason="model_manager module not yet implemented")

        from ptycho_torch.model_manager import save_torch_bundle
        from ptycho.config.config import update_legacy_dict

        # Populate params.cfg via config bridge (CONFIG-001 requirement)
        update_legacy_dict(params_cfg_snapshot, minimal_training_config)

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

            # Validate manifest.dill exists
            assert 'manifest.dill' in archive_files, (
                "Archive MUST contain manifest.dill at root "
                "(TensorFlow baseline: model_manager.py:361-364)"
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

            # Validate params.dill exists in each model directory
            assert 'autoencoder/params.dill' in archive_files, (
                "Each model directory MUST contain params.dill "
                "(CONFIG-001 requirement: params.cfg snapshot for load-time restoration)"
            )
            assert 'diffraction_to_obj/params.dill' in archive_files, (
                "Each model directory MUST contain params.dill "
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
            with zf.open('manifest.dill') as manifest_file:
                manifest = dill.load(manifest_file)

            assert 'models' in manifest, "manifest.dill MUST contain 'models' key"
            assert 'version' in manifest, "manifest.dill MUST contain 'version' key"
            assert set(manifest['models']) == {'autoencoder', 'diffraction_to_obj'}, (
                "manifest['models'] MUST list both model names"
            )
            assert manifest['version'] == '2.0-pytorch', (
                "manifest['version'] MUST be '2.0-pytorch' for format detection "
                "(enables cross-backend compatibility checks)"
            )

    def test_params_snapshot(
        self,
        tmp_path,
        params_cfg_snapshot,
        minimal_training_config,
        dummy_torch_models
    ):
        """
        CRITICAL CONFIG-001 TEST: params.dill must capture full params.cfg state.

        Requirement: Phase D3.A callchain finding #1 — TensorFlow loader calls
        `params.cfg.update(loaded_params)` at model_manager.py:119 to restore
        training-time configuration before model reconstruction. PyTorch MUST replicate
        this via dataclass_to_legacy_dict snapshot.

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
        from ptycho.config.config import update_legacy_dict, dataclass_to_legacy_dict

        # Populate params.cfg via config bridge (CONFIG-001 requirement)
        update_legacy_dict(params_cfg_snapshot, minimal_training_config)

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
            with zf.open('autoencoder/params.dill') as params_file:
                loaded_params = dill.load(params_file)

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
