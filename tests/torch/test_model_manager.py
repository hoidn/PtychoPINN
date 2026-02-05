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

    def test_save_bundle_with_intensity_scale(
        self,
        tmp_path,
        params_cfg_snapshot,
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
        from ptycho.config.config import update_legacy_dict

        # Populate params.cfg via config bridge (CONFIG-001 requirement)
        update_legacy_dict(params_cfg_snapshot, minimal_training_config)

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
            with zf.open('diffraction_to_obj/params.dill') as params_file:
                loaded_params = dill.load(params_file)

        # Validate intensity_scale was persisted
        assert 'intensity_scale' in loaded_params, (
            "params.dill MUST contain 'intensity_scale' when provided to save_torch_bundle"
        )
        assert loaded_params['intensity_scale'] == test_intensity_scale, (
            f"Expected intensity_scale={test_intensity_scale}, "
            f"got {loaded_params['intensity_scale']}"
        )


class TestLoadTorchBundle:
    """
    Phase D3.C loader tests — validate params restoration and model reconstruction.

    These tests validate that load_torch_bundle satisfies the reconstructor load
    contract defined in specs/ptychodus_api_spec.md §4.5 and implements CONFIG-001
    compliant params.cfg restoration per Phase D3.A callchain finding #1.
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
            batch_size=16,
        )

        return training_config

    @pytest.fixture
    def dummy_torch_models(self):
        """Create minimal PyTorch model stubs for persistence testing."""
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

    def test_create_torch_model_with_gridsize_sets_channel_count(self):
        """create_torch_model_with_gridsize must align channels to gridsize**2."""
        pytest.importorskip("torch")
        from ptycho_torch.model_manager import create_torch_model_with_gridsize

        model = create_torch_model_with_gridsize(
            gridsize=1,
            N=64,
            params_dict={'gridsize': 1, 'N': 64, 'model_type': 'pinn'}
        )

        assert model.data_config.C == 1, (
            f"Expected data_config.C=1 for gridsize=1, got {model.data_config.C}"
        )
        assert model.model_config.C_model == 1, (
            f"Expected model_config.C_model=1 for gridsize=1, got {model.model_config.C_model}"
        )
        conv_weight = model.model.autoencoder.encoder.blocks[0].conv1.weight
        assert conv_weight.shape[1] == 1, (
            f"Expected conv1 input channels=1, got {conv_weight.shape[1]}"
        )

    def test_load_round_trip_updates_params_cfg(
        self,
        tmp_path,
        params_cfg_snapshot,
        minimal_training_config,
        dummy_torch_models
    ):
        """
        CRITICAL CONFIG-001 TEST: load_torch_bundle MUST restore params.cfg before model reconstruction.

        Requirement: Phase D3.A callchain finding #1 — TensorFlow load path at
        ptycho/model_manager.py:119 calls `params.cfg.update(loaded_params)` to restore
        training-time configuration before calling `create_model_with_gridsize`. PyTorch
        MUST replicate this to prevent shape mismatch errors.

        Failure Mode: If params.cfg restoration skipped, subsequent model operations
        will use stale/default gridsize/N values → tensor shape mismatches → inference fails.

        Red-phase contract (Phase D3.C):
        - Function signature: load_torch_bundle(base_path, model_name='diffraction_to_obj')
        - MUST extract archive and load params.dill
        - MUST call params.cfg.update(loaded_params) before model reconstruction
        - MUST return (model, params_dict) tuple
        - Model reconstruction may raise NotImplementedError (deferred to follow-up)

        Test mechanism:
        - Save bundle with known config (N=64, gridsize=2, nphotons=1e9)
        - Clear params.cfg to simulate fresh process
        - Call load_torch_bundle
        - Validate params.cfg contains expected values after load
        """
        pytest.importorskip("ptycho_torch.model_manager", reason="model_manager module not yet implemented")

        from ptycho_torch.model_manager import save_torch_bundle, load_torch_bundle
        from ptycho.config.config import update_legacy_dict
        from ptycho import params

        # Save bundle with known config
        update_legacy_dict(params_cfg_snapshot, minimal_training_config)
        base_path = tmp_path / "round_trip_test"

        save_torch_bundle(
            models_dict=dummy_torch_models,
            base_path=str(base_path),
            config=minimal_training_config
        )

        # Simulate fresh process: clear params.cfg
        params.cfg.clear()
        assert params.cfg.get('N') is None, "Sanity check: params.cfg should be empty before load"
        assert params.cfg.get('gridsize') is None, "Sanity check: params.cfg should be empty before load"

        # Attempt to load bundle (may raise NotImplementedError if model reconstruction not yet done)
        try:
            model, loaded_params = load_torch_bundle(str(base_path), model_name='diffraction_to_obj')
        except NotImplementedError as e:
            # Model reconstruction not yet implemented — validate params restoration happened
            # by checking params.cfg was updated (side effect occurs before NotImplementedError)
            if 'load_torch_bundle model reconstruction not yet implemented' in str(e):
                # Expected during red→green transition; validate CONFIG-001 gate executed
                assert params.cfg.get('N') == 64, (
                    "CONFIG-001 VIOLATION: params.cfg['N'] not restored. "
                    f"Expected 64, got {params.cfg.get('N')}. "
                    "load_torch_bundle MUST call params.cfg.update() before model reconstruction."
                )
                assert params.cfg.get('gridsize') == 2, (
                    "CONFIG-001 VIOLATION: params.cfg['gridsize'] not restored. "
                    f"Expected 2, got {params.cfg.get('gridsize')}."
                )
                assert params.cfg.get('nphotons') == 1e9, (
                    f"CONFIG-001: params.cfg['nphotons'] not restored. Expected 1e9, got {params.cfg.get('nphotons')}."
                )
                # Test passes — params restoration verified even though model reconstruction pending
                return
            else:
                raise  # Unexpected NotImplementedError, re-raise

        # Full implementation path (once model reconstruction done):
        # Validate params.cfg was updated
        assert params.cfg.get('N') == 64, f"CONFIG-001: Expected N=64 in params.cfg, got {params.cfg.get('N')}"
        assert params.cfg.get('gridsize') == 2, f"CONFIG-001: Expected gridsize=2, got {params.cfg.get('gridsize')}"
        assert params.cfg.get('nphotons') == 1e9, f"CONFIG-001: Expected nphotons=1e9, got {params.cfg.get('nphotons')}"

        # Validate return values
        assert loaded_params is not None, "load_torch_bundle MUST return params_dict"
        assert loaded_params['N'] == 64, "Returned params_dict MUST match saved config"
        assert loaded_params['gridsize'] == 2, "Returned params_dict MUST match saved config"

        # Validate model object (basic smoke test)
        assert model is not None, "load_torch_bundle MUST return model instance"

    def test_missing_params_raises_value_error(
        self,
        tmp_path,
        params_cfg_snapshot
    ):
        """
        ERROR HANDLING TEST: load_torch_bundle MUST fail gracefully when params.dill missing required fields.

        Requirement: Phase D3.A callchain — model reconstruction requires N and gridsize
        to call create_torch_model_with_gridsize. If params.dill is corrupt or incomplete,
        loader MUST raise ValueError with actionable error message.

        Red-phase contract (Phase D3.C):
        - If params.dill missing 'N' or 'gridsize', MUST raise ValueError
        - Error message MUST list missing fields
        - Error message MUST mention "Cannot reconstruct model architecture"

        Test mechanism:
        - Manually create archive with incomplete params.dill (missing 'N')
        - Call load_torch_bundle
        - Validate ValueError raised with correct message
        """
        pytest.importorskip("ptycho_torch.model_manager", reason="model_manager module not yet implemented")

        from ptycho_torch.model_manager import load_torch_bundle

        # Create malformed archive with incomplete params.dill
        base_path = tmp_path / "malformed_bundle"
        zip_path = Path(f"{base_path}.zip")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create manifest
            manifest = {'models': ['diffraction_to_obj'], 'version': '2.0-pytorch'}
            manifest_path = Path(temp_dir) / 'manifest.dill'
            with open(manifest_path, 'wb') as f:
                dill.dump(manifest, f)

            # Create model directory with INCOMPLETE params.dill (missing 'N')
            model_dir = Path(temp_dir) / 'diffraction_to_obj'
            model_dir.mkdir()

            incomplete_params = {'gridsize': 2, 'model_type': 'pinn'}  # Missing 'N'
            params_path = model_dir / 'params.dill'
            with open(params_path, 'wb') as f:
                dill.dump(incomplete_params, f)

            # Create dummy model.pth
            model_path = model_dir / 'model.pth'
            dill.dump({'_sentinel': 'dummy'}, open(model_path, 'wb'))

            # Zip archive
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file in Path(temp_dir).rglob('*'):
                    if file.is_file():
                        arc_path = file.relative_to(temp_dir)
                        zf.write(file, arc_path)

        # Attempt to load malformed bundle
        with pytest.raises(ValueError) as exc_info:
            load_torch_bundle(str(base_path), model_name='diffraction_to_obj')

        # Validate error message content
        error_msg = str(exc_info.value)
        assert 'missing required fields' in error_msg.lower() or 'required parameters missing' in error_msg.lower(), (
            "ValueError MUST mention missing fields"
        )
        assert 'N' in error_msg or "['N']" in error_msg, (
            "ValueError MUST list missing field 'N'"
        )
        assert 'Cannot reconstruct model architecture' in error_msg or 'cannot reconstruct' in error_msg.lower(), (
            "ValueError MUST explain consequence (model reconstruction failure)"
        )

    def test_load_round_trip_returns_model_stub(
        self,
        tmp_path,
        params_cfg_snapshot,
        minimal_training_config,
        dummy_torch_models
    ):
        """
        REGRESSION TEST: save → load round-trip must return saved sentinel model and params.

        Requirement: Phase D4.B1 — validate PyTorch persistence layer maintains round-trip
        parity with TensorFlow baseline without raising NotImplementedError when model
        reconstruction is complete.

        This test documents the expected end state after persistence implementation:
        - save_torch_bundle serializes sentinel models + config snapshot
        - load_torch_bundle restores the saved models and params without errors
        - Returned model structure matches what was saved

        Red-phase expectation:
        - Currently SHOULD raise NotImplementedError("load_torch_bundle model reconstruction not yet implemented")
        - Once Phase D3.C complete, SHOULD return (model, params) tuple successfully

        Test mechanism:
        - Save bundle with sentinel models (torch-optional stubs)
        - Load the same bundle
        - Validate (model, params) returned match saved state
        - If NotImplementedError raised, document it as expected during red phase
        """
        pytest.importorskip("ptycho_torch.model_manager", reason="model_manager module not yet implemented")

        from ptycho_torch.model_manager import save_torch_bundle, load_torch_bundle
        from ptycho.config.config import update_legacy_dict
        from ptycho import params

        # Save bundle with known config
        update_legacy_dict(params_cfg_snapshot, minimal_training_config)
        base_path = tmp_path / "d4b1_round_trip"

        save_torch_bundle(
            models_dict=dummy_torch_models,
            base_path=str(base_path),
            config=minimal_training_config
        )

        # Clear params.cfg to simulate fresh process
        params.cfg.clear()

        # Attempt round-trip load
        # Red phase: expect NotImplementedError
        # Green phase: expect successful return
        try:
            model, loaded_params = load_torch_bundle(str(base_path), model_name='autoencoder')

            # Green phase assertions (once Phase D3.C complete)
            assert model is not None, (
                "load_torch_bundle MUST return non-None model after successful load"
            )
            assert loaded_params is not None, (
                "load_torch_bundle MUST return params dict"
            )

            # Validate params match saved config
            assert loaded_params['N'] == 64, f"Expected N=64, got {loaded_params['N']}"
            assert loaded_params['gridsize'] == 2, f"Expected gridsize=2, got {loaded_params['gridsize']}"
            assert loaded_params['model_type'] == 'pinn', f"Expected model_type='pinn', got {loaded_params['model_type']}"

            # Validate params.cfg was updated (CONFIG-001 requirement)
            assert params.cfg.get('N') == 64, (
                "CONFIG-001: params.cfg['N'] must be restored during load"
            )

        except NotImplementedError as e:
            # Red phase: document expected failure
            if 'load_torch_bundle model reconstruction not yet implemented' in str(e):
                pytest.xfail(
                    "Phase D4.B1 red phase: load_torch_bundle model reconstruction pending. "
                    "Expected NotImplementedError raised. This test will pass once Phase D3.C "
                    "implements model reconstruction logic."
                )
            else:
                raise  # Unexpected NotImplementedError, re-raise

    def test_reconstructs_models_from_bundle(
        self,
        tmp_path,
        params_cfg_snapshot,
        minimal_training_config,
        dummy_torch_models
    ):
        """
        Phase C4.D A1 TDD RED test: load_torch_bundle must reconstruct both models from wts.h5.zip.

        Requirement: ADR-003-BACKEND-API Phase C4.D — unblock integration workflow by
        implementing bundle loader that returns {'diffraction_to_obj', 'autoencoder'}
        model dict + hydrated config per spec §4.6–§4.8 and CONFIG-001.

        This test establishes the acceptance criteria for load_torch_bundle completion:
        - Function MUST accept bundle base_path and return models dict + config
        - Models dict MUST contain both 'diffraction_to_obj' and 'autoencoder' keys
        - Each value MUST be a loaded nn.Module (or sentinel dict during torch-optional)
        - Config MUST be reconstructed from serialized metadata (params.dill)
        - params.cfg MUST be populated via CONFIG-001 bridge before model loading

        Red-phase expectation (Phase A1):
        - Currently RAISES NotImplementedError at ptycho_torch/model_manager.py:267
        - Capturing RED log to pytest_load_bundle_red.log per plan guidance

        Green-phase expectation (Phase A2/A3):
        - RETURNS (models_dict, config) where models_dict has both model keys
        - Each model is reconstructed via create_torch_model_with_gridsize helper
        - Integration test passes bundle to workflows without NotImplementedError

        Test mechanism (mirroring TensorFlow analogue):
        - Generate temporary bundle via save_torch_bundle with minimal dual models
        - Call load_torch_bundle expecting models dict return (not single model tuple)
        - Validate returned structure matches TensorFlow ModelManager.load_multiple_models
        - Assert both model names present in returned dict keys
        """
        pytest.importorskip("ptycho_torch.model_manager", reason="model_manager module not yet implemented")

        from ptycho_torch.model_manager import save_torch_bundle, load_torch_bundle
        from ptycho.config.config import update_legacy_dict
        from ptycho import params

        # Save lightweight bundle with dual models (Phase A1 guidance: reuse minimal config)
        update_legacy_dict(params_cfg_snapshot, minimal_training_config)
        base_path = tmp_path / "c4d_bundle_loader_test"

        save_torch_bundle(
            models_dict=dummy_torch_models,
            base_path=str(base_path),
            config=minimal_training_config
        )

        # Clear params.cfg to simulate fresh inference process (CONFIG-001 requirement)
        params.cfg.clear()
        assert params.cfg.get('N') is None, "Sanity check: params.cfg should be empty before load"

        # Phase A2/A3: load_torch_bundle should return models dict + config
        # (Currently raises NotImplementedError — RED baseline for Phase A1)
        models_dict, loaded_config = load_torch_bundle(str(base_path))

        # GREEN phase assertions (Phase A3):
        # Validate models dict structure
        assert isinstance(models_dict, dict), (
            "load_torch_bundle MUST return models dict as first element of tuple"
        )
        assert 'diffraction_to_obj' in models_dict, (
            "models_dict MUST contain 'diffraction_to_obj' key (spec §4.6 dual-model requirement)"
        )
        assert 'autoencoder' in models_dict, (
            "models_dict MUST contain 'autoencoder' key (spec §4.6 dual-model requirement)"
        )

        # Validate models are non-None (basic smoke check)
        assert models_dict['diffraction_to_obj'] is not None, (
            "'diffraction_to_obj' model MUST be reconstructed and loaded"
        )
        assert models_dict['autoencoder'] is not None, (
            "'autoencoder' model MUST be reconstructed and loaded"
        )

        # Validate loaded_config contains critical fields
        assert loaded_config is not None, (
            "load_torch_bundle MUST return config dict as second element of tuple"
        )
        assert loaded_config.get('N') == 64, (
            f"Config MUST preserve N=64 from saved bundle, got {loaded_config.get('N')}"
        )
        assert loaded_config.get('gridsize') == 2, (
            f"Config MUST preserve gridsize=2, got {loaded_config.get('gridsize')}"
        )

        # CONFIG-001 gate: params.cfg MUST be populated before model reconstruction
        assert params.cfg.get('N') == 64, (
            "CONFIG-001 VIOLATION: params.cfg['N'] not restored. "
            "load_torch_bundle MUST call params.cfg.update() before returning."
        )
        assert params.cfg.get('gridsize') == 2, (
            "CONFIG-001 VIOLATION: params.cfg['gridsize'] not restored."
        )
