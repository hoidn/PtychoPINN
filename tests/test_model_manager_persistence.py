# file: tests/test_model_manager_persistence.py

import unittest
import tempfile
import zipfile
import dill
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np
import tensorflow as tf
import sys

# Add project root to path to allow for ptycho imports
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ptycho import params as p  # noqa: E402
from ptycho.grouping import CENTERED_NEAREST_GROUPING_CONTRACT  # noqa: E402
from ptycho.probe import get_default_probe  # noqa: E402

# Initialize probe before importing model to avoid KeyError
p.set('N', 64)
p.set('probe.type', 'gaussian')
p.set('probe.photons', 1e10)
probe = get_default_probe(64)
p.set('probe', probe)

from ptycho.model_manager import ModelManager  # noqa: E402
from ptycho.model import create_model_with_gridsize  # noqa: E402

class TestModelManagerPersistence(unittest.TestCase):

    def setUp(self):
        """Create a temporary directory and store original params."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.original_params = p.cfg.copy()
        p.set('N', 64)
        p.set('gridsize', 1)
        p.set('intensity_scale', 1.0)
        p.set('probe.type', 'gaussian')
        p.set('probe.photons', 1e10)
        print(f"\nCreated temp dir: {self.test_dir.name}")

    def tearDown(self):
        """Clean up the temporary directory and restore original params."""
        self.test_dir.cleanup()
        p.cfg.clear()
        p.cfg.update(self.original_params)
        print("Cleaned up temp dir and restored params.")

    def test_parameter_restoration_on_load(self):
        """
        CRITICAL TEST: Verify that loading a model restores its saved params.cfg,
        overwriting the current session's configuration.
        """
        print("--- Running Test: Parameter Restoration ---")
        # 1. Save a model with a specific, non-default configuration
        p.set('N', 128)
        p.set('gridsize', 2)
        p.set('nphotons', 1e8)
        model_to_save, _ = create_model_with_gridsize(gridsize=2, N=128)
        
        model_path = Path(self.test_dir.name) / "param_test_model"
        ModelManager.save_model(model_to_save, str(model_path), {}, 1.0)

        # 2. Change the current session's parameters to something different
        p.set('N', 64)
        p.set('gridsize', 1)
        p.set('nphotons', 1e9)
        
        # 3. Load the model
        _ = ModelManager.load_model(str(model_path))
        
        # 4. Assert that the global parameters have been updated to the saved values
        self.assertEqual(p.get('N'), 128, "Parameter 'N' was not restored correctly.")
        self.assertEqual(p.get('gridsize'), 2, "Parameter 'gridsize' was not restored correctly.")
        self.assertEqual(p.get('nphotons'), 1e8, "Parameter 'nphotons' was not restored correctly.")
        print("✅ Parameter restoration test passed.")

    def test_role_inference_allows_subclassed_models_without_output_names(self):
        """Subclassed Keras models may be saveable without a graph signature."""

        class SubclassedModel(tf.keras.Model):
            def call(self, inputs):
                return inputs

        model = SubclassedModel()
        model(tf.zeros((1, 1), dtype=tf.float32))
        self.assertFalse(hasattr(model, "output_names"))

        role = ModelManager._infer_model_role(
            model,
            str(Path(self.test_dir.name) / "custom_model"),
        )

        self.assertIsNone(role)
        self.assertEqual(
            ModelManager._infer_model_role(
                model,
                str(Path(self.test_dir.name) / "diffraction_to_obj") + "/",
            ),
            "diffraction_to_obj",
        )

    def test_architecture_aware_loading(self):
        """
        CRITICAL TEST: Test that a model's architecture is correctly rebuilt based on
        the parameters restored from the saved artifact.
        """
        print("--- Running Test: Architecture-Aware Loading ---")
        # 1. Save a model with gridsize=2 (which has 4 input channels)
        p.set('N', 64)
        p.set('gridsize', 2)
        model_gs2, _ = create_model_with_gridsize(gridsize=2, N=64)
        
        model_path = Path(self.test_dir.name) / "model_gs2"
        ModelManager.save_model(model_gs2, str(model_path), {}, 1.0)

        # 2. Set current session to a conflicting gridsize
        p.set('gridsize', 1)
        
        # 3. Load the gridsize=2 model. This should first restore gridsize=2,
        #    then build the model with the correct architecture.
        loaded_model = ModelManager.load_model(str(model_path))
        
        # 4. Assert that the loaded model has the correct architecture for gridsize=2
        # The input layer for a gridsize=2 model should have 4 channels.
        self.assertEqual(loaded_model.input_shape[0][-1], 4, "Loaded model has incorrect input shape for gridsize=2.")
        print("✅ Architecture-aware loading test passed.")

    def test_inference_consistency_after_load(self):
        """
        CRITICAL TEST: Ensure a loaded model produces identical output to the
        original model for the same input.
        """
        print("--- Running Test: Inference Consistency ---")
        p.set('N', 64)
        p.set('gridsize', 1)
        p.set('intensity_scale', 1.0)
        
        # 1. Create and save a model
        _, original_inference_model = create_model_with_gridsize(gridsize=1, N=64)
        
        # 2. Generate a dummy input
        dummy_diffraction = tf.random.normal((2, 64, 64, 1))
        dummy_positions = tf.zeros((2, 1, 2, 1))
        
        # 3. Get output from the original model
        original_output = original_inference_model.predict([dummy_diffraction, dummy_positions])
        
        model_path = Path(self.test_dir.name) / "inference_model"
        ModelManager.save_model(original_inference_model, str(model_path), {}, 1.0)

        params_path = model_path / "params.dill"
        with params_path.open("rb") as stream:
            archived_params = dill.load(stream)
        archived_params.pop("probe", None)
        with params_path.open("wb") as stream:
            dill.dump(archived_params, stream)

        # 4. Load the model
        with patch.object(
            tf.keras.models,
            "load_model",
            side_effect=AssertionError(
                "persisted model roles must avoid graph deserialization"
            ),
        ):
            loaded_model = ModelManager.load_model(str(model_path))

        # The directory name is arbitrary; loading must preserve the saved
        # inference-model role rather than silently selecting the autoencoder.
        self.assertEqual(tuple(loaded_model.output_names), ('trimmed_obj',))
        self.assertEqual(len(loaded_model.outputs), 1)
        
        # 5. Get output from the loaded model
        loaded_output = loaded_model.predict([dummy_diffraction, dummy_positions])
        
        # 6. Assert that the outputs are numerically identical
        np.testing.assert_allclose(original_output, loaded_output, rtol=1e-6,
                                   err_msg="Model output changed after save/load cycle.")
        print("✅ Inference consistency test passed.")

    def test_known_role_keras_loads_weights_without_probe_snapshot(self):
        """Known roles rebuild first and do not deserialize the saved graph."""
        p.set('N', 64)
        p.set('gridsize', 1)
        p.set('intensity_scale', 1.0)
        _, original_model = create_model_with_gridsize(gridsize=1, N=64)

        model_path = Path(self.test_dir.name) / "diffraction_to_obj"
        ModelManager.save_model(original_model, str(model_path), {}, 1.0)

        params_path = model_path / "params.dill"
        with params_path.open("rb") as stream:
            archived_params = dill.load(stream)
        self.assertNotIn("_model_role", archived_params)
        archived_params.pop("probe", None)
        with params_path.open("wb") as stream:
            dill.dump(archived_params, stream)

        with (model_path / "model_metadata.dill").open("rb") as stream:
            model_metadata = dill.load(stream)
        self.assertEqual(
            model_metadata["_model_role"],
            "diffraction_to_obj",
        )
        (model_path / "model_metadata.dill").unlink()

        with (
            patch.object(
                tf.keras.models,
                "load_model",
                side_effect=AssertionError(
                    "known model roles must load weights into rebuilt architectures"
                ),
            ),
            patch.object(
                tf.keras.config,
                "enable_unsafe_deserialization",
                side_effect=AssertionError(
                    "known model roles must not enable unsafe graph loading"
                ),
            ),
        ):
            loaded_model = ModelManager.load_model(str(model_path))

        self.assertEqual(tuple(loaded_model.output_names), ('trimmed_obj',))
        for expected, actual in zip(
            original_model.get_weights(),
            loaded_model.get_weights(),
        ):
            np.testing.assert_allclose(expected, actual, rtol=1e-6)

    def test_persisted_role_loads_only_the_selected_candidate(self):
        model_path = Path(self.test_dir.name) / "arbitrary_model_name"
        model_path.mkdir()
        (model_path / "model.keras").touch()

        with (model_path / "params.dill").open("wb") as stream:
            dill.dump(
                {
                    "_version": "1.0",
                    "N": 64,
                    "gridsize": 1,
                    "intensity_scale": 1.0,
                },
                stream,
            )
        with (model_path / "custom_objects.dill").open("wb") as stream:
            dill.dump({}, stream)
        with (model_path / "model_metadata.dill").open("wb") as stream:
            dill.dump(
                {"_model_role": "diffraction_to_obj"},
                stream,
            )

        autoencoder = MagicMock(spec=tf.keras.Model)
        autoencoder.output_names = [
            "trimmed_obj",
            "intensity_scaler_inv",
            "pred_intensity",
        ]
        diffraction_to_obj = MagicMock(spec=tf.keras.Model)
        diffraction_to_obj.output_names = ["trimmed_obj"]

        with (
            patch(
                "ptycho.model.create_model_with_gridsize",
                return_value=(autoencoder, diffraction_to_obj),
            ),
            patch.object(
                tf.keras.models,
                "load_model",
                side_effect=AssertionError(
                    "persisted roles must not deserialize the saved graph"
                ),
            ),
            patch.object(
                tf.keras.config,
                "enable_unsafe_deserialization",
                side_effect=AssertionError(
                    "persisted roles must not enable unsafe graph loading"
                ),
            ),
        ):
            loaded_model = ModelManager.load_model(str(model_path))

        self.assertIs(loaded_model, diffraction_to_obj)
        diffraction_to_obj.load_weights.assert_called_once_with(
            str(model_path / "model.keras")
        )
        autoencoder.load_weights.assert_not_called()

    def test_unknown_keras3_role_rejects_missing_output_signature(self):
        loaded_model = SimpleNamespace(output_names=['unknown_output'])
        candidates = {
            'autoencoder': SimpleNamespace(output_names=['reconstruction', 'scale']),
            'diffraction_to_obj': SimpleNamespace(output_names=['trimmed_obj']),
        }

        with self.assertRaisesRegex(
            ValueError,
            "No reconstructed model matches Keras 3 output signature",
        ):
            ModelManager._select_keras3_candidate_by_output_signature(
                loaded_model,
                candidates,
                artifact_name='unknown-artifact',
            )

    def test_unknown_keras3_role_rejects_ambiguous_output_signature(self):
        loaded_model = SimpleNamespace(output_names=['trimmed_obj'])
        candidates = {
            'candidate_a': SimpleNamespace(output_names=['trimmed_obj']),
            'candidate_b': SimpleNamespace(output_names=['trimmed_obj']),
        }

        with self.assertRaisesRegex(
            ValueError,
            "Multiple reconstructed models match Keras 3 output signature",
        ):
            ModelManager._select_keras3_candidate_by_output_signature(
                loaded_model,
                candidates,
                artifact_name='unknown-artifact',
            )

class TestTensorFlowBundleGroupingContract(unittest.TestCase):
    """Centered-nearest grouping boundary on multi-model TF archives.

    The bundle loader must preflight the root manifest and every listed
    role's params.dill against the grouping identity BEFORE any model is
    constructed, project the marker into params.cfg only after accepted
    validation, and never rewrite accepted archives.
    """

    _ROLES = ("autoencoder", "diffraction_to_obj")

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.original_params = p.cfg.copy()
        p.set('N', 64)
        p.set('gridsize', 1)
        p.set('intensity_scale', 1.0)
        p.set('probe.type', 'gaussian')
        p.set('probe.photons', 1e10)

    def tearDown(self):
        self.test_dir.cleanup()
        p.cfg.clear()
        p.cfg.update(self.original_params)

    def _write_bundle(self, archive_path, manifest, role_params):
        """Write a wts.h5-style zip: manifest.dill + per-role params.dill."""
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.dill", dill.dumps(manifest))
            for role, params_dict in role_params.items():
                zf.writestr(
                    f"{role}/params.dill",
                    dill.dumps(dict(params_dict)),
                )
                zf.writestr(
                    f"{role}/custom_objects.dill",
                    dill.dumps({}),
                )

    @staticmethod
    def _role_params(gridsize):
        return {
            role: {"N": 64, "gridsize": np.int64(gridsize)}
            for role in TestTensorFlowBundleGroupingContract._ROLES
        }

    def test_grouping_contract_matrix_rejects_before_construction(self):
        """Every incompatible bundle is rejected before model construction."""
        contract = CENTERED_NEAREST_GROUPING_CONTRACT
        cases = [
            (
                "v1 any role C>1 requires retraining",
                {"models": list(self._ROLES), "version": "1.0"},
                self._role_params(2),
                "version-1.0 C>1 bundle requires retraining",
            ),
            (
                "v1 missing gridsize",
                {"models": list(self._ROLES), "version": "1.0"},
                {
                    "autoencoder": {"N": 64},
                    "diffraction_to_obj": {"N": 64, "gridsize": 1},
                },
                "autoencoder params.dill lacks gridsize",
            ),
            (
                "v1 None gridsize",
                {"models": list(self._ROLES), "version": "1.0"},
                {
                    "autoencoder": {"N": 64, "gridsize": None},
                    "diffraction_to_obj": {"N": 64, "gridsize": 1},
                },
                "autoencoder params.dill gridsize must be a positive integer",
            ),
            (
                "v1 zero gridsize",
                {"models": list(self._ROLES), "version": "1.0"},
                {
                    "autoencoder": {"N": 64, "gridsize": 0},
                    "diffraction_to_obj": {"N": 64, "gridsize": 1},
                },
                "autoencoder params.dill gridsize must be a positive integer",
            ),
            (
                "v1 fractional gridsize",
                {"models": list(self._ROLES), "version": "1.0"},
                {
                    "autoencoder": {"N": 64, "gridsize": 1.5},
                    "diffraction_to_obj": {"N": 64, "gridsize": 1},
                },
                "autoencoder params.dill gridsize must be a positive integer",
            ),
            (
                "v1 bool gridsize",
                {"models": list(self._ROLES), "version": "1.0"},
                {
                    "autoencoder": {"N": 64, "gridsize": True},
                    "diffraction_to_obj": {"N": 64, "gridsize": 1},
                },
                "autoencoder params.dill gridsize must be a positive integer",
            ),
            (
                "v1 conflicting role gridsizes",
                {"models": list(self._ROLES), "version": "1.0"},
                {
                    "autoencoder": {"N": 64, "gridsize": 1},
                    "diffraction_to_obj": {"N": 64, "gridsize": 2},
                },
                "TensorFlow bundle roles disagree on gridsize",
            ),
            (
                "v2 missing marker",
                {
                    "models": list(self._ROLES),
                    "version": "2.0",
                },
                self._role_params(1),
                "version-2.0 bundle lacks centered-nearest-v1",
            ),
            (
                "v2 unknown marker",
                {
                    "models": list(self._ROLES),
                    "version": "2.0",
                    "grouping_contract": "quadrant-v9",
                },
                self._role_params(1),
                "version-2.0 bundle lacks centered-nearest-v1",
            ),
            (
                "v2 conflicting role gridsizes",
                {
                    "models": list(self._ROLES),
                    "version": "2.0",
                    "grouping_contract": contract,
                },
                {
                    "autoencoder": {"N": 64, "gridsize": 1},
                    "diffraction_to_obj": {"N": 64, "gridsize": 2},
                },
                "TensorFlow bundle roles disagree on gridsize",
            ),
            (
                "unsupported version",
                {"models": list(self._ROLES), "version": "3.0"},
                self._role_params(1),
                "unsupported TensorFlow bundle version '3.0'",
            ),
        ]
        for label, manifest, role_params, error_regex in cases:
            with self.subTest(label=label):
                base = Path(self.test_dir.name) / f"reject_{label.replace(' ', '_')}"
                self._write_bundle(f"{base}.zip", manifest, role_params)

                with (
                    patch(
                        "ptycho.model.create_model_with_gridsize"
                    ) as mock_create,
                ):
                    with self.assertRaisesRegex(ValueError, error_regex):
                        ModelManager.load_multiple_models(str(base))
                    mock_create.assert_not_called()

    def test_v1_all_roles_c1_accepted_in_memory_and_bytes_unchanged(self):
        """Accepted v1 C1 loads in memory without rewriting the archive."""
        base = Path(self.test_dir.name) / "v1_c1"
        self._write_bundle(
            f"{base}.zip",
            {"models": list(self._ROLES), "version": "1.0"},
            self._role_params(1),
        )
        archive_path = base.parent / "v1_c1.zip"
        snapshot = archive_path.read_bytes()

        autoencoder = MagicMock(spec=tf.keras.Model)
        diffraction_to_obj = MagicMock(spec=tf.keras.Model)
        with patch(
            "ptycho.model.create_model_with_gridsize",
            return_value=(autoencoder, diffraction_to_obj),
        ):
            loaded = ModelManager.load_multiple_models(str(base))

        self.assertEqual(set(loaded), set(self._ROLES))
        self.assertIs(loaded["diffraction_to_obj"], diffraction_to_obj)
        self.assertEqual(
            p.cfg.get("grouping_contract"),
            CENTERED_NEAREST_GROUPING_CONTRACT,
            "accepted v1 C1 load must expose the marker in params.cfg",
        )
        self.assertEqual(
            archive_path.read_bytes(),
            snapshot,
            "accepted v1 C1 loading must not rewrite the bundle",
        )

    def test_v2_marker_accepted_for_c1_and_c_gt_1(self):
        """Version-2.0 bundles with the marker accept agreeing C1 or C>1."""
        for gridsize in (1, 2):
            with self.subTest(gridsize=gridsize):
                base = Path(self.test_dir.name) / f"v2_gs{gridsize}"
                self._write_bundle(
                    f"{base}.zip",
                    {
                        "models": list(self._ROLES),
                        "version": "2.0",
                        "grouping_contract": CENTERED_NEAREST_GROUPING_CONTRACT,
                    },
                    self._role_params(gridsize),
                )
                archive_path = base.parent / f"v2_gs{gridsize}.zip"
                snapshot = archive_path.read_bytes()

                autoencoder = MagicMock(spec=tf.keras.Model)
                diffraction_to_obj = MagicMock(spec=tf.keras.Model)
                with patch(
                    "ptycho.model.create_model_with_gridsize",
                    return_value=(autoencoder, diffraction_to_obj),
                ):
                    loaded = ModelManager.load_multiple_models(str(base))

                self.assertEqual(set(loaded), set(self._ROLES))
                self.assertEqual(
                    p.cfg.get("grouping_contract"),
                    CENTERED_NEAREST_GROUPING_CONTRACT,
                )
                self.assertEqual(
                    archive_path.read_bytes(),
                    snapshot,
                    "accepted v2 loading must not rewrite the bundle",
                )

    def test_stale_per_role_marker_does_not_override_accepted_root_marker(self):
        """Accepted loads must keep the root-validated marker in params.cfg.

        A stale/conflicting grouping_contract persisted inside a role's
        params.dill must not win over the root-validated centered marker.
        """
        contract = CENTERED_NEAREST_GROUPING_CONTRACT
        cases = [
            (
                "v1 C1 root with stale per-role marker",
                {"models": list(self._ROLES), "version": "1.0"},
            ),
            (
                "v2 exact root marker with stale per-role marker",
                {
                    "models": list(self._ROLES),
                    "version": "2.0",
                    "grouping_contract": contract,
                },
            ),
        ]
        for label, manifest in cases:
            with self.subTest(label=label):
                base = Path(self.test_dir.name) / f"stale_{label.replace(' ', '_')}"
                self._write_bundle(
                    f"{base}.zip",
                    manifest,
                    {
                        role: {
                            "N": 64,
                            "gridsize": 1,
                            "grouping_contract": "quadrant-v9",
                        }
                        for role in self._ROLES
                    },
                )

                with patch(
                    "ptycho.model.create_model_with_gridsize",
                    return_value=(
                        MagicMock(spec=tf.keras.Model),
                        MagicMock(spec=tf.keras.Model),
                    ),
                ):
                    ModelManager.load_multiple_models(str(base))

                self.assertEqual(
                    p.cfg.get("grouping_contract"),
                    contract,
                    "stale per-role grouping_contract must not override "
                    "the accepted root marker",
                )

    def test_new_save_writes_v2_manifest_with_centered_marker(self):
        """save_multiple_models writes root manifest 2.0 + the marker."""
        p.set('gridsize', 1)
        p.set('intensity_scale', 1.0)
        base = Path(self.test_dir.name) / "new_save"
        ModelManager.save_multiple_models(
            {
                "autoencoder": MagicMock(spec=tf.keras.Model),
                "diffraction_to_obj": MagicMock(spec=tf.keras.Model),
            },
            str(base),
            {},
            1.0,
        )

        with zipfile.ZipFile(f"{base}.zip", 'r') as zf:
            manifest = dill.loads(zf.read("manifest.dill"))
            autoencoder_params = dill.loads(
                zf.read("autoencoder/params.dill")
            )

        self.assertEqual(manifest["version"], "2.0")
        self.assertEqual(
            manifest["grouping_contract"],
            CENTERED_NEAREST_GROUPING_CONTRACT,
        )
        self.assertEqual(set(manifest["models"]), set(self._ROLES))
        # Per-role params versions are unchanged.
        self.assertEqual(autoencoder_params["_version"], "1.0")


if __name__ == '__main__':
    unittest.main()
