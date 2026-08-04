# file: tests/test_model_manager_persistence.py

import unittest
import tempfile
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

if __name__ == '__main__':
    unittest.main()
