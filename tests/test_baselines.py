import unittest
import numpy as np
import tensorflow as tf
from ptycho.baselines import build_model

class TestBaselines(unittest.TestCase):
    def test_build_model_always_creates_single_channel_output(self):
        """
        This test will FAIL before the fix.
        It verifies that build_model is hardened to always produce a single-channel
        output, even when fed multi-channel input data.
        """
        # 1. Arrange: Create mock 4-channel input data
        mock_X_train = np.random.rand(16, 64, 64, 4).astype(np.float32)
        mock_Y_I_train = np.random.rand(16, 64, 64, 4).astype(np.float32)
        mock_Y_phi_train = np.random.rand(16, 64, 64, 4).astype(np.float32)

        # 2. Act: Build the model
        model = build_model(mock_X_train, mock_Y_I_train, mock_Y_phi_train)

        # 3. Assert: Check the output shape of the two decoder arms
        # The model has two outputs: [decoded1, decoded2] for amplitude and phase
        output_shape_amp = model.output_shape[0]
        output_shape_phase = model.output_shape[1]

        # This assertion will fail before the fix, as shape will be (None, 64, 64, 4)
        self.assertEqual(output_shape_amp[-1], 1, "Amplitude decoder should have 1 output channel.")
        self.assertEqual(output_shape_phase[-1], 1, "Phase decoder should have 1 output channel.")

if __name__ == '__main__':
    unittest.main()