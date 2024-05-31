import unittest
import tensorflow as tf
import numpy as np
from ptycho.tf_helper import get_mask, combine_complex, pad_obj


if __name__ == '__main__':
    unittest.main()
import unittest
import tensorflow as tf
from ptycho.tf_helper import get_mask, _fromgrid, params

class TestFromGrid(unittest.TestCase):

    def test_fromgrid(self):
        print("Debug: Starting test_fromgrid")
        # Set up parameters for the test
        gridsize = params()['gridsize']
        N = params()['N']
        print(f"Debug: Test parameters - gridsize = {gridsize}, N = {N}")
        # Create a sample input tensor in grid format
        input_tensor = tf.random.uniform((1, gridsize, gridsize, N, N), dtype=tf.float32)
        print(f"Debug: Input tensor shape = {input_tensor.shape}")
        # Calculate the expected output shape
        expected_shape = (1, N, N, 1)
        print(f"Debug: Expected output shape = {expected_shape}")
        # Run the _fromgrid function
        output_tensor = _fromgrid(input_tensor)
        print(f"Debug: Output tensor shape = {output_tensor.shape}")
        # Check if the output shape matches the expected shape
        self.assertEqual(output_tensor.shape, expected_shape)

with tf.device('/CPU:0'):
    def test_complexify_amp_phase():
        # Test with real tensor
        real_tensor = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        assert tf.math.reduce_all(complexified_amp_phase_fn(real_tensor) == real_tensor * 2), "Failed on real tensor"
    def test_get_mask():
        input_tensor = tf.constant([[0.1, 0.5], [0.9, 0.0]], dtype=tf.float32)
        expected_output = tf.constant([[0, 1], [1, 0]], dtype=tf.float32)
        threshold = 0.2
        output = get_mask(input_tensor, threshold)
        self.assertTrue(tf.reduce_all(tf.equal(output, expected_output)))

if __name__ == '__main__':
    unittest.main()
import unittest
import numpy as np
import tensorflow as tf
from ptycho.tf_helper import combine_complex

class TestCombineComplex(unittest.TestCase):

def test_combine_complex():
    amp = tf.constant([1.0, 2.0], dtype=tf.float32)
    phi = tf.constant([0.0, np.pi], dtype=tf.float32)
    expected_output = tf.constant([1.0 + 0j, -2.0 + 0j], dtype=tf.complex64)
    output_complex = combine_complex(amp, phi)
    # Use a tolerance when comparing complex numbers
    tolerance = 1e-5
    self.assertTrue(tf.reduce_all(tf.math.abs(output_complex - expected_output) < tolerance))

def test_pad_and_diffract():
    # Create a sample input tensor
    input_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    input_tensor = tf.reshape(input_tensor, (1, 2, 2, 1))  # Reshape to (batch, height, width, channels)
    # Define the desired output height and width
    desired_height = 4
    desired_width = 4
    # Expected output tensor values based on provided output
    expected_output_values = [0.0, 0.70710677, 1.0, 0.70710677, 0.35355338, 1.4577379, 1.9039432, 1.2747549, 0.5, 1.8027756]
    # Run pad_and_diffract function
    _, output_tensor = pad_and_diffract(input_tensor, desired_height, desired_width)
    # Flatten the output tensor and slice the first 10 values for comparison
    output_values = output_tensor.numpy().flatten()[:10]
    # Check if the output values match the expected values within a tolerance
    for expected, actual in zip(expected_output_values, output_values):
        self.assertAlmostEqual(expected, actual, places=5)

# Execute the tests
if __name__ == "__main__":
    test_complexify_function()
    with tf.device('/CPU:0'):
        # Force CPU execution because one of the first two tests fails on GPU
        test_complexify_amp_phase()
        test_get_mask()
        test_combine_complex()
        test_pad_and_diffract()

if __name__ == '__main__':
    unittest.main()

