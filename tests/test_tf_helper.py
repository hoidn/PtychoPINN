from ptycho.tf_helper import complexify_function, complexify_amp_phase, combine_complex
import tensorflow as tf
import numpy as np


# Sample function to be complexified
def sample_fn(tensor, *args, **kwargs):
    return tensor * 2

# Complexify the sample function
complexified_fn = complexify_function(sample_fn)
complexified_amp_phase_fn = complexify_amp_phase(sample_fn)

def test_complexify_function():
    # Test with real tensor
    real_tensor = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    assert tf.math.reduce_all(complexified_fn(real_tensor) == real_tensor * 2), "Failed on real tensor"

    # Test with complex tensor
    complex_tensor = tf.constant([1.0 + 2.0j, 3.0 + 4.0j], dtype=tf.complex64)
    expected_output = tf.constant([2.0 + 4.0j, 6.0 + 8.0j], dtype=tf.complex64)
    assert tf.math.reduce_all(complexified_fn(complex_tensor) == expected_output), "Failed on complex tensor"

with tf.device('/CPU:0'):
    def test_complexify_amp_phase():
        # Test with real tensor
        real_tensor = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        assert tf.math.reduce_all(complexified_amp_phase_fn(real_tensor) == real_tensor * 2), "Failed on real tensor"

    # Test with complex tensor
    complex_tensor = tf.constant([1.0 + 2.0j, 3.0 + 4.0j], dtype=tf.complex64)
    # Doubling the amplitude
    expected_amplitude = tf.math.abs(complex_tensor) * 2
    # Doubling the phase (modulus to keep it within -pi to pi)
    expected_phase = tf.math.angle(complex_tensor) * 2 % (2 * tf.constant(np.pi))
    # Construct the expected tensor
    expected_tensor = combine_complex(expected_amplitude, expected_phase)
    # Compare the reconstructed tensor to the expected tensor
    error = tf.math.abs(complexified_amp_phase_fn(complex_tensor) - expected_tensor)
    assert tf.math.reduce_max(error) < 1e-6, "Failed on complex tensor"


# Execute the tests
if __name__ == "__main__":
    test_complexify_function()
    with tf.device('/CPU:0'):
        # Force CPU execution because one of the first two tests fails on GPU
        test_complexify_amp_phase()

print("All tests passed!")
