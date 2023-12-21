import unittest
import tensorflow as tf
import numpy as np
from ptycho.tf_helper import get_mask, combine_complex, pad_obj


if __name__ == '__main__':
    unittest.main()
import unittest
from ptycho.tf_helper import get_mask
import tensorflow as tf

class TestGetMask(unittest.TestCase):

    def test_get_mask(self):
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

    def test_combine_complex(self):
        amp = tf.constant([1.0, 2.0], dtype=tf.float32)
        phi = tf.constant([0.0, np.pi], dtype=tf.float32)
        expected_output = tf.constant([1.0 + 0j, -2.0 + 0j], dtype=tf.complex64)
        output_complex = combine_complex(amp, phi)
        # Use a tolerance when comparing complex numbers
        tolerance = 1e-5
        self.assertTrue(tf.reduce_all(tf.math.abs(output_complex - expected_output) < tolerance))

if __name__ == '__main__':
    unittest.main()
import unittest
import tensorflow as tf
import numpy as np
from ptycho.tf_helper import pad_and_diffract

class TestPadAndDiffract(unittest.TestCase):

    def test_pad_and_diffract(self):
        # Create a sample input tensor
        input_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
        input_tensor = tf.reshape(input_tensor, (1, 2, 2, 1))  # Reshape to (batch, height, width, channels)
        # Define the desired output height and width
        desired_height = 4
        desired_width = 4
        # Run pad_and_diffract function
        _, output_tensor = pad_and_diffract(input_tensor, desired_height, desired_width)
        # Print part of the output tensor for inspection
        print("Part of the output tensor:", output_tensor.numpy().flatten()[:10])

if __name__ == '__main__':
    unittest.main()
