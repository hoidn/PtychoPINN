import unittest
import tensorflow as tf
import numpy as np
from tf_helper import get_mask, combine_complex, pad_obj

class TestTFHelper(unittest.TestCase):

    def test_get_mask(self):
        input_tensor = tf.constant([[0.1, 0.5], [0.9, 0.0]], dtype=tf.float32)
        expected_output = tf.constant([[0, 1], [1, 0]], dtype=tf.float32)
        threshold = 0.2
        output = get_mask(input_tensor, threshold)
        self.assertTrue(tf.reduce_all(tf.equal(output, expected_output)))

    def test_combine_complex(self):
        amp = tf.constant([[2.0, 3.0], [4.0, 5.0]], dtype=tf.float32)
        phi = tf.constant([[0.0, np.pi/2], [np.pi, -np.pi/2]], dtype=tf.float32)
        expected_output = tf.constant([[2.0+0j, 0+3j], [-4.0+0j, 0-5j]], dtype=tf.complex64)
        output = combine_complex(amp, phi)
        self.assertTrue(tf.reduce_all(tf.equal(output, expected_output)))

    def test_pad_obj(self):
        input_tensor = tf.constant([[1.0]], dtype=tf.float32)
        expected_output_shape = (3, 3)
        h, w = 2, 2
        output = pad_obj(input_tensor, h, w)
        self.assertEqual(output.shape, expected_output_shape)

if __name__ == '__main__':
    unittest.main()
import unittest
import tensorflow as tf
from ptycho.tf_helper import get_mask

class TestGetMask(unittest.TestCase):
    def test_get_mask(self):
        # Create a sample input tensor
        input_tensor = tf.constant([[0.1, 0.5, 0.9],
                                    [1.1, 0.2, 0.8]], dtype=tf.float32)
        # Define the support threshold
        threshold = 0.5
        # Expected output mask
        expected_mask = tf.constant([[0, 1, 1],
                                     [1, 0, 1]], dtype=tf.float32)
        
        # Call the get_mask function
        output_mask = get_mask(input_tensor, threshold)
        
        # Verify the output mask matches the expected mask
        self.assertTrue(tf.reduce_all(tf.equal(output_mask, expected_mask)))

if __name__ == '__main__':
    unittest.main()
