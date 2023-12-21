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
