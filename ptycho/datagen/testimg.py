import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter as gf
import os
from scipy import misc
from imageio import imread
from ptycho import tf_helper as hh
from ptycho import params
import tensorflow as tf

def first_and_last(it):
    it = iter(it)  # Ensure it's an iterator
    try:
        first = next(it)  # Get the first item
    except StopIteration:
        return  # If the iterator is empty, return an empty iterator
    last = None
    for last in it:  # Traverse the rest of the iterator to find the last item
        pass
    if last is None:
        yield first
    else:
        yield first
        yield last

path = './'

# Lazy pipeline state: (iterator, reversed_iterator). Import performs zero
# params.cfg reads and zero file I/O (W3.2); first get_block() initializes.
_state = None


def _load_patches():
    global _state
    if _state is None:
        image = imread(os.path.join(path, 'williamson.jpeg')).astype(float)
        image /= image.mean()
        image = image[None, 100:, :, :1]
        N = params.get('size')
        imgs = hh.extract_patches(image, N, N)
        imgs = tf.reshape(imgs, (-1,) + (N, N))
        imgs_np = imgs.numpy()
        rev_tensor = tf.convert_to_tensor(imgs_np[::-1], dtype=tf.float32)
        _state = (iter(imgs_np), iter(rev_tensor))
    return _state


def get_block(reverse = False):
    it, rev_it = _load_patches()
    if reverse:
        return np.array(next(rev_it))
    return np.array(next(it))

def get_img(N = None, sigma = .5, reverse = False):
    img = get_block(reverse = reverse)
    # Anti-aliasing
    img = gf(img, sigma)
    img = img[:, :, None]
    return img

