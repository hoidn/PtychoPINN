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
image= imread(os.path.join(path,'williamson.jpeg')).astype(float)
image /= image.mean()
image = image[None, 100:, :, :1]

N = params.get('size')
imgs = hh.extract_patches(image, N, N)
imgs = tf.reshape(imgs, (-1,) + (N, N))
it = iter(imgs)
#it = first_and_last(it)

def get_block(reverse = False):
    if reverse:
        return np.array(next(reversed(it)))
    return np.array(next(it))

def get_img(N = None, sigma = .5, reverse = False):
    img = get_block(reverse = reverse)
    # anti aliasing
    img = gf(img, sigma)
#    img = img + gf(img, 10 * sigma) * 5
    img = img[:, :, None]
    return img