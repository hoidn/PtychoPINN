import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter as gf
import os
from scipy import misc
from imageio import imread

from ptycho import tf_helper as hh
from ptycho import params
import tensorflow as tf

path = './'
image= imread(os.path.join(path,'williamson.jpeg')).astype(float)
image /= image.mean()
image = image[None, :, :, :1]

N = params.get('size')
imgs = hh.extract_patches(image, N, N)
imgs = tf.reshape(imgs, (-1,) + (N, N))
it = iter(imgs)

def get_block():
    return np.array(next(it))

def get_img(N = None, sigma = .5):
    img = get_block()
    # anti aliasing
    img = gf(img, sigma)
#    img = img + gf(img, 10 * sigma) * 5
    img = img[:, :, None]
    return img
