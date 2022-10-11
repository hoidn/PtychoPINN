from tensorflow.keras.layers import Lambda
from tensorflow.signal import fft2d, fftshift

import tensorflow as tf
from xrdc import fourier
def mk_rand():
    return int(N * np.random.uniform())

import numpy as np
import matplotlib.pyplot as plt
from skimage import draw, morphology

tfk = tf.keras
tfkl = tf.keras.layers

N = 64

from xrdc import fourier as f

def mk_lines_img():
    image = np.zeros((N, N))
    for _ in range(6):
        rr, cc = draw.line(mk_rand(), mk_rand(), mk_rand(), mk_rand())
        image[rr, cc] = 1
    #dilated = morphology.dilation(image, morphology.disk(radius=1))
    res = np.zeros((N, N, 3))
    #res[:, :, :] = dilated[..., None]
    res[:, :, :] = image[..., None]
    return f.gf(res, 1) + 2 * f.gf(res, 5)


ds_lines = {'train':  tf.data.Dataset.from_tensor_slices([tf.convert_to_tensor(mk_lines_img()) for _ in range(500)])}

def do_forward(sequential = None):
    """
    zero-pad the real-space object and then fourier transform it
    """
    if sequential is None:
        sequential = tfk.Sequential([])
    sequential.add(tfk.Input(shape = (N, N, 1)))
    sequential.add(Lambda(lambda resized: (fft2d(
        tf.squeeze(tf.cast(resized, tf.complex64))
    ))))
    sequential.add(Lambda(lambda X: tf.math.real(tf.math.conj(X) * X) / N**2))
    sequential.add(Lambda(lambda psd: 
                          tf.expand_dims(
                              tf.math.sqrt(
            fftshift(psd, (-2, -1))), 3)))
    #sequential.add(Lambda(lambda x: tf.math.log(1 + 5 * x)),)
    return sequential

rmod = do_forward()
rmod.compile(loss='mse')

def split_image(image3, tile_size):
    n, m =  image3.shape[0], image3.shape[1]
    N, M = (n // tile_size[0]) * tile_size[0], (m // tile_size[1]) * tile_size[1]
    image3 = image3[:N, :M, :]
    image_shape = tf.shape(image3)
    tile_rows = tf.reshape(image3, [image_shape[0], -1, tile_size[1], image_shape[2]])
    serial_tiles = tf.transpose(tile_rows, [1, 0, 2, 3])
    return tf.reshape(serial_tiles, [-1, tile_size[1], tile_size[0], image_shape[2]])


def split_tf_image(img):
    return tf.compat.v1.extract_image_patches(img[None, ...], [1, N, N, 1], [1, N // 4, N // 4, 1],
                                              [1, 1, 1, 1], padding = 'VALID')

def img_get(key, ds):
    for img in iter(ds[key]):
        yield split_tf_image(img)

def img_get(key, ds):
    foo = iter(ds[key])
    for i, img in enumerate(foo):
        if i > 5000:
            break
        yield split_tf_image(img)
#     foo =  iter(datasets2[key])
#     for _ in range(3):
#         yield split_tf_image(next(foo))

def mk_ds_slices(key, ds):
    tmp = list(img_get(key, ds))
    tmp = tf.concat(tmp, 0)
    tmp = tf.reshape(tmp, [-1, N, N, 3])[:, :, :, 0]
    tmp2 = tf.data.Dataset.from_tensor_slices(tmp)
    return tmp2

    orig_probe = tf.identity(image)
    rmod = do_forward()
    rmod.compile(loss='mse')
    print(image.shape)
    image = rmod(image)#rmod.predict(image)
    return image, image, orig_probe#, orig

def _preprocess(sample):
    image = tf.cast(tf.image.resize(sample['image'], [N, N]), 
                    tf.float32) / 255.  # Scale to unit interval.
#     print(image.shape)
#     image = image * tprobe
#     print(image.shape, tf.convert_to_tensor(probe, tf.float32)[..., None].shape)
    rmod = do_forward(do_resize(N))
    rmod.compile(loss='mse')
    orig = tf.identity(image)
    image = rmod(image)#rmod.predict(image)
    return image, image, orig

ds = mk_ds_slices('train', ds_lines)

DATASET_SIZE = len(ds)
train_size = int(0.8 * DATASET_SIZE)
# val_size = int(0.4 * DATASET_SIZE)
test_size = int(0.2 * DATASET_SIZE)

full_dataset = ds#tf.data.TFRecordDataset(FLAGS.input_file)
# full_dataset = full_dataset.shuffle(int(10e3))
train_dataset = full_dataset.take(train_size)
test_dataset = full_dataset.skip(train_size)
# val_dataset = test_dataset.skip(test_size)
test_dataset = test_dataset.take(test_size)

#train_dataset = (train_dataset
#                 .batch(128)
#                 .map(_preprocess)
#                 .cache()
#                 .prefetch(tf.data.AUTOTUNE)
#                 .shuffle(int(10e3))
#                )
#
#
#eval_dataset = (test_dataset
#                .batch(128)
#                .map(_preprocess)
#                .cache()
#                .prefetch(tf.data.AUTOTUNE))
#
