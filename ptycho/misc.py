"""Miscellaneous utility functions for PtychoPINN.

This module provides utility functions and decorators supporting the PtychoPINN 
workflow, including caching mechanisms for expensive computations and helper 
functions for output path generation and image processing.

Key Public Interface:
    @memoize_disk_and_memory: Decorator for caching function results to disk and memory
    @memoize_simulated_data: Specialized decorator for caching simulated ptychography data
    get_path_prefix(): Generate timestamped output directory paths
    colormap2arr(): Convert RGB colormap images to grayscale arrays
    cross_image(): Find offsets through 2D autocorrelation

The caching decorators significantly speed up repeated computations by storing 
results based on function parameters and configuration state. They are primarily 
used by simulation and data preprocessing modules.

Example Usage:
    from ptycho.misc import memoize_disk_and_memory, get_path_prefix
    
    @memoize_disk_and_memory
    def expensive_simulation(params):
        # Computation is cached automatically
        return simulate_diffraction_data(params)
    
    output_dir = get_path_prefix()  # e.g., "outputs/01-15-2024-14.30.15_experiment/"
"""

import numpy as np
import matplotlib.cm as cm
import scipy.cluster.vq as scv
from ptycho import params
from datetime import datetime

# TODO multiple creations of this directory
def get_path_prefix():
    label = params.cfg['label']
    prefix = params.params()['output_prefix']
    now = datetime.now() # current date and time
    try:
        date_time = params.get('timestamp')
    except KeyError:
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        params.set('timestamp', date_time)
    date_time = date_time.replace('/', '-').replace(':', '.').replace(', ', '-')

    #print('offset', offset)
    out_prefix = '{}/{}_{}/'.format(prefix, date_time, label)
    return out_prefix

# Convert RGB colormap images to grayscale
def colormap2arr(arr,cmap):
    # http://stackoverflow.com/questions/3720840/how-to-reverse-color-map-image-to-scalar-values/3722674#3722674
    gradient=cmap(np.linspace(0.0,1.0,1000))

    # Reshape arr to something like (240*240, 4), all the 4-tuples in a long list...
    arr2=arr.reshape((arr.shape[0]*arr.shape[1],arr.shape[2]))

    # Use vector quantization to shift the values in arr2 to the nearest point in
    # the code book (gradient).
    code,dist=scv.vq(arr2,gradient)

    # code is an array of length arr2 (240*240), holding the code book index for
    # each observation. (arr2 are the "observations".)
    # Scale the values so they are from 0 to 1.
    values=code.astype('float')/gradient.shape[0]

    # Reshape values back to (240,240)
    values=values.reshape(arr.shape[0],arr.shape[1])
    values=values[::-1]
    return values

import functools
import hashlib
import json
import os
import tensorflow as tf

#https://chat.openai.com/c/8273412b-f3fb-405c-a7a4-c0466bb43b04
import os
import functools
import hashlib
import json
import numpy as np
import tensorflow as tf

def memoize_disk_and_memory(func):
    from ptycho.params import cfg
    from ptycho import probe
    memory_cache = {}
    disk_cache_dir = 'memoized_data'
    if not os.path.exists(disk_cache_dir):
        os.makedirs(disk_cache_dir)

    def process_dict(d):
        processed = {}
        for k, v in d.items():
            if isinstance(v, tf.Tensor):
                processed[k] = ('tensor', v.numpy())
            elif isinstance(v, np.ndarray):
                processed[k] = ('array', v)
            elif isinstance(v, dict):
                processed[k] = ('dict', process_dict(v))
            else:
                processed[k] = ('primitive', v)
        return processed

    def reconstruct_dict(d):
        reconstructed = {}
        for k, (type_, value) in d.items():
            if type_ == 'tensor' or type_ == 'array':
                reconstructed[k] = value
            elif type_ == 'dict':
                reconstructed[k] = reconstruct_dict(value)
            else:  # primitive
                reconstructed[k] = value
        return reconstructed

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cfg_keys = ['offset', 'N', 'outer_offset_train', 'outer_offset_test',
                    'nphotons', 'nimgs_train', 'nimgs_test', 'set_phi',
                    'data_source', 'gridsize', 'big_gridsize', 'default_probe_scale']
        hash_input = {k: cfg[k] for k in cfg_keys if k in cfg}
        hash_input.update({f'arg_{i}': json.dumps(arg, default=str) for i, arg in enumerate(args)})
        hash_input.update({f'kwarg_{k}': json.dumps(v, default=str) for k, v in kwargs.items()})
        hash_input_str = json.dumps(hash_input, sort_keys=True).encode('utf-8')
        hash_hex = hashlib.sha1(hash_input_str).hexdigest()

        if hash_hex in memory_cache:
            print("Loading result from memory cache.")
            return memory_cache[hash_hex]
        else:
            disk_cache_file = os.path.join(disk_cache_dir, f'{hash_hex}.npz')
            if os.path.exists(disk_cache_file):
                print("Loading result from disk cache.")
                loaded_data = np.load(disk_cache_file, allow_pickle=True)
                if 'dict_data' in loaded_data:
                    result = reconstruct_dict(loaded_data['dict_data'].item())
                elif 'result' in loaded_data:
                    result = loaded_data['result']
                else:
                    result = tuple(loaded_data[key] for key in loaded_data.keys())
                    if len(result) == 1:
                        result = result[0]
            else:
                print("No cached result found. Calculating and caching the result.")
                result = func(*args, **kwargs)
                if isinstance(result, (np.ndarray, tf.Tensor)):
                    np.savez(disk_cache_file, result=result.numpy() if isinstance(result, tf.Tensor) else result)
                elif isinstance(result, tuple):
                    np.savez(disk_cache_file, **{f'arr_{i}': arr.numpy() if isinstance(arr, tf.Tensor) else arr for i, arr in enumerate(result)})
                elif isinstance(result, dict):
                    processed_dict = process_dict(result)
                    np.savez(disk_cache_file, dict_data=processed_dict)
                else:
                    raise ValueError("Invalid function output. Expected numpy array, TensorFlow tensor, tuple, or dictionary with values as arrays/tensors/primitives.")
                memory_cache[hash_hex] = result
        return result
    return wrapper


##########
# unit test
##########
#
#import numpy as np
#import tensorflow as tf
#
## Define test functions
#@memoize_disk_and_memory
#def test_function1(x):
#    return np.random.rand(x, x)
#
#@memoize_disk_and_memory
#def test_function2(x):
#    return tf.random.uniform((x, x))
#
#@memoize_disk_and_memory
#def test_function3(x):
#    return np.random.rand(x, x), tf.random.uniform((x, x))
#
## First run - cache miss
#result1_first = test_function1(5)
#result2_first = test_function2(5)
#result3_first = test_function3(5)
#
## Second run - cache hit
#result1_second = test_function1(5)
#result2_second = test_function2(5)
#result3_second = test_function3(5)
#
## Test if the memoized results match the first run results
#np.testing.assert_array_equal(result1_first, result1_second)
#np.testing.assert_array_equal(result2_first, result2_second)
#
#np.testing.assert_array_equal(result3_first[0], result3_second[0])
#np.testing.assert_array_equal(result3_first[1], result3_second[1])
#
## Test if memoization works with different function arguments
#result1_diff_arg = test_function1(6)
#np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, result1_first, result1_diff_arg)
#

import functools
import numpy as np
import tensorflow as tf

def make_invocation_counter():
    count = 0

    def increment():
        nonlocal count
        count += 1
        return count

    return increment

######
## logging decorator
######
# TODO deprecated, moved to logging.py
#def g(h):
#    increment_count = make_invocation_counter()
#
#    def wrapper(f):
#        @functools.wraps(f)
#        def inner(*args, **kwargs):
#            invocation_count = increment_count()
#            if invocation_count <= 2:
#                return h(f)(*args, **kwargs)
#            else:
#                return f(*args, **kwargs)
#
#        return inner
#
#    return wrapper
#
#@g
#def debug(func):
#    def wrapper(*args, **kwargs):
#        def get_type_and_shape(x):
#            if isinstance(x, np.ndarray):
#                return f"{type(x)} with shape {x.shape}"
#            elif isinstance(x, tf.Tensor):
#                return f"{type(x)} with shape {x.shape}"
#            else:
#                return str(type(x))
#
#        args_types = [get_type_and_shape(arg) for arg in args]
#        kwargs_types = {k: get_type_and_shape(v) for k, v in kwargs.items()}
#
#        print(f"Calling {func.__name__} with args types: {args_types}, kwargs types: {kwargs_types}")
#        result = func(*args, **kwargs)
#        
#        result_type = get_type_and_shape(result)
#        print(f"{func.__name__} returned {result_type}")
#        
#        return result
#    return wrapper

import scipy.signal
import functools
import hashlib
import json
import os
import numpy as np

def memoize_simulated_data(func):
    memory_cache = {}
    disk_cache_dir = 'memoized_simulated_data'
    if not os.path.exists(disk_cache_dir):
        os.makedirs(disk_cache_dir)

    def array_to_bytes(arr):
        return arr.tobytes(), arr.dtype.str, arr.shape

    def bytes_to_array(data, dtype, shape):
        return np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape)

    @functools.wraps(func)
    def wrapper(objectGuess, probeGuess, nimages, buffer, random_seed=None, return_patches=True):
        from ptycho.loader import RawData
        # Create a unique hash for the input parameters
        hash_input = {
            'objectGuess': array_to_bytes(objectGuess),
            'probeGuess': array_to_bytes(probeGuess),
            'nimages': nimages,
            'buffer': buffer,
            'random_seed': random_seed,
            'return_patches': return_patches
        }
        hash_str = json.dumps(hash_input, sort_keys=True).encode('utf-8')
        hash_hex = hashlib.sha256(hash_str).hexdigest()

        if hash_hex in memory_cache:
            print("Loading result from memory cache.")
            return memory_cache[hash_hex]

        disk_cache_file = os.path.join(disk_cache_dir, f'{hash_hex}.npz')
        if os.path.exists(disk_cache_file):
            print("Loading result from disk cache.")
            with np.load(disk_cache_file, allow_pickle=True) as data:
                raw_data_dict = data['raw_data'].item()
                raw_data = RawData(
                    xcoords=raw_data_dict['xcoords'],
                    ycoords=raw_data_dict['ycoords'],
                    xcoords_start=raw_data_dict['xcoords_start'],
                    ycoords_start=raw_data_dict['ycoords_start'],
                    diff3d=raw_data_dict['diff3d'],
                    probeGuess=raw_data_dict['probeGuess']
                )
                if return_patches:
                    patches = data['patches']
                    result = (raw_data, patches)
                else:
                    result = raw_data
        else:
            print("No cached result found. Calculating and caching the result.")
            result = func(objectGuess, probeGuess, nimages, buffer, random_seed, return_patches)
            
            if isinstance(result, tuple):
                raw_data, patches = result
            else:
                raw_data = result
                patches = None

            raw_data_dict = {
                'xcoords': raw_data.xcoords,
                'ycoords': raw_data.ycoords,
                'xcoords_start': raw_data.xcoords_start,
                'ycoords_start': raw_data.ycoords_start,
                'diff3d': raw_data.diff3d,
                'probeGuess': raw_data.probeGuess
            }

            np.savez(disk_cache_file, raw_data=raw_data_dict, patches=patches)

        memory_cache[hash_hex] = result
        return result

    return wrapper

def cross_image(im1, im2):
    """
    Find offsets through 2d autocorrelation
    """
    # get rid of the color channels by performing a grayscale transform
    # the type cast into 'float' is to avoid overflows
    im1_gray = im1#np.sum(im1.astype('float'), axis=2)
    im2_gray = im2#np.sum(im2.astype('float'), axis=2)

    # get rid of the averages, otherwise the results are not good
    im1_gray -= np.mean(im1_gray)
    im2_gray -= np.mean(im2_gray)

    # calculate the correlation image; note the flipping of onw of the images
    return scipy.signal.fftconvolve(im1_gray, im2_gray[::-1,::-1], mode='same')
