import numpy as np
import matplotlib.cm as cm
import scipy.cluster.vq as scv
from ptycho import params
from datetime import datetime

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

#import functools
#import hashlib
#import json
#import os
#import numpy as np
#import tensorflow as tf
#
#def memoize_disk_and_memory(func):
#    from ptycho.params import cfg
#    memory_cache = {}
#    disk_cache_dir = 'memoized_data'
#
#    if not os.path.exists(disk_cache_dir):
#        os.makedirs(disk_cache_dir)
#
#    # TODO probe
#    @functools.wraps(func)
#    def wrapper(*args, **kwargs):
#        cfg_keys = ['offset', 'N', 'bigoffset', 'sim_nphotons', 'nimgs_train', 'nimgs_test',
#                    'data_source', 'gridsize', 'big_gridsize']
#        hash_input = {k: cfg[k] for k in cfg_keys if k in cfg}
#        hash_input_str = json.dumps(hash_input, sort_keys=True).encode('utf-8')
#        hash_hex = hashlib.sha1(hash_input_str).hexdigest()
#
#        if hash_hex in memory_cache:
#            print("Loading result from memory cache.")
#            return memory_cache[hash_hex]
#        else:
#            disk_cache_file = os.path.join(disk_cache_dir, f'{hash_hex}.npz')
#
#            if os.path.exists(disk_cache_file):
#                print("Loading result from disk cache.")
#                loaded_data = np.load(disk_cache_file, allow_pickle=True)
#                result = tuple(loaded_data[key] for key in loaded_data.keys())
#                if len(result) == 1:
#                    result = result[0]
#            else:
#                print("No cached result found. Calculating and caching the result.")
#                result = func(*args, **kwargs)
#
#                if isinstance(result, (np.ndarray, tf.Tensor)):
#                    np.savez(disk_cache_file, result=result.numpy() if isinstance(result, tf.Tensor) else result)
#                elif isinstance(result, tuple):
#                    np.savez(disk_cache_file, **{f'arr_{i}': arr.numpy() if isinstance(arr, tf.Tensor) else arr for i, arr in enumerate(result)})
#                else:
#                    raise ValueError("Invalid function output. Expected numpy array, TensorFlow tensor, or tuple containing numpy arrays and/or TensorFlow tensors.")
#
#                memory_cache[hash_hex] = result
#
#        return result
#
#    return wrapper

import functools
import hashlib
import json
import os
import numpy as np
import tensorflow as tf

def memoize_disk_and_memory(func):
    from ptycho.params import cfg
    memory_cache = {}
    disk_cache_dir = 'memoized_data'

    if not os.path.exists(disk_cache_dir):
        os.makedirs(disk_cache_dir)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cfg_keys = ['offset', 'N', 'bigoffset', 'sim_nphotons', 'nimgs_train', 'nimgs_test',
                    'data_source', 'gridsize', 'big_gridsize']
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
                else:
                    raise ValueError("Invalid function output. Expected numpy array, TensorFlow tensor, or tuple containing numpy arrays and/or TensorFlow tensors.")

                memory_cache[hash_hex] = result

        return result

    return wrapper



#### 3.5 version
#
#import numpy as np
#import tensorflow as tf
#import hashlib
#import pickle
#import os
#
#def memoize(cfg_key_list):
#    def decorator(func):
#        cache_dir = './cache'
#        os.makedirs(cache_dir, exist_ok=True)
#
#        def hash_cfg(cfg):
#            cfg_subset = {k: cfg[k] for k in cfg_key_list}
#            cfg_str = str(cfg_subset)
#            return hashlib.sha256(cfg_str.encode()).hexdigest()
#
#        def memoized_func(cfg):
#            cfg_hash = hash_cfg(cfg)
#            cache_path = os.path.join(cache_dir, cfg_hash)
#
#            if os.path.exists(cache_path):
#                with open(cache_path, 'rb') as f:
#                    output = pickle.load(f)
#            else:
#                output = func(cfg)
#                with open(cache_path, 'wb') as f:
#                    pickle.dump(output, f)
#
#            return output
#
#        return memoized_func
#
#    return decorator

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




#########
## version that uses str instead of json.dumps
#######
#import functools
#import hashlib
#import os
#import numpy as np
#import tensorflow as tf
#
#def memoize_disk_and_memory(func):
#    memory_cache = {}
#    disk_cache_dir = 'memoized_data'
#
#    if not os.path.exists(disk_cache_dir):
#        os.makedirs(disk_cache_dir)
#
#    @functools.wraps(func)
#    def wrapper(*args, **kwargs):
#        cfg_keys = ['offset', 'N', 'bigoffset', 'sim_nphotons', 'nimgs_train', 'nimgs_test',
#                    'data_source', 'gridsize', 'big_gridsize']
#        hash_input = ''.join(str(cfg[k]) for k in cfg_keys if k in cfg)
#        hash_hex = hashlib.sha1(hash_input.encode('utf-8')).hexdigest()
#
#        if hash_hex in memory_cache:
#            print("Loading result from memory cache.")
#            return memory_cache[hash_hex]
#        else:
#            disk_cache_file = os.path.join(disk_cache_dir, f'{hash_hex}.npz')
#
#            if os.path.exists(disk_cache_file):
#                print("Loading result from disk cache.")
#                loaded_data = np.load(disk_cache_file, allow_pickle=True)
#                result = tuple(loaded_data[key] for key in loaded_data.keys())
#                if len(result) == 1:
#                    result = result[0]
#            else:
#                print("No cached result found. Calculating and caching the result.")
#                result = func(*args, **kwargs)
#
#                if isinstance(result, (np.ndarray, tf.Tensor)):
#                    np.savez(disk_cache_file, result=result.numpy() if isinstance(result, tf.Tensor) else result)
#                elif isinstance(result, tuple):
#                    np.savez(disk_cache_file, **{f'arr_{i}': arr.numpy() if isinstance(arr, tf.Tensor) else arr for i, arr in enumerate(result)})
#                else:
#                    raise ValueError("Invalid function output. Expected numpy array, TensorFlow tensor, or tuple containing numpy arrays and/or TensorFlow tensors.")
#
#                memory_cache[hash_hex] = result
#
#        return result
#
#    return wrapper
#
