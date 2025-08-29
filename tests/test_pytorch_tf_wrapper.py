import os
import numpy as np
import functools
import tensorflow as tf
import hashlib
import sys
import pytest

# Gracefully handle PyTorch import failures
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    torch = None
    import warnings
    warnings.warn(f"PyTorch not available: {e}. Skipping PyTorch-related tests.")

# Skip entire module if torch is not available
if not TORCH_AVAILABLE:
    pytest.skip("PyTorch not available", allow_module_level=True)

#Define helper functions found in wrapper "output_consistency"

#Create directory for function outputs to save
memo_dir = "function_output"
if not os.path.exists(memo_dir):
    os.makedirs(memo_dir)

def hash_tensor(tensor):
    """Generate a hash for a tensor for use in ids."""

    tensor_np = tensor.numpy() if isinstance(tensor, (tf.Tensor, torch.Tensor)) else np.array(tensor)
    #Transpose input tensor if tensor is tensorflow and greater than 3 dimensions, so the input hash is identical to pytorch
    if isinstance(tensor, tf.Tensor) and tensor.ndim > 3:
        #Transpose numpy tensor such that tensorflow/pytorch are consistent
        tensor_np = np.transpose(tensor_np, axes=[0,3,1,2])

    tensor_bytes = tensor_np.tobytes()

    return hashlib.md5(tensor_bytes).hexdigest()

def generate_unique_id(func, *args, **kwargs):
    """
    Generate unique ID for function output. Modified to exclude function name from unique ID, such that
    PyTorch and Tensorflow functions with identical inputs generate the same unique ID.
    """

    func_name = func.__name__
    args_hash = "_".join([hash_tensor(arg) if isinstance(arg, (tf.Tensor, torch.Tensor)) else str(arg) for arg in args])
    kwargs_hash = "_".join([f"{k}={hash_tensor(v) if isinstance(v, (tf.Tensor, torch.Tensor)) else v}" for k, v in kwargs.items()])
    unique_id = f"{args_hash}_{kwargs_hash}"
    #unique_id = f"{func_name}"

    return unique_id

def save_output(func_id, output):
    """
    Convert function output to numpy and save to file
    """
    output_path = os.path.join(memo_dir, f"{func_id}.npy")
    #Check if output is a tensorflow tensor dimensions greater than 3.
    #If output is 3 or less, than no color channel exists and this tensor does not need to be modified
    if isinstance(output, tf.Tensor) and output.ndim > 3:
        #Tranpose order of dimensions from (0,1,2,3) to (0,3,1,2). This brings the channel dimension to the second channel
        output = tf.transpose(output, perm=[0,3,1,2])

    np.save(output_path, output.numpy())

def load_output(func_id):
    """
    Load function output from numpy if path exists otherwise return None
    """
    output_path = os.path.join(memo_dir, f"{func_id}.npy")
    
    if os.path.exists(output_path):
        output = np.load(output_path)
        return output
    else:
        return None
    
def invocation_wrapper(outer_wrapper):
    """
    Wrap a function with a wrapper that tracks the number of times a function is called.
    Outer wrapper keeps track of number of calls
    Inner wrapper is the actual function being passed through (e.g. relu or forward pass of a neural network layer)
    """

    class Invocations:
        """
        Invocation class keeps track of number of times function is called. Each function has its own unique key with corresponding counts
        """
        def __init__(self):
            self.counts = {}

        def increment_count(self, f):
            if f not in self.counts:
                self.counts[f] = 1
            else:
                self.counts[f] += 1


    invocations = Invocations()

    @functools.wraps(outer_wrapper)
    def wrapper(f):
        @functools.wraps(f)
        def inner_wrapper(*args, **kwargs):
            #Increment invocations count (each function has a separate tracker)
            invocations.increment_count(f)
            #If function calls are above 2, do not save function outputs (simply pass func through)
            if invocations.counts[f] <= 2:
                return outer_wrapper(f)(*args, **kwargs)
            else:
                return f(*args, **kwargs)
            
        return inner_wrapper
    
    return wrapper

#Create instance of invocation wrapper that prints function details
@invocation_wrapper
def debug(func):
    def wrapper(*args, **kwargs):
        print(f"Function '{func.__name__}' called with args {args} and kwargs {kwargs}")

        #Save function outputs, etc.
        #Generate function id
        func_id = generate_unique_id(func, *args, **kwargs)

        # Generate default function output
        output = func(*args, **kwargs)

        # Load existing output if available
        existing_output = load_output(func_id)

        # Save output if not available
        if existing_output is None:
            save_output(func_id, output)
        else: #Check if output is consistent
            if not np.allclose(output.numpy(), existing_output):
                raise ValueError("Output mismatch for function '{func.__name__}' with id '{func_id}'")
            else:
                print(f"Output for function '{func.__name__}' with id '{func_id}' is consistent")   

        return output
    

    return wrapper


# def output_consistency(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):

#         #Generate function id
#         func_id = generate_unique_id(func, *args, **kwargs)

#         # Generate default function output
#         output = func(*args, **kwargs)

#         # Load existing output if available
#         existing_output = load_output(func_id)

#         # Save output if not available
#         if existing_output is None:
#             save_output(func_id, output)
#         else: #Check if output is consistent
#             if not np.allclose(output.numpy(), existing_output):
#                 raise ValueError("Output mismatch for function '{func.__name__}' with id '{func_id}'")
#             else:
#                 print(f"Output for function '{func.__name__}' with id '{func_id}' is consistent")

#         return output

#     return wrapper





