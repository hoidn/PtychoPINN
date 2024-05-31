import os
import numpy as np
import functools
import tensorflow as tf
import torch
import hashlib
import sys

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
    Generate unique ID for function output
    """

    func_name = func.__name__
    args_hash = "_".join([hash_tensor(arg) if isinstance(arg, (tf.Tensor, torch.Tensor)) else str(arg) for arg in args])
    kwargs_hash = "_".join([f"{k}={hash_tensor(v) if isinstance(v, (tf.Tensor, torch.Tensor)) else v}" for k, v in kwargs.items()])
    unique_id = f"{func_name}_{args_hash}_{kwargs_hash}"

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

def output_consistency(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):

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



