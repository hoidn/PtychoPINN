"""
Module for logging and inspecting function inputs, outputs, and execution times.

Provides the `debug` decorator to log function invocations, including serialized inputs,
outputs, and execution times. Supports logging to console and disk files.

Includes `load_logged_data` function to load logged data from disk for a specific invocation.

Handles serialization of NumPy arrays, TensorFlow tensors, and custom objects.

Logging controlled by `params.get('debug')` configuration.

Key components:
- `debug` decorator
- `load_logged_data` function
- Helper functions: `make_invocation_counter`, `serialize_input`
- Custom exceptions: `SerializationError`, `LoggedDataNotFoundError`
"""
import functools
import inspect
import json
import numpy as np
import os
import tensorflow as tf
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple

import ptycho.params as params

class SerializationError(Exception):
    pass

class LoggedDataNotFoundError(Exception):
    pass

def make_invocation_counter():
    count = 0
    def increment():
        nonlocal count
        count += 1
        return count
    return increment

# TODO surround each function's output section in xml tags with the function / 
# method path
def debug(log_to_file: bool = True):
    def decorator(func: Callable):
        increment_count = make_invocation_counter()

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if params.get('debug'):
                invocation_count = increment_count()

                if invocation_count <= 2:
                    timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
                    module_path = inspect.getmodule(func).__name__
                    function_name = func.__name__

                    def serialize_input(arg: Any) -> str:
                        if isinstance(arg, np.ndarray):
                            return f"NumPy array with shape {arg.shape} and data type {arg.dtype}"
                        elif isinstance(arg, tf.Tensor):
                            return f"TensorFlow tensor with shape {arg.shape} and data type {arg.dtype}"
                        elif isinstance(arg, (int, float, str, bool)):
                            return f"{type(arg).__name__} with value {arg}"
                        else:
                            return str(type(arg))

                    serializable_inputs = {
                        'args': [serialize_input(arg) for arg in args],
                        'kwargs': {key: serialize_input(value) for key, value in kwargs.items()}
                    }

                    log_message = f"Calling function {function_name} in module {module_path} with inputs: {json.dumps(serializable_inputs, default=str)}"
                    print(log_message)

                    if log_to_file:
                        log_directory = os.path.join(os.getcwd(), 'logs', module_path)
                        os.makedirs(log_directory, exist_ok=True)
                        log_file_path = os.path.join(log_directory, f"{function_name}_{timestamp}.log")
                        try:
                            with open(log_file_path, 'w') as log_file:
                                log_file.write(log_message + '\n')
                        except IOError as e:
                            print(f"Error writing log file: {e}")

                    start_time = datetime.now()
                    try:
                        result = func(*args, **kwargs)
                    except Exception as e:
                        error_message = f"Error executing function {function_name} in module {module_path}: {str(e)}"
                        print(error_message)
                        raise e
                    end_time = datetime.now()
                    execution_time = end_time - start_time

                    serializable_result = serialize_input(result)

                    log_message = f"Function {function_name} in module {module_path} returned: {serializable_result}"
                    print(log_message)
                    print(f"Execution time: {execution_time}")

                    if log_to_file:
                        try:
                            with open(log_file_path, 'a') as log_file:
                                log_file.write(log_message + '\n')
                                log_file.write(f"Execution time: {execution_time}\n")
                        except IOError as e:
                            print(f"Error writing log file: {e}")

                else:
                    result = func(*args, **kwargs)

            else:
                result = func(*args, **kwargs)

            return result

        return wrapper

    return decorator

def load_logged_data(module_path: str, function_name: str, invocation_index: int = 0) -> Tuple[Dict[str, Any], Any]:
    log_directory = os.path.join(os.getcwd(), 'logs', module_path)
    log_files = [f for f in os.listdir(log_directory) if f.startswith(f"{function_name}_")]
    log_files.sort()

    if invocation_index >= len(log_files):
        raise LoggedDataNotFoundError(f"Invocation index {invocation_index} not found for function {function_name} in module {module_path}")

    log_file_path = os.path.join(log_directory, log_files[invocation_index])

    try:
        with open(log_file_path, 'r') as log_file:
            lines = log_file.readlines()
            inputs_line = lines[0].strip()
            outputs_line = lines[1].strip()

            inputs_start = inputs_line.find(': ') + 2
            outputs_start = outputs_line.find(': ') + 2

            inputs_json = inputs_line[inputs_start:]
            outputs_str = outputs_line[outputs_start:]

            inputs = json.loads(inputs_json)
            outputs = outputs_str

            return inputs, outputs
    except (IOError, json.JSONDecodeError) as e:
        raise LoggedDataNotFoundError(f"Error loading logged data for function {function_name} in module {module_path}: {str(e)}")

import os
import json
from typing import List, Tuple, Union
from ptycho.logging import LoggedDataNotFoundError, load_logged_data

def get_type_and_dim(serialized_data: str) -> str:
    if serialized_data.startswith("NumPy array"):
        shape_start = serialized_data.find("shape") + len("shape")
        shape_end = serialized_data.find("and data type")
        shape = eval(serialized_data[shape_start:shape_end].strip())
        dtype = serialized_data[shape_end + len("and data type"):].strip()
        return f"NumPy array, shape: {shape}, dtype: {dtype}"
    elif serialized_data.startswith("TensorFlow tensor"):
        shape_start = serialized_data.find("shape") + len("shape")
        shape_end = serialized_data.find("and data type")
        shape = eval(serialized_data[shape_start:shape_end].strip())
        dtype = serialized_data[shape_end + len("and data type"):].strip()
        return f"TensorFlow tensor, shape: {shape}, dtype: {dtype}"
    else:
        return serialized_data.split(" ")[0]

def process_log_file(module_path: str, function_name: str) -> None:
    if function_name.startswith("__init__"):
        return

    invocation_index = 0
    try:
        inputs, outputs = load_logged_data(module_path, function_name, invocation_index)
    except LoggedDataNotFoundError:
        return

    input_types_dims = []
    for input_data in inputs["args"]:
        input_types_dims.append(get_type_and_dim(input_data))
    for input_name, input_data in inputs["kwargs"].items():
        input_types_dims.append(f"{input_name}: {get_type_and_dim(input_data)}")

    output_type_dim = get_type_and_dim(outputs)

    print(f"Module: {module_path}, Function: {function_name}")
    print("Input types and dimensionalities:")
    for input_type_dim in input_types_dims:
        print(f"  - {input_type_dim}")
    print(f"Output type and dimensionality: {output_type_dim}")
    print()

def extract_logged_data(log_directory: str) -> None:
    for module_name in os.listdir(log_directory):
        module_directory = os.path.join(log_directory, module_name)
        for log_file in os.listdir(module_directory):
            function_name = log_file.split("_")[0]
            process_log_file(module_name, function_name)

# TODO this function belongs among the tests
def main() -> None:
    log_directory = "logs/"
    extract_logged_data(log_directory)

####
# tests
####
# Test case 1: Function with serializable inputs and output
@debug()
def add_numbers(a: int, b: int) -> int:
    return a + b

# Test case 2: Function with NumPy array input and output
@debug()
def multiply_array(arr: np.ndarray) -> np.ndarray:
    return arr * 2

# Test case 3: Function with TensorFlow tensor input and output
@debug()
def add_tensors(t1: tf.Tensor, t2: tf.Tensor) -> tf.Tensor:
    return t1 + t2

# Test case 4: Function with mixed input types and custom object output
class CustomResult:
    def __init__(self, value: str):
        self.value = value

@debug()
def process_data(data: Any, flag: bool) -> CustomResult:
    if flag:
        return CustomResult("Processed: " + str(data))
    else:
        return CustomResult("Skipped: " + str(data))

# Test case 5: Function with exception
@debug()
def divide_numbers(a: int, b: int) -> float:
    return a / b

# Test case 6: Loading logged data from disk
@debug(log_to_file=True)
def multiply_numbers(a: int, b: int) -> int:
    return a * b

## Set the debug parameter to True
#params.cfg['debug'] = True
#
## Running the tests
#add_numbers(3, 5)
#add_numbers(4, 6)
#add_numbers(5, 7)  # This invocation will not be logged
#multiply_array(np.array([1, 2, 3]))
#multiply_array(np.array([4, 5, 6]))
#multiply_array(np.array([7, 8, 9]))  # This invocation will not be logged
#add_tensors(tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), tf.constant([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]))
#add_tensors(tf.constant([[1.0, 2.0], [3.0, 4.0]]), tf.constant([[5.0, 6.0], [7.0, 8.0]]))
#add_tensors(tf.constant([1.0, 2.0, 3.0]), tf.constant([4.0, 5.0, 6.0]))  # This invocation will not be logged
#process_data({"key": "value"}, True)
#process_data({"key": "value"}, False)
#process_data([1, 2, 3], True)  # This invocation will not be logged
#try:
#    divide_numbers(10, 0)
#except ZeroDivisionError:
#    pass
#try:
#    divide_numbers(20, 0)
#except ZeroDivisionError:
#    pass
#try:
#    divide_numbers(30, 0)  # This invocation will not be logged
#except ZeroDivisionError:
#    pass
#
#multiply_numbers(2, 3)
#multiply_numbers(4, 5)
#multiply_numbers(6, 7)  # This invocation will not be logged
#
## Loading logged data from disk
#module_path = "__main__"
#function_name = "multiply_numbers"
#invocation_index = 0
#
#inputs, output = load_logged_data(module_path, function_name, invocation_index)
#
#print(f"Loaded inputs: {inputs}")
#print(f"Loaded output: {output}")
#
## Cleanup: Remove the logged data files
#log_directory = os.path.join(os.getcwd(), 'logs', module_path)
#log_files = [f for f in os.listdir(log_directory) if f.startswith(f"{function_name}_")]
#for log_file in log_files:
#    log_file_path = os.path.join(log_directory, log_file)
#    os.remove(log_file_path)
#
## Set the debug parameter to False
#params.cfg['debug'] = False
#
## Running the tests again (no logging should occur)
#add_numbers(3, 5)
#multiply_array(np.array([1, 2, 3]))
#add_tensors(tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), tf.constant([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]))
#process_data({"key": "value"}, True)
#try:
#    divide_numbers(10, 0)
#except ZeroDivisionError:
#    pass
#multiply_numbers(2, 3)
#
#
