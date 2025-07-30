from .serializer import Serializer
from .logger import Logger
from .functionmapping import FunctionMapping
from .configuration import Configuration

# spec
#    @depends_on(Logger, Configuration, FunctionMapping)
#    interface Debug {
#        """
#        Applies the debugging process to the function.
#
#        Preconditions:
#        - `func` must be a callable.
#        - Configuration must allow debugging.
#
#        Postconditions:
#        - If debugging is allowed by the Configuration:
#          - Returns a new function that wraps the original function with debugging functionality.
#          - The returned function, when called, performs two forms of logging:
#            1. Prints function call and return information to the console, surrounded by XML tags
#               containing the callable's module path and name. The console log messages are in the
#               format `<module.function>CALL/RETURN args/result</module.function>`. For all array
#               or tensor types (i.e., objects with a .shape and/or .dtype attribute), the shapes
#               and data types are also printed.
#            2. Serializes function inputs and outputs to a log file using the `logCall` and `logReturn`
#               methods of the Logger interface. The serialized data can be loaded using the `LoadLog`
#               method. If serialization fails, the console logging still occurs, but no log file is
#               generated for that invocation.
#          - Logs only the first two invocations of the function.
#        - If debugging is not allowed by the Configuration:
#          - Returns the original function unchanged, without any debugging functionality.
#        """
#        Callable decorate(Callable func);
#    };

## implementation
import time
import os
import pickle
import json
from typing import Callable, Any, List, Union, Optional
import re

def make_invocation_counter():
    count = 0
    def increment():
        nonlocal count
        count += 1
        return count
    return increment

class Debug:
    def __init__(self):
        self.configuration = Configuration()
        self.serializer = Serializer()
        self.logger = Logger()
        self.function_mapping = FunctionMapping()

    def decorate(self, func: Callable) -> Callable:
        increment_count = make_invocation_counter()
        if not self.configuration.getDebugFlag():
            return func

        else:
            module_path = self.function_mapping.get_module_path(func)
            function_name = func.__name__

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                invocation_count = increment_count()
                if invocation_count > 2:
                    return func(*args, **kwargs)
                
                log_file_path = self.function_mapping.get_log_file_path(func)
                os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

                try:
                    serialized_args = self.serializer.serialize(args)
                    serialized_kwargs = self.serializer.serialize(kwargs)
                    self.logger.logCall(serialized_args, serialized_kwargs, log_file_path)
                except ValueError:
                    pass  # If serialization fails, just proceed with console logging

                console_log_start = f"<{module_path}.{function_name}>CALL"
                console_log_args = self._formatConsoleLog(args)
                console_log_kwargs = self._formatConsoleLog(kwargs)
                print(console_log_start)
                print(console_log_args)
                print(console_log_kwargs)

                start_time = time.time()

                result = func(*args, **kwargs)
                try:
                    serialized_result = self.serializer.serialize(result)
                    self.logger.logReturn(serialized_result, time.time() - start_time, log_file_path)

                    console_log_end = f"</{module_path}.{function_name}>RETURN"
                    console_log_result = self._formatConsoleLog(result)
                    print(console_log_end + " " + console_log_result)

                except Exception as e:
                    self.logger.logError(str(e), log_file_path)
                    print(f"<{module_path}.{function_name}>ERROR {str(e)}")
                return result

            return wrapper

    def _formatConsoleLog(self, data: Any) -> str:
        if not isinstance(data, tuple):
            data = (data,)

        formatted_data = []
        for item in data:
            if hasattr(item, 'shape') and hasattr(item, 'dtype'):
                formatted_data.append(f"type={type(item)}, shape={item.shape}, dtype={item.dtype}")
            elif isinstance(item, (int, float, str, bool)):
                formatted_data.append(f"type={type(item)}, {item}")
            else:
                formatted_data.append(f"type={type(item)}")
        return ", ".join(formatted_data)

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)

import unittest

class TestDebug(unittest.TestCase):
    def setUp(self):
        self.configuration = Configuration()
        self.serializer = Serializer()
        self.logger = Logger()
        self.function_mapping = FunctionMapping()
        self.debug = Debug(self.configuration, self.serializer, self.logger, self.function_mapping)

    def test_decorate_call(self):
        @self.debug.decorate
        def add(x, y):
            return x + y

        result = add(3, 4)
        self.assertEqual(result, 7)

    def test_decorate_return(self):
        @self.debug.decorate
        def multiply(x, y):
            return x * y

        result = multiply(2, 3)
        self.assertEqual(result, 6)
        result = multiply(4, 5)
        self.assertEqual(result, 20)
        result = multiply(6, 7)  # This call should not be logged
        self.assertEqual(result, 42)

    def test_decorate_error(self):
        @self.debug.decorate
        def divide(x, y):
            return x / y

        with self.assertRaises(ZeroDivisionError):
            divide(1, 0)


obj = Debug()
debug = obj.decorate

if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)

