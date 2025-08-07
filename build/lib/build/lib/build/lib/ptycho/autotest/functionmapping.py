# spec
#    interface FunctionMapping {
#        """
#        Retrieves the log file path for a given function.
#
#        Preconditions:
#        - `func` must be a callable.
#        - `log_directory` must be a valid directory path.
#        - Expected JSON format: { "log_directory": "string" }
#
#        Postconditions:
#        - Returns the log file path for the given function, formatted as `prefix/module.fname<suffix>.log`.
#        - If `log_directory` is not provided or is an empty string, returns an empty string.
#        """
#        string getLogFilePath(Callable func, string log_directory);
#
#        """
#        Loads a function given its log file path or module path.
#
#        Preconditions:
#        - `log_file_path` must be a valid log file path or empty string.
#        - `module_path` must be a valid module path or empty string.
#        - Expected JSON format: { "log_file_path": "string", "module_path": "string" }
#
#        Postconditions:
#        - Returns the function object if successfully loaded.
#        - If the function cannot be found or imported, returns None.
#        """
#        Union[Callable, None] loadFunction(string log_file_path, string module_path);
#
#        """
#        Retrieves the module path for a given function.
#
#        Preconditions:
#        - `func` must be a callable.
#
#        Postconditions:
#        - Returns the module path for the given function, formatted as `module.fname`.
#        - If `func` is a built-in function or does not have a valid module path, returns an empty string.
#        """
#        string getModulePath(Callable func);
#    };

# implementation
import os
import shutil
import importlib
from typing import Callable, Optional

def dprint(*args):
    pass

class FunctionMapping:
    def __init__(self, log_directory: str = "logs"):
        self.log_directory = log_directory

    def get_log_file_path(self, func: Callable) -> str:
        """
        Retrieves the log file path for a given function.
        
        Preconditions:
        - `func` must be a callable.
        
        Postconditions:
        - Returns the log file path for the given function, formatted as `prefix/module.fname<suffix>.log`.
        >>> function_mapping = FunctionMapping(log_directory="test_logs")
        >>> def sample_function():
        ...     return "sample function executed"
        >>> function_mapping.get_log_file_path(sample_function)
        'test_logs/__main__.sample_function.log'
        """
        module_name = func.__module__
        func_name = func.__name__
        log_file_path = f"{self.log_directory}/{module_name}.{func_name}.log"
        return log_file_path

    def save_function(self, log_file_path: str, func: Callable) -> None:
        module_path, func_name = self.get_module_and_function_from_log_path(log_file_path)
        module = importlib.import_module(module_path)
        setattr(module, func_name, func)

    def load_function_from_path(self, log_file_path: str) -> Optional[Callable]:
        try:
            dprint(f"log_file_path: {log_file_path}")
            module_path, func_name = self.get_module_and_function_from_log_path(log_file_path)
            dprint(f"module_path: {module_path}")
            dprint(f"func_name: {func_name}")
            dprint(f"Importing module: {module_path}")
            module = importlib.import_module(module_path)
            dprint(f"Imported module: {module}")
            dprint(f"Retrieving function: {func_name}")
            func = getattr(module, func_name, None)
            dprint(f"Retrieved function: {func}")
            return func
        except Exception as e:
            dprint(f"Error loading function: {e}")
            return None

    def get_module_and_function_from_log_path(self, log_file_path: str) -> tuple:
        dprint(f"log_file_path: {log_file_path}")
        log_file_path = log_file_path.replace(f"{self.log_directory}/", "")
        dprint(f"log_file_path after removing log_directory: {log_file_path}")
        log_file_path = log_file_path.replace(".log", "")
        dprint(f"log_file_path after removing .log: {log_file_path}")
        parts = log_file_path.rsplit(".", 1)
        print(parts)
        dprint(f"parts: {parts}")
        module_path = parts[0]
        dprint(f"module_path: {module_path}")
        func_name = parts[1]
        dprint(f"func_name: {func_name}")
        return module_path, func_name

    def load_function(self, log_file_path: str) -> Optional[Callable]:
        """
        Loads a function given its log file path.
        
        Preconditions:
        - `log_file_path` must be valid.
        
        Postconditions:
        - Returns the function object if successfully loaded.
        - If the function cannot be found or imported, returns None.
        >>> function_mapping = FunctionMapping(log_directory="test_logs")
        >>> def sample_function():
        ...     return "sample function executed"
        >>> log_file_path = function_mapping.get_log_file_path(sample_function)
        >>> loaded_func = function_mapping.load_function(log_file_path)
        """
        return self.load_function_from_path(log_file_path)

    def get_module_path(self, func: Callable) -> str:
        """
        Retrieves the module path for a given function.
        
        Preconditions:
        - `func` must be a callable.
        
        Postconditions:
        - Returns the module path for the given function, formatted as `module.fname`.
        >>> function_mapping = FunctionMapping(log_directory="test_logs")
        >>> def sample_function():
        ...     return "sample function executed"
        >>> function_mapping.get_module_path(sample_function)
        '__main__.sample_function'
        """
        module_name = func.__module__
        func_name = func.__name__
        module_path = f"{module_name}.{func_name}"
        return module_path


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)

def sample_function():
    return "sample function executed"

def another_function():
    return "another function executed"

def test_get_log_file_path():
    function_mapping = FunctionMapping(log_directory="test_logs")
    path = function_mapping.get_log_file_path(sample_function)
    assert path == 'test_logs/__main__.sample_function.log', f"Expected 'test_logs/__main__.sample_function.log', got '{path}'"

def test_load_function():
    function_mapping = FunctionMapping(log_directory="test_logs")
    log_file_path = function_mapping.get_log_file_path(sample_function)
    
    loaded_func = function_mapping.load_function(log_file_path=log_file_path)
    assert loaded_func is not None, "Expected function to be loaded, but got None"
    assert loaded_func.__name__ == 'sample_function', f"Expected 'sample_function', got '{loaded_func.__name__}'"

def test_get_module_path():
    function_mapping = FunctionMapping(log_directory="test_logs")
    path = function_mapping.get_module_path(sample_function)
    assert path == '__main__.sample_function', f"Expected '__main__.sample_function', got '{path}'"

if __name__ == "__main__":
    test_get_log_file_path()
    test_load_function()
    test_get_module_path()
    print("All tests passed!")
