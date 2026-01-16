from .serializer import Serializer
# spec
#    @depends_on(Serializer)
#    interface Logger {
#        """
#        Logs function call details to a specified log file.
#
#        Preconditions:
#        - `args` and `kwargs` are serialized using pickle.
#        - `log_file_path` must be a valid file path with write permissions.
#          The directory containing the file must exist.
#
#        Postconditions:
#        - The serialized function arguments and keyword arguments are written to the log file.
#          The log entry is formatted as a single line JSON string.
#        - If there is an error during logging, an error message is printed to stderr.
#        """
#        void logCall(bytes args, bytes kwargs, string log_file_path);
#
#        """
#        Logs function return details to the specified log file.
#
#        Preconditions:
#        - `result` is serialized using pickle.
#        - `log_file_path` must be a valid file path with write permissions.
#          The directory containing the file must exist.
#
#        Postconditions:
#        - The serialized `result` and `execution_time` are appended to the log file.
#          The log entry is formatted as a single line JSON string.
#        - If there is an error during logging, an error message is printed to stderr.
#        """
#        void logReturn(bytes result, float execution_time, string log_file_path);
#
#        """
#        Logs an error message to the specified log file.
#
#        Preconditions:
#        - `log_file_path` must be a valid file path with write permissions.
#          The directory containing the file must exist.
#
#        Postconditions:
#        - The `error` message is written to the log file.
#          The log entry is formatted as a single line JSON string.
#        - If there is an error during logging, an error message is printed to stderr.
#        """
#        void logError(string error, string log_file_path);
#
#        """
#        Loads a logged dataset from a log file.
#
#        Preconditions:
#        - `log_file_path` must be a valid file path with read permissions.
#          The file must contain valid JSON-formatted log entries.
#
#        Postconditions:
#        - Returns a list or tuple containing the logged inputs and output.
#        - If there is an error during loading, returns an empty list or tuple.
#        """
#        Union[list, tuple] loadLog(Configuration configuration);
#
#        """
#        Searches the log directory and returns all valid log file paths.
#
#        Preconditions:
#        - `log_directory` must be a valid directory path with read permissions.
#
#        Postconditions:
#        - Returns a list of valid log file paths adhering to the format ^(?P<log_path_prefix>[a-z0-9]+)/(?P<python_namespace_path>([a-z0-9]+\.)+)log$.?
#        - Invalid log file paths are filtered out using the validateLogFilePath method.
#        - If there are no valid log files or an error occurs during searching, returns an empty list.
#        """
#        list[str] searchLogDirectory(string log_directory);
#
#        """
#        Validates a log file path against the expected format.
#
#        Preconditions:
#        - `log_file_path` must be a string representing a file path.
#
#        Postconditions:
#        - Returns True if the `log_file_path` adheres to the format '^(?P<log_path_prefix>[a-z0-9]+)/(?P<python_namespace_path>([a-z0-9]+\.)+)log$.', False otherwise.
#        """
#        bool validateLogFilePath(string log_file_path);
#    };

import json
import os
import sys
import pickle
from typing import Any, Union, List
import re

class Logger:
    def __init__(self):
        self.serializer = Serializer()

    def logCall(self, args: bytes, kwargs: bytes, log_file_path: str) -> None:
        try:
            with open(log_file_path, 'a') as log_file:
                log_entry = json.dumps({
                    "args": args.hex(),
                    "kwargs": kwargs.hex()
                })
                log_file.write(log_entry + "\n")
        except Exception as e:
            print(f"Error logging function call: {e}", file=sys.stderr)

    def logReturn(self, result: bytes, execution_time: float, log_file_path: str) -> None:
        try:
            with open(log_file_path, 'a') as log_file:
                log_entry = json.dumps({
                    "result": result.hex(),
                    "execution_time": execution_time
                })
                log_file.write(log_entry + "\n")
        except Exception as e:
            print(f"Error logging function return: {e}", file=sys.stderr)

    def logError(self, error: str, log_file_path: str) -> None:
        pass
#        try:
#            with open(log_file_path, 'a') as log_file:
#                log_entry = json.dumps({
#                    "error": error
#                })
#                log_file.write(log_entry + "\n")
#        except Exception as e:
#            print(f"Error logging error: {e}", file=sys.stderr)

    def loadLog(self, log_file_path: str) -> Union[List, tuple]:
        logs = []
        try:
            with open(log_file_path, 'r') as log_file:
                for line in log_file:
                    log_entry = json.loads(line)
                    if "args" in log_entry:
                        log_entry["args"] = bytes.fromhex(log_entry["args"])
                    if "kwargs" in log_entry:
                        log_entry["kwargs"] = bytes.fromhex(log_entry["kwargs"])
                    if "result" in log_entry:
                        log_entry["result"] = bytes.fromhex(log_entry["result"])
                    logs.append(log_entry)
        except Exception as e:
            print(f"Error loading log: {e}", file=sys.stderr)
        return logs

    def searchLogDirectory(self, log_directory: str) -> List[str]:
        valid_log_files = []
        try:
            for root, _, files in os.walk(log_directory):
                for file in files:
                    file_path = os.path.relpath(os.path.join(root, file), start=log_directory)
                    if self.validateLogFilePath(file_path):
                        valid_log_files.append(os.path.join(log_directory, file_path))
        except Exception as e:
            print(f"Error searching log directory: {e}", file=sys.stderr)
        return valid_log_files

    def validateLogFilePath(self, log_file_path: str) -> bool:
        return True
        pattern = r'^(?P<log_path_prefix>[a-z0-9]+)/(?P<python_namespace_path>([a-z0-9]+\.)+)log$'
        return re.match(pattern, log_file_path) is not None

import unittest
import tempfile

class TestLogger(unittest.TestCase):
    def setUp(self):
        self.logger = Logger()
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_file = os.path.join(self.test_dir.name, 'test.log')
        
    def tearDown(self):
        self.test_dir.cleanup()

    def test_logCall(self):
        args = self.logger.serializer.serialize(('arg1', 'arg2'))
        kwargs = self.logger.serializer.serialize({'key': 'value'})
        self.logger.logCall(args, kwargs, self.test_file)
        
        with open(self.test_file, 'r') as log_file:
            log_entry = json.loads(log_file.readline())
            self.assertEqual(log_entry["args"], args.hex())
            self.assertEqual(log_entry["kwargs"], kwargs.hex())

    def test_logReturn(self):
        result = self.logger.serializer.serialize('result')
        execution_time = 0.123
        self.logger.logReturn(result, execution_time, self.test_file)
        
        with open(self.test_file, 'r') as log_file:
            log_entry = json.loads(log_file.readline())
            self.assertEqual(log_entry["result"], result.hex())
            self.assertEqual(log_entry["execution_time"], execution_time)

    def test_logError(self):
        error = "Test error message"
        self.logger.logError(error, self.test_file)
        
        with open(self.test_file, 'r') as log_file:
            log_entry = json.loads(log_file.readline())
            self.assertEqual(log_entry["error"], error)

    def test_loadLog(self):
        args = self.logger.serializer.serialize(('arg1', 'arg2'))
        kwargs = self.logger.serializer.serialize({'key': 'value'})
        result = self.logger.serializer.serialize('result')
        execution_time = 0.123
        
        self.logger.logCall(args, kwargs, self.test_file)
        self.logger.logReturn(result, execution_time, self.test_file)
        
        logs = self.logger.loadLog(self.test_file)
        self.assertEqual(len(logs), 2)
        self.assertEqual(logs[0]["args"], args)
        self.assertEqual(logs[0]["kwargs"], kwargs)
        self.assertEqual(logs[1]["result"], result)
        self.assertEqual(logs[1]["execution_time"], execution_time)

    def test_searchLogDirectory(self):
        valid_file = os.path.join(self.test_dir.name, 'logs/module.samplefunc.log')
        invalid_file = os.path.join(self.test_dir.name, 'invalid.log')
        
        os.makedirs(os.path.dirname(valid_file), exist_ok=True)
        
        with open(valid_file, 'w'), open(invalid_file, 'w'):
            pass
        
        valid_files = self.logger.searchLogDirectory(self.test_dir.name)
        self.assertIn(valid_file, valid_files)
        self.assertNotIn(invalid_file, valid_files)

    def test_validateLogFilePath(self):
        valid_path = 'logs/module.samplefunc.log'
        invalid_path = 'invalid.log'
        
        self.assertTrue(self.logger.validateLogFilePath(valid_path))
        self.assertFalse(self.logger.validateLogFilePath(invalid_path))

if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)
