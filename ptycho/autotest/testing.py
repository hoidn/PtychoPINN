from .logger import Logger
from .functionmapping import FunctionMapping
from .configuration import Configuration
import unittest
from logger import Logger
from functionmapping import FunctionMapping

from typing import List, Tuple, Any, Optional, Callable, Union

class TestSummary:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def increment_passed(self):
        self.passed += 1

    def increment_failed(self):
        self.failed += 1

    def increment_skipped(self):
        self.skipped += 1

    def __repr__(self):
        return f"TestSummary(passed={self.passed}, failed={self.failed}, skipped={self.skipped})"

class Testing:
    def __init__(self, logger: Logger, function_mapping: FunctionMapping):
        self.logger = logger
        self.function_mapping = function_mapping

class Testing:
    def __init__(self, logger: Logger, function_mapping: FunctionMapping):
        self.logger = logger
        self.function_mapping = function_mapping

    def testCallable(self, log_path_prefix: str, func: Callable) -> bool:
        print(f"Debug: testCallable called with log_path_prefix: {log_path_prefix}")
        log_files = self.logger.searchLogDirectory(log_path_prefix)
        print(f"Debug: Found log files: {log_files}")
        for log_file in log_files:
            logs = self.logger.loadLog(log_file)
            #print(f"Debug: Loaded logs: {logs}")
            for i in range(len(logs) // 2):
                args = logs[2 * i]['args']
                kwargs = logs[2 * i]['kwargs']
                expected_output = logs[2 * i + 1]['result']
                try:
                    deserialized_args = self.logger.serializer.deserialize(args)
                    deserialized_kwargs = self.logger.serializer.deserialize(kwargs)
                    deserialized_expected_output = self.logger.serializer.deserialize(expected_output)
                    actual_output = func(*deserialized_args, **deserialized_kwargs)
                    #print(f"Debug: Actual output: {actual_output}")
                    if actual_output != deserialized_expected_output:
                        print("Debug: Test failed")
                        return False
                except Exception as e:
                    print(f"Error testing function: {e}")
                    return False
        print("Debug: Test passed")
        return True

    def createTestCase(self, log_path_prefix: str) -> Union[tuple, None]:
        print(f"Debug: createTestCase called with log_path_prefix: {log_path_prefix}")
        log_files = self.logger.searchLogDirectory(log_path_prefix)
        print(f"Debug: Found log files: {log_files}")
        for log_file in log_files:
            logs = self.logger.loadLog(log_file)
            #print(f"Debug: Loaded logs: {logs}")
            if logs:
                log = logs[0]
                inputs = log['args']
                expected_output = log['result']
                func = self.function_mapping.load_function(log_file)
                print(f"Debug: Loaded function: {func}")
                if func is not None:
                    return (inputs, expected_output, func)
        print("Debug: No test case found")
        return None

    def runTestSuite(self, log_path_prefix: str) -> TestSummary:
        print(f"Debug: runTestSuite called with log_path_prefix: {log_path_prefix}")
        summary = TestSummary()
        log_files = self.logger.searchLogDirectory(log_path_prefix)
        print(f"Debug: Found log files: {log_files}")
        for log_file in log_files:
            test_case = self.createTestCase(log_path_prefix)
            if test_case is not None:
                inputs, expected_output, func = test_case
                if self.testCallable(log_path_prefix, func):
                    summary.increment_passed()
                else:
                    summary.increment_failed()
            else:
                summary.increment_skipped()
        print(f"Debug: Test summary: {summary}")
        return summary

class TestSummary:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def increment_passed(self):
        self.passed += 1

    def increment_failed(self):
        self.failed += 1

    def increment_skipped(self):
        self.skipped += 1

    def __repr__(self):
        return f"TestSummary(passed={self.passed}, failed={self.failed}, skipped={self.skipped})"


def add(x, y):
    return x + y

def multiply(x, y):
    return x * y

def divide(x, y):
    return x / y

class TestTesting(unittest.TestCase):
    def setUp(self):
        self.logger = Logger()
        self.function_mapping = FunctionMapping()
        self.testing = Testing(self.logger, self.function_mapping)

    def test_testCallable(self):
        log_path_prefix = 'test_logs'
        self.logger.logReturn(log_path_prefix + '/add', (3, 4), 7)
        self.assertTrue(self.testing.testCallable(log_path_prefix, add))

    def test_createTestCase(self):
        log_path_prefix = 'test_logs'
        self.logger.logReturn(log_path_prefix + '/add', (3, 4), 7)
        self.function_mapping.save_function(log_path_prefix + '/add', add)
        test_case = self.testing.createTestCase(log_path_prefix)
        self.assertIsNotNone(test_case)
        inputs, expected_output, func = test_case
        self.assertEqual(self.logger.serializer.deserialize(inputs), (3, 4))
        self.assertEqual(self.logger.serializer.deserialize(expected_output), 7)
        self.assertEqual(func, add)

    def test_runTestSuite(self):
        log_path_prefix = 'test_logs'
        self.logger.logReturn(log_path_prefix + '/add', (3, 4), 7)
        self.logger.logReturn(log_path_prefix + '/multiply', (3, 4), 12)
        self.function_mapping.save_function(log_path_prefix + '/add', add)
        self.function_mapping.save_function(log_path_prefix + '/multiply', multiply)
        summary = self.testing.runTestSuite(log_path_prefix)
        self.assertIsInstance(summary, TestSummary)
        self.assertEqual(summary.passed, 2)
        self.assertEqual(summary.failed, 0)
        self.assertEqual(summary.skipped, 0)

if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)
