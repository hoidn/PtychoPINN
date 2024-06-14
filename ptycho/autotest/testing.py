from .logger import Logger
from .functionmapping import FunctionMapping
from .configuration import Configuration

import unittest
from unittest.mock import MagicMock

# skipped_count

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
from logger import Logger
from functionmapping import FunctionMapping
from configuration import Configuration

import unittest
from unittest.mock import MagicMock

# skipped_count

from typing import List, Tuple, Any, Optional, Callable, Union

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
#
#class Testing:
#    def __init__(self, logger: Logger, function_mapping: FunctionMapping):
#        self.logger = logger
#        self.function_mapping = function_mapping
#
#    def testCallable(self, log_path_prefix: str, func: Callable) -> bool:
#        log_files = self.logger.searchLogDirectory(log_path_prefix)
#        for log_file in log_files:
#            logs = self.logger.loadLog(log_file)
#            for log in logs:
#                inputs = log['args']
#                expected_output = log['result']
#                try:
#                    deserialized_inputs = self.logger.serializer.deserialize(inputs)
#                    deserialized_expected_output = self.logger.serializer.deserialize(expected_output)
#                    actual_output = func(*deserialized_inputs)
#                    if actual_output != deserialized_expected_output:
#                        return False
#                except Exception as e:
#                    print(f"Error testing function: {e}")
#                    return False
#        return True
#
#    def createTestCase(self, log_path_prefix: str) -> Union[tuple, None]:
#        log_files = self.logger.searchLogDirectory(log_path_prefix)
#        for log_file in log_files:
#            logs = self.logger.loadLog(log_file)
#            if logs:
#                log = logs[0]
#                inputs = log['args']
#                expected_output = log['result']
#                func = self.function_mapping.load_function(log_file)
#                if func is not None:
#                    return (inputs, expected_output, func)
#        return None
#
#    def runTestSuite(self, log_path_prefix: str) -> TestSummary:
#        summary = TestSummary()
#        log_files = self.logger.searchLogDirectory(log_path_prefix)
#        for log_file in log_files:
#            test_case = self.createTestCase(log_path_prefix)
#            if test_case is not None:
#                inputs, expected_output, func = test_case
#                if self.testCallable(log_path_prefix, func):
#                    summary.increment_passed()
#                else:
#                    summary.increment_failed()
#            else:
#                summary.increment_skipped()
#        return summary
#
#
#
#def add(x, y):
#    return x + y
#
#def multiply(x, y):
#    return x * y
#
#def divide(x, y):
#    return x / y
#
#
#class TestTesting(unittest.TestCase):
#    def setUp(self):
#        self.logger = Logger()
#        self.function_mapping = FunctionMapping()
#        self.testing = Testing(self.logger, self.function_mapping)
#
#    def test_testCallable(self):
#        log_path_prefix = 'test_logs'
#        self.logger.logResult(log_path_prefix + '/add', (3, 4), 7)
#        self.assertTrue(self.testing.testCallable(log_path_prefix, add))
#
#    def test_createTestCase(self):
#        log_path_prefix = 'test_logs'
#        self.logger.logResult(log_path_prefix + '/add', (3, 4), 7)
#        self.function_mapping.save_function(log_path_prefix + '/add', add)
#        test_case = self.testing.createTestCase(log_path_prefix)
#        self.assertIsNotNone(test_case)
#        inputs, expected_output, func = test_case
#        self.assertEqual(self.logger.serializer.deserialize(inputs), (3, 4))
#        self.assertEqual(self.logger.serializer.deserialize(expected_output), 7)
#        self.assertEqual(func, add)
#
#    def test_runTestSuite(self):
#        log_path_prefix = 'test_logs'
#        self.logger.logResult(log_path_prefix + '/add', (3, 4), 7)
#        self.logger.logResult(log_path_prefix + '/multiply', (3, 4), 12)
#        self.function_mapping.save_function(log_path_prefix + '/add', add)
#        self.function_mapping.save_function(log_path_prefix + '/multiply', multiply)
#        summary = self.testing.runTestSuite(log_path_prefix)
#        self.assertIsInstance(summary, TestSummary)
#        self.assertEqual(summary.passed, 2)
#        self.assertEqual(summary.failed, 0)
#        self.assertEqual(summary.skipped, 0)
#
#if __name__ == '__main__':
#    unittest.main(argv=[''], verbosity=2, exit=False)
#
##from logger import Logger
##from functionmapping import FunctionMapping
##from configuration import Configuration
##
##import unittest
##from unittest.mock import MagicMock
##
### skipped_count
##
##from typing import List, Tuple, Any, Optional, Callable, Union
##
##class Testing:
##    def __init__(self, logger: Logger, function_mapping: FunctionMapping):
##        self.logger = logger
##        self.function_mapping = function_mapping
##
##    def testCallable(self, log_path_prefix: str, func: Callable) -> bool:
##        log_files = self.logger.searchLogDirectory(log_path_prefix)
##        for log_file in log_files:
##            logs = self.logger.loadLog(log_file)
##            for log in logs:
##                inputs = log['args']
##                expected_output = log['result']
##                try:
##                    deserialized_inputs = self.logger.serializer.deserialize(inputs)
##                    deserialized_expected_output = self.logger.serializer.deserialize(expected_output)
##                    actual_output = func(*deserialized_inputs)
##                    if actual_output != deserialized_expected_output:
##                        return False
##                except Exception as e:
##                    print(f"Error testing function: {e}")
##                    return False
##        return True
##
##    def createTestCase(self, log_path_prefix: str) -> Union[tuple, None]:
##        log_files = self.logger.searchLogDirectory(log_path_prefix)
##        for log_file in log_files:
##            logs = self.logger.loadLog(log_file)
##            if logs:
##                log = logs[0]
##                inputs = log['args']
##                expected_output = log['result']
##                func = self.function_mapping.load_function(log_file)
##                if func is not None:
##                    return (inputs, expected_output, func)
##        return None
##
##    def runTestSuite(self, log_path_prefix: str) -> TestSummary:
##        summary = TestSummary()
##        log_files = self.logger.searchLogDirectory(log_path_prefix)
##        for log_file in log_files:
##            test_case = self.createTestCase(log_path_prefix)
##            if test_case is not None:
##                inputs, expected_output, func = test_case
##                if self.testCallable(log_path_prefix, func):
##                    summary.increment_passed()
##                else:
##                    summary.increment_failed()
##            else:
##                summary.increment_skipped()
##        return summary
##
##class TestSummary:
##    def __init__(self):
##        self.passed = 0
##        self.failed = 0
##        self.skipped = 0
##
##    def increment_passed(self):
##        self.passed += 1
##
##    def increment_failed(self):
##        self.failed += 1
##
##    def increment_skipped(self):
##        self.skipped += 1
##
##    def __repr__(self):
##        return f"TestSummary(passed={self.passed}, failed={self.failed}, skipped={self.skipped})"
##
##
##def add(x, y):
##    return x + y
##
##def multiply(x, y):
##    return x * y
##
##def divide(x, y):
##    return x / y
##
##class TestTesting(unittest.TestCase):
##    def setUp(self):
##        self.logger = Logger()
##        self.function_mapping = FunctionMapping()
##        self.testing = Testing(self.logger, self.function_mapping)
##
##        # Mocking logger and function_mapping methods
##        self.logger.searchLogDirectory = MagicMock(return_value=['log1'])
##        self.logger.loadLog = MagicMock(return_value=[{
##            'args': self.logger.serializer.serialize((3, 4)),
##            'result': self.logger.serializer.serialize(7)
##        }])
##        self.function_mapping.load_function = MagicMock(return_value=add)
##
##    def test_testCallable(self):
##        log_path_prefix = 'test_logs'
##        self.assertTrue(self.testing.testCallable(log_path_prefix, add))
##
##    def test_createTestCase(self):
##        log_path_prefix = 'test_logs'
##        test_case = self.testing.createTestCase(log_path_prefix)
##        self.assertIsNotNone(test_case)
##
##    def test_runTestSuite(self):
##        log_path_prefix = 'test_logs'
##        summary = self.testing.runTestSuite(log_path_prefix)
##        self.assertIsInstance(summary, TestSummary)
##
##if __name__ == '__main__':
##    unittest.main(argv=[''], verbosity=2, exit=False)
##
##
###class TestSummary:
###    def __init__(self, passed: int, failed: int, skipped: int):
###        self.passed = passed
###        self.failed = failed
###        self.skipped = skipped
###
###class Testing:
###    def __init__(self, logger: Logger, function_mapping: FunctionMapping):
###        self.logger = logger
###        self.function_mapping = function_mapping
###
###    def testCallable(self, log_file_path: str, func: Callable) -> bool:
###        # Load the logged inputs and expected output from the log file
###        logged_data = self.logger.loadLog(log_file_path)
###        inputs, expected_output = logged_data
###        print(f"Loaded inputs: {inputs}")
###        print(f"Expected output: {expected_output}")
###
###        # Invoke the function with the logged inputs
###        if len(inputs) == 2:
###            actual_output = func(*inputs[0], **inputs[1])
###        elif len(inputs) == 1:
###            actual_output = func(*inputs[0])
###        else:
###            actual_output = func()
###
###        print(f"Actual output: {actual_output}")
###
###        # Compare the actual output with the expected output
###        return actual_output == expected_output
####    def testCallable(self, log_file_path: str, func: Callable) -> bool:
####        print(f"Testing callable for log file: {log_file_path}")
####        # Load the logged inputs and expected output from the log file
####        logged_data = self.logger.loadLog(log_file_path)
####        inputs, expected_output = logged_data
####        print(f"Loaded inputs: {inputs}")
####        print(f"Expected output: {expected_output}")
####
####        # Invoke the function with the logged inputs
####        actual_output = func(*inputs[0], **inputs[1])
####        print(f"Actual output: {actual_output}")
####
####        # Compare the actual output with the expected output
####        result = actual_output == expected_output
####        print(f"Test result: {result}")
####        return result
###
###    def createTestCase(self, log_file_path: str) -> Optional[Tuple[List[Any], Any, Callable]]:
###        print(f"Creating test case for log file: {log_file_path}")
###        # Load the logged inputs and output from the log file
###        logged_data = self.logger.loadLog(log_file_path)
###        inputs, expected_output = logged_data
###        print(f"Loaded inputs: {inputs}")
###        print(f"Expected output: {expected_output}")
###
###        # Load the function object from the log file path
###        func = self.function_mapping.load_function(log_file_path=log_file_path)
###        print(f"Loaded function: {func}")
###
###        if func is None:
###            print("Function not found. Skipping test case.")
###            return None
###
###        # Construct and return the test case tuple
###        test_case = (inputs, expected_output, func)
###        print(f"Created test case: {test_case}")
###        return test_case
###
###    def runTestSuite(self, log_file_paths: List[str]) -> TestSummary:
###        print("Running test suite...")
###        passed_count = 0
###        failed_count = 0
###        skipped_count = 0
###
###        for log_file_path in log_file_paths:
###            print(f"Processing log file: {log_file_path}")
###            # Create a test case from the log file
###            test_case = self.createTestCase(log_file_path)
###
###            if test_case is None:
###                skipped_count += 1
###                print("Test case skipped.")
###                continue
###
###            inputs, expected_output, func = test_case
###            print(f"Executing test case: {test_case}")
###
###            # Execute the test case
###            if self.testCallable(log_file_path, func):
###                passed_count += 1
###                print("Test case passed.")
###            else:
###                failed_count += 1
###                print("Test case failed.")
###
###        # Construct and return the test summary
###        test_summary = TestSummary(passed_count, failed_count, skipped_count)
###        print(f"Test summary: {test_summary.__dict__}")
###        return test_summary
###
###if __name__ == '__main__':
###    logdir = 'logs'
###    # Instantiate configuration with debugging enabled
###    configuration = Configuration(debug=True, log_file_prefix=logdir)
###    print(f"Configuration: {configuration.__dict__}")
###
###    # Instantiate Logger and FunctionMapping with new configuration
###    test_logger = Logger()
###    test_function_mapping = FunctionMapping(logdir)
###    print(f"Logger: {test_logger}")
###    print(f"FunctionMapping: {test_function_mapping}")
###
###    testing = Testing(test_logger, test_function_mapping)
###    test_log_paths = test_logger.searchLogDirectory(logdir)
###    print(f"Test log paths: {test_log_paths}")
###
###    # Example test: Assume we have valid log paths and functions are available
###    #test_log_paths = [f"{logdir}/__main__.valid_function.log", f"{logdir}/__main__.invalid_function.log"]
###    test_summary = testing.runTestSuite(test_log_paths)
###    print(f"Final test summary: {test_summary.__dict__}")
