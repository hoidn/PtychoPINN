"""
Unit tests for the ptycho.log_config module.
"""

import unittest
import tempfile
import logging
import sys
from io import StringIO
from pathlib import Path
from ptycho.log_config import setup_logging, restore_stdout


class TestLogConfig(unittest.TestCase):
    """Test cases for centralized logging configuration."""

    def setUp(self):
        """Clear any existing handlers before each test."""
        # Restore original stdout first
        restore_stdout()
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)
        # Reset root logger level
        root_logger.setLevel(logging.WARNING)

    def tearDown(self):
        """Clean up after each test."""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)
        # Reset root logger level
        root_logger.setLevel(logging.WARNING)
        # Restore original stdout
        restore_stdout()

    def test_default_setup_logging_creates_log_directory_and_file(self):
        """Test that default setup_logging creates the logs/ directory and debug.log file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Call setup_logging with defaults
            setup_logging(output_dir)
            
            # Verify logs directory is created
            logs_dir = output_dir / "logs"
            self.assertTrue(logs_dir.exists(), "logs/ directory should be created")
            self.assertTrue(logs_dir.is_dir(), "logs/ should be a directory")
            
            # Verify debug.log file is created
            debug_log = logs_dir / "debug.log"
            self.assertTrue(debug_log.exists(), "debug.log file should be created")
            self.assertTrue(debug_log.is_file(), "debug.log should be a file")
            
            # Test logging works by writing a message
            logging.info("Test message")
            
            # Verify the log file has content
            with open(debug_log, 'r') as f:
                content = f.read()
                self.assertIn("Test message", content, "Log message should be written to debug.log")

    def test_setup_logging_clears_existing_handlers(self):
        """Test that setup_logging removes existing handlers before setting up new ones."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Add a dummy handler first
            dummy_handler = logging.StreamHandler()
            logging.getLogger().addHandler(dummy_handler)
            initial_handler_count = len(logging.getLogger().handlers)
            
            # Call setup_logging
            setup_logging(output_dir)
            
            # Verify handlers were replaced, not just added
            final_handler_count = len(logging.getLogger().handlers)
            self.assertGreater(initial_handler_count, 0, "Should have had initial handlers")
            self.assertEqual(final_handler_count, 2, "Should have exactly 2 handlers after setup (file + console)")

    def test_quiet_mode_disables_console(self):
        """Test that quiet=True creates no console handler."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Call setup_logging with quiet=True
            setup_logging(output_dir, quiet=True)
            
            # Should have only 1 handler (file)
            handlers = logging.getLogger().handlers
            self.assertEqual(len(handlers), 1, "Should have exactly 1 handler in quiet mode")
            self.assertIsInstance(handlers[0], logging.FileHandler, "Single handler should be FileHandler")
            
            # Test that messages still go to file
            logging.info("Quiet test message")
            debug_log = output_dir / "logs" / "debug.log"
            with open(debug_log, 'r') as f:
                content = f.read()
                self.assertIn("Quiet test message", content, "Log message should still be written to file")

    def test_verbose_mode_enables_debug_console(self):
        """Test that verbose=True sets console to DEBUG level."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Capture console output
            console_capture = StringIO()
            
            # Call setup_logging with verbose=True
            setup_logging(output_dir, verbose=True)
            
            # Replace stdout with our capture for testing
            handlers = logging.getLogger().handlers
            console_handler = None
            for handler in handlers:
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                    console_handler = handler
                    handler.stream = console_capture
                    break
            
            self.assertIsNotNone(console_handler, "Should have console handler in verbose mode")
            self.assertEqual(console_handler.level, logging.DEBUG, "Console should be at DEBUG level")
            
            # Test DEBUG message goes to console
            logging.debug("Debug test message")
            console_output = console_capture.getvalue()
            self.assertIn("Debug test message", console_output, "DEBUG message should appear on console")

    def test_custom_console_level(self):
        """Test custom console_level parameter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Call setup_logging with WARNING level
            setup_logging(output_dir, console_level=logging.WARNING)
            
            # Find console handler (StreamHandler that's not FileHandler)
            console_handler = None
            for handler in logging.getLogger().handlers:
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                    console_handler = handler
                    break
            
            self.assertIsNotNone(console_handler, "Should have console handler")
            self.assertEqual(console_handler.level, logging.WARNING, "Console should be at WARNING level")

    def test_string_path_support(self):
        """Test that string paths are converted to Path objects."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Pass string instead of Path
            setup_logging(temp_dir)
            
            # Verify logs directory is created
            logs_dir = Path(temp_dir) / "logs"
            self.assertTrue(logs_dir.exists(), "logs/ directory should be created from string path")

    def test_backward_compatibility(self):
        """Test that old setup_logging(output_dir) calls still work."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Call with old signature (positional argument only)
            setup_logging(output_dir)
            
            # Should have 2 handlers (file + console) with default levels
            handlers = logging.getLogger().handlers
            self.assertEqual(len(handlers), 2, "Should have 2 handlers for backward compatibility")
            
            # Check handler types and levels
            file_handler = console_handler = None
            for handler in handlers:
                if isinstance(handler, logging.FileHandler):
                    file_handler = handler
                elif isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                    console_handler = handler
            
            self.assertIsNotNone(file_handler, "Should have file handler")
            self.assertIsNotNone(console_handler, "Should have console handler")
            self.assertEqual(file_handler.level, logging.DEBUG, "File should be DEBUG level")
            self.assertEqual(console_handler.level, logging.INFO, "Console should be INFO level")

    def test_conflicting_flags_verbose_overrides(self):
        """Test that verbose=True overrides console_level parameter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Call with both verbose and custom console_level
            setup_logging(output_dir, console_level=logging.WARNING, verbose=True)
            
            # Find console handler
            console_handler = None
            for handler in logging.getLogger().handlers:
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                    console_handler = handler
                    break
            
            # verbose should override console_level
            self.assertEqual(console_handler.level, logging.DEBUG, "verbose=True should override console_level")

    def test_quiet_flag_overrides_console_level(self):
        """Test that quiet=True disables console regardless of console_level."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Call with both quiet and console_level
            setup_logging(output_dir, console_level=logging.INFO, quiet=True)
            
            # Should have only file handler
            handlers = logging.getLogger().handlers
            self.assertEqual(len(handlers), 1, "quiet=True should disable console regardless of console_level")


if __name__ == '__main__':
    unittest.main()