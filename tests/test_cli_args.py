"""
Unit tests for the ptycho.cli_args module.
"""

import unittest
import argparse
import logging
from ptycho.cli_args import add_logging_arguments, get_logging_config


class TestCliArgs(unittest.TestCase):
    """Test cases for CLI argument handling."""

    def test_add_logging_arguments(self):
        """Test that logging arguments are added correctly to parser."""
        parser = argparse.ArgumentParser()
        add_logging_arguments(parser)
        
        # Parse with default values
        args = parser.parse_args([])
        
        self.assertFalse(args.quiet, "quiet should default to False")
        self.assertFalse(args.verbose, "verbose should default to False")  
        self.assertEqual(args.console_level, 'INFO', "console_level should default to INFO")

    def test_quiet_verbose_mutually_exclusive(self):
        """Test that quiet and verbose are mutually exclusive."""
        parser = argparse.ArgumentParser()
        add_logging_arguments(parser)
        
        # Should work individually
        args_quiet = parser.parse_args(['--quiet'])
        self.assertTrue(args_quiet.quiet)
        
        args_verbose = parser.parse_args(['--verbose'])
        self.assertTrue(args_verbose.verbose)
        
        # Should fail when both specified
        with self.assertRaises(SystemExit):
            parser.parse_args(['--quiet', '--verbose'])

    def test_console_level_choices(self):
        """Test that console-level accepts valid choices."""
        parser = argparse.ArgumentParser()
        add_logging_arguments(parser)
        
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        for level in valid_levels:
            args = parser.parse_args(['--console-level', level])
            self.assertEqual(args.console_level, level)
        
        # Should fail with invalid level
        with self.assertRaises(SystemExit):
            parser.parse_args(['--console-level', 'INVALID'])

    def test_get_logging_config_defaults(self):
        """Test get_logging_config with default arguments."""
        parser = argparse.ArgumentParser()
        add_logging_arguments(parser)
        args = parser.parse_args([])
        
        config = get_logging_config(args)
        
        expected = {
            'console_level': logging.INFO,
            'quiet': False,
            'verbose': False
        }
        
        self.assertEqual(config, expected)

    def test_get_logging_config_quiet(self):
        """Test get_logging_config with quiet flag."""
        parser = argparse.ArgumentParser()
        add_logging_arguments(parser)
        args = parser.parse_args(['--quiet'])
        
        config = get_logging_config(args)
        
        expected = {
            'console_level': logging.INFO,  # Still INFO, but quiet overrides
            'quiet': True,
            'verbose': False
        }
        
        self.assertEqual(config, expected)

    def test_get_logging_config_verbose(self):
        """Test get_logging_config with verbose flag."""
        parser = argparse.ArgumentParser()
        add_logging_arguments(parser)
        args = parser.parse_args(['--verbose'])
        
        config = get_logging_config(args)
        
        expected = {
            'console_level': logging.INFO,  # Still INFO, but verbose overrides
            'quiet': False,
            'verbose': True
        }
        
        self.assertEqual(config, expected)

    def test_get_logging_config_custom_level(self):
        """Test get_logging_config with custom console level."""
        parser = argparse.ArgumentParser()
        add_logging_arguments(parser)
        args = parser.parse_args(['--console-level', 'WARNING'])
        
        config = get_logging_config(args)
        
        expected = {
            'console_level': logging.WARNING,
            'quiet': False,
            'verbose': False
        }
        
        self.assertEqual(config, expected)


if __name__ == '__main__':
    unittest.main()