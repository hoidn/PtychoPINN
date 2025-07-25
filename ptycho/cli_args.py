"""
Shared command-line argument components for PtychoPINN scripts.

Provides reusable argument parser functions for consistent logging configuration.
Public interface: add_logging_arguments() and get_logging_config().

Example:
    add_logging_arguments(parser)
    setup_logging(output_dir, **get_logging_config(args))
"""

import argparse
import logging
from typing import Optional


def add_logging_arguments(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """
    Add standard logging-related arguments to any argument parser.
    
    This function adds a consistent set of logging options that can be used
    by any script to control console vs file output behavior.
    
    Args:
        parser: The argument parser to add logging arguments to
        
    Returns:
        The logging argument group for further customization if needed
        
    Added Arguments:
        --quiet: Disable console output (file logging only)
        --verbose: Enable DEBUG output to console  
        --console-level: Set specific console logging level
        
    Examples:
        parser = argparse.ArgumentParser(description="My script")
        add_logging_arguments(parser)
        args = parser.parse_args()
        
        # Use with setup_logging
        setup_logging(
            output_dir, 
            console_level=getattr(logging, args.console_level),
            quiet=args.quiet,
            verbose=args.verbose
        )
    """
    logging_group = parser.add_argument_group('logging options')
    
    # Mutual exclusion between quiet and verbose
    output_group = logging_group.add_mutually_exclusive_group()
    output_group.add_argument(
        '--quiet', 
        action='store_true',
        help='Disable console output (file logging only, automation-friendly)'
    )
    output_group.add_argument(
        '--verbose', 
        action='store_true', 
        help='Enable DEBUG output to console (for detailed troubleshooting)'
    )
    
    # Fine-grained console level control
    logging_group.add_argument(
        '--console-level', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set console logging level (default: INFO). Ignored if --quiet or --verbose is used.'
    )
    
    return logging_group


def get_logging_config(args: argparse.Namespace) -> dict:
    """
    Extract logging configuration from parsed arguments.
    
    This helper function converts argument namespace to a dictionary
    of kwargs suitable for passing to setup_logging().
    
    Args:
        args: Parsed arguments from argparse (must have logging args added)
        
    Returns:
        Dictionary of logging configuration parameters
        
    Example:
        args = parser.parse_args()
        logging_config = get_logging_config(args)
        setup_logging(output_dir, **logging_config)
    """
    console_level = getattr(logging, args.console_level)
    
    return {
        'console_level': console_level,
        'quiet': args.quiet,
        'verbose': args.verbose
    }