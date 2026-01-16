"""
Centralized logging system with enhanced tee-style output and stdout capture.

Provides authoritative logging configuration for all PtychoPINN workflows with comprehensive
output capture that logs ALL stdout (including print statements) while maintaining flexible
console control for interactive development and automation-friendly operation.

Key Features:
- Complete stdout/stderr capture with automatic exception handling
- Tee-style logging: simultaneous console and file output with independent level control
- Flexible modes: --quiet (file-only), --verbose (debug), or default (info to console)
- Per-run organization: all logs stored in output_dir/logs/debug.log for full records

Usage:
    from ptycho.log_config import setup_logging
    
    setup_logging(Path("my_run"))              # Standard: INFO to console, DEBUG to file
    setup_logging(Path("my_run"), quiet=True)  # Automation: file-only logging
    setup_logging(Path("my_run"), verbose=True) # Debug: DEBUG to both console and file

This system ensures every workflow execution has complete logging records for troubleshooting
and reproducibility across all training, inference, and simulation workflows.
"""

import logging
from pathlib import Path
import sys
from typing import Union, TextIO
from io import StringIO


class TeeStream:
    """Stream that writes to multiple outputs simultaneously."""
    
    def __init__(self, *streams):
        self.streams = streams
    
    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
    
    def flush(self):
        for stream in self.streams:
            stream.flush()
    
    def isatty(self):
        # Check if any of the streams is a tty
        return any(hasattr(stream, 'isatty') and stream.isatty() for stream in self.streams)


class LoggerWriter:
    """Writer that sends output to a logger at a specific level."""
    
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.buffer = []
    
    def write(self, message):
        if message and message != '\n':
            # Buffer the message to handle multi-line output properly
            self.buffer.append(message.rstrip('\n'))
    
    def flush(self):
        if self.buffer:
            # Join all buffered content and log as a single message
            full_message = ''.join(self.buffer)
            if full_message.strip():  # Only log non-empty messages
                self.logger.log(self.level, full_message.strip())
            self.buffer = []
    
    def isatty(self):
        return False


def setup_logging(
    output_dir: Union[str, Path],
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    quiet: bool = False,
    verbose: bool = False
) -> logging.Logger:
    """
    Set up tee-style logging that writes to both console and file.
    
    This function provides flexible logging control for both interactive and
    automated workflows. All messages are always written to a file for complete
    records, while console output can be customized or disabled entirely.
    
    Args:
        output_dir: Directory for log files (str or Path)
        console_level: Minimum level for console output (default: INFO)
        file_level: Minimum level for file output (default: DEBUG)
        quiet: If True, disable console output entirely (for automation)
        verbose: If True, set console_level to DEBUG (for debugging)
        
    Returns:
        Configured logger instance
        
    Examples:
        # Default: INFO to console, DEBUG to file
        setup_logging(Path("my_run"))
        
        # Quiet mode: file only (automation-friendly)
        setup_logging(Path("my_run"), quiet=True)
        
        # Verbose mode: DEBUG to both (debugging)
        setup_logging(Path("my_run"), verbose=True)
        
        # Custom levels
        setup_logging(Path("my_run"), console_level=logging.WARN)
    """
    # Handle string paths
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    
    # Process convenience flags
    if verbose:
        console_level = logging.DEBUG
    if quiet:
        console_level = None  # Will skip console handler
    
    # Set up log directory
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Remove all existing handlers to start fresh
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set root logger to capture everything needed
    if console_level is not None:
        root_logger.setLevel(min(file_level, console_level))
    else:
        # Quiet mode: only need file level
        root_logger.setLevel(file_level)
    
    # Standard formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler - always present for complete records
    file_handler = logging.FileHandler(log_dir / 'debug.log', mode='w')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Set up stdout redirection to capture print statements
    # Store original stdout and stderr for restoration if needed
    if not hasattr(sys, '_original_stdout'):
        sys._original_stdout = sys.stdout
    if not hasattr(sys, '_original_stderr'):
        sys._original_stderr = sys.stderr
    
    # Console handler - optional based on quiet mode
    # IMPORTANT: Use original stdout to avoid circular logging
    if console_level is not None:
        console_handler = logging.StreamHandler(sys._original_stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Create logger writers to capture print statements and stderr
    stdout_logger_writer = LoggerWriter(root_logger, logging.INFO)
    stderr_logger_writer = LoggerWriter(root_logger, logging.ERROR)
    
    # Set up stdout redirection
    if console_level is not None:
        # Normal mode: tee stdout to both console and logger
        sys.stdout = TeeStream(sys._original_stdout, stdout_logger_writer)
    else:
        # Quiet mode: redirect stdout only to logger
        sys.stdout = TeeStream(stdout_logger_writer)
    
    # Set up stderr redirection
    if console_level is not None:
        # Normal mode: tee stderr to both console and logger
        sys.stderr = TeeStream(sys._original_stderr, stderr_logger_writer)
    else:
        # Quiet mode: redirect stderr only to logger
        sys.stderr = TeeStream(stderr_logger_writer)
    
    # Install custom exception hook to capture uncaught exceptions
    def exception_handler(exc_type, exc_value, exc_traceback):
        # Allow KeyboardInterrupt to be handled normally
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        # Log the uncaught exception with full traceback
        uncaught_logger = logging.getLogger('uncaught_exceptions')
        uncaught_logger.critical(
            "Uncaught exception", 
            exc_info=(exc_type, exc_value, exc_traceback)
        )
        
        # Call original handler to maintain normal console behavior
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    
    # Store original exception hook if not already stored
    if not hasattr(sys, '_original_excepthook'):
        sys._original_excepthook = sys.excepthook
    
    # Install our custom exception handler
    sys.excepthook = exception_handler
    
    return root_logger


def restore_logging():
    """
    Restore original stdout, stderr, and exception hook.
    
    This function completely undoes the logging system modifications,
    restoring the system to its original state. Typically called in
    test teardown or when logging system needs to be reset.
    """
    if hasattr(sys, '_original_stdout'):
        sys.stdout = sys._original_stdout
    if hasattr(sys, '_original_stderr'):
        sys.stderr = sys._original_stderr
    if hasattr(sys, '_original_excepthook'):
        sys.excepthook = sys._original_excepthook


def restore_stdout():
    """Restore original stdout, typically called in test teardown."""
    if hasattr(sys, '_original_stdout'):
        sys.stdout = sys._original_stdout