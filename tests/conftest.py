"""
Global pytest configuration for PtychoPINN tests.

This file handles optional dependency management and provides fixtures
for the test suite.
"""

import pytest
import warnings
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--run-oom",
        action="store_true",
        default=False,
        help="Run OOM tests that intentionally exhaust memory",
    )


def pytest_configure(config):
    """Configure pytest with custom settings."""

    # Register custom markers
    config.addinivalue_line("markers", "torch: mark test as requiring PyTorch")
    config.addinivalue_line("markers", "optional: mark test as requiring optional dependencies")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "oom: mark test as OOM-triggering (requires --run-oom)")

def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to handle optional dependencies.

    In torch-required environments (Ptychodus CI), PyTorch must be installed;
    tests in tests/torch/ will FAIL (not skip) if torch unavailable.

    In TF-only CI environments, tests/torch/ directory is skipped entirely.

    OOM tests require --run-oom flag to execute (they intentionally exhaust memory).
    """

    # Check if PyTorch is available
    torch_available = True
    try:
        import torch
    except ImportError:
        torch_available = False

    # Check if --run-oom option was provided
    run_oom = config.getoption("--run-oom", default=False)

    # Skip torch tests only in TF-only CI environments (directory-based)
    # Skip OOM tests unless --run-oom is provided
    for item in items:
        if "torch" in str(item.fspath).lower() or item.get_closest_marker("torch"):
            if not torch_available:
                # No whitelist exceptions: ALL torch tests skip in TF-only CI
                item.add_marker(pytest.mark.skip(reason="PyTorch not available (TF-only CI)"))

        # Handle OOM tests
        if item.get_closest_marker("oom") and not run_oom:
            item.add_marker(pytest.mark.skip(reason="Need --run-oom option to run OOM tests"))

def pytest_ignore_collect(path, config):
    """Avoid importing torch test modules when PyTorch is unavailable.

    This prevents import-time errors from modules that import torch at module scope
    (before pytest can mark/skip them). When torch is missing, ignore collection of
    files under tests/torch/ entirely.
    """
    p = str(path).replace("\\", "/").lower()
    if "/tests/torch/" in p:
        try:
            import torch  # noqa: F401
            return False
        except Exception:
            return True
    return False

def pytest_runtest_setup(item):
    """
    Setup hook that runs before each test.
    Can be used to skip tests based on markers or other conditions.
    """
    
    # Check for torch marker
    torch_marker = item.get_closest_marker("torch")
    if torch_marker is not None:
        try:
            import torch
        except ImportError as e:
            pytest.skip(f"PyTorch not available: {e}")
    
    # Check for optional marker
    optional_marker = item.get_closest_marker("optional")
    if optional_marker is not None:
        # Add any other optional dependency checks here
        pass

@pytest.fixture(scope="session", autouse=True)
def suppress_warnings():
    """Suppress common warnings during testing."""
    
    # Suppress specific warnings that are not relevant for testing
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    warnings.filterwarnings("ignore", message="PyTorch not available.*")
    
    yield
    
    # Reset warning filters after tests
    warnings.resetwarnings()


def pytest_report_header(config):
    """Add information to the test report header."""
    
    # Check optional dependencies
    torch_status = "available"
    try:
        import torch
        torch_version = torch.__version__
    except ImportError as e:
        torch_status = f"not available ({e})"
        torch_version = "N/A"
    
    return [
        f"PyTorch: {torch_status}",
        f"PyTorch version: {torch_version}" if torch_version != "N/A" else "",
    ]
