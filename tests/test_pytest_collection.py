from pathlib import Path

from tests import conftest
import pytest


def test_ignore_hook_defers_non_torch_paths_to_other_plugins():
    assert conftest.pytest_ignore_collect(
        Path("/repo/tests/studies/test_ignored.py"), None
    ) is None


def test_ignore_hook_defers_torch_paths_when_torch_is_available():
    pytest.importorskip("torch")
    assert conftest.pytest_ignore_collect(
        Path("/repo/tests/torch/test_ignored.py"), None
    ) is None
