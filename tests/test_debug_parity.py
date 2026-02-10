import os
import logging

import numpy as np


def test_parity_debug_enabled_env(monkeypatch):
    from ptycho import debug_parity

    monkeypatch.delenv("PTYCHO_DEBUG_PARITY", raising=False)
    assert debug_parity.parity_debug_enabled() is False

    monkeypatch.setenv("PTYCHO_DEBUG_PARITY", "1")
    assert debug_parity.parity_debug_enabled() is True

    monkeypatch.setenv("PTYCHO_DEBUG_PARITY", "true")
    assert debug_parity.parity_debug_enabled() is True

    monkeypatch.setenv("PTYCHO_DEBUG_PARITY", "no")
    assert debug_parity.parity_debug_enabled() is False


def test_summarize_array_basic():
    from ptycho import debug_parity

    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    stats = debug_parity.summarize_array(arr)

    assert stats["shape"] == (2, 2)
    assert stats["dtype"] == "float32"
    assert stats["min"] == 1.0
    assert stats["max"] == 4.0
    assert stats["mean"] == 2.5


def test_summarize_offsets_unique():
    from ptycho import debug_parity

    offsets = np.array([[0.0, 1.0], [0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    stats = debug_parity.summarize_offsets(offsets)

    assert stats["unique_count"] == 2
    assert stats["shape"] == (3, 2)


def test_log_array_stats_prints_when_no_handlers(monkeypatch, capsys):
    from ptycho import debug_parity

    monkeypatch.setenv("PTYCHO_DEBUG_PARITY", "1")

    logger = logging.getLogger("ptycho.debug_parity")
    root = logging.getLogger()
    saved_logger_handlers = list(logger.handlers)
    saved_root_handlers = list(root.handlers)
    logger.handlers = []
    root.handlers = []
    try:
        debug_parity.log_array_stats("sample", np.array([1.0, 2.0], dtype=np.float32))
        captured = capsys.readouterr().out
    finally:
        logger.handlers = saved_logger_handlers
        root.handlers = saved_root_handlers

    assert "[PARITY]" in captured


def test_log_array_stats_prints_when_root_level_filters(monkeypatch, capsys):
    from ptycho import debug_parity

    monkeypatch.setenv("PTYCHO_DEBUG_PARITY", "1")

    root = logging.getLogger()
    handler = logging.StreamHandler()
    saved_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.WARNING)
    try:
        debug_parity.log_array_stats("sample", np.array([1.0, 2.0], dtype=np.float32))
        captured = capsys.readouterr().out
    finally:
        root.removeHandler(handler)
        root.setLevel(saved_level)

    assert "[PARITY]" in captured
