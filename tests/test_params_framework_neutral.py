"""Focused contract tests for the legacy parameter foundation."""

import subprocess
import sys

import numpy as np

from ptycho import params


def test_params_import_and_dictionary_api_do_not_require_tensorflow():
    code = r"""
import builtins
import sys

real_import = builtins.__import__

def reject_tensorflow(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "tensorflow" or name.startswith("tensorflow."):
        raise AssertionError(f"ptycho.params requested TensorFlow import: {name}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = reject_tensorflow

from ptycho import params

cfg_identity = id(params.cfg)
params.unseal()
params.set("N", 64)
assert params.get("N") == 64
params.cfg.update({"gridsize": 1})
assert params.get("gridsize") == 1
assert id(params.cfg) == cfg_identity
assert not any(
    name == "tensorflow" or name.startswith("tensorflow.")
    for name in sys.modules
)
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_print_params_keeps_array_diagnostics_without_tensorflow(capsys):
    previous = dict(params.cfg)
    cfg_identity = id(params.cfg)
    try:
        params.cfg.clear()
        params.cfg.update(params.DEFAULT_CFG)
        params.cfg["probe"] = np.array([1.0, 3.0])

        params.print_params()

        output = capsys.readouterr().out
        assert "probe:\n" in output
        assert "  shape: (2,)\n" in output
        assert "  mean: 2.000\n" in output
    finally:
        params.cfg.clear()
        params.cfg.update(previous)

    assert id(params.cfg) == cfg_identity
