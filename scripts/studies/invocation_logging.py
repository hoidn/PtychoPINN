"""Study-facing compatibility exports for invocation artifact helpers."""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
from typing import Any, Dict

from ptycho.invocation_logging import (
    build_command_line,
    capture_runtime_provenance,
    get_git_commit,
    get_git_dirty,
    update_invocation_artifacts,
    write_invocation_artifacts,
)


def capture_neuralop_provenance() -> Dict[str, Any]:
    """Capture neuraloperator/neuralop/UNO API provenance for U-NO rows.

    Required by the lines128 U-NO design (each U-NO row must record neuraloperator
    package version, neuralop.__version__, and the UNO constructor signature).
    """
    payload: Dict[str, Any] = {
        "neuraloperator_package_version": None,
        "neuralop_module_version": None,
        "uno_signature": None,
    }
    try:
        payload["neuraloperator_package_version"] = importlib.metadata.version("neuraloperator")
    except Exception:
        pass
    try:
        neuralop = importlib.import_module("neuralop")
        payload["neuralop_module_version"] = getattr(neuralop, "__version__", None)
    except Exception:
        return payload
    try:
        uno_module = importlib.import_module("neuralop.models")
        uno_cls = getattr(uno_module, "UNO", None)
        if uno_cls is not None:
            payload["uno_signature"] = str(inspect.signature(uno_cls))
    except Exception:
        pass
    return payload
