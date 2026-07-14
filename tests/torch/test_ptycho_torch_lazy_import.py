from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


EXPECTED_EXPORTS = {
    "MemmapDatasetBridge",
    "PtychoDataContainerTorch",
    "RawDataTorch",
    "TORCH_AVAILABLE",
    "load_inference_bundle_torch",
    "run_cdi_example_torch",
    "to_inference_config",
    "to_model_config",
    "to_training_config",
    "train_cdi_model_torch",
}


def test_package_public_exports_remain_lazy_and_discoverable() -> None:
    import ptycho_torch

    assert set(ptycho_torch.__all__) == EXPECTED_EXPORTS
    assert EXPECTED_EXPORTS <= set(dir(ptycho_torch))
    assert ptycho_torch.TORCH_AVAILABLE is True
    assert "torch" in sys.modules


def test_cold_package_import_loads_torch_but_keeps_other_exports_lazy() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = r"""
import json
import sys

import ptycho_torch

lazy_modules = {
    "ptycho_torch.config_bridge",
    "ptycho_torch.data_container_bridge",
    "ptycho_torch.memmap_bridge",
    "ptycho_torch.raw_data_bridge",
    "ptycho_torch.workflows.components",
}
print(json.dumps({
    "lazy_loaded": sorted(lazy_modules.intersection(sys.modules)),
    "torch_available": ptycho_torch.TORCH_AVAILABLE,
    "torch_loaded": "torch" in sys.modules,
}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result == {
        "lazy_loaded": [],
        "torch_available": True,
        "torch_loaded": True,
    }


def test_import_failing_torch_raises_actionable_runtime_error() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = r"""
import builtins
import importlib.util

assert importlib.util.find_spec("torch") is not None
real_import = builtins.__import__

def failing_import(name, *args, **kwargs):
    if name == "torch":
        raise ImportError("simulated binary import failure")
    return real_import(name, *args, **kwargs)

builtins.__import__ = failing_import
try:
    import ptycho_torch
except RuntimeError as exc:
    print(str(exc))
else:
    raise AssertionError("ptycho_torch import unexpectedly succeeded")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "PyTorch is required for ptycho_torch package" in completed.stdout
    assert "pip install torch>=2.2" in completed.stdout


def test_concurrent_named_export_resolution_is_stable() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = r"""
from concurrent.futures import ThreadPoolExecutor

import ptycho_torch

def load_export(_index):
    return ptycho_torch.to_model_config

with ThreadPoolExecutor(max_workers=8) as pool:
    exports = list(pool.map(load_export, range(32)))
assert all(item is exports[0] for item in exports)
assert exports[0].__module__ == "ptycho_torch.config_bridge"
print("stable")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "stable"


def test_from_import_preserves_named_exports_and_submodules() -> None:
    from ptycho_torch import config_bridge, helper, reassembly, to_model_config

    assert config_bridge.to_model_config is to_model_config
    assert helper.__name__ == "ptycho_torch.helper"
    assert reassembly.__name__ == "ptycho_torch.reassembly"
