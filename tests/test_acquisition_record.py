"""Focused contracts for the framework-neutral acquisition handoff."""

from __future__ import annotations

import subprocess
import sys

import numpy as np


def test_acquisition_record_imports_without_tensorflow_or_torch():
    code = r"""
import builtins
import sys

real_import = builtins.__import__

def reject_frameworks(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "tensorflow" or name.startswith("tensorflow."):
        raise AssertionError(f"acquisition record imported TensorFlow: {name}")
    if name == "torch" or name.startswith("torch."):
        raise AssertionError(f"acquisition record imported Torch: {name}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = reject_frameworks

from ptycho.acquisition import AcquisitionRecord

assert AcquisitionRecord.__module__ == "ptycho.acquisition"
assert not any(
    name == "tensorflow" or name.startswith("tensorflow.")
    or name == "torch" or name.startswith("torch.")
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


def test_raw_data_torch_round_trip_preserves_selected_acquisition_boundary(monkeypatch):
    from ptycho import params
    from ptycho.acquisition import AcquisitionRecord
    from ptycho.config.config import ModelConfig, TrainingConfig
    from ptycho.raw_data import RawData
    from ptycho_torch.raw_data_bridge import RawDataTorch
    from ptycho_torch.workflows import components

    previous = dict(params.cfg)
    try:
        n_points = 9
        patch_size = 4
        xcoords = np.arange(n_points, dtype=np.float64)
        ycoords = np.arange(n_points, dtype=np.float64) * 2
        xcoords_start = xcoords + 0.25
        ycoords_start = ycoords - 0.5
        diffraction = np.arange(
            n_points * patch_size * patch_size, dtype=np.float32
        ).reshape(n_points, patch_size, patch_size)
        probe = np.ones((patch_size, patch_size), dtype=np.complex64) * (1 + 2j)
        scan_index = np.arange(n_points, dtype=np.int32)
        object_guess = np.ones((32, 32), dtype=np.complex64) * (2 - 1j)
        target = np.arange(
            n_points * patch_size * patch_size, dtype=np.float32
        ).reshape(n_points, patch_size, patch_size).astype(np.complex64)
        normalization = np.float32(3.5)
        metadata = {"source": "slice-4a"}
        sample_indices = np.array([1, 3, 5], dtype=np.int64)

        raw = RawData(
            xcoords,
            ycoords,
            xcoords_start,
            ycoords_start,
            diffraction,
            probe,
            scan_index,
            objectGuess=object_guess,
            Y=target,
            norm_Y_I=normalization,
            metadata=metadata,
        )
        raw.sample_indices = sample_indices
        raw.subsample_seed = 17

        record = AcquisitionRecord.from_raw_data(raw)
        config = TrainingConfig(
            model=ModelConfig(N=patch_size, gridsize=1),
            n_groups=3,
            neighbor_count=1,
            nphotons=1e6,
        )
        adapter = RawDataTorch.from_acquisition(record, config=config)
        restored = adapter.to_acquisition()

        for field in (
            "xcoords",
            "ycoords",
            "xcoords_start",
            "ycoords_start",
            "diff3d",
            "probeGuess",
            "scan_index",
            "objectGuess",
            "Y",
            "sample_indices",
        ):
            np.testing.assert_array_equal(getattr(restored, field), getattr(record, field))
        assert restored.norm_Y_I == record.norm_Y_I
        assert restored.metadata is metadata
        assert restored.subsample_seed == 17

        expected = raw.generate_grouped_data(
            N=patch_size,
            K=1,
            nsamples=3,
            sequential_sampling=True,
            gridsize=1,
        )
        actual = adapter.generate_grouped_data(
            N=patch_size,
            K=1,
            nsamples=3,
            sequential_sampling=True,
            gridsize=1,
        )

        assert actual.keys() == expected.keys()
        for key in expected:
            if expected[key] is None:
                assert actual[key] is None
            else:
                np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=0)

        production_handoff = {}
        from_acquisition = RawDataTorch.from_acquisition

        def capture_handoff(handoff_record, config=None):
            production_handoff["record"] = handoff_record
            return from_acquisition(handoff_record, config=config)

        monkeypatch.setattr(
            RawDataTorch,
            "from_acquisition",
            staticmethod(capture_handoff),
        )
        container = components._ensure_container(raw, config)

        handoff_record = production_handoff["record"]
        assert isinstance(handoff_record, AcquisitionRecord)
        np.testing.assert_array_equal(handoff_record.xcoords_start, xcoords_start)
        np.testing.assert_array_equal(handoff_record.Y, target)
        assert handoff_record.metadata is metadata
        assert container.metadata is metadata
        container_indices = container.nn_indices.cpu().numpy()
        expected_container_y = np.transpose(
            target[container_indices],
            (0, 2, 3, 1),
        )
        np.testing.assert_array_equal(
            container.Y.cpu().numpy(),
            expected_container_y,
        )
    finally:
        params.cfg.clear()
        params.cfg.update(previous)
