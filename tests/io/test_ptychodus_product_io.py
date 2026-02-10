import os
from pathlib import Path

import numpy as np
import pytest

# Import exporter/importer without bringing in tensorflow by avoiding RawData import
from ptycho.io.ptychodus_product_io import (
    ExportMeta,
    export_product_from_rawdata,
    import_product_to_rawdata,
)


def _make_synthetic_rawdata(n_points: int = 4, N_probe: int = 8, N_obj: int = 16):
    class _Stub:
        pass

    stub = _Stub()
    stub.xcoords = np.arange(n_points, dtype=float)
    stub.ycoords = np.arange(n_points, dtype=float) * 2.0
    stub.xcoords_start = stub.xcoords
    stub.ycoords_start = stub.ycoords
    stub.scan_index = np.zeros(n_points, dtype=int)
    stub.diff3d = None
    stub.probeGuess = (np.ones((N_probe, N_probe), dtype=np.complex64) + 0j).astype(np.complex64)
    stub.objectGuess = (np.ones((N_obj, N_obj), dtype=np.complex64) + 0j).astype(np.complex64)
    return stub


def test_export_writes_minimal_hdf5(tmp_path: Path):
    raw = _make_synthetic_rawdata()
    out_path = tmp_path / "product.h5"

    meta = ExportMeta(
        name="Example",
        comments="Unit test export",
        detector_distance_m=0.75,
        probe_energy_eV=8000.0,
        exposure_time_s=0.1,
        probe_photon_count=0.0,
        mass_attenuation_m2_kg=0.0,
        tomography_angle_deg=0.0,
        object_pixel_width_m=5.0e-8,
        object_pixel_height_m=5.0e-8,
        probe_pixel_width_m=1.25e-7,
        probe_pixel_height_m=1.25e-7,
        object_center_x_m=0.0,
        object_center_y_m=0.0,
    )

    # RED: expect NotImplemented until exporter is implemented
    export_product_from_rawdata(raw, out_path, meta)

    # On GREEN: validate HDF5 contents
    # import h5py
    # with h5py.File(out_path, 'r') as f:
    #     assert f.attrs['name'] == 'Example'
    #     assert 'probe' in f and 'object' in f
    #     assert 'probe_position_indexes' in f
    #     assert 'probe_position_x_m' in f and 'probe_position_y_m' in f
    #     # Ensure loss_values exists (may be empty)
    #     assert 'loss_values' in f


def test_import_reads_hdf5_to_rawdata(tmp_path: Path):
    # Create a minimal valid HDF5 file per spec
    p = tmp_path / "input_product.h5"
    import h5py

    with h5py.File(p, 'w') as f:
        # Root attrs
        f.attrs['name'] = 'Example'
        f.attrs['comments'] = 'Unit test import'
        f.attrs['detector_object_distance_m'] = 0.75
        f.attrs['probe_energy_eV'] = 8000.0
        f.attrs['probe_photon_count'] = 0.0
        f.attrs['exposure_time_s'] = 0.1
        f.attrs['mass_attenuation_m2_kg'] = 0.0
        f.attrs['tomography_angle_deg'] = 0.0

        # Positions
        f.create_dataset('probe_position_indexes', data=np.zeros(4, dtype=np.int32))
        f.create_dataset('probe_position_x_m', data=np.array([0.0, 5e-8, 1e-7, 1.5e-7]))
        f.create_dataset('probe_position_y_m', data=np.array([0.0, 5e-8, 1e-7, 1.5e-7]))

        # Probe
        probe = (np.ones((8, 8), dtype=np.complex64) + 0j).astype(np.complex64)
        d_probe = f.create_dataset('probe', data=probe)
        d_probe.attrs['pixel_width_m'] = 1.25e-7
        d_probe.attrs['pixel_height_m'] = 1.25e-7

        # Object
        obj = (np.ones((16, 16), dtype=np.complex64) + 0j).astype(np.complex64)
        d_obj = f.create_dataset('object', data=obj)
        d_obj.attrs['center_x_m'] = 0.0
        d_obj.attrs['center_y_m'] = 0.0
        d_obj.attrs['pixel_width_m'] = 5.0e-8
        d_obj.attrs['pixel_height_m'] = 5.0e-8
        f.create_dataset('object_layer_spacing_m', data=np.array([], dtype=np.float64))

        # Losses - empty arrays allowed
        f.create_dataset('loss_values', data=np.array([], dtype=np.float64))
        f.create_dataset('loss_epochs', data=np.array([], dtype=np.int32))

    # RED: expect NotImplemented until importer is implemented
    raw = import_product_to_rawdata(p)

    # On GREEN: validate RawData basics
    # assert isinstance(raw, RawData)
    # assert raw.probeGuess is not None and raw.objectGuess is not None
    # assert raw.xcoords.shape[0] == 4 and raw.ycoords.shape[0] == 4


@pytest.mark.slow
def test_cli_convert_run1084_smoke(tmp_path: Path):
    # Smoke: convert the provided Run1084 dataset (if present) to HDF5 product
    in_npz = Path('datasets/Run1084_recon3_postPC_shrunk_3.npz')
    if not in_npz.exists():
        pytest.skip("Run1084 dataset not present")

    out_product = tmp_path / "run1084_product.h5"
    cmd = (
        f"python scripts/tools/convert_to_ptychodus_product.py "
        f"--input-npz {in_npz} --output-product {out_product} "
        f"--name Run1084 --comments smoke --object-pixel-size-m 5e-8 --probe-pixel-size-m 1.25e-7"
    )
    rc = os.system(cmd)

    # RED: exporter not implemented yet; expect non-zero return
    assert rc == 0

    # On GREEN: verify file exists and has basic datasets
    # import h5py
    # with h5py.File(out_product, 'r') as f:
    #     assert 'probe' in f and 'object' in f
    #     assert 'probe_position_indexes' in f
