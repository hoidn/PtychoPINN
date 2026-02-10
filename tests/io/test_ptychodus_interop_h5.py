import sys
from pathlib import Path

import numpy as np
import pytest

from ptycho.io.ptychodus_product_io import ExportMeta, export_product_from_rawdata


def _stub_raw(n_points: int = 5, N_probe: int = 8, N_obj: int = 16):
    class _Stub:
        pass

    s = _Stub()
    s.xcoords = np.arange(n_points, dtype=float)
    s.ycoords = np.arange(n_points, dtype=float) * 2.0
    s.xcoords_start = s.xcoords
    s.ycoords_start = s.ycoords
    s.scan_index = np.arange(n_points, dtype=int) % 2
    s.diff3d = None
    s.probeGuess = (np.ones((N_probe, N_probe), dtype=np.complex64) + 0j).astype(np.complex64)
    s.objectGuess = (np.ones((N_obj, N_obj), dtype=np.complex64) + 0j).astype(np.complex64)
    return s


def test_interop_h5_reader(tmp_path: Path):
    # Export a minimal product file using our exporter
    raw = _stub_raw()
    meta = ExportMeta(
        name="InteropTest",
        comments="interop smoke",
        detector_distance_m=0.5,
        probe_energy_eV=9000.0,
        exposure_time_s=0.2,
        object_pixel_width_m=5.0e-8,
        object_pixel_height_m=5.0e-8,
        probe_pixel_width_m=1.25e-7,
        probe_pixel_height_m=1.25e-7,
        object_center_x_m=0.0,
        object_center_y_m=0.0,
    )
    out = tmp_path / "interop.h5"
    export_product_from_rawdata(raw, out, meta)

    # Ensure ptychodus/src is importable so we can use its HDF5 reader plugin
    proj_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(proj_root / "ptychodus" / "src"))

    # Skip gracefully when the optional ptychodus dependency is unavailable.
    pytest.importorskip(
        "ptychodus",
        reason="ptychodus plugins not available; see specs/ptychodus_api_spec.md",
    )

    from ptychodus.plugins.h5_product_file import H5ProductFileIO

    product = H5ProductFileIO().read(out)

    # Metadata
    assert product.metadata.name == meta.name
    assert product.metadata.probe_energy_eV == pytest.approx(meta.probe_energy_eV)

    # Positions count
    assert len(product.probe_positions) == len(raw.xcoords)

    # Probe geometry
    pg = product.probes.get_pixel_geometry()
    assert pg.width_m == pytest.approx(meta.probe_pixel_width_m)
    assert pg.height_m == pytest.approx(meta.probe_pixel_height_m)

    # Object geometry
    og = product.object_.get_geometry()
    assert og.pixel_width_m == pytest.approx(meta.object_pixel_width_m)
    assert og.pixel_height_m == pytest.approx(meta.object_pixel_height_m)
    assert og.center_x_m == pytest.approx(meta.object_center_x_m)
    assert og.center_y_m == pytest.approx(meta.object_center_y_m)

    # Losses are empty in current exporter
    assert len(product.losses) == 0
