#!/usr/bin/env python3
"""Convert a NPZ dataset to a Ptychodus HDF5 product (scaffold).

This CLI parses inputs and prepares for conversion via
`ptycho.io.ptychodus_product_io.export_product_from_rawdata`. The exporter is
currently a scaffold and will be implemented in a follow-up step.

Usage
-----
python scripts/tools/convert_to_ptychodus_product.py \
  --input-npz datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --output-product outputs/ptychodus_products/run1084_product.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from pathlib import Path as _P
import sys as _sys

# Ensure project root is on sys.path so 'ptycho' can be imported when running from scripts/tools
_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))

from ptycho.io.ptychodus_product_io import ExportMeta, export_product_from_rawdata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert NPZ dataset to Ptychodus HDF5 product (scaffold)",
    )
    parser.add_argument(
        "--input-npz",
        required=True,
        type=Path,
        help="Path to input NPZ file (e.g., Run1084_recon3_postPC_shrunk_3.npz)",
    )
    parser.add_argument(
        "--output-product",
        required=True,
        type=Path,
        help="Path to output HDF5 product (*.h5|*.hdf5)",
    )

    # Optional metadata (scaffold defaults are applied when omitted)
    parser.add_argument("--name", default="", help="Product name")
    parser.add_argument("--comments", default="", help="Product comments")
    parser.add_argument("--detector-distance-m", type=float, default=0.0)
    parser.add_argument("--probe-energy-eV", type=float, default=0.0)
    parser.add_argument("--exposure-time-s", type=float, default=0.0)
    parser.add_argument("--probe-photon-count", type=float, default=0.0)
    parser.add_argument("--mass-attenuation-m2-kg", type=float, default=0.0)
    parser.add_argument("--tomography-angle-deg", type=float, default=0.0)

    parser.add_argument("--object-pixel-size-m", type=float, default=5.0e-8)
    parser.add_argument("--probe-pixel-size-m", type=float, default=1.25e-7)
    parser.add_argument("--object-center-x-m", type=float, default=0.0)
    parser.add_argument("--object-center-y-m", type=float, default=0.0)

    # Bundle raw data by default; allow disabling with --no-include-diffraction
    parser.add_argument(
        "--include-diffraction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include raw diffraction + coords under /raw_data (default: true)",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Load NPZ directly to avoid tensorflow dependency in RawData
    with np.load(str(args.input_npz), allow_pickle=True) as data:
        class _RawStub:
            pass

        raw = _RawStub()
        raw.xcoords = data["xcoords"]
        raw.ycoords = data["ycoords"]
        raw.xcoords_start = data.get("xcoords_start", raw.xcoords)
        raw.ycoords_start = data.get("ycoords_start", raw.ycoords)
        raw.probeGuess = data["probeGuess"]
        raw.objectGuess = data.get("objectGuess", np.ones((16, 16), dtype=np.complex64))
        raw.scan_index = data.get("scan_index", np.zeros_like(raw.xcoords, dtype=int))
        # Diffraction key may be 'diff3d' or 'diffraction'
        raw.diff3d = data.get("diff3d")
        if raw.diff3d is None and "diffraction" in data:
            raw.diff3d = data["diffraction"]

    # Build metadata (scaffold defaults fill in missing fields)
    meta = ExportMeta(
        name=args.name,
        comments=args.comments,
        detector_distance_m=args.detector_distance_m,
        probe_energy_eV=args.probe_energy_eV,
        exposure_time_s=args.exposure_time_s,
        probe_photon_count=args.probe_photon_count,
        mass_attenuation_m2_kg=args.mass_attenuation_m2_kg,
        tomography_angle_deg=args.tomography_angle_deg,
        object_pixel_width_m=args.object_pixel_size_m,
        object_pixel_height_m=args.object_pixel_size_m,
        probe_pixel_width_m=args.probe_pixel_size_m,
        probe_pixel_height_m=args.probe_pixel_size_m,
        object_center_x_m=args.object_center_x_m,
        object_center_y_m=args.object_center_y_m,
    )

    try:
        export_product_from_rawdata(
            raw,
            args.output_product,
            meta,
            include_raw=bool(args.include_diffraction),
        )
    except NotImplementedError as e:
        print(f"Exporter not yet implemented: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Conversion failed: {e}", file=sys.stderr)
        return 3

    print(f"Wrote Ptychodus product: {args.output_product}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
