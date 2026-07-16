"""Structural contract for the deterministic probe-extension visual check."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np


def test_renderer_writes_large_png_and_complete_matching_sidecar(tmp_path: Path):
    from scripts.simulation.render_probe_extension_check import (
        render_probe_extension_check,
    )

    yy, xx = np.indices((8, 8))
    probe = (
        (1.0 + 0.05 * xx + 0.03 * yy)
        * np.exp(1j * (0.02 * (xx**2 + yy**2) + 0.15 * np.sin(xx)))
    ).astype(np.complex64)
    source = tmp_path / "probe.npz"
    np.savez(source, probeGuess=probe)
    output = tmp_path / "check.png"

    returned = render_probe_extension_check(
        source_probe=source,
        target_size=16,
        smoothing=0.5,
        output=output,
    )

    sidecar_path = output.with_suffix(".json")
    assert output.is_file()
    assert sidecar_path.is_file()
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert returned == sidecar
    assert sidecar["schema_version"] == "probe_extension_visual_check_v1"
    assert sidecar["canonical_pipeline"] == (
        "smooth:0.5|pad_extrapolate_boundary_matched:16"
    )
    assert sidecar["source_shape"] == [8, 8]
    assert sidecar["target_shape"] == [16, 16]
    assert sidecar["source_rows"] == [4, 12]
    assert sidecar["source_columns"] == [4, 12]
    assert sidecar["center_difference_max"] == 0.0
    assert sidecar["seam_error_max"] <= 2e-8
    assert 0.0 < sidecar["adjacent_phase_step_max"] <= np.pi
    assert sidecar["laplacian_residual"] <= sidecar["solver_tolerance"]
    assert sidecar["inner_boundary_overlay"] is True
    assert sidecar["outer_boundary_overlay"] is True
    assert sidecar["boundary_method"] == "harmonic_dirichlet_c0"
    assert sidecar["solver"] == "scipy.sparse.linalg.spsolve"
    for field in (
        "source_file_sha256",
        "source_probe_sha256",
        "prepared_probe_sha256",
        "legacy_probe_sha256",
        "boundary_matched_probe_sha256",
    ):
        assert len(sidecar[field]) == 64
    assert {
        "prepared amplitude",
        "prepared wrapped phase",
        "legacy global amplitude",
        "legacy global wrapped phase",
        "boundary-matched amplitude",
        "boundary-matched wrapped phase",
        "center absolute difference",
        "across-seam wrapped phase step",
        "outer phase residual",
        "horizontal unwrapped phase profile",
        "vertical unwrapped phase profile",
        "identity and solver annotation",
    } == set(sidecar["panel_labels"])

    image = mpimg.imread(output)
    assert image.shape[1] >= 2400
    assert image.shape[0] >= 1800
    assert sidecar["png_width_pixels"] == image.shape[1]
    assert sidecar["png_height_pixels"] == image.shape[0]
