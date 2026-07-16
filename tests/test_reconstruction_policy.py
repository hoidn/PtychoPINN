"""Focused contracts for reconstruction-policy identity and composition."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import subprocess
import sys

import pytest


def test_reconstruction_policy_import_loads_neither_framework():
    code = r"""
import builtins
import sys

real_import = builtins.__import__

def reject_frameworks(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "tensorflow" or name.startswith("tensorflow."):
        raise AssertionError(f"reconstruction policy imported TensorFlow: {name}")
    if name == "torch" or name.startswith("torch."):
        raise AssertionError(f"reconstruction policy imported Torch: {name}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = reject_frameworks

from ptycho.reconstruction_policy import ReconstructionPolicy

assert ReconstructionPolicy.__module__ == "ptycho.reconstruction_policy"
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


@pytest.mark.parametrize(
    (
        "patch_weighting",
        "varpro_scaling",
        "algorithm",
        "window",
        "calibration",
        "route",
    ),
    [
        (
            "uniform",
            False,
            "legacy_position_v1",
            "legacy_stitch_crop",
            "identity_v1",
            "uniform",
        ),
        (
            "probe",
            False,
            "barycentric_v1",
            "middle_trim",
            "identity_v1",
            "barycentric",
        ),
        (
            "uniform",
            True,
            "barycentric_v1",
            "middle_trim",
            "varpro_s1s2_v1",
            "barycentric",
        ),
        (
            "probe",
            True,
            "barycentric_v1",
            "middle_trim",
            "varpro_s1s2_v1",
            "barycentric",
        ),
    ],
)
def test_cli_policy_mapping(
    patch_weighting,
    varpro_scaling,
    algorithm,
    window,
    calibration,
    route,
):
    from ptycho.reconstruction_policy import resolve_cli_reconstruction_policy

    policy = resolve_cli_reconstruction_policy(patch_weighting, varpro_scaling)

    assert policy.assembly.algorithm == algorithm
    assert policy.assembly.weighting == patch_weighting
    assert policy.assembly.window == window
    assert policy.calibration.method == calibration
    assert policy.output.representation == "amplitude_phase"
    assert policy.output.final_crop == "none"
    assert policy.compatibility_route == route
    assert policy.identity == "/".join(
        (
            "reconstruction_policy_v1",
            algorithm,
            patch_weighting,
            window,
            calibration,
            "amplitude_phase",
            "none",
            "preserve_calibrated_canvas",
            "none",
        )
    )


def test_unknown_patch_weighting_fails_closed():
    from ptycho.reconstruction_policy import resolve_cli_reconstruction_policy

    with pytest.raises(ValueError, match="patch_weighting"):
        resolve_cli_reconstruction_policy("central_mask", False)


def test_legacy_position_rejects_probe_weighting():
    from ptycho.reconstruction_policy import AssemblySpec

    with pytest.raises(ValueError, match="weighting"):
        AssemblySpec(
            algorithm="legacy_position_v1",
            weighting="probe",
            window="legacy_stitch_crop",
        )


def test_legacy_position_rejects_middle_trim_window():
    from ptycho.reconstruction_policy import AssemblySpec

    with pytest.raises(ValueError, match="window"):
        AssemblySpec(
            algorithm="legacy_position_v1",
            weighting="uniform",
            window="middle_trim",
        )


def test_varpro_policy_rejects_legacy_position_assembly():
    from ptycho.reconstruction_policy import (
        AssemblySpec,
        CalibrationSpec,
        ReconstructionPolicy,
    )

    with pytest.raises(ValueError, match="VarPro"):
        ReconstructionPolicy(
            assembly=AssemblySpec(
                algorithm="legacy_position_v1",
                weighting="uniform",
                window="legacy_stitch_crop",
            ),
            calibration=CalibrationSpec(method="varpro_s1s2_v1"),
        )


def test_policy_identity_has_no_scale_profile_axis():
    from ptycho.reconstruction_policy import (
        ReconstructionPolicy,
        resolve_cli_reconstruction_policy,
    )

    field_names = {field.name for field in fields(ReconstructionPolicy)}
    assert "scale_contract_version" not in field_names
    assert "measurement_domain" not in field_names

    identity = resolve_cli_reconstruction_policy("probe", True).identity
    assert "ci_intensity" not in identity
    assert "legacy_v1" not in identity
    assert "count_intensity" not in identity


def test_policy_records_are_frozen():
    from ptycho.reconstruction_policy import resolve_cli_reconstruction_policy

    policy = resolve_cli_reconstruction_policy("probe", True)
    with pytest.raises(FrozenInstanceError):
        policy.assembly = policy.assembly
    with pytest.raises(FrozenInstanceError):
        policy.calibration.method = "identity_v1"
    with pytest.raises(FrozenInstanceError):
        policy.output.final_crop = "none"
