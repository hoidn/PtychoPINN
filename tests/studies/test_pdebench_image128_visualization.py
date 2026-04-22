import numpy as np


def test_cfd_cns_visual_spec_uses_field_specific_colormaps():
    from scripts.studies.pdebench_image128.visualization import cfd_cns_field_visual_spec

    scalar = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

    assert cfd_cns_field_visual_spec("density", [scalar])["cmap"] == "cividis"
    assert cfd_cns_field_visual_spec("Vx", [scalar])["cmap"] == "RdBu_r"
    assert cfd_cns_field_visual_spec("Vy", [scalar])["cmap"] == "PuOr"
    assert cfd_cns_field_visual_spec("pressure", [scalar])["cmap"] == "magma"
    assert cfd_cns_field_visual_spec("density", [scalar], is_error=True)["cmap"] == "magma"


def test_cfd_cns_visual_spec_uses_symmetric_limits_for_signed_velocity_fields():
    from scripts.studies.pdebench_image128.visualization import cfd_cns_field_visual_spec

    vx = np.array([[-2.0, 1.0], [0.5, 3.5]], dtype=np.float32)
    spec = cfd_cns_field_visual_spec("Vx", [vx])

    assert spec["vmin"] == -3.5
    assert spec["vmax"] == 3.5


def test_cfd_cns_visual_spec_uses_combined_range_for_positive_fields():
    from scripts.studies.pdebench_image128.visualization import cfd_cns_field_visual_spec

    density_a = np.array([[1.0, 2.0]], dtype=np.float32)
    density_b = np.array([[0.5, 4.5]], dtype=np.float32)
    spec = cfd_cns_field_visual_spec("density", [density_a, density_b])

    assert spec["vmin"] == 0.5
    assert spec["vmax"] == 4.5


def test_cfd_cns_visual_spec_uses_zero_based_limits_for_error_panels():
    from scripts.studies.pdebench_image128.visualization import cfd_cns_field_visual_spec

    err_a = np.array([[0.0, 2.0]], dtype=np.float32)
    err_b = np.array([[1.0, 4.0]], dtype=np.float32)
    spec = cfd_cns_field_visual_spec("pressure", [err_a, err_b], is_error=True)

    assert spec["vmin"] == 0.0
    assert spec["vmax"] == 4.0
