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


def test_cfd_cns_shared_scale_bundle_uses_signed_value_scale_and_zero_error_floor():
    from scripts.studies.pdebench_image128.visualization import cfd_cns_shared_scale_bundle

    target = np.array([[-1.0, 0.5], [0.0, 2.0]], dtype=np.float32)
    prediction = np.array([[0.25, -3.0], [1.0, 0.0]], dtype=np.float32)
    abs_error = np.abs(prediction - target)

    spec = cfd_cns_shared_scale_bundle("Vx", value_arrays=[target, prediction], error_arrays=[abs_error])

    assert spec["value_scale"]["vmin"] == -3.0
    assert spec["value_scale"]["vmax"] == 3.0
    assert spec["error_scale"]["vmin"] == 0.0
    assert spec["error_scale"]["vmax"] == float(np.max(abs_error))


def test_cfd_cns_shared_scale_bundle_is_deterministic_across_input_order():
    from scripts.studies.pdebench_image128.visualization import cfd_cns_shared_scale_bundle

    density_a = np.array([[1.0, 2.5]], dtype=np.float32)
    density_b = np.array([[0.5, 4.0]], dtype=np.float32)
    err_a = np.array([[0.0, 0.3]], dtype=np.float32)
    err_b = np.array([[0.1, 0.8]], dtype=np.float32)

    spec_a = cfd_cns_shared_scale_bundle(
        "density",
        value_arrays=[density_a, density_b],
        error_arrays=[err_a, err_b],
    )
    spec_b = cfd_cns_shared_scale_bundle(
        "density",
        value_arrays=[density_b, density_a],
        error_arrays=[err_b, err_a],
    )

    assert spec_a == spec_b
