def test_dataset_norm_y_i():
    import numpy as np
    import tensorflow as tf
    from ptycho.loader import PtychoDataset

    # --- Test: norm_Y_I provided in train_data ---
    class FakeData:
        def __init__(self, X, norm_Y_I):
            self.X = X
            self.norm_Y_I = norm_Y_I

    X = np.ones((10, 64, 64))
    provided_norm = 5.0
    train_data = FakeData(X, provided_norm)
    test_data = FakeData(X, provided_norm)
    ds = PtychoDataset(train_data, test_data)
    assert ds.norm_Y_I == provided_norm, "norm_Y_I should be taken directly from train_data when provided"

    # --- Test: Fallback calculation when norm_Y_I is None ---
    train_data_no_norm = FakeData(X, None)
    ds_fallback = PtychoDataset(train_data_no_norm, test_data)
    # Check that fallback returns a numeric value (scale_nphotons returns a float/numeric tensor)
    assert ds_fallback.norm_Y_I is not None, "norm_Y_I fallback should be computed when missing"
    assert isinstance(ds_fallback.norm_Y_I, (float, np.floating, np.ndarray)), "Fallback norm_Y_I should be numeric"

    # --- Test: Error case when train_data lacks X attribute ---
    class FakeDataNoX:
        def __init__(self, norm_Y_I):
            self.norm_Y_I = norm_Y_I

    try:
        FakeDataNoX_instance = FakeDataNoX(None)
        _ = PtychoDataset(FakeDataNoX_instance, test_data)
        assert False, "Expected AttributeError when train_data.X is missing"
    except AttributeError:
        pass
