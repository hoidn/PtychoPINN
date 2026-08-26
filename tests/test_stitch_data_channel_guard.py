"""stitch_data channel guard: mismatched channel dim must fail loudly (AssertionError),
not TypeError — params.get(['N']) was an unhashable-key latent bug (W3.2)."""
import numpy as np
import pytest


def test_channel_mismatch_raises_assertion_not_typeerror():
    from ptycho import params
    from ptycho.config.legacy_state import legacy_params_scope
    from ptycho.data_preprocessing import stitch_data

    with legacy_params_scope():
        params.cfg.update({'gridsize': 2, 'N': 4})
        bad = np.zeros((8, 4, 4, 2), dtype=np.float32)  # channel dim 2 != 1 and != N
        with pytest.raises(AssertionError):
            stitch_data(bad, nimgs=1, outer_offset=4)
