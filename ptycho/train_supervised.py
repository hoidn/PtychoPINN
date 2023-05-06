from ptycho.generate_data import *
from ptycho import tf_helper as hh
from ptycho import baselines as bl
from ptycho import params as p

# For comparison to the 'baseline' model (PtychoNN) we need to crop/shift in a different way
def xyshift(arr4d, dx, dy):
    assert len(arr4d.shape) == 4
    from scipy.ndimage.interpolation import shift
    arr4d = np.roll(arr4d, dx, axis = 1)
    arr4d = np.roll(arr4d, dy, axis = 2)
    return arr4d

if p.cfg['gridsize'] == 2:
    history = bl.train((X_train[:, :, :, :4]),
                              Y_I_train[:, :, :, :4], Y_phi_train[:, :, :, :4])

    baseline_overlap_pred_I, baseline_overlap_pred_phi = history.predict(
        [X_test[:, :, :, :4]  * bl.params.params()['intensity_scale']])
    baseline_overlap_stitched = stitch(baseline_overlap_pred_I[:, :, :, :1], norm_Y_I_test)
    YY_baseline_overlap = xyshift(baseline_overlap_stitched, -2, -2)
    YY_phi_baseline_overlap = xyshift(stitch(baseline_overlap_pred_phi[:, :, :, :1], 1), -2, -2)
    stitched_obj = hh.combine_complex(YY_baseline_overlap, YY_phi_baseline_overlap)

elif p.cfg['gridsize'] == 1:
    history = bl.train((X_train[:, :, :, :1]), Y_I_train[:, :, :, :1], Y_phi_train[:, :, :, :1])
    baseline_pred_I, baseline_pred_phi = history.predict([X_test[:, :, :, 0] * bl.params.params()['intensity_scale']])
    baseline_stitched = stitch(baseline_pred_I, norm_Y_I_test, nsegments=37)

    YY_baseline = baseline_stitched
    YY_phi_baseline = stitch(baseline_pred_phi[:, :, :, :1], 1)
    stitched_obj = hh.combine_complex(YY_baseline, YY_phi_baseline)
else:
    raise ValueError

