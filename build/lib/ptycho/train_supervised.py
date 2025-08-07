from ptycho.generate_data import *
from ptycho import tf_helper as hh
from ptycho import baselines as bl
from ptycho import params as p

offset = p.get('offset')

# For comparison to the 'baseline' model (PtychoNN) we need to crop/shift in a different way
def xyshift(arr4d, dx, dy):
    assert len(arr4d.shape) == 4
    from scipy.ndimage.interpolation import shift
    arr4d = np.roll(arr4d, dx, axis = 1)
    arr4d = np.roll(arr4d, dy, axis = 2)
    return arr4d

def get_recon_patches_single_channel(X):
    """
    Reconstructs obj patches using a single channel of X (assumes 'model' is the vanilla
    supervised model)
    """
    baseline_pred_I, baseline_pred_phi = model.predict([X[:, :, :, 0] * bl.params.params()['intensity_scale']])
    return hh.combine_complex(baseline_pred_I, baseline_pred_phi)
#    baseline_pred_I, baseline_pred_phi = model.predict([X[:, :, :, 0]])
#    return hh.combine_complex(baseline_pred_I, baseline_pred_phi)

def get_recon_patches_grid(X):
    """
    Reconstructs obj patches using a single channel of X (assumes 'model' is the vanilla
    supervised model)
    """
    baseline_overlap_pred_I, baseline_overlap_pred_phi = model.predict(
        [X_test[:, :, :, :4]  * bl.params.params()['intensity_scale']])
    obj_stitched = hh.combine_complex(baseline_overlap_pred_I[:, :, :, :1], baseline_overlap_pred_phi[:, :, :, :1])
    return xyshift(obj_stitched, -offset // 2, -offset // 2)

if p.cfg['gridsize'] == 2:
    model, history = bl.train((X_train[:, :, :, :4]),
                              Y_I_train[:, :, :, :4], Y_phi_train[:, :, :, :4])

    reconstructed_obj = get_recon_patches_grid(X_test)
    #stitched_obj = reassemble(reconstructed_obj, part = 'complex')

elif p.cfg['gridsize'] == 1:
    model, history = bl.train((X_train[:, :, :, :1]), Y_I_train[:, :, :, :1], Y_phi_train[:, :, :, :1])

    # TODO match above
    reconstructed_obj = get_recon_patches_single_channel(X_test)
    #stitched_obj = reassemble(reconstructed_obj, part = 'complex')

    reconstructed_obj_train = get_recon_patches_single_channel(X_train)

else:
    raise ValueError

try:
    stitched_obj = reassemble(reconstructed_obj, part='complex')
except (ValueError, TypeError) as e:
    print('object stitching failed:', e)
