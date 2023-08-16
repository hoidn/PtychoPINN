from ptycho.generate_data import *
from ptycho import params, model

# data parameters
offset = params.cfg['offset']
N = params.cfg['N']
gridsize = params.cfg['gridsize']
jitter_scale = params.params()['sim_jitter_scale']

# training parameters
nepochs = params.cfg['nepochs']
batch_size = params.cfg['batch_size']


# TODO reconstructed_obj -> pred_Y or something
if params.params()['positions.provided']:
    print('using provided scan positions for training')
    history = model.train(nepochs, X_train, coords_train_true, Y_I_train)
    reconstructed_obj, pred_amp, reg, L2_error = model.autoencoder.predict([X_test * model.params()['intensity_scale'],
                                                                     coords_test_true])

else:
    print('using nominal scan positions for training')
    history = model.train(nepochs, X_train, coords_train_nominal, Y_I_train)
    reconstructed_obj, pred_amp, reg, L2_error = model.autoencoder.predict([X_test * model.params()['intensity_scale'],
                                                                     coords_test_nominal])

try:
    stitched_obj = reassemble(reconstructed_obj, part='complex')
except (ValueError, TypeError) as e:
    print('object stitching failed:', e)
