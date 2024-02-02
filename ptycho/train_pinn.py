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

## TODO reconstructed_obj -> pred_Y or something
history = model.train(nepochs, ptycho_dataset.train_data)
reconstructed_obj, pred_amp, reconstructed_obj_cdi = model.autoencoder.predict([ptycho_dataset.test_data.X * model.params()['intensity_scale'],
                                                                 ptycho_dataset.test_data.coords_nominal ])

try:
    stitched_obj = reassemble(reconstructed_obj, part='complex')
except (ValueError, TypeError) as e:
    print('object stitching failed:', e)
