from ptycho.generate_data import reassemble
from ptycho import params, model

#offset = params.cfg['offset']
#N = params.cfg['N']
#gridsize = params.cfg['gridsize']
#jitter_scale = params.params()['sim_jitter_scale']
#batch_size = params.cfg['batch_size']

def train(train_data):
    # training parameters
    nepochs = params.cfg['nepochs']
    return model.train(nepochs, train_data)

def eval(test_data, history):
    reconstructed_obj, pred_amp, reconstructed_obj_cdi = history.predict([test_data.X * model.params()['intensity_scale'], test_data.coords_nominal ])
    try:
        stitched_obj = reassemble(reconstructed_obj, part='complex')
    except (ValueError, TypeError) as e:
        print('object stitching failed:', e)
    return reconstructed_obj, pred_amp, reconstructed_obj_cdi, stitched_obj

def train_eval(ptycho_dataset):
    ## TODO reconstructed_obj -> pred_Y or something
    history = train(ptycho_dataset)
    reconstructed_obj, pred_amp, reconstructed_obj_cdi, stitched_obj = eval(ptycho_dataset.test_data, history)
    return history, reconstructed_obj, pred_amp, reconstructed_obj_cdi, stitched_obj
