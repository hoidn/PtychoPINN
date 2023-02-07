cfg = {'N': 64, 'offset': 4, 'gridsize': 2, 'bigoffset': 4, 'batch_size': 16,
    'nepochs': 60, 'h': 64, 'w': 64, 'intensity_scale.trainable': False,
    'probe.trainable': False, 'n_filters_scale': 2, 'output_prefix': 'outputs',
    'big_gridsize': 10}
#'h': 64, 'w': 64, 'intensity_scale.trainable': True, 'probe.trainable': True,

# TODO bigoffset should be a derived quantity, at least for simulation

def get_bigN():
    N = cfg['N']
    gridsize = cfg['gridsize']
    offset = cfg['offset']
    return N + (gridsize - 1) * offset

def params():
    d = {k:v for k, v in cfg.items()}
    d['bigN'] = get_bigN()
    return d
