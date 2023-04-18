"""
Stores global variables for data generation and model configuration
"""
# TODO naming convention for different types of parameters
cfg = {
    'N': 64, 'offset': 4, 'gridsize': 2, 'bigoffset': 4, 'batch_size': 16,
    'nepochs': 60, 'h': 64, 'w': 64, 'n_filters_scale': 2, 'output_prefix': 'outputs',
    'big_gridsize': 10, 'max_position_jitter': 0, 'sim_jitter_scale': 0.,
    'default_probe_scale': 0.7, 'mae_weight': 0., 'nll_weight': 1.,
    'sim_nphotons': 1e9, 'nsegments': 37,# TODO parameterize
    'data_source': 'lines', 'probe.trainable': False,
    'intensity_scale.trainable': False, 'positions.provided': False,
    'object.big': False, 'set_phi': False
    }

# TODO h, w
# TODO bigoffset should be a derived quantity, at least for simulation
def get_bigN():
    N = cfg['N']
    gridsize = cfg['gridsize']
    offset = cfg['offset']
    return N + (gridsize - 1) * offset

def get_padding_size():
    buffer = cfg['max_position_jitter']
    gridsize = cfg['gridsize']
    offset = cfg['offset']
    return (gridsize - 1) * offset + buffer

def get_padded_size():
    bigN = get_bigN()
    buffer = cfg['max_position_jitter']
    return bigN + buffer

def params():
    d = {k:v for k, v in cfg.items()}
    d['bigN'] = get_bigN()
    return d

def validate():
    assert cfg['data_source'] in ['lines', 'grf', 'experimental']
    # TODO
    return True

def set(key, value):
    cfg[key] = value
    assert validate()

def get(key):
    if key == 'bigN':
        cfg['bigN'] = get_bigN()
        return cfg['bigN']
    return cfg[key]
