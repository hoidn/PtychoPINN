cfg = {'N': 64, 'offset': 4, 'gridsize': 2, 'bigoffset': 4, 'batch_size': 16,
    'h': 64, 'w': 64, 'intensity_scale.trainable': False}

def get_bigN():
    N = cfg['N']
    gridsize = cfg['gridsize']
    offset = cfg['offset']
    return N + (gridsize - 1) * offset

def params():
    d = {k:v for k, v in cfg.items()}
    d['bigN'] = get_bigN()
    return d

