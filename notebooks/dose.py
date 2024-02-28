import argparse

def init(nphotons):
    from ptycho.params import cfg
    p.cfg['positions.provided'] = False
    p.cfg['data_source'] = 'lines'
    p.cfg['set_phi'] = False
    p.cfg['nepochs'] = 60

    p.cfg['offset'] = 4
    p.cfg['max_position_jitter'] = 3
    p.cfg['output_prefix'] = 'lines3'

    p.cfg['gridsize'] = 2
    p.cfg['n_filters_scale'] = 2
    p.cfg['object.big'] = True
    p.cfg['intensity_scale.trainable'] = True
    p.cfg['probe.trainable'] = False

    p.cfg['outer_offset_train'] = 8
    p.cfg['outer_offset_test'] = 20
    p.cfg['nimgs_train'] = 2
    p.cfg['nimgs_test'] = 2

    p.cfg['nphotons'] = nphotons

def execute():
    from ptycho.evaluation import save_metrics, trim
    from ptycho.tf_helper import pad

    from ptycho.params import cfg
    from ptycho import generate_data as init

    from ptycho.params import cfg
    p.cfg['data_source'] = 'lines'
    p.cfg['offset'] = 4
    p.cfg['max_position_jitter'] = 10
    p.cfg['output_prefix'] = 'lines2'

    p.cfg['gridsize'] = 2
    p.cfg['n_filters_scale'] = 2
    p.cfg['object.big'] = True
    p.cfg['intensity_scale.trainable'] = True

    from ptycho.train import train
from ptycho.model import Conv_Pool_block, Conv_Up_block
    # reload(model)
    # reload(train)

    # print(p.cfg)
    from ptycho.train import stitched_obj, YY_ground_truth

    from ptycho.train_pinn import train as train_pinn, eval as eval_pinn

    d = save_metrics(stitched_obj, YY_ground_truth, label = 'PINN,NLL,overlaps')
    d
    #d0 = d

    import matplotlib.pyplot as plt
    import numpy as np

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # reconstructed amplitude images
    img1 = axs[0, 0].imshow(np.absolute(stitched_obj)[0], cmap='jet', interpolation='none')
    axs[0, 0].set_title('Reconstructed amplitude')

    # reconstructed phase images
    axs[0, 1].imshow(np.angle(stitched_obj)[0], cmap='jet')
    fig.colorbar(img1, ax=axs[0, 1])
    axs[0, 1].set_title('Reconstructed phase')

    # ground truth amplitude images
    img = axs[1, 0].imshow(np.absolute(init.YY_ground_truth), interpolation='none', cmap='jet')
    axs[1, 0].set_title('Ground truth amplitude')

    # ground truth phase images
    img = axs[1, 1].imshow(np.angle(init.YY_ground_truth), interpolation='none', cmap='jet')
    axs[1, 1].set_title('Ground truth phase')
    fig.colorbar(img, ax=axs[1, 1])
    return d, init.YY_ground_truth, stitched_obj

def parse_arguments():
    parser = argparse.ArgumentParser(description='Ptychographic reconstruction script.')
    parser.add_argument('nphotons', type=float, help='Number of photons')
    args = parser.parse_args()
    return args.nphotons

if __name__ == '__main__':
    nphotons = parse_arguments()
    init(nphotons)

    d, YY_ground_truth, stitched_obj  = execute()
