import os
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import dill
import argparse
from ptycho import params
from ptycho import params as p

save_model = False
save_data = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        prog = 'PtychoPINN',
                        description = 'Generate / load data and train the model',
                        )
    parser.add_argument('offset', type = int)
    args = parser.parse_args()
    # offset between neighboring scan points, in pixels
    offset = params.cfg['offset'] = args.offset
else:
    offset = params.cfg['offset']

prefix = params.params()['output_prefix']
now = datetime.now() # current date and time
date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
date_time = date_time.replace('/', '-').replace(':', '.').replace(', ', '-')

print('offset', offset)
out_prefix = '{}/{}_{}/'.format(prefix, date_time, offset)
os.makedirs(out_prefix, exist_ok=True)

from ptycho.initialize_data import *

#i = 1
#plt.imshow(np.log(normed_ff_np
#                  (np.array(hh.combine_complex(Y_I_train, Y_phi_train))[0, :, :, 0])), cmap = 'jet')
#plt.colorbar()

#plt.imshow(np.log(X_train[0, :, :, 0]), cmap = 'jet')
#plt.colorbar()

from ptycho import model

matplotlib.rcParams['font.size'] = 12

#Create a TensorBoard callback
logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = '500,520')

 #TODO handle intensity scaling more gracefully
# TODO descriptive names (b, a, ....  O.o)
if params.params()['positions.provided']:
    print('using provided scan positions for training')
    history = model.train(nepochs, X_train, coords_train_true, Y_I_train)
    b, a, reg, L2_error = model.autoencoder.predict([X_test * model.params()['intensity_scale'],
                                                    coords_test_true])

else:
    print('using nominal scan positions for training')
    history = model.train(nepochs, X_train, coords_train_nominal, Y_I_train)
    b, a, reg, L2_error = model.autoencoder.predict([X_test * model.params()['intensity_scale'],
                                                    coords_test_nominal])

from ptycho import baselines as bl
from ptycho.params import params

# TODO dict reference -> function call
bigoffset = params()['bigoffset']

def show_groundtruth():
    plt.rcParams["figure.figsize"] = (10, 10)
    # ground truth
    plt.imshow(YY_I_test_full[0, clipleft: -clipright, clipleft: -clipright], interpolation = 'none',
              cmap = 'jet')
#    vmin = np.min(YY_I_test_full[0, clipleft: -clipright, clipleft: -clipright])
#    vmax = np.max(YY_I_test_full[0, clipleft: -clipright, clipleft: -clipright])




# reconstructed
stitched = stitch(b, norm_Y_I_test,
                  norm = False)

plt.imsave(out_prefix + 'recon.png', stitched[0][:, :, 0], cmap = 'jet')
plt.imsave(out_prefix + 'orig.png', YY_I_test_full[0, clipleft: -clipright, clipleft: -clipright, 0],
          cmap = 'jet')

with open(out_prefix + '/history.dill', 'wb') as file_pi:
    dill.dump(history.history, file_pi)

with open(out_prefix + '/params.dill', 'wb') as f:
    dill.dump(p.cfg, f)

if save_model:
    model.autoencoder.save('{}.h5'.format(out_prefix + 'wts'), save_format="tf")

if save_data:
    with open(out_prefix + '/test_data.dill', 'wb') as f:
        dill.dump(
            {'YY_I_test_full': YY_I_test_full,
            'Y_I_test': Y_I_test,
            'Y_phi_test': Y_phi_test,
            'X_test': X_test}, f)

#with open('/trainHistoryDict', "rb") as file_pi:
#    history = pickle.load(file_pi)
#plt.imshow(np.absolute(model.autoencoder.variables[-1]), cmap = 'jet')
#plt.colorbar()
