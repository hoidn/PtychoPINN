import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def symmetrize(arr):
    return (arr + arr[::-1, ::-1]) / 2

def symmetrize_3d(arr):
    return (arr + arr[:, ::-1, ::-1]) / 2

def cropshow(arr, *args, **kwargs):
    arr = arr[16:-16, 16:-16]
    plt.imshow(arr, *args, **kwargs)

from scipy.ndimage import gaussian_filter as gf
def summarize(i, a, b, X_test, Y_I_test, Y_phi_test, probe, channel = 0):
    plt.rcParams["figure.figsize"] = (10, 10)
    vmin = 0
    vmax = np.absolute(b)[i].max()

    aa, bb = 3, 3
    plt.subplot(aa, bb, 1)
    plt.title('True amp.\n(illuminated)')
    #cropshow((Y_I_test[i]), cmap = 'jet', vmin = vmin, vmax = vmax)
    cropshow((Y_I_test[i, :, :, channel]), cmap = 'jet')

    plt.subplot(aa, bb, 2)
    plt.title('Reconstructed amp.\n(illuminated)')
#    cropshow(
#        gf((np.absolute(b))[i] * probe[..., None], .7), cmap = 'jet')
    cropshow((np.absolute(b))[i] * probe[..., None], cmap = 'jet')
#     cropshow((np.absolute(b))[i] * probe[..., None], vmin = np.min(Y_I_test[i]),
#              vmax = np.max(Y_I_test[i]), cmap = 'jet')

    plt.subplot(aa, bb, 3)
    plt.title('True phase')
    cropshow(((Y_phi_test * (probe > .01)[..., None]))[i, :, :, channel], cmap = 'jet')
    plt.colorbar()

    plt.subplot(aa, bb, 4)
    plt.title('True amp.\n(full)')
    cropshow((Y_I_test[i, :, :, channel] / (probe + 1e-9)), cmap = 'jet')

    plt.subplot(aa, bb, 5)
    plt.title('Reconstructed amp. (full)')
    #plt.imshow((np.absolute(b))[i], cmap = 'jet')
    cropshow((np.absolute(b))[i], cmap = 'jet')

    plt.subplot(aa, bb, 6)
    plt.title('Reconstructed phase')
    cropshow((np.angle(b) * (probe > .01)[..., None])[i], cmap = 'jet')
    plt.colorbar()
    print('phase min:', np.min((np.angle(b) * (probe > .01)[..., None])),
        'phase max:', np.max((np.angle(b) * (probe > .01)[..., None])))

    plt.subplot(aa, bb, 7)
    plt.title('True diffraction')
    plt.imshow(np.log(X_test)[i, :, :, channel], cmap = 'jet')

    plt.subplot(aa, bb, 8)
    plt.title('Recon diffraction')
    plt.imshow(np.log(a)[i, :, :, channel], cmap = 'jet')

def plt_metrics(history, loss_type = 'MAE', metric2 = 'padded_obj_loss'):
    hist=history
    epochs=np.asarray(history.epoch)+1

    plt.style.use('seaborn-white')
    matplotlib.rc('font',family='Times New Roman')
    matplotlib.rcParams['font.size'] = 12

    f, axarr = plt.subplots(2, sharex=True, figsize=(12, 8))

    axarr[0].set(ylabel='Loss')
    axarr[0].plot(epochs,hist.history['loss'], 'C3o', label='Diffraction {} Training'.format(loss_type))
    axarr[0].plot(epochs,hist.history['val_loss'], 'C3-', label='Diffraction {} Validation'.format(loss_type))
    axarr[0].grid()
    axarr[0].legend(loc='center right', bbox_to_anchor=(1.5, 0.5))

    axarr[1].set(ylabel='Loss')
    axarr[1].plot(epochs,hist.history[metric2], 'C0o', label='Object {} Training'.format(loss_type))
    axarr[1].plot(epochs,hist.history['val_' + metric2], 'C0-', label='Object {} Validation'.format(loss_type))
    axarr[1].legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
    plt.xlabel('Epochs')
    plt.tight_layout()
    #plt.semilogy()
    axarr[1].grid()
