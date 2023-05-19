import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf

from ptycho import params
from ptycho import misc

def recon_patches(patches):
    """
    chop channel dimension size to 1, then patch together a single image
    """
    from ptycho import generate_data as data
    return data.reassemble(patches[:, :, :, :1])[0]

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

import scipy.fftpack as fftpack
fp = fftpack

def trim(arr2d):
    offset = params.get('offset')
    assert not (offset % 2)
    return arr2d[offset // 2:-offset // 2, offset // 2:-offset // 2]

def mae(target, pred, normalize = True):
    """
    mae for an entire (stitched-together) reconstruction.
    """
    if normalize:
        scale = np.mean(target) / np.mean(pred)
    else:
        scale = 1
    print('mean scale adjustment:', scale)
    return np.mean(np.absolute(target - scale * pred))

def mse(target, pred, normalize = True):
    """
    mae for an entire (stitched-together) reconstruction.
    """
    if normalize:
        scale = np.mean(target) / np.mean(pred)
    else:
        scale = 1
    print('mean scale adjustment:', scale)
    return np.mean((target - scale * pred)**2)

def psnr(target, pred, normalize = True, shift = False):
    """
    for phase inputs, assume that global shift has already been taken care off
    """
    import cv2
    target = np.array(target)
    pred = np.array(pred)
    if normalize:
        scale = np.mean(target) / np.mean(pred)
    else:
        scale = 1
    if shift:
        offset = min(np.min(target), np.min(pred))
        target = target - offset
        pred = pred - offset
    pred = scale * pred
    return cv2.PSNR(target, pred)

def fft2d(aphi):
    F1 = fftpack.fft2((aphi).astype(float))
    F2 = fftpack.fftshift(F1)
    return F2

def highpass2d(aphi, n = 2):
    if n == 1:
        print('subtracting mean', np.mean(aphi))
        return aphi - np.mean(aphi)
    F2 = fft2d(aphi)
    (w, h) = aphi.shape
    half_w, half_h = int(w/2), int(h/2)

    # high pass filter

    F2[half_w-n:half_w+n+1,half_h-n:half_h+n+1] = 0

    im1 = fp.ifft2(fftpack.ifftshift(F2)).real
    return im1

def lowpass2d(aphi, n = 2):
    if n == 1:
        print('subtracting mean', np.mean(aphi))
        return aphi - np.mean(aphi)
    F2 = fft2d(aphi)
    (w, h) = aphi.shape
    half_w, half_h = int(w/2), int(h/2)

    # high pass filter
    mask = np.zeros_like(F2)
    mask[half_w-n:half_w+n+1,half_h-n:half_h+n+1] = 1.
    F2 = F2 * mask

    im1 = fp.ifft2(fftpack.ifftshift(F2)).real
    return im1

def frc50(target, pred):
    if np.isnan(pred).all():
        raise ValueError
    if np.max(target) == np.min(target) == 0:
        return None, np.nan
    from FRC import fourier_ring_corr as frc
    shellcorr = frc.FSC(np.array(target), np.array(pred))
    return shellcorr, np.where(shellcorr < .5)[0][0]


def eval_reconstruction(stitched_obj, ground_truth_obj, lowpass_n = 1,
        label = ''):
    assert stitched_obj.shape[1] == ground_truth_obj.shape[1]
    YY_ground_truth = np.absolute(ground_truth_obj)
    YY_phi_ground_truth = np.angle(ground_truth_obj)

    phi_pred = trim(
        highpass2d(
            np.squeeze(np.angle(stitched_obj)[0]), n = lowpass_n
        )
    )
    phi_target = trim(
        highpass2d(
            np.squeeze(YY_phi_ground_truth), n = lowpass_n
        )
    )
    amp_target = tf.cast(trim(YY_ground_truth), tf.float32)
    amp_pred = trim(np.absolute(stitched_obj)[0])

    # TODO complex FRC?
    mae_amp = mae(amp_target, amp_pred) # PINN
    mse_amp = mse(amp_target, amp_pred) # PINN
    psnr_amp = psnr(amp_target[:, :, 0], amp_pred[:, :, 0], normalize = True,
        shift = False)
    frc_amp, frc50_amp = frc50(amp_target[:, :, 0], amp_pred[:, :, 0])

    mae_phi = mae(phi_target, phi_pred, normalize=False) # PINN
    mse_phi = mse(phi_target, phi_pred, normalize=False) # PINN
    psnr_phi = psnr(phi_target, phi_pred, normalize = False, shift = True)
    frc_phi, frc50_phi = frc50(phi_target, phi_pred)

    return {'mae': (mae_amp, mae_phi),
        'mse': (mse_amp, mse_phi),
        'psnr': (psnr_amp, psnr_phi),
        'frc50': (frc50_amp, frc50_phi),
        'frc': (frc_amp, frc_phi)}

import pandas as pd
import os
import dill
def save_metrics(stitched_obj, YY_ground_truth,  label = ''):
    """
    evaluate reconstruction and save the result to disk.
    """
    out_prefix = misc.get_path_prefix()
    os.makedirs(out_prefix, exist_ok=True)
    metrics = eval_reconstruction(stitched_obj, YY_ground_truth, label = label)
    metrics['label'] = label
    d = {**params.cfg, **metrics}
    with open(out_prefix + '/params.dill', 'wb') as f:
        dill.dump(d, f)
    df = pd.DataFrame({k: d[k] for k in ['mae', 'mse', 'psnr', 'frc50']})
    df.to_csv(out_prefix + '/metrics.csv')
    return {k: metrics[k] for k in ['mae', 'mse', 'psnr', 'frc50']}
