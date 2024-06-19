import numpy as np
from ptycho import params
from ptycho import diffsim as datasets


def load_simulated_data(probe, cfg):
    np.random.seed(1)
    X_train, Y_I_train, Y_phi_train, intensity_scale, YY_train_full, _, (coords_train_nominal, coords_train_true) = (
        datasets.mk_simdata(cfg['nimgs_train'], cfg['size'], probe, cfg['outer_offset_train'], jitter_scale=cfg['jitter_scale'], which="train", cfg = cfg)
    )
    cfg["intensity_scale"] = intensity_scale

    np.random.seed(2)
    X_test, Y_I_test, Y_phi_test, _, YY_test_full, norm_Y_I_test, (coords_test_nominal, coords_test_true) = (
        datasets.mk_simdata(cfg['nimgs_test'], cfg['size'], probe, cfg['outer_offset_test'], intensity_scale, jitter_scale=cfg['jitter_scale'], which="test", cfg = cfg)
    )

    return (
        X_train,
        Y_I_train,
        Y_phi_train,
        X_test,
        Y_I_test,
        Y_phi_test,
        intensity_scale,
        YY_train_full,
        YY_test_full,
        norm_Y_I_test,
        coords_train_nominal,
        coords_train_true,
        coords_test_nominal,
        coords_test_true,
    )

def load_experimental_data(probe, cfg):
    from ptycho import experimental

    YY_I, YY_phi = experimental.get_full_experimental("train")
    X_train, Y_I_train, Y_phi_train, intensity_scale, YY_train_full, _, (coords_train_nominal, coords_train_true) = (
        datasets.mk_simdata(cfg['nimgs_train'], experimental.train_size, probe, cfg['outer_offset_train'], jitter_scale=cfg['jitter_scale'], YY_I=YY_I, YY_phi=YY_phi, cfg = cfg)
    )

    YY_I, YY_phi = experimental.get_full_experimental("test")
    X_test, Y_I_test, Y_phi_test, _, YY_test_full, norm_Y_I_test, (coords_test_nominal, coords_test_true) = (
        datasets.mk_simdata(cfg['nimgs_test'], experimental.test_size, probe, cfg['outer_offset_test'], intensity_scale, jitter_scale=cfg['jitter_scale'], YY_I=YY_I, YY_phi=YY_phi, cfg = cfg)
    )

    return (
        X_train,
        Y_I_train,
        Y_phi_train,
        X_test,
        Y_I_test,
        Y_phi_test,
        intensity_scale,
        YY_train_full,
        YY_test_full,
        norm_Y_I_test,
        coords_train_nominal,
        coords_train_true,
        coords_test_nominal,
        coords_test_true,
    )

def load_xpp_data(probeGuess):
    from ptycho import xpp

    train_data_container = loader.load(xpp.get_data, probeGuess, which="train")
    test_data_container = loader.load(xpp.get_data, probeGuess, which="test")
    return train_data_container, test_data_container

def load_generic_data(probeGuess, cfg):
    N = cfg['N']
    from ptycho.loader import RawData

    train_data_file_path = params.get("train_data_file_path")
    test_data_file_path = params.get("test_data_file_path")

    train_raw, test_raw = RawData.from_files(train_data_file_path, test_data_file_path)

    dset_train = train_raw.generate_grouped_data(N, K=7, nsamples=1)
    dset_test = test_raw.generate_grouped_data(N, K=7, nsamples=1)

    train_data_container = loader.load(lambda: dset_train, probeGuess, which=None, create_split=False, cfg = cfg)
    test_data_container = loader.load(lambda: dset_test, probeGuess, which=None, create_split=False, cfg = cfg)
    return train_data_container, test_data_container

from ptycho import loader

