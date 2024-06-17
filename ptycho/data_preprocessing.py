from ptycho.data_loading import (
    load_simulated_data,
    load_experimental_data,
    load_xpp_data,
    load_generic_data,
)
from ptycho.data_processing import process_simulated_data

from ptycho.loader import PtychoDataContainer, PtychoDataset
from ptycho import params
from ptycho import probe

def create_ptycho_dataset(
    X_train,
    Y_I_train,
    Y_phi_train,
    intensity_scale,
    YY_train_full,
    coords_train_nominal,
    coords_train_true,
    X_test,
    Y_I_test,
    Y_phi_test,
    YY_test_full,
    coords_test_nominal,
    coords_test_true,
):
    return PtychoDataset(
        PtychoDataContainer(
            X_train,
            Y_I_train,
            Y_phi_train,
            intensity_scale,
            YY_train_full,
            coords_train_nominal,
            coords_train_true,
            None,
            None,
            None,
            probe.get_probe(fmt="np"),
        ),
        PtychoDataContainer(
            X_test,
            Y_I_test,
            Y_phi_test,
            intensity_scale,
            YY_test_full,
            coords_test_nominal,
            coords_test_true,
            None,
            None,
            None,
            probe.get_probe(fmt="np"),
        ),
    )

def load_simulated_data_container(probeGuess, config=None):
    if config:
        for key, value in config.items():
            params.set(key, value)

    size = params.cfg["size"]
    outer_offset_train = params.cfg["outer_offset_train"]
    outer_offset_test = params.cfg["outer_offset_test"]
    jitter_scale = params.cfg["sim_jitter_scale"]

    (
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
    ) = load_simulated_data(size, probeGuess, outer_offset_train, outer_offset_test, jitter_scale)

    X_train, Y_I_train, Y_phi_train, YY_ground_truth = process_simulated_data(
        X_train, Y_I_train, Y_phi_train, X_test, Y_I_test, Y_phi_test, YY_test_full, outer_offset_test
    )

    train_data_container = PtychoDataContainer(
        X_train,
        Y_I_train,
        Y_phi_train,
        intensity_scale,
        YY_train_full,
        coords_train_nominal,
        coords_train_true,
        None,
        None,
        None,
        probeGuess,
    )
    test_data_container = PtychoDataContainer(
        X_test,
        Y_I_test,
        Y_phi_test,
        norm_Y_I_test,
        YY_test_full,
        coords_test_nominal,
        coords_test_true,
        None,
        None,
        None,
        probeGuess,
    )

    return train_data_container, test_data_container, YY_ground_truth, YY_test_full

def generate_data(probeGuess=None):
    data_source = params.params()["data_source"]
    probe_np = probe.get_probe(fmt="np")
    outer_offset_train = params.cfg["outer_offset_train"]
    outer_offset_test = params.cfg["outer_offset_test"]
    YY_test_full = None
    norm_Y_I_test = None

    if data_source in ["lines", "grf", "points", "testimg", "diagonals", "V"]:
        train_data_container, test_data_container, YY_ground_truth, YY_test_full = load_simulated_data_container(probe_np)
        intensity_scale = train_data_container.norm_Y_I
        ptycho_dataset = PtychoDataset(train_data_container, test_data_container)
    elif data_source == "experimental":
        (
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
        ) = load_experimental_data(probe_np, outer_offset_train, outer_offset_test, params.params()["sim_jitter_scale"])
        X_train, Y_I_train, Y_phi_train, YY_ground_truth = process_simulated_data(
            X_train, Y_I_train, Y_phi_train, X_test, Y_I_test, Y_phi_test, YY_test_full, outer_offset_test
        )
        ptycho_dataset = create_ptycho_dataset(
            X_train,
            Y_I_train,
            Y_phi_train,
            intensity_scale,
            YY_train_full,
            coords_train_nominal,
            coords_train_true,
            X_test,
            Y_I_test,
            Y_phi_test,
            YY_test_full,
            coords_test_nominal,
            coords_test_true,
        )
    elif data_source == "xpp":
        test_data_container, train_data_container = load_xpp_data(probeGuess)
        intensity_scale = train_data_container.norm_Y_I
        ptycho_dataset = PtychoDataset(train_data_container, test_data_container)
        YY_ground_truth = None
        YY_test_full = None
    elif data_source == "generic":
        train_data_container, test_data_container = load_generic_data(probeGuess, params.cfg["N"])
        intensity_scale = train_data_container.norm_Y_I
        ptycho_dataset = PtychoDataset(train_data_container, test_data_container)
        YY_ground_truth = None
        print("INFO: train data:")
        print(train_data_container)
        print("INFO: test data:")
        print(test_data_container)
    else:
        raise ValueError("Invalid data source")

    params.cfg["intensity_scale"] = intensity_scale
    return (
        ptycho_dataset.train_data.X,
        ptycho_dataset.train_data.Y_I,
        ptycho_dataset.train_data.Y_phi,
        ptycho_dataset.test_data.X,
        ptycho_dataset.test_data.Y_I,
        ptycho_dataset.test_data.Y_phi,
        YY_ground_truth,
        ptycho_dataset,
        YY_test_full,
        norm_Y_I_test,
    )
