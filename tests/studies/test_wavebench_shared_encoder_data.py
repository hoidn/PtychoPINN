from pathlib import Path

import torch

from scripts.studies.wavebench_shared_encoder import data as wb_data


def test_load_locked_contract_uses_authoritative_wavebench_inputs():
    contract = wb_data.load_locked_contract(Path("."))

    assert contract["selected_variant"] == "time_varying/is/thick_lines_gaussian_lens"
    assert (
        contract["selected_dataset_member"]
        == "wavebench_dataset/time_varying/is/thick_lines_gaussian_lens.beton"
    )
    assert contract["stable_dataset_target"]["repo_relative"] == "wavebench_dataset/time_varying/is/"
    assert contract["split"] == {"train": 9000, "val": 500, "test": 500, "seed": 42}
    assert contract["tensor_contract"]["input_shape"] == [1, 128, 128]
    assert contract["tensor_contract"]["target_shape"] == [1, 128, 128]
    assert tuple(contract["latent_channels"]) == (32, 64)
    assert tuple(contract["row_roster"]) == (
        "cnn",
        "hybrid_resnet",
        "spectral_resnet_bottleneck_net",
        "fno",
        "ffno",
    )


def test_build_split_indices_matches_locked_counts_without_overlap():
    indices = wb_data.build_split_indices()

    assert len(indices["train"]) == 9000
    assert len(indices["val"]) == 500
    assert len(indices["test"]) == 500
    assert len(set(indices["train"]) & set(indices["val"])) == 0
    assert len(set(indices["train"]) & set(indices["test"])) == 0
    assert len(set(indices["val"]) & set(indices["test"])) == 0
    assert sorted(indices["train"] + indices["val"] + indices["test"]) == list(range(10000))


def test_trim_split_indices_preserves_locked_order_prefix():
    indices = wb_data.build_split_indices()
    trimmed = wb_data.trim_split_indices(indices, train=8, val=4, test=4)

    assert trimmed["train"] == indices["train"][:8]
    assert trimmed["val"] == indices["val"][:4]
    assert trimmed["test"] == indices["test"][:4]


def test_summarize_tensor_batch_reports_locked_tensor_contract():
    inputs = torch.zeros(2, 1, 128, 128, dtype=torch.float32)
    targets = torch.ones(2, 1, 128, 128, dtype=torch.float32)

    summary = wb_data.summarize_tensor_batch(inputs, targets)

    assert summary["input_shape"] == [2, 1, 128, 128]
    assert summary["target_shape"] == [2, 1, 128, 128]
    assert summary["input_dtype"] == "torch.float32"
    assert summary["target_dtype"] == "torch.float32"
    assert summary["input_min"] == 0.0
    assert summary["target_max"] == 1.0
