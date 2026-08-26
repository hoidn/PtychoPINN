"""Regression gates for the converged RAM/mmap training batch rail."""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import default_collate

from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
from ptycho_torch.dataloader import PtychoDataset


N = 64


def _configs(grid_size=1, *, ci=False, batch_size=4, num_workers=0, strategy="auto"):
    from ptycho.config.config import (
        ModelConfig as PublicModelConfig,
        PyTorchExecutionConfig,
        TrainingConfig as PublicTrainingConfig,
    )

    data = DataConfig(
        N=N,
        neighbor_count=1 if grid_size == 1 else 6,
        K_quadrant=8,
        n_raw_frames_selected=1,
        gridsize=grid_size,
        neighbor_function="Nearest" if grid_size == 1 else "4_quadrant",
        scan_pattern="Isotropic",
        x_bounds=(0.0, 1.0),
        y_bounds=(0.0, 1.0),
        normalize="Batch",
        scale_contract_version="ci_intensity_v2" if ci else "legacy_v1",
        measurement_domain="count_intensity" if ci else "normalized_amplitude",
    )
    model = ModelConfig(
        object_big=grid_size > 1,
        physics_forward_mode="rectangular_scaled" if ci else "amplitude",
        cnn_output_mode="real_imag" if ci else "amp_phase",
    )
    training = TrainingConfig(
        batch_size=batch_size,
        device="cpu",
        strategy=strategy,
        num_workers=num_workers,
        orchestrator="Mlflow",
    )
    public = PublicTrainingConfig(
        model=PublicModelConfig(N=N, gridsize=grid_size, object_big=grid_size > 1),
        batch_size=batch_size,
        sequential_sampling=True,
        backend="pytorch",
    )
    payload = SimpleNamespace(
        tf_training_config=public,
        pt_data_config=data,
        pt_model_config=model,
        pt_training_config=training,
        execution_config=PyTorchExecutionConfig(
            accelerator="cpu",
            devices=2 if strategy == "ddp" else 1,
            strategy=strategy,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=2 if num_workers else None,
            logger_backend=None,
            enable_checkpointing=False,
        ),
    )
    return public, payload


def _write_npz(path, *, seed=0, count_intensity=False, probe_factor=1.0):
    rng = np.random.default_rng(seed)
    if count_intensity:
        diffraction = rng.integers(1, 40, size=(64, N, N)).astype(np.float32)
    else:
        diffraction = rng.random((64, N, N), dtype=np.float32)
        diffraction /= np.sqrt(
            np.square(diffraction).sum(axis=(-2, -1), keepdims=True)
        )
    yy, xx = np.meshgrid(
        np.arange(8, dtype=np.float64),
        np.arange(8, dtype=np.float64),
        indexing="ij",
    )
    probe = probe_factor * (
        rng.random((N, N)) + 1j * rng.random((N, N))
    )
    obj = rng.random((N, N)) + 1j * rng.random((N, N))
    np.savez(
        path,
        diff3d=diffraction.astype(np.float32),
        xcoords=xx.ravel(),
        ycoords=yy.ravel(),
        probeGuess=probe.astype(np.complex64),
        objectGuess=obj.astype(np.complex64),
    )


def _dataset(tmp_path, payload, *, ci=False, two_experiments=False):
    source = tmp_path / "npz"
    source.mkdir(parents=True)
    _write_npz(source / "experiment-0.npz", seed=11, count_intensity=ci)
    if two_experiments:
        _write_npz(
            source / "experiment-1.npz",
            seed=22,
            count_intensity=ci,
            probe_factor=2.0 + 1.0j,
        )
    return PtychoDataset(
        ptycho_dir=str(source),
        model_config=payload.pt_model_config,
        data_config=payload.pt_data_config,
        training_config=payload.pt_training_config,
        data_dir=str(tmp_path / "map" / "tensors"),
        remake_map=True,
    )


def _ram_container(dataset):
    fields = dataset.mmap_ptycho
    return {
        "X": fields["images"].permute(0, 2, 3, 1).clone(),
        "coords_relative": fields["coords_relative"].permute(0, 2, 3, 1).clone(),
        "experiment_id": fields["experiment_id"].clone(),
        "object_index": fields["object_index"].clone(),
        "nn_indices": fields["nn_indices"].clone(),
        "rms_scaling_constant": fields["rms_scaling_constant"].clone(),
        "physics_scaling_constant": fields["physics_scaling_constant"].clone(),
        "probe": dataset.data_dict["probes"][0].clone(),
        "scaling_constant": dataset.data_dict["probe_scaling"][:1].clone(),
    }


@pytest.mark.parametrize("grid_size", [1, 2])
def test_ram_and_mmap_emit_the_same_named_batch(grid_size, tmp_path):
    from ptycho_torch.workflows.components import _build_lightning_dataloaders

    public, payload = _configs(grid_size)
    mmap_dataset = _dataset(tmp_path, payload)
    ram_loader, _ = _build_lightning_dataloaders(
        _ram_container(mmap_dataset),
        None,
        public,
        payload=payload,
        torch_training_seed=19,
    )
    mmap_loader, _ = _build_lightning_dataloaders(
        mmap_dataset,
        None,
        public,
        payload=payload,
        torch_training_seed=19,
    )

    ram = next(iter(ram_loader))
    mmap = next(iter(mmap_loader))
    for name in (
        "images",
        "coords_relative",
        "rms_scaling_constant",
        "physics_scaling_constant",
        "experiment_id",
        "object_index",
        "nn_indices",
    ):
        torch.testing.assert_close(ram[0][name], mmap[0][name], msg=name)
    torch.testing.assert_close(ram[1], mmap[1], msg="probe")
    torch.testing.assert_close(ram[2], mmap[2], msg="probe_scaling")


def test_multi_experiment_mmap_batch_selects_probe_and_ci_statistics(tmp_path):
    from ptycho_torch.workflows.components import _build_lightning_dataloaders

    public, payload = _configs(ci=True, batch_size=128)
    dataset = _dataset(tmp_path, payload, ci=True, two_experiments=True)
    loader, _ = _build_lightning_dataloaders(
        dataset,
        None,
        public,
        payload=payload,
        torch_training_seed=3,
    )
    seen = set()
    for fields, probe, probe_scaling in loader:
        experiment_ids = fields["experiment_id"].long()
        for row, experiment_id in enumerate(experiment_ids.tolist()):
            seen.add(experiment_id)
            expected_probe = dataset.data_dict["probes"][experiment_id].unsqueeze(0)
            torch.testing.assert_close(probe[row], expected_probe)
            torch.testing.assert_close(fields["probe_training"][row], expected_probe)
            torch.testing.assert_close(
                fields["rms_input_scale"][row],
                dataset.data_dict["ci_statistics"]["rms_input_scale"][
                    experiment_id
                ].view(1, 1, 1),
            )
            torch.testing.assert_close(
                probe_scaling[row],
                dataset.data_dict["probe_scaling"][experiment_id].view(1, 1, 1),
            )
    assert seen == {0, 1}


def test_ram_supervised_mode_requires_both_named_labels():
    from ptycho_torch.dataloader import _PtychoContainerDataset

    with pytest.raises(ValueError, match="requires label_amp and label_phase"):
        _PtychoContainerDataset(
            {
                "X": torch.ones(2, N, N, 1),
                "coords_relative": torch.zeros(2, 1, 2, 1),
                "probe": torch.ones(N, N, dtype=torch.complex64),
            },
            model_config=ModelConfig(mode="Supervised", object_big=False),
        )


def test_ram_supervised_container_emits_only_real_labels():
    from ptycho_torch.data_container_bridge import PtychoDataContainerTorch
    from ptycho_torch.dataloader import _PtychoContainerDataset

    labels = np.full((2, N, N, 1), 3.0 + 4.0j, dtype=np.complex64)
    grouped = {
        "X_full": np.ones((2, N, N, 1), dtype=np.float32),
        "Y": labels,
        "coords_relative": np.zeros((2, 1, 2, 1), dtype=np.float32),
        "coords_offsets": np.zeros((2, 1, 2, 1), dtype=np.float64),
        "nn_indices": np.zeros((2, 1), dtype=np.int32),
    }
    model = ModelConfig(mode="Supervised", object_big=False)
    container = PtychoDataContainerTorch(
        grouped,
        np.ones((N, N), dtype=np.complex64),
    )
    dataset = _PtychoContainerDataset(container, model_config=model)

    fields, _probe, _scale = dataset.__getitems__([1, 0])
    torch.testing.assert_close(
        fields["label_amp"],
        container.Y_I[[1, 0]].permute(0, 3, 1, 2),
    )
    torch.testing.assert_close(
        fields["label_phase"],
        container.Y_phi[[1, 0]].permute(0, 3, 1, 2),
    )

    unlabeled = PtychoDataContainerTorch(
        {**grouped, "Y": None},
        np.ones((N, N), dtype=np.complex64),
    )
    with pytest.raises(ValueError, match="requires label_amp and label_phase"):
        _PtychoContainerDataset(unlabeled, model_config=model)


@pytest.mark.parametrize("channels", [1, 4])
def test_vectorized_ram_batch_preserves_legacy_collated_strides(channels):
    from ptycho_torch.dataloader import _PtychoContainerDataset

    rows = 5
    images = torch.arange(
        rows * N * N * channels, dtype=torch.float32
    ).reshape(rows, N, N, channels)
    container = {
        "X": images,
        "observed_images": images,
        "measured_intensity": images,
        "coords_relative": torch.arange(
            rows * 2 * channels, dtype=torch.float32
        ).reshape(rows, 1, 2, channels),
        "probe_training": torch.ones(N, N, dtype=torch.complex64),
        "probe_physical": torch.ones(N, N, dtype=torch.complex64),
        "probe_normalization": torch.tensor(2.0),
        "rms_input_scale": torch.tensor(3.0),
        "mean_measured_intensity": torch.tensor(4.0),
    }
    dataset = _PtychoContainerDataset(
        container,
        model_config=ModelConfig(object_big=channels > 1),
        ci_active=True,
    )
    indices = [4, 1, 3]

    vectorized = dataset.__getitems__(indices)[0]
    legacy = default_collate([dataset[index] for index in indices])[0]

    for name in (
        "images",
        "observed_images",
        "measured_intensity",
        "coords_relative",
    ):
        torch.testing.assert_close(vectorized[name], legacy[name], msg=name)
        assert vectorized[name].stride() == legacy[name].stride(), name


class _WorkerTaggedPtychoDataset(PtychoDataset):
    def __getitem__(self, index):
        fields, probe, scaling = super().__getitem__(index)
        fields = fields.clone()
        fields["fetch_pid"] = torch.full(
            fields.batch_size,
            os.getpid(),
            dtype=torch.int64,
        )
        return fields, probe, scaling

    def __getitems__(self, indices):
        return self[indices]


def test_mmap_num_workers_fetches_in_a_worker_process(tmp_path):
    from ptycho_torch.workflows.components import _build_lightning_dataloaders

    public, payload = _configs(num_workers=1)
    dataset = _dataset(tmp_path, payload)
    dataset.__class__ = _WorkerTaggedPtychoDataset
    loader, _ = _build_lightning_dataloaders(
        dataset,
        None,
        public,
        payload=payload,
        torch_training_seed=5,
    )

    fields, _probe, _scaling = next(iter(loader))
    assert set(fields["fetch_pid"].tolist()) != {os.getpid()}


@pytest.mark.integration
def test_real_lightning_cpu_ddp_shards_train_and_uses_explicit_validation_map(
    tmp_path,
):
    script = tmp_path / "ddp_batch_rail.py"
    output = tmp_path / "evidence"
    script.write_text(
        textwrap.dedent(
            """
            import json
            import os
            from pathlib import Path
            import sys
            from types import SimpleNamespace

            import lightning.pytorch as L
            import numpy as np
            import torch

            sys.path.insert(0, sys.argv[2])
            from ptycho.config.config import ModelConfig as PublicModelConfig
            from ptycho.config.config import PyTorchExecutionConfig
            from ptycho.config.config import TrainingConfig as PublicTrainingConfig
            from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
            from ptycho_torch.dataloader import PtychoDataset
            from ptycho_torch.workflows.components import _build_lightning_dataloaders


            N = 64


            def write_source(path, ids):
                path.mkdir(parents=True, exist_ok=True)
                images = np.zeros((len(ids), N, N), dtype=np.float32)
                for row, sample_id in enumerate(ids):
                    images[row].flat[sample_id] = 1.0
                coords = np.arange(len(ids), dtype=np.float64)
                np.savez(
                    path / "data.npz",
                    diff3d=images,
                    xcoords=coords,
                    ycoords=coords,
                    probeGuess=np.ones((N, N), dtype=np.complex64),
                    objectGuess=np.ones((N, N), dtype=np.complex64),
                )


            def configs():
                data = DataConfig(
                    N=N, neighbor_count=1, n_raw_frames_selected=1, gridsize=1,
                    x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0),
                    scale_contract_version="legacy_v1",
                    measurement_domain="normalized_amplitude",
                )
                model = ModelConfig(object_big=False)
                training = TrainingConfig(
                    batch_size=2, device="cpu", strategy="ddp", num_workers=0,
                    orchestrator="Mlflow",
                )
                public = PublicTrainingConfig(
                    model=PublicModelConfig(N=N, gridsize=1),
                    batch_size=2,
                    sequential_sampling=False,
                    backend="pytorch",
                )
                payload = SimpleNamespace(
                    tf_training_config=public,
                    pt_data_config=data,
                    pt_model_config=model,
                    pt_training_config=training,
                    execution_config=PyTorchExecutionConfig(
                        accelerator="cpu", devices=2, strategy="ddp",
                        num_workers=0, pin_memory=False, persistent_workers=False,
                        logger_backend=None, enable_checkpointing=False,
                    ),
                )
                return public, payload


            class Recorder(L.LightningModule):
                def __init__(self, output):
                    super().__init__()
                    self.weight = torch.nn.Parameter(torch.tensor(0.0))
                    self.output = Path(output)
                    self.train_ids = []
                    self.val_ids = []

                @staticmethod
                def ids(batch):
                    images = batch[0]["images"]
                    return images.flatten(1).argmax(dim=1).tolist()

                def training_step(self, batch, _batch_idx):
                    self.train_ids.extend(self.ids(batch))
                    return self.weight * 0.0

                def validation_step(self, batch, _batch_idx):
                    self.val_ids.extend(self.ids(batch))

                def configure_optimizers(self):
                    return torch.optim.SGD(self.parameters(), lr=0.0)

                def on_fit_end(self):
                    self.output.mkdir(parents=True, exist_ok=True)
                    (self.output / f"rank-{self.global_rank}.json").write_text(
                        json.dumps({"train": self.train_ids, "val": self.val_ids})
                    )


            def main():
                root = Path(sys.argv[1])
                train_map = root / "train-map" / "tensors"
                val_map = root / "val-map" / "tensors"
                public, payload = configs()
                if "LOCAL_RANK" not in os.environ:
                    write_source(root / "train-source", range(8))
                    write_source(root / "val-source", range(100, 104))
                    PtychoDataset(
                        str(root / "train-source"), payload.pt_model_config,
                        payload.pt_data_config, payload.pt_training_config,
                        data_dir=str(train_map), remake_map=True,
                    )
                    PtychoDataset(
                        str(root / "val-source"), payload.pt_model_config,
                        payload.pt_data_config, payload.pt_training_config,
                        data_dir=str(val_map), remake_map=True,
                    )
                train = PtychoDataset.from_existing_map(
                    train_map, payload.pt_model_config, payload.pt_data_config
                )
                val = PtychoDataset.from_existing_map(
                    val_map, payload.pt_model_config, payload.pt_data_config
                )
                datamodule = _build_lightning_dataloaders(
                    train, val, public, payload=payload, torch_training_seed=7
                )
                trainer = L.Trainer(
                    accelerator="cpu", devices=2, strategy="ddp", max_epochs=1,
                    logger=False, enable_checkpointing=False,
                    enable_progress_bar=False, num_sanity_val_steps=0,
                    use_distributed_sampler=True,
                )
                trainer.fit(Recorder(root / "evidence"), datamodule=datamodule)


            if __name__ == "__main__":
                main()
            """
        )
    )
    source_root = Path(__file__).parents[2]
    env = os.environ.copy()
    for name in (
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
        "NODE_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        env.pop(name, None)
    completed = subprocess.run(
        [sys.executable, str(script), str(tmp_path), str(source_root)],
        cwd=source_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    records = [
        json.loads((output / f"rank-{rank}.json").read_text())
        for rank in range(2)
    ]
    train_by_rank = [record["train"] for record in records]
    val_by_rank = [record["val"] for record in records]
    assert set(train_by_rank[0]).isdisjoint(train_by_rank[1])
    assert sorted(train_by_rank[0] + train_by_rank[1]) == list(range(8))
    assert set(val_by_rank[0]).isdisjoint(val_by_rank[1])
    assert sorted(val_by_rank[0] + val_by_rank[1]) == list(range(100, 104))
