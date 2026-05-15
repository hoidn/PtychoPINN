"""Barebones training script for PC-CCNF with indexed dataloader.

Usage:
    python -m ptycho_torch.beta_modules.train_ccnf \
        --ptycho_dir data/pinn_velo_ic_2 \
        --config ptycho_torch/configs/testing_configs/ccnf_indexed.json \
        --output_dir /tmp/ccnf_run
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from ptycho_torch.config_params import (
    DataConfig, ModelConfig, TrainingConfig, InferenceConfig,
    update_existing_config,
)
from ptycho_torch.utils import load_config_from_json, validate_and_process_config, auto_set_num_datasets
from ptycho_torch.model import PtychoPINN_Lightning
from ptycho_torch.beta_modules.dataloader_index import PtychoDatasetIndexed
from ptycho_torch.dataloader import TensorDictDataLoader, Collate_Lightning
from ptycho_torch.train_utils import (
    set_seed, get_training_strategy, find_learning_rate,
    is_effectively_global_rank_zero, LightningConfigSaveCallback,
    resolve_n_devices,
)


class IndexedDataModule(L.LightningDataModule):
    """Lightning DataModule wrapping PtychoDatasetIndexed for DDP training.

    DDP synchronization uses Lightning's prepare_data/setup protocol:
    - prepare_data() runs on rank 0 only and creates the memory map
    - Lightning provides a barrier between prepare_data and setup
    - setup() runs on all ranks and loads the existing memory map

    This is safe for both local DDP (Lightning-managed) and torchrun launches.
    """

    def __init__(self, ptycho_dir, model_config, data_config, training_config,
                 val_split=0.05, val_seed=42, data_dir='data/memmap_indexed',
                 min_overlap_frac=0.25, temperature=0.5,
                 K_candidates=30, aspect_range=(0.7, 1.3)):
        super().__init__()
        self.ptycho_dir = ptycho_dir
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config
        self.val_split = val_split
        self.val_seed = val_seed
        self.data_dir = data_dir
        self.min_overlap_frac = min_overlap_frac
        self.temperature = temperature
        self.K_candidates = K_candidates
        self.aspect_range = aspect_range
        self._is_setup_done = False

    def prepare_data(self):
        """Rank 0 only: create memory map on shared filesystem."""
        self.training_config.orchestrator = "Lightning"
        PtychoDatasetIndexed(
            ptycho_dir=self.ptycho_dir,
            model_config=self.model_config,
            data_config=self.data_config,
            training_config=self.training_config,
            data_dir=self.data_dir,
            remake_map=True,
            min_overlap_frac=self.min_overlap_frac,
            temperature=self.temperature,
            K_candidates=self.K_candidates,
            aspect_range=self.aspect_range,
        )

    def setup(self, stage=None):
        if self._is_setup_done:
            return
        if stage == "fit" or stage is None:
            self.training_config.orchestrator = "Lightning"

            full_dataset = PtychoDatasetIndexed(
                ptycho_dir=self.ptycho_dir,
                model_config=self.model_config,
                data_config=self.data_config,
                training_config=self.training_config,
                data_dir=self.data_dir,
                remake_map=False,
                min_overlap_frac=self.min_overlap_frac,
                temperature=self.temperature,
                K_candidates=self.K_candidates,
                aspect_range=self.aspect_range,
            )

            dataset_size = len(full_dataset)
            val_size = int(self.val_split * dataset_size)
            train_size = dataset_size - val_size
            print(f"Dataset split: Total={dataset_size}, Train={train_size}, Val={val_size}")

            generator = torch.Generator().manual_seed(self.val_seed)
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size], generator=generator
            )
        self._is_setup_done = True

    def train_dataloader(self):
        return TensorDictDataLoader(
            self.train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=self.training_config.num_workers,
            collate_fn=Collate_Lightning(pin_memory_if_cuda=True),
            pin_memory=True,
            persistent_workers=True if self.training_config.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return TensorDictDataLoader(
            self.val_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            num_workers=self.training_config.num_workers,
            collate_fn=Collate_Lightning(pin_memory_if_cuda=True),
            pin_memory=True,
            persistent_workers=True if self.training_config.num_workers > 0 else False,
        )


def main(ptycho_dir, config_path, output_dir):
    """Train PC-CCNF model with indexed dataloader."""
    print("Loading configs...")
    config_data = load_config_from_json(config_path)
    d_replace, m_replace, t_replace, i_replace, _ = validate_and_process_config(config_data)

    data_config = DataConfig()
    if d_replace:
        update_existing_config(data_config, d_replace)

    model_config = ModelConfig()
    if m_replace:
        update_existing_config(model_config, m_replace)

    training_config = TrainingConfig()
    if t_replace:
        update_existing_config(training_config, t_replace)

    inference_config = InferenceConfig()
    if i_replace:
        update_existing_config(inference_config, i_replace)

    auto_set_num_datasets(model_config, ptycho_dir)

    resolve_n_devices(training_config)
    set_seed(42, n_devices=training_config.n_devices)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"ccnf_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Mmap path must be deterministic across DDP ranks. In subprocess DDP
    # each rank re-runs main() independently, so datetime.now() can differ.
    # Use output_dir (from CLI args, identical across ranks) for the mmap.
    mmap_dir = os.path.join(output_dir, "memmap_indexed")

    print(f"Architecture: {getattr(model_config, 'architecture', 'unet')}")
    print(f"Encoder type: {getattr(model_config, 'encoder_type', 'cnn')}")
    print(f"Decoder type: {getattr(model_config, 'ccnf_decoder_type', 'neural_field')}")
    print(f"C={data_config.C}, N={data_config.N}, devices={training_config.n_devices}")

    data_module = IndexedDataModule(
        ptycho_dir=ptycho_dir,
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        data_dir=mmap_dir,
    )

    model = PtychoPINN_Lightning(model_config, data_config, training_config, inference_config)
    model.training = True

    updated_lr = find_learning_rate(
        training_config.learning_rate,
        training_config.n_devices,
        training_config.batch_size,
    )
    model.lr = updated_lr
    print(f"Learning rate: {updated_lr:.6f}")

    val_loss_label = model.val_loss_name

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(run_dir, "checkpoints"),
        monitor=val_loss_label,
        mode="min",
        save_top_k=1,
        filename="best-checkpoint",
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor=val_loss_label,
        mode="min",
        patience=50,
        verbose=True,
    )

    config_dict = {
        "data_config": data_config,
        "model_config": model_config,
        "training_config": training_config,
        "inference_config": inference_config,
    }
    config_save_callback = LightningConfigSaveCallback(
        config_map=config_dict,
        base_output_dir=run_dir,
    )

    trainer = L.Trainer(
        max_epochs=training_config.epochs,
        default_root_dir=run_dir,
        devices=training_config.n_devices,
        accelerator="gpu" if training_config.n_devices > 0 and torch.cuda.is_available() else "cpu",
        strategy=get_training_strategy(training_config.n_devices, training_config.strategy),
        callbacks=[checkpoint_callback, early_stop_callback, config_save_callback],
        enable_checkpointing=True,
        enable_progress_bar=True,
        logger=False,
    )

    print(f"Starting training for {training_config.epochs} epochs...")
    trainer.fit(model, datamodule=data_module)

    if dist.is_initialized():
        dist.barrier()

    if is_effectively_global_rank_zero():
        print(f"Training complete. Outputs at: {run_dir}")

    return model, trainer, run_dir


def cli_main():
    parser = argparse.ArgumentParser(
        description="Train PC-CCNF with indexed dataloader",
    )
    parser.add_argument("--ptycho_dir", type=str, required=True,
                        help="Directory containing NPZ scan files")
    parser.add_argument("--config", type=str, required=True,
                        help="JSON config file path")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for checkpoints and logs")
    args = parser.parse_args()

    main(args.ptycho_dir, args.config, args.output_dir)


if __name__ == "__main__":
    cli_main()
