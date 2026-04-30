"""
Red-then-green tests for Lightning checkpoint hyperparameter serialization.

Per Phase D1c requirements in plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md,
this module validates that PtychoPINN_Lightning checkpoints include serialized
hyperparameters so that load_from_checkpoint() works without explicit config kwargs.

Expected failure mode (RED phase):
- checkpoint['hyper_parameters'] returns None
- load_from_checkpoint raises TypeError for missing 4 positional arguments

Implementation target (GREEN phase):
- self.save_hyperparameters() called in PtychoPINN_Lightning.__init__()
- Config objects serialized as dicts (Path/Tensor fields converted)
- load_from_checkpoint reconstructs module from checkpoint alone
"""

import pytest
import tempfile
from pathlib import Path

# torch-optional imports per POLICY-001
try:
    import torch
    import lightning.pytorch as pl
    from lightning.pytorch import Trainer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from ptycho.config.config import ModelConfig as TFModelConfig, TrainingConfig as TFTrainingConfig, update_legacy_dict
    from ptycho import params as p
    from ptycho_torch.config_params import ModelConfig, DataConfig, TrainingConfig, InferenceConfig
    from ptycho_torch.model import PtychoPINN_Lightning


GENERATOR_CLASS_BY_ARCHITECTURE = {
    "ffno": "FfnoGeneratorModule",
    "fno": "CascadedFNOGenerator",
    "hybrid": "HybridUNOGenerator",
    "stable_hybrid": "StableHybridUNOGenerator",
    "fno_vanilla": "FnoVanillaGeneratorModule",
    "neuralop_uno": "NeuralopUnoGeneratorModule",
    "hybrid_resnet": "HybridResnetGeneratorModule",
    "spectral_resnet_bottleneck_net": "SpectralResnetBottleneckGeneratorModule",
}


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch backend requires torch>=2.2")
class TestLightningCheckpointSerialization:
    """
    Validate Lightning checkpoint hyperparameter persistence contract.

    Tests ensure checkpoints can be loaded without providing explicit config objects,
    satisfying ptychodus API spec §4.6 model persistence requirements.
    """

    @pytest.fixture
    def minimal_config(self, tmp_path):
        """
        Create minimal PyTorch config objects for checkpoint testing.

        Uses deterministic CPU settings with minimal dataset size for fast test execution.
        """
        # Create PyTorch config objects directly
        data_cfg = DataConfig(
            N=64,
            grid_size=(1, 1),  # Minimal gridsize for fast test
        )

        model_cfg = ModelConfig(
            mode='Unsupervised',
            n_filters_scale=1,
        )

        train_cfg = TrainingConfig(
            epochs=0,  # Zero epochs for checkpoint creation only
            batch_size=16,
            learning_rate=1e-4,
        )

        infer_cfg = InferenceConfig()

        # Bridge to legacy params.cfg for CONFIG-001 compliance
        # Create TF config for params.cfg population
        tf_model_cfg = TFModelConfig(
            N=64,
            gridsize=1,
            model_type='pinn',
            amp_activation='silu',
            n_filters_scale=1,
        )
        tf_train_cfg = TFTrainingConfig(
            model=tf_model_cfg,
            train_data_file=Path('dummy_train.npz'),
            test_data_file=None,
            n_groups=16,
            batch_size=16,
            nepochs=0,
            nphotons=1e6,
            neighbor_count=4,
            output_dir=tmp_path,
        )
        update_legacy_dict(p.cfg, tf_train_cfg)

        return model_cfg, data_cfg, train_cfg, infer_cfg

    @pytest.fixture
    def lightning_module(self, minimal_config):
        """
        Instantiate PtychoPINN_Lightning with canonical config objects.

        Returns module ready for checkpoint persistence.
        """
        # Unpack config tuple
        model_cfg, data_cfg, train_cfg, infer_cfg = minimal_config

        # Create Lightning module
        module = PtychoPINN_Lightning(
            model_config=model_cfg,
            data_config=data_cfg,
            training_config=train_cfg,
            inference_config=infer_cfg,
        )

        return module

    def test_checkpoint_contains_hyperparameters(self, lightning_module, tmp_path):
        """
        RED TEST: Assert checkpoint includes 'hyper_parameters' key with config payload.

        Expected to FAIL initially because PtychoPINN_Lightning.__init__() does not call
        self.save_hyperparameters(). After fix, checkpoint should contain four config dicts.
        """
        # Arrange: Create checkpoint via Trainer
        ckpt_path = tmp_path / "test.ckpt"
        trainer = Trainer(
            max_epochs=0,  # No training, just checkpoint creation
            enable_checkpointing=True,
            deterministic=True,
            default_root_dir=tmp_path,
            logger=False,
            enable_progress_bar=False,
            accelerator='cpu',  # Force CPU for test
        )

        # Attach model to trainer by setting internal attribute
        # (Lightning doesn't expose a public setter for .model)
        trainer.strategy._lightning_module = lightning_module

        # Act: Save checkpoint explicitly
        trainer.save_checkpoint(ckpt_path)

        # Load checkpoint to inspect structure
        checkpoint = torch.load(ckpt_path)

        # Assert: hyper_parameters key exists and is not None
        assert 'hyper_parameters' in checkpoint, \
            "Checkpoint missing 'hyper_parameters' key"

        hyper_params = checkpoint['hyper_parameters']
        assert hyper_params is not None, \
            "checkpoint['hyper_parameters'] is None — save_hyperparameters() not called"

        # Assert: Contains expected config keys
        expected_keys = {'model_config', 'data_config', 'training_config', 'inference_config'}
        actual_keys = set(hyper_params.keys())
        assert expected_keys.issubset(actual_keys), \
            f"Missing config keys. Expected {expected_keys}, got {actual_keys}"

    def test_load_from_checkpoint_without_kwargs(self, lightning_module, tmp_path):
        """
        RED TEST: Assert load_from_checkpoint succeeds without explicit config args.

        Expected to FAIL with TypeError: missing 4 required positional arguments
        (model_config, data_config, training_config, inference_config) because
        Lightning cannot reconstruct the module from an empty hyper_parameters dict.

        After fix, Lightning restores configs from checkpoint and instantiates module.
        """
        # Arrange: Create checkpoint
        ckpt_path = tmp_path / "reload_test.ckpt"
        trainer = Trainer(
            max_epochs=0,
            enable_checkpointing=True,
            deterministic=True,
            default_root_dir=tmp_path,
            logger=False,
            enable_progress_bar=False,
            accelerator='cpu',
        )
        trainer.strategy._lightning_module = lightning_module
        trainer.save_checkpoint(ckpt_path)

        # Act: Attempt to load without providing kwargs
        # Expected to fail with TypeError initially
        loaded_module = PtychoPINN_Lightning.load_from_checkpoint(str(ckpt_path))

        # Assert: Successfully loaded (GREEN phase validation)
        assert loaded_module is not None
        assert isinstance(loaded_module, PtychoPINN_Lightning)

        # Validate configs were restored
        assert hasattr(loaded_module, 'hparams')
        assert 'model_config' in loaded_module.hparams
        assert 'data_config' in loaded_module.hparams
        assert 'training_config' in loaded_module.hparams
        assert 'inference_config' in loaded_module.hparams

    def test_checkpoint_configs_are_serializable(self, lightning_module, tmp_path):
        """
        RED TEST: Validate config objects can round-trip through checkpoint serialization.

        Tests that Path objects, enums, and other non-primitive types are correctly
        serialized and deserialized. Expected to pass once save_hyperparameters() fix
        includes proper dataclass→dict conversion.
        """
        # Arrange: Create and reload checkpoint
        ckpt_path = tmp_path / "serialization_test.ckpt"
        trainer = Trainer(
            max_epochs=0,
            enable_checkpointing=True,
            deterministic=True,
            default_root_dir=tmp_path,
            logger=False,
            enable_progress_bar=False,
            accelerator='cpu',
        )
        trainer.strategy._lightning_module = lightning_module
        trainer.save_checkpoint(ckpt_path)

        # Act: Load checkpoint and inspect hyper_parameters
        checkpoint = torch.load(ckpt_path)
        hyper_params = checkpoint['hyper_parameters']

        # Assert: All config values are serializable primitives (dicts, not dataclass instances)
        assert isinstance(hyper_params['model_config'], dict), \
            "model_config should be serialized as dict, not dataclass instance"
        assert isinstance(hyper_params['data_config'], dict), \
            "data_config should be serialized as dict, not dataclass instance"
        assert isinstance(hyper_params['training_config'], dict), \
            "training_config should be serialized as dict, not dataclass instance"
        assert isinstance(hyper_params['inference_config'], dict), \
            "inference_config should be serialized as dict, not dataclass instance"

        # Assert: No Path objects in serialized form (should be strings)
        for cfg_name, cfg_dict in hyper_params.items():
            if isinstance(cfg_dict, dict):
                for key, value in cfg_dict.items():
                    assert not isinstance(value, Path), \
                        f"{cfg_name}.{key} contains Path object {value} — must convert to str"

    @pytest.mark.parametrize(
        ("architecture", "mode"),
        [("ffno", "Supervised")],
    )
    def test_supervised_generator_checkpoint_rebuilds_configured_module(
        self,
        tmp_path,
        architecture,
        mode,
    ):
        """Checkpoint-only reload must restore the configured supervised generator."""
        data_cfg, model_cfg, train_cfg, infer_cfg = self._build_generator_checkpoint_config(
            tmp_path,
            architecture=architecture,
            mode=mode,
        )
        lightning_module = PtychoPINN_Lightning(
            model_config=model_cfg,
            data_config=data_cfg,
            training_config=train_cfg,
            inference_config=infer_cfg,
        )
        ckpt_path = self._save_checkpoint(lightning_module, tmp_path / f"{architecture}_{mode}.ckpt", tmp_path)

        loaded_module = PtychoPINN_Lightning.load_from_checkpoint(str(ckpt_path))

        assert loaded_module.model_config.architecture == architecture
        assert loaded_module.model_config.mode == mode
        assert type(loaded_module.model.autoencoder).__name__ == GENERATOR_CLASS_BY_ARCHITECTURE[architecture]
        assert type(loaded_module.model.autoencoder).__name__ != "Autoencoder"

    @pytest.mark.parametrize(
        "architecture",
        [
            "ffno",
            "fno",
            "hybrid",
            "stable_hybrid",
            "fno_vanilla",
            "neuralop_uno",
            "hybrid_resnet",
            "spectral_resnet_bottleneck_net",
        ],
    )
    def test_generator_architecture_checkpoint_rebuilds_without_manual_injection(
        self,
        tmp_path,
        architecture,
    ):
        """Registered generator architectures must reload from checkpoint alone."""
        data_cfg, model_cfg, train_cfg, infer_cfg = self._build_generator_checkpoint_config(
            tmp_path,
            architecture=architecture,
            mode="Unsupervised",
        )
        lightning_module = PtychoPINN_Lightning(
            model_config=model_cfg,
            data_config=data_cfg,
            training_config=train_cfg,
            inference_config=infer_cfg,
        )
        ckpt_path = self._save_checkpoint(lightning_module, tmp_path / f"{architecture}.ckpt", tmp_path)

        loaded_module = PtychoPINN_Lightning.load_from_checkpoint(str(ckpt_path))

        assert loaded_module.model_config.architecture == architecture
        assert type(loaded_module.model.autoencoder).__name__ == GENERATOR_CLASS_BY_ARCHITECTURE[architecture]

    @staticmethod
    def _save_checkpoint(lightning_module, ckpt_path, root_dir):
        trainer = Trainer(
            max_epochs=0,
            enable_checkpointing=True,
            deterministic=True,
            default_root_dir=root_dir,
            logger=False,
            enable_progress_bar=False,
            accelerator='cpu',
        )
        trainer.strategy._lightning_module = lightning_module
        trainer.save_checkpoint(ckpt_path)
        return ckpt_path

    @staticmethod
    def _build_generator_checkpoint_config(tmp_path, *, architecture, mode):
        image_size = 128 if architecture == "neuralop_uno" else 64
        data_cfg = DataConfig(
            N=image_size,
            C=1,
            grid_size=(1, 1),
        )
        model_cfg = ModelConfig(
            mode=mode,
            architecture=architecture,
            generator_output_mode="real_imag",
            probe_mask=False,
            object_big=False,
            probe_big=False,
            loss_function="MAE" if mode == "Supervised" else "Poisson",
        )
        train_cfg = TrainingConfig(
            epochs=0,
            batch_size=2,
            learning_rate=1e-4,
            torch_loss_mode="mae" if mode == "Supervised" else "poisson",
            output_dir=str(tmp_path),
        )
        infer_cfg = InferenceConfig()

        tf_model_cfg = TFModelConfig(
            N=image_size,
            gridsize=1,
            model_type='supervised' if mode == "Supervised" else 'pinn',
            amp_activation='silu',
            n_filters_scale=1,
            architecture=architecture,
        )
        tf_train_cfg = TFTrainingConfig(
            model=tf_model_cfg,
            train_data_file=Path('dummy_train.npz'),
            test_data_file=None,
            n_groups=2,
            batch_size=2,
            nepochs=0,
            nphotons=1e6,
            neighbor_count=4,
            output_dir=tmp_path,
            torch_loss_mode="mae" if mode == "Supervised" else "poisson",
        )
        update_legacy_dict(p.cfg, tf_train_cfg)

        return data_cfg, model_cfg, train_cfg, infer_cfg
