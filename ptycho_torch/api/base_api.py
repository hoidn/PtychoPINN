from typing import Optional, Dict, Any, Tuple, Union, Protocol
from enum import Enum
from dataclasses import dataclass, replace
import json
from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig, InferenceConfig, DatagenConfig, update_existing_config
from ptycho_torch.utils import load_all_configs_from_mlflow, load_config_from_json, validate_and_process_config
import os


class ConfigManager:
    """
    Manages config handling. Assumes configs are fully expoed to user

    Can load configs from either:
    1. Previous mlflow training run (which includes all necessary replication artifacts)
    2. Config file containing all replacement configs

    These are both built off the base config classes in ptycho_torch, which start of with default values
    See ptycho_torch.config_params.py for details
    
    """

    def __init__(
            self,
            data_config: Optional[DataConfig],
            model_config: Optional[ModelConfig],
            training_config: Optional[TrainingConfig],
            inference_config: Optional[InferenceConfig],
            datagen_config: Optional[DatagenConfig]):

            self.data_config = data_config or DataConfig()
            self.model_config = model_config or ModelConfig()
            self.training_config = training_config or TrainingConfig()
            self.inference_config = inference_config or InferenceConfig()
            self.datagen_config = datagen_config or DatagenConfig()

    @classmethod
    def _from_mlflow(
          cls,
          run_id: str,
          mlflow_tracking_uri: Optional[str] = None
    ) -> 'ConfigManager':
        """
        Loads configs via artifacts from mflow run (based on run-id)
        """

        try:
            configs = load_all_configs_from_mlflow(run_id, mlflow_tracking_uri)
            print(f"Successfully loaded configuration from run {run_id} from tracking uri {mlflow_tracking_uri}")
        except Exception as e:
            print("Failed to load configs from MlFlow. Defaulting to vanilla...")
            configs = (None, None, None, None, None)

        return cls(*configs)
    
    @classmethod
    def _from_json(
        cls,
        json_path: str
    ) -> 'ConfigManager':
        """
        Loads configs from JSON file
        """
        #Try loading from json, otherwise defaults to vanilla behavior
        json_loaded = False
        try:
            config_data = load_config_from_json(json_path)
            d_cfg, m_cfg, t_cfg, i_cfg, dgen_cfg = validate_and_process_config(config_data)
            json_loaded = True
        except Exception as e:
            d_cfg, m_cfg, t_cfg, i_cfg, dgen_cfg = {}, {}, {}, {}, {} 

        #Create configs and update with JSON values
        data_config = DataConfig()
        model_config = ModelConfig()
        training_config = TrainingConfig()
        datagen_config = DatagenConfig()
        inference_config = InferenceConfig()

        update_existing_config(data_config, d_cfg)       
        update_existing_config(model_config, m_cfg)       
        update_existing_config(training_config, t_cfg)
        update_existing_config(datagen_config, dgen_cfg)        
        update_existing_config(inference_config, i_cfg)

        return cls(data_config, model_config, training_config, inference_config, datagen_config), json_loaded
    
    @classmethod
    def _from_lightning_json(
        cls,
        json_base_path: str
    ) -> 'ConfigManager':
        
        from ptycho_torch.api.api_helper import load_configs_from_local_dir

        try:
            configs = load_configs_from_local_dir(json_base_path)
        except Exception as e:
            print("Failed to load default configs, using empty defaults instead.")
            configs = (None, None, None, None, None)
        
        return cls(*configs)

    @classmethod
    def _flexible_load(
         cls,
         run_id: Optional[str],
         json_path: str,
         mlflow_tracking_uri: Optional[str] = None
    ) -> 'ConfigManager':
         """
         Optionally load from mlflow config first before overriding with JSON in relevant fields
         """

         #Start with mlflow
         mlflow_manager = cls._from_mlflow(run_id, mlflow_tracking_uri)

         #Return setup with json
         json_manager, json_loaded = cls._from_json(json_path)

         #Apply overrides
         if json_loaded:
            update_existing_config(mlflow_manager.data_config, json_manager.data_config)
            update_existing_config(mlflow_manager.model_config, json_manager.model_config)
            update_existing_config(mlflow_manager.training_config, json_manager.training_config)
            update_existing_config(mlflow_manager.inference_config, json_manager.inference_config)
            update_existing_config(mlflow_manager.datagen_config, json_manager.datagen_config)

         return mlflow_manager

    def update(
        self,
        data_config: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        inference_config: Optional[Dict[str, Any]] = None,
        datagen_config: Optional[Dict[str, Any]] = None,
    ):
        "Manually update config fields, exposing this to the user"
        if data_config:
            update_existing_config(self.data_config, data_config)
        if model_config:
            update_existing_config(self.model_config, model_config)
        if training_config:
            update_existing_config(self.training_config, training_config)
        if inference_config:
            update_existing_config(self.inference_config, inference_config)
        if datagen_config:
            update_existing_config(self.datagen_config, datagen_config)

    def to_tuple(self) -> Tuple[DataConfig, ModelConfig, TrainingConfig, InferenceConfig, DatagenConfig]:
        """
        Export as tuple for compatibility with instantiation code
        """
        return (
            self.data_config,
            self.model_config,
            self.training_config,
            self.inference_config,
            self.datagen_config
        )
    @staticmethod
    def _parse_config(config: Optional[Union[Any, Dict]], ConfigClass):
        """
        Parses a single config to ensure that the format abides by 
        """
        if config is None:
            return ConfigClass()
        elif isinstance(config, dict):
            instance = ConfigClass()
            update_existing_config(instance, config)
            return instance
        else:
            return config
    @classmethod
    def from_configs(
        cls,
        model_config: Optional[Union[ModelConfig,Dict]] = None,
        data_config: Optional[Union[ModelConfig,Dict]] = None,
        training_config: Optional[Union[ModelConfig,Dict]] = None,
        inference_config: Optional[Union[ModelConfig,Dict]] = None,
        datagen_config: Optional[Union[ModelConfig,Dict]] = None
    ) -> 'ConfigManager':
        """
        Creates a ConfigManager from individual config objects or dicts.
        Provides flexibility for users who want to specify configs individually.
        """
        return cls(
            data_config=cls._parse_config(data_config, DataConfig),
            model_config=cls._parse_config(model_config, ModelConfig),
            training_config=cls._parse_config(training_config, TrainingConfig),
            inference_config=cls._parse_config(inference_config, InferenceConfig),
            datagen_config=cls._parse_config(datagen_config, DatagenConfig)
        )

from ptycho_torch.train_utils import PtychoDataModule
from typing import Iterator
import torch

class DataLoaderProtocol(Protocol):
    """
    Generic protocol for dataloader
    """
    def __iter__(self) -> Iterator[Tuple[Any, torch.Tensor, float]]:
        #Specifically outputs in a style that works with my current implementation
        #Tuple is: (Tensordict, probe tensors, scaling constant)
        ...
    def __len__(self) -> int:
        ...

class DataloaderFormats(Enum):
    """
    Different dataloader approaches depending on desired outcome
    
    Lightning_module is used for training with the datamodule from pytorch lightning
    Tensordict is the custom dataloader used in Albert's publication, used for inference
    """
    LIGHTNING_MODULE = 'lightning_module' #For Mlflow + lightning
    LIGHTNING_ONLY_MODULE = 'lightning_only_module' #For lightning-only
    TENSORDICT = 'tensordict'
    DATALOADER = 'pytorch'

class PtychoDataLoader:
    """
    Wrapper around custom PtychoDataLoader as well as data module.
    """

    def __init__(
        self,
        data_dir: str,
        config_manager: Optional[ConfigManager] = None,
        data_config: Optional[Union[DataConfig, Dict[str, Any]]] = None,
        model_config: Optional[Union[ModelConfig, Dict[str, Any]]] = None,
        training_config: Optional[Union[TrainingConfig, Dict[str, Any]]] = None,
        data_format: Union[DataloaderFormats, str] = DataloaderFormats.LIGHTNING_MODULE,
        output_dir: Optional[str] = None,
        timestamp: Optional[str] = None,
        memory_map_dir: Optional[str] = 'data/memmap'
    ):
        """
        Args:
            data_dir: Single directory with all training files
        """
        if config_manager is not None:
            self.data_config = config_manager.data_config
            self.model_config = config_manager.model_config
            self.training_config = config_manager.training_config
        else:
            self.data_config = ConfigManager._parse_config(data_config, DataConfig)
            self.model_config = ConfigManager._parse_config(model_config, ModelConfig)
            self.training_config = ConfigManager._parse_config(training_config, TrainingConfig)
        
        self.data_dir = data_dir
        self.data_format = data_format
        self.memory_map_dir = memory_map_dir

        #Can expose these later if need be, or integrate into training configs
        self.val_seed = 42
        self.val_split = 0.05

        if isinstance(data_format, str):
            data_format = DataloaderFormats(data_format)

        if data_format == DataloaderFormats.LIGHTNING_MODULE:
            self._setup_lightning_datamodule()
        elif data_format == DataloaderFormats.LIGHTNING_ONLY_MODULE:
            self._setup_lightning_only_datamodule(output_dir, timestamp = timestamp)
        elif data_format == DataloaderFormats.TENSORDICT:
            self._setup_tensordict_dataloader()
        elif data_format == DataloaderFormats.DATALOADER:
            pass # Not implemented yet

    def _setup_lightning_datamodule(self):
        """
        Initialize PtychoDataModule for training specifically(flexible for multi-instanced gpu training)
        Specifically needs comptability with lightning
        Needs more complex handling 
        """
        self.data_module = PtychoDataModule(
            self.data_dir,
            self.model_config,
            self.data_config,
            self.training_config,
            initial_remake_map = True,
            val_split=0.05,
            val_seed=self.val_seed,
            memory_map_dir=self.memory_map_dir
        )

        print("Set up lightning datamodule")
    
    def _setup_lightning_only_datamodule(self, output_dir, timestamp = None):
        """
        Initialize PtychoDataModule for training specifically(flexible for multi-instanced gpu training)
        Specifically needs comptability with lightning
        Needs more complex handling 
        """
        from datetime import datetime
        from ptycho_torch.train_utils import is_effectively_global_rank_zero

        # Use provided timestamp or generate new one
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Define the root run directory
        self.output_dir = os.path.join(output_dir, f"run_{timestamp}")

        self.data_module = PtychoDataModule(
            self.data_dir,
            self.model_config,
            self.data_config,
            self.training_config,
            initial_remake_map = True,
            val_split=0.05,
            val_seed=self.val_seed,
            memory_map_dir=self.memory_map_dir
        )

        print("Set up lightning datamodule")

    def _setup_tensordict_dataloader(self):
        from ptycho_torch.dataloader import PtychoDataset, TensorDictDataLoader, Collate_Lightning

        self.dataset = PtychoDataset(
                        ptycho_dir=self.data_dir,
                        model_config=self.model_config,
                        data_config=self.data_config,
                        training_config = self.training_config,
                        data_dir = self.memory_map_dir,
                        remake_map= True
                    )
        dataset_size = len(self.dataset)
        val_size = int(self.val_split * dataset_size)
        train_size = dataset_size - val_size
        #Seeding for dataloader
        generator = torch.Generator().manual_seed(self.val_seed)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size], generator=generator
        )

        print("Setup ptycho dataset")

        #Note, collate function is currently integrated with pytorch lightning for multi gpu inference.
        self.dataloader = TensorDictDataLoader(
                self.train_dataset,
                batch_size=self.training_config.batch_size,
                shuffle=True,
                num_workers=self.training_config.num_workers,
                collate_fn=Collate_Lightning(pin_memory_if_cuda=True), # Lightning handles device placement
                pin_memory=True, # Lightning DDP often benefits from this
                persistent_workers=True,
                prefetch_factor = 4,
            )
        
        print("Setup ptycho dataloader")
        

    def module_train_dataloader(self):
        """Training dataloader"""
        return self.data_module.train_dataloader()
    
    def module_val_dataloader(self):
        """Validation dataloader"""
        return self.data_module.val_dataloader()

    def __iter__(self):
        return iter(self.train_dataloader())
    
    def __len__(self):
        return len(self.data_module.train_dataloader())

from torch.nn import Module
import mlflow
from mlflow.tracking import MlflowClient
import shutil
from pathlib import Path

class Orchestration(Enum):
    """
    Enumeration of supported orchestration strategies
    Used specifically for dataloading and training
    """
    MLFLOW = "mlflow"
    PYTORCH = "pytorch"
    LIGHTNING = "lightning"

class PtychoModel:
    """
    Wrapper around generic module with save/load utils
    """
    def __init__(
        self,
        model_config: Optional[Union[ModelConfig,Dict]] = None,
        data_config: Optional[Union[ModelConfig,Dict]] = None,
        training_config: Optional[Union[ModelConfig,Dict]] = None,
        inference_config: Optional[Union[ModelConfig,Dict]] = None
    ):
        """
        Initializes with just settings, will load model with methods later (since they're setting-dependent)
        Model is generic, can handle PtychoPINN_Lightning (for PtychoPINN_torch).
        Parse config is called here in case someone wants to instantiate the model without class methods
        """

        self.data_config = ConfigManager._parse_config(data_config, DataConfig)
        self.model_config = ConfigManager._parse_config(model_config, ModelConfig)
        self.training_config = ConfigManager._parse_config(training_config, TrainingConfig)
        self.inference_config = ConfigManager._parse_config(inference_config, InferenceConfig)

    def save(
        self,
        path: str,
        strategy: Union[Orchestration, str] = Orchestration.MLFLOW,
        **kwargs
    ) -> str:
        """
        Generic save method that delegates to specific strategy.
        
        Args:
            path: Destination path (interpretation depends on strategy)
            strategy: Save strategy to use
            **kwargs: Strategy-specific arguments
            
        Returns:
            Path or URI where model was saved
        """
        if isinstance(strategy, str):
            strategy = Orchestration(strategy)
        
        if strategy == Orchestration.MLFLOW:
            return self.save_mlflow(path, **kwargs)
        elif strategy == Orchestration.PYTORCH:
            return self.save_pytorch(path, **kwargs)
        elif strategy == Orchestration.LIGHTNING:
            return self.save_lightning(path, **kwargs)
        else:
            raise ValueError(f"Unknown save strategy: {strategy}")
    
    @classmethod
    def _load(
        cls,
        config_manager: Optional[ConfigManager] = None,
        model_config: Optional[Union[ModelConfig,Dict]] = None,
        data_config: Optional[Union[ModelConfig,Dict]] = None,
        training_config: Optional[Union[ModelConfig,Dict]] = None,
        inference_config: Optional[Union[ModelConfig,Dict]] = None,
        strategy: Union[Orchestration, str] = Orchestration.MLFLOW,
        **kwargs
    ) -> 'PtychoModel':
        """
        Generic load method that delegates to specific strategy.
        
        Args:
            path: Source path (interpretation depends on strategy)
            strategy: Load strategy to use
            model: Model class to instantiate
            **kwargs: Strategy-specific arguments
            
        Returns:
            Loaded PtychoModel instance
        """
        if config_manager is None:
            config_manager = ConfigManager.from_configs(
                model_config = model_config,
                data_config = data_config,
                training_config = training_config,
                inference_config = inference_config
            )
        
        if isinstance(strategy, str):
            strategy = Orchestration(strategy)

        instance = cls(model_config = config_manager.model_config,
                       data_config = config_manager.data_config,
                       training_config = config_manager.training_config,
                       inference_config = config_manager.inference_config)
        
        if strategy == Orchestration.MLFLOW:
            instance.model, instance.run_id = cls.load_from_mlflow(**kwargs)
            instance.mlflow_tracking_uri = kwargs.get('mlflow_tracking_uri')
        elif strategy == Orchestration.PYTORCH:
            instance.model = cls.load_from_pytorch(**kwargs)
        elif strategy == Orchestration.LIGHTNING:
            print("Loading lightning model")
            instance.model = cls.load_from_lightning(data_config = config_manager.data_config,
                                                     model_config = config_manager.model_config,
                                                     training_config = config_manager.training_config,
                                                     inference_config = config_manager.inference_config,
                                                     **kwargs,
                                                     )
        else:
            raise ValueError(f"Unknown load strategy: {strategy}")
        
        return instance

    @classmethod    
    def _new_model(
        cls,
        model=None,
        config_manager: Optional[ConfigManager] = None,
        model_config: Optional[Union[ModelConfig, Dict]] = None,
        data_config: Optional[Union[DataConfig, Dict]] = None,
        training_config: Optional[Union[TrainingConfig, Dict]] = None,
        inference_config: Optional[Union[InferenceConfig, Dict]] = None
    ) -> 'PtychoModel': 
        """
        Creates PtychoModel class with new specified model
        """
        if config_manager is None:
            config_manager = ConfigManager.from_configs(
                model_config=model_config,
                data_config=data_config,
                training_config=training_config,
                inference_config=inference_config
            )

        model_instance = model(
            config_manager.model_config,
            config_manager.data_config,
            config_manager.training_config,
            config_manager.inference_config
        )

        instance = cls(
            model_config=config_manager.model_config,
            data_config=config_manager.data_config,
            training_config=config_manager.training_config,
            inference_config=config_manager.inference_config
        )
        
        instance.model = model_instance
        
        return instance
    
    @staticmethod
    def load_from_mlflow(
        run_id: str = None,
        mlflow_tracking_uri: Optional[str] = None
    ) -> 'PtychoModel':
        """
        Loads model from mlflow run. Requires underlying model to be integrated with mlflow infra
        """       
        #Getting mlflow to read the correc path
        tracking_uri = f"file:{os.path.abspath(mlflow_tracking_uri)}"
        mlflow.set_tracking_uri(tracking_uri)
        model_uri = f"runs:/{run_id}/model"

        model = mlflow.pytorch.load_model(model_uri)
        run_id = run_id

        return model, run_id
    
    @staticmethod
    def load_from_pytorch(
        path: str = None,
        config_path: Optional[str] = None
    ) -> 'PtychoModel':
        pass #TBD
    
    def load_from_checkpoint(
        checkpoint_path: str,
        model: Optional[Any] = None,
        config_path: Optional[str] = None
    ):
        pass

    @staticmethod
    def load_from_lightning(
        data_config: DataConfig,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        inference_config: InferenceConfig,
        run_path: str,
        model_class: Any
    ) -> Any:
        """
        Loads model from Lightning checkpoint.
        
        Args:
            run_path: Full path to run directory (e.g., 'base_dir/run_20240115_143022/')
            model_class: Lightning module class to instantiate
            
        Returns:
            Loaded Lightning model
            
        Raises:
            FileNotFoundError: If checkpoint not found
        """
        from pathlib import Path
        
        run_path = Path(run_path)
        checkpoint_path = run_path / "checkpoints" / "best-checkpoint.ckpt"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path}. "
                f"Expected structure: {run_path}/checkpoints/best-checkpoint.ckpt"
            )
        
        print(f"Loading model from {checkpoint_path}")
        model = model_class.load_from_checkpoint(str(checkpoint_path),
                                                 model_config = model_config,
                                                 data_config = data_config,
                                                 training_config = training_config,
                                                 inference_config = inference_config)
        
        return model
    
    def save_mlflow(
        self,
        destination_path: str,
        current_mlflow_tracking_uri: Optional[str] = None
    ):
        """
        Copies MLflow run while preserving both experiment ID and run ID.
        Works across storage backends.
        """
        import tempfile
        import yaml
        
        if self.run_id is None:
            raise ValueError("No run_id associated with this model.")
        
        # Set source
        source_uri = self.mlflow_tracking_uri or current_mlflow_tracking_uri
        if source_uri:
            mlflow.set_tracking_uri(source_uri)
        
        source_client = MlflowClient()
        run = source_client.get_run(self.run_id)
        exp_id = run.info.experiment_id
        
        # Setup destination paths
        dest_path = Path(destination_path)
        dest_mlruns = dest_path / "mlruns"
        dest_exp_dir = dest_mlruns / exp_id
        dest_run_dir = dest_exp_dir / self.run_id
        
        # Create directory structure
        dest_run_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy experiment metadata
        source_exp = source_client.get_experiment(exp_id)
        exp_meta = {
            'experiment_id': exp_id,
            'name': source_exp.name,
            'artifact_location': str(dest_exp_dir / "artifacts"),
            'lifecycle_stage': source_exp.lifecycle_stage,
        }
        
        with open(dest_exp_dir / 'meta.yaml', 'w') as f:
            yaml.dump(exp_meta, f)
        
        # Download artifacts to correct location
        artifacts_dir = dest_run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            source_client.download_artifacts(self.run_id, "", dst_path=str(artifacts_dir))
        except Exception as e:
            print(f"Warning: Could not download artifacts: {e}")
        
        # Save run metadata with preserved IDs
        from mlflow.entities import RunStatus
        
        meta = {
            'artifact_uri': str(artifacts_dir.absolute()),
            'end_time': run.info.end_time,
            'entry_point_name': '',
            'experiment_id': exp_id,
            'lifecycle_stage': 'active',
            'run_id': self.run_id, 
            'run_name': run.data.tags.get('mlflow.runName', ''),
            'run_uuid': self.run_id, 
            'source_name': '',
            'source_type': 4,
            'source_version': '',
            'start_time': run.info.start_time,
            'status': RunStatus.from_string(run.info.status),  # Convert to int
            'tags': [],
            'user_id': run.info.user_id,
        }
        
        with open(dest_run_dir / 'meta.yaml', 'w') as f:
            yaml.dump(meta, f, default_flow_style=False)
        
        # Save params
        params_dir = dest_run_dir / "params"
        params_dir.mkdir(exist_ok=True)
        for key, value in run.data.params.items():
            with open(params_dir / key, 'w') as f:
                f.write(str(value))
        
        # Save metrics with full history
        metrics_dir = dest_run_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        for key in run.data.metrics.keys():
            metric_history = source_client.get_metric_history(self.run_id, key)
            with open(metrics_dir / key, 'w') as f:
                for metric in metric_history:
                    f.write(f"{metric.timestamp} {metric.value} {metric.step}\n")
        
        # Save tags
        tags_dir = dest_run_dir / "tags"
        tags_dir.mkdir(exist_ok=True)
        for key, value in run.data.tags.items():
            safe_key = key.replace('/', '_').replace('\\', '_')
            with open(tags_dir / safe_key, 'w') as f:
                f.write(str(value))
        
        dest_uri = f"file://{dest_mlruns.absolute()}"
        
        print(f"✓ New tracking URI: {dest_uri}")
        
        return dest_uri

    def save_pytorch(self, destination_path: str, checkpoint_path: Optional[str] = None) -> Path:
        """
        Minimal persistence shim for PyTorch backend (Phase R reactivation).

        Emits a Lightning checkpoint + manifest bundle referencing params.cfg so
        ptychodus loaders can track backend provenance per specs/ptychodus_api_spec.md §4.6.

        This is a transitional implementation until full .h5.zip adapter lands in Phase 5.

        Args:
            destination_path: Output directory for the bundle
            checkpoint_path: Optional path to Lightning .ckpt file (if None, searches default locations)

        Returns:
            Path to the created manifest file

        Raises:
            FileNotFoundError: If checkpoint_path not provided and can't find .ckpt in destination
            ValueError: If params.cfg is empty (CONFIG-001 violation)

        Example:
            >>> model = PtychoModel(...)
            >>> manifest_path = model.save_pytorch('outputs/exp001')
            >>> # Creates: outputs/exp001/checkpoint.ckpt + manifest.json

        See also:
            - TensorFlow persistence: ptycho/model_manager.py (save_model, save_multiple_models)
            - Spec contract: specs/ptychodus_api_spec.md §4.6
        """
        import json
        from pathlib import Path
        import ptycho.params as params

        dest = Path(destination_path)
        dest.mkdir(parents=True, exist_ok=True)

        # Validate params.cfg populated (CONFIG-001 gate)
        if not params.cfg:
            raise ValueError(
                "params.cfg is empty. Must call update_legacy_dict(params.cfg, config) "
                "before save_pytorch(). See CONFIG-001 in docs/findings.md."
            )

        # Locate Lightning checkpoint
        if checkpoint_path is None:
            # Search for .ckpt in destination directory
            ckpt_candidates = list(dest.glob('**/*.ckpt'))
            if not ckpt_candidates:
                raise FileNotFoundError(
                    f"No Lightning checkpoint (.ckpt) found in {dest}. "
                    "Provide checkpoint_path explicitly or ensure training saved a checkpoint."
                )
            checkpoint_path = str(ckpt_candidates[0])

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Create manifest bundle
        manifest = {
            'backend': 'pytorch',
            'checkpoint': str(ckpt_path.name),  # Relative to destination_path
            'params_cfg_snapshot': dict(params.cfg),  # Serialize params.cfg for provenance
            'version': '1.0',
            'notes': 'Minimal PyTorch persistence shim (Phase R reactivation)'
        }

        manifest_path = dest / 'pytorch_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)  # default=str handles Path objects

        print(f"✓ PyTorch bundle saved to {dest}")
        print(f"  - Checkpoint: {ckpt_path.name}")
        print(f"  - Manifest: {manifest_path.name}")

        return manifest_path
    
    def save_lightning(
        self,
        destination_base_path: str,
        source_run_path: str
    ) -> Path:
        """
        Copies Lightning training run (checkpoint + configs) to new base directory.
        Preserves the datetime-stamped run folder name.
        
        Args:
            destination_base_path: New base directory (e.g., 'new_experiments/')
            source_run_path: Source run directory (e.g., 'old_base/run_20240115_143022/')
            
        Returns:
            Path to copied run directory
            
        Raises:
            FileNotFoundError: If source directory or required subdirectories don't exist
            
        Example:
            >>> model.save_lightning('new_base/', 'old_base/run_20240115_143022/')
            >>> # Creates: new_base/run_20240115_143022/
        """
        from pathlib import Path
        import shutil
        
        source_path = Path(source_run_path)
        
        # Validate source directory structure
        required_dirs = ['checkpoints', 'configs']
        optional_dirs = ['finetune_checkpoints', 'logs_finetune']
        
        for dir_name in required_dirs:
            if not (source_path / dir_name).exists():
                raise FileNotFoundError(
                    f"Required directory '{dir_name}' not found in {source_path}"
                )
        
        # Extract datetime folder name (e.g., 'run_20240115_143022')
        run_folder_name = source_path.name
        
        # Create destination path preserving datetime folder
        dest_path = Path(destination_base_path) / run_folder_name
        
        # Copy entire directory structure
        if dest_path.exists():
            print(f"Warning: Destination {dest_path} already exists. Overwriting...")
            shutil.rmtree(dest_path)
        
        shutil.copytree(source_path, dest_path)
        
        print(f"✓ Lightning run copied to {dest_path}")
        print(f"  - Checkpoints: {run_folder_name}/checkpoints/")
        print(f"  - Configs: {run_folder_name}/configs/")
        
        # Check for optional fine-tuning artifacts
        if (dest_path / 'finetune_checkpoints').exists():
            print(f"  - Fine-tune checkpoints: {run_folder_name}/finetune_checkpoints/")
        if (dest_path / 'logs_finetune').exists():
            print(f"  - Fine-tune logs: {run_folder_name}/logs_finetune/")
        
        return dest_path

    def save_checkpoint(self, destination_path=str):
        pass

class TrainStrategy(Enum):
    """Enumeration of supported train strategies"""
    LIGHTNING = "lightning"
    PYTORCH = "pytorch" #unsupported at the moment
    TENSORFLOW = "tensorflow" #unsupported at the moment

class Trainer:
    """
    High-level training orchestration
    Supports Lightning Trainer/Mlflow
    """
    def __init__(self,
                 model: PtychoModel,
                 dataloader: PtychoDataLoader,
                 config_manager: Optional[ConfigManager],
                 training_config: Optional[Union[TrainingConfig,Dict]],
                 strategy: TrainStrategy):
        """
        Args:
            model: PtychoModel instance
            dataloader: PtychoDataLoader instance
            config_manager: ConfigManager instance
            training_config: ConfigManager instance
        """
        self.ptycho_model = model
        self.dataloader = dataloader
        self.config_manager = config_manager #Need config manager here to extract datagenconfig
        self.training_config = training_config
        self.strategy = strategy
        self._trainer = None
        

    @classmethod
    def _from_lightning(cls,
                       model: PtychoModel,
                       dataloader: PtychoDataLoader, 
                       orchestration: Orchestration,
                       config_manager: Optional[ConfigManager] = None,
                       training_config: Optional[Union[TrainingConfig, Dict]] = None) -> 'Trainer':
        
        if config_manager is not None:
            parsed_config = config_manager.training_config
        else:
            parsed_config = ConfigManager._parse_config(training_config, TrainingConfig)

        instance = cls(model = model,
                       dataloader = dataloader,
                       config_manager = config_manager,
                       training_config = parsed_config,
                       strategy = TrainStrategy.LIGHTNING)
        
        instance._setup_lightning_trainer(orchestration, dataloader.output_dir)

        return instance
    
    @classmethod
    def _from_pytorch(cls,
                     model: PtychoModel,
                     dataloader: PtychoDataLoader, 
                     config_manager: Optional[ConfigManager] = None,
                     training_config: Optional[Union[TrainingConfig, Dict]] = None) -> 'Trainer':
        pass #TBD
    
    def _setup_lightning_trainer(self, orchestration, output_dir):
        """
        Lightning-specific setup that was copied from my own training code in train.py
        """
        from ptycho_torch.api.trainer_api import setup_lightning_trainer

        self._trainer, self._output_dir = setup_lightning_trainer(self.ptycho_model,
                                                self.config_manager,
                                                orchestration,
                                                output_dir)
        
    def _setup_pytorch_trainer(self):
        """
        Pytorch-specific setup
        """
        pass
    
    def train(self, orchestration: str, experiment_name: str):
        """
        Generic train script that delegates to specific strategy

        Outputs:
            result - Arbitrary dictionary containing useful metadata
        """
        if isinstance(orchestration, str):
            strategy = Orchestration(orchestration)

        if strategy == Orchestration.MLFLOW:
            result = self._train_with_mlflow(experiment_name)
        elif strategy == Orchestration.PYTORCH:
            pass
        elif strategy == Orchestration.LIGHTNING:
            result = self._train_with_lightning(self._output_dir)
        else:
            raise ValueError(f"Unknown load strategy: {strategy}")
        
        return result

    def _train_with_mlflow(self, experiment_name: Optional[str] = None):
        """
        Executes training using MlFlow api
        """
        from ptycho_torch.train_utils import find_learning_rate, is_effectively_global_rank_zero, log_parameters_mlflow, print_auto_logged_info, ModelFineTuner

        exp_name = experiment_name or self.training_config.experiment_name

        # updated_lr = find_learning_rate(
        #     self.training_config.learning_rate,
        #     self.training_config.n_devices,
        #     self.training_config.batch_size
        # )

        updated_lr = self.training_config.learning_rate

        self.ptycho_model.model.lr = updated_lr
        self.ptycho_model.model.training = True

        run_ids = {}
        
        # Run DDP checks and initialize MLflow run if on main process
        if is_effectively_global_rank_zero():
            mlflow.set_experiment(exp_name)
            mlflow.pytorch.autolog(checkpoint_monitor = self.ptycho_model.model.val_loss_name)

            with mlflow.start_run() as run:
                #Log all configs
                log_parameters_mlflow(
                    self.dataloader.data_config, 
                    self.ptycho_model.model_config,
                    self.training_config,
                    self.ptycho_model.inference_config,
                    self.config_manager.datagen_config
                )

                #Set tags
                mlflow.set_tag("stage", "training")
                mlflow.set_tag("encoder_frozen", "False")
                mlflow.set_tag("model_name", self.training_config.model_name)
                if self.training_config.notes:
                    mlflow.set_tag("notes", self.training_config.notes)

                #Train
                print("Beginning model training...")
                self._trainer.fit(
                    self.ptycho_model.model,
                    datamodule = self.dataloader.data_module
                )

                print("Training Complete!!")
                print_auto_logged_info(mlflow.get_run(run_id = run.info.run_id))
                run_ids['training'] = run.info.run_id

        if self.training_config.epochs_fine_tune > 0:
            fine_tuner = ModelFineTuner(
                self.ptycho_model.model,
                self.dataloader.data_module,
                self.training_config
            )
            fine_tuning_run_id = fine_tuner.fine_tune(experiment_name = exp_name)

            if is_effectively_global_rank_zero():
                run_ids['fine_tune'] = fine_tuning_run_id
            
        if is_effectively_global_rank_zero():
            print(f"Training run_id: {run_ids.get('training')}")
            print(f"Fine tune run_id: {run_ids.get('fine_tune')}")

        return run_ids
    
    def _train_with_lightning(self,
                              output_dir = None):
        """
        Lightning trainer has been prepped in previous step before this is called.

        This includes setting up a global directory path for which both the regular and fine-tune training runs
        will be saved. That is passed to this function so the fine tuner knows where to write.

        Returns:
            output_dir - Output directory for all training artifacts, configs included
        """
        
        from ptycho_torch.train_utils import find_learning_rate, ModelFineTuner_Lightning

        updated_lr = find_learning_rate(
            self.training_config.learning_rate,
            self.training_config.n_devices,
            self.training_config.batch_size
        )

        print(f"Updated learning rate is: {updated_lr}")

        self.ptycho_model.model.lr = updated_lr
        self.ptycho_model.model.training = True

        #Regular train
        self._trainer.fit(
                    self.ptycho_model.model,
                    datamodule = self.dataloader.data_module
                )
        
        if self.training_config.epochs_fine_tune > 0:
            print("Beginning fine tuning...")
            # Note: In DDP, all ranks reach this point after trainer.fit finishes.
            fine_tuner = ModelFineTuner_Lightning(
                model=self.ptycho_model.model, 
                train_module=self.dataloader.data_module, 
                training_config=self.training_config,
                run_dir=output_dir
            )
            fine_tuner.fine_tune()

        return output_dir

class Datagen:
    """
    Supports synthetic data generation in easy-to-access way
    Creates new synthetic data directory with provided probes

    Inherently contains a list of probes. Can then be used to simulate objects and then save them via a directory
    """
    def __init__(self,
                 probe_list: Tuple,
                 probe_arg: Dict,
                 datagen_config: DatagenConfig,
                 data_config: DataConfig):
        
        self.datagen_config = datagen_config
        self.data_config = data_config
        self.probe_list = probe_list
        self.probe_arg = probe_arg

    @classmethod
    def _from_npz(cls,
                 npz_path,
                 config_manager: Optional[ConfigManager] = None,
                 datagen_config: Optional[Union[DatagenConfig, Dict]] = None) -> 'Datagen':
        """
        Abstracted class method to generate a new instance of datagen
        """
        from ptycho_torch.datagen.datagen import assemble_precomputed_images

        if config_manager is not None:
            datagen_config = config_manager.datagen_config
            data_config = config_manager.data_config
        elif datagen_config is not None:
            datagen_config = ConfigManager._parse_config(datagen_config,
                                                         DatagenConfig)
            data_config = ConfigManager._parse_config(data_config,
                                                         DataConfig)       
        # Only rank 0 does the actual data generation
        print("Rank 0: Preparing synthetic data...")
        
        # Legacy object argument, unused for now but allows for easy prototyping
        probe_arg = {}
        
        # Assemble probe lists
        exp_probe_list = assemble_precomputed_images(npz_path, 'probe', True)
        probe_list = [item for item in exp_probe_list for _ in range(datagen_config.objects_per_probe)]
        probe_name_idx = [idx for idx in list(range(len(exp_probe_list))) for _ in range(datagen_config.objects_per_probe)]
        probe_arg['probe_name_idx'] = probe_name_idx

        return cls(probe_list, probe_arg,
                   datagen_config, data_config)
    
    def _create_synthetic_objects(self):
        from ptycho_torch.datagen.datagen import simulate_synthetic_objects

        #Dummy argument: Deprecated for now
        obj_arg = {}
        # Try creating synthetic object
        try:
            print(f"Creating objects for class: {self.datagen_config.object_class}")
            image_size = self.datagen_config.image_size
            objects_per_probe = self.datagen_config.objects_per_probe
            object_class = self.datagen_config.object_class
            #Dataconfig currently unused, so just passing vanilla DataConfig class instance
            self.object_list = simulate_synthetic_objects(image_size, DataConfig(), objects_per_probe,
                                                            object_class, obj_arg)
        except:
            raise ValueError("Inputted synthetic object class not valid")
    
    def _generate_simulated_data(self,
                                 synthetic_path):
        
        from ptycho_torch.utils import remove_all_files
        from ptycho_torch.datagen.datagen import simulate_multiple_experiments

        # Remove all existing files from directory
        if not os.path.exists(synthetic_path):
            os.mkdir(synthetic_path)
        else:
            remove_all_files(synthetic_path)

        # Simulate and fill directory
        print("Simulating experiments...")

        self.probe_arg['beamstop_diameter'] = self.datagen_config.beamstop_diameter
        image_size = self.datagen_config.image_size
        diff_per_object = self.datagen_config.diff_per_object

        simulate_multiple_experiments(self.object_list, self.probe_list,
                                      diff_per_object,
                                      image_size, self.data_config, self.probe_arg,
                                      synthetic_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
class InferenceEngine:
    """
    Class designed to perform inference using api-level calls
    """
    def __init__(self,
                 ptycho_model: PtychoModel,
                 config_manager: Optional[ConfigManager] = None,
                 model_config: Optional[Union[ModelConfig, Dict]] = None,
                 data_config: Optional[Union[DataConfig, Dict]] = None,
                 training_config: Optional[Union[TrainingConfig, Dict]] = None,
                 inference_config: Optional[Union[InferenceConfig, Dict]] = None,
                 ):
        """
        Instantiates base class with just configurations. Other methods will set up the data
        """
        
        if config_manager is not None:
            self.data_config = config_manager.data_config
            self.model_config = config_manager.model_config
            self.training_config = config_manager.training_config
            self.inference_config = config_manager.inference_config
        else:
            self.data_config = ConfigManager._parse_config(data_config, DataConfig)
            self.model_config = ConfigManager._parse_config(model_config, ModelConfig)
            self.training_config = ConfigManager._parse_config(training_config, TrainingConfig)
            self.inference_config = ConfigManager._parse_config(inference_config, InferenceConfig)

        self.model = ptycho_model.model

    def predict(self,
                ptycho_dataloader: PtychoDataLoader,
                device = 'cuda'
                ) -> torch.Tensor:
        """
        Runs feed-forward prediction without stitching
        """
        self.model.to(device)
        res = []
        
        with torch.no_grad():
            for batch in ptycho_dataloader.dataloader:
                batch = batch.to(device)
                output = self.model.forward_predict(batch)
                res.append(output.cpu())

        final_output = torch.cat(res, dim=0)
        
        return final_output
    
    def predict_and_stitch(self,
                           ptycho_dataloader: PtychoDataLoader) -> torch.Tensor:
        """
        Calls on predict + stitch methods already in ptychopinn_torch library. Automatically handles multi-gpu inference (if available)
        as well as cpu if gpus not available.

        Should work on most data formats given that they provide positional data and diffraction data

        If there are multiple files in the inference directory that exist in the dataloader, then one must modify
        the "experiment_number" parameter in the inference_configs. This will predict only on a subset of the PtychoDataset class,
        which corresponds to one specific .npz file within the inference directory
        """
        from ptycho_torch.reassembly import reconstruct_image_barycentric

        result, _, _ = reconstruct_image_barycentric(model = self.model,
                                                     ptycho_dset = ptycho_dataloader.dataset,
                                                     training_config = self.training_config,
                                                     data_config = self.data_config,
                                                     model_config = self.model_config,
                                                     inference_config = self.inference_config,
                                                     gpu_ids = None,
                                                     use_mixed_precision=True, verbose = False)

        #Save results
        result_cpu = result.to('cpu')

        return result_cpu
    
#Usage examples

#1.a Instantiate ConfigManager instance, either using json file, mlflow load or existing settings
# config_manager = ConfigManager._flexible_load(run_id run_id) <- load from mlflow
# config_manager = ConfigManager._flexible_load(json_path = '/path/to/config.json') <- load from json
#1.b Update ConfigManager parameters using update function if using user interface to modify any values
# config_manager.update(data_config = new_data_config_dict) <- Example of updating certain parameters in data_config for any keys in the dict

#2.a Instantiate PtychoDataloader instance, using built in class methods depending on dataloader format.

#3. Instantiate PtychoModel instance, using built in class methods depending on model format

#4. Training only: Instantiate trainer based on orchestration
        

        
        






    

        
    




    
    
        







