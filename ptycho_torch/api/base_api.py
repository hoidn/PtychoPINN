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
        from ptycho_torch.api.api_helper import config_from_json

        config_data, json_loaded = config_from_json(json_path)
        
        return cls(**config_data), json_loaded

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
         from ptycho_torch.api.api_helper import update_manager_with_json

         #Start with mlflow
         mlflow_manager = cls._from_mlflow(run_id, mlflow_tracking_uri)

         #Return setup with json
         json_manager, json_loaded = cls._from_json(json_path)

         #Apply overrides
         update_manager_with_json(mlflow_manager = mlflow_manager,
                                  json_loaded = json_loaded,
                                  json_manager = json_manager)

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
    LIGHTNING_MODULE = 'lightning_module'
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
        from ptycho_torch.api.api_helper import setup_lightning_datamodule

        setup_lightning_datamodule(self)

    def _setup_tensordict_dataloader(self):
        
        from ptycho_torch.api.api_helper import setup_tensordict_dataloader

        setup_tensordict_dataloader(self)

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

class PtychoModel:
    """
    Wrapper around generic module with save/load utils
    """
    def __init__(
        self,
        model_config: Optional[Union[ModelConfig,Dict]] = None,
        data_config: Optional[Union[ModelConfig,Dict]] = None,
        training_config: Optional[Union[ModelConfig,Dict]] = None,
        inference_config: Optional[Union[ModelConfig,Dict]] = None,
        model = None
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
        self.model = model

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
            return instance
        elif strategy == Orchestration.PYTORCH:
            instance.model = cls.load_from_pytorch(**kwargs)
            return instance
        else:
            raise ValueError(f"Unknown load strategy: {strategy}")

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
        from ptycho_torch.api.api_helper import create_new_model

        model_data = create_new_model(model = model,
                                      config_manger = config_manager,
                                      model_config = model_config,
                                      data_config = data_config,
                                      training_config = training_config,
                                      inference_config = inference_config)

        instance = cls(**model_data)
        
        return instance
    
    @staticmethod
    def load_from_mlflow(
        run_id: str = None,
        mlflow_tracking_uri: Optional[str] = None
    ):
        from ptycho_torch.api.api_helper import load_with_mlflow
        """
        Loads model from mlflow run. Requires underlying model to be integrated with mlflow infra
        """

        model, run_id = load_with_mlflow(run_id = run_id,
                                          mlflow_tracking_uri = mlflow_tracking_uri)

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
    
    def save_mlflow(
        self,
        destination_path: str,
        current_mlflow_tracking_uri: Optional[str] = None
    ):
        """
        Copies MLflow run while preserving both experiment ID and run ID.
        Works across storage backends.
        """
        from ptycho_torch.api.api_helper import save_with_mlflow
        
        source_uri = current_mlflow_tracking_uri or self.mlflow_tracking_uri

        dest_uri = save_with_mlflow(run_id = self.run_id,
                                   destination_path = destination_path,
                                   mlflow_tracking_uri = source_uri,)
        
        return dest_uri

    def save_pytorch(self, destination_path = str):
        pass

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
        
        instance._setup_lightning_trainer()

        return instance
    
    @classmethod
    def _from_pytorch(cls,
                     model: PtychoModel,
                     dataloader: PtychoDataLoader, 
                     config_manager: Optional[ConfigManager] = None,
                     training_config: Optional[Union[TrainingConfig, Dict]] = None) -> 'Trainer':
        pass #TBD
    
    def _setup_lightning_trainer(self):
        """
        Lightning-specific setup that was copied from my own training code in train.py
        """
        from ptycho_torch.api.api_helper import setup_lightning_trainer

        self._trainer = setup_lightning_trainer(self.ptycho_model,
                                                self.training_config)
        
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
        else:
            raise ValueError(f"Unknown load strategy: {strategy}")
        
        return result

    def _train_with_mlflow(self, experiment_name: Optional[str] = None):
        """
        Executes training using MlFlow api
        """
        from ptycho_torch.api.api_helper import train_with_mlflow

        run_ids = train_with_mlflow(self, experiment_name)

        return run_ids

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
                 datagen_config: Optional[Union[DatagenConfig, Dict]] = None,
                 probe_arg = None) -> 'Datagen':
        """
        Abstracted class method to generate a new instance of datagen
        """
        from ptycho_torch.api.api_helper import assemble_probes_from_npz
        
        probe_arg = {}

        probe_list, probe_arg, data_config = assemble_probes_from_npz(npz_path = npz_path,
                                                         config_manager = config_manager,
                                                         probe_arg = probe_arg,
                                                         datagen_config = datagen_config)

        return cls(probe_list, probe_arg,
                   datagen_config, data_config)
    
    def _create_synthetic_objects(self,
                                  obj_arg = None):
        from ptycho_torch.datagen.datagen import simulate_synthetic_objects
        from ptycho_torch.api.api_helper import simulate_synthetic_object_list
        
        #Obj_arg is deprecated but this exposes in case anything need to be modified
        if obj_arg is None:
            obj_arg = {}

        simulate_synthetic_object_list(self, obj_arg)
    
    def _generate_simulated_data(self,
                                 synthetic_path):
        
        from ptycho_torch.api.api_helper import generate_simulated_data

        generate_simulated_data(self, synthetic_path)
        
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

        from ptycho_torch.api.api_helper import predict_only

        return predict_only(self,
                            dataloader = ptycho_dataloader,
                            device = 'cuda')
    
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
        from ptycho_torch.api.api_helper import predict_and_stitch_barycentric

        return predict_and_stitch_barycentric(self, ptycho_dataloader)
    
#Usage examples

#1.a Instantiate ConfigManager instance, either using json file, mlflow load or existing settings
# config_manager = ConfigManager._flexible_load(run_id run_id) <- load from mlflow
# config_manager = ConfigManager._flexible_load(json_path = '/path/to/config.json') <- load from json
#1.b Update ConfigManager parameters using update function if using user interface to modify any values
# config_manager.update(data_config = new_data_config_dict) <- Example of updating certain parameters in data_config for any keys in the dict

#2.a Instantiate PtychoDataloader instance, using built in class methods depending on dataloader format.

#3. Instantiate PtychoModel instance, using built in class methods depending on model format

#4. Training only: Instantiate trainer based on orchestration
        

        
        






    

        
    




    
    
        







