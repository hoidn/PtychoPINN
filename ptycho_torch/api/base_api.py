from typing import Optional, Dict, Any, Tuple, Union, Protocol
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
    def from_mlflow(
          cls,
          run_id: str,
          mlflow_tracking_uri: Optional[str] = None
    ) -> 'ConfigManager':
        """
        Loads configs via artifacts from mflow run (based on run-id)
        """

        try:
            configs = load_all_configs_from_mlflow(run_id, mlflow_tracking_uri)
        except Exception as e:
            print("Failed to load configs from MlFlow. Defaulting to vanilla...")
            configs = (None, None, None, None, None)

        return cls(*configs)
    
    @classmethod
    def from_json(
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
    def flexible_load(
         cls,
         run_id: Optional[str],
         json_path: str,
         mlflow_tracking_uri: Optional[str] = None
    ) -> 'ConfigManager':
         """
         Optionally load from mlflow config first before overriding with JSON in relevant fields
         """

         #Start with mlflow
         mlflow_manager = cls.from_mlflow(run_id, mlflow_tracking_uri)

         #Return setup with json
         json_manager, json_loaded = cls.from_json(json_path)

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
    def _parse_config(self, config: Optional[Union[Any, Dict]], ConfigClass):
        if config is None:
            return ConfigClass()
        elif isinstance(config, dict):
            instance = ConfigClass()
            update_existing_config(instance, config)
            return instance
        else:
            return config

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
        data_module_bool: bool = True
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

        if data_module_bool:
            self._setup_datamodule()


    def _setup_datamodule(self):
        """
        Initialize PtychoDataModule (flexible for multi-instanced gpu training)
        Needs more complex handling 
        """
        self.data_module = PtychoDataModule(
            self.data_dir,
            self.model_config,
            self.data_config,
            self.training_config,
            initial_remake_map = True,
            val_split=0.05,
            val_seed=42
        )

    def train_dataloader(self):
        """Training dataloader"""
        return self.data_module.train_dataloader()
    
    def val_dataloader(self):
        """Validation dataloader"""
        return self.data_module.val_dataloader()

    def __iter__(self):
        return iter(self.train_dataloader())
    
    def __len__(self):
        return len(self.data_module.train_dataloader())

from torch.nn import Module
from typing import Enum
import mlflow
from mlflow.tracking import MlflowClient
import shutil
from pathlib import Path

class Orchestration(Enum):
    """Enumeration of supported orchestration strategies"""
    MLFLOW = "mlflow"
    PYTORCH = "pytorch"

class PtychoModel:
    """
    Wrapper around generic module with save/load utils
    """
    def __init__(
        self,
        config_manager: Optional[ConfigManager] = None,
        model_config: Optional[Union[ModelConfig,Dict]] = None,
        data_config: Optional[Union[ModelConfig,Dict]] = None,
        training_config: Optional[Union[ModelConfig,Dict]] = None,
        inference_config: Optional[Union[ModelConfig,Dict]] = None,
        model: Optional[Any] = None,
        run_id: Optional[str] = None
    ):
        """
        Loads model with configs using config manager
        model is generic, can handle PtychoPINN_Lightning (for PtychoPINN_torch).
        Includes a run_id parameter in case it was instantiated from a trained model
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

        self.model = model(
            self.model_config,
            self.data_config,
            self.training_config,
            self.inference_config
        )

        self.run_id = run_id

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
            return self._save_mlflow(path, **kwargs)
        elif strategy == Orchestration.PYTORCH:
            return self._save_pytorch(path, **kwargs)
        else:
            raise ValueError(f"Unknown save strategy: {strategy}")
        
    @classmethod
    def load(
        cls,
        path: str,
        strategy: Union[Orchestration, str] = Orchestration.MLFLOW,
        model: Optional[Any] = None,
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
        if isinstance(strategy, str):
            strategy = Orchestration(strategy)
        
        if strategy == Orchestration.MLFLOW:
            return cls._load_mlflow(path, model=model, **kwargs)
        elif strategy == Orchestration.PYTORCH:
            return cls._load_pytorch(path, model=model, **kwargs)
        else:
            raise ValueError(f"Unknown load strategy: {strategy}")

    @classmethod
    def load_from_mlflow(
        cls,
        run_id: str,
        mlflow_tracking_uri: Optional[str] = None,
        model: Optional[Any] = None
    ) -> 'PtychoModel':
        """
        Loads model from mlflow run. Requires underlying model to be integrated with mlflow infra
        """
        config_manager = ConfigManager.from_mlflow(run_id, mlflow_tracking_uri)
        loaded_model = cls(config_manager = config_manager,
                           model = model,
                           run_id = run_id)

        return loaded_model

    @classmethod
    def _load_pytorch(
        cls,
        path: str,
        model: Optional[Any] = None,
        config_path: Optional[str] = None
    ) -> 'PtychoModel':
        pass #TBD
    
    def load_from_checkpoint(
        checkpoint_path: str,
        model: Optional[Any] = None,
        config_path: Optional[str] = None
    ):
        pass
        

    def save_mlflow(self, destination_path: str,
            mlflow_tracking_uri: Optional[str] = None
        ):
        
        if self.run_id is None:
            raise ValueError("No run id associated with this model, cannot save to new location. See MlFlow documentation")
        
        #Set up mlflow run
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)

        client = MlflowClient()
        run = client.get_run(self.run_id)

        #Destination details
        dest_path = Path(destination_path)
        mlruns_path = dest_path / "mlruns"
        experiment_id = run.info.experiment_id
        run_path = mlruns_path / experiment_id / self.run_id

        run_path.mkdir(parents=True, exist_ok = True)

        #Download artifacts
        artifacts_path = run_path / "artifacts"
        if run.info.artifact_uri:
            try:
                client.download_artifacts(self.run_id, "",
                                          dst_path = str(artifacts_path))
            except Exception as e:
                print(f"Warning. Could not download artifacts because of {e}")

        meta_path = run_path / "meta.yaml"

        #Saving run metadata at destination
        with open(meta_path, 'w') as f:
            import yaml
            meta = {
            'run_id': self.run_id,
            'experiment_id': experiment_id,
            'status': run.info.status,
            'start_time': run.info.start_time,
            'end_time': run.info.end_time,
            'artifact_uri': str(artifacts_path),
            }

            yaml.dump(meta, f)
        
        #Saving artifacts at destination

        # Save params
        params_path = run_path / "params"
        params_path.mkdir(exist_ok=True)
        for key, value in run.data.params.items():
            with open(params_path / key, 'w') as f:
                f.write(value)
        
        # Save metrics
        metrics_path = run_path / "metrics"
        metrics_path.mkdir(exist_ok=True)
        for key, value in run.data.metrics.items():
            with open(metrics_path / key, 'w') as f:
                f.write(f"{run.info.start_time} {value} 0\n")
        
        # Save tags
        tags_path = run_path / "tags"
        tags_path.mkdir(exist_ok=True)
        for key, value in run.data.tags.items():
            with open(tags_path / key, 'w') as f:
                f.write(value)
        
        print(f"Saved MLflow run {self.run_id} to {run_path}")
        print(f"New tracking URI: file://{mlruns_path.absolute()}")
        
        return str(mlruns_path.absolute())

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
    def from_lightning(cls,
                       model: PtychoModel,
                       dataloader: PtychoDataLoader, 
                       config_manager: Optional[ConfigManager] = None,
                       training_config: Optional[Union[TrainingConfig, Dict]] = None) -> 'Trainer':
        
        parsed_config = ConfigManager._parse_config()

        instance = cls(model = model,
                       dataloader = dataloader,
                       config_manager = config_manager,
                       parsed_config = parsed_config,
                       strategy = TrainStrategy.LIGHTNING)
        
        instance._setup_lightning_trainer()

        return instance
    
    @classmethod
    def from_pytorch(cls,
                     model: PtychoModel,
                     dataloader: PtychoDataLoader, 
                     config_manager: Optional[ConfigManager] = None,
                     training_config: Optional[Union[TrainingConfig, Dict]] = None) -> 'Trainer':
        pass #TBD
    
    def _setup_lightning_trainer(self):
        """
        Lightning-specific setup that was copied from my own training code in train.py
        """
        from ptycho_torch.api.trainer_api import setup_lightning_trainer

        self._trainer = setup_lightning_trainer(self.model,
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
            strategy = Orchestration(strategy)

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
        from ptycho_torch.train_utils import find_learning_rate, is_effectively_global_rank_zero, log_parameters_mlflow, print_auto_logged_info, ModelFineTuner

        exp_name = experiment_name or self.training_config.experiment_name

        updated_lr = find_learning_rate(
            self.training_config.learning_rate,
            self.training_config.n_devices,
            self.training_config.batch_size
        )

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
    def from_npz(cls,
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
    
    @staticmethod
    def _parse_datagen_config(config_manager: Optional[ConfigManager],
                               datagen_config = Optional[Union[DatagenConfig, Dict]]):
        if config_manager is not None:
            return config_manager.datagen_config
        elif datagen_config is not None:
            if isinstance(datagen_config, dict):
                config = DatagenConfig()
                update_existing_config(config, datagen_config)
                return config
            else:
                return datagen_config
            
        return DatagenConfig() #Defaults to vanilla training config if nothing works
        

        






    

        
    




    
    
        







