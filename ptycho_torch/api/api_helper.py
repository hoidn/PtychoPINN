import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from ptycho_torch.api.base_api import PtychoModel, PtychoDataLoader, ConfigManager, Trainer, Datagen
from typing import Optional, Dict, Any, Tuple, Union, Protocol
from ptycho_torch.config_params import TrainingConfig, DatagenConfig, DataConfig, ModelConfig, InferenceConfig, update_existing_config
from ptycho_torch.train_utils import get_training_strategy
import mlflow
import torch
import os

def config_from_json(json_path):

    from ptycho_torch.utils import load_config_from_json, validate_and_process_config
    
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

    return {
        "data_config": data_config,
        "model_config": model_config,
        "training_config": training_config,
        "datagen_config": datagen_config,
        "inference_config": inference_config,
    }, json_loaded

def update_manager_with_json(mlflow_manager: ConfigManager,
                             json_loaded,
                             json_manager: ConfigManager = None):
    """
    Quick helper function to override configs from mlflow load with json if doing flexible load
    """
    if json_loaded:
        update_existing_config(mlflow_manager.data_config, json_manager.data_config)
        update_existing_config(mlflow_manager.model_config, json_manager.model_config)
        update_existing_config(mlflow_manager.training_config, json_manager.training_config)
        update_existing_config(mlflow_manager.inference_config, json_manager.inference_config)
        update_existing_config(mlflow_manager.datagen_config, json_manager.datagen_config)

    return mlflow_manager


def create_new_model(model=None,
                    config_manager: Optional[ConfigManager] = None,
                    model_config: Optional[Union[ModelConfig, Dict]] = None,
                    data_config: Optional[Union[DataConfig, Dict]] = None,
                    training_config: Optional[Union[TrainingConfig, Dict]] = None,
                    inference_config: Optional[Union[InferenceConfig, Dict]] = None
                    ):
    
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

    return {
        'model_config': config_manager.model_config,
        'data_config': config_manager.data_config,
        'training_config': config_manager.training_config,
        'inference_config': config_manager.inference_config,
        'model': model_instance
    }

def setup_lightning_trainer(ptycho_model: PtychoModel,
                            training_config: TrainingConfig) -> L.Trainer:
    
    val_loss_label = ptycho_model.model.val_loss_name #Nested models because Ptycho

    checkpoint_callback = ModelCheckpoint(
        monitor = val_loss_label,
        mode = 'min',
        save_top_k = 1,
        filename = 'best-checkpoint'
    )

    early_stop_callback = EarlyStopping(
        monitor = val_loss_label,
        mode='min',
        patience=100,
        verbose = True,
        strict = True
    )

    #Calculate total epochs
    if training_config.stage_2_epochs > 0 or training_config.stage_3_epochs > 0:
        total_epochs = (
            training_config.stage_1_epochs +
            training_config.stage_2_epochs + 
            training_config.stage_3_epochs
        )
    else:
        total_epochs = training_config.epochs
    
    #Instantiate lightning trainer
    trainer = L.Trainer(
        max_epochs = total_epochs,
        devices = training_config.n_devices,
        accelerator = 'gpu',
        callbacks = [checkpoint_callback, early_stop_callback],
        strategy = get_training_strategy(training_config.n_devices),
        check_val_every_n_epoch=1,
        enable_checkpointing=True,
        enable_progress_bar=True
    )

    return trainer

def train_with_mlflow(instance,
                      experiment_name: str,
                      ):
    """
    Mlflow training script to externalize api call
    """
    from ptycho_torch.train_utils import find_learning_rate, is_effectively_global_rank_zero, log_parameters_mlflow, print_auto_logged_info, ModelFineTuner
    
    training_config = config_manager.training_config
    config_manager = instance.config_manager,
    dataloader = instance.dataloader
    ptycho_model = instance.ptycho_model
    trainer = instance.trainer

    exp_name = experiment_name or training_config.experiment_name

    updated_lr = find_learning_rate(
        training_config.learning_rate,
        training_config.n_devices,
        training_config.batch_size
    )

    ptycho_model.model.lr = updated_lr
    ptycho_model.model.training = True

    run_ids = {}
    
    # Run DDP checks and initialize MLflow run if on main process
    if is_effectively_global_rank_zero():
        mlflow.set_experiment(exp_name)
        mlflow.pytorch.autolog(checkpoint_monitor = ptycho_model.model.val_loss_name)

        with mlflow.start_run() as run:
            #Log all configs
            log_parameters_mlflow(
                dataloader.data_config, 
                ptycho_model.model_config,
                training_config,
                ptycho_model.inference_config,
                config_manager.datagen_config
            )

            #Set tags
            mlflow.set_tag("stage", "training")
            mlflow.set_tag("encoder_frozen", "False")
            mlflow.set_tag("model_name", training_config.model_name)
            if training_config.notes:
                mlflow.set_tag("notes", training_config.notes)

            #Train
            print("Beginning model training...")
            trainer.fit(
                ptycho_model.model,
                datamodule = dataloader.data_module
            )

            print("Training Complete!!")
            print_auto_logged_info(mlflow.get_run(run_id = run.info.run_id))
            run_ids['training'] = run.info.run_id

    if training_config.epochs_fine_tune > 0:
        fine_tuner = ModelFineTuner(
            ptycho_model.model,
            dataloader.data_module,
            training_config
        )
        fine_tuning_run_id = fine_tuner.fine_tune(experiment_name = exp_name)

        if is_effectively_global_rank_zero():
            run_ids['fine_tune'] = fine_tuning_run_id
        
    if is_effectively_global_rank_zero():
        print(f"Training run_id: {run_ids.get('training')}")
        print(f"Fine tune run_id: {run_ids.get('fine_tune')}")

    return run_ids

def load_with_mlflow(
        run_id: str = None,
        mlflow_tracking_uri: Optional[str] = None):
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

def save_with_mlflow(
        run_id,
        destination_path: str,
        mlflow_tracking_uri: Optional[str] = None
    ):
    """
    Copies MLflow run while preserving both experiment ID and run ID.
    Works across storage backends.
    """
    import tempfile
    import yaml
    from mlflow import MlflowClient
    from pathlib import Path
    
    if run_id is None:
        raise ValueError("No run_id associated with this model.")
    
    # Set source
    source_uri = mlflow_tracking_uri
    if source_uri:
        mlflow.set_tracking_uri(source_uri)
    
    source_client = MlflowClient()
    run = source_client.get_run(run_id)
    exp_id = run.info.experiment_id
    
    # Setup destination paths
    dest_path = Path(destination_path)
    dest_mlruns = dest_path / "mlruns"
    dest_exp_dir = dest_mlruns / exp_id
    dest_run_dir = dest_exp_dir / run_id
    
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
        source_client.download_artifacts(run_id, "", dst_path=str(artifacts_dir))
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
        'run_id': run_id, 
        'run_name': run.data.tags.get('mlflow.runName', ''),
        'run_uuid': run_id, 
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
        metric_history = source_client.get_metric_history(run_id, key)
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

def setup_lightning_datamodule(instance: PtychoDataLoader):
    from ptycho_torch.train_utils import PtychoDataModule

    instance.data_module = PtychoDataModule(
            instance.data_dir,
            instance.model_config,
            instance.data_config,
            instance.training_config,
            initial_remake_map = True,
            val_split=0.05,
            val_seed=instance.val_seed,
            memory_map_dir=instance.memory_map_dir
        )

    print("Set up lightning datamodule")

def setup_tensordict_dataloader(instance: PtychoDataLoader):

    from ptycho_torch.dataloader import PtychoDataset, TensorDictDataLoader, Collate_Lightning

    instance.dataset = PtychoDataset(
                    ptycho_dir=instance.data_dir,
                    model_config=instance.model_config,
                    data_config=instance.data_config,
                    data_dir = instance.memory_map_dir,
                    remake_map= True
                )
    dataset_size = len(instance.dataset)
    val_size = int(instance.val_split * dataset_size)
    train_size = dataset_size - val_size
    #Seeding for dataloader
    generator = torch.Generator().manual_seed(instance.val_seed)
    instance.train_dataset, instance.val_dataset = torch.utils.data.random_split(
        instance.dataset, [train_size, val_size], generator=generator
    )

    print("Setup ptycho dataset")

    #Note, collate function is currently integrated with pytorch lightning for multi gpu inference.
    instance.dataloader = TensorDictDataLoader(
            instance.train_dataset,
            batch_size=instance.training_config.batch_size,
            shuffle=True,
            num_workers=instance.training_config.num_workers,
            collate_fn=Collate_Lightning(pin_memory_if_cuda=True), # Lightning handles device placement
            pin_memory=True, # Lightning DDP often benefits from this
            persistent_workers=True,
            prefetch_factor = 4,
        )
    
    print("Setup ptycho dataloader")

def assemble_probes_from_npz(npz_path,
                             config_manager: ConfigManager,
                             probe_arg: Dict,
                             datagen_config: DatagenConfig):
    """
    Assembles list of probes from npz path. Duplicates exist if multiple objects per probe
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

    #Check for multiple specified objects
    if isinstance(datagen_config.object_class, str):
        num_obj = 1
    elif isinstance(datagen_config.object_class, list):
        num_obj = len(datagen_config.object_class)

    exp_probe_list = assemble_precomputed_images(npz_path, 'probe', True)
    probe_list = [item for item in exp_probe_list for _ in range(num_obj * datagen_config.objects_per_probe)]
    probe_name_idx = [idx for idx in list(range(len(exp_probe_list))) for _ in range(datagen_config.objects_per_probe)]
    probe_arg['probe_name_idx'] = probe_name_idx

    return probe_list, probe_arg, data_config

def simulate_synthetic_object_list(instance: Datagen,
                               obj_arg: Dict):
    """
    Simulates synthetic objects
    """
    from ptycho_torch.datagen.datagen import simulate_synthetic_objects
    # Try creating synthetic object
    try:
        print(f"Creating objects for class: {instance.datagen_config.object_class}")
        image_size = instance.datagen_config.image_size
        objects_per_probe = instance.datagen_config.objects_per_probe
        object_class = instance.datagen_config.object_class
        #Dataconfig currently unused, so just passing vanilla DataConfig class instance
        instance.object_list = simulate_synthetic_objects(image_size, DataConfig(), objects_per_probe,
                                                        object_class, obj_arg)
    except:
        raise ValueError("Inputted synthetic object class not valid")
    
def generate_simulated_data(instance: Datagen,
                            synthetic_path):
    """
    Extended function for generating simulated data from object/probe lists
    """
    from ptycho_torch.utils import remove_all_files
    from ptycho_torch.datagen.datagen import simulate_multiple_experiments

    # Remove all existing files from directory
    if not os.path.exists(synthetic_path):
        os.mkdir(synthetic_path)
    else:
        remove_all_files(synthetic_path)

    # Simulate and fill directory
    print("Simulating experiments...")

    instance.probe_arg['beamstop_diameter'] = instance.datagen_config.beamstop_diameter
    image_size = instance.datagen_config.image_size
    diff_per_object = instance.datagen_config.diff_per_object

    simulate_multiple_experiments(instance.object_list, instance.probe_list,
                                  diff_per_object,
                                  image_size, instance.data_config, instance.probe_arg,
                                  synthetic_path)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def predict_only(instance,
                 dataloader: PtychoDataLoader,
                 device = 'cuda'):

    instance.model.to(device)
    res = []
    
    with torch.no_grad():
        for batch in dataloader.dataloader:
            batch = batch.to(device)
            output = instance.model.forward_predict(batch)
            res.append(output.cpu())

    final_output = torch.cat(res, dim=0)
    
    return final_output

def predict_and_stitch_barycentric(instance,
                                   dataloader: PtychoDataLoader):
    
    from ptycho_torch.reassembly import reconstruct_image_barycentric

        

    result, _, _ = reconstruct_image_barycentric(model = instance.model,
                                                    ptycho_dset = dataloader.dataset,
                                                    training_config = instance.training_config,
                                                    data_config = instance.data_config,
                                                    model_config = instance.model_config,
                                                    inference_config = instance.inference_config,
                                                    gpu_ids = None,
                                                    use_mixed_precision=True, verbose = False)

    #Save results
    result_cpu = result.to('cpu')

    return result_cpu
    
