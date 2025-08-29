#!/usr/bin/env python

import logging
import sys

# Set up file handler for debug logging
file_handler = logging.FileHandler('train_debug.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Set up console handler for info logging
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Configure root logger
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger().addHandler(file_handler)
logging.getLogger().addHandler(console_handler)

from ptycho.workflows.components import (
    parse_arguments,
    setup_configuration,
    load_data,
    run_cdi_example,
    save_outputs,
    logger
)
from ptycho.config.config import TrainingConfig, update_legacy_dict
from ptycho import model_manager, params

def interpret_n_images_parameter(n_images: int, gridsize: int) -> tuple[int, str]:
    """
    Interpret --n-images parameter based on gridsize.
    
    For gridsize=1: n_images refers to individual images (traditional behavior)
    For gridsize>1: n_images refers to number of neighbor groups
    
    Args:
        n_images: User-specified number from --n-images
        gridsize: Current gridsize setting
        
    Returns:
        tuple: (actual_n_images, interpretation_message)
    """
    if gridsize == 1:
        message = f"Parameter interpretation: --n-images={n_images} refers to individual images (gridsize=1)"
        return n_images, message
    else:
        total_patterns = n_images * gridsize * gridsize
        message = f"Parameter interpretation: --n-images={n_images} refers to neighbor groups (gridsize={gridsize}, total patterns={total_patterns})"
        return n_images, message

def interpret_sampling_parameters(config: TrainingConfig) -> tuple[int, int, str]:
    """
    Interpret sampling parameters with support for independent control.
    
    Priority:
    1. If n_subsample is specified: use it for data subsampling
    2. Otherwise: use n_groups for legacy behavior
    
    Args:
        config: Training configuration with sampling parameters
        
    Returns:
        tuple: (n_subsample, n_groups, interpretation_message)
    """
    gridsize = config.model.gridsize
    
    # Case 1: Independent control with n_subsample
    if config.n_subsample is not None:
        n_subsample = config.n_subsample
        n_groups = config.n_groups
        
        if gridsize == 1:
            message = (f"Independent sampling control: subsampling {n_subsample} images, "
                      f"using {n_groups} groups for training")
        else:
            total_from_groups = n_groups * gridsize * gridsize
            message = (f"Independent sampling control: subsampling {n_subsample} images, "
                      f"creating {n_groups} groups (approx {total_from_groups} patterns from groups)")
        
        return n_subsample, n_groups, message
    
    # Case 2: Legacy behavior - n_groups controls both
    else:
        # For backward compatibility, n_groups controls subsampling
        if gridsize == 1:
            n_subsample = config.n_groups
            n_groups = config.n_groups
            message = f"Legacy mode: using {n_groups} groups (gridsize=1)"
        else:
            # For gridsize > 1, we need to subsample enough to create the groups
            n_subsample = config.n_groups  # This will be interpreted as groups by generate_grouped_data
            n_groups = config.n_groups
            total_patterns = n_groups * gridsize * gridsize
            message = (f"Legacy mode: --n-groups={n_groups} refers to neighbor groups "
                      f"(gridsize={gridsize}, approx {total_patterns} patterns)")
        
        return n_subsample, n_groups, message
def main() -> None:
    """Main function to orchestrate the CDI example script execution."""
    args = parse_arguments()
    
    # Handle legacy argument name
    if hasattr(args, 'train_data_file_path'):
        args.train_data_file = args.train_data_file_path
        delattr(args, 'train_data_file_path')
        
    config = setup_configuration(args, args.config)
    
    # Interpret sampling parameters with new independent control support
    n_subsample, n_groups, interpretation_message = interpret_sampling_parameters(config)
    logger.info(interpretation_message)
    
    # Log warning if potentially problematic configuration
    if config.n_subsample is not None and config.model.gridsize > 1:
        min_required = n_groups * config.model.gridsize * config.model.gridsize
        if n_subsample < min_required:
            logger.warning(f"n_subsample ({n_subsample}) may be too small to create {n_groups} "
                         f"groups of size {config.model.gridsize}². Consider increasing n_subsample to at least {min_required}")
    
    # Update global params with new-style config at entry point
    update_legacy_dict(params.cfg, config)
    
    try:
        logger.info(f"Starting training with n_subsample={n_subsample}, n_groups={n_groups}, "
                   f"stitching={'enabled' if args.do_stitching else 'disabled'}")

        # Load data with new independent sampling parameters
        # Note: load_data still uses n_images parameter name internally
        ptycho_data = load_data(
            str(config.train_data_file), 
            n_images=n_groups,  # Pass n_groups as n_images to maintain API compatibility
            n_subsample=n_subsample,
            subsample_seed=config.subsample_seed
        )
        
        # Check for metadata and override nphotons if present
        from ptycho.metadata import MetadataManager
        try:
            _, metadata = MetadataManager.load_with_metadata(str(config.train_data_file))
            # Check both top-level and physics_parameters for nphotons
            metadata_nphotons = None
            if metadata:
                if 'nphotons' in metadata:
                    metadata_nphotons = float(metadata['nphotons'])
                elif 'physics_parameters' in metadata and 'nphotons' in metadata['physics_parameters']:
                    metadata_nphotons = float(metadata['physics_parameters']['nphotons'])
                
                if metadata_nphotons is not None:
                    original_nphotons = config.nphotons
                    # Update config with metadata nphotons value (create new instance for dataclass)
                    config = config.__class__(
                        **{**config.__dict__, 'nphotons': metadata_nphotons}
                    )
                    logger.info(f"Overriding nphotons from config ({original_nphotons:.1e}) with value from dataset metadata: {metadata_nphotons:.1e}")
                    # Update the legacy params dict as well
                    params.cfg['nphotons'] = metadata_nphotons
        except Exception as e:
            logger.debug(f"No metadata found or error reading metadata: {e}")
        
        test_data = None
        if config.test_data_file:
            test_data = load_data(str(config.test_data_file))
            logger.info(f"Loaded test data from {config.test_data_file}")

        recon_amp, recon_phase, results = run_cdi_example(ptycho_data, test_data, config, do_stitching=args.do_stitching)
        model_manager.save(str(config.output_dir))
        save_outputs(recon_amp, recon_phase, results, str(config.output_dir))
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        raise

if __name__ == "__main__":
    main()
