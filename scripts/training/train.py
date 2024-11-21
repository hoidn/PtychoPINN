#!/usr/bin/env python

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
def main() -> None:
    """Main function to orchestrate the CDI example script execution."""
    args = parse_arguments()
    config: TrainingConfig = setup_configuration(args, args.config)
    
    # Update global params with new-style config at entry point
    update_legacy_dict(params.cfg, config)
    
    try:

        #ptycho_data, ptycho_data_train, obj = load_and_prepare_data(config['train_data_file_path'])
        ptycho_data = load_data(config['train_data_file_path'], n_images = 512)
        
        test_data = None
        if config['test_data_file_path']:

            #test_data, test_ptycho_data_train, test_obj = load_and_prepare_data(config['test_data_file_path'])
            test_data = load_data(config['test_data_file_path'])

        recon_amp, recon_phase, results = run_cdi_example(ptycho_data, test_data, config)
        model_manager.save(config['output_prefix'])
        save_outputs(recon_amp, recon_phase, results, config['output_prefix'])
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        raise

if __name__ == "__main__":
    main()
