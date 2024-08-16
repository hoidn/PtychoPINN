#!/usr/bin/env python

from ptycho.workflows.train_script_components import (
    parse_arguments,
    setup_configuration,
    load_and_prepare_data,
    run_cdi_example,
    save_outputs,
    logger
)

def main() -> None:
    """Main function to orchestrate the CDI example script execution."""
    args = parse_arguments()
    config = setup_configuration(args, args.config)
    
    try:
        ptycho_data, ptycho_data_train, obj = load_and_prepare_data(config['train_data_file_path'])
        
        test_data = None
        if config['test_data_file_path']:
            test_ptycho_data, test_ptycho_data_train, test_obj = load_and_prepare_data(config['test_data_file_path'])
            test_data = test_ptycho_data
        
        recon_amp, recon_phase, results = run_cdi_example(ptycho_data, test_data, config)
        save_outputs(recon_amp, recon_phase, results, config['output_prefix'])
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        raise

if __name__ == "__main__":
    main()
