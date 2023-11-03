#!/bin/bash
# Script for PINN ablation study

# Help message function
usage() {
  echo "Usage: $0 [-h] data_source prefix nepochs outer_offset_train outer_offset_test"
  echo
  echo "  -h, --help                  Display this help and exit."
  echo "  data_source                 Data source parameter to be passed to the train.py script. Selects a simulated object type."
  echo "  prefix                      Output prefix for the training session."
  echo "  nepochs                     Number of epochs for the training."
  echo "  outer_offset_train          Scan grid offset value for training."
  echo "  outer_offset_test           Scan grid offset value for evaluation."
  echo
  echo "Example:"
  echo "  $0 grf grf2 100 8 20"
}

# Check for '-h' or '--help' and invoke usage function
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

# Check for the correct number of arguments
if [[ "$#" -ne 5 ]]; then
  echo "Error: Incorrect number of arguments."
  usage
  exit 1
fi

# Assign arguments to variables
dsource=$1
prefix=$2
nepochs=$3
outer_offset_train=$4 # 8
outer_offset_test=$5 # 20

# Invocation 1
train.py  --data_source $dsource --nepochs $nepochs --offset 4  --output_prefix $prefix --model_type pinn --gridsize 2 --n_filters_scale 2 --object_big True --intensity_scale_trainable True --label "PINN,NLL,overlaps" --nimgs_train 2 --nimgs_test 1 --outer_offset_train $outer_offset_train --outer_offset_test $outer_offset_test --set_phi 

## Invocation 2
#train.py  --data_source $dsource --nepochs $nepochs --offset 4  --output_prefix $prefix --model_type pinn --nll_weight 0.0 --mae_weight 1.0 --intensity_scale_trainable True --object_big True --n_filters_scale 2 --label "PINN,overlaps" --nimgs_train 2 --nimgs_test 1 --outer_offset_train $outer_offset_train --outer_offset_test $outer_offset_test --set_phi 
#
## Invocation 3
#train.py  --data_source $dsource --nepochs $nepochs --offset 4  --output_prefix $prefix --model_type supervised --n_filters_scale 1 --gridsize 2 --label "overlaps" --nimgs_train 2 --nimgs_test 1 --outer_offset_train $outer_offset_train --outer_offset_test $outer_offset_test --set_phi 
#
## Invocation 4
#train.py  --data_source $dsource --nepochs $nepochs --offset 4  --output_prefix $prefix --model_type pinn --gridsize 1 --nll_weight 1.0 --mae_weight 0.0 --label "PINN,NLL" --nimgs_train 2 --nimgs_test 1 --outer_offset_train $outer_offset_train --outer_offset_test $outer_offset_test --set_phi 
#
## Invocation 5
#train.py  --data_source $dsource --nepochs $nepochs --offset 4  --output_prefix $prefix --model_type pinn --gridsize 1 --nll_weight 0.0 --mae_weight 1.0 --label "PINN" --nimgs_train 2 --nimgs_test 1 --outer_offset_train $outer_offset_train --outer_offset_test $outer_offset_test --set_phi 
#
## Invocation 6
#train.py  --data_source $dsource --nepochs $nepochs --offset 4  --output_prefix $prefix --model_type supervised --gridsize 1 --n_filters_scale 1 --label "none" --nimgs_train 2 --nimgs_test 1 --outer_offset_train $outer_offset_train --outer_offset_test $outer_offset_test --set_phi 
