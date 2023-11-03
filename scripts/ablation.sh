#!/bin/bash

# Default values
DEFAULT_DSOURCE="lines"
DEFAULT_PREFIX="tmp"
DEFAULT_NEPOCHS=60
DEFAULT_OUTER_OFFSET_TRAIN=8
DEFAULT_OUTER_OFFSET_TEST=20

# Help message function
usage() {
    echo "Usage: $0 [-h] [-d <data_source>] [-p <prefix>] [-n <nepochs>] [-o <outer_offset_train>] [-t <outer_offset_test>]"
    echo "  -h                      Display this help message."
    echo "  -d <data_source>        Data source for training. Default: $DEFAULT_DSOURCE"
    echo "  -p <prefix>             Output prefix for the files. Default: $DEFAULT_PREFIX"
    echo "  -n <nepochs>            Number of epochs for training. Default: $DEFAULT_NEPOCHS"
    echo "  -o <outer_offset_train> Outer offset for training. Default: $DEFAULT_OUTER_OFFSET_TRAIN"
    echo "  -t <outer_offset_test>  Outer offset for testing. Default: $DEFAULT_OUTER_OFFSET_TEST"
    exit 1
}

# Parse command line options
while getopts ":hd:p:n:o:t:" opt; do
  case ${opt} in
    h )
      usage
      ;;
    d )
      dsource=$OPTARG
      ;;
    p )
      prefix=$OPTARG
      ;;
    n )
      nepochs=$OPTARG
      ;;
    o )
      outer_offset_train=$OPTARG
      ;;
    t )
      outer_offset_test=$OPTARG
      ;;
    \? )
      echo "Invalid option: -$OPTARG" 1>&2
      usage
      ;;
    : )
      echo "Invalid option: -$OPTARG requires an argument" 1>&2
      usage
      ;;
  esac
done

# Assign default values if variables are empty
dsource=${dsource:-$DEFAULT_DSOURCE}
prefix=${prefix:-$DEFAULT_PREFIX}
nepochs=${nepochs:-$DEFAULT_NEPOCHS}
outer_offset_train=${outer_offset_train:-$DEFAULT_OUTER_OFFSET_TRAIN}
outer_offset_test=${outer_offset_test:-$DEFAULT_OUTER_OFFSET_TEST}

# Invocation: Example of how you'd call one of the training commands
invoke_training() {
train.py --data_source "$dsource" --nepochs "$nepochs" --offset 4 --output_prefix "$prefix" "$@"
}

invoke_training --model_type pinn --gridsize 2 --n_filters_scale 2 --object_big True --intensity_scale_trainable True --label "PINN,NLL,overlaps" --nimgs_train 2 --nimgs_test 1 --outer_offset_train "$outer_offset_train" --outer_offset_test "$outer_offset_test" --set_phi

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
