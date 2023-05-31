#!/bin/bash
#dsource=grf
#prefix=grf2
dsource=$1
prefix=$2
nepochs=$3

# Invocation 1
python ptycho/train.py  --data_source $dsource --nepochs $nepochs --offset 4  --output_prefix $prefix --model_type pinn --gridsize 2 --n_filters_scale 2 --object_big True --intensity_scale_trainable True --label "PINN,NLL,overlaps" --nimgs_train 1 --nimgs_test 1 --outer_offset_train 4 --outer_offset_test 12

# Invocation 2
python ptycho/train.py  --data_source $dsource --nepochs $nepochs --offset 4  --output_prefix $prefix --model_type pinn --nll_weight 0.0 --mae_weight 1.0 --intensity_scale_trainable True --object_big True --n_filters_scale 2 --label "PINN,overlaps" --nimgs_train 1 --nimgs_test 1 --outer_offset_train 4 --outer_offset_test 12

# Invocation 3
python ptycho/train.py  --data_source $dsource --nepochs $nepochs --offset 4  --output_prefix $prefix --model_type supervised --n_filters_scale 1 --gridsize 2 --label "overlaps" --nimgs_train 1 --nimgs_test 1 --outer_offset_train 4 --outer_offset_test 12

# Invocation 4
python ptycho/train.py  --data_source $dsource --nepochs $nepochs --offset 4  --output_prefix $prefix --model_type pinn --gridsize 1 --nll_weight 1.0 --mae_weight 0.0 --label "PINN,NLL" --nimgs_train 1 --nimgs_test 1 --outer_offset_train 4 --outer_offset_test 12

# Invocation 5
python ptycho/train.py  --data_source $dsource --nepochs $nepochs --offset 4  --output_prefix $prefix --model_type pinn --gridsize 1 --nll_weight 0.0 --mae_weight 1.0 --label "PINN" --nimgs_train 1 --nimgs_test 1 --outer_offset_train 4 --outer_offset_test 12

# Invocation 6
python ptycho/train.py  --data_source $dsource --nepochs $nepochs --offset 4  --output_prefix $prefix --model_type supervised --gridsize 1 --n_filters_scale 1 --label "none" --nimgs_train 1 --nimgs_test 1 --outer_offset_train 4 --outer_offset_test 12
