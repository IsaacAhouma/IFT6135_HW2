#!/bin/bash

#PBS -A colosse-users
#PBS -l feature=k80
#PBS -l nodes=1:gpus=1
#PBS -l walltime=10:00:00

s_exec python 'ptb-lm.py' --model=RNN --optimizer=SGD --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35

# echo "Copying files to local hard drive..."
# cp -r $TMP_RESULTS_DIR $ROOT_DIR

# echo "Cleaning up data and results..."
# rm -r $TMP_DATA_DIR
# rm -r $TMP_RESULTS_DIR
