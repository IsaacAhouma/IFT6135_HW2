#!/bin/bash

#PBS -A colosse-users
#PBS -l feature=k20
#PBS -l nodes=1:gpus=1
#PBS -l walltime=10:00:00

s_exec python 'ptb-lm.py' --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=.9
# echo "Copying files to local hard drive..."
# cp -r $TMP_RESULTS_DIR $ROOT_DIR
# echo "Cleaning up data and results..."
# rm -r $TMP_DATA_DIR
# rm -r $TMP_RESULTS_DIR
