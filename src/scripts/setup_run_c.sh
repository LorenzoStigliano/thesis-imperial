#!/bin/bash

# Activate virtual environment
source /vol/bitbucket/ls1121/doscond/bin/activate

# Set PATH environment variable
export PATH=/vol/cuda/10.1.243/bin:$PATH

# Set CPATH environment variable
export CPATH=/vol/cuda/10.1.243/include:$CPATH

# Set LD_LIBRARY_PATH environment variable
export LD_LIBRARY_PATH=/vol/cuda/10.1.243/lib64:$LD_LIBRARY_PATH

# STILL NEED TO RUN GAT -> GAT
python /homes/ls1121/thesis-imperial/src/main_lsp.py  --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 3
python /homes/ls1121/thesis-imperial/src/main_fitnet.py  --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 3
python /homes/ls1121/thesis-imperial/src/main_fitnet_1.py  --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 3
