#!/bin/bash

# Activate virtual environment
source /vol/bitbucket/ls1121/doscond/bin/activate

# Set PATH environment variable
export PATH=/vol/cuda/10.1.243/bin:$PATH

# Set CPATH environment variable
export CPATH=/vol/cuda/10.1.243/include:$CPATH

# Set LD_LIBRARY_PATH environment variable
export LD_LIBRARY_PATH=/vol/cuda/10.1.243/lib64:$LD_LIBRARY_PATH

# STILL NEED TO RUN GAT -> GCN
python /homes/ls1121/thesis-imperial/src/main_lsp_ensamble_BreastMNIST_1.py  --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 10
python /homes/ls1121/thesis-imperial/src/main_lsp_ensamble_BreastMNIST_4.py  --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 10
python /homes/ls1121/thesis-imperial/src/main_lsp_ensamble_BreastMNIST_2.py  --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 10
python /homes/ls1121/thesis-imperial/src/main_lsp_ensamble_BreastMNIST_3.py  --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 10