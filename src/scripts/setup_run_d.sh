#!/bin/bash

# Activate virtual environment
source /vol/bitbucket/ls1121/doscond/bin/activate

# Set PATH environment variable
export PATH=/vol/cuda/10.1.243/bin:$PATH

# Set CPATH environment variable
export CPATH=/vol/cuda/10.1.243/include:$CPATH

# Set LD_LIBRARY_PATH environment variable
export LD_LIBRARY_PATH=/vol/cuda/10.1.243/lib64:$LD_LIBRARY_PATH

#ABLATION STUDY FOR BreastMNIST 
python /homes/ls1121/thesis-imperial/src/main_cross_validation.py  --runs 0 1 2 3 4 --n_jobs 5 
