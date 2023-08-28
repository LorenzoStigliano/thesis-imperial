#!/bin/bash

# Activate virtual environment
source /Users/lorenzostigliano/Documents/University/Imperial/env-test/bin/activate

# Navigate to the src directory
cd ../

# Set PATH environment variable
export PATH=/vol/cuda/10.1.243/bin:$PATH

# Set CPATH environment variable
export CPATH=/vol/cuda/10.1.243/include:$CPATH

# Set LD_LIBRARY_PATH environment variable
export LD_LIBRARY_PATH=/vol/cuda/10.1.243/lib64:$LD_LIBRARY_PATH

python main_cross_validation.py --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 1 --dataset="gender_data"
python main_cross_validation.py --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 1 --dataset="BreastMNIST"