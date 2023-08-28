#!/bin/bash

# Activate virtual environment
source /Users/lorenzostigliano/Documents/University/Imperial/env-test/bin/activate

# Navigate to the src directory
cd ../

python main_cross_validation.py --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 1 --dataset="gender_data"
python main_cross_validation.py --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 1 --dataset="BreastMNIST"