#!/bin/bash

# Activate virtual environment
source /vol/bitbucket/ls1121/doscond/bin/activate

python /homes/ls1121/thesis-imperial/src/main_rep_kd.py --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 1 --dataset="gender_data"
python /homes/ls1121/thesis-imperial/src/main_rep_kd.py --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 1 --dataset="BreastMNIST"