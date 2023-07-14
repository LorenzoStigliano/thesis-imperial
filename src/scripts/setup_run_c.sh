#!/bin/bash

# Activate virtual environment
source /vol/bitbucket/ls1121/doscond/bin/activate

# Set PATH environment variable
export PATH=/vol/cuda/10.1.243/bin:$PATH

# Set CPATH environment variable
export CPATH=/vol/cuda/10.1.243/include:$CPATH

# Set LD_LIBRARY_PATH environment variable
export LD_LIBRARY_PATH=/vol/cuda/10.1.243/lib64:$LD_LIBRARY_PATH

#HYPERPARAMETER SENSITIVITY - Disentanglement 
python /homes/ls1121/thesis-imperial/src/main_ensamble_ablation_1_lambda.py  --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 10 
python /homes/ls1121/thesis-imperial/src/main_ensamble_ablation_2_lambda.py  --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 10 
python /homes/ls1121/thesis-imperial/src/main_ensamble_ablation_3_lambda.py  --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 10 
python /homes/ls1121/thesis-imperial/src/main_ensamble_ablation_4_lambda.py  --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 10 
python /homes/ls1121/thesis-imperial/src/main_ensamble_ablation_5_lambda.py  --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 10 
python /homes/ls1121/thesis-imperial/src/main_ensamble_ablation_6_lambda.py  --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 10 

#HYPERPARAMETER SENSITIVITY - Student ensemble  
python /homes/ls1121/thesis-imperial/src/main_ensamble_ablation_1_gamma.py  --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 10 
python /homes/ls1121/thesis-imperial/src/main_ensamble_ablation_2_gamma.py  --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 10 
python /homes/ls1121/thesis-imperial/src/main_ensamble_ablation_3_gamma.py  --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 10 
python /homes/ls1121/thesis-imperial/src/main_ensamble_ablation_4_gamma.py  --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 10 
python /homes/ls1121/thesis-imperial/src/main_ensamble_ablation_5_gamma.py  --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 10 
python /homes/ls1121/thesis-imperial/src/main_ensamble_ablation_6_gamma.py  --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 10 
python /homes/ls1121/thesis-imperial/src/main_ensamble_ablation_7_gamma.py  --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 10 
