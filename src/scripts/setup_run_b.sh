#!/bin/bash

# Activate virtual environment
source /vol/bitbucket/ls1121/doscond/bin/activate

# Set PATH environment variable
export PATH=/vol/cuda/10.1.243/bin:$PATH

# Set CPATH environment variable
export CPATH=/vol/cuda/10.1.243/include:$CPATH

# Set LD_LIBRARY_PATH environment variable
export LD_LIBRARY_PATH=/vol/cuda/10.1.243/lib64:$LD_LIBRARY_PATH

#GAT FOR GSP 
python /homes/ls1121/thesis-imperial/src/main_teacher_student.py  --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 5 

#ON HOLD:

#python /homes/ls1121/thesis-imperial/src/main_lsp.py  --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 5 
#python /homes/ls1121/thesis-imperial/src/main_fitnet.py  --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 5 
#python /homes/ls1121/thesis-imperial/src/main_fitnet_1.py  --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 5
#TODO: python /homes/ls1121/thesis-imperial/src/main_ensamble.py  --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 3  
#TODO: #python /homes/ls1121/thesis-imperial/src/main_lsp_ensamble.py  --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 3 
