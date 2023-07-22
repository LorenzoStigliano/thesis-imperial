#!/bin/bash

# Activate virtual environment
source /vol/bitbucket/ls1121/doscond/bin/activate

# Set PATH environment variable
export PATH=/vol/cuda/10.1.243/bin:$PATH

# Set CPATH environment variable
export CPATH=/vol/cuda/10.1.243/include:$CPATH

# Set LD_LIBRARY_PATH environment variable
export LD_LIBRARY_PATH=/vol/cuda/10.1.243/lib64:$LD_LIBRARY_PATH

#GCN -> GAT GSP 
python /homes/ls1121/thesis-imperial/src/main_teacher_student.py  --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 10 
#VANILLA EXPLORATION
python /homes/ls1121/thesis-imperial/src/main_teacher_student_1.py  --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 10
python /homes/ls1121/thesis-imperial/src/main_teacher_student_2.py  --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 10
python /homes/ls1121/thesis-imperial/src/main_teacher_student_3.py  --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 10
python /homes/ls1121/thesis-imperial/src/main_teacher_student_4.py  --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 10
python /homes/ls1121/thesis-imperial/src/main_teacher_student_5.py  --runs 0 1 2 3 4 5 6 7 8 9 --n_jobs 10
