import argparse
import random
import numpy as np
import torch

from models.model_config import * 
from trainers.mskd_trainer import cross_validation

from utils.builders import new_folder
from utils.loaders import load_data

import joblib
from joblib import Parallel, delayed

def train_main_model(dataset, model, view, cv_number, run=0):
    
    torch.manual_seed(run)
    np.random.seed(run)
    random.seed(run)

    cv_name = str(cv_number)+"Fold"
    model_name = "MainModel_"+cv_name+"_"+dataset+"_"+model

    G_list = load_data(dataset, view, NormalizeInputGraphs=False)

    new_folder(model, gcn_student_args["evaluation_method"])
    
    if gcn_args["evaluation_method"] == "model_assessment":
            model_name += f"_run_{run}_fixed_init"
    
    if model == "mskd":
        cross_validation(mskd_student_args, G_list, view, model_name, cv_number, run)

def parrallel_run(run):
    print(run)
    datasets_asdnc = ['gender_data']
    views = [0, 2, 4, 5] #0, 2, 4, 5
    for dataset_i in datasets_asdnc:
        for view_i in views:
            models = ["mskd"] #"gcn_student", "gcn_student_teacher_weight"
            for model in models:
                for cv in [3, 5, 10]:
                    train_main_model(dataset_i, model, view_i, cv, run)

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'results'])
    parser.add_argument('--runs', nargs='+', help='Enter a list of seeds for the runs: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9', default=[0])
    parser.add_argument('--n_jobs', type=int, help='parrallel jobs', default=1)

    args = parser.parse_args()
    runs = [int(run) for run in args.runs]
    n_jobs = int(args.n_jobs)
    
    if args.mode == 'train':
        '''
        Training GNN Models with datasets of data directory.
        '''
        runs = runs # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        #joblib.cpu_count()
        Parallel(n_jobs=n_jobs)(delayed(parrallel_run)(run) for run in runs)