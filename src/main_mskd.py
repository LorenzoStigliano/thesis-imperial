import argparse
import random
import numpy as np
import torch

from models_config.model_config_GSP import * 
from models_config.model_config_BreastMNIST import * 
from trainers.mskd_trainer import cross_validation

from utils.builders import new_folder
from utils.loaders import load_data
from utils.config import SAVE_DIR_MODEL_DATA, SAVE_DIR_DATA

import joblib
from joblib import Parallel, delayed

def train_main_model(dataset, model, view, cv_number, model_args, run=0):
    
    torch.manual_seed(run)
    np.random.seed(run)
    random.seed(run)

    cv_name = str(cv_number)+"Fold"
    model_name = "MainModel_"+cv_name+"_"+dataset+"_"+model

    G_list = load_data(dataset, view, SAVE_DIR_DATA, NormalizeInputGraphs=False)

    new_folder(model, model_args["evaluation_method"], SAVE_DIR_MODEL_DATA, dataset=model_args["dataset"], backbone=model_args["backbone"])
    
    if model_args["evaluation_method"] == "model_assessment":
            model_name += f"_run_{run}_fixed_init"
    
    if model_args["model_name"] == "mskd" or model_args["model_name"] == "mskd_gat":
        cross_validation(model_args, G_list, view, model_name, cv_number, run)

def parrallel_run(run, dataset):
    print(run)
    views = [0, 2, 4, 5] #0, 2, 4, 5
    if dataset == "gender_data":
        for view_i in views:
            models = [gcn_mskd_student_args] #ADD MODEL CONFIG HERE 
            for model in models:
                for cv in [3, 5, 10]:
                    train_main_model(dataset, model["model_name"], view_i, cv, model, run)
    else:
        models = [gcn_mskd_student_BreastMNIST_args] #ADD MODEL CONFIG HERE 
        for model in models:
            for cv in [3, 5, 10]:
                train_main_model(dataset, model["model_name"], -1, cv, model, run) 

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train'])
    parser.add_argument('--runs', nargs='+', help='Enter a list of seeds for the runs: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9', default=[0])
    parser.add_argument('--n_jobs', type=int, help='parrallel jobs', default=1)
    parser.add_argument('--dataset', type=str, help='dataset to use', default='gender_data', choices=['gender_data', 'BreastMNIST'])

    args = parser.parse_args()
    runs = [int(run) for run in args.runs]
    n_jobs = int(args.n_jobs)
    dataset = str(args.dataset)

    if args.mode == 'train':
        '''
        Training GNN Models with datasets of data directory.
        '''
        runs = runs # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        Parallel(n_jobs=n_jobs)(delayed(parrallel_run)(run, dataset) for run in runs)
