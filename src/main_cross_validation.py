import os
import torch
import random
import argparse
import numpy as np

from trainers.model_trainer import cross_validation

from models.model_config import * 

from utils.builders import new_folder
from utils.loaders import load_data

import joblib
from joblib import Parallel, delayed

def train_main_model(dataset, model, view, cv_number, model_args, run=0):
    """
    Parameters
    ----------
    dataset : dataset
    model : GNN model (diffpool, gat, gcn, gunet or sag)
    view : index of cortical morphological network. 
    
    Description
    ----------
    This method trains selected GNN model with 5-Fold Cross Validation.
    
    """    
    torch.manual_seed(run)
    np.random.seed(run)
    random.seed(run)

    name = str(cv_number)+"Fold"
    model_name = "MainModel_"+name+"_"+dataset+"_"+model
    
    G_list = load_data(dataset, view, NormalizeInputGraphs=False)

    new_folder(model, model_args["evaluation_method"], backbone=model_args["backbone"])
        
    if model=='gcn' and model_args["layers"]==2:
        if model_args["evaluation_method"] == "model_assessment":
            model_name += f"_run_{run}_fixed_init"
        cross_validation(model_args, G_list, view, model_name, cv_number, run)
    
    if model=='gcn' and model_args["layers"]!=2:
        if model_args["evaluation_method"] == "model_assessment":
            layers = model_args["layers"]
            model_name += f"_run_{run}_fixed_init_layers_{str(layers)}"
        cross_validation(model_args, G_list, view, model_name, cv_number, run)
    
    elif model=='gcn_student':
        if model_args["evaluation_method"] == "model_assessment":
            model_name += f"_run_{run}_fixed_init"
        cross_validation(model_args, G_list, view, model_name, cv_number, run)
    
    elif model=='mlp':
        if model_args["evaluation_method"] == "model_assessment":
            model_name += f"_run_{run}_fixed_init"
        cross_validation(model_args, G_list, view, model_name, cv_number, run)
    
    elif model=='gat':
        if model_args["evaluation_method"] == "model_assessment":
            model_name += f"_run_{run}_fixed_init"
        cross_validation(model_args, G_list, view, model_name, cv_number, run)
    
def parrallel_run(run):
    print(run)
    datasets = ['gender_data']
    views = [0, 2, 4, 5] #0, 2, 4, 5
    for dataset_i in datasets:
        for view_i in views:
            models = [gat_args] # "gcn", "gcn_student" "gcn_3_args" args  gcn_student_args
            for model in models:
                for cv in [3, 5, 10]:
                    train_main_model(dataset_i, model["model_name"], view_i, cv, model, run)
                
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
        Parallel(n_jobs=n_jobs)(delayed(parrallel_run)(run) for run in runs)
        