import os
import torch
import random
import argparse
import numpy as np

from trainers.model_trainer import cross_validation

from models.model_config import * 

from utils.builders import new_folder

from utils.loaders import load_data

def train_main_model(dataset, model, view, cv_number, run=0):
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

    new_folder(model, gcn_args["evaluation_method"])
        
    if model=='gcn':
        if gcn_args["evaluation_method"] == "model_assessment":
            model_name += f"_run_{run}_fixed_init"
        cross_validation(gcn_args, G_list, view, model_name, cv_number, run)
    
    elif model=='gcn_student':
        if gcn_student_args["evaluation_method"] == "model_assessment":
            model_name += f"_run_{run}_fixed_init"
        cross_validation(gcn_student_args, G_list, view, model_name, cv_number, run)
    
    elif model=='mlp':
        if mlp_args["evaluation_method"] == "model_assessment":
            model_name += f"_run_{run}_fixed_init"
        cross_validation(mlp_args, G_list, view, model_name, cv_number, run)
    
if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'results'])
    args = parser.parse_args()
    
    if args.mode == 'train':
        '''
        Training GNN Models with datasets of data directory.
        '''
        runs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        datasets_asdnc = ['gender_data']
        views = [0, 2, 4, 5] #0, 2, 4, 5
        for run in runs:
            for dataset_i in datasets_asdnc:
                for view_i in views:
                    models = ["gcn", "gcn_student"] # "gcn", "gcn_student"
                    for model in models:
                        for cv in [3, 5, 10]:
                            train_main_model(dataset_i, model, view_i, cv, run)
                  