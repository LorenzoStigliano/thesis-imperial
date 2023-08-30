import os
import torch
import random
import argparse
import numpy as np

from trainers.model_trainer import cross_validation

from models_config.model_config_GSP import * 

from utils.builders import new_folder
from utils.loaders import load_data
from utils.config import SAVE_DIR_MODEL_DATA, SAVE_DIR_DATA

import joblib
from joblib import Parallel, delayed

def train_main_model(dataset, model, view, cv_number, model_args, run=0):
    """
    Train the main GNN model on the given dataset using cross-validation.

    Parameters:
        dataset (str): Name of the dataset.
        model (str): Name of the GNN model.
        view (int): View number (for multi-view datasets) or -1 for single-view datasets.
        cv_number (int): Number of folds for cross-validation.
        model_args (dict): Dictionary containing model configuration arguments.
        run (int, optional): Random seed for reproducibility. Defaults to 0.
    """  
    torch.manual_seed(run)
    np.random.seed(run)
    random.seed(run)

    name = str(cv_number)+"Fold"
    model_name = "MainModel_"+name+"_"+dataset+"_"+model
    
    G_list = load_data(dataset, view, SAVE_DIR_DATA, NormalizeInputGraphs=False)

    new_folder(model, model_args["evaluation_method"], SAVE_DIR_MODEL_DATA, dataset=model_args["dataset"], backbone=model_args["backbone"])
        
    if model=='gcn' and model_args["layers"]==2:
        if model_args["evaluation_method"] == "model_assessment" or  model_args["evaluation_method"] == "model_selection":
            model_name += f"_run_{run}_fixed_init"
        cross_validation(model_args, G_list, view, model_name, cv_number, run)
    else:
        if model_args["evaluation_method"] == "model_assessment" or  model_args["evaluation_method"] == "model_selection":
            model_name += f"_run_{run}_fixed_init"
        cross_validation(model_args, G_list, view, model_name, cv_number, run)

def parrallel_run_A(run, dataset):
    """
    Execute training of GNN models in parallel for the specified run and dataset.

    Parameters:
        run (int): Seed for the current run.
        dataset (str): Name of the dataset.

    Note:
        This function is meant to be used in parallel processing for training models.
    """
    views = [0] #0, 2, 4, 5
    if dataset == "gender_data":
        for view_i in views:
            models = [gcn_args, gcn_student_args] #ADD MODEL CONFIG HERE 
            for model in models:
                for cv in [3, 5, 10]:
                    train_main_model(dataset, model["model_name"], view_i, cv, model, run)

        