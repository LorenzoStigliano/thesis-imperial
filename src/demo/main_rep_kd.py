import argparse
import random
import numpy as np
import torch

from model_config_GSP import * 

from trainers.rep_kd_ensemble.rep_kd_ensemble_2_trainer import lsp_cross_validation_2
from trainers.rep_kd_ensemble.rep_kd_ensemble_3_trainer import lsp_cross_validation_3
from trainers.rep_kd_ensemble.rep_kd_ensemble_4_trainer import lsp_cross_validation_4
from trainers.rep_kd_ensemble.rep_kd_ensemble_5_trainer import lsp_cross_validation_5

from builders import new_folder
from loaders import load_data
from config import SAVE_DIR_MODEL_DATA, SAVE_DIR_DATA

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

    cv_name = str(cv_number)+"Fold"
    model_name = "MainModel_"+cv_name+"_"+dataset+"_"+model_args["model_name"]

    G_list = load_data(dataset, view, SAVE_DIR_DATA, NormalizeInputGraphs=False)

    new_folder(model_args["model_name"], model_args["evaluation_method"], SAVE_DIR_MODEL_DATA, backbone=model_args["backbone"], dataset=model_args["dataset"])
    
    if model_args["evaluation_method"] == "model_assessment" or  model_args["evaluation_method"] == "model_selection":
            model_name += f"_run_{run}_fixed_init"
            
    if model_args["model_name"] == "gcn_student_lsp_ensamble_2":
        lsp_cross_validation_2(model_args, G_list, view, model_name, cv_number, n_students=2, run=run)
    if model_args["model_name"] == "gcn_student_lsp_ensamble_3" :
        lsp_cross_validation_3(model_args, G_list, view, model_name, cv_number, n_students=3, run=run)
    if model_args["model_name"] == "gcn_student_lsp_ensamble_4":
        lsp_cross_validation_4(model_args, G_list, view, model_name, cv_number, n_students=4, run=run)
    if model_args["model_name"] == "gcn_student_lsp_ensamble_5":
        lsp_cross_validation_5(model_args, G_list, view, model_name, cv_number, n_students=5, run=run)

def parrallel_run_B(run, dataset):
    """
    Execute training of GNN models in parallel for the specified run and dataset.

    Parameters:
        run (int): Seed for the current run.
        dataset (str): Name of the dataset.

    Note:
        This function is meant to be used in parallel processing for training models.
    """
    print(run)
    views = [0, 2, 4, 5] #0, 2, 4, 5
    if dataset == "gender_data":
        for view_i in views:
            #ADD MODEL CONFIG HERE 
            models = [ 
                gcn_student_lsp_ensamble_2_args,
                gcn_student_lsp_ensamble_3_args,
                gcn_student_lsp_ensamble_4_args,
                gcn_student_lsp_ensamble_5_args
            ] 
            for model in models:
                for cv in [3, 5, 10]:
                    train_main_model(dataset, model["model_name"], view_i, cv, model, run)
