import optuna 
import time
import pickle
import random
import shutil 
import numpy as np
import torch
import torch.nn as nn

from utils.builders import new_folder
from utils.loaders import load_data

from trainers.teacher_student_trainer import cross_validation
from model_config import *

def train_main_model(dataset, model, view, cv_number, model_args, run=0):
    
    torch.manual_seed(run)
    np.random.seed(run)
    random.seed(run)

    model_strip = "_".join(model.split("_")[:2]) 
    cv_name = str(cv_number)+"Fold"
    model_name = "MainModel_"+cv_name+"_"+dataset+"_"+model_strip

    G_list = load_data(dataset, view, NormalizeInputGraphs=False)
    
    if model_args["evaluation_method"] == "model_assessment" or model_args["evaluation_method"] == "model_selection":
            model_name += f"_run_{run}_fixed_init"
    
    if model_args["model_name"] == "gcn_student" or model_args["model_name"] == "gat_student":
        return cross_validation(model_args, G_list, view, model_name, cv_number, run)

def objective(trial):

    run = 0
    datasets_asdnc="gender_data"
    views=[0]
    model = gcn_student_args
    # Define the hyperparameters to optimize
    lr = trial.suggest_categorical("lr", [1e-3, 1e-4, 1e-5, 1e-6])
    alpha_ce = trial.suggest_categorical("alpha_ce", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2])
    alpha_soft_ce = trial.suggest_categorical("alpha_soft_ce", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    validation_loss_all = []
    for dataset_i in datasets_asdnc:
        if dataset_i == "gender_data":
            for view_i in views:
                for cv in [3]:
                    model["lr"] = lr
                    model["alpha_ce"] = alpha_ce
                    model["alpha_soft_ce"] = alpha_soft_ce   
                    validation_loss = train_main_model(dataset_i, model["model_name"], view_i, cv, model, run)
                    validation_loss_all.append(validation_loss)
        else:
            for cv in [3]:
                model["lr"] = lr
                model["alpha_ce"] = alpha_ce
                model["alpha_soft_ce"] = alpha_soft_ce  
                validation_loss = train_main_model(dataset_i, model["model_name"], -1, cv, model, run)
                validation_loss_all.append(validation_loss)
    
    # Optimize for higher accuracy
    return np.mean(validation_loss_all)

# Create an Optuna study object and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=400)

# Print the best hyperparameters and their corresponding accuracy
best_params = study.best_params
best_accuracy = study.best_value
print('Best Hyperparameters:', best_params)
print('Best Accuracy:', best_accuracy)