import os
import torch
import random
import argparse
import numpy as np

from utils.trainers.model_trainer import test_scores, two_shot_trainer
import utils.trainers.gunet_trainer as gunet_trainer
import utils.trainers.sag_trainer as sag_trainer

from models.model_config import * 

from utils.builders import new_folder

from utils.loaders import load_data

from utils.split_cv import transform_Data
from utils.split_few_shot import transform_Data_FewShot
from utils.analysis import  Rep_histograms, Models_trained, Rep_heatmap

def train_main_model(dataset,model,view, cv_number):
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
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)   

    name = str(cv_number)+"Fold"
    model_name = "MainModel_"+name+"_"+dataset+"_"+model
    
    new_folder(model)

    G_list = load_data(dataset, view, NormalizeInputGraphs=False)
        
    if model=='gcn':
        test_scores(gcn_args, G_list, view, model_name, cv_number)
    
    elif model=='gat':
        test_scores(gat_args, G_list, view, model_name, cv_number)
    
    elif model=='diffpool':
        test_scores(diffpool_args, G_list, view, model_name, cv_number)
    
    elif model == "gunet":
        transform_Data(cv_number, dataset)
        gunet_trainer.cv_benchmark(dataset, view, cv_number)
    
    elif model == "sag":
        transform_Data(cv_number, dataset)
        sag_trainer.cv_benchmark(dataset, view, cv_number)

def two_shot_train(dataset, model, view, num_shots):
    """
    Parameters
    ----------
    dataset : dataset
    model : GNN model (diffpool, gat, gcn, gunet or sag)
    view : index of cortical morphological network.
    
    Description
    ----------
    This method trains selected GNN model with Two shot learning.
    
    """
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    transform_Data_FewShot(dataset)
    new_folder(model)
    
    model_name = "Few_Shot_"+dataset+"_"+model
    
    if model == "gcn":
        two_shot_trainer(dataset, view, num_shots, model_name, gcn_args)
    
    elif model == "gat":
        two_shot_trainer(dataset, view, num_shots, model_name, gat_args)
    
    elif model == "diffpool":
        two_shot_trainer(dataset, view, num_shots, model_name, diffpool_args)
    
    elif model == "gunet":
        gunet_trainer.two_shot_trainer(dataset, view, num_shots)
    
    elif model == "sag":
        sag_trainer.two_shot_trainer(dataset, view, num_shots)

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'results'])
    #parser.add_argument('--v', type=str, default=0, help='index of cortical morphological network.')
    parser.add_argument('--cv_number', type=str, default=3, help='number of cross validations.')
    parser.add_argument('--num_shots', type=str, default=5, help='number of runs for the FS learning.')
    #parser.add_argument('--data', type=str, default='Demo', choices = [ f.path[5:] for f in os.scandir("data") if f.is_dir() ], help='Name of dataset')
    args = parser.parse_args()
    #view = args.v
    #dataset = args.data
    num_shots = args.num_shots
    cv_number = args.cv_number
    
    if args.mode == 'train':
        '''
        Training GNN Models with datasets of data directory.
        '''
        datasets_asdnc = ['gender_data']
        
        views = [1]
        for dataset_i in datasets_asdnc:
            for view_i in views:
                models = ["sag"]
                for model in models:
                    train_main_model(dataset_i, model, view_i, 3)
                    #two_shot_train(dataset_i, model, view_i, num_shots)
                   
            print("All GNN architectures are trained with dataset: "+dataset_i)
          
        
    elif args.mode == 'results':
        '''
        if Models_trained(dataset, view):
            print("Models are not trained with"+dataset+" dataset view:"+str(view))
        else:
            Rep_histograms(dataset, view)
            Rep_heatmap(dataset, view)
            print("Reproducibility Histogram of dataset "+dataset+" is saved into results file.")
        '''
 