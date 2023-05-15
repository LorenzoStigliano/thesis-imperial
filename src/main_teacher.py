import time
import argparse
import random

from models.model_config import * 
from trainers.teacher_trainer import train_teacher

from utils.helpers import *
from utils.builders import new_folder
from utils.loaders import load_data

def train_main_model(dataset, model, view):
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

    model_name = "MainModel_"+dataset+"_"+model

    G_list = load_data(dataset, view, NormalizeInputGraphs=False)
    
    new_folder(model)
    train_teacher(gcn_args, G_list, view, model_name)
        

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    
    '''
    Training GNN Models with datasets of data directory.
    '''
    datasets = ['gender_data']
    views = [0] #0, 1, 2, 3, 4, 5
    for dataset in datasets:
        for view in views:
            models = ["gcn"]
            for model in models:
                    train_main_model(dataset, model, view)