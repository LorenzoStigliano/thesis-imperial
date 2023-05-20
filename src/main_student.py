import argparse
import random

from models.model_config import * 
from trainers.student_trainer import cross_validation

from utils.helpers import *
from utils.builders import new_folder
from utils.loaders import load_data

def train_main_model(dataset, model, view, cv_number, run=0):
    
    torch.manual_seed(run)
    np.random.seed(run)
    random.seed(run)   

    cv_name = str(cv_number)+"Fold"
    model_name = "MainModel_"+cv_name+"_"+dataset+"_"+model

    G_list = load_data(dataset, view, NormalizeInputGraphs=False)

    new_folder(model, gcn_student_args["evaluation_method"])
    
    if gcn_args["evaluation_method"] == "model_assessment":
            model_name += f"_run_{run}"
    
    cross_validation(gcn_student_args, G_list, view, model_name, cv_number)
        
if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'results'])
    args = parser.parse_args()
    
    if args.mode == 'train':
        '''
        Training GNN Models with datasets of data directory.
        '''
        runs = [1,2] #0, 1 ,2
        datasets_asdnc = ['gender_data']
        views = [0, 2, 4, 5] #0, 1, 2, 3, 4, 5
        for run in runs:
            for dataset_i in datasets_asdnc:
                for view_i in views:
                    models = ["gcn_student"]
                    for model in models:
                        for cv in [3, 5, 10]:
                            train_main_model(dataset_i, model, view_i, cv, run)