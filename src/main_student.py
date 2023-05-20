import argparse
import random

from models.model_config import * 
from trainers.student_trainer import test_scores

from utils.helpers import *
from utils.builders import new_folder
from utils.loaders import load_data

def train_main_model(dataset,model,view, cv_number):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)   

    name = str(cv_number)+"Fold"
    model_name = "MainModel_"+name+"_"+dataset+"_"+model

    G_list = load_data(dataset, view, NormalizeInputGraphs=False)
    
    new_folder(model)
    test_scores(gcn_student_args, G_list, view, model_name, cv_number)
        
if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'results'])
    args = parser.parse_args()
    
    if args.mode == 'train':
        '''
        Training GNN Models with datasets of data directory.
        '''
        datasets_asdnc = ['gender_data']
        views = [0, 1, 2, 3, 4, 5] #0, 1, 2, 3, 4, 5
        for dataset_i in datasets_asdnc:
            for view_i in views:
                models = ["gcn_student"]
                for model in models:
                    for cv in [3, 5, 10]:
                        train_main_model(dataset_i, model, view_i, cv)