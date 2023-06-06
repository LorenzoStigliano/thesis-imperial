import torch
import pickle
import io
import numpy as np 
from config import SAVE_DIR_MODEL_DATA

############ GETTERS FOR LABELS OF MODELS ############

def get_labels_and_preds(dataset, model, analysis_type, training_type, cv_n, view, run, dataset_split):
    if analysis_type == "model_assessment":
        if "teacher" in model:
            if "weight" in model:
                model = "_".join(model.split("_")[:2]) 
                cv_path = SAVE_DIR_MODEL_DATA+f'model_assessment/{model}/labels_and_preds/MainModel_{training_type}_{dataset}_{model}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_with_teacher_weight_matching_{dataset_split}.pickle'
            else:
                model = "_".join(model.split("_")[:2]) 
                cv_path = SAVE_DIR_MODEL_DATA+f'model_assessment/{model}/labels_and_preds/MainModel_{training_type}_{dataset}_{model}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_with_teacher_{dataset_split}.pickle'    
        else:
            cv_path = SAVE_DIR_MODEL_DATA+f'model_assessment/{model}/labels_and_preds/MainModel_{training_type}_{dataset}_{model}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_{dataset_split}.pickle'   
    #TODO: model_selection
    else:
        return 
    
    with open(cv_path,'rb') as f:
        labels_and_preds = pickle.load(f)

    return labels_and_preds

############ GETTERS FOR METRICS OF MODELS ############

def extract_metrics(dataset, model, analysis_type, training_type, view, run, dataset_split, metric):
    metrics = []
    if training_type == '3Fold':
        for cv_i in range(3):
            metrics.append(get_metrics(dataset, model, analysis_type, training_type, cv_i, view, run, dataset_split, metric))
    if training_type == '5Fold':
        for cv_i in range(5):
            metrics.append(get_metrics(dataset, model, analysis_type, training_type, cv_i, view, run, dataset_split, metric))
    if training_type == '10Fold':
        for cv_i in range(10):
            metrics.append(get_metrics(dataset, model, analysis_type, training_type, cv_i, view, run, dataset_split, metric))
    metrics = np.array(metrics)
    return metrics

def get_metrics(dataset, model, analysis_type, training_type, cv_n, view, run, dataset_split, metric):
    if analysis_type == "model_assessment":
        if "teacher" in model:
            if "weight" in model:
                model = "_".join(model.split("_")[:2]) 
                cv_path = SAVE_DIR_MODEL_DATA+f'model_assessment/{model}/metrics/MainModel_{training_type}_{dataset}_{model}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_with_teacher_weight_matching_{dataset_split}_{metric}.pickle'
            else:
                model = "_".join(model.split("_")[:2]) 
                cv_path = SAVE_DIR_MODEL_DATA+f'model_assessment/{model}/metrics/MainModel_{training_type}_{dataset}_{model}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_with_teacher_{dataset_split}_{metric}.pickle'    
        else:
            cv_path = SAVE_DIR_MODEL_DATA+f'model_assessment/{model}/metrics/MainModel_{training_type}_{dataset}_{model}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_{dataset_split}_{metric}.pickle'   
    #TODO: model_selection
    else:
        return 
    
    with open(cv_path,'rb') as f:
        metrics = pickle.load(f)

    return metrics

############ GETTERS FOR WEIGHTS OF MODELS ############

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def extract_weights(dataset, view, model, training_type, run):
    runs = []
    if training_type == '3Fold':
        for cv_i in range(3):
            runs.append(get_weight(dataset, view, model, training_type, 0, cv_i, run))
    if training_type == '5Fold':
        for cv_i in range(5):
            runs.append(get_weight(dataset, view, model, training_type, 0, cv_i, run))
    if training_type == '10Fold':
        for cv_i in range(10):
            runs.append(get_weight(dataset, view, model, training_type, 0, cv_i, run))
    runs = np.array(runs)
    weights = np.mean(runs, axis=0)
    return weights

def get_weight(dataset, view, model, training_type, shot_n, cv_n, run):
    
    if "teacher" in model:
        if "weight" in model:
            model = "_".join(model.split("_")[:2]) 
            cv_path = SAVE_DIR_MODEL_DATA+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_fixed_init_CV_{}_view_{}_with_teacher_weight_matching.pickle'.format(model, training_type, dataset, model, run, cv_n, view)
        else:
            model = "_".join(model.split("_")[:2]) 
            cv_path = SAVE_DIR_MODEL_DATA+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_fixed_init_CV_{}_view_{}_with_teacher.pickle'.format(model, training_type, dataset, model, run, cv_n, view)        
    else:
        cv_path = SAVE_DIR_MODEL_DATA+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_fixed_init_CV_{}_view_{}.pickle'.format(model,training_type, dataset, model, run, cv_n, view)

    x_path = cv_path 
    with open(x_path,'rb') as f:
        weights = pickle.load(f)

    if model == 'sag':
        weights_vector = torch.mean(weights['w'], 1).detach().numpy()
    if model == 'diffpool':
        weights_vector = torch.mean(weights['w'], 1).detach().numpy()
    if model == 'gcn':
        weights_vector = weights['w'].squeeze().detach().numpy()
    if model == 'gcn_student':
        weights_vector = weights['w'].squeeze().detach().numpy()
    if model == 'gat':
        weights_vector = weights['w'].squeeze().detach().numpy()
    if model == 'gunet':
        weights_vector = torch.mean(weights['w'], 0).detach().numpy()    
    return weights_vector