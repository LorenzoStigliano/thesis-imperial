import pickle
import numpy as np 
from config import SAVE_DIR_MODEL_DATA

def get_labels_and_preds(dataset, model, analysis_type, training_type, cv_n, view, run, dataset_split):
    if analysis_type == "model_assessment":
        if "teacher" in model:
            if "weight" in model:
                model = "_".join(model.split("_")[:2]) 
                cv_path = SAVE_DIR_MODEL_DATA+f'model_assessment/{model}/labels_and_preds/MainModel_{training_type}_{dataset}_{model}_run_{run}_CV_{cv_n}_view_{view}_with_teacher_weight_matching_{dataset_split}.pickle'
            else:
                model = "_".join(model.split("_")[:2]) 
                cv_path = SAVE_DIR_MODEL_DATA+f'model_assessment/{model}/labels_and_preds/MainModel_{training_type}_{dataset}_{model}_run_{run}_CV_{cv_n}_view_{view}_with_teacher_{dataset_split}.pickle'    
        else:
            cv_path = SAVE_DIR_MODEL_DATA+f'model_assessment/{model}/labels_and_preds/MainModel_{training_type}_{dataset}_{model}_run_{run}_CV_{cv_n}_view_{view}_{dataset_split}.pickle'   
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
                cv_path = SAVE_DIR_MODEL_DATA+f'model_assessment/{model}/metrics/MainModel_{training_type}_{dataset}_{model}_run_{run}_CV_{cv_n}_view_{view}_with_teacher_weight_matching_{dataset_split}_{metric}.pickle'
            else:
                model = "_".join(model.split("_")[:2]) 
                cv_path = SAVE_DIR_MODEL_DATA+f'model_assessment/{model}/metrics/MainModel_{training_type}_{dataset}_{model}_run_{run}_CV_{cv_n}_view_{view}_with_teacher_{dataset_split}_{metric}.pickle'    
        else:
            cv_path = SAVE_DIR_MODEL_DATA+f'model_assessment/{model}/metrics/MainModel_{training_type}_{dataset}_{model}_run_{run}_CV_{cv_n}_view_{view}_{dataset_split}_{metric}.pickle'   
    #TODO: model_selection
    else:
        return 
    
    with open(cv_path,'rb') as f:
        metrics = pickle.load(f)

    return metrics