import torch
import pickle
import io
import numpy as np 
from config import SAVE_DIR_MODEL_DATA

############ GETTERS FOR LABELS OF MODELS ############

def get_labels_and_preds(dataset, model, analysis_type, training_type, cv_n, view, run, dataset_split, student=0, model_args=None):
    if analysis_type == "model_assessment":
        if "ensamble" in model:
            alpha = str(model_args["alpha"])
            beta = str(model_args["beta"])
            gamma = str(model_args["gamma"])
            lambda_ = str(model_args["lambda"])
            if student == -1:
                cv_path =  SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/model_assessment/{model}/labels_and_preds/MainModel_{training_type}_{dataset}_{model_args["model_name"]}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_alpha_{alpha}_beta_{beta}_gamma_{gamma}_lambda_{lambda_}_{dataset_split}_ensamble.pickle'
            else:
                cv_path =  SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/model_assessment/{model}/labels_and_preds/MainModel_{training_type}_{dataset}_{model_args["model_name"]}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_alpha_{alpha}_beta_{beta}_gamma_{gamma}_lambda_{lambda_}_{dataset_split}_student_{student}.pickle'
        
        elif "teacher" in model:
            if "weight" in model:
                model = "_".join(model.split("_")[:2]) 
                cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/model_assessment/{model}/labels_and_preds/MainModel_{training_type}_{dataset}_{model}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_with_teacher_weight_matching_{dataset_split}.pickle'
            else:
                model = "_".join(model.split("_")[:2]) 
                cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/model_assessment/{model}/labels_and_preds/MainModel_{training_type}_{dataset}_{model}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_with_teacher_{dataset_split}.pickle'    
        
        else:
            cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/model_assessment/{model}/labels_and_preds/MainModel_{training_type}_{dataset}_{model}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_{dataset_split}.pickle'   
    #TODO: model_selection
    else:
        return 
    with open(cv_path,'rb') as f:
        labels_and_preds = pickle.load(f)
    return labels_and_preds

############ GETTERS FOR METRICS OF MODELS ############

def extract_metrics(dataset, model, analysis_type, training_type, view, run, dataset_split, metric, model_args=None):
    metrics = []
    if training_type == '3Fold':
        for cv_i in range(3):
            metrics.append(get_metrics(dataset, model, analysis_type, training_type, cv_i, view, run, dataset_split, metric, model_args))
    if training_type == '5Fold':
        for cv_i in range(5):
            metrics.append(get_metrics(dataset, model, analysis_type, training_type, cv_i, view, run, dataset_split, metric, model_args))
    if training_type == '10Fold':
        for cv_i in range(10):
            metrics.append(get_metrics(dataset, model, analysis_type, training_type, cv_i, view, run, dataset_split, metric, model_args))
    metrics = np.array(metrics)
    return metrics

def get_metrics(dataset, model, analysis_type, training_type, cv_n, view, run, dataset_split, metric, model_args=None):
    if analysis_type == "model_assessment": 
        if "teacher" in model:
            if "weight" in model:
                model = "_".join(model.split("_")[:2]) 
                cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/model_assessment/{model}/metrics/MainModel_{training_type}_{dataset}_{model}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_with_teacher_weight_matching_{dataset_split}_{metric}.pickle'
            else:
                model = "_".join(model.split("_")[:2]) 
                cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/model_assessment/{model}/metrics/MainModel_{training_type}_{dataset}_{model}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_with_teacher_{dataset_split}_{metric}.pickle'    
        
        elif "gcn_student" == model:
            cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/model_assessment/{model}/metrics/MainModel_{training_type}_{dataset}_{model}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_{dataset_split}_{metric}.pickle'    
        
        elif "gat_student" == model:
            cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/model_assessment/{model}/metrics/MainModel_{training_type}_{dataset}_{model}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_{dataset_split}_{metric}.pickle'    
        
        elif "gat" == model:
            cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/model_assessment/{model}/metrics/MainModel_{training_type}_{dataset}_{model}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_{dataset_split}_{metric}.pickle'    

        elif "mlp" == model:
            cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/model_assessment/{model}/metrics/MainModel_{training_type}_{dataset}_{model}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_with_teacher_{dataset_split}_{metric}.pickle'    
        
        elif "lsp" == model:
            cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/model_assessment/{model}/metrics/MainModel_{training_type}_{dataset}_{model}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_{model}_{dataset_split}_{metric}.pickle'    
        
        elif "mskd" == model:
            cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/model_assessment/{model}/metrics/MainModel_{training_type}_{dataset}_{model}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_{model}_{dataset_split}_{metric}.pickle'    
        
        elif "fitnet" == model:
            cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/model_assessment/{model}/metrics/MainModel_{training_type}_{dataset}_{model}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_{model}_{dataset_split}_{metric}.pickle'    

        elif model_args != None:
            if "layers" in model_args.keys():
                if model_args["layers"] == 3 or model_args["layers"] == 4:
                    cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/'+'model_assessment/{}/metrics/MainModel_{}_{}_{}_run_{}_fixed_init_layers_{}_CV_{}_view_{}_{}_{}.pickle'.format(model,training_type, dataset, model, run, model_args["layers"],cv_n, view, dataset_split, metric)
                else:
                    cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/model_assessment/{model}/metrics/MainModel_{training_type}_{dataset}_{model}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_{dataset_split}_{metric}.pickle'   
        else:
            cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/model_assessment/{model}/metrics/MainModel_{training_type}_{dataset}_{model}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_{dataset_split}_{metric}.pickle'   
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

def extract_weights(dataset, view, model, training_type, run, student=0, model_args=None):
    runs = []
    if training_type == '3Fold':
        for cv_i in range(3):
            runs.append(get_weight(dataset, view, model, training_type, 0, cv_i, run, student, model_args))
    if training_type == '5Fold':
        for cv_i in range(5):
            runs.append(get_weight(dataset, view, model, training_type, 0, cv_i, run, student, model_args))
    if training_type == '10Fold':
        for cv_i in range(10):
            runs.append(get_weight(dataset, view, model, training_type, 0, cv_i, run, student, model_args))
    runs = np.array(runs)
    weights = np.mean(runs, axis=0)
    return weights

def get_weight(dataset, view, model, training_type, shot_n, cv_n, run, student, model_args=None):
    if "ensamble" in model:
        
        alpha = str(model_args["alpha"])
        beta = str(model_args["beta"])
        gamma = str(model_args["gamma"])
        lambda_ = str(model_args["lambda"])
        
        #SPECIAL CASE WHERE WE ANALYSE THE ENSAMBLE 
        if student == -1:
            all_weights = []
            for student in range(model_args["n_students"]):
                cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/'+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_fixed_init_student_{}_CV_{}_view_{}_alpha_{}_beta_{}_gamma_{}_lambda_{}.pickle'.format(model,training_type, dataset, model_args["model_name"], run, student, cv_n, view, alpha, beta, gamma, lambda_)
                with open(cv_path,'rb') as f:
                 weights = pickle.load(f)
                weights_vector = weights['w'].squeeze().detach().numpy() 
                all_weights.append(weights_vector)
            return np.mean(all_weights, axis=0)

        else:
            cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/'+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_fixed_init_student_{}_CV_{}_view_{}_alpha_{}_beta_{}_gamma_{}_lambda_{}.pickle'.format(model,training_type, dataset, model_args["model_name"], run, student, cv_n, view, alpha, beta, gamma, lambda_)
    
    elif "gat_student" == model:
        cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/'+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_fixed_init_CV_{}_view_{}.pickle'.format(model,training_type, dataset, model, run, cv_n, view)
    
    elif "gat" == model:
        cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/'+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_fixed_init_CV_{}_view_{}.pickle'.format(model,training_type, dataset, model, run, cv_n, view)

    elif "teacher" in model:
        if "weight" in model:
            model = "_".join(model.split("_")[:2]) 
            cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/'+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_fixed_init_CV_{}_view_{}_with_teacher_weight_matching.pickle'.format(model, training_type, dataset, model, run, cv_n, view)
        else:
            model = "_".join(model.split("_")[:2]) 
            cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/'+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_fixed_init_CV_{}_view_{}_with_teacher.pickle'.format(model, training_type, dataset, model, run, cv_n, view)        
    
    elif "gcn_student" == model:
        cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/'+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_fixed_init_CV_{}_view_{}.pickle'.format(model,training_type, dataset, model, run, cv_n, view)

    elif "lsp" == model:
        cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/'+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_fixed_init_CV_{}_view_{}_lsp.pickle'.format(model,training_type, dataset, model, run, cv_n, view)

    elif "mskd" == model:
        cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/'+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_fixed_init_CV_{}_view_{}_mskd.pickle'.format(model,training_type, dataset, model, run, cv_n, view)

    elif "fitnet" == model:
        cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/'+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_fixed_init_CV_{}_view_{}_fitnet.pickle'.format(model,training_type, dataset, model, run, cv_n, view)

    elif model_args != None:
        if "layers" in model_args.keys():
            if model_args["layers"] == 3 or model_args["layers"] == 4:
                cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/'+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_fixed_init_layers_{}_CV_{}_view_{}.pickle'.format(model,training_type, dataset, model, run, model_args["layers"],cv_n, view)
            else:
                cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/'+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_fixed_init_CV_{}_view_{}.pickle'.format(model,training_type, dataset, model, run, cv_n, view)
        else:
            cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/'+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_fixed_init_CV_{}_view_{}_with_teacher.pickle'.format(model,training_type, dataset, model, run, cv_n, view)
    else:
        cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/'+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_fixed_init_CV_{}_view_{}.pickle'.format(model,training_type, dataset, model, run, cv_n, view)
    
    x_path = cv_path 
    with open(x_path,'rb') as f:
        weights = pickle.load(f)

    weights_vector = weights['w'].squeeze().detach().numpy() 
    
    return weights_vector