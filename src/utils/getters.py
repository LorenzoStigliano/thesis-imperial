import torch
import pickle
import io
import numpy as np 
from config import SAVE_DIR_MODEL_DATA

########################################################################################
################################## GETTERS FOR LABELS OF MODELS ########################
########################################################################################

def get_labels_and_preds(dataset, model, analysis_type, training_type, cv_n, view, run, dataset_split, student=0, model_args=None):
    """
    Get labels and predictions for a specific analysis type, model, and training configuration.

    Parameters:
        dataset (str): Name of the dataset.
        model (str): Name of the GNN model or KD method.
        analysis_type (str): Type of analysis ("model_assessment" or other).
        training_type (str): Type of training configuration (e.g., "3Fold", "5Fold", "10Fold").
        cv_n (int): Index of the cross-validation fold.
        view (int): Index of the data view.
        run (int): Index of the model training run.
        dataset_split (str): Split of the dataset ("train", "val", or "test").
        student (int, optional): Index of the student in the ensemble (default is 0).
        model_args (dict, optional): Dictionary containing model-specific arguments (default is None).

    Returns:
        labels_and_preds (dict) A dictionary containing labels and model predictions if analysis_type is "model_assessment",
        otherwise returns None.
    """"
    if analysis_type == "model_assessment":
        if "ensamble" in model:
            alpha = str(model_args["alpha"])
            beta = str(model_args["beta"])
            gamma = str(model_args["gamma"])
            lambda_ = str(model_args["lambda"])
            if student == -1:
                cv_path =  SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/model_assessment/{model}/labels_and_preds/MainModel_{training_type}_{dataset}_{model_args["model_name"]}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_alpha_{alpha}_beta_{beta}_gamma_{gamma}_lambda_{lambda_}_{dataset_split}_ensamble.pickle'
            
            if "gcn_student_lsp_ensamble_4_temperature" == model:
                T = str(model_args["T"])
                cv_path =  SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/model_assessment/{model}/labels_and_preds/MainModel_{training_type}_{dataset}_{model_args["model_name"]}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_alpha_{alpha}_beta_{beta}_gamma_{gamma}_lambda_{lambda_}_T_{T}_{dataset_split}_student_{student}.pickle'        
            
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
    else:
        return 
    with open(cv_path,'rb') as f:
        labels_and_preds = pickle.load(f)
    return labels_and_preds

########################################################################################
################################## GETTERS FOR METRICS OF MODELS #######################
########################################################################################

def extract_metrics(dataset, model, analysis_type, training_type, view, run, dataset_split, metric, model_args=None):
    """
    Extract and aggregate metrics across cross-validation folds for a specific analysis.

    Parameters:
        dataset (str): Name of the dataset.
        model (str): Name of the GNN model or KD method.
        analysis_type (str): Type of analysis ("model_assessment" or other).
        training_type (str): Type of training configuration (e.g., "3Fold", "5Fold", "10Fold").
        view (int): Index of the data view.
        run (int): Index of the model training run.
        dataset_split (str): Split of the dataset ("train", "val", or "test").
        metric (str): Name of the metric to extract.
        model_args (dict, optional): Dictionary containing model-specific arguments (default is None).

    Returns:
        metrics (numpy.ndarray): Array of aggregated metrics across cross-validation folds.
    """
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
    """
    Get model evaluation metrics from saved files.

    Parameters:
        dataset (str): Name of the dataset.
        model (str): Name of the GNN model or KD method.
        analysis_type (str): Type of analysis ("model_assessment" or other).
        training_type (str): Type of training configuration (e.g., "3Fold", "5Fold", "10Fold").
        cv_n (int): Index of the cross-validation fold.
        view (int): Index of the data view.
        run (int): Index of the model training run.
        dataset_split (str): Split of the dataset ("train", "val", or "test").
        metric (str): Name of the metric to extract.
        model_args (dict, optional): Dictionary containing model-specific arguments (default is None).

    Returns:
        metrics (dict): Dictionary containing model evaluation metrics if analysis_type is "model_assessment",
            otherwise returns None.
    """
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
        
        elif "lsp_gat" == model:
            cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/model_assessment/{model}/metrics/MainModel_{training_type}_{dataset}_{model}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_lsp_{dataset_split}_{metric}.pickle'    
        
        elif "lsp_gcn" == model:
            cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/model_assessment/{model}/metrics/MainModel_{training_type}_{dataset}_{"lsp"}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_lsp_{dataset_split}_{metric}.pickle'    
        
        elif "mskd" == model:
            cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/model_assessment/{model}/metrics/MainModel_{training_type}_{dataset}_{model}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_{model}_{dataset_split}_{metric}.pickle'    
        
        elif "mskd_gat" == model:
            cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/model_assessment/{model}/metrics/MainModel_{training_type}_{dataset}_{model}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_mskd_{dataset_split}_{metric}.pickle'    
        
        elif "mskd_gcn" == model:
            cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/model_assessment/{model}/metrics/MainModel_{training_type}_{dataset}_{"mskd"}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_mskd_{dataset_split}_{metric}.pickle'    
        
        elif "fitnet" == model:
            cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/model_assessment/{model}/metrics/MainModel_{training_type}_{dataset}_{model}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_{model}_{dataset_split}_{metric}.pickle'    

        elif "fitnet_gat" == model:
            cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/model_assessment/{model}/metrics/MainModel_{training_type}_{dataset}_{model}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_fitnet_{dataset_split}_{metric}.pickle'    

        elif "fitnet_gcn" == model:
            cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/model_assessment/{model}/metrics/MainModel_{training_type}_{dataset}_{"fitnet"}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_fitnet_{dataset_split}_{metric}.pickle'    

        elif model_args != None:
            if "layers" in model_args.keys():
                if model_args["layers"] == 3 or model_args["layers"] == 4:
                    cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/'+'model_assessment/{}/metrics/MainModel_{}_{}_{}_run_{}_fixed_init_layers_{}_CV_{}_view_{}_{}_{}.pickle'.format(model,training_type, dataset, model, run, model_args["layers"],cv_n, view, dataset_split, metric)
                else:
                    cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/model_assessment/{model}/metrics/MainModel_{training_type}_{dataset}_{model}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_{dataset_split}_{metric}.pickle'   
        else:
            cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/model_assessment/{model}/metrics/MainModel_{training_type}_{dataset}_{model}_run_{run}_fixed_init_CV_{cv_n}_view_{view}_{dataset_split}_{metric}.pickle'   
    else:
        return 
    with open(cv_path,'rb') as f:
        metrics = pickle.load(f)
    return metrics

########################################################################################
################################## GETTERS FOR WEIGHTS OF MODELS #######################
########################################################################################

class CPU_Unpickler(pickle.Unpickler):
    """
    Unpickler class to load torch objects on CPU.

    This custom unpickler is designed to load torch objects from pickled data while
    mapping them to the CPU. It specifically handles the case where the module is
    'torch.storage' and the name is '_load_from_bytes'.

    Usage:
    custom_unpickler = CPU_Unpickler(file, fix_imports=False, encoding="latin1")
    object = custom_unpickler.load()
    """
    def find_class(self, module, name):
        """
        Find and return the class to load from pickled data.

        Parameters:
            module (str): Name of the module.
            name (str): Name of the class.

        Returns:
            class: The class to load from pickled data, or super's find_class result.
        """
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def extract_weights(dataset, view, model, training_type, run, student=0, model_args=None):
    """
    Extract and average model weights from different cross-validation folds.

    This function extracts model weights from different cross-validation folds and
    averages them to provide a consolidated set of weights.

    Parameters:
        dataset (str): Name of the dataset.
        view (int): Index of the data view.
        model (str): Name of the GNN model.
        training_type (str): Type of training configuration (e.g., "3Fold", "5Fold", "10Fold").
        run (int): Index of the model training run.
        student (int, optional): Index of the student in the ensemble (default is 0).
        model_args (dict, optional): Dictionary containing model-specific arguments (default is None).

    Returns:
        weights (np.ndarray): Averaged model weights.
    """
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
    """
    Get model weights from specified configuration.

    This function retrieves model weights from a specified configuration and returns them.

    Parameters:
        dataset (str): Name of the dataset.
        view (int): Index of the data view.
        model (str): Name of the GNN model.
        training_type (str): Type of training configuration (e.g., "3Fold", "5Fold", "10Fold").
        shot_n (int): Not used.
        cv_n (int): Index of the cross-validation fold.
        run (int): Index of the model training run.
        student (int): Index of the student in the ensemble.
        model_args (dict, optional): Dictionary containing model-specific arguments (default is None).

    Returns:
        weights_vector (np.ndarray): Model weights vector.
    """
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

        if "gcn_student_lsp_ensamble_4_temperature" == model:
            T = str(model_args["T"])
            cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/'+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_fixed_init_student_{}_CV_{}_view_{}_alpha_{}_beta_{}_gamma_{}_lambda_{}_T_{}.pickle'.format(model,training_type, dataset, model_args["model_name"], run, student, cv_n, view, alpha, beta, gamma, lambda_, T)
        
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
    
    elif "gcn_student_vanilla_hyperparameter" == model:
        cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/'+'model_assessment/gcn_student_vanilla_hyperparameter/weights/W_MainModel_{}_{}_{}_run_{}_fixed_init_CV_{}_view_{}_with_teacher.pickle'.format(training_type, dataset, "gcn_student", run, cv_n, view)

    elif "lsp" == model:
        cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/'+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_fixed_init_CV_{}_view_{}_lsp.pickle'.format(model,training_type, dataset, model, run, cv_n, view)

    elif "lsp_gat" == model:
        cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/'+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_fixed_init_CV_{}_view_{}_lsp.pickle'.format(model,training_type, dataset, model, run, cv_n, view)

    elif "lsp_gcn" == model:
        cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/'+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_fixed_init_CV_{}_view_{}_lsp.pickle'.format(model,training_type, dataset, "lsp", run, cv_n, view)

    elif "mskd" == model:
        cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/'+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_fixed_init_CV_{}_view_{}_mskd.pickle'.format(model,training_type, dataset, model, run, cv_n, view)

    elif "mskd_gat" == model:
        cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/'+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_fixed_init_CV_{}_view_{}_mskd.pickle'.format(model,training_type, dataset, model, run, cv_n, view)

    elif "mskd_gcn" == model:
        cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/'+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_fixed_init_CV_{}_view_{}_mskd.pickle'.format(model,training_type, dataset, "mskd", run, cv_n, view)

    elif "fitnet_gat" == model:
        cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/'+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_fixed_init_CV_{}_view_{}_fitnet.pickle'.format(model,training_type, dataset, model, run, cv_n, view)

    elif "fitnet_gcn" == model:
        cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/'+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_fixed_init_CV_{}_view_{}_fitnet.pickle'.format(model,training_type, dataset, "fitnet", run, cv_n, view)

    elif "fitnet" == model:
        cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/'+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_fixed_init_CV_{}_view_{}_fitnet.pickle'.format(model,training_type, dataset, model, run, cv_n, view)

    elif model_args != None:
        if "layers" in model_args.keys():
            if model_args["layers"] == 3 or model_args["layers"] == 4:
                cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/'+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_fixed_init_layers_{}_CV_{}_view_{}.pickle'.format(model,training_type, dataset, model, run, model_args["layers"],cv_n, view)
            else:
                cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/'+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_fixed_init_CV_{}_view_{}.pickle'.format(model, training_type, dataset, model, run, cv_n, view)
        else:
            cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/'+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_fixed_init_CV_{}_view_{}_with_teacher.pickle'.format(model,training_type, dataset, model, run, cv_n, view)
    else:
        cv_path = SAVE_DIR_MODEL_DATA+f'{dataset}/{model_args["backbone"]}/'+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_fixed_init_CV_{}_view_{}.pickle'.format(model,training_type, dataset, model, run, cv_n, view)
    
    x_path = cv_path 
    with open(x_path,'rb') as f:
        weights = pickle.load(f)

    weights_vector = weights['w'].squeeze().detach().numpy() 
    
    return weights_vector