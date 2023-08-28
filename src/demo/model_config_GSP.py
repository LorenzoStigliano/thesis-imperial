
"""
Model Configuration File for GSP datset

This file contains example configuration parameters for training and evaluating graph neural network (GNN) models.
These parameters are defined here as templates but are hard-coded in their corresponding 'main*.py' files for specific
experiments and tasks.

The 'model_config' file serves as a reference for the configuration parameters used in training and evaluation of GNN models.
Each configuration defines hyperparameters and settings that shape the behavior of the GNN model during training and testing.
The 'main*.py' scripts that perform the actual training and evaluation import these configurations and use them for execution.

It's important to customize these configurations based on the requirements of the specific GNN model, dataset, and task.
By centralizing these parameters, it becomes easier to manage and experiment with different configurations without
directly modifying the main code.

Usage:
    To use this 'model_config' file, select or modify the example configurations provided below to match your specific
    GNN model and application. These configurations will then be imported and used in the 'main*.py' scripts for
    training and evaluation.

Note:
    These example configurations should be adjusted according to the specific GNN model and application.
    Parameters like 'num_epochs', 'hidden_dim', 'dropout', 'threshold', and others should be customized
    based on the nature of the problem being tackled.
"""
#############################################################################################################################
######################################## GCN TEACHER MODEL PARAMETERS  ######################################################
######################################## GCN STUDENT MODEL PARAMETERS  ######################################################
#############################################################################################################################

gcn_args = {
    "num_epochs":1, 
    "lr": 0.0001,
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn",
    "backbone":"gcn",
    "dataset":"gender_data",
    "layers":2,
    "evaluation_method": "model_assessment" # model selection or model assessment
}

# ENSAMBLE WITH LSP PARAMS
gcn_student_lsp_ensamble_2_args = {
    "num_epochs":1, 
    "lr": 0.0001, # 0.0001 when training without teacher
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_lsp_ensamble_2",
    "backbone":"gcn",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 1, # ensamble ce loss
    "beta": 2,  # ensamble kd loss
    "gamma": 2, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":2 # TOTAL number of students in ensamble 
}

gcn_student_lsp_ensamble_3_args = {
    "num_epochs":1, 
    "lr": 0.0001, # 0.0001 when training without teacher
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_lsp_ensamble_3",
    "backbone":"gcn",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 1, # ensamble ce loss
    "beta": 2,  # ensamble kd loss
    "gamma": 2, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":3 # TOTAL number of students in ensamble 
}

gcn_student_lsp_ensamble_4_args = {
    "num_epochs":1, 
    "lr": 0.0001, # 0.0001 when training without teacher
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_lsp_ensamble_4",
    "backbone":"gcn",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 1, # ensamble ce loss
    "beta": 2,  # ensamble kd loss
    "gamma": 2, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":4 # TOTAL number of students in ensamble 
}

gcn_student_lsp_ensamble_5_args = {
    "num_epochs":1, 
    "lr": 0.0001, # 0.0001 when training without teacher
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_lsp_ensamble_5",
    "backbone":"gcn",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 1, # ensamble ce loss
    "beta": 2,  # ensamble kd loss NOTE: You need to take into account T^2 to this value
    "gamma": 2, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":5 # TOTAL number of students in ensamble 
}