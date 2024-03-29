
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
    "num_epochs":50, 
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

gcn_3_args = {
    "num_epochs":50, 
    "lr": 0.0001,
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn",
    "backbone":"gcn",
    "dataset":"gender_data",
    "layers":3,
    "evaluation_method": "model_assessment" # model selection or model assessment
}

#Baselines

gcn_student_args = {
    "num_epochs":50, 
    "lr": 0.0001, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student",
    "backbone":"gcn",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 1, 
    "T": 3, 
    "alpha_soft_ce": 2
}

gcn_lsp_student_args = {
    "num_epochs":50, 
    "lr": 0.001, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"lsp",
    "backbone":"gcn",
    'student_type':"gcn",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 1, 
    "T": 3, 
    "alpha_soft_ce": 2,
    "alpha_kd_lsp":2
}

gcn_mskd_student_args = {
    "num_epochs":50, 
    "lr": 0.001, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"mskd",
    'student_type':"gcn",
    "backbone":"gcn",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 1, 
    "T": 3, 
    "alpha_soft_ce": 4,
    "alpha_mskd": 2
}

gcn_fitnet_student_args_0_4 = {
    "num_epochs":50, 
    "lr": 0.001, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"fitnet",
    'student_type':"gcn",
    "backbone":"gcn",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 1, 
    "T": 3, 
    "alpha_soft_ce": 2,
    "alpha_ht": 0.5
}

gcn_fitnet_student_args = {
    "num_epochs":50, 
    "lr": 0.001, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"fitnet",
    'student_type':"gcn",
    "backbone":"gcn",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 1, 
    "T": 3, 
    "alpha_soft_ce": 2,
    "alpha_ht": 0.2
}

# Ensamble method

gcn_student_ensamble_2_args = {
    "num_epochs":50, 
    "lr": 0.0001,
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_ensamble_2",
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

gcn_student_ensamble_3_args = {
    "num_epochs":50, 
    "lr": 0.0001, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_ensamble_3",
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

gcn_student_ensamble_4_args = {
    "num_epochs":50, 
    "lr": 0.0001, # 0.0001 when training without teacher
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_ensamble_4",
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

gcn_student_ensamble_5_args = {
    "num_epochs":50, 
    "lr": 0.0001, # 0.0001 when training without teacher
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_ensamble_5",
    "backbone":"gcn",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 1, # ensamble ce loss
    "beta": 2,  # ensamble kd loss
    "gamma": 2, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":5 # TOTAL number of students in ensamble 
}

# ENSAMBLE WITH LSP PARAMS

gcn_student_lsp_ensamble_2_args = {
    "num_epochs":50, 
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
    "num_epochs":50, 
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
    "num_epochs":50, 
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
    "num_epochs":50, 
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

#ABALATION STUDY LOGITS GCN 

gcn_student_ensamble_4_args_1 = {
    "num_epochs":50, 
    "lr": 0.0001, # 0.0001 when training without teacher
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_ensamble_4",
    "backbone":"gcn",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 0, # ensamble ce loss
    "beta": 2,  # ensamble kd loss
    "gamma": 2, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":4 # TOTAL number of students in ensamble 
}

gcn_student_ensamble_4_args_2 = {
    "num_epochs":50, 
    "lr": 0.0001, # 0.0001 when training without teacher
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_ensamble_4",
    "backbone":"gcn",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 1, # ensamble ce loss
    "beta": 0,  # ensamble kd loss
    "gamma": 2, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":4 # TOTAL number of students in ensamble 
}

gcn_student_ensamble_4_args_3 = {
    "num_epochs":50, 
    "lr": 0.0001, # 0.0001 when training without teacher
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_ensamble_4",
    "backbone":"gcn",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 1, # ensamble ce loss
    "beta": 2,  # ensamble kd loss
    "gamma": 0, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":4 # TOTAL number of students in ensamble 
}

gcn_student_ensamble_4_args_4 = {
    "num_epochs":50, 
    "lr": 0.0001, # 0.0001 when training without teacher
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_ensamble_4",
    "backbone":"gcn",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 1, # ensamble ce loss
    "beta": 2,  # ensamble kd loss
    "gamma": 2, # sum of student ce loss
    "lambda":0, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":4 # TOTAL number of students in ensamble 
}

#ABALATION STUDY LSP GCN 

gcn_student_lsp_ensamble_4_args_1 = {
    "num_epochs":50, 
    "lr": 0.0001, # 0.0001 when training without teacher
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_lsp_ensamble_4",
    "backbone":"gcn",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 0, # ensamble ce loss
    "beta": 2,  # ensamble kd loss
    "gamma": 2, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":4 # TOTAL number of students in ensamble 
}
gcn_student_lsp_ensamble_4_args_2 = {
    "num_epochs":50, 
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
    "beta": 0,  # ensamble kd loss
    "gamma": 2, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":4 # TOTAL number of students in ensamble 
}

gcn_student_lsp_ensamble_4_args_3 = {
    "num_epochs":50, 
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
    "gamma": 0, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":4 # TOTAL number of students in ensamble 
}

gcn_student_lsp_ensamble_4_args_4 = {
    "num_epochs":50, 
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
    "lambda":0, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":4 # TOTAL number of students in ensamble 
}


#############################################################################################################################
######################################## GAT TEACHER MODEL PARAMETERS  ######################################################
######################################## GCN STUDENT MODEL PARAMETERS  ######################################################
#############################################################################################################################

gat_args = {
    "num_epochs":50, 
    "lr":0.0001, 
    "weight_decay":5e-4,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "hidden_dim":8,
    "nb_heads":8, # Attention heads
    "alpha":0.2, # Alpha for the leaky_relu.
    "dropout": 0.1,
    "model_name":"gat",
    "evaluation_method": "model_assessment",
    "dataset":"gender_data",
    "backbone":"gat"
}

gat_student_args = {
    "num_epochs":50, 
    "lr": 0.0001, 
    "weight_decay":5e-4, 
    "hidden_dim":8,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student",
    'student_type':"gcn",
    "backbone":"gat",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 1, 
    "T": 3, 
    "alpha_soft_ce": 7
}

gat_lsp_student_args = {
    "num_epochs":50, 
    "lr": 0.001, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"lsp",
    'student_type':"gcn",
    "backbone":"gat",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 1, 
    "T": 3, 
    "alpha_soft_ce": 2,
    "alpha_kd_lsp":2,
    "alpha_weight": 0
}

gat_mskd_student_args = {
    "num_epochs":50, 
    "lr": 0.001, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"mskd",
    'student_type':"gcn",
    "backbone":"gat",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 1, 
    "T": 3, 
    "alpha_soft_ce": 4,
    "alpha_mskd": 2
}

gat_fitnet_student_args_0_4 = {
    "num_epochs":50, 
    "lr": 0.001, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"fitnet",
    'student_type':"gcn",
    "backbone":"gat",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 1, 
    "T": 3, 
    "alpha_soft_ce": 2,
    "alpha_ht": 0.5
}

gat_fitnet_student_args = {
    "num_epochs":50, 
    "lr": 0.001, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"fitnet",
    'student_type':"gcn",
    "backbone":"gat",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 1, 
    "T": 3, 
    "alpha_soft_ce": 2,
    "alpha_ht": 0.2
}

# Ensamble method

gat_student_ensamble_2_args = {
    "num_epochs":50, 
    "lr": 0.0001, # 0.0001 when training without teacher
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_ensamble_2",
    'student_type':"gcn",
    "backbone":"gat",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 1, # ensamble ce loss
    "beta": 2,  # ensamble kd loss
    "gamma": 2, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":2 # TOTAL number of students in ensamble 
}

gat_student_ensamble_3_args = {
    "num_epochs":50, 
    "lr": 0.0001, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_ensamble_3",
    'student_type':"gcn",
    "backbone":"gat",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 1, # ensamble ce loss
    "beta": 2,  # ensamble kd loss
    "gamma": 2, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":3 # TOTAL number of students in ensamble 
}

gat_student_ensamble_4_args = {
    "num_epochs":50, 
    "lr": 0.0001, # 0.0001 when training without teacher
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_ensamble_4",
    'student_type':"gcn",
    "backbone":"gat",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 1, # ensamble ce loss
    "beta": 2,  # ensamble kd loss
    "gamma": 2, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":4 # TOTAL number of students in ensamble 
}

gat_student_ensamble_5_args = {
    "num_epochs":50, 
    "lr": 0.0001, # 0.0001 when training without teacher
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_ensamble_5",
    'student_type':"gcn",
    "backbone":"gat",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 1, # ensamble ce loss
    "beta": 2,  # ensamble kd loss
    "gamma": 2, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":5 # TOTAL number of students in ensamble 
}

# ENSAMBLE WITH LSP PARAMS

gat_student_lsp_ensamble_2_args = {
    "num_epochs":50, 
    "lr": 0.0001, # 0.0001 when training without teacher
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_lsp_ensamble_2",
    'student_type':"gcn",
    "backbone":"gat",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 1, # ensamble ce loss
    "beta": 2,  # ensamble kd loss
    "gamma": 2, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":2 # TOTAL number of students in ensamble 
}

gat_student_lsp_ensamble_3_args = {
    "num_epochs":50, 
    "lr": 0.0001, # 0.0001 when training without teacher
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_lsp_ensamble_3",
    'student_type':"gcn",
    "backbone":"gat",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 1, # ensamble ce loss
    "beta": 2,  # ensamble kd loss
    "gamma": 2, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":3 # TOTAL number of students in ensamble 
}

gat_student_lsp_ensamble_4_args = {
    "num_epochs":50, 
    "lr": 0.0001, # 0.0001 when training without teacher
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_lsp_ensamble_4",
    'student_type':"gcn",
    "backbone":"gat",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 1, # ensamble ce loss
    "beta": 2,  # ensamble kd loss
    "gamma": 2, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":4 # TOTAL number of students in ensamble 
}

gat_student_lsp_ensamble_5_args = {
    "num_epochs":50, 
    "lr": 0.0001, # 0.0001 when training without teacher
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_lsp_ensamble_5",
    'student_type':"gcn",
    "backbone":"gat",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 1, # ensamble ce loss
    "beta": 2,  # ensamble kd loss NOTE: You need to take into account T^2 to this value
    "gamma": 2, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":5 # TOTAL number of students in ensamble 
}

#############################################################################################################################
######################################## GAT TEACHER MODEL PARAMETERS  ######################################################
######################################## GAT STUDENT MODEL PARAMETERS  ######################################################
#############################################################################################################################

gat_gat_student_args = {
    "num_epochs":50, 
    "lr": 0.0001, 
    "weight_decay":5e-4, 
    "hidden_dim":8,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gat_student",
    'student_type':"gat",
    "backbone":"gat",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 1, 
    "T": 3, 
    "alpha_soft_ce": 7
}

gat_gat_lsp_student_args = {
    "num_epochs":50, 
    "lr": 0.001, 
    "weight_decay":5e-4, 
    "hidden_dim":8,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"lsp_gat",
    'student_type':"gat",
    "backbone":"gat",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 1, 
    "T": 3, 
    "alpha_soft_ce": 2,
    "alpha_kd_lsp":2,
    "alpha_weight": 0
}

gat_gat_mskd_student_args = {
    "num_epochs":50, 
    "lr": 0.001, 
    "weight_decay":5e-4, 
    "hidden_dim":8,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"mskd_gat",
    'student_type':"gat",
    "backbone":"gat",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 1, 
    "T": 3, 
    "alpha_soft_ce": 4,
    "alpha_mskd": 2
}

gat_gat_fitnet_student_args_0_4 = {
    "num_epochs":50, 
    "lr": 0.001, 
    "weight_decay":5e-4, 
    "hidden_dim":8,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"fitnet_gat",
    'student_type':"gat",
    "backbone":"gat",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 1, 
    "T": 3, 
    "alpha_soft_ce": 2,
    "alpha_ht": 0.5
}

gat_gat_fitnet_student_args = {
    "num_epochs":50, 
    "lr": 0.001, 
    "weight_decay":5e-4, 
    "hidden_dim":8,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"fitnet_gat",
    'student_type':"gat",
    "backbone":"gat",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 1, 
    "T": 3, 
    "alpha_soft_ce": 2,
    "alpha_ht": 0.2
}

# ENSAMBLE WITH LSP PARAMS

gat_gat_student_lsp_ensamble_2_args = {
    "num_epochs":50, 
    "lr": 0.0001, # 0.0001 when training without teacher
    "weight_decay":5e-4, 
    "hidden_dim":8,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gat_student_lsp_ensamble_2",
    'student_type':"gat",
    "backbone":"gat",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 1, # ensamble ce loss
    "beta": 2,  # ensamble kd loss
    "gamma": 2, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":2 # TOTAL number of students in ensamble 
}

gat_gat_student_lsp_ensamble_3_args = {
    "num_epochs":50, 
    "lr": 0.0001, # 0.0001 when training without teacher
    "weight_decay":5e-4, 
    "hidden_dim":8,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gat_student_lsp_ensamble_3",
    'student_type':"gat",
    "backbone":"gat",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 1, # ensamble ce loss
    "beta": 2,  # ensamble kd loss
    "gamma": 2, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":3 # TOTAL number of students in ensamble 
}

gat_gat_student_lsp_ensamble_4_args = {
    "num_epochs":50, 
    "lr": 0.0001, # 0.0001 when training without teacher
    "weight_decay":5e-4, 
    "hidden_dim":8,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gat_student_lsp_ensamble_4",
    'student_type':"gat",
    "backbone":"gat",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 1, # ensamble ce loss
    "beta": 2,  # ensamble kd loss
    "gamma": 2, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":4 # TOTAL number of students in ensamble 
}

gat_gat_student_lsp_ensamble_5_args = {
    "num_epochs":50, 
    "lr": 0.0001, # 0.0001 when training without teacher
    "weight_decay":5e-4, 
    "hidden_dim":8,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gat_student_lsp_ensamble_5",
    'student_type':"gat",
    "backbone":"gat",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 1, # ensamble ce loss
    "beta": 2,  # ensamble kd loss NOTE: You need to take into account T^2 to this value
    "gamma": 2, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":5 # TOTAL number of students in ensamble 
}

#############################################################################################################################
######################################## GCN TEACHER MODEL PARAMETERS  ######################################################
######################################## GAT STUDENT MODEL PARAMETERS  ######################################################
#############################################################################################################################

gcn_gat_student_args = {
    "num_epochs":50, 
    "lr": 0.0001, 
    "weight_decay":5e-4, 
    "hidden_dim":8,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gat_student",
    'student_type':"gat",
    "backbone":"gcn",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 1, 
    "T": 3, 
    "alpha_soft_ce": 7
}

gcn_gat_lsp_student_args = {
    "num_epochs":50, 
    "lr": 0.001, 
    "weight_decay":5e-4, 
    "hidden_dim":8,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"lsp_gat",
    'student_type':"gat",
    "backbone":"gcn",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 1, 
    "T": 3, 
    "alpha_soft_ce": 2,
    "alpha_kd_lsp":2,
    "alpha_weight": 0
}

gcn_gat_mskd_student_args = {
    "num_epochs":50, 
    "lr": 0.001, 
    "weight_decay":5e-4, 
    "hidden_dim":8,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"mskd_gat",
    'student_type':"gat",
    "backbone":"gcn",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 1, 
    "T": 3, 
    "alpha_soft_ce": 4,
    "alpha_mskd": 2
}

gcn_gat_fitnet_student_args_0_4 = {
    "num_epochs":50, 
    "lr": 0.001, 
    "weight_decay":5e-4, 
    "hidden_dim":8,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"fitnet_gat",
    'student_type':"gat",
    "backbone":"gcn",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 1, 
    "T": 3, 
    "alpha_soft_ce": 2,
    "alpha_ht": 0.5
}

gcn_gat_fitnet_student_args = {
    "num_epochs":50, 
    "lr": 0.001, 
    "weight_decay":5e-4, 
    "hidden_dim":8,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"fitnet_gat",
    'student_type':"gat",
    "backbone":"gcn",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 1, 
    "T": 3, 
    "alpha_soft_ce": 2,
    "alpha_ht": 0.2
}

# ENSAMBLE WITH LSP PARAMS

gcn_gat_student_lsp_ensamble_2_args = {
    "num_epochs":50, 
    "lr": 0.0001, # 0.0001 when training without teacher
    "weight_decay":5e-4, 
    "hidden_dim":8,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gat_student_lsp_ensamble_2",
    'student_type':"gat",
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

gcn_gat_student_lsp_ensamble_3_args = {
    "num_epochs":50, 
    "lr": 0.0001, # 0.0001 when training without teacher
    "weight_decay":5e-4, 
    "hidden_dim":8,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gat_student_lsp_ensamble_3",
    'student_type':"gat",
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

gcn_gat_student_lsp_ensamble_4_args = {
    "num_epochs":50, 
    "lr": 0.0001, # 0.0001 when training without teacher
    "weight_decay":5e-4, 
    "hidden_dim":8,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gat_student_lsp_ensamble_4",
    'student_type':"gat",
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

gcn_gat_student_lsp_ensamble_5_args = {
    "num_epochs":50, 
    "lr": 0.0001, # 0.0001 when training without teacher
    "weight_decay":5e-4, 
    "hidden_dim":8,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gat_student_lsp_ensamble_5",
    'student_type':"gat",
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

#############################################################################################################################
######################################## GCN TEACHER MODEL PARAMETERS  ######################################################
######################################## GCN STUDENT MODEL PARAMETERS  ######################################################
#############################################################################################################################

# ENSAMBLE WITH EMB PARAMS

gcn_student_emb_ensamble_2_args = {
    "num_epochs":50, 
    "lr": 0.0001, # 0.0001 when training without teacher
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_emb_ensamble_2",
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

gcn_student_emb_ensamble_3_args = {
    "num_epochs":50, 
    "lr": 0.0001, # 0.0001 when training without teacher
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_emb_ensamble_3",
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

gcn_student_emb_ensamble_4_args = {
    "num_epochs":50, 
    "lr": 0.0001, # 0.0001 when training without teacher
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_emb_ensamble_4",
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

gcn_student_emb_ensamble_5_args = {
    "num_epochs":50, 
    "lr": 0.0001, # 0.0001 when training without teacher
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_emb_ensamble_5",
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