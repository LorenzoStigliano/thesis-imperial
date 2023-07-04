
######################################## GCN BACKBONE MODEL PARAMETERS ########################################

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
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 1, 
    "T": 3, 
    "alpha_soft_ce": 2,
    "alpha_kd_lsp":2,
    "alpha_weight": 0
}

gcn_mskd_student_args = {
    "num_epochs":50, 
    "lr": 0.001, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"mskd",
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
    "backbone":"gcn",
    "dataset":"gender_data",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 1, 
    "T": 2, 
    "alpha_soft_ce": 2,
    "alpha_ht": 0.2
}

# Ensamble method

gcn_student_ensamble_2_args = {
    "num_epochs":50, 
    "lr": 0.0001, # 0.0001 when training without teacher
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
    "lr": 0.0001, # 0.0001 when training without teacher
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
    "beta": 2,  # ensamble kd loss
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


######################################## GAT BACKBONE MODEL PARAMETERS ########################################

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
    "lr":0.0001, 
    "weight_decay":5e-4,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "hidden_dim":4,
    "nb_heads":4, # Attention heads
    "alpha":0.2, # Alpha for the leaky_relu.
    "dropout": 0.1,
    "model_name":"gat_student",
    "evaluation_method": "model_assessment",
    "dataset":"gender_data",
    "backbone":"gat"
}