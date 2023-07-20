
#############################################################################################################################
######################################## GCN TEACHER MODEL PARAMETERS w/BreastMNIST ########################################
######################################## GCN STUDENT MODEL PARAMETERS w/BreastMNIST ########################################
#############################################################################################################################

gcn_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn",
    "backbone":"gcn",
    "dataset":"BreastMNIST",
    "layers":2,
    "evaluation_method": "model_assessment" # model selection or model assessment
}

gcn_student_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-5, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student",
    'student_type':"gcn",
    "backbone":"gcn",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 0.8, 
    "T": 3, 
    "alpha_soft_ce": 9
}

gcn_fitnet_student_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"fitnet",
    'student_type':"gcn",
    "backbone":"gcn",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 2, 
    "T": 3, 
    "alpha_soft_ce": 7,
    "alpha_ht": 0.1
}

gcn_lsp_student_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"lsp",
    'student_type':"gcn",
    "backbone":"gcn",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 1, 
    "T": 3, 
    "alpha_soft_ce": 2,
    "alpha_kd_lsp":2,
    "alpha_weight": 0
}

gcn_mskd_student_BreastMNIST_args = {
    "num_epochs":50, 
    "lr":  1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"mskd",
    'student_type':"gcn",
    "backbone":"gcn",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 1, 
    "T": 3, 
    "alpha_soft_ce": 4,
    "alpha_mskd": 2
}

gcn_student_ensamble_2_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_ensamble_2",
    'student_type':"gcn",
    "backbone":"gcn",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":2 # TOTAL number of students in ensamble 
}

gcn_student_ensamble_3_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_ensamble_3",
    'student_type':"gcn",
    "backbone":"gcn",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":3 # TOTAL number of students in ensamble 
}

gcn_student_ensamble_4_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_ensamble_4",
    'student_type':"gcn",
    "backbone":"gcn",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":4 # TOTAL number of students in ensamble 
}

gcn_student_ensamble_5_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_ensamble_5",
    'student_type':"gcn",
    "backbone":"gcn",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":5 # TOTAL number of students in ensamble 
}

# ENSAMBLE WITH LSP PARAMS

gcn_student_lsp_ensamble_2_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_lsp_ensamble_2",
    'student_type':"gcn",
    "backbone":"gcn",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":2 # TOTAL number of students in ensamble 
}

gcn_student_lsp_ensamble_3_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_lsp_ensamble_3",
    'student_type':"gcn",
    "backbone":"gcn",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":3 # TOTAL number of students in ensamble 
}

gcn_student_lsp_ensamble_4_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_lsp_ensamble_4",
    'student_type':"gcn",
    "backbone":"gcn",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":4 # TOTAL number of students in ensamble 
}

gcn_student_lsp_ensamble_5_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_lsp_ensamble_5",
    'student_type':"gcn",
    "backbone":"gcn",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":5 # TOTAL number of students in ensamble 
}

#ABLATION STUDY

gcn_student_lsp_ensamble_3_BreastMNIST_args_1 = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_lsp_ensamble_3",
    'student_type':"gcn",
    "backbone":"gcn",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 0, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":3 # TOTAL number of students in ensamble 
}

gcn_student_lsp_ensamble_3_BreastMNIST_args_2 = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_lsp_ensamble_3",
    'student_type':"gcn",
    "backbone":"gcn",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 0,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":3 # TOTAL number of students in ensamble 
}

gcn_student_lsp_ensamble_3_BreastMNIST_args_3 = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_lsp_ensamble_3",
    'student_type':"gcn",
    "backbone":"gcn",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 0, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":3 # TOTAL number of students in ensamble 
}

gcn_student_lsp_ensamble_3_BreastMNIST_args_4 = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_lsp_ensamble_3",
    'student_type':"gcn",
    "backbone":"gcn",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":0, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":3 # TOTAL number of students in ensamble 
}

#############################################################################################################################
######################################## GAT BACKBONE MODEL PARAMETERS w/BreastMNIST ########################################
#############################################################################################################################

gat_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-3, 
    "weight_decay":5e-4,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "hidden_dim":8,
    "nb_heads":8, # Attention heads
    "alpha":0.2, # Alpha for the leaky_relu.
    "dropout": 0.1,
    "model_name":"gat",
    'student_type':"gcn",
    "evaluation_method": "model_assessment",
    "dataset":"BreastMNIST",
    "backbone":"gat"
}

gat_student_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-6, 
    "weight_decay":5e-4, 
    "hidden_dim":2,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gat_student",
    'student_type':"gcn",
    "backbone":"gat",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 0.8, 
    "T": 3, 
    "alpha_soft_ce": 9
}

#############################################################################################################################
######################################## GAT TEACHER MODEL PARAMETERS w/BreastMNIST ########################################
######################################## GCN STUDENT MODEL PARAMETERS w/BreastMNIST ########################################
#############################################################################################################################

gat_gcn_student_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-5, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student",
    'student_type':"gcn",
    "backbone":"gat",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 0.8, 
    "T": 3, 
    "alpha_soft_ce": 9
}

gat_gcn_fitnet_student_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"fitnet",
    'student_type':"gcn",
    "backbone":"gat",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 2, 
    "T": 3, 
    "alpha_soft_ce": 7,
    "alpha_ht": 0.1
}

gat_gcn_lsp_student_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"lsp",
    'student_type':"gcn",
    "backbone":"gat",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 1, 
    "T": 3, 
    "alpha_soft_ce": 2,
    "alpha_kd_lsp":2,
    "alpha_weight": 0
}

gat_gcn_mskd_student_BreastMNIST_args = {
    "num_epochs":50, 
    "lr":  1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"mskd",
    'student_type':"gcn",
    "backbone":"gat",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 1, 
    "T": 3, 
    "alpha_soft_ce": 4,
    "alpha_mskd": 2
}

gat_gcn_student_ensamble_2_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_ensamble_2",
    'student_type':"gcn",
    "backbone":"gat",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":2 # TOTAL number of students in ensamble 
}

gat_gcn_student_ensamble_3_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_ensamble_3",
    'student_type':"gcn",
    "backbone":"gat",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":3 # TOTAL number of students in ensamble 
}

gat_gcn_student_ensamble_4_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_ensamble_4",
    'student_type':"gcn",
    "backbone":"gat",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":4 # TOTAL number of students in ensamble 
}

gat_gcn_student_ensamble_5_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_ensamble_5",
    'student_type':"gcn",
    "backbone":"gat",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":5 # TOTAL number of students in ensamble 
}

# ENSAMBLE WITH LSP PARAMS

gat_gcn_student_lsp_ensamble_2_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_lsp_ensamble_2",
    'student_type':"gcn",
    "backbone":"gat",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":2 # TOTAL number of students in ensamble 
}

gat_gcn_student_lsp_ensamble_3_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_lsp_ensamble_3",
    'student_type':"gcn",
    "backbone":"gat",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":3 # TOTAL number of students in ensamble 
}

gat_gcn_student_lsp_ensamble_4_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_lsp_ensamble_4",
    'student_type':"gcn",
    "backbone":"gat",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":4 # TOTAL number of students in ensamble 
}

gat_gcn_student_lsp_ensamble_5_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn_student_lsp_ensamble_5",
    'student_type':"gcn",
    "backbone":"gat",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":5 # TOTAL number of students in ensamble 
}


#############################################################################################################################
######################################## GAT TEACHER MODEL PARAMETERS w/BreastMNIST ########################################
######################################## GAT STUDENT MODEL PARAMETERS w/BreastMNIST ########################################
#############################################################################################################################

gat_gat_student_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-6, 
    "weight_decay":5e-4, 
    "hidden_dim":2,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gat_student",
    'student_type':"gat",
    "backbone":"gat",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 0.8, 
    "T": 3, 
    "alpha_soft_ce": 9
}

gat_gat_fitnet_student_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":2,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"fitnet_gat",
    'student_type':"gat",
    "backbone":"gat",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 2, 
    "T": 3, 
    "alpha_soft_ce": 7,
    "alpha_ht": 0.1
}

gat_gat_lsp_student_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":2,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"lsp_gat",
    'student_type':"gat",
    "backbone":"gat",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 1, 
    "T": 3, 
    "alpha_soft_ce": 2,
    "alpha_kd_lsp":2,
    "alpha_weight": 0
}

gat_gat_mskd_student_BreastMNIST_args = {
    "num_epochs":50, 
    "lr":  1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":2,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"mskd_gat",
    'student_type':"gat",
    "backbone":"gat",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 1, 
    "T": 3, 
    "alpha_soft_ce": 4,
    "alpha_mskd": 2
}

gat_gat_student_ensamble_2_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":2,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gat_student_ensamble_2",
    'student_type':"gat",
    "backbone":"gat",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":2 # TOTAL number of students in ensamble 
}

gat_gat_student_ensamble_3_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":2,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gat_student_ensamble_3",
    'student_type':"gat",
    "backbone":"gat",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":3 # TOTAL number of students in ensamble 
}

gat_gat_student_ensamble_4_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":2,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gat_student_ensamble_4",
    'student_type':"gat",
    "backbone":"gat",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":4 # TOTAL number of students in ensamble 
}

gat_gat_student_ensamble_5_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":2,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gat_student_ensamble_5",
    'student_type':"gat",
    "backbone":"gat",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":5 # TOTAL number of students in ensamble 
}

# ENSAMBLE WITH LSP PARAMS

gat_gat_student_lsp_ensamble_2_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":2,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gat_student_lsp_ensamble_2",
    'student_type':"gat",
    "backbone":"gat",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":2 # TOTAL number of students in ensamble 
}

gat_gat_student_lsp_ensamble_3_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":2,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gat_student_lsp_ensamble_3",
    'student_type':"gat",
    "backbone":"gat",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":3 # TOTAL number of students in ensamble 
}

gat_gat_student_lsp_ensamble_4_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":2,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gat_student_lsp_ensamble_4",
    'student_type':"gat",
    "backbone":"gat",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":4 # TOTAL number of students in ensamble 
}

gat_gat_student_lsp_ensamble_5_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":2,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gat_student_lsp_ensamble_5",
    'student_type':"gat",
    "backbone":"gat",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":5 # TOTAL number of students in ensamble 
}

#############################################################################################################################
######################################## GCN TEACHER MODEL PARAMETERS w/BreastMNIST ########################################
######################################## GAT STUDENT MODEL PARAMETERS w/BreastMNIST ########################################
#############################################################################################################################

gcn_gat_student_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-6, 
    "weight_decay":5e-4, 
    "hidden_dim":2,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gat_student",
    'student_type':"gat",
    "backbone":"gcn",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 0.8, 
    "T": 3, 
    "alpha_soft_ce": 9
}

gcn_gat_fitnet_student_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":2,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"fitnet_gat",
    'student_type':"gat",
    "backbone":"gcn",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 2, 
    "T": 3, 
    "alpha_soft_ce": 7,
    "alpha_ht": 0.1
}

gcn_gat_lsp_student_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":2,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"lsp_gat",
    'student_type':"gat",
    "backbone":"gcn",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 1, 
    "T": 3, 
    "alpha_soft_ce": 2,
    "alpha_kd_lsp":2,
    "alpha_weight": 0
}

gcn_gat_mskd_student_BreastMNIST_args = {
    "num_epochs":50, 
    "lr":  1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":2,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"mskd_gat",
    'student_type':"gat",
    "backbone":"gcn",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha_ce": 1, 
    "T": 3, 
    "alpha_soft_ce": 4,
    "alpha_mskd": 2
}

gcn_gat_student_ensamble_2_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":2,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gat_student_ensamble_2",
    'student_type':"gat",
    "backbone":"gcn",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":2 # TOTAL number of students in ensamble 
}

gcn_gat_student_ensamble_3_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":2,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gat_student_ensamble_3",
    'student_type':"gat",
    "backbone":"gcn",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":3 # TOTAL number of students in ensamble 
}

gcn_gat_student_ensamble_4_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":2,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gat_student_ensamble_4",
    'student_type':"gat",
    "backbone":"gcn",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":4 # TOTAL number of students in ensamble 
}

gcn_gat_student_ensamble_5_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":2,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gat_student_ensamble_5",
    'student_type':"gat",
    "backbone":"gcn",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":5 # TOTAL number of students in ensamble 
}

# ENSAMBLE WITH LSP PARAMS

gcn_gat_student_lsp_ensamble_2_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":2,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gat_student_lsp_ensamble_2",
    'student_type':"gat",
    "backbone":"gcn",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":2 # TOTAL number of students in ensamble 
}

gcn_gat_student_lsp_ensamble_3_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":2,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gat_student_lsp_ensamble_3",
    'student_type':"gat",
    "backbone":"gcn",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":3 # TOTAL number of students in ensamble 
}

gcn_gat_student_lsp_ensamble_4_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":2,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gat_student_lsp_ensamble_4",
    'student_type':"gat",
    "backbone":"gcn",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":4 # TOTAL number of students in ensamble 
}

gcn_gat_student_lsp_ensamble_5_BreastMNIST_args = {
    "num_epochs":50, 
    "lr": 1e-4, 
    "weight_decay":5e-4, 
    "hidden_dim":2,
    "nb_heads":2, # Attention heads
    "alpha":0.2,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gat_student_lsp_ensamble_5",
    'student_type':"gat",
    "backbone":"gcn",
    "dataset":"BreastMNIST",
    "evaluation_method": "model_assessment", # model selection or model assessment
    "alpha": 2, # ensamble ce loss
    "beta": 7,  # ensamble kd loss
    "gamma": 7, # sum of student ce loss
    "lambda":1, # disentanglement loss
    "T": 3, #Temperature parameter for soft logit target 
    "n_students":5 # TOTAL number of students in ensamble 
}
