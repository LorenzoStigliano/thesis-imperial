import torch
import pickle
import numpy as np

from config import SAVE_DIR_MODEL_DATA

def extract_weights_single(dataset, view, model, training_type, shot_n, cv_n, run):
    if "teacher" in model:
        if "weight" in model:
            model = "_".join(model.split("_")[:2]) 
            cv_path = SAVE_DIR_MODEL_DATA+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_CV_{}_view_{}_with_teacher_weight_matching.pickle'.format(model, training_type, dataset, model, run, cv_n, view)
        else:
            model = "_".join(model.split("_")[:2]) 
            cv_path = SAVE_DIR_MODEL_DATA+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_CV_{}_view_{}_with_teacher.pickle'.format(model, training_type, dataset, model, run, cv_n, view)        
    else:
        cv_path = SAVE_DIR_MODEL_DATA+'model_assessment/{}/weights/W_MainModel_{}_{}_{}_run_{}_CV_{}_view_{}.pickle'.format(model,training_type, dataset, model, run, cv_n, view)

    if training_type == 'Few_Shot':
        x_path = fs_path
    else: 
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

def extract_weights(dataset, view, model, training_type, run):
    runs = []
    if training_type == 'Few_Shot':
        for shot_i in range(5):
            runs.append(extract_weights_single(dataset, view, model, training_type, shot_i, 0, run))
    if training_type == '3Fold':
        for cv_i in range(3):
            runs.append(extract_weights_single(dataset, view, model, training_type, 0, cv_i, run))
    if training_type == '5Fold':
        for cv_i in range(5):
            runs.append(extract_weights_single(dataset, view, model, training_type, 0, cv_i, run))
    if training_type == '10Fold':
        for cv_i in range(10):
            runs.append(extract_weights_single(dataset, view, model, training_type, 0, cv_i, run))
    runs = np.array(runs)
    weights = np.mean(runs, axis=0)
    return weights

def top_biomarkers(weights, K_i):
    weights_normalized = np.abs(weights)
    return list(weights_normalized.argsort()[::-1][:K_i])

def sim(nodes1, nodes2):
    if len(nodes1)==len(nodes2):
        counter = 0
        for i in nodes1:
            for k in nodes2:
                if i==k:
                    counter+=1
        return counter/len(nodes1)

    else:
        print('nodes vectors are not compatible')

def view_specific_rep(dataset, view, model, CV, run):

    Ks = [5, 10, 15, 20]
    rep = np.zeros([len(CV), len(CV), len(Ks)])

    for i in range(rep.shape[0]):
        for j in range(rep.shape[1]):
            weights_i = extract_weights(dataset, view, model, CV[i], run)
            weights_j = extract_weights(dataset, view, model, CV[j], run)
            
            for k in range(rep.shape[2]):
                top_bio_i = top_biomarkers(weights_i, Ks[k])
                top_bio_j = top_biomarkers(weights_j, Ks[k])
                rep[i,j,k] = sim(top_bio_i, top_bio_j)
                
    rep_mean = np.mean(rep, axis=2)
    # Get the elements above the diagonal
    elements_above_diagonal = np.where(np.triu(np.ones_like(rep_mean), k=1), rep_mean, np.nan)

    # Drop the zeros and calculate the average
    average = np.nanmean(elements_above_diagonal)
    std = np.nanstd(elements_above_diagonal)
    
    return average, std

####### USAGE #######
"""
for view in views: 
    gcn_rep_score             = view_specific_rep(dataset="gender_data", view=view, model="gcn", CV=CV)
    gcn_student_score         = view_specific_rep(dataset="gender_data", view=view, model="gcn_student", CV=CV)
    gcn_student_teacher_score = view_specific_rep(dataset="gender_data", view=view, model="gcn_student_teacher", CV=CV)
"""

"""
CV = ["3Fold", "5Fold", "10Fold"]
views = [0, 2, 4, 5]
models = ["gcn","gcn_student", "gcn_student_teacher", "gcn_student_teacher_weight"] 
reproducibilty_data = {}

for view in views:
    print("____________________________________________")
    print(f"View: {view}")
    
    for model in models:
        
        model_data = []
        
        for run in [0,1,2]:
            model_data.append(view_specific_rep(dataset="gender_data", view=view, model=model, CV=CV, run=run))
        
        reproducibilty_data[model] = {
            "mean": np.mean(model_data),
            "std":np.std(model_data),
            "values":model_data
        }

    for model in models:
        print(model, reproducibilty_data[model]["mean"], reproducibilty_data[model]["std"])
"""

#COMPARE DIFFERENT WEIGHTS
"""
cv_path_1 = SAVE_DIR_MODEL_DATA+'model_assessment/gcn_student/weights/W_MainModel_3Fold_gender_data_gcn_student_run_0_CV_0_view_0_with_teacher_weight_matching.pickle'
cv_path_2 = '/Users/lorenzostigliano/Documents/University/Imperial/Summer Term/model_data_STABLE/model_assessment/gcn_student/weights/W_MainModel_3Fold_gender_data_gcn_student_run_0_CV_0_view_0_with_teacher_weight_matching.pickle'

with open(cv_path_1,'rb') as f:
    weights_1 = pickle.load(f)

with open(cv_path_2,'rb') as f:
    weights_2 = pickle.load(f)

print(weights_1)
print(weights_2)
"""
"""
Ks = [5, 10, 15, 20]
teacher_weights = weights_1["w"].squeeze().detach().numpy()
student_weight = weights_2["w"].squeeze().detach().numpy()
for k in Ks:
    top_bio_i = top_biomarkers(teacher_weights, k)
    top_bio_j = top_biomarkers(student_weight, k)
    print(top_bio_i)
    print(top_bio_j)
    print(sim(top_bio_i, top_bio_j))
"""
