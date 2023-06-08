import numpy as np
from getters import *

############ ANALYSIS OF REPRODUCIBILITY FOR MODELS ############

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

def view_specific_rep(dataset, view, model, CV, run, student=0, model_args=None):
    """
    USAGE:
    CV = ["3Fold", "5Fold", "10Fold"]
    views = [0,1,4,5]

    for view in views: 
        gcn_rep_score             = view_specific_rep(dataset="gender_data", view=view, model="gcn", CV=CV)
        gcn_student_score         = view_specific_rep(dataset="gender_data", view=view, model="gcn_student", CV=CV)
        gcn_student_teacher_score = view_specific_rep(dataset="gender_data", view=view, model="gcn_student_teacher", CV=CV)
    """

    Ks = [5, 10, 15, 20]
    rep = np.zeros([len(Ks), len(CV), len(CV)])

    for k in range(rep.shape[0]):
        for i in range(rep.shape[1]):
            for j in range(rep.shape[2]):
                weights_i = extract_weights(dataset, view, model, CV[i], run, student, model_args)
                weights_j = extract_weights(dataset, view, model, CV[j], run, student, model_args)
                top_bio_i = top_biomarkers(weights_i, Ks[k])
                top_bio_j = top_biomarkers(weights_j, Ks[k])
                rep[k,i,j] = sim(top_bio_i, top_bio_j)

    rep_mean = np.mean(rep, axis=0)
    # Get the elements above the diagonal
    elements_above_diagonal = np.where(np.triu(np.ones_like(rep_mean), k=1), rep_mean, np.nan)

    # Drop the zeros and calculate the average
    average = np.nanmean(elements_above_diagonal)
    std = np.nanstd(elements_above_diagonal)
    
    return average, std

def view_reproducibility_analysis(dataset, models, CV, views, run, student=0, model_args=None):
    """
    Reproducibility analysis for a single run
    """

    view_data_mean = []
    view_data_std = []

    for view in views:
        
        model_result_mean = []
        model_result_std = []
        
        for model in models:
            rep_score, std = view_specific_rep(dataset=dataset, view=view, model=model, run=run, CV=CV, student=student, model_args=model_args)
            model_result_mean.append(rep_score)
            model_result_std.append(std)
        
        view_data_mean.append(model_result_mean)
        view_data_std.append(model_result_std)

    view_data_std.append(list(np.std(view_data_mean, axis=0)))
    view_data_std = np.array(view_data_std).T

    view_data_mean.append(list(np.mean(view_data_mean, axis=0)))

    view_data_mean = np.array(view_data_mean).T
    
    return view_data_mean, view_data_std 

############ ANALYSIS OF METIRC FOR MODELS ############

def metric_and_view_analysis(models, CV, analysis_type, view, run, dataset_split, dataset, metric):
    """
    Mean of metric for a specific CV -> 3, 5 or 10
    """

    all_data_mean = []
    all_data_std = []
    
    for model in models:
        
        model_results_mean = []
        model_results_std = []
    
        for training_type in CV:
            metrics = extract_metrics(dataset=dataset, model=model, analysis_type=analysis_type, training_type=training_type, view=view, run=run, dataset_split=dataset_split, metric=metric)
            mean = np.mean([metric[-1] for metric in metrics])
            std = np.std([metric[-1] for metric in metrics])
            model_results_mean.append(mean)
            model_results_std.append(std)
        
        all_data_mean.append(model_results_mean)
        all_data_std.append(model_results_std)
    
    return all_data_mean, all_data_std

def view_metric_analysis(models, CV, view, run, metric, dataset, dataset_split, analysis_type):
    """
    Getting all the means across for a run
    """
    view_data_mean = []
    view_data_std = []

    mean, std = metric_and_view_analysis(models=models, 
                                    CV=CV, 
                                    analysis_type=analysis_type, 
                                    view=view, 
                                    run=run, 
                                    dataset= dataset,
                                    dataset_split=dataset_split, 
                                    metric=metric)
    view_data_mean.append(mean)
    view_data_std.append(std)
    
    return view_data_mean, view_data_std 

####### USAGE #######
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
    weights_1 = CPU_Unpickler(f).load()

with open(cv_path_2,'rb') as f:
    weights_2 = CPU_Unpickler(f).load()

print(weights_1)
print(weights_2)

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
