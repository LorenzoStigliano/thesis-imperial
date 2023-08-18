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

def view_reproducibility_analysis(dataset, models, CV, views, run, students=0, model_args=None):
    """
    Reproducibility analysis for a single run
    """

    view_data_mean = []
    view_data_std = []

    for view in views:
        
        model_result_mean = []
        model_result_std = []

        if model_args==None:
        
            for model in models:
                rep_score, std = view_specific_rep(dataset=dataset, view=view, model=model, run=run, CV=CV, student=students, model_args=model_args)
                model_result_mean.append(rep_score)
                model_result_std.append(std)
        
        else:
            
            for i, model in enumerate(models):
                rep_score, std = view_specific_rep(dataset=dataset, view=view, model=model, run=run, CV=CV, student=students[i], model_args=model_args[i])
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

def metric_and_view_analysis(models, CV, analysis_type, view, run, dataset_split, dataset, metric, model_args=None):
    """
    Mean of metric for a specific CV -> 3, 5 or 10 for a specific run 
    """

    all_data_mean = []
    all_data_std = []
    if model_args == None:
        for model in models:
            
            model_results_mean = []
            model_results_std = []
            for training_type in CV:
                metrics = extract_metrics(dataset=dataset, 
                                        model=model, 
                                        analysis_type=analysis_type, 
                                        training_type=training_type, 
                                        view=view, 
                                        run=run, 
                                        dataset_split=dataset_split, 
                                        metric=metric,
                                        model_args=model_args)
                mean = np.mean([metric[-1] for metric in metrics])
                std = np.std([metric[-1] for metric in metrics])
                model_results_mean.append(mean)
                model_results_std.append(std)
            
            all_data_mean.append(model_results_mean)
            all_data_std.append(model_results_std)
    else:
        for i, model in enumerate(models):
            
            model_results_mean = []
            model_results_std = []
        
            for training_type in CV:
                metrics = extract_metrics(dataset=dataset, 
                                        model=model, 
                                        analysis_type=analysis_type, 
                                        training_type=training_type, 
                                        view=view, 
                                        run=run, 
                                        dataset_split=dataset_split, 
                                        metric=metric,
                                        model_args=model_args[i])
                mean = np.mean([metric[-1] for metric in metrics])
                std = np.std([metric[-1] for metric in metrics])
                model_results_mean.append(mean)
                model_results_std.append(std)
            
            all_data_mean.append(model_results_mean)
            all_data_std.append(model_results_std)        
    
    return all_data_mean, all_data_std

def view_metric_analysis(models, CV, view, run, metric, dataset, dataset_split, analysis_type,model_args=None):
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
                                    metric=metric, 
                                    model_args=model_args)
    view_data_mean.append(mean)
    view_data_std.append(std)
    
    return view_data_mean, view_data_std 

def get_student_model_metric(dataset, model, CV, runs, analysis_type, dataset_split, view, model_args):
    #Get average student modele metrics across all runs and all cv for all models in ensemble 
    #EACH FOLD INDIVIDUAL

    student_data_mean = [] 
    student_data_var = [] 
    
    for student in range(model_args["n_students"]):

        run_data = []   

        for run in runs:

            model_metrics = []
            
            for training_type in CV:

                model_cv_metrics = get_mean_CV_metric_student_model(
                    dataset=dataset, 
                    model=model, 
                    analysis_type=analysis_type, 
                    training_type=training_type, 
                    view=view, 
                    run=run, 
                    student=student, 
                    dataset_split=dataset_split,
                    model_args=model_args
                    )
                model_metrics.append(model_cv_metrics)
            
            run_data.append(model_metrics)
        
        student_data_mean.append(np.mean(run_data, axis=0))   
        student_data_var.append(np.std(run_data, axis=0))        

    return student_data_mean, student_data_var

def get_best_student_ensamble_detailed(model, view, CV, runs, dataset, dataset_split, analysis_type, model_args):
    import warnings
    warnings.filterwarnings('ignore')
    
    student_rep = []
    student_var = []

    # get the best reproducibility across all runs 
    for run in runs:

        mean_all_runs = []

        for student in range(model_args["n_students"]):
            view_rep, _ = view_reproducibility_analysis(
                dataset=dataset, 
                models=[model], 
                CV=CV, 
                views=[view], 
                run=run, 
                students=[student], 
                model_args=[model_args]
            )
            mean_all_runs.append(view_rep[0][0])
        #store the reproducibility score for all students for the run 
        student_rep.append(mean_all_runs)
    
    #get the mean and variance for the reproducibility scores for all the students
    student_var = np.var(student_rep, axis=0)
    student_rep = np.mean(student_rep, axis=0)

    #get the metrics of all the student models
    all_student_metrics_mean, all_student_metrics_var = get_student_model_metric_all_folds(
        dataset=dataset, 
        model=model, 
        CV=CV, 
        runs=runs, 
        analysis_type=analysis_type, 
        dataset_split=dataset_split, 
        view=view, 
        model_args=model_args
    )
    
    #get the best model based soley on max reproducibility 
    best_rep_max = 0
    metric = 0
    student_max = -1
    for i, rep in enumerate(student_rep):
        if rep > metric:
            best_rep_max = rep 
            metric = rep
            student_max = i

    #get the best model based on max accuracy
    best_max_acc = 0
    metric = 0
    student_max_acc_index = -1
    for i, rep in enumerate(student_rep):
        student_acc = all_student_metrics_mean[i][0]
        if student_acc > metric:
            metric = student_acc
            best_max_acc = rep
            student_max_acc_index = i

    #get the best model based on max reproducibility and f1 score
    best_max_f1 = 0
    metric = 0
    student_max_f1_index = -1
    for i, rep in enumerate(student_rep):
        student_f1 = all_student_metrics_mean[i][1]
        if student_f1 > metric:
            metric = student_f1
            best_max_f1 = rep
            student_max_f1_index = i

    #get the best model based on max reproducibility and accuracy
    best_rep_acc = 0
    metric = 0
    student_acc_index = -1
    for i, rep in enumerate(student_rep):
        student_acc = all_student_metrics_mean[i][0]
        if (rep+student_acc)/2 > metric:
            metric = (rep+student_acc)/2 
            best_rep_acc = rep
            student_acc_index = i
    
    #get the best model based on max reproducibility and f1 score
    best_rep_f1 = 0
    metric = 0
    student_f1_index = -1
    for i, rep in enumerate(student_rep):
        student_f1 = all_student_metrics_mean[i][1]
        if (rep+student_f1)/2 > metric:
            metric = (rep+student_f1)/2 
            best_rep_f1 = rep
            student_f1_index = i
    
    return all_student_metrics_mean, all_student_metrics_var, [[best_rep_max, student_var[student_max], student_max], [best_max_acc, student_var[student_max_acc_index], student_max_acc_index], [best_max_f1, student_var[student_max_f1_index], student_max_f1_index], [best_rep_acc, student_var[student_acc_index], student_acc_index], [best_rep_f1, student_var[student_f1_index], student_f1_index]]
    
def get_student_model_metric_all_folds(dataset, model, CV, runs, analysis_type, dataset_split, view, model_args):
    #Get average student modele metrics across all runs and all cv for all models in ensemble 
    """
    USAGE:
    runs = [i for i in range(10)]
    CV=["3Fold", "5Fold", "10Fold"]
    model = "gcn_student_ensamble_3"
    analysis_type="model_assessment"
    model_args= gcn_student_ensamble_args
    dataset_split="val"
    view=2

    get_student_model_metric_all_folds(dataset, model, CV, runs, analysis_type, dataset_split, view, model_args)
    """

    all_student_metrics_mean = []
    all_student_metrics_var = []

    for student in range(model_args["n_students"]):

        model_metrics_runs = []
        
        for run in runs:
                
            model_metrics = []
            
            for training_type in CV:
                
                model_metrics.append(get_mean_CV_metric_student_model(
                    dataset=dataset, 
                    model=model, 
                    analysis_type=analysis_type, 
                    training_type=training_type, 
                    view=view, 
                    run=run, 
                    student=student, 
                    dataset_split=dataset_split,
                    model_args=model_args
                )
                )
            
            model_metrics = np.mean(model_metrics, axis=0)
            model_metrics_runs.append(model_metrics)
        all_student_metrics_var.append(np.var(model_metrics_runs, axis=0))
        all_student_metrics_mean.append(np.mean(model_metrics_runs, axis=0))

    return all_student_metrics_mean, all_student_metrics_var

def view_reproducibility_analysis_student_specific(dataset, models, CV, views, run, students=[0], model_args=None):
    """
    Reproducibility analysis for a single run for specific students in ensamble 
    student length = number of views, it is the specific student for each view
    """

    view_data_mean = []
    view_data_std = []

    for i, view in enumerate(views):
        
        model_result_mean = []
        model_result_std = []
        ensamble_count = 0

        for j, model in enumerate(models):

            rep_score, std = view_specific_rep(dataset=dataset, view=view, model=model, run=run, CV=CV, student=students[ensamble_count][i], model_args=model_args[j])
            model_result_mean.append(rep_score)
            model_result_std.append(std)
            
            if "ensamble" in models:
                ensamble_count += 1

        view_data_mean.append(model_result_mean)
        view_data_std.append(model_result_std)

    view_data_std.append(list(np.std(view_data_mean, axis=0)))
    view_data_std = np.array(view_data_std).T

    view_data_mean.append(list(np.mean(view_data_mean, axis=0)))

    view_data_mean = np.array(view_data_mean).T
    
    return view_data_mean, view_data_std 

def reproducibility_mulitple_runs_student_specific(dataset, views, models, CV, runs, students=0, model_args=None):

    mean_all_runs = []
    runs=[i for i in range(10)]
    for run in runs:
        view_data_mean, _ = view_reproducibility_analysis_student_specific(dataset, models, CV, views, run, students=students, model_args=model_args)
        mean_all_runs.append(view_data_mean)

    mean_all_std = np.std(mean_all_runs, axis=0)
    mean_all_runs = np.mean(mean_all_runs, axis=0)
    
    return mean_all_runs, mean_all_std

def get_mean_CV_metric_student_model(dataset, model, analysis_type, training_type, view, run, student, dataset_split, model_args):

    import sklearn.metrics as metrics

    #get the mean metric for a student for a particular CV training_type 
    
    student_acc, student_recall, student_precision, student_f1 = 0, 0, 0, 0
    acc_mean, recall_mean, precision_mean, f1_mean = [], [], [], []
    
    if training_type == "3Fold":
        cv_number = 3
    if training_type == "5Fold":
        cv_number = 5
    if training_type == "10Fold":
        cv_number = 10  
    
    for i in range(cv_number):
        x = get_labels_and_preds(dataset=dataset, 
                                model=model,
                                analysis_type=analysis_type, 
                                training_type=training_type,  
                                cv_n=i, 
                                view=view, 
                                run=run, 
                                dataset_split=dataset_split, 
                                student=student, 
                                model_args=model_args)
        result = {
            'prec': metrics.precision_score(x['labels'],  x['preds']),
            'recall': metrics.recall_score(x['labels'],  x['preds']),
            'acc': metrics.accuracy_score(x['labels'],  x['preds']),
            'F1': metrics.f1_score(x['labels'],  x['preds'])
        }   
        acc_mean.append(result['acc'])
        recall_mean.append(result['recall'])
        precision_mean.append(result['prec'])
        f1_mean.append(result['F1'])
    
    student_acc = np.mean(acc_mean)
    student_recall = np.mean(recall_mean)
    student_precision = np.mean(precision_mean)
    student_f1 = np.mean(f1_mean)

    return student_acc, student_f1, student_recall, student_precision

def get_all_best_student(analysis_type, dataset_split, dataset, models_args, views):

    all_model_metrics_mean = []
    all_model_metrics_var = []
    all_model_best_student = []

    for model_arg in models_args:
        
        print(model_arg["model_name"])

        model_metrics_mean = []
        model_metrics_var = []
        model_best_student = []

        for view in views:
            mean, var, best_students = get_best_student_ensamble_detailed(
                model=model_arg["model_name"], 
                view=view, 
                CV=["3Fold", "5Fold", "10Fold"],
                runs=[i for i in range(10)], 
                analysis_type=analysis_type,
                dataset_split=dataset_split,
                dataset=dataset,
                model_args=model_arg
            )
            print(best_students)
            model_metrics_mean.append(mean)
            model_metrics_var.append(var)
            model_best_student.append(best_students)

        all_model_metrics_mean.append(model_metrics_mean)

        all_model_metrics_var.append(model_metrics_var)
        all_model_best_student.append(model_best_student)
    
    return all_model_metrics_mean, all_model_metrics_var, all_model_best_student

def final_student(all_model_metrics_mean, all_model_metrics_var, all_model_best_student, selection_method='weighted acc'):
    import pandas as pd
    #GET THE REPRODUCIBILITY OF THE BEST STUDENT ACROSS ALL DATASETS FOR ALL MODELS
    df_best_student = np.array(all_model_best_student)
    df = []
    #iterate over number of datasets
    for i in range(len(all_model_metrics_mean[1])):
        df.append(df_best_student[:, i, :, 0].flatten())
    df = pd.DataFrame(np.array(df))
    # Calculate the mean of columns
    mean_row = df.mean(axis=0)

    # Append the mean row to the DataFrame
    df = df.append(mean_row, ignore_index=True).T
    index_values = ['max rep', 'max acc', 'max f1', 'weighted acc', 'weighted f1'] * (len(df) // 5) + ['max rep', 'max acc', 'max f1', 'weighted acc', 'weighted f1'][:len(df) % 5]

    # Assign the new index to the DataFrame
    df.index = index_values

    #GET THE MODEL PERFORMANCE
    index = df_best_student[:,:,:,2]
    all_data_accuracy = []
    #iterate over number of datasets
    for view_index in range(len(index[1])):
        view_data = []
        ensamble_indexes = index[:,view_index,:]
        ensamble_data = np.array(all_model_metrics_mean)[:,view_index]
        #number of ensembles 
        for i in range(4):
            view_data.append([ensamble_data[i][int(best_student_index)][0] for best_student_index in ensamble_indexes[i]])
        all_data_accuracy.append(np.array(view_data).flatten())
    
    df_acuracy = pd.DataFrame(np.array(all_data_accuracy))
    # Calculate the mean of columns
    mean_row = df_acuracy.mean(axis=0)

    # Append the mean row to the DataFrame
    df_acuracy = df_acuracy.append(mean_row, ignore_index=True).T
    # Create the list of repeating index values
    index_values = ['max rep', 'max acc', 'max f1', 'weighted acc', 'weighted f1'] * (len(df_acuracy) // 5) + ['max rep', 'max acc', 'max f1', 'weighted acc', 'weighted f1'][:len(df_acuracy) % 5]
    # Assign the new index to the DataFrame
    df_acuracy.index = index_values
    
    #GET THE MODEL PERFORMANCE STD
    all_data_var = []
    #iterate over number of datasets
    for view_index in range(len(all_model_metrics_mean[1])):
        view_data = []
        ensamble_indexes = index[:,view_index,:]
        ensamble_data = np.array(all_model_metrics_var)[:,view_index]
        #number of ensembles 
        for i in range(len(all_model_metrics_mean[0])):
            view_data.append([ensamble_data[i][int(best_student_index)][0] for best_student_index in ensamble_indexes[i]])
        all_data_var.append(np.array(view_data).flatten())
    

    df_var = pd.DataFrame(np.array(all_data_var))
    # Calculate the mean of columns
    mean_row = df_var.mean(axis=0)

    # Append the mean row to the DataFrame
    df_var = df_var.append(mean_row, ignore_index=True).T
    # Create the list of repeating index values
    index_values = ['max rep', 'max acc', 'max f1', 'weighted acc', 'weighted f1'] * (len(df_var) // 5) + ['max rep', 'max acc', 'max f1', 'weighted acc', 'weighted f1'][:len(df_var) % 5]

    # Assign the new index to the DataFrame
    df_var.index = index_values

    #GET METRIC BASED ON SELECTION CRITERIA
    df_acuracy = df_acuracy.loc[selection_method].T
    df_rep = df.loc[selection_method].T
    df_var = df_var.loc[selection_method].T
    df_var = np.sqrt(df_var)

    #LABEL THEM WITH NUMBER OF STUDENTS IN ENSEMBL
    new_labels = ['2', '3', '4',  '5']

    # Rename the columns using the new labels
    df_rep.columns = new_labels
    df_acuracy.columns = new_labels
    df_var.columns = new_labels

    #GET INDEX OF STUDNET IN ENSEMBLE
    if selection_method == 'max rep':
        df_index = pd.DataFrame(index[:, :, 0].T)
        df_index.columns = new_labels
        return df_rep, df_acuracy, df_var, df_index
    elif selection_method == 'max acc':
        df_index = pd.DataFrame(index[:, :, 1].T)
        df_index.columns = new_labels
        return df_rep, df_acuracy, df_var, df_index
    elif selection_method == 'max f1':        
        df_index = pd.DataFrame(index[:, :, 2].T)
        df_index.columns = new_labels
        return df_rep, df_acuracy, df_var, df_index
    elif selection_method == 'weighted acc': 
        df_index = pd.DataFrame(index[:, :, 3].T)
        df_index.columns = new_labels
        return df_rep, df_acuracy, df_var, df_index
    elif selection_method == 'weighted f1':
        df_index =pd.DataFrame(index[:, :, 4].T)
        df_index.columns = new_labels
        return df_rep, df_acuracy, df_var,df_index

def performance_mulitple_runs_student_specific(final_model, final_model_args, best_student, baseline_models, baseline_models_args, views, CV, runs, dataset, dataset_split, analysis_type, metrics):
    import pandas as pd

    view_mean = []
    view_std = []
        
    for best_student_i, view in enumerate(views):

        mean_all_runs = []
        for run in runs:
            view_data_mean, _ = view_metric_analysis(models=baseline_models, CV=CV, view=view, run=run, metric=metrics[0], dataset=dataset, 
                                                    dataset_split=dataset_split, analysis_type=analysis_type, model_args=baseline_models_args)
            mean_all_runs.append(view_data_mean)

        mean_all_std = np.std(mean_all_runs, axis=0).squeeze()
        mean_all_runs = np.mean(mean_all_runs, axis=0).squeeze()

        index_student = best_student[best_student_i]

        model_mean = []
        model_std = []

        for i, model_arg in enumerate(final_model_args):
            mean, std = get_student_model_metric(dataset, final_model[i], CV, runs, analysis_type, dataset_split, view, model_arg)
            model_mean.append(mean[index_student])
            model_std.append(std[index_student])
        
        for i, metric in enumerate(metrics):

            if metric == 'acc':
                mean_df = np.array(model_mean[0])[:,0]
                mean_df = mean_df[np.newaxis, :]
                
                std_df = np.array(model_std[0])[:,0]
                std_df = std_df[np.newaxis, :]
            
            if metric == 'recall':
                mean_df = np.array(model_mean[0])[:,1]
                mean_df = mean_df[np.newaxis, :]
                
                std_df = np.array(model_std[0])[:,1]
                std_df = std_df[np.newaxis, :]
            
            if metric == 'precision':
                mean_df = np.array(model_mean[0])[:,2]
                mean_df = mean_df[np.newaxis, :]
                
                std_df = np.array(model_std[0])[:,2]
                std_df = std_df[np.newaxis, :]
            
            if metric == 'f1':
                mean_df = np.array(model_mean[0])[:,3]
                mean_df = mean_df[np.newaxis, :]
                
                std_df = np.array(model_std[0])[:,3]
                std_df = std_df[np.newaxis, :]

            final_mean_df = np.r_[mean_all_runs, mean_df]
            final_var_df = np.r_[mean_all_std, std_df]

        view_mean.append(np.c_[ final_mean_df, np.mean(final_mean_df, axis=1)])
        view_std.append(np.c_[ final_var_df, np.std(final_var_df, axis=1)])

    average_across_views_metric = []
    for i in range(len(views)):
        average_across_views_metric.append(view_mean[i][:,-1])

    average_across_views_std = []
    for i in range(len(views)):
        average_across_views_std.append(view_std[i][:,-1])
    

    average_across_views_df = pd.DataFrame(average_across_views_metric)
    # Calculate the mean of each row
    average_across_views_df_mean = average_across_views_df.mean(axis=0)  
    # Create a DataFrame from the row means
    average_across_views_df_mean = pd.DataFrame(average_across_views_df_mean).T  
    # Append the row mean DataFrame to the original DataFrame
    average_across_views_df = average_across_views_df.append(average_across_views_df_mean, ignore_index=True)

    average_across_views_df_std_mean = pd.DataFrame(average_across_views_std)

    return average_across_views_df, average_across_views_df_std_mean