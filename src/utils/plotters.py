import random
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import sklearn.metrics as metrics
import os
import pickle

from loaders import load_data
from config import SAVE_DIR_FIGS, SAVE_DIR_MODEL_DATA
from analysis import view_specific_rep
from getters import * 

def plot_random_sample(dataset, views, save_fig=False):
    """
    Function to plot a random sample on a specified dataset and view
    """

    sample_index = random.choice(range(len(load_data(dataset, 1, False))))

    for view in views:
        
        sample = load_data(dataset, view, False)[sample_index]
        ax = sns.heatmap(sample["adj"])
        title = "Dataset:{}, Sample:{}, View:{}".format(dataset, sample["id"], view)
        plt.title(title)
        
        if save_fig:
            if not os.path.exists(SAVE_DIR_FIGS+"samples/"):
                os.makedirs(SAVE_DIR_FIGS+"samples/")
            
            plt.savefig(SAVE_DIR_FIGS+"samples/"+title+".png", dpi=150)
            plt.clf()
        
        else:
            plt.show()
            plt.clf()    

def plot_learning_curves(dataset, views, model, evaluation_method, folds, with_teacher=False, save_fig=False, run=0):

    for view in views:
        if folds == 3:
            fig, ax = plt.subplots(1,3, figsize=(30,10))
        if folds == 5:
            fig, ax = plt.subplots(1,5, figsize=(30,10))

        for fold in range(folds):
            if evaluation_method=="model_selection":
                if with_teacher:
                    directory_train_losses = f'{SAVE_DIR_MODEL_DATA}{evaluation_method}/{model}/training_loss/training_loss_MainModel_{folds}Fold_{dataset}_{model}_CV_{fold}_view_{view}_with_teacher.pickle'
                    directory_validation_losses =  f'{SAVE_DIR_MODEL_DATA}{evaluation_method}/{model}/validation_loss/validation_loss_MainModel_{folds}Fold_{dataset}_{model}_CV_{fold}_view_{view}_with_teacher.pickle'
                else:
                    directory_train_losses = f'{SAVE_DIR_MODEL_DATA}{evaluation_method}/{model}/training_loss/training_loss_MainModel_{folds}Fold_{dataset}_{model}_CV_{fold}_view_{view}.pickle'
                    directory_validation_losses =  f'{SAVE_DIR_MODEL_DATA}{evaluation_method}/{model}/validation_loss/validation_loss_MainModel_{folds}Fold_{dataset}_{model}_CV_{fold}_view_{view}.pickle'

            else:
                if with_teacher:
                    directory_train_losses = f'{SAVE_DIR_MODEL_DATA}{evaluation_method}/{model}/training_loss/training_loss_MainModel_{folds}Fold_{dataset}_{model}_run_{run}_CV_{fold}_view_{view}_with_teacher.pickle'
                    directory_validation_losses = f'{SAVE_DIR_MODEL_DATA}{evaluation_method}/{model}/validation_loss/validation_loss_MainModel_{folds}Fold_{dataset}_{model}_run_{run}_CV_{fold}_view_{view}_with_teacher.pickle'
                else:
                    directory_train_losses = f'{SAVE_DIR_MODEL_DATA}{evaluation_method}/{model}/training_loss/training_loss_MainModel_{folds}Fold_{dataset}_{model}_run_{run}_CV_{fold}_view_{view}.pickle'
                    directory_validation_losses = f'{SAVE_DIR_MODEL_DATA}{evaluation_method}/{model}/validation_loss/validation_loss_MainModel_{folds}Fold_{dataset}_{model}_run_{run}_CV_{fold}_view_{view}=.pickle'


            with open(directory_train_losses,'rb') as f:
                train_losses = pickle.load(f)  
    
            with open(directory_validation_losses,'rb') as f:
                val_losses = pickle.load(f)  
                
            epochs = range(len(train_losses['loss']))
            if isinstance(train_losses['loss'][0], float):
                train_losses = train_losses['loss']
            else:
                train_losses = [i.item() for i in (train_losses['loss'])]
            if isinstance(val_losses['loss'][0], float):
                val_losses = val_losses['loss']
            else:
                val_losses = [i.item() for i in (val_losses['loss'])]
                
            ax[fold].plot(epochs, train_losses, label='Train')
            ax[fold].plot(epochs, val_losses  , label='Val')
            ax[fold].legend()
            ax[fold].grid()    

        if with_teacher:
            title = "Dataset:{}, Model:{}_with_teacher, View:{}, CV:{}, Epochs:{}".format(dataset, model, view, folds, len(train_losses))
        else:
            title = "Dataset:{}, Model:{}, View:{}, CV:{}, Epochs:{}".format(dataset, model, view, folds, len(train_losses))
        
        if save_fig: 
            fig.suptitle(title)
            if not os.path.exists(SAVE_DIR_FIGS+"loss_curves/"):
                os.makedirs(SAVE_DIR_FIGS+"loss_curves/")
            
            fig.savefig(SAVE_DIR_FIGS+"loss_curves/"+title+".png", dpi=150)
            plt.clf()
        
        else:
            fig.suptitle(title)
            plt.show()
            plt.clf()     

def plot_metric_curves(dataset, views, model, evaluation_method, folds, metric, with_teacher=False, save_fig=False):
    
    for view in views:
        if folds == 3:
            fig, ax = plt.subplots(1,3, figsize=(15,10))
        if folds == 5:
            fig, ax = plt.subplots(1,5, figsize=(15,10))
    
        for fold in range(folds):
            if with_teacher:
                with open('{}{}/{}/metrics/MainModel_{}Fold_{}_{}_CV_{}_view_{}_with_teacher_train_{}.pickle'.format(SAVE_DIR_MODEL_DATA,
                                                                                                    evaluation_method,
                                                                                                    model,
                                                                                                    folds,
                                                                                                    dataset,
                                                                                                    model,
                                                                                                    fold,
                                                                                                    view,
                                                                                                    metric),'rb') as f:
                    train_metric = pickle.load(f)  
                with open('{}{}/{}/metrics/MainModel_{}Fold_{}_{}_CV_{}_view_{}_with_teacher_val_{}.pickle'.format(SAVE_DIR_MODEL_DATA,
                                                                                                    evaluation_method,
                                                                                                    model,
                                                                                                    folds,
                                                                                                    dataset,
                                                                                                    model,
                                                                                                    fold,
                                                                                                    view,
                                                                                                    metric),'rb') as f:
                    val_metric = pickle.load(f)  
            else:
                with open('{}{}/{}/metrics/MainModel_{}Fold_{}_{}_CV_{}_view_{}_train_{}.pickle'.format(SAVE_DIR_MODEL_DATA,
                                                                                                    evaluation_method,
                                                                                                    model,
                                                                                                    folds,
                                                                                                    dataset,
                                                                                                    model,
                                                                                                    fold,
                                                                                                    view,
                                                                                                    metric),'rb') as f:
                    train_metric = pickle.load(f)  
                with open('{}{}/{}/metrics/MainModel_{}Fold_{}_{}_CV_{}_view_{}_val_{}.pickle'.format(SAVE_DIR_MODEL_DATA,
                                                                                                    evaluation_method,
                                                                                                    model,
                                                                                                    folds,
                                                                                                    dataset,
                                                                                                    model,
                                                                                                    fold,
                                                                                                    view,
                                                                                                    metric),'rb') as f:
                    val_metric = pickle.load(f)  
                
            epochs = range(len(train_metric))
            ax[fold].plot(epochs, train_metric, label='Train')
            ax[fold].plot(epochs, val_metric  , label='Val')
            ax[fold].legend()
            ax[fold].grid()
        
        if with_teacher:
            title = "Metric:{} Dataset:{}, Model:{}_with_teacher, View:{}, CV:{}, Epochs:{}".format(metric, dataset, model, view, folds, len(val_metric))
        else:
            title = "Metric:{} Dataset:{}, Model:{}, View:{}, CV:{}, Epochs:{}".format(metric, dataset, model, view, folds, len(val_metric))
        
        if save_fig: 
            fig.suptitle(title)
            if not os.path.exists(SAVE_DIR_FIGS+"metric_curves/"):
                os.makedirs(SAVE_DIR_FIGS+"metric_curves/")
            
            fig.savefig(SAVE_DIR_FIGS+"metric_curves/"+title+".png", dpi=150)
            plt.clf()
        
        else:
            fig.suptitle(title)
            plt.show()
            plt.clf()  
            

def plot_bar_chart(dataset, views, models, folds, metric, save_fig=False):

    for view in views:
        
        barWidth = 1/(len(models)+1)

        fold_data = []
        for fold in range(folds):
            model_result = []
            for model in models:
                if "teacher" in model:
                    model = "_".join(model.split("_")[:2]) 
                    with open('{}{}/labels_and_preds/MainModel_{}Fold_{}_{}_CV_{}_view_{}_teacher.pickle'.format(SAVE_DIR_MODEL_DATA,
                                                                                                                    model,
                                                                                                                    folds,
                                                                                                                    dataset,
                                                                                                                    model,
                                                                                                                    fold,
                                                                                                                    view),'rb') as f:
                        pred_data = pickle.load(f)  
                else:
                    with open('{}{}/labels_and_preds/MainModel_{}Fold_{}_{}_CV_{}_view_{}.pickle'.format(SAVE_DIR_MODEL_DATA,
                                                                                                                    model,
                                                                                                                    folds,
                                                                                                                    dataset,
                                                                                                                    model,
                                                                                                                    fold,
                                                                                                                    view),'rb') as f:
                        pred_data = pickle.load(f)  
                
                labels = pred_data['labels']
                preds = pred_data['preds']
                result = {
                    'prec': metrics.precision_score(labels, preds, average='macro'),
                    'recall': metrics.recall_score(labels, preds, average='macro'),
                    'acc': metrics.accuracy_score(labels, preds),
                    'F1': metrics.f1_score(labels, preds, average="micro")
                }
                model_result.append(result[metric])
            fold_data.append(model_result)
        
        fold_data.append(np.mean(fold_data, axis=0))
        fold_data = np.array(fold_data).T
        X = np.arange(folds+1)
        sep = 0.00
        for i, fold_d in enumerate(fold_data):
            plt.bar(X + sep, fold_d, width = barWidth, edgecolor ='grey', label=models[i])
            sep += barWidth
            
        title = "Dataset:{}, Metric:{}, View:{}, CV:{}".format(dataset, metric, view, folds)
        
        plt.ylabel('Metric:{}'.format(metric))
        x_ticks = ["CV {}".format(i) for i in range(folds)] + ["Average"]
        
        plt.xticks([r + barWidth for r in range(len(fold_data[0]))], x_ticks)
        plt.title(title)
        plt.legend()

        if save_fig:
            if not os.path.exists(SAVE_DIR_FIGS+"results/"):
                os.makedirs(SAVE_DIR_FIGS+"results/")
            
            plt.savefig(SAVE_DIR_FIGS+"results/"+title+".png", dpi=150)
            plt.clf()
        
        else:
            plt.show()
            plt.clf()     

def plot_bar_chart_rep(dataset, views, models, CV, run, save_fig=False):
    
    plt.rcParams["figure.figsize"] = (10,5)

    view_data = []
    
    for view in views:
        
        barWidth = 1/(len(models)+1)
        model_result = []
        
        for model in models:
            rep_score, std = view_specific_rep(dataset=dataset, view=view, model=model, run=run, CV=CV)
            model_result.append(rep_score)
        
        view_data.append(model_result)

    view_data.append(list(np.mean(view_data, axis=0)))
    view_data = np.array(view_data).T

    X = np.arange(len(views)+1)
    sep = 0.00
    for i, view_d in enumerate(view_data):
        plt.bar(X + sep, view_d, width = barWidth, edgecolor ='grey', label=models[i])
        sep += barWidth
    
    max_y_lim = np.amax(view_data) + 0.01
    min_y_lim = np.amin(view_data) - 0.01
    plt.ylim(min_y_lim, max_y_lim)
    
    title = "Reproducibility Score for Dataset:{}".format(dataset)
    
    plt.ylabel("Reproducibility Score")
    x_ticks = ["View {}".format(i) for i in views]+ ["Average"]
    
    plt.xticks([r + barWidth for r in range(len(view_data[0]))], x_ticks)
    plt.title(title)
    plt.legend()
    

    if save_fig:
        if not os.path.exists(SAVE_DIR_FIGS+"reproducibility/"):
            os.makedirs(SAVE_DIR_FIGS+"reproducibility/")
        
        plt.savefig(SAVE_DIR_FIGS+"reproducibility/"+title+".png", dpi=150)
        plt.clf()
    
    else:
        plt.show()
        plt.clf()    

def plot_bar_chart_reproducibility_mulitple_runs(dataset, views, models, CV, runs, save_fig=False):
    
    def view_reproducibility_analysis(models, CV, views, run):

        view_data_mean = []
        view_data_std = []

        for view in views:
            
            barWidth = 1/(len(models)+1)
            model_result_mean = []
            model_result_std = []
            
            for model in models:
                rep_score, std = view_specific_rep(dataset=dataset, view=view, model=model, run=run, CV=CV)
                model_result_mean.append(rep_score)
                model_result_std.append(std)
            
            view_data_mean.append(model_result_mean)
            view_data_std.append(model_result_std)

        view_data_std.append(list(np.std(view_data_mean, axis=0)))
        view_data_std = np.array(view_data_std).T

        view_data_mean.append(list(np.mean(view_data_mean, axis=0)))

        view_data_mean = np.array(view_data_mean).T
        
        return view_data_mean, view_data_std 
    
    barWidth = 1/(len(models)+1)

    mean_all_runs = []
    
    for run in runs:
        view_data_mean, view_data_std = view_reproducibility_analysis(models, CV, views, run)
        mean_all_runs.append(view_data_mean)

    mean_all_std = np.std(mean_all_runs, axis=0)
    mean_all_runs = np.mean(mean_all_runs, axis=0)

    X = np.arange(len(views)+1)
    sep = 0.00
    for i, view_d in enumerate(mean_all_runs):
        plt.bar(X + sep, view_d, yerr=mean_all_std[i], capsize=4, width = barWidth, edgecolor ='grey', label=models[i], alpha=0.5)
        sep += barWidth
    
    max_y_lim = np.amax(mean_all_runs) + 0.01
    min_y_lim = np.amin(mean_all_runs) - 0.01
    plt.ylim(min_y_lim, max_y_lim)
    
    title = f"Reproducibility Score for Dataset:{dataset} across {len(runs)} different seeds."
    
    plt.ylabel("Reproducibility Score")
    x_ticks = ["View {}".format(i) for i in views]+ ["Average"]
    
    plt.xticks([r + barWidth for r in range(len(mean_all_runs[0]))], x_ticks)
    plt.title(title)
    plt.grid(axis = 'y')
    plt.legend()
    

    if save_fig:
        if not os.path.exists(SAVE_DIR_FIGS+"reproducibility/"):
            os.makedirs(SAVE_DIR_FIGS+"reproducibility/")
        
        plt.savefig(SAVE_DIR_FIGS+"reproducibility/"+title+".png", dpi=150)
        plt.clf()
    
    else:
        plt.show()
        plt.clf()    

def plot_bar_chart_metric_multiple_runs(dataset, view, models, CV, runs, metric, dataset_split, analysis_type, save_fig=False):
    
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
    
    barWidth = 1/(len(models)+1)

    mean_all_runs = []
    for run in [i for i in range(10)]:
        view_data_mean, _ = view_metric_analysis(models=models, CV=CV, view=view, run=run, metric=metric, dataset=dataset, dataset_split=dataset_split, analysis_type=analysis_type)
        mean_all_runs.append(view_data_mean)

    mean_all_std = np.std(mean_all_runs, axis=0).squeeze()
    mean_all_runs = np.mean(mean_all_runs, axis=0).squeeze()

    #GET MEAN AND STD ACROSS MEAN OF RUNS
    mean_all_runs = np.c_[ mean_all_runs, np.mean(mean_all_runs, axis=1)]     
    mean_all_std = np.c_[ mean_all_std, np.std(mean_all_runs, axis=1)]  

    mean_all_runs, mean_all_std

    mean_all_std = np.std(mean_all_runs, axis=0)
    mean_all_runs = np.mean(mean_all_runs, axis=0)

    X = np.arange(len(CV)+1)
    sep = 0.00
    for i, view_d in enumerate(mean_all_runs):
        plt.bar(X + sep, view_d, yerr=mean_all_std[i], capsize=4, width = barWidth, edgecolor ='grey', label=models[i], alpha=0.5)
        sep += barWidth
    
    max_y_lim = np.amax(mean_all_runs) + 0.05
    min_y_lim = np.amin(mean_all_runs) - 0.05
    plt.ylim(min_y_lim, max_y_lim)
    
    title = f"Dataset:{dataset}, Metric:{metric}, View:{view}, Across: {len(runs)} seeds"
    
    plt.ylabel(f"Metric: {metric}")
    x_ticks = ["CV {}".format(i) for i in CV]+ ["Average"]
    
    plt.xticks([r + barWidth for r in range(len(mean_all_runs))], x_ticks)
    plt.title(title)
    plt.grid(axis = 'y')
    plt.legend()
    

    if save_fig:
        if not os.path.exists(SAVE_DIR_FIGS+"reproducibility/"):
            os.makedirs(SAVE_DIR_FIGS+"reproducibility/")
        
        plt.savefig(SAVE_DIR_FIGS+"reproducibility/"+title+".png", dpi=150)
        plt.clf()
    
    else:
        plt.show()
        plt.clf()   
################################ PLOT DATA ################################

np.random.seed(0)
random.seed(0)

"""
plot_learning_curves(dataset="gender_data", evaluation_method='model_selection', views=[0, 2, 4, 5], model="gcn", folds=3, save_fig=True)
plot_metric_curves(dataset="gender_data", evaluation_method='model_selection', views=[0, 2, 4, 5], model="gcn", folds=3, metric='acc', save_fig=True)
plot_metric_curves(dataset="gender_data", evaluation_method='model_selection', views=[0, 2, 4, 5], model="gcn", folds=3, metric='f1', save_fig=True)
plot_metric_curves(dataset="gender_data", evaluation_method='model_selection', views=[0, 2, 4, 5], model="gcn", folds=3, metric='recall', save_fig=True)
plot_metric_curves(dataset="gender_data", evaluation_method='model_selection', views=[0, 2, 4, 5], model="gcn", folds=3, metric='precision', save_fig=True)

plot_learning_curves(dataset="gender_data", evaluation_method='model_selection', views=[0, 2, 4, 5], model="gcn_student", folds=3, save_fig=True)
plot_metric_curves(dataset="gender_data", evaluation_method='model_selection', views=[0, 2, 4, 5], model="gcn_student", folds=3, metric='acc', save_fig=True)
plot_metric_curves(dataset="gender_data", evaluation_method='model_selection', views=[0, 2, 4, 5], model="gcn_student", folds=3, metric='f1', save_fig=True)
plot_metric_curves(dataset="gender_data", evaluation_method='model_selection', views=[0, 2, 4, 5], model="gcn_student", folds=3, metric='recall', save_fig=True)
plot_metric_curves(dataset="gender_data", evaluation_method='model_selection', views=[0, 2, 4, 5], model="gcn_student", folds=3, metric='precision', save_fig=True)

plot_learning_curves(dataset="gender_data", evaluation_method='model_selection', views=[0, 2, 4, 5], model="gcn_student", folds=3, with_teacher=True, save_fig=True)
plot_metric_curves(dataset="gender_data", evaluation_method='model_selection', views=[0, 2, 4, 5], model="gcn_student", folds=3, metric='acc', with_teacher=True, save_fig=True)
plot_metric_curves(dataset="gender_data", evaluation_method='model_selection', views=[0, 2, 4, 5], model="gcn_student", folds=3, metric='f1', with_teacher=True, save_fig=True)
plot_metric_curves(dataset="gender_data", evaluation_method='model_selection', views=[0, 2, 4, 5], model="gcn_student", folds=3, metric='recall', with_teacher=True, save_fig=True)
plot_metric_curves(dataset="gender_data", evaluation_method='model_selection', views=[0, 2, 4, 5], model="gcn_student", folds=3, metric='precision', with_teacher=True, save_fig=True)
"""

"""
plot_bar_chart(dataset="gender_data", views=[0, 1], models=[ "gcn", "gat", "diffpool", "gunet", "sag"], folds=3, metric="acc", save_fig=True)
plot_bar_chart(dataset="gender_data", views=[0, 1], models=[ "gcn", "gat", "diffpool", "gunet", "sag"], folds=3, metric="prec", save_fig=True)
plot_bar_chart(dataset="gender_data", views=[0, 1], models=[ "gcn", "gat", "diffpool", "gunet", "sag"], folds=3, metric="recall", save_fig=True)
plot_bar_chart(dataset="gender_data", views=[0, 1], models=[ "gcn", "gat", "diffpool", "gunet", "sag"], folds=3, metric="F1", save_fig=True)

for model in [ "gcn", "gat", "diffpool", "gunet", "sag"]:
    plot_learning_curves(dataset="gender_data", views=[0, 1], model=model, folds=3, save_fig=True)

plot_random_sample("gender_data", [0,1], True)
plot_random_sample("gender_data", [0,1], True)
plot_random_sample("gender_data", [0,1], True)
"""

"""
for cv in [3, 5, 10]:
    plot_bar_chart(dataset="gender_data", views=[0], models=["gcn", "gcn_student", "gcn_student_teacher"], folds=cv, metric="acc", save_fig=False) "gcn_student_teacher", "gcn_student_teacher_weight"
"""
"""
"""
#plot_bar_chart_rep(dataset="gender_data", views=[0, 2, 4, 5], models=["gcn", "gcn_student","gcn_student_teacher",  "gcn_student_teacher_weight"], CV=["3Fold", "5Fold", "10Fold"], run=5, save_fig=False)

plot_bar_chart_reproducibility_mulitple_runs(dataset="gender_data", views=[0, 2, 4, 5], models=["gcn", "gcn_student","gcn_student_teacher",  "gcn_student_teacher_weight"], CV=["3Fold", "5Fold", "10Fold"], runs=[i for i in range(10)], save_fig=False)

#plot_bar_chart_metric_multiple_runs(dataset="gender_data", view=0, models=["gcn", "gcn_student","gcn_student_teacher",  "gcn_student_teacher_weight"], CV=["3Fold", "5Fold", "10Fold"], runs=[i for i in range(10)], metric="acc", dataset_split="val", analysis_type="model_assessment")