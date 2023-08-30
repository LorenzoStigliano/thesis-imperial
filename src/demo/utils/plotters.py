import random
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import sklearn.metrics as metrics
import os
import pickle

from utils.loaders import load_data
from utils.config import SAVE_DIR_FIGS, SAVE_DIR_MODEL_DATA
from utils.analysis import view_specific_rep, view_reproducibility_analysis, view_metric_analysis

def plot_random_sample(dataset, views, save_fig=False):
    """
    Plot a random sample from a specified dataset and views.

    Parameters:
        dataset (str): Name of the dataset.
        views (list): List of views to plot.
        save_fig (bool, optional): Whether to save the figures (default is False).
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
    """
    Plot learning curves for a specified dataset, views, and model.

    Parameters:
        dataset (str): Name of the dataset.
        views (list): List of views to plot.
        model (str): Model name.
        evaluation_method (str): Evaluation method.
        folds (int): Number of folds.
        with_teacher (bool, optional): Whether to include teacher model (default is False).
        save_fig (bool, optional): Whether to save the figures (default is False).
        run (int, optional): Run number (default is 0).
    """
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
                    directory_validation_losses = f'{SAVE_DIR_MODEL_DATA}{evaluation_method}/{model}/validation_loss/validation_loss_MainModel_{folds}Fold_{dataset}_{model}_run_{run}_CV_{fold}_view_{view}.pickle'

            print(directory_train_losses)
            print(directory_validation_losses)
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
    """
    Plot metric curves for a specified dataset, views, and model.

    Parameters:
        dataset (str): Name of the dataset.
        views (list): List of views to plot.
        model (str): Model name.
        evaluation_method (str): Evaluation method.
        folds (int): Number of folds.
        metric (str): Metric to plot.
        with_teacher (bool, optional): Whether to include teacher model (default is False).
        save_fig (bool, optional): Whether to save the figures (default is False).
    """
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
    """
    Plot bar charts for a specified dataset, views, models, and metric.

    Parameters:
        dataset (str): Name of the dataset.
        views (list): List of views to plot.
        models (list): List of model names.
        folds (int): Number of folds.
        metric (str): Metric to plot.
        save_fig (bool, optional): Whether to save the figures (default is False).
    """
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
    """
    Plot bar charts for reproducibility scores across different runs.

    Parameters:
        dataset (str): Name of the dataset.
        views (list): List of views to plot.
        models (list): List of model names.
        CV (list): List of CV types.
        run (int): Run number.
        save_fig (bool, optional): Whether to save the figures (default is False).
    """
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
        if model[i] == "gcn":
            plt.bar(X + sep, view_d, width = barWidth, edgecolor ='grey', label=models[i]+"_teacher")
        else:
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

def plot_bar_chart_reproducibility_mulitple_runs(dataset, views, models, CV, runs, students=0, model_args=None, save_fig=False):
    """
    Plot bar charts for reproducibility scores across multiple runs.

    Parameters:
        dataset (str): Name of the dataset.
        views (list): List of views to plot.
        models (list): List of model names.
        CV (list): List of CV types.
        runs (list): List of run numbers.
        students (int, optional): Number of student models (default is 0).
        model_args (dict, optional): Model arguments (default is None).
        save_fig (bool, optional): Whether to save the figures (default is False).
    """
    plt.rcParams["figure.figsize"] = (20,8)

    barWidth = 1/(len(models)+1)

    mean_all_runs = []
    
    for run in runs:
        view_data_mean, _ = view_reproducibility_analysis(dataset, models, CV, views, run, students, model_args)
        mean_all_runs.append(view_data_mean)

    mean_all_std = np.std(mean_all_runs, axis=0)
    mean_all_runs = np.mean(mean_all_runs, axis=0)

    X = np.arange(len(views)+1)
    sep = 0.00
    for i, view_d in enumerate(mean_all_runs):
        if models[i] == "gcn":
            plt.bar(X + sep, view_d, yerr=mean_all_std[i], capsize=4, width = barWidth, edgecolor ='grey', label=models[i]+"_teacher", alpha=0.5)
        else:
            plt.bar(X + sep, view_d, yerr=mean_all_std[i], capsize=4, width = barWidth, edgecolor ='grey', label=models[i], alpha=0.5)
        
        sep += barWidth
    
    max_y_lim = 1 if np.amax(mean_all_runs) + np.max(mean_all_std) > 1 else np.amax(mean_all_runs) + np.max(mean_all_std)
    min_y_lim = 0 if np.amin(mean_all_runs) - np.max(mean_all_std) - 0.01 < 0 else np.amin(mean_all_runs) - np.max(mean_all_std) - 0.01
    plt.ylim(min_y_lim, max_y_lim)
    
    title = f"Reproducibility Score for Dataset:{dataset} across {len(runs)} different seeds"
    
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

def plot_bar_chart_metric_multiple_runs(dataset, view, models, CV, runs, metric, dataset_split, analysis_type, model_args=None, save_fig=False):
    """
    Plot bar charts for metric scores across multiple runs.

    Parameters:
        dataset (str): Name of the dataset.
        view (int): View to plot.
        models (list): List of model names.
        CV (list): List of CV types.
        runs (list): List of run numbers.
        metric (str): Metric to plot.
        dataset_split (str): Dataset split ('train', 'val', 'test').
        analysis_type (str): Analysis type ('model_selection', 'model_assessment').
        model_args (dict, optional): Model arguments (default is None).
        save_fig (bool, optional): Whether to save the figures (default is False).
    """
    barWidth = 1/(len(models)+1)

    mean_all_runs = []
    for run in runs:
        view_data_mean, _ = view_metric_analysis(models=models, CV=CV, view=view, run=run, metric=metric, dataset=dataset, dataset_split=dataset_split, analysis_type=analysis_type, model_args=model_args)
        mean_all_runs.append(view_data_mean)

    mean_all_std = np.std(mean_all_runs, axis=0).squeeze()
    mean_all_runs = np.mean(mean_all_runs, axis=0).squeeze()

    #GET MEAN AND STD ACROSS MEAN OF RUNS
    mean_all_runs = np.c_[ mean_all_runs, np.mean(mean_all_runs, axis=1)]     
    mean_all_std = np.c_[ mean_all_std, np.std(mean_all_runs, axis=1)]  
    X = np.arange(len(CV)+1)
    sep = 0.00
    for i, view_d in enumerate(mean_all_runs):
        if models[i] == "gcn":
             plt.bar(X + sep, view_d, yerr=mean_all_std[i], capsize=4, width = barWidth, edgecolor ='grey', label=models[i]+"_teacher", alpha=0.5)
        else:
            plt.bar(X + sep, view_d, yerr=mean_all_std[i], capsize=4, width = barWidth, edgecolor ='grey', label=models[i], alpha=0.5)
        sep += barWidth
    
    max_y_lim = np.amax(mean_all_runs) + 0.05
    min_y_lim = np.amin(mean_all_runs) - 0.05
    plt.ylim(min_y_lim, max_y_lim)
    
    #title = f"Dataset:{dataset}, Metric:{metric}, View:{view}, Across: {len(runs)} seeds with fixed init"
    title = f"{metric} across view {view} for 3, 5 and 10-Fold CV"
    
    plt.ylabel(f"Metric: {metric}")
    x_ticks = [i for i in CV]+ ["Average"]
    
    plt.xticks([r + barWidth for r in range(len(CV)+1)], x_ticks)
    plt.title(title)
    plt.grid(axis = 'y')
    plt.legend()
    

    if save_fig:
        if not os.path.exists(SAVE_DIR_FIGS+"metric_results/"):
            os.makedirs(SAVE_DIR_FIGS+"metric_results/")
        
        plt.savefig(SAVE_DIR_FIGS+"metric_results/"+title+".png", dpi=150)
        plt.clf()
    
    else:
        plt.show()
        plt.clf()   
