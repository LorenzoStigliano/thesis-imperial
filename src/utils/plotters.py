import random
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import sklearn.metrics as metrics
import os
import pickle

from loaders import load_data
from config import SAVE_DIR_FIGS, SAVE_DIR_MODEL_DATA
from self_reproducibility import view_specific_rep

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

def plot_learning_curves(dataset, views, model, folds, save_fig=False):
    
    for view in views:
        for fold in range(folds):
            with open('{}{}/training_loss/Training_loss_MainModel_{}Fold_{}_{}_CV_{}_view_{}.pickle'.format(SAVE_DIR_MODEL_DATA,
                                                                                                            model,
                                                                                                            folds,
                                                                                                            dataset,
                                                                                                            model,
                                                                                                            fold,
                                                                                                            view),'rb') as f:
                losses = pickle.load(f)  
            epochs = range(len(losses['loss']))
            if isinstance(losses['loss'][0], float):
                losses = losses['loss']
            else:
                losses = [i.item() for i in (losses['loss'])]
                
            plt.plot(epochs, losses, label='CV {}'.format(fold))
        
        title = "Dataset:{}, Model:{}, View:{}".format(dataset, model, view)
        plt.title(title)
        plt.legend()

        if save_fig:
            if not os.path.exists(SAVE_DIR_FIGS+"loss_curves/"):
                os.makedirs(SAVE_DIR_FIGS+"loss_curves/")
            
            plt.savefig(SAVE_DIR_FIGS+"loss_curves/"+title+".png", dpi=150)
            plt.clf()
        
        else:
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

def plot_bar_chart_rep(dataset, views, models, CV, save_fig=False):
    
    view_data = []
    
    for view in views:
        
        barWidth = 1/(len(models)+1)
        model_result = []
        
        for model in models:
            rep_score = view_specific_rep(dataset=dataset, view=view, model=model, CV=CV)
            model_result.append(rep_score)
        
        view_data.append(model_result)

    view_data = np.array(view_data).T
    X = np.arange(len(views))
    sep = 0.00
    for i, view_d in enumerate(view_data):
        plt.bar(X + sep, view_d, width = barWidth, edgecolor ='grey', label=models[i])
        sep += barWidth
        
    title = "Reproducibility Score for Dataset:{}".format(dataset)
    
    plt.ylabel("Reproducibility Score")
    x_ticks = ["View {}".format(i) for i in range(len(views))]
    
    plt.xticks([r + barWidth for r in range(len(views))], x_ticks)
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

################################ PLOT DATA ################################

np.random.seed(0)
random.seed(0)

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

#plot_bar_chart(dataset="gender_data", views=[5], models=["gcn", "gcn_student", "gcn_student_teacher"], folds=3, metric="acc", save_fig=False)

"""
for cv in [3, 5, 10]:
    plot_bar_chart(dataset="gender_data", views=[0,1,2,3,4,5], models=["gcn", "gcn_student"], folds=cv, metric="acc", save_fig=True)

plot_bar_chart_rep(dataset="gender_data", views=[0,1,2,3,4,5], models=["gcn", "gcn_student", "gcn_student_teacher"], CV=["3Fold", "5Fold", "10Fold"], save_fig=False)

"""

plot_bar_chart_rep(dataset="gender_data", views=[0], models=["gcn", "gcn_student"], CV=["3Fold", "5Fold", "10Fold"], save_fig=False)
