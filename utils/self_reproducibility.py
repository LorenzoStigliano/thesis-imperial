import pickle
import numpy as np
import os 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
import torch

from config import SAVE_DIR_MODEL_DATA


def extract_weights_single(dataset, view, model, training_type, shot_n, cv_n):
    fs_path = SAVE_DIR_MODEL_DATA+'{}/weights/W_{}_{}_{}{}_view_{}.pickle'.format(model, training_type, dataset, model, shot_n, view)
    cv_path = SAVE_DIR_MODEL_DATA+'{}/weights/W_MainModel_{}_{}_{}_CV_{}_view_{}.pickle'.format(model,training_type, dataset, model, cv_n, view)
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

def extract_weights(dataset, view, model, training_type):
    runs = []
    if training_type == 'Few_Shot':
        for shot_i in range(5):
            runs.append(extract_weights_single(dataset, view, model, training_type, shot_i, 0))
    if training_type == '3Fold':
        for cv_i in range(3):
            runs.append(extract_weights_single(dataset, view, model, training_type, 0, cv_i))
    if training_type == '5Fold':
        for cv_i in range(5):
            runs.append(extract_weights_single(dataset, view, model, training_type, 0, cv_i))
    if training_type == '10Fold':
        for cv_i in range(10):
            runs.append(extract_weights_single(dataset, view, model, training_type, 0, cv_i))
    runs = np.array(runs)
    weights = np.mean(runs, axis=0)
    return weights

def top_biomarkers(weights, K_i):
    weights_normalized = np.abs(weights)
    result = []
    w_sorted = weights_normalized.argsort()  #verified
    for i in range(1, 1+K_i):
        result.append(w_sorted[-1*i])
    return result

def sim(nodes1, nodes2):
    if len(nodes1)==len(nodes2):
        counter = 0
        for i in nodes1:
            for k in nodes2:
                if i==k:
                    counter+=1
        return counter/len(nodes1)
    else:
        print('nodes vectors are not caompatible')

def view_specific_rep(dataset, view, model, CV):
    #models = ['diffpool', 'gat', 'gcn', 'gunet', 'sag']
    Ks = [5, 10, 15, 20]
    #CV = ["3Fold", "5Fold", "10Fold"]
    rep = np.zeros([len(CV), len(CV), len(Ks)])
    
    for i in range(rep.shape[0]):
        for j in range(rep.shape[1]):
            weights_i = extract_weights(dataset, view, model, CV[i])
            weights_j = extract_weights(dataset, view, model, CV[j])
            for k in range(rep.shape[2]):
                top_bio_i = top_biomarkers(weights_i, Ks[k])
                top_bio_j = top_biomarkers(weights_j, Ks[k])
                rep[i,j,k] = sim(top_bio_i, top_bio_j)
    
    rep_mean = np.mean(rep, axis=2)
    
    # Get the elements above the diagonal
    elements_above_diagonal = np.where(np.triu(np.ones_like(rep_mean), k=1), rep_mean, np.nan)

    # Drop the zeros and calculate the average
    average = np.nanmean(elements_above_diagonal)
    
    return average

CV = ["3Fold", "5Fold", "10Fold"]
views = [0, 1, 2, 3, 4, 5]

for view in views: 

    gcn_rep_score     = view_specific_rep(dataset="gender_data", view=view, model="gcn", CV=CV)
    gcn_student_score = view_specific_rep(dataset="gender_data", view=view, model="gcn_student", CV=CV)
    
    print("View:{}, Teacher network score:{}, Student network score:{}".format(view, gcn_rep_score, gcn_student_score))

