import math
import numpy as np
import torch
import os 
import torch.utils.data

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

class GraphSampler(torch.utils.data.Dataset):
    
    def __init__(self, G_list):
        self.adj_all = []
        self.label_all = []
        self.id_all = []
        
        for i in range(len(G_list)):
            self.adj_all.append(G_list[i]['adj'])
            self.label_all.append(G_list[i]['label'])
            self.id_all.append(G_list[i]['id'])

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        return {'adj':self.adj_all[idx],
                'label':self.label_all[idx],
                'id':self.id_all[idx]}

# Returns train, validation and test sets.
def datasets_splits(folds, args, val_idx):
    train = []
    validation = []
    test = []
    train_folds = []
    for i in range(len(folds)):
        if i==val_idx:
            test.extend(folds[i])
        else:
            train_folds.append(folds[i])
    validation.extend(train_folds[0])
    for i in range(1, len(train_folds)):
        train.extend(train_folds[i])
    return train, validation, test

def model_selection_split(train, validation, args):
    print('Num training graphs: ', len(train), '; Num test graphs: ', len(validation))
    
    # minibatch
    dataset_sampler = GraphSampler(train)
    train_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size = 1,  
            shuffle = False)  

    dataset_sampler = GraphSampler(validation)
    val_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size = 1,  
            shuffle = False) 
    train_mean, train_median = get_stats(train)
    if(args['threshold'] == 'median'):
        threshold_value = train_median
    elif(args['threshold'] == 'mean'):
        threshold_value = train_mean
    else:
        threshold_value = 0.0
    return train_dataset_loader, val_dataset_loader, threshold_value
    
def model_assessment_split(train, validation, test, args):
    
    train.extend(validation)
    print('Num training graphs: ', len(train), '; Num test graphs: ', len(test))
    
    # minibatch
    dataset_sampler = GraphSampler(train)
    train_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size = 1,  
            shuffle = False)  

    dataset_sampler = GraphSampler(test)
    test_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size = 1,  
            shuffle = False) 
    train_mean, train_median = get_stats(train)
    if(args['threshold'] == 'median'):
        threshold_value = train_median
    if(args['threshold'] == 'mean'):
        threshold_value = train_mean
    
    return train_dataset_loader, test_dataset_loader, threshold_value


# Splits the dataset into k-folds     
def stratify_splits(graphs, cv_number):
    graphs_0 = []
    graphs_1 = []
    for i in range(len(graphs)):
        if graphs[i]['label'] == 0:
            graphs_0.append(graphs[i])
        if graphs[i]['label'] == 1:
            graphs_1.append(graphs[i])
    graphs_0_folds = []
    graphs_1_folds = []
    pop_0_fold_size = math.floor(len(graphs_0) / cv_number)
    pop_1_fold_size = math.floor(len(graphs_1) / cv_number)
    graphs_0_folds = [graphs_0[i:i + pop_0_fold_size] for i in range(0, len(graphs_0), pop_0_fold_size)]
    graphs_1_folds = [graphs_1[i:i + pop_1_fold_size] for i in range(0, len(graphs_1), pop_1_fold_size)]
    folds = []
    for i in range(cv_number):
        fold = []
        fold.extend(graphs_0_folds[i])
        fold.extend(graphs_1_folds[i])
        folds.append(fold)
    
    if len(graphs_0_folds) >cv_number:
        folds[cv_number-1].extend(graphs_0_folds[cv_number])
    if len(graphs_1_folds) > cv_number:
        folds[cv_number-1].extend(graphs_1_folds[cv_number])
    return folds

def get_stats(list_train):
    train_features = []
    for i in range(len(list_train)):
        ut_x_indexes = np.triu_indices(list_train[i]['adj'].shape[0], k=1)
        ut_x = list_train[i]['adj'][ut_x_indexes]
    
    for i in range(len(list_train)):
        ut_x_indexes = np.triu_indices(list_train[i]['adj'].shape[0], k=1)
        ut_x = list_train[i]['adj'][ut_x_indexes]
        train_features.extend(list(ut_x))
        
    train_features = np.array(train_features)
    train_mean = np.mean(train_features)
    train_median = np.median(train_features)
    
    return train_mean, train_median
