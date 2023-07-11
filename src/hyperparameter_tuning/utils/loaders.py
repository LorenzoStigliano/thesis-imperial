from sklearn import preprocessing
from torch_geometric import utils
from torch_geometric.data import Data

from config import SAVE_DIR_DATA

import torch
import pickle
import numpy as np

#TODO: make sure we can use these classes for data with more than 2 classes!

def load_data(dataset, view, NormalizeInputGraphs=False):
    """
    Parameters
    ----------

    Description
    ----------
    This methods loads the adjacency matrices representing the args.view -th view in dataset
    
    Returns
    -------
    List of dictionaries{adj, label, id}
    """
    if dataset =='gender_data':
        with open(SAVE_DIR_DATA+dataset+'/'+dataset+'_edges','rb') as f:
            multigraphs = pickle.load(f)        
        with open(SAVE_DIR_DATA+dataset+'/'+dataset+'_labels','rb') as f:
            labels = pickle.load(f)
        adjacencies = [multigraphs[i][:,:,view] for i in range(len(multigraphs))]
        #Normalize inputs
        if NormalizeInputGraphs==True:
            for subject in range(len(adjacencies)):
                adjacencies[subject] = minmax_sc(adjacencies[subject])

        #Create List of Dictionaries
        G_list=[]
        for i in range(len(labels)):
            if  labels[i] == -1: 
                    G_element = {"adj": adjacencies[i],"label": 0,"id": i}
            else:
                G_element = {"adj": adjacencies[i],"label": labels[i],"id":  i}
            G_list.append(G_element)
        return G_list
    
    else:
        with open(SAVE_DIR_DATA+dataset+'/'+dataset+'_images','rb') as f:
            images = pickle.load(f)        
        with open(SAVE_DIR_DATA+dataset+'/'+dataset+'_labels','rb') as f:
            labels = pickle.load(f)

        #Create List of Dictionaries
        G_list=[]
        for i in range(len(labels)):
            adj = images[i]
            G_element = {"adj": adj,"label": labels[i], "id":  i}
            G_list.append(G_element)
        
        return G_list        

def load_data_pg(dataset, view, NormalizeInputGraphs):
    """
    Parameters
    ----------

    Description
    ----------
    This methods loads the adjacency matrices representing the args.view -th view in dataset
    
    Returns
    -------
    List of dictionaries{adj, label, id}
    """
    with open(SAVE_DIR_DATA+dataset+'/'+dataset+'_edges','rb') as f:
        multigraphs = pickle.load(f)        
    with open(SAVE_DIR_DATA+dataset+'/'+dataset+'_labels','rb') as f:
        labels = pickle.load(f)
    adjacencies = [multigraphs[i][:,:,view] for i in range(len(multigraphs))]
    #Normalize inputs
    if NormalizeInputGraphs==True:
        for subject in range(len(adjacencies)):
            adjacencies[subject] = minmax_sc(adjacencies[subject])
    
    #Create List of Dictionaries
    G_list=[]
    for i in range(len(labels)):
        adj = adjacencies[i]
        edge_index, edge_values = utils.dense_to_sparse(adj)
        x = torch.eye(adj.shape[0])
        if  labels[i] == -1: 
            G_element = Data(x=x, edge_index=edge_index, edge_attr=edge_values, adj=adj, y=torch.tensor([0]))
        else:
            G_element =  Data(x=x, edge_index=edge_index, edge_attr=edge_values, adj=adj, y=torch.tensor([1]))
        G_list.append(G_element)
    
    return G_list

def minmax_sc(x):
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    return x

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
    