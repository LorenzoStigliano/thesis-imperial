from sklearn import preprocessing
from torch_geometric import utils
from torch_geometric.data import Data

import torch
import pickle
import numpy as np

def load_data(dataset, view, SAVE_DIR_DATA, NormalizeInputGraphs=False):
    """
    Load data for the specified dataset and view.

    Parameters:
        dataset (str): Name of the dataset ('gender_data' or other).
        view (int): Index of the view to load.
        SAVE_DIR_DATA (str): Directory where data is stored.
        NormalizeInputGraphs (bool, optional): Whether to normalize input graphs (default is False).

    Returns:
        list: List of dictionaries, each containing 'adj', 'label', and 'id' keys.
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

def minmax_sc(x):
    """
    Apply min-max scaling to input data.

    Parameters:
        x (numpy.ndarray): Input data.

    Returns:
        numpy.ndarray: Scaled data.
    """
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    return x

class GraphSampler(torch.utils.data.Dataset):
    """
    Custom dataset class for graph data.

    Parameters:
        G_list (list): List of dictionaries containing graph data.

    Returns:
        dict: A dictionary containing graph adjacency, labels, and ID.
    """
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
    