
import torch
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sklearn.metrics as metrics

from models.gcn import GCN
from models.gcn_student import GCN_STUDENT
from models.model_config import * 
from utils.helpers import *
from config import SAVE_DIR_MODEL_DATA

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train(model_args, train_dataset, model, threshold_value, model_name):

    model.to(device)

    # Define Loss
    criterion = nn.CrossEntropyLoss(reduction='mean')

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=model_args["lr"], weight_decay=model_args['weight_decay'])

    test_accs = []
    train_loss=[]

    model.train()
    for epoch in range(model_args["num_epochs"]):
        
        print("Epoch ",epoch)
        print("Size of Training Set:" + str(len(train_dataset)))
        
        preds = []
        labels = []

        ## Optimize parameters
        total_loss = 0.0
        for _, data in enumerate(train_dataset):
            # Initialize gradients with 0
            optimizer.zero_grad()

            # Transfer device
            adj = Variable(data['adj'].float(), requires_grad=False).to(device)
            adj = torch.squeeze(adj)
            
            label = Variable(data['label'].long()).to(device)     
            
            features = np.identity(adj.shape[0])
            features = Variable(torch.from_numpy(features).float(), requires_grad=False).to(device)       
            
            if model_args["threshold"] in ["median", "mean"]:
                adj = torch.where(adj > threshold_value, torch.tensor([1.0]).to(device), torch.tensor([0.0]).to(device))
            
            # Predict
            y_pred = model(features, adj)

            # Compute loss (foward propagation)
            loss = model.loss(y_pred, label)
            
            # Compute gradients (backward propagation)
            loss.backward()
            
            # Update parameters
            optimizer.step()
                    
            total_loss += loss.item()

            _, indices = torch.max(y_pred, 1)
            preds.append(indices.cpu().data.numpy())
            labels.append(data['label'].long().numpy())

        # Save weights of GNN model
        if epoch==model_args["num_epochs"]-1:
              model.is_trained = True

        #list_lr_momentum_scheduler.step()
        preds = np.hstack(preds)
        labels = np.hstack(labels)
        print("Train accuracy : ", np.mean( preds == labels ))
        print('Loss: {}'.format(total_loss / len(train_dataset)))

        train_loss.append(total_loss)

    # Save Model
    torch.save(model, SAVE_DIR_MODEL_DATA+model_args['model_name']+"/models/"+model_args['model_name']+"_"+model_name+".pt")

def train_teacher(model_args, G_list, view, model_name):

    dataset_sampler = GraphSampler(G_list)
    train_dataset = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size = 1,  
            shuffle = False)  
    
    num_nodes = G_list[0]['adj'].shape[0]
    num_classes = 2 #TODO: make sure it can be used with any number of classes
    model = GCN(
        nfeat = num_nodes,
        nhid = model_args["hidden_dim"],
        nclass = num_classes,
        dropout = model_args["dropout"]
    ).to(device) 
    
    train_mean, train_median = get_stats(G_list)
    if model_args["threshold"] == "mean":
        threshold_value = train_mean
    else:
        threshold_value = train_median

    train(model_args, train_dataset, model, threshold_value, model_name+"_view_"+str(view)+"_teacher")
 

 