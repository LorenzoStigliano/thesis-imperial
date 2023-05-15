
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

from models.gcn_student import GCN_STUDENT
from models.model_config import * 
from utils.helpers import *
from config import SAVE_DIR_MODEL_DATA

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class CrossEntropyLossForSoftTarget(nn.Module):
    def __init__(self, T=3, alpha=1):
        super(CrossEntropyLossForSoftTarget, self).__init__()
        self.T = T
        self.alpha = alpha
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
    def forward(self, y_pred, y_gt):
        y_pred_soft = y_pred.div(self.T)
        y_gt_soft = y_gt.div(self.T)
        return -(self.softmax(y_gt_soft)*self.logsoftmax(y_pred_soft)).mean().mul(self.alpha)

def train(model_args, train_dataset, val_dataset, student_model, threshold_value, model_name):

    # Create model
    teacher_model = torch.load(SAVE_DIR_MODEL_DATA+"gcn/models/gcn_MainModel_gender_data_gcn_view_0_teacher.pt")
    teacher_model.is_trained = False
    teacher_model.eval()

    # Transfer
    teacher_model.to(device)
    student_model.to(device)

    # Define Loss
    criterion = nn.CrossEntropyLoss(reduction='mean')
    criterion_soft = CrossEntropyLossForSoftTarget()

    # Define optimizer
    optimizer = optim.Adam(student_model.parameters(), lr=model_args["lr"], weight_decay=model_args['weight_decay'])

    test_accs = []
    train_loss=[]

    student_model.train()
    for epoch in range(model_args["num_epochs"]):
        
        print("Epoch ",epoch)
        print("Size of Training Set:" + str(len(train_dataset)))
        print("Size of Validation Set:" + str(len(val_dataset)))
        
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
            
            y_gt = label.to(device)

            # Compute soft label
            y_soft = teacher_model(features, adj)

            # Predict
            y_pred = student_model(features, adj)

            # Compute loss (foward propagation)
            loss = criterion(y_pred, y_gt) + criterion_soft(y_pred, y_soft)
            
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
              student_model.is_trained = True

        #list_lr_momentum_scheduler.step()
        preds = np.hstack(preds)
        labels = np.hstack(labels)
        print("Train accuracy : ", np.mean( preds == labels ))
        print('Loss: {}'.format(total_loss / len(train_dataset)))

        test_acc = evaluate(val_dataset, student_model, model_args, threshold_value, model_name)
        test_accs.append(test_acc)
        train_loss.append(total_loss)

    # Save training loss of GNN model
    los_p = {'loss':train_loss}
    with open(SAVE_DIR_MODEL_DATA+model_args['model_name']+"/training_loss/Training_loss_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
      
    # Save Model
    torch.save(student_model, SAVE_DIR_MODEL_DATA+model_args['model_name']+"/models/"+model_args['model_name']+"_"+model_name+".pt")
    
    # Save weights
    if model_args['model_name'] == "diffpool":
        w_dict = {"w": student_model.state_dict()["assign_conv_first_modules.0.weight"]}
        with open(SAVE_DIR_MODEL_DATA+model_args['model_name']+'/weights/W_'+model_name+'.pickle', 'wb') as f:
            pickle.dump(w_dict, f)
    else:
        path = SAVE_DIR_MODEL_DATA+model_args['model_name']+'/weights/W_'+model_name+'.pickle'
        
        if os.path.exists(path):
            os.remove(path)

        os.rename(model_args['model_name']+'_W.pickle'.format(), path)

    return test_acc


def evaluate(dataset, model, model_args, threshold_value, model_name):
    """
    Parameters
    ----------
    dataset : dataloader (dataloader for the validation/test dataset).
    model : nn model (GCN model).
    model_args : arguments
    threshold_value : float (threshold for adjacency matrices).
    
    Description
    ----------
    This methods performs the evaluation of the model on test/validation dataset
    
    Returns
    -------
    test accuracy.
    """
    model.eval()
    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).to(device)
        labels.append(data['label'].long().numpy())
        
        adj = torch.squeeze(adj)
        
        features = np.identity(adj.shape[0])
        
        features = Variable(torch.from_numpy(features).float(), requires_grad=False).to(device)
        
        if model_args["threshold"] in ["median", "mean"]:
            adj = torch.where(adj > threshold_value, torch.tensor([1.0]).to(device), torch.tensor([0.0]).to(device))
        
        if model_args["model_name"] == 'diffpool':
            batch_num_nodes=np.array([adj.shape[1]])
            features = torch.unsqueeze(features, 0)
            assign_input = np.identity(adj.shape[1])
            assign_input = Variable(torch.from_numpy(assign_input).float(), requires_grad=False).to(device)
            assign_input = torch.unsqueeze(assign_input, 0)
            ypred= model(features, adj, batch_num_nodes, assign_x=assign_input)
        
        elif model_args["model_name"] == "mlp":
            features = torch.mean(features, axis=1)
            ypred = model(features)[1]
        else:
            ypred = model(features, adj)

        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

    labels = np.hstack(labels)
    preds = np.hstack(preds)

    simple_r = {'labels':labels,'preds':preds}
    # Save labels and predictions of model.
    with open(SAVE_DIR_MODEL_DATA+model_args["model_name"]+"/labels_and_preds/"+model_name+".pickle", 'wb') as f:
      pickle.dump(simple_r, f)
    
    if model_args["evaluation_method"] == 'model assessment':
        name = 'Test'
    if model_args["evaluation_method"] == 'model selection':
        name = 'Validation'
    
    print(name, " accuracy:", metrics.accuracy_score(labels, preds))
    
    return metrics.accuracy_score(labels, preds)

def cv_benchmark(model_args, G_list, model_name, cv_number, view):
    """
    Parameters
    ----------
    model_args : Arguments
    Description
    ----------
    Initiates the model and performs train/test or train/validation splits and calls train() to execute training and evaluation.
    Returns
    -------
    test_accs : test accuracies (list)
    """
    test_accs = []
    folds = stratify_splits(G_list, cv_number)
    
    [random.shuffle(folds[i]) for i in range(len(folds))]
    
    for i in range(cv_number):
        train_set, validation_set, test_set = datasets_splits(folds, model_args, i)
        if model_args["evaluation_method"] =='model selection':
            train_dataset, val_dataset, threshold_value = model_selection_split(train_set, validation_set, model_args)
        if model_args["evaluation_method"] =='model assessment':
            train_dataset, val_dataset, threshold_value = model_assessment_split(train_set, validation_set, test_set, model_args)
        
        print("CV : ",i)
        
        num_nodes = G_list[0]['adj'].shape[0]
        num_classes = 2 #TODO: make sure it can be used with any number of classes
        student_model = GCN_STUDENT(
            nfeat = num_nodes,
            nhid = model_args["hidden_dim"],
            nclass = num_classes,
            dropout = model_args["dropout"]
        ).to(device) 
        test_acc = train(model_args, train_dataset, val_dataset, student_model, threshold_value, model_name+"_CV_"+str(i)+"_view_"+str(view)+"_teacher")
        test_accs.append(test_acc)
    
    return test_accs

def test_scores(model_args, G_list, view, model_name, cv_number):
    
    print("Main : ", view, model_name, cv_number)
    test_accs = cv_benchmark(model_args, G_list, model_name, cv_number, view)
    print("test accuracies ", test_accs)
    
    return test_accs
 

 