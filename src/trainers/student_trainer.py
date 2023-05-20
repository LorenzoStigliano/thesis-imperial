
import time
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
    def __init__(self, T=3, alpha=2):
        super(CrossEntropyLossForSoftTarget, self).__init__()
        self.T = T
        self.alpha = alpha
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
    def forward(self, y_pred, y_gt):
        y_pred_soft = y_pred.div(self.T)
        y_gt_soft = y_gt.div(self.T)
        return -(self.softmax(y_gt_soft)*self.logsoftmax(y_pred_soft)).mean().mul(self.alpha)
    
def weight_similarity_loss(w_teacher, w_student):
    """
    Compute the mean squared error (MSE) loss between the weights of the last layer
    of two networks.
    """    
    # Concatenate and compute the mean squared error
    mse_loss = nn.MSELoss()
    return mse_loss(w_teacher, w_student)

def cross_validation(model_args, G_list, view, model_name, cv_number, run=0):
    start = time.time() 
    print("Run : ",run)
    print("--------------------------------------------------------------------------")
    print("Main : ", view, model_name, cv_number)
    
    folds = stratify_splits(G_list, cv_number)
    
    [random.shuffle(folds[i]) for i in range(len(folds))]
    
    for cv in range(cv_number):

        train_set, validation_set, test_set = datasets_splits(folds, model_args, cv)
        
        if model_args["evaluation_method"] =='model_selection':
            train_dataset, val_dataset, threshold_value = model_selection_split(train_set, validation_set, model_args)
        if model_args["evaluation_method"] =='model_assessment':
            train_dataset, val_dataset, threshold_value = model_assessment_split(train_set, validation_set, test_set, model_args)
        
        print(f"CV : {cv}")
        name = model_name+"_CV_"+str(cv)+"_view_"+str(view)+"_with_teacher"
        print(name)

        train_set, validation_set, test_set = datasets_splits(folds, model_args, cv)
        num_nodes = G_list[0]['adj'].shape[0]
        num_classes = 2 
        student_model = GCN_STUDENT(
            nfeat = num_nodes,
            nhid = model_args["hidden_dim"],
            nclass = num_classes,
            dropout = model_args["dropout"]
        ).to(device) 

        if model_args["evaluation_method"] =='model_selection':
            #Here we leave out the test set since we are not evaluating we can see the performance on the test set after training
            train(model_args, train_dataset, val_dataset, student_model, threshold_value, name,cv, view, cv_number)
            #See performance on the held-out test set 
            dataset_sampler = GraphSampler(test_set)
            test_dataset = torch.utils.data.DataLoader(
                dataset_sampler, 
                batch_size = 1,  
                shuffle = False) 
            test(test_dataset, student_model, model_args, threshold_value, name)

        if model_args["evaluation_method"] =='model_assessment':
            #Here we join the train and validation dataset
            train(model_args, train_dataset, val_dataset, student_model, threshold_value, name,  cv, view, cv_number, run)
    
    print('Time taken', time.time()-start)

def train(model_args, train_dataset, val_dataset, student_model, threshold_value, model_name, cv, view, cv_number, run=0, alpha=2):
    """
    Parameters
    ----------
    model_args : arguments
    train_dataset : dataloader (dataloader for the validation/test dataset).
    val_dataset : dataloader (dataloader for the validation/test dataset).
    model : nn model (GCN model).
    threshold_value : float (threshold for adjacency matrices).
    
    Description
    ----------
    This methods performs the training of the model on train dataset and calls evaluate() method for evaluation.
    """
    # Load teacher model
    if model_args['evaluation_method'] == "model_selection":
       teacher_model = torch.load(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+f"/gcn/models/gcn_MainModel_{cv_number}Fold_gender_data_gcn_CV_{cv}_view_{view}.pt")
    else:
       teacher_model = torch.load(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+f"/gcn/models/gcn_MainModel_{cv_number}Fold_gender_data_gcn_run_{run}_CV_{cv}_view_{view}.pt")
    
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
    
    # Metrics 
    train_loss=[]
    train_acc=[]
    train_ce_loss=[]
    train_soft_ce_loss=[]
    train_f1=[]
    train_recall=[]
    train_precision=[]

    validation_loss=[]
    validation_ce_loss=[]
    validation_soft_ce_loss=[]
    validation_acc=[]
    validation_f1=[]
    validation_recall=[]
    validation_precision=[]
    
    print(f"Size of Training Set: {str(len(val_dataset))}")
    print(f"Size of Validation Set: {str(len(val_dataset))}")
    
    for epoch in range(model_args["num_epochs"]):
        
        student_model.train()
        total_time = 0
        total_loss = 0
        ce_loss = 0
        soft_ce_loss = 0
        
        preds = []
        labels = []
        for _, data in enumerate(train_dataset):
            begin_time = time.time()
            
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
            ypred = student_model(features, adj)

            # Compute loss (foward propagation)
            loss = criterion(ypred, y_gt) + criterion_soft(ypred, y_soft)

            _, indices = torch.max(ypred, 1)
            preds.append(indices.cpu().data.numpy())
            labels.append(data['label'].long().numpy())
            
            # Compute gradients (backward propagation)
            loss.backward()
            
            # Update parameters
            optimizer.step()
                    
            total_loss += loss.item()
            ce_loss += criterion(ypred, y_gt).item()
            soft_ce_loss += criterion_soft(ypred, y_soft).item()

            elapsed = time.time() - begin_time
            total_time += elapsed
        
        # Save weights of GNN model
        if epoch==model_args["num_epochs"]-1:
              student_model.is_trained = True
        
        preds = np.hstack(preds)
        labels = np.hstack(labels)
        result = {
                    'prec': metrics.precision_score(labels, preds),
                    'recall': metrics.recall_score(labels, preds),
                    'acc': metrics.accuracy_score(labels, preds),
                    'F1': metrics.f1_score(labels, preds)
        }
              
        print("---------------------------------")
        print(f"Time taken for epoch {epoch}: {total_time}")
        print(f"Train accuracy: {result['acc']}")
        print(f"Train loss: {total_loss / len(train_dataset)}")

        
        train_loss.append(total_loss / len(train_dataset))
        train_ce_loss.append(ce_loss / len(train_dataset))
        train_soft_ce_loss.append(soft_ce_loss / len(train_dataset))
        train_acc.append(result['acc'])
        train_f1.append(result['F1'])
        train_recall.append(result['recall'])
        train_precision.append(result['prec'])
        
        val_loss, val_ce_loss, val_soft_ce_loss, val_acc, val_precision, val_recall, val_f1 = validate(val_dataset, student_model, model_args, threshold_value, model_name, teacher_model, alpha)
        validation_loss.append(val_loss)
        validation_ce_loss.append(val_ce_loss)
        validation_soft_ce_loss.append(val_soft_ce_loss)
        validation_acc.append(val_acc)
        validation_f1.append(val_f1)
        validation_recall.append(val_recall)
        validation_precision.append(val_precision)
        
    #Save train metrics
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args["model_name"]+"/metrics/"+model_name+"_train_acc.pickle", 'wb') as f:
      pickle.dump(train_acc, f)
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args["model_name"]+"/metrics/"+model_name+"_train_f1.pickle", 'wb') as f:
      pickle.dump(train_f1, f)
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args["model_name"]+"/metrics/"+model_name+"_train_recall.pickle", 'wb') as f:
      pickle.dump(train_recall, f)
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args["model_name"]+"/metrics/"+model_name+"_train_precision.pickle", 'wb') as f:
      pickle.dump(train_precision, f)

    #Save validation metrics
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args["model_name"]+"/metrics/"+model_name+"_val_acc.pickle", 'wb') as f:
      pickle.dump(validation_acc, f)
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args["model_name"]+"/metrics/"+model_name+"_val_f1.pickle", 'wb') as f:
      pickle.dump(validation_f1, f)
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args["model_name"]+"/metrics/"+model_name+"_val_recall.pickle", 'wb') as f:
      pickle.dump(validation_recall, f)
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args["model_name"]+"/metrics/"+model_name+"_val_precision.pickle", 'wb') as f:
      pickle.dump(validation_precision, f)

    # Save final labels and predictions of model on train set 
    simple_r = {'labels':labels,'preds':preds}
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args["model_name"]+"/labels_and_preds/"+model_name+"_train.pickle", 'wb') as f:
      pickle.dump(simple_r, f)
    
    # Save training loss of GNN model
    los_p = {'loss':train_loss}
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args['model_name']+"/training_loss/training_loss_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    los_p = {'loss':train_ce_loss}
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args['model_name']+"/training_loss/training_loss_ce_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    los_p = {'loss':train_soft_ce_loss}
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args['model_name']+"/training_loss/training_loss_soft_ce_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    
    # Save validation loss of GNN model
    los_p = {'loss':validation_loss}
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args['model_name']+"/validation_loss/validation_loss_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    los_p = {'loss':validation_ce_loss}
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args['model_name']+"/training_loss/training_loss_ce_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    los_p = {'loss':validation_soft_ce_loss}
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args['model_name']+"/training_loss/training_loss_soft_ce_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    
    # Save Model
    torch.save(student_model, SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args['model_name']+"/models/"+model_args['model_name']+"_"+model_name+".pt")
    
    # Save weights
    if model_args['model_name'] == "diffpool":
        w_dict = {"w": student_model.state_dict()["assign_conv_first_modules.0.weight"]}
        with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args['model_name']+'/weights/W_'+model_name+'.pickle', 'wb') as f:
            pickle.dump(w_dict, f)
    else:
        path = SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args['model_name']+'/weights/W_'+model_name+'.pickle'
        
        if os.path.exists(path):
            os.remove(path)

        os.rename(model_args['model_name']+'_W.pickle'.format(),path)

def validate(dataset, model, model_args, threshold_value, model_name, teacher_model, alpha):
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
    total_loss = 0
    ce_loss = 0
    soft_ce_loss = 0

    criterion = nn.CrossEntropyLoss(reduction='mean')
    criterion_soft = CrossEntropyLossForSoftTarget()

    for _, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).to(device)
        labels.append(data['label'].long().numpy())
        label = Variable(data['label'].long()).to(device)        
        
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

        # Ground truth label 
        y_gt = label.to(device)
        # Compute soft label
        y_soft = teacher_model(features, adj)
        loss = criterion(ypred, y_gt) + alpha*criterion_soft(ypred, y_soft)
        
        total_loss += loss.item()
        ce_loss += criterion(ypred, y_gt).item()
        soft_ce_loss += criterion_soft(ypred, y_soft).item()

        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

    labels = np.hstack(labels)
    preds = np.hstack(preds)

    result = {
                'prec': metrics.precision_score(labels, preds),
                'recall': metrics.recall_score(labels, preds),
                'acc': metrics.accuracy_score(labels, preds),
                'F1': metrics.f1_score(labels, preds)
    }

    simple_r = {'labels':labels,'preds':preds}
    # Save labels and predictions of model on test set 
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args["model_name"]+"/labels_and_preds/"+model_name+"_val.pickle", 'wb') as f:
      pickle.dump(simple_r, f)

    val_total_loss = total_loss / len(dataset)
    val_ce_loss = ce_loss / len(dataset)
    val_soft_ce_loss = soft_ce_loss / len(dataset)
    print(f"Validation accuracy: {result['acc']}")
    print(f"Validation Loss: {val_total_loss}")

    return val_total_loss, val_ce_loss, val_soft_ce_loss, result['acc'], result['prec'], result['recall'], result['F1']

def test(dataset, model, model_args, threshold_value, model_name):
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
    total_loss = 0

    for _, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).to(device)
        labels.append(data['label'].long().numpy())
        label = Variable(data['label'].long()).to(device)        
        
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
        
        total_loss += model.loss(ypred, label).item()
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

    labels = np.hstack(labels)
    preds = np.hstack(preds)

    simple_r = {'labels':labels,'preds':preds}
    # Save labels and predictions of model on test set 
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args["model_name"]+"/labels_and_preds/"+model_name+"_test.pickle", 'wb') as f:
      pickle.dump(simple_r, f)

    print('Held-out test set loss: {}'.format(total_loss / len(dataset)))
    print("Test accuracy:", metrics.accuracy_score(labels, preds))    
