
import time
import pickle
import random
import shutil 
import psutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sklearn.metrics as metrics

from models.gcn_student import GCN_STUDENT
from models.gat.gat_student import GAT_STUDENT
from utils.helpers import *
from utils.config import SAVE_DIR_MODEL_DATA

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

class CrossEntropyLossForSoftTarget(nn.Module):
    """
    Initializes the CrossEntropyLossForSoftTarget module.
    
    Parameters:
      T (float): Temperature parameter for softening the labels. Default is 3.
      alpha (float): Weight parameter for adjusting the loss. Default is 2.
    """
    def __init__(self, T=3, alpha=2):
        super(CrossEntropyLossForSoftTarget, self).__init__()
        self.T = T
        self.alpha = alpha
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
    def forward(self, y_pred, y_gt):
        """
        Computes the forward pass of the loss function.
        
        Parameters:
          y_pred (tensor): Predicted logits from the model.
          y_gt (tensor): Ground truth labels (softened).
        
        Returns:
          loss (tensor): Computed loss value.
        """
        y_pred_soft = y_pred.div(self.T)
        y_gt_soft = y_gt.div(self.T)
        return -(self.softmax(y_gt_soft)*self.logsoftmax(y_pred_soft)).mean().mul(self.alpha)
    
def cross_validation(model_args, G_list, view, model_name, cv_number, run=0):
    """
    Perform cross-validation training and evaluation of the model.
    
    Parameters:
        model_args (dict): Model configuration parameters.
        G_list (list): List of graph data for different views.
        view (int): View index for the current cross-validation.
        model_name (str): Name of the model being trained.
        cv_number (int): Number of cross-validation folds.
        run (int, optional): Run number. Default is 0.
    """
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
        #add hyperparameters here     
        name = model_name+"_CV_"+str(cv)+"_view_"+str(view)+"_with_teacher"
        
        print(name)

        train_set, validation_set, test_set = datasets_splits(folds, model_args, cv)
        num_nodes = G_list[0]['adj'].shape[0]
        num_classes = 2 

        if model_args["model_name"] == "gcn_student":
          student_model = GCN_STUDENT(
              nfeat = num_nodes,
              nhid = model_args["hidden_dim"],
              nclass = num_classes,
              dropout = model_args["dropout"],
              run = run,
              dataset = model_args["dataset"]
          ).to(device) 

        elif model_args["model_name"] == "gat_student":
          student_model = GAT_STUDENT(
              nfeat=num_nodes, 
              nhid=model_args['hidden_dim'], 
              nclass=num_classes, 
              dropout=model_args['dropout'], 
              nheads=model_args['nb_heads'], 
              alpha=model_args['alpha'],
              run = run,
              dataset = model_args["dataset"]
          ).to(device)  
          print(student_model)

        if model_args["evaluation_method"] =='model_selection':
            #Here we leave out the test set since we are not evaluating we can see the performance on the test set after training
            train(model_args, train_dataset, val_dataset, student_model, threshold_value, name,  cv, view, cv_number, run)
            #See performance on the held-out test set 
            dataset_sampler = GraphSampler(test_set)
            test_dataset = torch.utils.data.DataLoader(
                dataset_sampler, 
                batch_size = 1,  
                shuffle = False) 
            test(test_dataset, student_model, model_args, threshold_value)

        if model_args["evaluation_method"] =='model_assessment':
            #Here we join the train and validation dataset
            train(model_args, train_dataset, val_dataset, student_model, threshold_value, name,  cv, view, cv_number, run)
    
    print('Time taken', time.time()-start)

def train(model_args, train_dataset, val_dataset, student_model, threshold_value, model_name, cv, view, cv_number, run=0, alpha=2):
    """
    Train the model on the training dataset and evaluate on the validation dataset.
    
    Parameters:
        model_args (dict): Model configuration parameters.
        train_dataset (dataloader): Dataloader for the training dataset.
        val_dataset (dataloader): Dataloader for the validation dataset.
        student_model (nn.Module): Student graph neural network model to be trained.
        threshold_value (float): Threshold value for adjacency matrices.
        model_name (str): Name of the model being trained.
        cv (int): Cross-validation fold number.
        view (int): View index.
        cv_number (int): Number of cross-validation folds.
        run (int, optional): Run number. Default is 0.
        alpha (int, optional): Weight for the loss. Default is 2.
    """
    # Load teacher model
    teacher_model = torch.load(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+f"/{model_args['backbone']}/models/{model_args['backbone']}_MainModel_{cv_number}Fold_{model_args['dataset']}_{model_args['backbone']}_run_{run}_fixed_init_CV_{cv}_view_{view}.pt")
    teacher_model.is_trained = False
    teacher_model.eval()

    # Transfer
    teacher_model.to(device)
    student_model.to(device)

    # Define Loss
    criterion = nn.CrossEntropyLoss(reduction='mean')
    criterion_soft = CrossEntropyLossForSoftTarget(T=model_args["T"], alpha=model_args["alpha_soft_ce"])

    # Define optimizer
    optimizer = optim.Adam(student_model.parameters(), lr=model_args["lr"], weight_decay=model_args['weight_decay'])
    
    # Metrics 
    train_loss=[]
    train_acc=[]
    train_ce_loss=[]
    train_soft_ce_loss=[]
    train_weight_loss=[]
    train_f1=[]
    train_recall=[]
    train_precision=[]

    validation_loss=[]
    validation_ce_loss=[]
    validation_soft_ce_loss=[]
    validation_weight_loss=[]
    validation_acc=[]
    validation_f1=[]
    validation_recall=[]
    validation_precision=[]

    time_per_epoch = []
    memory_usage_per_epoch = []    
    
    print(f"Size of Training Set: {str(len(train_dataset))}")
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
            
            # Ground truth label
            y_gt = label.to(device)

            # Compute soft label
            y_soft, _ = teacher_model(features, adj)
            
            # Predict
            ypred, _ = student_model(features, adj)
            # Compute loss (foward propagation)
            loss_ce = criterion(ypred, y_gt)
            loss_soft = criterion_soft(ypred, y_soft)
            loss = model_args["alpha_ce"]*loss_ce + criterion_soft(ypred, y_soft)
            #Save pred
            _, indices = torch.max(ypred, 1)
            preds.append(indices.cpu().data.numpy())
            labels.append(data['label'].long().numpy())
            
            # Compute gradients (backward propagation)
            loss.backward()
            
            # Update parameters
            optimizer.step()
                    
            total_loss += loss.item()
            ce_loss += loss_ce.item()
            soft_ce_loss += loss_soft.item()

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
        print(f"Train Soft CE loss: {soft_ce_loss / len(train_dataset)}")
        print(f"Train CE loss: {ce_loss / len(train_dataset)}")
 
        train_loss.append(total_loss / len(train_dataset))
        train_ce_loss.append(ce_loss / len(train_dataset))
        train_soft_ce_loss.append(soft_ce_loss / len(train_dataset))
        train_acc.append(result['acc'])
        train_f1.append(result['F1'])
        train_recall.append(result['recall'])
        train_precision.append(result['prec'])
        
        val_loss, val_ce_loss, val_soft_ce_loss, val_acc, val_precision, val_recall, val_f1 = validate(val_dataset, student_model, model_args, threshold_value, model_name, teacher_model)
        validation_loss.append(val_loss)
        validation_ce_loss.append(val_ce_loss)
        validation_soft_ce_loss.append(val_soft_ce_loss)
        validation_acc.append(val_acc)
        validation_f1.append(val_f1)
        validation_recall.append(val_recall)
        validation_precision.append(val_precision)
        time_per_epoch.append(total_time)
        process = psutil.Process()
        memory_usage_per_epoch.append(process.memory_info().rss / 1024 ** 2)
      
    print(f"Average Memory Usage: {np.mean(memory_usage_per_epoch)} MB, Std: {np.std(memory_usage_per_epoch)}")
    print(f"Average Time: {np.mean(time_per_epoch)}, Std: {np.std(time_per_epoch)}")

    #Save train metrics
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args["model_name"]+"/metrics/"+model_name+"_train_acc.pickle", 'wb') as f:
      pickle.dump(train_acc, f)
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args["model_name"]+"/metrics/"+model_name+"_train_f1.pickle", 'wb') as f:
      pickle.dump(train_f1, f)
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args["model_name"]+"/metrics/"+model_name+"_train_recall.pickle", 'wb') as f:
      pickle.dump(train_recall, f)
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args["model_name"]+"/metrics/"+model_name+"_train_precision.pickle", 'wb') as f:
      pickle.dump(train_precision, f)

    #Save validation metrics
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args["model_name"]+"/metrics/"+model_name+"_val_acc.pickle", 'wb') as f:
      pickle.dump(validation_acc, f)
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args["model_name"]+"/metrics/"+model_name+"_val_f1.pickle", 'wb') as f:
      pickle.dump(validation_f1, f)
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args["model_name"]+"/metrics/"+model_name+"_val_recall.pickle", 'wb') as f:
      pickle.dump(validation_recall, f)
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args["model_name"]+"/metrics/"+model_name+"_val_precision.pickle", 'wb') as f:
      pickle.dump(validation_precision, f)

    # Save final labels and predictions of model on train set 
    simple_r = {'labels':labels,'preds':preds}
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args["model_name"]+"/labels_and_preds/"+model_name+"_train.pickle", 'wb') as f:
      pickle.dump(simple_r, f)
    
    # Save training loss of GNN model
    los_p = {'loss':train_loss}
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args['model_name']+"/training_loss/training_loss_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    los_p = {'loss':train_ce_loss}
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args['model_name']+"/training_loss/training_loss_ce_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    los_p = {'loss':train_soft_ce_loss}
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args['model_name']+"/training_loss/training_loss_soft_ce_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    los_p = {'loss':train_weight_loss}
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args['model_name']+"/training_loss/training_loss_weight_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    
    # Save validation loss of GNN model
    los_p = {'loss':validation_loss}
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args['model_name']+"/validation_loss/validation_loss_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    los_p = {'loss':validation_ce_loss}
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args['model_name']+"/validation_loss/validation_loss_ce_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    los_p = {'loss':validation_soft_ce_loss}
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args['model_name']+"/validation_loss/validation_loss_soft_ce_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    los_p = {'loss':validation_weight_loss}
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args['model_name']+"/validation_loss/validation_loss_weight_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    
    # Save Model
    torch.save(student_model, SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args['model_name']+"/models/"+model_args['model_name']+"_"+model_name+".pt")
    
    # Save weights
    path = SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args['model_name']+'/weights/W_'+model_name+'.pickle'
    if os.path.exists(path):
       os.remove(path)

    shutil.move(model_args['model_name']+"_"+str(run)+"_"+str(model_args['dataset'])+'_W.pickle', path)

def validate(dataset, model, model_args, threshold_value, model_name, teacher_model):
    """
    Evaluate the model on the validation dataset.
    
    Parameters:
        dataset (dataloader): Dataloader for the validation/test dataset.
        model (nn.Module): Graph convolutional network (GCN) model.
        model_args (dict): Arguments for configuring the model.
        threshold_value (float): Threshold value for adjacency matrices.
        model_name (str): Name of the model being evaluated.
        teacher_model (nn.Module): Teacher model for knowledge distillation.
    
    Returns:
        val_total_loss (float): Validation total loss.
        val_ce_loss (float): Validation cross-entropy loss.
        val_soft_ce_loss (float): Validation soft cross-entropy loss.
        val_acc (float): Validation accuracy.
        val_prec (float): Validation precision.
        val_recall (float): Validation recall.
        val_f1 (float): Validation F1 score.
    """
    model.eval()
    labels = []
    preds = []
    total_loss = 0
    ce_loss = 0
    soft_ce_loss = 0

    # Define Loss
    criterion = nn.CrossEntropyLoss(reduction='mean')
    criterion_soft = CrossEntropyLossForSoftTarget(T=model_args["T"], alpha=model_args["alpha_soft_ce"])

    for _, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).to(device)
        labels.append(data['label'].long().numpy())
        label = Variable(data['label'].long()).to(device)        
        
        adj = torch.squeeze(adj)
        
        features = np.identity(adj.shape[0])
        
        features = Variable(torch.from_numpy(features).float(), requires_grad=False).to(device)
        
        if model_args["threshold"] in ["median", "mean"]:
            adj = torch.where(adj > threshold_value, torch.tensor([1.0]).to(device), torch.tensor([0.0]).to(device))
        
        # Ground truth label 
        y_gt = label.to(device)
        # Compute soft label
        y_soft , _ = teacher_model(features, adj)

        # Predict
        ypred, _ = model(features, adj)
        # Compute loss (foward propagation)
        loss_ce = criterion(ypred, y_gt)
        loss_soft = criterion_soft(ypred, y_soft)
        loss = model_args["alpha_ce"]*loss_ce + criterion_soft(ypred, y_soft)
        #Save pred
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

        total_loss += loss.item()
        ce_loss += loss_ce.item()
        soft_ce_loss += loss_soft.item()

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
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args["model_name"]+"/labels_and_preds/"+model_name+"_val.pickle", 'wb') as f:
      pickle.dump(simple_r, f)

    val_total_loss = total_loss / len(dataset)
    val_ce_loss = ce_loss / len(dataset)
    val_soft_ce_loss = soft_ce_loss / len(dataset)
    print(f"Validation accuracy: {result['acc']}")
    print(f"Validation Loss: {val_total_loss}")

    return val_total_loss, val_ce_loss, val_soft_ce_loss, result['acc'], result['prec'], result['recall'], result['F1']

def test(dataset, model, model_args, threshold_value):
    """
    Evaluate the model on the test/validation dataset.
    
    Parameters:
        dataset (dataloader): Dataloader for the validation/test dataset.
        model (nn.Module): Graph convolutional network (GCN) model.
        model_args (dict): Arguments for configuring the model.
        threshold_value (float): Threshold value for adjacency matrices.
    """
    model.eval()
    labels = []
    preds = []

    for _, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).to(device)
        labels.append(data['label'].long().numpy())
        label = Variable(data['label'].long()).to(device)        
        
        adj = torch.squeeze(adj)
        
        features = np.identity(adj.shape[0])
        
        features = Variable(torch.from_numpy(features).float(), requires_grad=False).to(device)
        
        if model_args["threshold"] in ["median", "mean"]:
            adj = torch.where(adj > threshold_value, torch.tensor([1.0]).to(device), torch.tensor([0.0]).to(device))
        
        # Ground truth label 
        y_gt = label.to(device)
        # Predict
        ypred, _ = model(features, adj)
        #Save pred
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
    print(f"Test accuracy: {result['acc']}")
