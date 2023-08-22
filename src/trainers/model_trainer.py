import time
import torch
import pickle
import random
import shutil 
import psutil
import numpy as np
import sklearn.metrics as metrics
from torch.autograd import Variable
import torch.nn.functional as F

from models.gcn import GCN
from models.gcn_3_layers import GCN3
from models.gcn_student import GCN_STUDENT

from models.gat.gat import GAT
from models.gat.gat_student import GAT_STUDENT

from utils.helpers import *
from utils.config import SAVE_DIR_MODEL_DATA

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def cross_validation(model_args, G_list, view, model_name, cv_number, run=0):
    """
    Performs cross-validation training and evaluation of the model.
    
    Parameters:
        model_args (dict): Model configuration parameters.
        G_list (list): List of graph data for different views.
        view (int): View index for the current cross-validation.
        model_name (str): Name of the model being trained.
        cv_number (int): Number of cross-validation folds.
        run (int, optional): Run number. Default is 0.
    """
    start = time.time() 
    print("Run :", run)
    print("--------------------------------------------------------------------------")
    print("Main : ", view, model_name, cv_number)

    folds = stratify_splits(G_list, cv_number)
    
    [random.shuffle(folds[i]) for i in range(len(folds))]
    
    for i in range(cv_number):
        
        print("CV : ",i)
        
        train_set, validation_set, test_set = datasets_splits(folds, model_args, i)
        
        if model_args["evaluation_method"] =='model_selection':
            train_dataset, val_dataset, threshold_value = model_selection_split(train_set, validation_set, model_args)
        
        if model_args["evaluation_method"] =='model_assessment':
            train_dataset, val_dataset, threshold_value = model_assessment_split(train_set, validation_set, test_set, model_args)
         
        num_nodes = G_list[0]['adj'].shape[0]
        num_classes = 2 #TODO: make sure it can be used with any number of classes
    
        if model_args["model_name"]=='gcn' and model_args["layers"]==2:
            model = GCN(
                nfeat = num_nodes,
                nhid = model_args["hidden_dim"],
                nclass = num_classes,
                dropout = model_args["dropout"],
                run = run,
                dataset = model_args["dataset"]
            ).to(device)
        
        elif model_args["model_name"]=='gcn'and model_args["layers"]==3:
            model = GCN3(
                nfeat = num_nodes,
                nhid = model_args["hidden_dim"],
                nclass = num_classes,
                dropout = model_args["dropout"],
                run = run,
                dataset = model_args["dataset"]
            ).to(device)
        
        elif model_args["model_name"]=='gat':
            model = GAT(
                nfeat=num_nodes, 
                nhid=model_args['hidden_dim'], 
                nclass=num_classes, 
                dropout=model_args['dropout'], 
                nheads=model_args['nb_heads'], 
                alpha=model_args['alpha'],
                run = run,
                dataset = model_args["dataset"]
            ).to(device)   
        
        elif model_args["model_name"] == "gcn_student":
            model = GCN_STUDENT(
                nfeat = num_nodes,
                nhid = model_args["hidden_dim"],
                nclass = num_classes,
                dropout = model_args["dropout"],
                run = run,
                dataset = model_args["dataset"]
            ).to(device) 
        
        elif model_args["model_name"]=='gat_student':
            model = GAT_STUDENT(
                nfeat=num_nodes, 
                nhid=model_args['hidden_dim'], 
                nclass=num_classes, 
                dropout=model_args['dropout'], 
                nheads=model_args['nb_heads'], 
                alpha=model_args['alpha'],
                run = run,
                dataset = model_args["dataset"]
            ).to(device)   

        if model_args["evaluation_method"] =='model_selection': 
            train(model_args, train_dataset, val_dataset, model, threshold_value, model_name+"_CV_"+str(i)+"_view_"+str(view), run)
            #See performance on the held-out test set 
            dataset_sampler = GraphSampler(test_set)
            test_dataset = torch.utils.data.DataLoader(
                dataset_sampler, 
                batch_size = 1,  
                shuffle = False)   
            test(test_dataset, model, model_args, threshold_value)
            
        if model_args["evaluation_method"] =='model_assessment': 
          #add hyperparameters here     
          train(model_args, train_dataset, val_dataset, model, threshold_value, model_name+"_CV_"+str(i)+"_view_"+str(view), run)
    
    print('Time taken', time.time()-start)

def train(model_args, train_dataset, val_dataset, model, threshold_value, model_name, run):
    """
    Trains the graph neural network model and evaluates on the validation data.
    
    Parameters:
        model_args (dict): Model configuration parameters.
        train_dataset (dataloader): Dataloader for the training dataset.
        val_dataset (dataloader): Dataloader for the validation dataset.
        model (nn.Module): Graph neural network model to be trained.
        threshold_value (float): Threshold value for adjacency matrices.
        model_name (str): Name of the model being trained.
        run (int): Run number.
    """
    params = list(model.parameters()) 
    optimizer = torch.optim.Adam(params, lr=model_args["lr"], weight_decay=model_args['weight_decay'])
    
    # Metrics 
    train_loss=[]
    train_acc=[]
    train_f1=[]
    train_recall=[]
    train_precision=[]

    validation_loss=[]
    validation_acc=[]
    validation_f1=[]
    validation_recall=[]
    validation_precision=[]

    time_per_epoch = []
    memory_usage_per_epoch = []
    
    print(f"Size of Training Set: {str(len(train_dataset))}")
    print(f"Size of Validation Set: {str(len(val_dataset))}")
    
    for epoch in range(model_args["num_epochs"]):
      model.train()
      total_time = 0
      total_loss = 0
      
      preds = []
      labels = []

      for batch_idx, data in enumerate(train_dataset):
          begin_time = time.time()
          adj = Variable(data['adj'].float(), requires_grad=False).to(device)
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
              features = torch.mean(adj, axis=1)
              logits = model(features)[1]
              ypred = torch.sigmoid(logits)[0]
          else:
              ypred, _ = model(features, adj)
          
          if model_args["model_name"] == "mlp":
            pred_label = 1 if ypred >= 0.5 else 0
            preds.append(np.array(pred_label))
            labels.append(data['label'].long().numpy())
          else:
            _, indices = torch.max(ypred, 1)
            preds.append(indices.cpu().data.numpy())
            labels.append(data['label'].long().numpy())        

          loss = model.loss(ypred, label)
          
          model.zero_grad()
          loss.backward()
          optimizer.step()
          total_loss += loss
          elapsed = time.time() - begin_time
          total_time += elapsed
      
      # Save weights of GNN model
      if epoch==model_args["num_epochs"]-1:
            model.is_trained = True
      
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
      train_acc.append(result['acc'])
      train_f1.append(result['F1'])
      train_recall.append(result['recall'])
      train_precision.append(result['prec'])
      
      val_loss, val_acc, val_precision, val_recall, val_f1 = validate(val_dataset, model, model_args, threshold_value, model_name)
      validation_acc.append(val_acc)
      validation_f1.append(val_f1)
      validation_recall.append(val_recall)
      validation_precision.append(val_precision)
      
      validation_loss.append(val_loss)

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
    
    # Save validation loss of GNN model
    los_p = {'loss':validation_loss}
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args['model_name']+"/validation_loss/validation_loss_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    
    # Save Model
    torch.save(model, SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args['model_name']+"/models/"+model_args['model_name']+"_"+model_name+".pt")
    
    # Save weights
    if model_args['model_name'] == "diffpool":
        w_dict = {"w": model.state_dict()["assign_conv_first_modules.0.weight"]}
        with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args['model_name']+'/weights/W_'+model_name+'.pickle', 'wb') as f:
            pickle.dump(w_dict, f)
    else:
        path = SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args['model_name']+'/weights/W_'+model_name+'.pickle'
        
        if os.path.exists(path):
            os.remove(path)
        shutil.move(model_args['model_name']+"_"+str(run)+'_'+str(model_args["dataset"])+'_W.pickle', path)

def validate(dataset, model, model_args, threshold_value, model_name):
    """
    Evaluates the graph neural network model on the validation data.
    
    Parameters:
        dataset (dataloader): Dataloader for the validation dataset.
        model (nn.Module): Graph neural network model to be evaluated.
        model_args (dict): Model configuration parameters.
        threshold_value (float): Threshold value for adjacency matrices.
        model_name (str): Name of the model being evaluated.
    
    Returns:
        val_loss (float): Validation loss.
        val_acc (float): Validation accuracy.
        val_precision (float): Validation precision.
        val_recall (float): Validation recall.
        val_f1 (float): Validation F1 score.
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
            features = torch.mean(adj, axis=1)
            logits = model(features)[1]
            ypred = torch.sigmoid(logits)[0]
        else:
            ypred, _ = model(features, adj)
        
        total_loss += model.loss(ypred, label).item()
    
        if model_args["model_name"] == "mlp":
            pred_label = 1 if  ypred >= 0.5 else 0
            preds.append(np.array(pred_label))
        else:
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
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args["model_name"]+"/labels_and_preds/"+model_name+"_val.pickle", 'wb') as f:
      pickle.dump(simple_r, f)

    val_loss = total_loss / len(dataset)
    print(f"Validation accuracy: {result['acc']}")
    print(f"Validation Loss: {val_loss}")

    return val_loss, result['acc'], result['prec'], result['recall'], result['F1']

def test(dataset, model, model_args, threshold_value):
    """
    Evaluates the graph neural network model on the test data.
    
    Parameters:
        dataset (dataloader): Dataloader for the test dataset.
        model (nn.Module): Graph neural network model to be evaluated.
        model_args (dict): Model configuration parameters.
        threshold_value (float): Threshold value for adjacency matrices.
    
    Returns:
        None
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
            features = torch.mean(adj, axis=1)
            logits = model(features)[1]
            ypred = torch.sigmoid(logits)[0]
        else:
            ypred, _ = model(features, adj)
        
        total_loss += model.loss(ypred, label).item()
    
        if model_args["model_name"] == "mlp":
            pred_label = 1 if  ypred >= 0.5 else 0
            preds.append(np.array(pred_label))
        else:
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

    test_loss = total_loss / len(dataset)
    print(f"Test accuracy: {result['acc']}")
    print(f"Test Loss: {test_loss}")
