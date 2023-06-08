import time
import torch
import pickle
import random
import shutil 
import numpy as np
import sklearn.metrics as metrics
from torch.autograd import Variable
import torch.nn.functional as F

from models.gcn import GCN
from models.gcn_3_layers import GCN3
from models.gcn_4_layers import GCN4
from models.gcn_student import GCN_STUDENT
from models.mlp import MLP
from models.gat import GAT
import models.diffpool as DIFFPOOL

from utils.helpers import *
from config import SAVE_DIR_MODEL_DATA

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def cross_validation(model_args, G_list, view, model_name, cv_number, run=0):
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
    
        if model_args["model_name"]=='diffpool':
            input_dim = num_nodes
            model = DIFFPOOL.SoftPoolingGcnEncoder(
                num_nodes, 
                input_dim, model_args['hidden_dim'], model_args['output_dim'], num_classes, model_args['num_gc_layers'],
                model_args['hidden_dim'], assign_ratio=model_args['assign_ratio'], num_pooling=model_args['num_pool'],
                bn=model_args['bn'], dropout=model_args['dropout'], linkpred=model_args['linkpred'], args=model_args,
                assign_input_dim=num_nodes).to(device)     
        
        elif model_args["model_name"]=='gcn' and model_args["layers"]==2:
            model = GCN(
                nfeat = num_nodes,
                nhid = model_args["hidden_dim"],
                nclass = num_classes,
                dropout = model_args["dropout"],
                run = run
            ).to(device)
        
        elif model_args["model_name"]=='gcn'and model_args["layers"]==3:
            model = GCN3(
                nfeat = num_nodes,
                nhid = model_args["hidden_dim"],
                nclass = num_classes,
                dropout = model_args["dropout"],
                run = run
            ).to(device)
        
        elif model_args["model_name"]=='gcn'and model_args["layers"]==4:
            model = GCN4(
                nfeat = num_nodes,
                nhid = model_args["hidden_dim"],
                nclass = num_classes,
                dropout = model_args["dropout"],
                run = run
            ).to(device)

        elif model_args["model_name"]=='gat':
            model = GAT(
                nfeat=num_nodes, 
                nhid=model_args['hidden_dim'], 
                nclass=num_classes, 
                dropout=model_args['dropout'], 
                nheads=model_args['nb_heads'], 
                alpha=model_args['alpha']).to(device)   
        
        elif model_args["model_name"] == "gcn_student":
            model = GCN_STUDENT(
                nfeat = num_nodes,
                nhid = model_args["hidden_dim"],
                nclass = num_classes,
                dropout = model_args["dropout"],
                run = run
            ).to(device) 
        
        elif model_args["model_name"] == "mlp":
            model = MLP(
                num_layers=model_args["num_layers"], 
                input_dim=num_nodes, 
                hidden_dim=model_args["hidden_dim"], 
                output_dim=2, 
                dropout_ratio=model_args["dropout_ratio"]
                )
        if model_args["evaluation_method"] =='model_selection': 
            train(model_args, train_dataset, val_dataset, model, threshold_value, model_name+"_CV_"+str(i)+"_view_"+str(view))
            #See performance on the held-out test set 
            dataset_sampler = GraphSampler(test_set)
            test_dataset = torch.utils.data.DataLoader(
                dataset_sampler, 
                batch_size = 1,  
                shuffle = False) 
            #add hyperparameters here     
            test(test_dataset, model, model_args, threshold_value, model_name+"_CV_"+str(i)+"_view_"+str(view))
            
        if model_args["evaluation_method"] =='model_assessment': 
          #add hyperparameters here     
          train(model_args, train_dataset, val_dataset, model, threshold_value, model_name+"_CV_"+str(i)+"_view_"+str(view), run)
    
    print('Time taken', time.time()-start)

def train(model_args, train_dataset, val_dataset, model, threshold_value, model_name, run):
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
              features = torch.mean(features, axis=1)
              ypred = model(features)[1]
          else:
              ypred= model(features, adj)

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
    
    # Save validation loss of GNN model
    los_p = {'loss':validation_loss}
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args['model_name']+"/validation_loss/validation_loss_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    
    # Save Model
    torch.save(model, SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args['model_name']+"/models/"+model_args['model_name']+"_"+model_name+".pt")
    
    # Save weights
    if model_args['model_name'] == "diffpool":
        w_dict = {"w": model.state_dict()["assign_conv_first_modules.0.weight"]}
        with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args['model_name']+'/weights/W_'+model_name+'.pickle', 'wb') as f:
            pickle.dump(w_dict, f)
    else:
        path = SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args['model_name']+'/weights/W_'+model_name+'.pickle'
        
        if os.path.exists(path):
            os.remove(path)

        shutil.move(model_args['model_name']+"_"+str(run)+'_W.pickle', path)

def validate(dataset, model, model_args, threshold_value, model_name):
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

    val_loss = total_loss / len(dataset)
    print(f"Validation accuracy: {result['acc']}")
    print(f"Validation Loss: {val_loss}")

    return val_loss, result['acc'], result['prec'], result['recall'], result['F1']

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
