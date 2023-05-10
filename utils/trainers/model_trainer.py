import time
import torch
import pickle
import random
import numpy as np
import sklearn.metrics as metrics
from torch.autograd import Variable

from models.gcn import GCN
from models.gat import GAT
import models.diffpool as DIFFPOOL


from utils.helpers import *
from utils.config import SAVE_DIR_MODEL_DATA, SAVE_DIR_TWO_SHOT_VIEWS

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

    result = {
        'prec': metrics.precision_score(labels, preds, average='macro'),
        'recall': metrics.recall_score(labels, preds, average='macro'),
        'acc': metrics.accuracy_score(labels, preds),
        'F1': metrics.f1_score(labels, preds, average="micro")
    }
    if model_args["evaluation_method"] == 'model assessment':
        name = 'Test'
    if model_args["evaluation_method"] == 'model selection':
        name = 'Validation'
    print(name, " accuracy:", result['acc'])
    return result['acc']

def train(model_args, train_dataset, val_dataset, model, threshold_value, model_name):
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
    
    Returns
    -------
    test accuracy.
    """
    params = list(model.parameters()) 
    optimizer = torch.optim.Adam(params, lr=model_args["lr"], weight_decay=model_args['weight_decay'])
    test_accs = []
    train_loss=[]
    val_acc=[]
    

    for epoch in range(model_args["num_epochs"]):
        print("Epoch ",epoch)
        print("Size of Training Set:" + str(len(train_dataset)))
        print("Size of Validation Set:" + str(len(val_dataset)))
        model.train()
        total_time = 0
        avg_loss = 0.0
        
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
            else:
                ypred= model(features, adj)

            _, indices = torch.max(ypred, 1)
            preds.append(indices.cpu().data.numpy())
            labels.append(data['label'].long().numpy())
            loss = model.loss(ypred, label)
            
            model.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss
            elapsed = time.time() - begin_time
            total_time += elapsed
        
        # Save weights of GNN model
        if epoch==model_args["num_epochs"]-1:
              model.is_trained = True
              
        preds = np.hstack(preds)
        labels = np.hstack(labels)
        print("Train accuracy : ", np.mean( preds == labels ))
        test_acc = evaluate(val_dataset, model, model_args, threshold_value, model_name)
        val_acc.append(test_acc)
        print('Avg loss: ', avg_loss, '; epoch time: ', total_time)
        test_accs.append(test_acc)
        train_loss.append(avg_loss)
  
    # Save training loss of GNN model
    los_p = {'loss':train_loss}
    with open(SAVE_DIR_MODEL_DATA+model_args['model_name']+"/training_loss/Training_loss_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
      
    # Save Model
    torch.save(model, SAVE_DIR_MODEL_DATA+model_args['model_name']+"/models/"+model_args['model_name']+"_"+model_name+".pt")
    
    # Save weights
    if model_args['model_name'] == "diffpool":
        w_dict = {"w": model.state_dict()["assign_conv_first_modules.0.weight"]}
        with open(SAVE_DIR_MODEL_DATA+model_args['model_name']+'/weights/W_'+model_name+'.pickle', 'wb') as f:
            pickle.dump(w_dict, f)
    else:
        path = SAVE_DIR_MODEL_DATA+model_args['model_name']+'/weights/W_'+model_name+'.pickle'
        
        if os.path.exists(path):
            os.remove(path)

        os.rename(model_args['model_name']+'_W.pickle'.format(),path)

    return test_acc

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
    
    print(len(folds[0]))
    
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
    
        if model_args["model_name"]=='diffpool':
            input_dim = num_nodes
            model = DIFFPOOL.SoftPoolingGcnEncoder(
                num_nodes, 
                input_dim, model_args['hidden_dim'], model_args['output_dim'], num_classes, model_args['num_gc_layers'],
                model_args['hidden_dim'], assign_ratio=model_args['assign_ratio'], num_pooling=model_args['num_pool'],
                bn=model_args['bn'], dropout=model_args['dropout'], linkpred=model_args['linkpred'], args=model_args,
                assign_input_dim=num_nodes).to(device)     
        
        elif model_args["model_name"]=='gcn':
            model = GCN(
                nfeat = num_nodes,
                nhid = model_args["hidden_dim"],
                nclass = num_classes,
                dropout = model_args["dropout"]
            ).to(device)
        
        elif model_args["model_name"]=='gat':
            model = GAT(
                nfeat=num_nodes, 
                nhid=model_args['hidden_dim'], 
                nclass=num_classes, 
                dropout=model_args['dropout'], 
                nheads=model_args['nb_heads'], 
                alpha=model_args['alpha']).to(device)   

        test_acc = train(model_args, train_dataset, val_dataset, model, threshold_value, model_name+"_CV_"+str(i)+"_view_"+str(view))
        test_accs.append(test_acc)
    
    return test_accs

def test_scores(model_args, G_list, view, model_name, cv_number):
    
    print("Main : ", view, model_name, cv_number)
    test_accs = cv_benchmark(model_args, G_list, model_name, cv_number, view)
    print("test accuracies ", test_accs)
    
    return test_accs

def two_shot_trainer(dataset, view, num_shots, model_name, model_args):    
    start = time.time()
    num_classes = 2 #TODO: make sure it can be used with any number of classes
    print("VIEW", view)
    
    for i in range(num_shots):
        
        print("Shot : ",i)
        with open(SAVE_DIR_TWO_SHOT_VIEWS+dataset+'_view_'+str(view)+'_shot_'+str(i)+'_train','rb') as f:
            train_set = pickle.load(f)
        with open(SAVE_DIR_TWO_SHOT_VIEWS+dataset+'_view_'+str(view)+'_shot_'+str(i)+'_test','rb') as f:
            test_set = pickle.load(f)
        
        num_nodes = train_set[0]['adj'].shape[0]
        
        if model_args["model_name"]=='diffpool':
            input_dim = num_nodes
            model = DIFFPOOL.SoftPoolingGcnEncoder(
                num_nodes, 
                input_dim, model_args['hidden_dim'], model_args['output_dim'], num_classes, model_args['num_gc_layers'],
                model_args['hidden_dim'], assign_ratio=model_args['assign_ratio'], num_pooling=model_args['num_pool'],
                bn=model_args['bn'], dropout=model_args['dropout'], linkpred=model_args['linkpred'], args=model_args,
                assign_input_dim=num_nodes).to(device)     
        
        elif model_args["model_name"]=='gcn':
            model = GCN(
                nfeat = num_nodes,
                nhid = model_args["hidden_dim"],
                nclass = num_classes,
                dropout = model_args["dropout"]
            ).to(device)
        
        elif model_args["model_name"]=='gat':
            model = GAT(
                nfeat=num_nodes, 
                nhid=model_args['hidden_dim'], 
                nclass=num_classes, 
                dropout=model_args['dropout'], 
                nheads=model_args['nb_heads'], 
                alpha=model_args['alpha']).to(device)  
        
        train_dataset, val_dataset, threshold_value = two_shot_loader(train_set, test_set, model_args)
        
        test_acc = train(model_args, train_dataset, val_dataset, model, threshold_value, model_name+str(i)+"_view_"+str(view))
        
        print("Test accuracy:"+str(test_acc))
        print('Time taken', time.time()-start)
