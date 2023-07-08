import time
import torch
import pickle
import random
import shutil 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sklearn.metrics as metrics

from models.gcn.gcn_student_ensamble import GCN_STUDENT_ENSAMBLE
from models.gat.gat_student_ensamble import GAT_STUDENT_ENSAMBLE
from models.model_config import * 
from utils.helpers import *
from utils.config import SAVE_DIR_MODEL_DATA

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

class CrossEntropyLossForSoftTarget(nn.Module):
    def __init__(self, T=3):
        super(CrossEntropyLossForSoftTarget, self).__init__()
        self.T = T
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
    def forward(self, y_pred, y_gt):
        y_pred_soft = y_pred.div(self.T)
        y_gt_soft = y_gt.div(self.T)
        return -(self.softmax(y_gt_soft)*self.logsoftmax(y_pred_soft)).mean().mul(self.T^2)
    
def weight_similarity_loss(w_teacher, w_student):
    """
    Compute the KL loss between the weights of the last layer
    of two networks.
    """    
    # Concatenate and compute the cosine similarity
    loss = nn.CosineSimilarity()
    return loss(w_student, w_teacher).abs()

def cross_validation_5(model_args, G_list, view, model_name, cv_number, n_students, run=0):
    start = time.time() 
    print("Run : ",run)
    print("--------------------------------------------------------------------------")
    print("Main : ", view, model_name, cv_number)
    
    folds = stratify_splits(G_list, cv_number)
    
    [random.shuffle(folds[i]) for i in range(len(folds))]
    
    for cv in range(cv_number):
        
        student_names = []
        students = []
        train_set, validation_set, test_set = datasets_splits(folds, model_args, cv)
        
        if model_args["evaluation_method"] =='model_selection':
            train_dataset, val_dataset, threshold_value = model_selection_split(train_set, validation_set, model_args)
        if model_args["evaluation_method"] =='model_assessment':
            train_dataset, val_dataset, threshold_value = model_assessment_split(train_set, validation_set, test_set, model_args)
        print(f"CV : {cv}")

        alpha = str(model_args["alpha"])
        beta = str(model_args["beta"])
        gamma = str(model_args["gamma"])
        lambda_ = str(model_args["lambda"])

        for i in range(n_students):
            #add hyperparameters here    
            name = model_name+f"_student_{str(i)}_CV_{str(cv)}_view_{str(view)}_alpha_{alpha}_beta_{beta}_gamma_{gamma}_lambda_{lambda_}"
            student_names.append(name)
        #add hyperparameters here     
        name = model_name+f"_CV_{str(cv)}_view_{str(view)}_alpha_{alpha}_beta_{beta}_gamma_{gamma}_lambda_{lambda_}"
        num_nodes = G_list[0]['adj'].shape[0]
        num_classes = 2 
        
        for i in range(n_students):
            if model_args["backbone"] == "gcn":
              student_model = GCN_STUDENT_ENSAMBLE(
                  nfeat = num_nodes,
                  nhid = model_args["hidden_dim"],
                  nclass = num_classes,
                  dropout = model_args["dropout"],
                  seed = i,
                  run = run, 
                  number = i,
                  total_number = model_args["n_students"],
                  dataset = model_args["dataset"]
              ).to(device)
            else:
              student_model = GAT_STUDENT_ENSAMBLE(
                  nfeat=num_nodes, 
                  nhid=model_args['hidden_dim'], 
                  nclass=num_classes, 
                  dropout=model_args['dropout'], 
                  nheads=model_args['nb_heads'], 
                  alpha=model_args['alpha'],
                  seed = i,
                  run = run, 
                  number = i,
                  total_number = model_args["n_students"],
                  dataset = model_args["dataset"]
              ).to(device)  
            students.append(student_model)

        train_set, validation_set, test_set = datasets_splits(folds, model_args, cv)

        if model_args["evaluation_method"] =='model_selection':
            #Here we leave out the test set since we are not evaluating we can see the performance on the test set after training
            train(model_args, train_dataset, val_dataset, students, student_names, threshold_value, name,  cv, view, cv_number, run)
            #See performance on the held-out test set 
            dataset_sampler = GraphSampler(test_set)
            test_dataset = torch.utils.data.DataLoader(
                dataset_sampler, 
                batch_size = 1,  
                shuffle = False) 
            test(test_dataset, students, model_args, threshold_value, name)

        if model_args["evaluation_method"] =='model_assessment':
            #Here we join the train and validation dataset
            train(model_args, train_dataset, val_dataset, students, student_names, threshold_value, name,  cv, view, cv_number, run)
    
    print('Time taken', time.time()-start)

def train(model_args, train_dataset, val_dataset, students, student_names, threshold_value, model_name, cv, view, cv_number, run=0):
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
    teacher_model = torch.load(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+f"/{model_args['backbone']}/models/{model_args['backbone']}_MainModel_{cv_number}Fold_gender_data_{model_args['backbone']}_run_{run}_fixed_init_CV_{cv}_view_{view}.pt")
    teacher_model.is_trained = False
    teacher_model.eval()

    # Transfer
    teacher_model.to(device)
    student_model_1 = students[0].to(device)
    student_model_2 = students[1].to(device)
    student_model_3 = students[2].to(device)
    student_model_4 = students[3].to(device)
    student_model_5 = students[4].to(device)

    # Define Loss
    criterion = nn.CrossEntropyLoss(reduction='mean')
    criterion_soft = CrossEntropyLossForSoftTarget(T=model_args["T"])

    # Define optimizer
    optimizer_1 = optim.Adam(student_model_1.parameters(), lr=model_args["lr"], weight_decay=model_args['weight_decay'])
    optimizer_2 = optim.Adam(student_model_2.parameters(), lr=model_args["lr"], weight_decay=model_args['weight_decay'])
    optimizer_3 = optim.Adam(student_model_3.parameters(), lr=model_args["lr"], weight_decay=model_args['weight_decay'])
    optimizer_4 = optim.Adam(student_model_4.parameters(), lr=model_args["lr"], weight_decay=model_args['weight_decay'])
    optimizer_5 = optim.Adam(student_model_5.parameters(), lr=model_args["lr"], weight_decay=model_args['weight_decay'])
    
    # Metrics 
    # total train loss 
    total_train_loss = []
    # cumulative loss of teacher and student soft-ce
    train_loss_teacher_student = []
    # ensamble loss
    train_loss_ensamble_ce = []
    # soft ensamble loss
    train_ensamble_soft_ce_loss=[]
    # loss of weights within the students 
    train_loss_within_student=[]

    # Validation losses
    validation_total_loss = []
    validation_loss_teacher_student = []
    validation_loss_ensamble_ce = []
    validation_ensamble_soft_ce_loss = []
    validation_loss_within_student = []
    
    print(f"Size of Training Set: {str(len(train_dataset))}")
    print(f"Size of Validation Set: {str(len(val_dataset))}")
    
    for epoch in range(model_args["num_epochs"]):
        
        student_model_1.train()
        student_model_2.train()
        student_model_3.train()
        student_model_4.train()
        student_model_5.train()

        total_time = 0
        total_loss = 0
        t_loss_teacher_student = 0
        t_loss_ensamble_ce = 0
        t_ensamble_soft_ce_loss = 0
        t_loss_within_student = 0
        
        preds_ensamble, labels_ensamble  = [], []
        preds_1, labels_1 = [], []
        preds_2, labels_2 = [], []
        preds_3, labels_3 = [], []
        preds_4, labels_4 = [], []
        preds_5, labels_5 = [], []

        for _, data in enumerate(train_dataset):
            begin_time = time.time()
            
            # Initialize gradients
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            optimizer_3.zero_grad()
            optimizer_4.zero_grad()
            optimizer_5.zero_grad()

            # Transfer device
            adj = Variable(data['adj'].float(), requires_grad=False).to(device)
            adj = torch.squeeze(adj)
            
            label = Variable(data['label'].long()).to(device)     
            
            features = np.identity(adj.shape[0])
            features = Variable(torch.from_numpy(features).float(), requires_grad=False).to(device)       
            
            if model_args["threshold"] in ["median", "mean"]:
                adj = torch.where(adj > threshold_value, torch.tensor([1.0]).to(device), torch.tensor([0.0]).to(device))
            
            # extract student weight
            student_weights_1 = student_model_1.LinearLayer.weight
            student_weights_2 = student_model_2.LinearLayer.weight
            student_weights_3 = student_model_3.LinearLayer.weight
            student_weights_4 = student_model_4.LinearLayer.weight
            student_weights_5 = student_model_5.LinearLayer.weight

            y_gt = label.to(device)

            # Compute soft label
            y_soft, _ = teacher_model(features, adj)

            # Predict
            ypred_1 = student_model_1(features, adj)
            ypred_2 = student_model_2(features, adj)
            ypred_3 = student_model_3(features, adj)
            ypred_4 = student_model_4(features, adj)
            ypred_5 = student_model_5(features, adj)
            y_pred_ensamble = torch.unsqueeze(sum(ypred_1 + ypred_2 + ypred_3 + ypred_4 + ypred_5)/5, dim=0)

            # Compute loss (foward propagation)
            loss_teacher_student = criterion_soft(ypred_1, y_soft) + criterion_soft(ypred_2, y_soft) + criterion_soft(ypred_3, y_soft) + criterion_soft(ypred_4, y_soft) + criterion_soft(ypred_5, y_soft)
            loss_within_student = weight_similarity_loss(student_weights_1, student_weights_2) + weight_similarity_loss(student_weights_1, student_weights_3) + weight_similarity_loss(student_weights_2, student_weights_3) + weight_similarity_loss(student_weights_1, student_weights_4) + weight_similarity_loss(student_weights_2, student_weights_4) + weight_similarity_loss(student_weights_3, student_weights_4) + weight_similarity_loss(student_weights_1, student_weights_5) + weight_similarity_loss(student_weights_2, student_weights_5) + weight_similarity_loss(student_weights_3, student_weights_5) + weight_similarity_loss(student_weights_4, student_weights_5)
            loss_ensamble_soft_ce = criterion_soft(y_pred_ensamble, y_soft)
            loss_ensamble_ce = criterion(y_pred_ensamble, y_gt)

            loss = model_args["alpha"]*loss_ensamble_ce + model_args["beta"]*loss_ensamble_soft_ce + model_args["gamma"]*loss_teacher_student + model_args["lambda"]*loss_within_student
            
            # Compute gradients (backward propagation)
            loss.backward()
            
            # Update parameters
            optimizer_1.step()
            optimizer_2.step()
            optimizer_3.step()
            optimizer_4.step()
            optimizer_5.step()

            total_loss += loss.item()
            t_loss_teacher_student += loss_teacher_student.item()
            t_loss_within_student += loss_within_student.item()
            t_ensamble_soft_ce_loss += loss_ensamble_soft_ce.item()
            t_loss_ensamble_ce += loss_ensamble_ce.item()

            elapsed = time.time() - begin_time
            total_time += elapsed
        
        # Save weights of GNN model
        if epoch==model_args["num_epochs"]-1:
              student_model_1.is_trained = True
              student_model_2.is_trained = True
              student_model_3.is_trained = True
              student_model_4.is_trained = True
              student_model_5.is_trained = True

              # Get the predictions of the ensamble and the individual models
              _, indices = torch.max(y_pred_ensamble, 1)
              preds_ensamble.append(indices.cpu().data.numpy())
              labels_ensamble.append(data['label'].long().numpy())

              _, indices = torch.max(ypred_1, 1)
              preds_1.append(indices.cpu().data.numpy())
              labels_1.append(data['label'].long().numpy())

              _, indices = torch.max(ypred_2, 1)
              preds_2.append(indices.cpu().data.numpy())
              labels_2.append(data['label'].long().numpy())

              _, indices = torch.max(ypred_3, 1)
              preds_3.append(indices.cpu().data.numpy())
              labels_3.append(data['label'].long().numpy())

              _, indices = torch.max(ypred_4, 1)
              preds_4.append(indices.cpu().data.numpy())
              labels_4.append(data['label'].long().numpy())

              _, indices = torch.max(ypred_5, 1)
              preds_5.append(indices.cpu().data.numpy())
              labels_5.append(data['label'].long().numpy())
              
        print("---------------------------------")
        print(f"Time taken for epoch {epoch}: {total_time}")
        print(f"Train total loss: {total_loss / len(train_dataset)}")
        print(f"Train teacher and student loss: {t_loss_teacher_student / len(train_dataset)}")
        print(f"Train within student loss for weights: {t_loss_within_student / len(train_dataset)}")

        total_train_loss.append(total_loss / len(train_dataset))
        train_loss_teacher_student.append(t_loss_teacher_student / len(train_dataset)) 
        train_loss_ensamble_ce.append(t_loss_ensamble_ce / len(train_dataset)) 
        # soft ensamble loss
        train_ensamble_soft_ce_loss.append(t_ensamble_soft_ce_loss / len(train_dataset)) 
        # loss of weights within the students 
        train_loss_within_student.append(t_loss_within_student / len(train_dataset)) 
        
        val_total_loss, val_loss_teacher_student, val_loss_ensamble_ce, val_ensamble_soft_ce_loss, val_loss_within_student = validate(val_dataset, students, model_args, threshold_value, model_name, teacher_model)
        validation_total_loss.append(val_total_loss)
        validation_loss_teacher_student.append(val_loss_teacher_student)
        validation_loss_ensamble_ce.append(val_loss_ensamble_ce)
        validation_ensamble_soft_ce_loss.append(val_ensamble_soft_ce_loss)
        validation_loss_within_student.append(val_loss_within_student)
    
    # Save final labels and predictions of model on train set for ensamble and indiviudal students in ensamble (5)
    simple_r = {'labels':labels_ensamble,'preds':preds_ensamble}
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args["model_name"]+"/labels_and_preds/"+model_name+"_train_ensemble.pickle", 'wb') as f:
      pickle.dump(simple_r, f)
    
    simple_r = {'labels':np.hstack(labels_1),'preds':np.hstack(preds_1)}
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args["model_name"]+"/labels_and_preds/"+model_name+"_train_student_0.pickle", 'wb') as f:
      pickle.dump(simple_r, f)
    
    simple_r = {'labels':np.hstack(labels_2),'preds':np.hstack(preds_2)}
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args["model_name"]+"/labels_and_preds/"+model_name+"_train_student_1.pickle", 'wb') as f:
      pickle.dump(simple_r, f)
    
    simple_r = {'labels':np.hstack(labels_3),'preds':np.hstack(preds_3)}
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args["model_name"]+"/labels_and_preds/"+model_name+"_train_student_2.pickle", 'wb') as f:
      pickle.dump(simple_r, f) 

    simple_r = {'labels':np.hstack(labels_4),'preds':np.hstack(preds_4)}
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args["model_name"]+"/labels_and_preds/"+model_name+"_train_student_3.pickle", 'wb') as f:
      pickle.dump(simple_r, f)    

    simple_r = {'labels':np.hstack(labels_5),'preds':np.hstack(preds_5)}
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args["model_name"]+"/labels_and_preds/"+model_name+"_train_student_4.pickle", 'wb') as f:
      pickle.dump(simple_r, f) 
    
    # Save training loss of GNN model (5)
    los_p = {'loss':total_train_loss}
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args['model_name']+"/training_loss/training_loss_ensemble_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    los_p = {'loss':train_loss_teacher_student}
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args['model_name']+"/training_loss/training_loss_teacher_student_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    los_p = {'loss':train_loss_ensamble_ce}
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args['model_name']+"/training_loss/training_loss_ensamble_ce_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    los_p = {'loss':train_ensamble_soft_ce_loss}
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args['model_name']+"/training_loss/training_loss_ensamble_soft_ce_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    los_p = {'loss':train_loss_within_student}
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args['model_name']+"/training_loss/training_loss_within_student_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)

    # Save validation loss of GNN model (5)
    los_p = {'loss':validation_total_loss}
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args['model_name']+"/validation_loss/validation_loss_ensemble_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    los_p = {'loss':validation_loss_teacher_student}
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args['model_name']+"/validation_loss/validation_loss_teacher_student_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    los_p = {'loss':validation_loss_ensamble_ce}
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args['model_name']+"/validation_loss/validation_loss_ensamble_ce_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    los_p = {'loss':validation_ensamble_soft_ce_loss}
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args['model_name']+"/validation_loss/validation_loss_ensamble_soft_ce_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    los_p = {'loss':validation_loss_within_student}
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args['model_name']+"/validation_loss/validation_loss_within_student_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)    
    
    # Save Model (4)
    number = 0
    for student_name, student_model in zip(student_names, [student_model_1, student_model_2, student_model_3, student_model_4, student_model_5]):
      print(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args['model_name']+"/models/"+student_name+".pt")
      torch.save(student_model, SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args['model_name']+"/models/"+student_name+".pt")
    
      # Save weights
      path = SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args['model_name']+'/weights/W_'+student_name+'.pickle'
      
      if os.path.exists(path):
          os.remove(path)

      dataset_NAME = model_args['dataset']
      shutil.move(model_args['model_name']+f'_number_{number}_run_{run}_{dataset_NAME}_W.pickle', path)   
      
      number+=1



def validate(dataset, students, model_args, threshold_value, model_name, teacher_model):
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
    student_model_1 = students[0].eval()
    student_model_2 = students[1].eval()
    student_model_3 = students[2].eval()
    student_model_4 = students[3].eval()
    student_model_5 = students[4].eval()

    t_loss_teacher_student = 0
    t_loss_ensamble_ce = 0
    t_ensamble_soft_ce_loss = 0
    t_loss_within_student = 0
    total_loss = 0

    preds_ensamble, labels_ensamble  = [], []
    preds_1, labels_1 = [], []
    preds_2, labels_2 = [], []
    preds_3, labels_3 = [], []
    preds_4, labels_4 = [], []
    preds_5, labels_5 = [], []
    
    student_weights_1 = student_model_1.LinearLayer.weight
    student_weights_2 = student_model_2.LinearLayer.weight
    student_weights_3 = student_model_3.LinearLayer.weight
    student_weights_4 = student_model_4.LinearLayer.weight
    student_weights_5 = student_model_5.LinearLayer.weight

    criterion = nn.CrossEntropyLoss(reduction='mean')
    criterion_soft = CrossEntropyLossForSoftTarget(T=model_args["T"])

    for _, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).to(device)
        label = Variable(data['label'].long()).to(device)        
        
        adj = torch.squeeze(adj)
        
        features = np.identity(adj.shape[0])
        
        features = Variable(torch.from_numpy(features).float(), requires_grad=False).to(device)
        
        if model_args["threshold"] in ["median", "mean"]:
            adj = torch.where(adj > threshold_value, torch.tensor([1.0]).to(device), torch.tensor([0.0]).to(device))
        
        ypred_1 = student_model_1(features, adj)
        ypred_2 = student_model_2(features, adj)
        ypred_3 = student_model_3(features, adj)
        ypred_4 = student_model_4(features, adj)
        ypred_5 = student_model_5(features, adj)
        y_pred_ensamble = torch.unsqueeze(sum(ypred_1 + ypred_2 + ypred_3 + ypred_4+ ypred_5)/5, dim=0)

        # Ground truth label 
        y_gt = label.to(device)
        # Compute soft label
        y_soft, _ = teacher_model(features, adj)

        loss_teacher_student = criterion_soft(ypred_1, y_soft) + criterion_soft(ypred_2, y_soft) + criterion_soft(ypred_3, y_soft) + criterion_soft(ypred_4, y_soft) + criterion_soft(ypred_5, y_soft)
        loss_within_student = weight_similarity_loss(student_weights_1, student_weights_2) + weight_similarity_loss(student_weights_1, student_weights_3) + weight_similarity_loss(student_weights_2, student_weights_3) + weight_similarity_loss(student_weights_1, student_weights_4) + weight_similarity_loss(student_weights_2, student_weights_4) + weight_similarity_loss(student_weights_3, student_weights_4) + weight_similarity_loss(student_weights_1, student_weights_5) + weight_similarity_loss(student_weights_2, student_weights_5) + weight_similarity_loss(student_weights_3, student_weights_5) + weight_similarity_loss(student_weights_4, student_weights_5)
        loss_ensamble_soft_ce = criterion_soft(y_pred_ensamble, y_soft)
        loss_ensamble_ce = criterion(y_pred_ensamble, y_gt)

        loss = model_args["alpha"]*loss_ensamble_ce + model_args["beta"]*loss_ensamble_soft_ce + model_args["gamma"]*loss_teacher_student + model_args["lambda"]*loss_within_student

        total_loss += loss.item()
        t_loss_teacher_student += loss_teacher_student
        t_loss_within_student += loss_within_student
        t_ensamble_soft_ce_loss += loss_ensamble_soft_ce
        t_loss_ensamble_ce += loss_ensamble_ce

        # Get the predictions of the ensamble and the individual models
        _, indices = torch.max(y_pred_ensamble, 1)
        preds_ensamble.append(indices.cpu().data.numpy())
        labels_ensamble.append(data['label'].long().numpy())

        _, indices = torch.max(ypred_1, 1)
        preds_1.append(indices.cpu().data.numpy())
        labels_1.append(data['label'].long().numpy())

        _, indices = torch.max(ypred_2, 1)
        preds_2.append(indices.cpu().data.numpy())
        labels_2.append(data['label'].long().numpy())

        _, indices = torch.max(ypred_3, 1)
        preds_3.append(indices.cpu().data.numpy())
        labels_3.append(data['label'].long().numpy())

        _, indices = torch.max(ypred_4, 1)
        preds_4.append(indices.cpu().data.numpy())
        labels_4.append(data['label'].long().numpy())

        _, indices = torch.max(ypred_5, 1)
        preds_5.append(indices.cpu().data.numpy())
        labels_5.append(data['label'].long().numpy())
    
    simple_r = {'labels':labels_ensamble,'preds':preds_ensamble}
    # Save labels and predictions of model on test set  (4)
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args["model_name"]+"/labels_and_preds/"+model_name+"_val_ensamble.pickle", 'wb') as f:
      pickle.dump(simple_r, f)
    
    simple_r = {'labels':labels_1,'preds':preds_1}
    # Save labels and predictions of model on test set 
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args["model_name"]+"/labels_and_preds/"+model_name+"_val_student_0.pickle", 'wb') as f:
      pickle.dump(simple_r, f)   

    simple_r = {'labels':labels_2,'preds':preds_2}
    # Save labels and predictions of model on test set 
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args["model_name"]+"/labels_and_preds/"+model_name+"_val_student_1.pickle", 'wb') as f:
      pickle.dump(simple_r, f)   

    simple_r = {'labels':labels_3,'preds':preds_3}
    # Save labels and predictions of model on test set 
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args["model_name"]+"/labels_and_preds/"+model_name+"_val_student_2.pickle", 'wb') as f:
      pickle.dump(simple_r, f)   

    simple_r = {'labels':labels_4,'preds':preds_4}
    # Save labels and predictions of model on test set 
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args["model_name"]+"/labels_and_preds/"+model_name+"_val_student_3.pickle", 'wb') as f:
      pickle.dump(simple_r, f)   

    simple_r = {'labels':labels_5,'preds':preds_5}
    # Save labels and predictions of model on test set 
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args["model_name"]+"/labels_and_preds/"+model_name+"_val_student_4.pickle", 'wb') as f:
      pickle.dump(simple_r, f)   

    val_total_loss = total_loss / len(dataset)
    val_loss_teacher_student = t_loss_teacher_student / len(dataset)
    val_loss_ensamble_ce = t_loss_ensamble_ce / len(dataset)
    val_ensamble_soft_ce_loss = t_ensamble_soft_ce_loss / len(dataset)
    val_loss_within_student = t_loss_within_student / len(dataset)
    print(f"Validation accuracy ensamble: {metrics.accuracy_score(np.hstack(labels_ensamble), np.hstack(preds_ensamble))}")
    print(f"Validation accuracy model 1: {metrics.accuracy_score(np.hstack(labels_1), np.hstack(preds_1))}")
    print(f"Validation accuracy model 2: {metrics.accuracy_score(np.hstack(labels_2), np.hstack(preds_2))}")
    print(f"Validation accuracy model 3: {metrics.accuracy_score(np.hstack(labels_3), np.hstack(preds_3))}")
    print(f"Validation accuracy model 4: {metrics.accuracy_score(np.hstack(labels_4), np.hstack(preds_4))}")
    print(f"Validation accuracy model 5: {metrics.accuracy_score(np.hstack(labels_5), np.hstack(preds_5))}")
    print(f"Validation Loss: {val_total_loss}")

    return val_total_loss, val_loss_teacher_student, val_loss_ensamble_ce, val_ensamble_soft_ce_loss, val_loss_within_student

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
    with open(SAVE_DIR_MODEL_DATA+model_args['dataset']+"/"+model_args['backbone']+"/"+model_args['evaluation_method']+"/"+model_args["model_name"]+"/labels_and_preds/"+model_name+"_test.pickle", 'wb') as f:
      pickle.dump(simple_r, f)

    print('Held-out test set loss: {}'.format(total_loss / len(dataset)))
    print("Test accuracy:", metrics.accuracy_score(labels, preds))    
