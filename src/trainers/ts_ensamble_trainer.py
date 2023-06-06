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

from models.gcn_student_ensamble import GCN_STUDENT_ENSAMBLE
from models.model_config import * 
from utils.helpers import *
from utils.analysis import * 
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
    Compute the KL loss between the weights of the last layer
    of two networks.
    """    
    # Concatenate and compute the cosine similarity
    loss = nn.CosineSimilarity()
    return loss(w_student, w_teacher).abs()

def cross_validation(model_args, G_list, view, model_name, cv_number, n_students, run=0):
    start = time.time() 
    print("Run : ",run)
    print("--------------------------------------------------------------------------")
    print("Main : ", view, model_name, cv_number)
    
    folds = stratify_splits(G_list, cv_number)
    student_names = []
    students = []
    
    [random.shuffle(folds[i]) for i in range(len(folds))]
    
    for cv in range(cv_number):

        train_set, validation_set, test_set = datasets_splits(folds, model_args, cv)
        
        if model_args["evaluation_method"] =='model_selection':
            train_dataset, val_dataset, threshold_value = model_selection_split(train_set, validation_set, model_args)
        if model_args["evaluation_method"] =='model_assessment':
            train_dataset, val_dataset, threshold_value = model_assessment_split(train_set, validation_set, test_set, model_args)
        print(f"CV : {cv}")

        for i in range(n_students):
            name = model_name+f"_student_{str(i)}"+"_CV_"+str(cv)+"_view_"+str(view)
            student_names.append(name)
        
        print(model_name)
        print(student_names)

        train_set, validation_set, test_set = datasets_splits(folds, model_args, cv)
        num_nodes = G_list[0]['adj'].shape[0]
        num_classes = 2 
        for i in range(n_students):
            student_model = GCN_STUDENT_ENSAMBLE(
                nfeat = num_nodes,
                nhid = model_args["hidden_dim"],
                nclass = num_classes,
                dropout = model_args["dropout"],
                seed = i,
                run = run, 
                number = i,
                total_number = model_args["n_students"]
            ).to(device)
            students.append(student_model)

        if model_args["evaluation_method"] =='model_selection':
            #Here we leave out the test set since we are not evaluating we can see the performance on the test set after training
            train(model_args, train_dataset, val_dataset, student_model, threshold_value, model_name, cv, view, cv_number)
            #See performance on the held-out test set 
            dataset_sampler = GraphSampler(test_set)
            test_dataset = torch.utils.data.DataLoader(
                dataset_sampler, 
                batch_size = 1,  
                shuffle = False) 
            test(test_dataset, students, model_args, threshold_value, model_name)

        if model_args["evaluation_method"] =='model_assessment':
            #Here we join the train and validation dataset
            train(model_args, train_dataset, val_dataset, students, student_names, threshold_value, model_name,  cv, view, cv_number, run)
    
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
    print("In training")
    print(model_name)
    # Load teacher model
    if model_args['evaluation_method'] == "model_selection":
       teacher_model = torch.load(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+f"/gcn/models/gcn_MainModel_{cv_number}Fold_gender_data_gcn_CV_{cv}_view_{view}.pt")
       teacher_weights_path = SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+f"/gcn/weights/W_MainModel_{cv_number}Fold_gender_data_gcn_CV_{cv}_view_{view}.pickle"
       with open(teacher_weights_path,'rb') as f:
          teacher_weights = pickle.load(f)
    else:
       teacher_model = torch.load(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+f"/gcn/models/gcn_MainModel_{cv_number}Fold_gender_data_gcn_run_{run}_fixed_init_CV_{cv}_view_{view}.pt")
       teacher_weights_path = SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+f"/gcn/weights/W_MainModel_{cv_number}Fold_gender_data_gcn_run_{run}_fixed_init_CV_{cv}_view_{view}.pickle"
       with open(teacher_weights_path,'rb') as f:
          teacher_weights = pickle.load(f)
    
    teacher_model.is_trained = False
    teacher_model.eval()

    #Extract teacher weights
    teacher_weights = teacher_weights['w'].detach()
    # Transfer
    teacher_model.to(device)
    student_model_1 = students[0].to(device)
    student_model_2 = students[1].to(device)
    student_model_3 = students[2].to(device)

    # Define Loss
    criterion = nn.CrossEntropyLoss(reduction='mean')
    criterion_soft = CrossEntropyLossForSoftTarget(T=model_args["T"], alpha=model_args["alpha_soft_ce"])

    # Define optimizer
    optimizer_1 = optim.Adam(student_model_1.parameters(), lr=model_args["lr"], weight_decay=model_args['weight_decay'])
    optimizer_2 = optim.Adam(student_model_2.parameters(), lr=model_args["lr"], weight_decay=model_args['weight_decay'])
    optimizer_3 = optim.Adam(student_model_3.parameters(), lr=model_args["lr"], weight_decay=model_args['weight_decay'])
    
    # Metrics 
    # total train loss 
    total_train_loss = []
    # soft-cross entropy between students and teacher
    train_loss_1, train_loss_2, train_loss_3 = [], [], []
    # performance accuracy of each student
    train_acc_1, train_acc_2, train_acc_3 = [], [], []
    # cumulative loss of teacher and student soft-ce
    train_loss_teacher_student = []
    # ensamble loss
    train_loss_ensamble_ce = []
    # soft ensamble loss
    train_ensamble_soft_ce_loss=[]
    # loss of weights within the students 
    train_loss_within_student=[]
    # ensamble accuracy 
    train_ensamble_acc = []

    # Validation losses
    validation_total_loss = []
    validation_loss_1, validation_loss_2, validation_loss_3 = [], [], []
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

        total_time = 0
        total_loss = 0
        t_loss_1, t_loss_2, t_loss_3 = 0, 0, 0
        t_loss_teacher_student = 0
        t_loss_ensamble_ce = 0
        t_ensamble_soft_ce_loss = 0
        t_loss_within_student = 0
        
        preds_ensamble, labels_ensamble  = [], []
        preds_1, labels_1 = [], []
        preds_2, labels_2 = [], []
        preds_3, labels_3 = [], []

        for _, data in enumerate(train_dataset):
            begin_time = time.time()
            
            # Initialize gradients
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            optimizer_3.zero_grad()

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

            y_gt = label.to(device)

            # Compute soft label
            y_soft = teacher_model(features, adj)

            # Predict
            ypred_1 = student_model_1(features, adj)
            ypred_2 = student_model_2(features, adj)
            ypred_3 = student_model_3(features, adj)
            y_pred_ensamble = torch.unsqueeze(sum(ypred_1 + ypred_2 + ypred_3)/3, dim=0)

            # Compute loss (foward propagation)
            loss_teacher_student = criterion_soft(ypred_1, y_soft) + criterion_soft(ypred_2, y_soft) + criterion_soft(ypred_3, y_soft)
            loss_within_student = weight_similarity_loss(student_weights_1, student_weights_2) + weight_similarity_loss(student_weights_1, student_weights_2) + weight_similarity_loss(student_weights_2, student_weights_3)
            loss_ensamble_ce = criterion_soft(y_pred_ensamble, y_soft)
            loss_ensamble_soft_ce = model_args["alpha_ce"]*criterion(y_pred_ensamble, y_gt)
                
            loss = loss_teacher_student + loss_within_student + loss_ensamble_ce + loss_ensamble_soft_ce

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
            
            # Compute gradients (backward propagation)
            loss.backward()
            
            # Update parameters
            optimizer_1.step()
            optimizer_2.step()
            optimizer_3.step()

            total_loss += loss.item()
            t_loss_1+= criterion_soft(ypred_1, y_soft)
            t_loss_2+= criterion_soft(ypred_2, y_soft)
            t_loss_3+= criterion_soft(ypred_3, y_soft)
            t_loss_teacher_student += loss_teacher_student
            t_loss_within_student += loss_within_student.item()
            t_ensamble_soft_ce_loss += loss_ensamble_soft_ce
            t_loss_ensamble_ce += loss_ensamble_ce

            elapsed = time.time() - begin_time
            total_time += elapsed
        
        # Save weights of GNN model
        if epoch==model_args["num_epochs"]-1:
              student_model_1.is_trained = True
              student_model_2.is_trained = True
              student_model_3.is_trained = True
        
        preds_ensamble = np.hstack(preds_ensamble)
        labels_ensamble  = np.hstack(labels_ensamble)
        result = {
            'prec': metrics.precision_score(labels_ensamble, preds_ensamble),
            'recall': metrics.recall_score(labels_ensamble, preds_ensamble),
            'acc': metrics.accuracy_score(labels_ensamble, preds_ensamble),
            'F1': metrics.f1_score(labels_ensamble, preds_ensamble)
        }
        train_acc_1.append(metrics.accuracy_score(np.hstack(labels_1), np.hstack(preds_1)))
        train_acc_2.append(metrics.accuracy_score(np.hstack(labels_2), np.hstack(preds_2)))
        train_acc_3.append(metrics.accuracy_score(np.hstack(labels_3), np.hstack(preds_3)))
        train_ensamble_acc.append(result['acc'])
              
        print("---------------------------------")
        print(f"Time taken for epoch {epoch}: {total_time}")
        print(f"Train ensamble accuracy: {result['acc']}")
        print(f"Train model 1 accuracy: {train_acc_1[-1]}")
        print(f"Train model 2 accuracy: {train_acc_2[-1]}")
        print(f"Train model 3 accuracy: {train_acc_3[-1]}")
        print(f"Train total loss: {total_loss / len(train_dataset)}")
        print(f"Train teacher and student loss: {t_loss_teacher_student / len(train_dataset)}")
        print(f"Train within student loss for weights: {t_loss_within_student / len(train_dataset)}")

        total_train_loss.append(total_loss / len(train_dataset))
        train_loss_1.append( t_loss_1/ len(train_dataset))
        train_loss_2.append( t_loss_2/ len(train_dataset))
        train_loss_3.append( t_loss_3/ len(train_dataset))
        train_loss_teacher_student.append(t_loss_teacher_student / len(train_dataset)) 
        train_loss_ensamble_ce.append(t_loss_ensamble_ce / len(train_dataset)) 
        # soft ensamble loss
        train_ensamble_soft_ce_loss.append(t_ensamble_soft_ce_loss / len(train_dataset)) 
        # loss of weights within the students 
        train_loss_within_student.append(t_loss_within_student / len(train_dataset)) 
        
        val_total_loss, val_loss_1, val_loss_2, val_loss_3, val_loss_teacher_student, val_loss_ensamble_ce, val_ensamble_soft_ce_loss, val_loss_within_student = validate(val_dataset, students, model_args, threshold_value, model_name, teacher_model, teacher_weights)
        validation_total_loss.append(val_total_loss)
        validation_loss_1.append(val_loss_1)
        validation_loss_2.append(val_loss_2)
        validation_loss_3.append(val_loss_3)
        validation_loss_teacher_student.append(val_loss_teacher_student)
        validation_loss_ensamble_ce.append(val_loss_ensamble_ce)
        validation_ensamble_soft_ce_loss.append(val_ensamble_soft_ce_loss)
        validation_loss_within_student.append(val_loss_within_student)
    
    
    #Save train metrics acc
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args["model_name"]+"/metrics/"+model_name+"_train_acc_ensamble.pickle", 'wb') as f:
      pickle.dump(train_ensamble_acc, f)
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args["model_name"]+"/metrics/"+model_name+"_train_acc_student_1.pickle", 'wb') as f:
      pickle.dump(train_acc_1, f)
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args["model_name"]+"/metrics/"+model_name+"_train_acc_student_2.pickle", 'wb') as f:
      pickle.dump(train_acc_2, f)
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args["model_name"]+"/metrics/"+model_name+"_train_acc_student_3.pickle", 'wb') as f:
      pickle.dump(train_acc_3, f)

    # Save final labels and predictions of model on train set for ensamble and indiviudal students in ensamble 
    simple_r = {'labels':labels_ensamble,'preds':preds_ensamble}
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args["model_name"]+"/labels_and_preds/"+model_name+"_train_ensemble.pickle", 'wb') as f:
      pickle.dump(simple_r, f)
    
    simple_r = {'labels':np.hstack(labels_1),'preds':np.hstack(preds_1)}
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args["model_name"]+"/labels_and_preds/"+model_name+"_train_student_1.pickle", 'wb') as f:
      pickle.dump(simple_r, f)
    
    simple_r = {'labels':np.hstack(labels_2),'preds':np.hstack(preds_2)}
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args["model_name"]+"/labels_and_preds/"+model_name+"_train_student_2.pickle", 'wb') as f:
      pickle.dump(simple_r, f)
    
    simple_r = {'labels':np.hstack(labels_3),'preds':np.hstack(preds_3)}
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args["model_name"]+"/labels_and_preds/"+model_name+"_train_student_3.pickle", 'wb') as f:
      pickle.dump(simple_r, f)    
    
    # Save training loss of GNN model
    los_p = {'loss':total_train_loss}
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args['model_name']+"/training_loss/training_loss_ensemble_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    los_p = {'loss':train_loss_1}
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args['model_name']+"/training_loss/training_loss_student_1_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    los_p = {'loss':train_loss_2}
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args['model_name']+"/training_loss/training_loss_student_2_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    los_p = {'loss':train_loss_3}
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args['model_name']+"/training_loss/training_loss_student_3_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    los_p = {'loss':train_loss_teacher_student}
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args['model_name']+"/training_loss/training_loss_teacher_student_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    los_p = {'loss':train_loss_ensamble_ce}
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args['model_name']+"/training_loss/training_loss_ensamble_ce_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    los_p = {'loss':train_ensamble_soft_ce_loss}
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args['model_name']+"/training_loss/training_loss_ensamble_soft_ce_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    los_p = {'loss':train_loss_within_student}
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args['model_name']+"/training_loss/training_loss_within_student_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)

    # Save validation loss of GNN model
    los_p = {'loss':validation_total_loss}
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args['model_name']+"/validation_loss/validation_loss_ensemble_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    los_p = {'loss':validation_loss_1}
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args['model_name']+"/validation_loss/validation_loss_student_1_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    los_p = {'loss':validation_loss_2}
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args['model_name']+"/validation_loss/validation_loss_student_2_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    los_p = {'loss':validation_loss_3}
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args['model_name']+"/validation_loss/validation_loss_student_3_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    los_p = {'loss':validation_loss_teacher_student}
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args['model_name']+"/validation_loss/validation_loss_teacher_student_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    los_p = {'loss':validation_loss_ensamble_ce}
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args['model_name']+"/validation_loss/validation_loss_ensamble_ce_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    los_p = {'loss':validation_ensamble_soft_ce_loss}
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args['model_name']+"/validation_loss/validation_loss_ensamble_soft_ce_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    los_p = {'loss':validation_loss_within_student}
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args['model_name']+"/validation_loss/validation_loss_within_student_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)    
    
    # Save Model
    number = 0
    for student_name, student_model in zip(student_names, [student_model_1, student_model_2, student_model_3]):
      torch.save(student_model, SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args['model_name']+"/models/"+student_name+".pt")
    
      # Save weights
      if model_args['model_name'] == "diffpool":
          w_dict = {"w": student_model.state_dict()["assign_conv_first_modules.0.weight"]}
          with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args['model_name']+'/weights/W_'+student_name+'.pickle', 'wb') as f:
              pickle.dump(w_dict, f)
      else:
          path = SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args['model_name']+'/weights/W_'+student_name+'.pickle'
          
          if os.path.exists(path):
              os.remove(path)

          os.rename(model_args['model_name']+f'_number_{number}_run_{run}_W.pickle', path)  
      
      number+=1



def validate(dataset, students, model_args, threshold_value, model_name, teacher_model, teacher_weights):
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

    t_loss_1, t_loss_2, t_loss_3 = 0, 0, 0
    t_loss_teacher_student = 0
    t_loss_ensamble_ce = 0
    t_ensamble_soft_ce_loss = 0
    t_loss_within_student = 0
    total_loss = 0

    preds_ensamble, labels_ensamble  = [], []
    preds_1, labels_1 = [], []
    preds_2, labels_2 = [], []
    preds_3, labels_3 = [], []
    
    student_weights_1 = student_model_1.LinearLayer.weight
    student_weights_2 = student_model_2.LinearLayer.weight
    student_weights_3 = student_model_3.LinearLayer.weight

    criterion = nn.CrossEntropyLoss(reduction='mean')
    criterion_soft = CrossEntropyLossForSoftTarget(T=model_args["T"], alpha=model_args["alpha_soft_ce"])

    for _, data in enumerate(dataset):
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
            ypred_1 = student_model_1(features, adj)
            ypred_2 = student_model_2(features, adj)
            ypred_3 = student_model_3(features, adj)
            y_pred_ensamble = torch.unsqueeze(sum(ypred_1 + ypred_2 + ypred_3)/3, dim=0)
        else:
            ypred_1 = student_model_1(features, adj)
            ypred_2 = student_model_2(features, adj)
            ypred_3 = student_model_3(features, adj)
            y_pred_ensamble = torch.unsqueeze(sum(ypred_1 + ypred_2 + ypred_3)/3, dim=0)

        # Ground truth label 
        y_gt = label.to(device)
        # Compute soft label
        y_soft = teacher_model(features, adj)

        loss_teacher_student = criterion_soft(ypred_1, y_soft) + criterion_soft(ypred_2, y_soft) + criterion_soft(ypred_3, y_soft)
        loss_within_student = weight_similarity_loss(student_weights_1, student_weights_2) + weight_similarity_loss(student_weights_1, student_weights_2) + weight_similarity_loss(student_weights_2, student_weights_3)
        loss_ensamble_ce = criterion_soft(y_pred_ensamble, y_soft)
        loss_ensamble_soft_ce = model_args["alpha_ce"]*criterion(y_pred_ensamble, y_gt)

        loss = loss_teacher_student + loss_within_student + loss_ensamble_ce + loss_ensamble_soft_ce
        
        total_loss += loss.item()
        t_loss_1+= criterion_soft(ypred_1, y_soft)
        t_loss_2+= criterion_soft(ypred_2, y_soft)
        t_loss_3+= criterion_soft(ypred_3, y_soft)
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
    
    simple_r = {'labels':labels_ensamble,'preds':preds_ensamble}
    # Save labels and predictions of model on test set 
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args["model_name"]+"/labels_and_preds/"+model_name+"_val_ensamble.pickle", 'wb') as f:
      pickle.dump(simple_r, f)
    
    simple_r = {'labels':labels_2,'preds':preds_1}
    # Save labels and predictions of model on test set 
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args["model_name"]+"/labels_and_preds/"+model_name+"_val_student_1.pickle", 'wb') as f:
      pickle.dump(simple_r, f)   

    simple_r = {'labels':labels_2,'preds':preds_2}
    # Save labels and predictions of model on test set 
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args["model_name"]+"/labels_and_preds/"+model_name+"_val_student_2.pickle", 'wb') as f:
      pickle.dump(simple_r, f)   

    simple_r = {'labels':labels_2,'preds':preds_3}
    # Save labels and predictions of model on test set 
    with open(SAVE_DIR_MODEL_DATA+model_args['evaluation_method']+"/"+model_args["model_name"]+"/labels_and_preds/"+model_name+"_val_student_3.pickle", 'wb') as f:
      pickle.dump(simple_r, f)   

    val_total_loss = total_loss / len(dataset)
    val_loss_1, val_loss_2, val_loss_3 = t_loss_1 / len(dataset), t_loss_2 / len(dataset), t_loss_3 / len(dataset)
    val_loss_teacher_student = t_loss_teacher_student / len(dataset)
    val_loss_ensamble_ce = t_loss_ensamble_ce / len(dataset)
    val_ensamble_soft_ce_loss = t_ensamble_soft_ce_loss / len(dataset)
    val_loss_within_student = t_loss_within_student / len(dataset)
    print(f"Validation accuracy ensamble: {metrics.accuracy_score(np.hstack(labels_ensamble), np.hstack(preds_ensamble))}")
    print(f"Validation accuracy model 1: {metrics.accuracy_score(np.hstack(labels_1), np.hstack(preds_1))}")
    print(f"Validation accuracy model 2: {metrics.accuracy_score(np.hstack(labels_2), np.hstack(preds_2))}")
    print(f"Validation accuracy model 3: {metrics.accuracy_score(np.hstack(labels_3), np.hstack(preds_3))}")
    print(f"Validation Loss: {val_total_loss}")

    return val_total_loss, val_loss_1, val_loss_2, val_loss_3, val_loss_teacher_student, val_loss_ensamble_ce, val_ensamble_soft_ce_loss, val_loss_within_student

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
