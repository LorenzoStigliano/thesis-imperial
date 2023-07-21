import os 
import sys
import pickle
import numpy as np
import scipy.io as sio

#import medmnist
#from medmnist import INFO

#import torch.utils.data as data
#import torchvision.transforms as transforms

sys.path.append("/Users/lorenzostigliano/Documents/University/Imperial/Summer Term/thesis-imperial/src/utils")

from config import SAVE_DIR_MODEL_DATA

def dump_data_MEDMNIST(save_dir, dataset):
    """
    #1. download and save data all together to a directory and split labels 

    USAGE:
    dump_data_MEDMNIST(SAVE_DIR_DATA, 'BreastMNIST')
    dump_data_MEDMNIST(SAVE_DIR_DATA, 'PneumoniaMNIST')
    """
    data_flag=dataset
    download = True
    BATCH_SIZE = 1
    data_flag = data_flag.lower()
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    # preprocessing
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # load the data
    train_dataset = DataClass(split='train', transform=data_transform, download=download)
    test_dataset = DataClass(split='test', transform=data_transform, download=download)
    val_dataset = DataClass(split='val', transform=data_transform, download=download)

    # encapsulate data into dataloader form
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(train_dataset)
    
    images, labels = [], []
    for image, label in train_loader: 
        images.append(image.squeeze().numpy()) 
        labels.append(int(label[0][0])) 
    for image, label in test_loader: 
        images.append(image.squeeze().numpy()) 
        labels.append(int(label[0][0]))
    for image, label in val_loader: 
        images.append(image.squeeze().numpy()) 
        labels.append(int(label[0][0]))

    if not os.path.exists(save_dir + dataset):
        os.makedirs(save_dir + dataset) 

    with open(save_dir + dataset +'/'+dataset+'_images', 'wb') as f:
        pickle.dump(images, f)
    with open(save_dir + dataset +'/'+dataset+'_labels', 'wb') as f:
        pickle.dump(labels, f)

def dump_data_MEDMNIST_adj(save_dir, dataset):
    """
    #1. download and save data all together to a directory and split labels 

    USAGE:
    dump_data_MEDMNIST(SAVE_DIR_DATA, 'BreastMNIST')
    dump_data_MEDMNIST(SAVE_DIR_DATA, 'PneumoniaMNIST')
    """
    data_flag=dataset
    download = True
    BATCH_SIZE = 1
    data_flag = data_flag.lower()
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    # preprocessing
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # load the data
    train_dataset = DataClass(split='train', transform=data_transform, download=download)
    test_dataset = DataClass(split='test', transform=data_transform, download=download)
    val_dataset = DataClass(split='val', transform=data_transform, download=download)

    # encapsulate data into dataloader form
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(train_dataset)
    
    images, labels = [], []
    for image, label in train_loader: 
        images.append(img_to_adj(image.squeeze().numpy()))
        labels.append(int(label[0][0])) 
    for image, label in test_loader: 
        images.append(img_to_adj(image.squeeze().numpy()))
        labels.append(int(label[0][0]))
    for image, label in val_loader: 
        images.append(img_to_adj(image.squeeze().numpy()))
        labels.append(int(label[0][0]))

    if not os.path.exists(save_dir + dataset+"_ADJ"):
        os.makedirs(save_dir + dataset+"_ADJ") 

    with open(save_dir + dataset+"_ADJ" +'/'+dataset+"_ADJ"+'_images', 'wb') as f:
        pickle.dump(images, f)
    with open(save_dir + dataset+"_ADJ" +'/'+dataset+"_ADJ"+'_labels', 'wb') as f:
        pickle.dump(labels, f)

def dump_data_gender_data(data_dir, save_dir, dataset):

  adjs, ages, labels = [], [], []

  for file in os.listdir(data_dir):
      if ".mat" in file:
          mat = sio.loadmat(data_dir+str(file))
          tensor = mat['Tensor'].squeeze()
          age = mat['age'][0][0]
          gender = mat['gender'][0][0] if mat['gender'][0][0] == 1 else 0

          adjs.append(tensor)
          ages.append(age)
          labels.append(gender)

  if not os.path.exists(save_dir + dataset):
     os.makedirs(save_dir + dataset) 

  with open(save_dir + dataset +'/'+dataset+'_edges', 'wb') as f:
    pickle.dump(adjs, f)
  with open(save_dir + dataset +'/'+dataset+'_labels', 'wb') as f:
    pickle.dump(labels, f)
  with open(save_dir + dataset +'/'+dataset+'_ages', 'wb') as f:
    pickle.dump(ages, f)

def new_folder(model, evaluation_method, backbone="gcn", dataset="gender_data"):
    """
    Parameters
    ----------
    model : GNN model (diffpool, gat, gcn, gunet or sag)
    
    Description
    ----------
    Creates GNN directories if not exist.
    """
    print(SAVE_DIR_MODEL_DATA+dataset+"/"+backbone+"/"+evaluation_method+"/"+model)
    if not os.path.exists(SAVE_DIR_MODEL_DATA+dataset+"/"+backbone+"/"+evaluation_method+"/"+model):
        os.makedirs(SAVE_DIR_MODEL_DATA+dataset+"/"+backbone+"/"+evaluation_method+"/"+model)
        os.makedirs(SAVE_DIR_MODEL_DATA+dataset+"/"+backbone+"/"+evaluation_method+"/"+model+"/weights")
        os.makedirs(SAVE_DIR_MODEL_DATA+dataset+"/"+backbone+"/"+evaluation_method+"/"+model+"/training_loss")
        os.makedirs(SAVE_DIR_MODEL_DATA+dataset+"/"+backbone+"/"+evaluation_method+"/"+model+"/validation_loss")
        os.makedirs(SAVE_DIR_MODEL_DATA+dataset+"/"+backbone+"/"+evaluation_method+"/"+model+"/models")
        os.makedirs(SAVE_DIR_MODEL_DATA+dataset+"/"+backbone+"/"+evaluation_method+"/"+model+"/labels_and_preds")
        os.makedirs(SAVE_DIR_MODEL_DATA+dataset+"/"+backbone+"/"+evaluation_method+"/"+model+"/metrics")

def get_adjs(i, j, n):
    """
    Parameters
    ----------
    i : row of pixel
    j : column of pixel
    n : 1-D size of a squared image (nxn)
    
    Description
    ----------
    This method returns the adjacency list of specific pixel
    """
    adj_list = []
    # Upper-Left
    uli, ulj = i - 1, j - 1
    if 0 <= uli < n and 0 <= ulj < n:
        adj_list.append((uli, ulj))
    
    # Up
    ui, uj = i - 1, j
    if 0 <= ui < n and 0 <= uj < n:
        adj_list.append((ui, uj))
    
    # Upper-Right
    uri, urj = i - 1, j + 1
    if 0 <= uri < n and 0 <= urj < n:
        adj_list.append((uri, urj))
    
    # Left
    li, lj = i, j - 1
    if 0 <= li < n and 0 <= lj < n:
        adj_list.append((li, lj))
    
    # Right
    ri, rj = i, j + 1
    if 0 <= ri < n and 0 <= rj < n:
        adj_list.append((ri, rj))

    # Lower-Left
    lli, llj = i + 1, j - 1
    if 0 <= lli < n and 0 <= llj < n:
        adj_list.append((lli, llj))

    # Down
    di, dj = i + 1, j
    if 0 <= di < n and 0 <= dj < n:
        adj_list.append((di, dj))
    
    # Lower-right
    lri, lrj = i + 1, j + 1
    if 0 <= lri < n and 0 <= lrj < n:
        adj_list.append((lri, lrj))
    
    return adj_list

def img_to_adj(image):
    """
    Parameters
    ----------
    image : nxn square image
    
    Description
    ----------
    This method returns the weighted adjacency matrix
    Weigths are determined by the absolute differences of adjacent pixels

    """
    n = image.shape[0]
    adj = np.zeros((n * n,n * n), np.float32)
    for i in range(0,n):
        for j in range(0 ,n):
            adjList = get_adjs(i, j, n)
            for u,v in adjList:
                weight = np.abs(image[i, j] - image[u, v])
                adj[n * i + j, n * u + v] = weight
                adj[n * u + v, n * i + j] = weight
    return adj
