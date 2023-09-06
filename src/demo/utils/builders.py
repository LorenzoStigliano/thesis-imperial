import os 
import sys
import pickle
import numpy as np
import scipy.io as sio

import medmnist
from medmnist import INFO

import torch.utils.data as data
import torchvision.transforms as transforms

def dump_data_MEDMNIST(save_dir, dataset):
    """
    Download, preprocess, and save MEDMNIST dataset to a directory while splitting labels.

    This function downloads the specified MEDMNIST dataset, preprocesses it, and saves the data along with the corresponding labels to a specified directory.

    Parameters:
        save_dir (str): Directory where the dataset will be saved.
        dataset (str): Name of the MEDMNIST dataset to download (e.g., 'BreastMNIST').

    Usage:
    dump_data_MEDMNIST(SAVE_DIR_DATA, 'BreastMNIST')

    Adapted from: https://github.com/basiralab/reproducibleFedGNN
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

def dump_data_gender_data(data_dir, save_dir, dataset):
    """
    Load and preprocess gender-related data from MAT files and save to a directory.

    This function reads gender-related data from MAT files located in the specified directory,
    preprocesses the data (including adjacency tensors, ages, and labels), and saves them
    along with the corresponding labels and ages to a specified directory.

    Parameters:
        data_dir (str): Directory containing the MAT files to be processed.
        save_dir (str): Directory where the preprocessed data will be saved.
        dataset (str): Name of the dataset (used for creating subdirectories).

    Usage:
    dump_data_gender_data(DATA_DIR, SAVE_DIR_DATA, 'GenderData')
    
    Adapted from: https://github.com/basiralab/RG-Select
    """

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

def new_folder(model, evaluation_method, SAVE_DIR_MODEL_DATA, backbone="gcn", dataset="gender_data"):
    """
    Create directories for storing GNN model-related data.

    This function creates directories for a specified GNN model, evaluation method, backbone,
    and dataset. The directories are organized to store various data related to the model,
    such as weights, training and validation loss, trained models, labels and predictions,
    and metrics.

    Parameters:
    model (str): Name of the model or KD method (gcn, gcn_student,...).
    evaluation_method (str): Name of the evaluation method.
    SAVE_DIR_MODEL_DATA (str): Directory where the model-related data will be saved.
    backbone (str, optional): Name of the GNN backbone (default is "gcn").
    dataset (str, optional): Name of the dataset (default is "gender_data").

    Usage:
    new_folder("gcn", "evaluation_method", SAVE_DIR_MODEL_DATA, backbone="gcn", dataset="gender_data")
    
    Adapted from: https://github.com/basiralab/RG-Select
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
