import os 
import scipy.io as sio
import pickle
from config import DATA_DIR, SAVE_DIR_DATA, DATASET, SAVE_DIR_MODEL_DATA

def dump_data(data_dir, save_dir, dataset):

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

def new_folder(model):
    """
    Parameters
    ----------
    model : GNN model (diffpool, gat, gcn, gunet or sag)
    
    Description
    ----------
    Creates GNN directories if not exist.
    """
    if not os.path.exists(SAVE_DIR_MODEL_DATA+model):
        os.makedirs(SAVE_DIR_MODEL_DATA+model)
        os.makedirs(SAVE_DIR_MODEL_DATA+model+"/weights")
        os.makedirs(SAVE_DIR_MODEL_DATA+model+"/training_loss")
        os.makedirs(SAVE_DIR_MODEL_DATA+model+"/models")
        os.makedirs(SAVE_DIR_MODEL_DATA+model+"/labels_and_preds")

dump_data(DATA_DIR, SAVE_DIR_DATA, DATASET)