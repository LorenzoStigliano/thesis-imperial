# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 17:08:19 2021

@author: user
"""

import os
import numpy as np
import random
import pickle
from utils.config import SAVE_DIR_DATA

def simulate_data(subjects, nodes, views, sigma, mu):
    #Adapted from: https://github.com/basiralab/reproducibleFedGNN
    edges = int(nodes*(nodes-1)/2)
    adjs = []
    for subject in range(subjects):
        dist_mat = np.zeros((nodes,nodes,views)) # Initialize nxn matrix
        for view in range(views):
            dist_arr = np.random.normal(mu, sigma, edges)
            dist_list = dist_arr.tolist()
            k = 0
            for i in range(nodes):
                for j in range(nodes):
                    if i>j:
                        dist_mat[i,j,view] = dist_list[k]
                        dist_mat[j,i,view] = dist_mat[i,j,view]
                        k+=1
        adjs.append(dist_mat)
    return adjs
