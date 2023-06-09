# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 21:17:31 2020
@author: Mohammed Amine
"""

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN3(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, run, dataset):
        torch.manual_seed(run)
        super(GCN3, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        #for each class we get a score and then softmax over the classes 
        self.LinearLayer = nn.Linear(nfeat,1)
        self.is_trained = False
        self.run = run
        self.dataset = dataset

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = self.gc3(x, adj)
        x = F.log_softmax(x, dim=1)
        x = self.LinearLayer(torch.transpose(x,0,1))
        
        if self.is_trained:
          w_dict = {"w": self.LinearLayer.weight}
          with open("gcn_"+str(self.run)+"_"+str(self.dataset)+"_W.pickle", 'wb') as f:
            pickle.dump(w_dict, f)
          self.is_trained = False
          print("GCN Weights are saved:")
          print(self.LinearLayer.weight)
        
        x = torch.transpose(x,0,1)
        return x
    
    def loss(self, pred, label, type='softmax'):
        return F.cross_entropy(pred, label, reduction='mean')