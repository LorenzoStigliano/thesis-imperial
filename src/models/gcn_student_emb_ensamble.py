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

class GCN_STUDENT_ENSAMBLE(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, seed, run, number, total_number, dataset):
        torch.manual_seed(seed*10+number)
        super(GCN_STUDENT_ENSAMBLE, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nclass)
        self.LinearLayer = nn.Linear(nfeat,1)
        self.is_trained = False
        self.number = number
        self.run = run
        self.total_number=total_number
        self.dataset = dataset

    def forward(self, x, adj):
        node_embeddings = F.relu(self.gc1(x, adj))
        #x = F.dropout(x, self.dropout, training=self.training)
        #x = self.gc2(x, adj)
        x = F.log_softmax(node_embeddings, dim=1)
        x = self.LinearLayer(torch.transpose(x,0,1))
        
        if self.is_trained:
          w_dict = {"w": self.LinearLayer.weight}
          with open(f"gcn_student_emb_ensamble_{self.total_number}_number_{self.number}_run_{self.run}_{self.dataset}_W.pickle", 'wb') as f:
            pickle.dump(w_dict, f)
          self.is_trained = False
          print("GCN Weights are saved:")
          print(self.LinearLayer.weight)
        
        x = torch.transpose(x,0,1)
        return x, node_embeddings
    
    def loss(self, pred, label, type='softmax'):
        return F.cross_entropy(pred, label, reduction='mean')