import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        run,
        norm_type="none",
    ):
        torch.manual_seed(run)
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.is_trained = False
        self.run = run

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = F.relu(h)
                h = self.dropout(h)
        h = torch.reshape(h, (1, 1)) 
        if self.is_trained:
          linear_layer = self.layers[0]
          w_dict = {"w": linear_layer.weight}
          with open("mlp_"+str(self.run)+"_W.pickle", 'wb') as f:
            pickle.dump(w_dict, f)
          self.is_trained = False
          print("MLP Weights are saved:")
          print(linear_layer.weight)
        return h_list, h
    
    def loss(self, pred, label, type='softmax'):
        return F.binary_cross_entropy(pred.float(), label.float(), reduction='mean')
