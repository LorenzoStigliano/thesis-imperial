
gcn_args = {
    "num_epochs":25, 
    "lr": 0.0001,
    "weight_decay":5e-4, 
    "hidden_dim":64,
    "dropout":0,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "model_name":"gcn",
    "evaluation_method": "model selection" # model selection or model assessment
}

gat_args = {
    "num_epochs":25, 
    "lr":0.0001, 
    "weight_decay":5e-4,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "hidden_dim":8,
    "nb_heads":8, # Attention heads
    "alpha":0.2, # Alpha for the leaky_relu.
    "dropout": 0.1,
    "model_name":"gat",
    "evaluation_method": "model assessment"
}

diffpool_args = {
    "num_epochs":25, 
    "lr":0.0001, 
    "weight_decay":5e-4,
    "threshold":"median", # Threshold the graph adjacency matrix. Possible values: no_threshold, median, mean
    "hidden_dim":126,
    "output_dim":256,
    "num_gc_layers":3, # Number of graph convolution layers before each pooling
    "assign_ratio":0.1, # Ratio of number of nodes in consecutive layers
    "num_pool":1, # Number of pooling layers
    "bn":True, #Whether batch normalization is used'
    "dropout": 0.1, 
    "linkpred": False, #Whether link prediction side objective is used
    "bias": True, #'Whether to add bias. Default to True.'
    "clip":2.0, #Gradient clipping
    "model_name":"diffpool",
    "evaluation_method": "model assessment"
}
