# models/layers.py 

"""
This script holds custom GNN layer implementations for the GNN-Playground.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv


ACTIVATION_FUNCTIONS = {
    "relu": F.relu,
    "elu": F.elu,
    "leaky_relu": F.leaky_relu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
}

ACTIVATION_MODULES = {
    "relu": torch.nn.ReLU(),
    "elu": torch.nn.ELU(),
    "leaky_relu": torch.nn.LeakyReLU(),
    "sigmoid": torch.nn.Sigmoid(),
    "tanh": torch.nn.Tanh()
}

class GNNLayer(torch.nn.Module):
    """ Base class for GNN layers. """
    def forward(self, x, edge_index):
        # return updeted representations
        raise NotImplementedError("Use provided GNN Wrapper classes instead.")

class GCNWrapper(GNNLayer):
    """ 
    Graph Convolutional Network (Kipf & Welling, 2017) 
    Applies symmetric normalization, linear transform, ReLU, and optional dropout.
    """
    def __init__(self, in_dim, out_dim, dropout_value=0.0, activation="default"):
        super().__init__()
        self.conv = GCNConv(in_dim, out_dim)
        self.dropout_value = dropout_value
        self.activation = ACTIVATION_FUNCTIONS.get(activation, F.relu)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.activation(x) 
        if self.dropout_value > 0:
            x = F.dropout(x, p=self.dropout_value, training=self.training)
        return x

class GraphSAGEWrapper(GNNLayer):
    """ 
    GraphSAGE layer (Hamilton et al., 2018). 
    Samples and aggregates neighbor features using mean, sum, or max, then applies linear + ReLU + dropout.
    """
    def __init__(self, in_dim, out_dim, aggregator="mean", dropout_value=0.0, activation="default"):
        super().__init__()
        self.dropout_value = dropout_value
        self.conv = SAGEConv(in_dim, out_dim, aggr=aggregator) # mean, max or lstm
        self.activation = ACTIVATION_FUNCTIONS.get(activation, F.relu)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.activation(x)  
        if self.dropout_value > 0:
            x = F.dropout(x, p=self.dropout_value, training=self.training)
        return x

class GATWrapper(GNNLayer):
    """ 
    Graph Attention Network layer (Veličković et al., 2018). 
    Uses self-attention on neighborhoods, supports multi-heads attention.
    """
    def __init__(self, in_dim, out_dim, heads=1, dropout_value=0.0, activation="default"):
        super().__init__()
        self.conv = GATConv(in_dim, out_dim, heads=heads, dropout=dropout_value)
        self.dropout_value = dropout_value
        self.activation = ACTIVATION_FUNCTIONS.get(activation, F.elu)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.activation(x)  # default ELU
        if self.dropout_value > 0:
            x = F.dropout(x, p=self.dropout_value, training=self.training)
        return x
    
class BLSTM(torch.nn.Module):
    """ 
    Bidirectional LSTM layer for sequence modeling. 
    Applied in the GINWrapper for node feature updates.
    """
    def __init__(self, in_dim, output_dim):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            in_dim, 
            output_dim // 2, 
            num_layers=1, 
            bidirectional=True, 
            batch_first=True
        )

    def forward(self, x):
        # x should hve shape [num_nodes, dim] treated as one batch
        output, _ = self.lstm(x.unsqueeze(0))   # [1, num_nodes, output_dim]
        output = output.squeeze(0)              # [num_nodes, output_dim]
        return output

class GINWrapper(GNNLayer):
    """ 
    Graph Isomorphism Network layer (Xu et al., 2019). 
    Sum aggregation followed by MLP update with learnable epsilon.
    """
    def __init__(self, in_dim, out_dim, eps=0.0, dropout_value=0.0, activation="default", update_func="MLP"):
        super().__init__()
        self.activation_module = ACTIVATION_MODULES.get(activation, torch.nn.ReLU())
        # Choose learnable update function
        if update_func == "MLP":
            # the original implementation, 2-layer MLP
            updater = torch.nn.Sequential(
                torch.nn.Linear(in_dim, out_dim),
                self.activation_module, 
                torch.nn.Linear(out_dim, out_dim)
            )
        elif update_func == "BLSTM":
            # Bidirectional LSTM for node feature updates
            updater = BLSTM(in_dim, out_dim)
        else: 
            raise ValueError(f"Use 'MLP' or 'BLSTM' as update_func.")

        self.conv = GINConv(updater, eps=eps, train_eps=True)
        self.dropout_value = dropout_value

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.activation_module(x)
        if self.dropout_value > 0:
            x = F.dropout(x, p=self.dropout_value, training=self.training)
        return x