# models/layers.py 

"""
This script holds custom GNN layer implementations for the GNN-Playground.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv


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
    def __init__(self, in_dim, out_dim, dropout_value=0.0):
        super().__init__()
        self.conv = GCNConv(in_dim, out_dim)
        self.dropout_value = dropout_value

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.relu(x)
        if self.dropout_value > 0:
            x = F.dropout(x, p=self.dropout_value, training=self.training)
        return x

class GraphSAGEWrapper(GNNLayer):
    """ 
    GraphSAGE layer (Hamilton et al., 2018). 
    Samples and aggregates neighbor features using mean, sum, or max, then applies linear + ReLU + dropout.
    """
    def __init__(self, in_dim, out_dim, aggregator="mean", dropout_value=0.0):
        super().__init__()
        self.dropout_value = dropout_value
        self.conv = SAGEConv(in_dim, out_dim, aggr=aggregator) # mean, max or lstm

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.relu(x)
        if self.dropout_value > 0:
            x = F.dropout(x, p=self.dropout_value, training=self.training)
        return x

class GATWrapper(GNNLayer):
    """ 
    Graph Attention Network layer (Veličković et al., 2018). 
    Uses self-attention on neighborhoods, supports multi-heads attention.
    """
    def __init__(self, in_dim, out_dim, heads=1, dropout_value=0.0):
        super().__init__()
        self.conv = GATConv(in_dim, out_dim, heads=heads, dropout=dropout_value)
        self.dropout_value = dropout_value

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.elu(x)
        if self.dropout_value > 0:
            x = F.dropout(x, p=self.dropout_value, training=self.training)
        return x
    
class GINWrapper(GNNLayer):
    """ 
    Graph Isomorphism Network layer (Xu et al., 2019). 
    Sum aggregation followed by MLP update with learnable epsilon.
    """
    def __init__(self, in_dim, out_dim, eps=0.0, dropout_value=0.0):
        super().__init__()
        # Learnable update function
        mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(out_dim, out_dim)
        )
        self.conv = GINConv(mlp, eps=eps, train_eps=True)
        self.dropout_value = dropout_value

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        if self.dropout_value > 0:
            x = F.dropout(x, p=self.dropout_value, training=self.training)
        return x