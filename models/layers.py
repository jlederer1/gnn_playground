# models/layers.py 

"""
This script holds custom GNN layer implementations for the GNN-Playground.
"""

import torch


class GNNLayer(torch.nn.Module):
    """ Base class for GNN layers. """
    def forward(self, x, edge_index):
        # return updeted representations
        raise NotImplementedError("Implement GNNLayer.")

class GCNWrapper(GNNLayer):
    """ 
    Graph Convolutional Network (Kipf & Welling, 2016) 
    Applies symmetric normalization, linear transform, ReLU, and optional dropout.
    """
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        raise NotImplementedError("Implement GCNWrapper.")

    def forward(self, x, edge_index):
        raise NotImplementedError("Implement GCNWrapper.")

class GraphSAGEWrapper(GNNLayer):
    """ 
    GraphSAGE layer (Hamilton et al., 2017). 
    Samples and aggregates neighbor features using mean, sum, or max, then applies linear + ReLU + dropout.
    """
    def __init__(self, in_dim, out_dim, aggregator="mean", dropout=0.0):
        super().__init__()
        raise NotImplementedError("Implement GraphSAGEWrapper.")

    def forward(self, x, edge_index):
        raise NotImplementedError("Implement GraphSAGEWrapper.")

class GATWrapper(GNNLayer):
    """ 
    Graph Attention Network layer (Veličković et al., 2018). 
    Uses self-attention on neighborhoods, supports multi-heads attention.
    """
    def __init__(self, in_dim, out_dim, heads=1, dropout=0.0):
        super().__init__()
        raise NotImplementedError("Implement GraphSAGEWrapper.")

    def forward(self, x, edge_index):
        raise NotImplementedError("Implement GraphSAGEWrapper.")
    
class GINWrapper(GNNLayer):
    """ 
    Graph Isomorphism Network layer (Xu et al., 2019). 
    Sum aggregation followed by MLP update with learnable epsilon.
    """
    def __init__(self, in_dim, out_dim, eps=0.0, dropout=0.0):
        super().__init__()
        raise NotImplementedError("Implement GraphSAGEWrapper.")

    def forward(self, x, edge_index):
        raise NotImplementedError("Implement GraphSAGEWrapper.")