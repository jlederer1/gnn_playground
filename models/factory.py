# models/factory.py

"""
This is the layer factory to dynamically construct GNN models.
It maps "layer_type" variables in configs to a corresponding PyTorch Geometric layer class to use in the competition
"""

from models.layers import GCNWrapper, GraphSAGEWrapper, GATWrapper, GINWrapper


# Factory dictionary to map layer names to their corresponding wrapper classes
LAYERS = {
    "GCN": GCNWrapper,              # Graph Convolutional Network (Kipf & Welling, 2016)
    "GAT": GATWrapper,              # Graph Attention Network (Veličković et al., 2017)
    "GraphSAGE": GraphSAGEWrapper,  # SAmple and aggreGatE framework by Hamilton et al. (2017)
    "GIN": GINWrapper               # Graph Isomorphism Network (Xu et al., 2019)
}

def get_layer(name, **kwargs):
    """
    Retrieve a GNN layer instance by name. 
    """
    return LAYERS[name](**kwargs)
