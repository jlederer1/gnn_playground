# models/net.py 

"""
GNN model classes for the GNN Playground.
"""

import torch 
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from models.factory import get_layer

POOLERS = {
    "mean": global_mean_pool,
    "add": global_add_pool,
    "sum": global_add_pool,  # for confusion 
    "max": global_max_pool,
}

class GNNModel(nn.Module):
    """
    Base class for GNN models. 
    A generic GNN consisting of a stack of layers and a task-specific head.
    Layers types and pooling method are based on the config.

    Args:
        in_dim (int): Input feature size per node.
        hidden_dim (int): Hidden embedding size.
        out_dim (int): Number of output classes on node or graph-level.
        num_layers (int): How many GNN layers to stack
        layer_type (str): Type of GNN layer ('GCN', 'GraphSAGE', 'GAT', 'GIN').
        dropout_value (float): Dropout probability each layer.
        glob_pooler (str): Global pooling method ('mean', 'max', 'add').
        task (str): Task type ('node' or 'graph').
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, layer_type, dropout_value, glob_pooler, task, **layer_kwrds):
        super().__init__()
        self.task = task
        self.dropout_value = dropout_value

        # Collect dimensionalities of the node features at each layer
        dims = [in_dim] + [hidden_dim] * num_layers # only head maps to out_dim
        # Instantiate all layers
        self.layers = nn.ModuleList([
            get_layer(
                layer_type, 
                in_dim=dims[i], 
                out_dim=dims[i + 1], 
                dropout_value=dropout_value,
                **layer_kwrds)
            for i in range(num_layers)
        ])
        
        # MLP classification-head 
        if task == "graph":
            self.pooling = POOLERS[glob_pooler]
            self.head = nn.Linear(hidden_dim, out_dim)
        elif task == "node":
            self.head = nn.Linear(hidden_dim, out_dim)
        else:
            raise ValueError(f"Choose 'node' or 'graph' as task type.")

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # run through all layers
        for layer in self.layers:
            x = layer(x, edge_index)
        
        if self.task == "graph":
            # Apply global pooling to get a graph-level representation
            if hasattr(data, 'batch'): # and isinstance(data, torch_geometric.data.batch.Batch):
                x = self.pooling(x, data.batch)
            else: 
                x = x.mean(dim=0, keepdim=True) # [batch_size, hidden_dim]

        if self.dropout_value > 0:
            x = F.dropout(x, p=self.dropout_value, training=self.training) 
        return self.head(x)
    
    def loss(self, pred, data):
        """
        Compute the CE loss based on the task type.

        Args:
            pred (torch.Tensor): Model predictions.
            data (torch_geometric.data.Data): Input data with labels.
        Returns:
            torch.Tensor: Computed loss value.
        """
        if self.task == "node":
            if self.training:
                mask = data.train_mask
            else:
                mask = data.test_mask 
            return F.cross_entropy(pred[mask], data.y[mask]) # expects [N, C]
        
        elif self.task == "graph":
            # use flat prediction tensor and casted targets 
            return F.binary_cross_entropy_with_logits(pred.view(-1), data.y.float()) 
        else:
            raise ValueError(f"Unsupported task type: {self.task}")