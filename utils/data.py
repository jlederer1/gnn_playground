# utils/data.py 

"""
Data utilities for GNN-Playground.
"""

import random 
import numpy as np
import torch
from torch_geometric.datasets import KarateClub, TUDataset 
# TODO: Check applicability of other options 
# like Planetoid, TUDataset, Amazon, Coauthor
from torch_geometric.loader import DataLoader
from pathlib import Path


def set_seed(seed):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed at {seed}")

def load_node_dataset(name, root='data/', seed=None, store=False):
    """
    Loads a node-level dataset by name.
    Returns a single graph as torch_geometric.data object.
    Contains x, edge_index, y, train_mask, and test_mask.
    """
    if name == 'KarateClub':
        dataset = KarateClub() # holds one PyTorch Geometric Data object with 34 nodes and 78 edges
        data = dataset[0] 
    else:
        raise ValueError(f"Use supported dataset (KarateClub)")
    
    # Simple train/test split on nodes
    if seed: 
        set_seed(seed)  # Ensure reproducibility
    num_nodes = data.num_nodes
    indices = list(range(num_nodes))
    random.shuffle(indices)
    split = int(0.5 * num_nodes)  # 80% for training, 20% for testing
    train_indices = indices[:split]
    test_indices = indices[split:]
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[train_indices] = True
    data.test_mask[test_indices] = True

    # Save datasplit
    if store: 
        root_dir = Path(__file__).resolve().parent.parent / root # data root directory
        root_dir.mkdir(parents=True, exist_ok=True)
        cache_filename = root_dir / f"{name}_node_data.pt"
        torch.save(data, cache_filename)
        print(f"Saved node dataset to {cache_filename}")

    return data

def load_graph_dataset(name, root='data/', split_ratio=0.8, batch_size=32, shuffle=True, seed=None, store=False):
    """
    Loads a graph-level dataset by name.
    Splits the dataset into training and validation sets and batches.
    Returns two PyTorch Geometric dataloader objects.
    """
    root_dir = Path(__file__).resolve().parent.parent / root # data root directory
    train_cache = root_dir / f"{name}_train.pt"
    test_cache = root_dir / f"{name}_test.pt"

    # Load cached datasets if available
    if store and train_cache.exists() and test_cache.exists():
        train_dataset = torch.load(train_cache)
        test_dataset = torch.load(test_cache)

    # or load a new split
    else:
        if name == 'MUTAG':
            dataset = TUDataset(root=root, name=name)  
        else:
            raise ValueError(f"Use supported dataset (MUTAG, ENZYMES)")
        
        if seed:
            set_seed(seed)
        if shuffle:
            dataset = dataset.shuffle()
        
        # reproducible split
        train_size = int(len(dataset) * split_ratio)
        train_dataset, test_dataset = dataset[:train_size], dataset[train_size:]

        # Save datasplit
        if store:
            train_cache.parent.mkdir(parents=True, exist_ok=True)
            test_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(train_dataset, train_cache)
            torch.save(test_dataset, test_cache)
            print(f"Saved graph datasets to {train_cache} and {test_cache}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size, 
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if seed else None
    ) 
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False)
    
    return train_loader, test_loader