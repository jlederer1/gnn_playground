# utils/config.py 

"""
Utilities for loading and validating experiment configurations for GNN-Playground.
Shared scheema for gnn_competition.py and notebook application.
"""

from pathlib import Path
import yaml


DEFAULT = {
    "data": {
        "name": "MUTAG",
        "task": "graph",
        "device": "auto",
        "random_seed": 42
    },
    "model": {
        "num_layers": 3,
        "hidden_dim": 64,
        "dropout_value": 0.5,
        "layer_type": "GCN",
        "aggregator": "mean",
        "glob_pooler": "mean",
        "activation": "default", # according to papers of GNN layers
        "update_func": "MLP" # only for GINConv layer
    },
    "optimizer": {
        "type": "Adam",
        "lr": 0.001,
        "weight_decay": 0.0
    },
    "training": {
        "batch_size": 32,
        "epochs": 100,
        "val_split": 0.5
    }
}

def merge_configs(default, custom):
    """
    Merges default configuration with custom overrides.
    Custom config can override any default value.
    """
    merged = default.copy()
    for key, value in custom.items():
        if isinstance(value, dict) and key in merged:
            # recursive override call
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged

def resolve_device(device):
    """
    Turns "auto" into "cuda", "mps" or "cpu" based on availability.
    """
    if device == "auto":
        # yield correct device
        import torch
        return (
            "cuda" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available() 
            else "cpu"
        )
    # Or keep as is
    elif device in ["cpu", "cuda", "mps"]:
        return device
    # Sanity
    else: # device not in ["cpu", "cuda", "mps"]:
        raise ValueError(f"Use 'cpu', 'cuda', 'mps' or 'auto' for data.device.")

def load_config(config_path = "configs/example.yaml"):
    """
    Loads and validates the experiment configuration from a YAML file.
    Returns merged configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config {config_path} does not exist.")
    
    with config_path.open("r", encoding="utf-8") as file:
        try:
            custom_config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing config YAML {config_path}: {e}")
    
    # Merge with defaults
    merged_config = merge_configs(DEFAULT, custom_config)

    # Validation checks 
    if merged_config["data"]["name"] not in ["MUTAG", "KarateClub"]:
        raise ValueError(f"Use 'MUTAG' or 'KarateClub' for data.name.")
    if merged_config["data"]["task"] not in ["node", "graph"]:
        raise ValueError(f"Use 'node' or 'graph' for data.task.")
    if merged_config["model"]["layer_type"] not in ["GCN", "GAT", "GraphSAGE", "GIN"]:
        raise ValueError(f"Use 'GCN', 'GAT', 'GraphSAGE' or 'GIN' for model.layer_type.")
    if merged_config["model"]["activation"] not in ["relu", "tanh", "sigmoid", "leaky_relu", "elu", "default"]:
        raise ValueError(f"Use 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu' or 'default' for model.activation.")
    if merged_config["optimizer"]["type"] not in ["Adam", "SGD"]:
        raise ValueError(f"Use 'Adam' or 'SGD' for optimizer.type.")
    
    # Optional auto device selection
    merged_config["data"]["device"] = resolve_device(merged_config["data"]["device"])
    
    return merged_config