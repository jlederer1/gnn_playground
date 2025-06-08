# gnn_competition.py

"""
This is the entry point for the CLI-based GNN-Playground.
It parses YAML configs or a textual TUI to select and run 2 GNN experiments at a time. 
It displays live tracking metrics and calls the certificate generator in the end. 
"""

import argparse, glob, os
from utils.config import load_config
from utils.data import load_node_dataset, load_graph_dataset
from models.net import GNNModel
from utils.train import train_model
from torch_geometric.loader import DataLoader

def main():
    parser = argparse.ArgumentParser(description="GNN-Playground CLI")
    parser.add_argument("--config", help="path to YAML config")
    args = parser.parse_args()

    if args.config is None:
        # find all .yaml in configs/ and choose the newest by last modification time
        files = glob.glob("configs/*.yaml")
        args.config = max(files, key=os.path.getmtime)
        print(f"Using latest config: {args.config}")

    # DATA
    config = load_config(args.config)

    if config["data"]["task"] == "node":
        data = load_node_dataset(config["data"]["name"], seed=config["data"]["random_seed"])
        train_loader = DataLoader([data], batch_size=1, shuffle=False)
        val_loader = DataLoader([data], batch_size=1, shuffle=False) # val_mask in model.loss() will be used here
    else: 
        train_loader, val_loader = load_graph_dataset(
            config["data"]["name"], 
            split_ratio=config["training"]["val_split"],
            batch_size=config["training"]["batch_size"],
            seed=config["data"]["random_seed"]
        )
    
    # MODEL 
    in_dim = train_loader.dataset[0].num_node_features
    output_dim = data.y.max().item() + 1 if config["data"]["task"] == "node" else 1
    model = GNNModel(
        in_dim=in_dim,
        hidden_dim=config["model"]["hidden_dim"],
        out_dim=output_dim,
        num_layers=config["model"]["num_layers"],
        layer_type=config["model"]["layer_type"],
        dropout_value=config["model"]["dropout_value"],
        glob_pooler=config["model"]["glob_pooler"],
        task=config["data"]["task"],
        aggregator=config["model"]["aggregator"],
        activation=config["model"]["activation"],
        update_func=config["model"]["update_func"],
    )

    # TRAINING 
    train_model(
        model=model,
        train_loader=train_loader,
        epochs=config["training"]["epochs"],
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
        optimizer_type=config["optimizer"]["type"],
        device=config["data"]["device"],
        val_loader=val_loader 
    )

if __name__ == "__main__":
    main()