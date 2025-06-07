# gnn_competition.py

"""
This is the entry point for the CLI-based GNN-Playground.
It parses YAML configs or a textual TUI to select and run 2 GNN experiments at a time. 
It displays live tracking metrics and calls the certificate generator in the end. 
"""

import argparse, glob, os

parser = argparse.ArgumentParser(description="GNN-Playground CLI")
parser.add_argument("--config", help="path to YAML config")
args = parser.parse_args()

if args.config is None:
    # find all .yaml in configs/ and choose the newest by last modification time
    files = glob.glob("configs/*.yaml")
    args.config = max(files, key=os.path.getmtime)
    print(f"Using latest config: {args.config}")