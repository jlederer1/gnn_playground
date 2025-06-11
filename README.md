# GNN-Playground

This is a **hands-on sandbox** for experimenting with Graph Neural Networks (GNNs) on tiny cpu-friendly datasets.
You can customize architecture and training hyperparameters in a single YAML file and then compare two models side-by-side. 
The repository provides you with 3 alternative to proceed:

- **Jupyter Notebook** (`notebooks/playground.ipynb`)
- **Terminal Script** (`gnn_competition.py --config configs/example.yaml`)
- **Textual User Interface (TUI)** (`gnn_competition.py --tui`)

--- 

## 🌟 Features 
- **Three small datasets** (MUTAG, ENZYME & KarateClub) for graph and node-based prediction tasks.
- **Multiple layer types**: GCN, GAT, GraphSAGE, GIN
- **Live plots** of loss and accuracy
- **PDF certificates** summarizing your interactions with the demo and results.
- **CPU-only** (supports `cpu`, `cuda` or `mps`)
- **Few dependancies** (PyTorch, PyG, pyyaml, matplotlib, textual, livelossplot, rich, fpdf2)

--- 

## 🚀 Quickstart

1. **Clone & install**
```bash
# Load repository
git clone https://github.com/jlederer1/gnn_playground.git
cd gnn_playground
# create and activate new environment
python3 -m venv .venv
source .venv/bin/activate       # on Linux/MaxOS
.\.venv\Scrips\activate         # in Windows shell
# Install dependancies
pip install -r requirements.txt
```
3. **Run in notebook**
```bash
jupyter notebook notebooks/playground.ipynb
```
3. **Edit a new configuration**
```bash
# copies example config and creates new_experiment_XX.yaml
python utils/new_experiment.py 
# then adjust hyperparameters like dataset, task, layers, lr, etc.
```
4. **Run via Command Line Interface (CLI)**
```bash
python gnn_competition.py --config configs/example.yaml
```
`--config` is optional, latest experiment config is chosen at default.

5. **Run in TUI**
```bash
python gnn_competition.py --tui
```
Note: The TUI is not supported for jupyter notebook.
After completion you can check `certificates/` to view a summary of the results of your interactions :)

---

## 📂 Project structure

...

--- 

## 🤝 Contributing 

Feel free to open issues and PRs to add new datasets, layers and ideas! 