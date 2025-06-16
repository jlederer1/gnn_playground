# utils/tui.py 

"""
Implements the Text-based User Interface (TUI) for the GNN Playground.

It incrementally covers: 
- data.name [KarateClub | MUTAG | ENZYMES]
- data.task [node | graph]
- data.device [auto | cuda | mps | cpu] (TODO: implement and test cuda and mps versions)
- data.random_seed [int]
- model.num_layers [int]
- model.hidden_dim [int]
- model.dropout_value [float]
- model.layer_type [GCN | GAT | GraphSAGE | GIN]
- model.aggregator [mean | sum | max] (only for GraphSAGE)
- model.update_func [MLP | BLSTM] (only for GIN)
- model.glob_pooler [mean | add | max] (only for graph tasks)
- model.activation [default | relu | leaky_relu | sigmoid | tanh]
- optimizer.type [Adam | SGD]
- optimizer.lr [float]
- optimizer.weight_decay [float]
- training.batch_size [int]
- training.epochs [int]
- training.val_split [float]

Press X to exit and retur the configuration.
"""

import glob, os
from pathlib import Path
from rich.syntax import Syntax # For syntax highlighting of the YAML block
from textual.app import App, ComposeResult
from textual.widgets import Button # Fires an event on click 
from textual.widgets import Input # Text input widget for numerical configs
from textual.widgets import Select # Dropdown widget
from textual.widgets import Static # To display static text
from textual.containers import Vertical # Stacks layout container top to bottom
from textual.containers import  Horizontal # Stacks layout container left to right
import yaml

from utils.config import DEFAULT, merge_configs, merge_configs, resolve_device, load_config


def get_latest_config(path="configs/*.yaml"):
    files = glob.glob(path)
    if files:
        latest = max(files, key=os.path.getmtime)
        return load_config(latest)
    return None

def yaml_block(config, active=None):
    """
    Formats the configuration dictionary as a YAML block for TUI display.
    """
    yaml_string = yaml.dump(config, sort_keys=False, width=60)
    lines = yaml_string.splitlines()

    highlight = set()
    if active:
        for idx, line in enumerate(lines, start=0):
            # Highlight the active line
            if active in line.lstrip():
                if active == "device:": idx += 1    # dont know why exactly... :(
                if active == "random_seed:": idx += 1   # dont know why exactly... :(
                highlight = [int(idx)]  # Set the line number to highlight
                break

    return Syntax("\n".join(lines), "yaml", theme="monokai" ,line_numbers=True, highlight_lines=highlight) 

# Map widged IDs (TUI components) to the associated config keys.
ID2KEY = {
    "pick_dataset": "name:", # leading spaces stripped from nested yaml structure lines..
    "pick_task": "task:",
    "pick_device": "device:",
    "pick_seed": "random_seed:",

    "pick_num_layers": "num_layers:",
    "pick_hidden_dim": "hidden_dim:",
    "pick_dropout_value": "dropout_value:",
    "pick_layer_type": "layer_type:",
    "pick_aggregator": "aggregator:",
    "pick_update_func": "update_func:",
    "pick_glob_pooler": "glob_pooler:",
    "pick_activation": "activation:",

    "pick_optimizer": "type:",
    "pick_lr": "lr:",
    "pick_weight_decay": "weight_decay:",

    "pick_batch_size": "batch_size:",
    "pick_epochs": "epochs:",
    "pick_val_split": "val_split:",
}

class YamlPreview(Static):
    can_focus = True # to allow focus on the preview widget when scrolling through hyperparameter options

class TUIInterface(App): 
    CSS = """
    Screen { align: center middle;}
    #panel { border: round $accent; width: 100; height: 40; }
    #buttons { dock: bottom; height: 3; content-align: center middle; }
    Select { border: round $accent; width: 20; padding: 1; }
    Input { border: round $accent; width: 20; padding: 1; }
    Select:focus { border: heavy $accent; background: $boost; }
    Input:focus { border: heavy $accent; background: $boost; }
    #preview { border: round $accent; }
    #preview:focus { border: heavy $accent; background: $boost; }

    Horizontal { height: 1; }
    .label { width: 18; min-width: 2; height: 6; }
    .unit { width: 4; min-width: 2; height: 6; }
    .static { width: 12; min-width: 2; height: 6; }
    Select, Input { width: 1fr; height: 1fr; }
    """

    BINDINGS = [
        ("tab", "focus_next", "Next"), # tab through options 
        ("shift tab", "focus_previous", "Previous"), # tab back
        ("X", "confirm", "Train Now"), # X to confirm config and start training
        ("escape", "quit", "Cancel"), # Escape to cancel and exit
    ]

    def __init__(self):
        """TUI Setup"""
        super().__init__()
        # self.config = merge_configs(DEFAULT, {})  # Start with default config
        self.config = get_latest_config() 

        # Options for Select widgets
        self.datasets = [("KarateClub", "KarateClub"), ("MUTAG", "MUTAG"), ("ENZYMES", "ENZYMES")]
        self.tasks = [("node", "node"), ("graph", "graph")] # always ("label", "value") pairs
        self.devices = [("auto", "auto"), ("cuda", "cuda"), ("mps", "mps"), ("cpu", "cpu")]
        self.layer_types = [("GCN", "GCN"), ("GAT", "GAT"), ("GraphSAGE", "GraphSAGE"), ("GIN", "GIN")] 
        self.aggregators = [("mean", "mean"), ("sum", "sum"), ("max", "max")] # (max only for GraphSAGE)
        self.update_funcs = [("MLP", "MLP"), ("BLSTM", "BLSTM")] # (only for GIN)
        self.glob_poolers = [("mean", "mean"), ("add", "add"), ("max", "max")] # (only for graph tasks)
        self.activations = [("default", "default"), ("relu", "relu"), ("leaky_relu", "leaky_relu"),
                            ("sigmoid", "sigmoid"), ("tanh", "tanh")]
        self.optimizers = [("Adam", "Adam"), ("SGD", "SGD")] 
    
    def compose(self):
        """Declare simple sequential TUI hirarchy"""
        with Vertical(id="panel", classes="panel"):
            # Title
            yield Static("GNN Playground Configuration \n('Tab' or click to navigate, 'enter' to modify & type to input)\n", classes="title")

            # Dataset and Task selection
            with Horizontal():
                yield Static("Dataset:", classes="static")
                yield Select(options=self.datasets, prompt="Dataset:", allow_blank=False, id="pick_dataset", classes="label")
                yield Static("Task:", classes="static")
                yield Select(options=self.tasks, prompt="Task:", allow_blank=False, id="pick_task", classes="label")

            # Device and Seed selection
            with Horizontal():
                yield Static("Device:", classes="static")
                yield Select(options=self.devices, prompt="Device:", allow_blank=False, id="pick_device", classes="label")
                yield Static("Random Seed:", classes="static")
                yield Input(value=str(self.config["data"]["random_seed"]), id="pick_seed", classes="unit")
            
            # Model hyperparameters
            with Horizontal():
                yield Static("Num. Layers:", classes="static")
                yield Input(value=str(self.config["model"]["num_layers"]), id="pick_num_layers", classes="unit")
                yield Static("Hidden Dim:", classes="static")
                yield Input(value=str(self.config["model"]["hidden_dim"]), id="pick_hidden_dim", classes="unit")
                yield Static("Dropout Value:", classes="static")
                yield Input(value=str(self.config["model"]["dropout_value"]), id="pick_dropout_value", classes="unit")
            
            # Layer type specifications
            with Horizontal():
                yield Static("Layer Type:", classes="static")
                yield Select(options=self.layer_types, prompt="Layer Type:", allow_blank=False, id="pick_layer_type", classes="label")
                
                yield Static("Aggregator:", classes="static")
                yield Select(options=self.aggregators, prompt="Aggregator:", allow_blank=False, id="pick_aggregator", classes="label")
                yield Static("Update Func:", classes="static")
                yield Select(options=self.update_funcs, prompt="Update func:", allow_blank=False, id="pick_update_func", classes="label")
            
            # Pooling and activation
            with Horizontal():
                yield Static("Glob. Pooler:", classes="static")
                yield Select(options=self.glob_poolers, prompt="Glob. Pooler:", allow_blank=False, id="pick_glob_pooler", classes="label")
                yield Static("Activation:", classes="static")
                yield Select(options=self.activations, prompt="Activation:", allow_blank=False, id="pick_activation", classes="label"    )
            
            # Optimizer 
            with Horizontal():
                yield Static("Optimizer:", classes="static")
                yield Select(options=self.optimizers, prompt="Optimizer:", allow_blank=False, id="pick_optimizer", classes="label")
                yield Static("Learning Rate:", classes="static")
                yield Input(value=str(self.config["optimizer"]["lr"]), id="pick_lr", classes="unit")
                yield Static("Weight Decay:", classes="static")
                yield Input(value=str(self.config["optimizer"]["weight_decay"]), id="pick_weight_decay", classes="unit")
            
            # Training parameters
            with Horizontal():
                yield Static("Batch Size:", classes="static")
                yield Input(value=str(self.config["training"]["batch_size"]), id="pick_batch_size", classes="unit")
                yield Static("Epochs:", classes="static")
                yield Input(value=str(self.config["training"]["epochs"]), id="pick_epochs", classes="unit")
                yield Static("Val. Split:", classes="static")
                yield Input(value=str(self.config["training"]["val_split"]), id="pick_val_split", classes="unit")
            
            # YAML preview
            self.preview = YamlPreview(yaml_block(self.config), id="preview", expand=True)
            yield self.preview

            # Action buttons 
            with Horizontal(id="buttons"):
                yield Button("'X' to Train", id="btn_train", variant="success")
                yield Button("'ESC' to cancel", id="btn_cancel", variant="error")
    
    def on_mount(self):
        """Initialize TUI with default values"""
        self.query_one("#pick_seed").value = str(self.config["data"]["random_seed"])
        self.query_one("#pick_dataset").value = self.config["data"]["name"] # initial focus (TODO)
        self.query_one("#pick_task").value = self.config["data"]["task"]
        self.query_one("#pick_device").value = self.config["data"]["device"]

        self.query_one("#pick_num_layers").value = str(self.config["model"]["num_layers"])
        self.query_one("#pick_hidden_dim").value = str(self.config["model"]["hidden_dim"])
        self.query_one("#pick_dropout_value").value = str(self.config["model"]["dropout_value"])
        self.query_one("#pick_layer_type").value = self.config["model"]["layer_type"]
        self.query_one("#pick_aggregator").value = self.config["model"]["aggregator"]
        self.query_one("#pick_update_func").value = self.config["model"]["update_func"]
        self.query_one("#pick_glob_pooler").value = self.config["model"]["glob_pooler"]
        self.query_one("#pick_activation").value = self.config["model"]["activation"]
        self.query_one("#pick_optimizer").value = self.config["optimizer"]["type"]
        self.query_one("#pick_lr").value = str(self.config["optimizer"]["lr"])
        self.query_one("#pick_weight_decay").value = str(self.config["optimizer"]["weight_decay"])
        self.query_one("#pick_batch_size").value = str(self.config["training"]["batch_size"])
        self.query_one("#pick_epochs").value = str(self.config["training"]["epochs"])
        self.query_one("#pick_val_split").value = str(self.config["training"]["val_split"])

        self.active_key = ID2KEY["pick_dataset"]  # Current yaml key
        self.update_preview()  # Initial preview update

    def update_preview(self):
        # Reload TUI preview after each config update or focus event  
        self.preview.update(yaml_block(self.config, active=self.active_key))  
    
    def on_select_changed(self, event):
        """Handles events emitted by Textual's Select widget."""
        # Data selection events
        self.active_key = ID2KEY[event.select.id] # highlight line in yaml preview
        if event.select.id == "pick_dataset":
            self.config["data"]["name"] = event.value
            # align tasks and datasets for KarateClub, MUTAG, ENZYMES
            if event.value == "KarateClub":
                self.config["data"]["task"] = "node"
                #self.query_one("#pick_task").value = "node"
            elif event.value in ["MUTAG", "ENZYMES"]:
                self.config["data"]["task"] = "graph"
                #self.query_one("#pick_task").value = "graph"
        elif event.select.id == "pick_task":
            self.config["data"]["task"] = event.value
            # align datasets with task type
            if event.value == "node" and self.config["data"]["name"] != "KarateClub":
                self.config["data"]["name"] = "KarateClub"
                #self.query_one("#pick_dataset").value = "KarateClub"
            elif event.value == "graph" and self.config["data"]["name"] == "KarateClub":
                self.config["data"]["name"] = "MUTAG"
                #self.query_one("#pick_dataset").value = "MUTAG"
        # Device and Seed selection
        elif event.select.id == "pick_device":
            self.config["data"]["device"] = event.value
        # Model hyperparameter events
        elif event.select.id == "pick_layer_type":
            self.config["model"]["layer_type"] = event.value
        elif event.select.id == "pick_aggregator":
            self.config["model"]["aggregator"] = event.value
        elif event.select.id == "pick_update_func":
            self.config["model"]["update_func"] = event.value
        elif event.select.id == "pick_glob_pooler":
            self.config["model"]["glob_pooler"] = event.value
        elif event.select.id == "pick_activation":
            self.config["model"]["activation"] = event.value
        # Optimizer selection events
        elif event.select.id == "pick_optimizer":
            self.config["optimizer"]["type"] = event.value
        
        self.update_preview()  # Update YAML preview

        # Enable/disable options based on task and layer type
        self.query_one("#pick_aggregator").disabled   = (self.config["model"]["layer_type"] != "GraphSAGE")
        self.query_one("#pick_update_func").disabled  = (self.config["model"]["layer_type"] != "GIN")
        self.query_one("#pick_glob_pooler").disabled  = (self.config["data"]["task"] != "graph")

    def on_input_changed(self, event): 
        """Handles events emitted by Textual's Input widget """
        self.active_key = ID2KEY[event.input.id]
        if event.input.id == "pick_seed":
            try:
                self.config["data"]["random_seed"] = int(event.value)
            except:
                event.input.value = str(self.config["data"]["random_seed"])
        # Numeric model hyperparameter input events
        elif event.input.id == "pick_num_layers":
            self.config["model"]["num_layers"] = int(event.value)
        elif event.input.id == "pick_hidden_dim":
            self.config["model"]["hidden_dim"] = int(event.value)
        elif event.input.id == "pick_dropout_value":
            self.config["model"]["dropout_value"] = float(event.value)
        # Optimizer hyperparameter input events
        elif event.input.id == "pick_lr":
            self.config["optimizer"]["lr"] = float(event.value)
        elif event.input.id == "pick_weight_decay":
            self.config["optimizer"]["weight_decay"] = float(event.value)
        # Training hyperparameter input events  
        elif event.input.id == "pick_batch_size":
            self.config["training"]["batch_size"] = int(event.value)
        elif event.input.id == "pick_epochs":
            self.config["training"]["epochs"] = int(event.value)
        elif event.input.id == "pick_val_split":
            self.config["training"]["val_split"] = float(event.value)

        self.update_preview()  # Update YAML preview
    
    def action_confirm(self):
        """Confirms config choice on btn_train click."""
        self.exit(self.config)
    
    def on_button_pressed(self, event):
        """Handles button clicks."""
        # Selection successful
        if event.button.id == "btn_train":
            self.action_confirm()
        # Selection aborted
        else:
            self.exit(None)
    
# The only public method to run TUI
def run_tui():
    """Run the TUI application and return the final configuration."""
    config = TUIInterface().run()
    if config is None:
        print("TUI cancelled. No configuration returned.")
        return None
    # check device type
    config["data"]["device"] = resolve_device(config["data"]["device"])
    return config