# utils/tui.py 

"""
Implements the Text-based User Interface (TUI) for the GNN Playground.

It incrementally covers: 
- data.name [KarateClub | MUTAG]
- data.task [node | graph]
- data.device [auto | cuda | mps | cpu] (TODO: implement and test cuda and mps versions)
- data.random_seed [int]
The rest of the config will folllow later... (TODO)

Press X to exit and retur the configuration.
"""

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

from utils.config import DEFAULT, merge_configs, merge_configs, resolve_device


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
                if active == "device:": idx += 1    # dont know why... :(
                if active == "random_seed:": idx += 1   # dont know why... :(
                highlight = [int(idx)]  # Set the line number to highlight
                #raise ValueError(f"Check match: <{active}> in <{line.lstrip()}, highlight={highlight}>")
                break

    return Syntax("\n".join(lines), "yaml", theme="monokai" ,line_numbers=True, highlight_lines=highlight) 

# Map widged IDs (TUI components) to the associated config keys.
ID2KEY = {
    "pick_dataset": "name:", # leading spaces stripped from nested yaml structure lines..
    "pick_task": "task:",
    "pick_device": "device:",
    "pick_seed": "random_seed:",
}

class YamlPreview(Static):
    can_focus = True # to allow focus on the preview widget when scrolling through hyperparameter options

class TUIInterface(App): 
    CSS = """
    Screen { align: center middle;}
    #panel { border: round $accent; width: 80; height: 30; }
    #buttons { dock: bottom; height: 3; content-align: center middle; }
    Select { border: round $accent; width: 20; padding: 1; }
    Input { border: round $accent; width: 20; padding: 1; }
    Select:focus { border: heavy $accent; background: $boost; }
    Input:focus { border: heavy $accent; background: $boost; }
    #preview { border: round $accent; }
    #preview:focus { border: heavy $accent; background: $boost; }
    """

    BINDINGS = [
        #("tab", "focus_next", "Next"),
        ("X", "confirm", "Train Now"),
        ("escape", "quit", "Cancel"),
    ]

    def __init__(self):
        """TUI Setup"""
        super().__init__()
        self.config = merge_configs(DEFAULT, {})  # Start with default config

        # Options for Select widgets
        self.datasets = [("KarateClub", "KarateClub"), ("MUTAG", "MUTAG")]
        self.tasks = [("node", "node"), ("graph", "graph")] # always ("label", "value") pairs
        self.devices = [("auto", "auto"), ("cuda", "cuda"), ("mps", "mps"), ("cpu", "cpu")]

    def compose(self):
        """Declare simple sequential TUI hirarchy"""
        with Vertical(id="panel", classes="panel"):
            # Title
            yield Static("GNN Playground Configuration", classes="title")

            # Dataset and Task selection
            with Horizontal():
                yield Static("Dataset:")
                yield Select(options=self.datasets, id="pick_dataset")
                yield Static("Task:")
                yield Select(options=self.tasks, id="pick_task")

            # Device and Seed selection
            with Horizontal():
                yield Static("Device:")
                yield Select(options=self.devices, id="pick_device")
                yield Static("Seed:")
                yield Input(value=str(self.config["data"]["random_seed"]), id="pick_seed")
            
            # YAML preview
            self.preview = YamlPreview(yaml_block(self.config), id="preview", expand=True)
            yield self.preview

            # Action buttons 
            with Horizontal(id="buttons"):
                yield Button("X to Train", id="btn_train", variant="success")
                yield Button("ESC to cancel", id="btn_cancel", variant="error")
    
    def on_mount(self):
        """Initialize TUI with default values"""
        self.query_one("#pick_seed").value = str(self.config["data"]["random_seed"])
        self.query_one("#pick_dataset").value = self.config["data"]["name"]
        self.query_one("#pick_task").value = self.config["data"]["task"]
        self.query_one("#pick_device").value = self.config["data"]["device"]
        self.active_key = "pick_dataset"  # Current yaml key
        self.update_preview()  # Initial preview update

    def update_preview(self):
        # Reload TUI preview after each config update or focus event  
        self.preview.update(yaml_block(self.config, active=self.active_key))  # active=self.active_key))
    
    # def on_focus(self, event):
    #     """Handles focus events on yaml."""
    #     raise ValueError(f"Check focus event: {wid.id} vs {ID2KEY.keys()}") 

    #     wid = event.control
    #     if wid and wid.id in ID2KEY:
    #         self.active_key = ID2KEY[wid.id] # TODO: add sanity checks
    #     else:
    #         self.active_key = None 
    #     self.update_preview()
    
    # def on_blur(self, event):
    #     """Handles blur events on yaml."""
    #     self.active_key = None
    #     self.update_preview()  
    
    def on_select_changed(self, event):
        self.active_key = ID2KEY[event.select.id] # highlight line in yaml preview
        if event.select.id == "pick_dataset":
            self.config["data"]["name"] = event.value
            # align tasks and datasets for KarateClub and MUTAG
            if event.value == "KarateClub":
                self.config["data"]["task"] = "node"
                self.query_one("#pick_task").value = "node"
            elif event.value == "MUTAG":
                self.config["data"]["task"] = "graph"
                self.query_one("#pick_task").value = "graph"
        elif event.select.id == "pick_task":
            self.config["data"]["task"] = event.value
        elif event.select.id == "pick_device":
            self.config["data"]["device"] = event.value
        self.update_preview()  # Update YAML preview

    def on_input_changed(self, event): 
        # Handles events emitted by Textual's Input widget 
        self.active_key = ID2KEY[event.input.id]
        if event.input.id == "pick_seed":
            try:
                self.config["data"]["random_seed"] = int(event.value)
            except:
                event.input.value = str(self.config["data"]["random_seed"])
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