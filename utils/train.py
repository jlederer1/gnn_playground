# utils/train.py 

"""
Training setup for Graph Neural Networks.
Includes data loading, model initialization, optimizer setup, epoch loop, batch processing and live plotting hooks. 
"""

# import time # TODO: timestamp logs checkpoints and results 
import torch 
from torch.optim import Adam, SGD 
from torch_geometric.loader import DataLoader
from livelossplot import PlotLosses  
from livelossplot.outputs import MatplotlibPlot
from tqdm import trange
import matplotlib.pyplot as plt


OPTIMIZERS = {
    'Adam': Adam,
    'SGD': SGD,
}

def batch_accuracy(model, output, batch):
    """
    Computes #correct and #total predictions for a batch, both node-level and graph-level.
    Returned as a tuple (correct, total).
    """
    if model.task == 'node':
        # Node-level accuracy
        mask = batch.train_mask if model.training else batch.test_mask
        predictions = output.argmax(dim=1)  # Get predicted class indices
        correct = (predictions[mask] == batch.y[mask]).sum().item()
        total = int(mask.sum().item())  # Number of nodes in the mask
    elif model.task == 'graph':
        if output.shape[1] == 1: # Graph-level accuracy - binary
            logits = output.view(-1)  # flat
            predictions = (torch.sigmoid(logits) > 0.5).long() # Convert to binary predictions
            correct = (predictions == batch.y.view(-1)).sum().item()
            total = batch.y.numel()  # Total number of graphs
        else:  # Graph-level accuracy - multiclass
            logits = output.view(-1, output.shape[1]) # [batch_size, num_classes]
            predictions = logits.argmax(dim=1)
            correct = (predictions == batch.y.view(-1)).sum().item()
            total = batch.y.numel()
    else:
        raise ValueError(f"Unsupported task type: {model.task}")
    
    return correct, total


def train_model(
    model, 
    train_loader: DataLoader, 
    epochs = 100,  
    lr = 0.001, 
    weight_decay = 0.0,
    optimizer_type = 'Adam', 
    device = None,
    val_loader = None
    ):
    """
    Train a GNN model and viz with livelossplot.
    
    Args:
        model: GNN model to train.
        train_loader: Training DataLoader.
        epochs: Number of training epochs, default 100.
        lr: Learning rate for the optimizer, default 0.001.
        weight_decay: Weight decay for the optimizer, default 0.0.
        optimizer_type: Name of the optimizer to use, default 'Adam'.
        device: Device for training, default CPU or GPU if available.
        val_loader: Optional validation DataLoader for evaluation during training.
    
    Returns:
       History dictionary with training and validation loss.
    """
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.to(device)
    
    # Initialize optimizer
    if optimizer_type in OPTIMIZERS:
        optimizer_class = OPTIMIZERS[optimizer_type]
        optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Supported optimizers are {list(OPTIMIZERS.keys())}.")
    
    # Use livelossplot only for CPU, GPUs dont like it...
    if device == 'cpu':
        mpl_plot = MatplotlibPlot() 
        liveloss = PlotLosses(outputs=[mpl_plot]) # We dont want unneccessary console outputs... 

    history = {
        'loss': [],
        'val_loss': [],
        'accuracy': [],
        'val_accuracy': []
    }
    
    # Training loop
    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            # model forward/backward
            output = model(batch)
            loss = model.loss(output, batch)  # loss bundled in GNN model
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() # TODO: * batch_size  
            c, t = batch_accuracy(model, output, batch)
            correct += c
            total += t
        
        epoch_loss /= len(train_loader.dataset)  
        train_accuracy = correct / max(total, 1)  # Avoid division by zero
        history['loss'].append(epoch_loss)
        history['accuracy'].append(train_accuracy)
    
        # Validation 
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = val_batch.to(device)
                    val_output = model(val_batch)
                    val_loss += model.loss(val_output, val_batch).item()
                    c, t = batch_accuracy(model, val_output, val_batch)
                    correct += c
                    total += t
                val_loss /= len(val_loader.dataset)  
                val_accuracy = correct / max(total, 1)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
        else:
            val_loss = None
            val_accuracy = None
        
        if device == 'cpu':
            # Update livelossplot
            logs = {
                'loss': epoch_loss,
                'accuracy': train_accuracy,
            }
            if val_loss is not None:
                logs['val_loss'] = val_loss
                logs['val_accuracy'] = val_accuracy  
            
            liveloss.update(logs)
            plt.close('all') # delete previous figure
            liveloss.draw()

        if epoch % 10 == 0 or epoch == epochs:
        
            msg = f"Epoch {epoch+1:03d}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {train_accuracy:.4f}"
            if val_loss is not None:
                msg += f", Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
            print(msg)
    
    return history