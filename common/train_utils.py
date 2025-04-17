# common/train_utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# --- Import necessary scheduler ---
from torch.optim.lr_scheduler import _LRScheduler # Use base class for type hinting
from torch.utils.data import DataLoader
from common.utils import get_device
from typing import Tuple, Optional, Callable # Import Optional
from tqdm import tqdm


def train_epoch(
        model: nn.Module,
        data_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: Callable, # to keep the same functionalities as before
) -> Tuple[float, float]:
    model.train()
    device = get_device()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    # Use tqdm description for clarity
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc="Training Epoch")
    for batch_idx, (data, target) in pbar:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # Ensure model outputs log-probabilities if using nll_loss this is modified now
        loss =  criterion(output, target) # for easy config between NLL and CrossEntropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        batch_correct = (output.argmax(1) == target).sum().item()
        batch_samples = data.size(0)

        total_loss += batch_loss * batch_samples
        total_correct += batch_correct
        total_samples += batch_samples

        # Update tqdm progress bar
        pbar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{batch_correct/batch_samples:.4f}")

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: Optional[Callable] = None, # Make criterion optional
) -> Tuple[float, float]:
    model.eval()
    device = get_device()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    # Default to NLLLoss if no criterion is provided
    if criterion is None:
        criterion = nn.NLLLoss()

    with torch.no_grad():
        pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc="Testing")
    for batch_idx, (data, target) in pbar:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target) # Assuming model outputs log_softmax

        total_loss += loss.item() * data.size(0)
        total_correct += (output.argmax(1) == target).sum().item()
        total_samples += data.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


# --- Modified train function ---
def train(
        model: nn.Module,
        train_loader: DataLoader, # Renamed for clarity
        optimizer: optim.Optimizer,
        epochs: int = 10,
        scheduler: Optional[_LRScheduler] = None, # Add optional scheduler argument
        test_loader: Optional[DataLoader] = None, # Optional test_loader for evaluation per epoch
        loss_type: str = 'nll' # default is nll to keep the same functionalities as before! wow omg
) -> None:
    print("Starting Training with ltyp...")
    model.to(get_device())

    # fpr the loss functions
    if loss_type.lower() == 'nll':
        criterion = nn.NLLLoss()
        print("Using NLL")
    elif loss_type.lower() == 'cross_entropy' or loss_type.lower() == 'ce':
        criterion = nn.CrossEntropyLoss()
        print("Using CrossEntropyLoss")
    else:
        raise ValueError(f"Brr Error loss type: {loss_type}")
    
        # for plotting 
    training_plot={
        'train_loss':[],
        'train_acc':[],
        'test_loss':[],
        'test_acc':[],
        'lr': []
    }


    for epoch in range(1, epochs + 1): # Start epoch count from 1
        # --- Train for one epoch ---
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)

        # --- Call scheduler.step() after the epoch ---
        current_lr = optimizer.param_groups[0]['lr'] # Get LR *before* scheduler steps
        if scheduler:
            scheduler.step() # Step the scheduler

        # --- Print epoch summary ---
        print(f"\nEpoch {epoch}/{epochs} Summary | LR: {current_lr:.6f}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        # --- Optional: Evaluate on test set each epoch ---
        if test_loader:
            test_loss, test_acc = evaluate(model, test_loader,criterion)
            print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.4f}")

    print("\nTraining complete!")
    return training_plot 