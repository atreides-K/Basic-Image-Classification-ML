# evaluate_model.py

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import argparse

# --- Assume these utils are in common/ ---
try:
    from common.utils import get_data, get_device
    # Assuming evaluate function signature is evaluate(model, test_loader, criterion, device)
    # If it doesn't take device, remove it from the call below.
    from common.train_utils import evaluate
except ImportError as e:
    print(f"Error importing from 'common': {e}")
    print("Please ensure 'common/utils.py' and 'common/train_utils.py' exist.")
    sys.exit(1)

# --- Assume model definitions are importable ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")
    if models_dir not in sys.path: sys.path.insert(0, models_dir)

    from ResNet import ResNet, PlainNet
    from BaseCNN import Net # Assuming BaseCNN.py contains Net class
except ImportError as e:
    print(f"Error importing model definitions: {e}")
    print("Please ensure 'models/ResNet.py' and 'models/BaseCNN.py' exist.")
    sys.exit(1)

# --- Configuration ---
DEVICE = get_device()
# ----- SET Correct Checkpoint Directory -----
CHECKPOINT_DIR = "." # Or "" or "." if files are in the same dir as script
MODEL_DIR = "models" # For verification if needed

# --- Models to Evaluate (Central Definition - Order Matters for Indexing) ---
# ----- VERIFY Checkpoint Paths and Loss Type (for selecting criterion) -----
models_to_evaluate_config = [
    # Index 0
    {
        "name": "BaseCNN_Best",
        "ckpt_path": os.path.join(CHECKPOINT_DIR, "best.pth"), # VERIFY FILENAME
        "arch_type": "net", "n_value": None, "loss_type": "cross_entropy" # Use CE for eval
    },
    # Index 1
    {
        "name": "ResNet-20",
        "ckpt_path": os.path.join(CHECKPOINT_DIR, "resnet_neo_v1_5e4_20.pth"), # VERIFY FILENAME
        "arch_type": "resnet", "n_value": 3, "loss_type": "cross_entropy"
    },
    # Index 2
    {
        "name": "ResNet-56",
        "ckpt_path": os.path.join(CHECKPOINT_DIR, "resnet_neo_v2_5e4_56.pth"), # VERIFY FILENAME (v1 or v2?)
        "arch_type": "resnet", "n_value": 9, "loss_type": "cross_entropy"
    },
    # Index 3
    {
        "name": "ResNet-110",
        "ckpt_path": os.path.join(CHECKPOINT_DIR, "resnet_neo_v3_5e4_110_2.pth"), # VERIFY FILENAME (_2?)
        "arch_type": "resnet", "n_value": 18, "loss_type": "cross_entropy"
    },
    # Index 4
    {
        "name": "PlainNet-20",
        "ckpt_path": os.path.join(CHECKPOINT_DIR, "plainnet_neo_v1_5e4_20.pth"), # VERIFY FILENAME
        "arch_type": "plainnet", "n_value": 3, "loss_type": "cross_entropy"
    },
    # Index 5
    {
        "name": "PlainNet-56",
        "ckpt_path": os.path.join(CHECKPOINT_DIR, "plainnet_neo_v2_5e4_56.pth"), # VERIFY FILENAME (v1 or v2?)
        "arch_type": "plainnet", "n_value": 9, "loss_type": "cross_entropy"
    },
    # Index 6
    {
        "name": "PlainNet-110",
        "ckpt_path": os.path.join(CHECKPOINT_DIR, "plainnet_neo_v3_5e4_110_2.pth"), # VERIFY FILENAME (_2?)
        "arch_type": "plainnet", "n_value": 18, "loss_type": "cross_entropy"
    },
]

# --- Main Execution Block ---
if __name__ == '__main__':

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Evaluate a trained model checkpoint.")
    num_models = len(models_to_evaluate_config)
    parser.add_argument('--model_index', type=int, required=True,
                        help=f'Index of the model to evaluate (0 to {num_models-1}).')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for evaluation loader.')


    args = parser.parse_args()

    # --- Validate Model Index ---
    if not 0 <= args.model_index < num_models:
        print(f"Error: Invalid model index {args.model_index}. Please provide an index between 0 and {num_models-1}.")
        print("Available models:")
        for idx, config in enumerate(models_to_evaluate_config):
            print(f"  Index {idx}: {config['name']}")
        sys.exit(1)

    # --- Get the configuration for the requested model index ---
    model_config = models_to_evaluate_config[args.model_index]
    model_name = model_config["name"]
    ckpt_path = model_config["ckpt_path"]
    arch_type = model_config["arch_type"]
    n_value = model_config["n_value"]
    loss_type = model_config["loss_type"] # Needed to select correct criterion

    print(f"\n--- Evaluating Model: {model_name} ---")
    print(f"Checkpoint Path: {ckpt_path}")

    # --- Check if Checkpoint File Exists ---
    if not os.path.exists(ckpt_path):
        print(f"!!! Error: Checkpoint file not found at {ckpt_path}")
        sys.exit(1)

    # --- Load Test Data ---
    print(f"Loading CIFAR-10 test data (Batch Size: {args.batch_size})...")
    # Use the standard test transform (usually just ToTensor and Normalize)
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    try:
        # Only need the test loader
        _, test_loader = get_data('cifar10', batch_size=args.batch_size, transform=eval_transform)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    print("Test data loaded.")

    # --- Instantiate Model Architecture ---
    print("Instantiating model architecture...")
    try:
        if arch_type == "net": model = Net(num_classes=10)
        elif arch_type == "resnet": model = ResNet(n=n_value)
        elif arch_type == "plainnet": model = PlainNet(n=n_value)
        else: raise ValueError(f"Unknown architecture type '{arch_type}'")
    except Exception as e:
        print(f"Error instantiating model '{model_name}': {e}")
        sys.exit(1)

    # --- Load Weights ---
    print(f"Loading weights from {ckpt_path}...")
    try:
        state_dict = torch.load(ckpt_path, map_location=torch.device('cpu')) # Load to CPU first
        # Handle potential nested dictionaries
        if 'state' in state_dict and isinstance(state_dict['state'], dict):
            load_dict = state_dict['state']
            print(" Loaded state_dict from 'state' key.")
        elif isinstance(state_dict, dict):
            load_dict = state_dict
            print(" Loaded state_dict directly.")
        else:
             raise TypeError("Checkpoint file did not contain a dictionary (state_dict).")

        model.load_state_dict(load_dict)
        model.to(DEVICE) # Move model to target device
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"!!! Error loading weights: {e}")
        sys.exit(1)

    # --- Define Loss Criterion ---
    # Needed for the evaluate function, even if only accuracy is reported
    if loss_type.lower() == 'nll':
        # Important: If using NLLLoss, the loaded model MUST output log-softmax
        print("Using NLLLoss criterion (Model MUST output LogSoftmax)")
        criterion = nn.NLLLoss()
    elif loss_type.lower() in ['cross_entropy', 'ce']:
        print("Using CrossEntropyLoss criterion (Model should output logits)")
        criterion = nn.CrossEntropyLoss()
    else:
        print(f"!!! Invalid loss_type '{loss_type}' specified in config. Cannot select criterion.")
        sys.exit(1)

    # --- Evaluate the Model ---
    print(f"Running evaluation on {DEVICE}...")
    # Set model to evaluation mode (disables dropout, uses running stats for BN)
    model.eval()

    # Call your evaluate function
    # Assuming evaluate signature: evaluate(model, test_loader, criterion, device)
    # If your evaluate function doesn't need 'device', remove it from the call.
    try:
        test_loss, test_acc = evaluate(model, test_loader, criterion) # Pass device
    except Exception as e:
        print(f"Error during evaluation: {e}")
        # Check if the error might be related to evaluate not needing device
        print("Ensure the 'evaluate' function in common/train_utils.py is compatible.")
        sys.exit(1)

    # --- Print Results ---
    print("\n--- Evaluation Results ---")
    print(f"Model:           {model_name}")
    print(f"Checkpoint:      {ckpt_path}")
    print(f"Test Loss:       {test_loss:.4f}")
    print(f"Test Accuracy:   {test_acc:.4f}%") # Assuming evaluate returns accuracy in %
    print("--------------------------")