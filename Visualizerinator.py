# visualize_landscape.py

import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting
import os
import copy # For deep copying models/parameters

# --- Assume these utils are in common/ ---
from common.utils import get_data, get_device, save_training_plot # Reusing save history function
from common.train_utils import evaluate # Need evaluate to calculate loss

# --- Assume model definitions are importable ---
# Make sure ResNet.py is in models/ and has the integer division fix
from models import ResNet, PlainNet
# Make sure you have a file (e.g., base_cnn_model.py) with your best Net class
from BaseCNN import Net # Or whatever you called the file/class

# --- Configuration ---
DEVICE = get_device()
CHECKPOINT_DIR = ""
PLOT_DIR = "plots"
HISTORY_DIR = "history" # Where history .npz files are saved
MODEL_DIR = "models" # Where ResNet.py etc. are

# --- Visualization Parameters ---
RESOLUTION = 50 # Grid resolution (e.g., 25x25 points). Higher = slower but smoother plots. Start low (e.g., 10 or 15) for testing.
ALPHA_RANGE = (-1.0, 1.0) # Range for alpha direction
BETA_RANGE = (-1.0, 1.0) # Range for beta direction
LOADER_BATCH_SIZE = 256 # Batch size for calculating loss during visualization (can be different from training)
SUBSET_SIZE = None # Use None for full test set, or e.g., 1000 for a faster subset evaluation

# --- Models to Visualize ---
# List of dictionaries defining each model to process
# Make sure 'ckpt_path' points to your actual saved model files
# and 'arch_type'/'n_value' match the model definition
models_to_visualize = [
    {
        "name": "BaseCNN_Best",
        "ckpt_path": os.path.join("best.pth"),
        "arch_type": "net",
        "n_value": None, # Not applicable for Net
        "loss_type": "nll" # Or 'cross_entropy' if your best base CNN used that
    },
    {
        "name": "ResNet-20",
        "ckpt_path": os.path.join("resnet_neo_v1_5e4_20.pth"), # Use your actual filename
        "arch_type": "resnet",
        "n_value": 3,
        "loss_type": "cross_entropy"
    },
    {
        "name": "ResNet-56",
        "ckpt_path": os.path.join("resnet_neo_v2_5e4_56.pth"), # Use your actual filename
        "arch_type": "resnet",
        "n_value": 9,
        "loss_type": "cross_entropy"
    },
    {
        "name": "ResNet-110",
        "ckpt_path": os.path.join("resnet_neo_v3_5e4_110.pth"), # Use your actual filename
        "arch_type": "resnet",
        "n_value": 18,
        "loss_type": "cross_entropy"
    },
    {
        "name": "PlainNet-20",
        "ckpt_path": os.path.join("plainnet_neo_v1_5e4_20.pth"), # Use your actual filename
        "arch_type": "plainnet",
        "n_value": 3,
        "loss_type": "cross_entropy"
    },
    {
        "name": "PlainNet-56",
        "ckpt_path": os.path.join("plainnet_neo_v2_5e4_56.pth"), # Use your actual filename
        "arch_type": "plainnet",
        "n_value": 9,
        "loss_type": "cross_entropy"
    },
    {
        "name": "PlainNet-110",
        "ckpt_path": os.path.join("plainnet_neo_v3_5e4_110.pth"), # Use your actual filename
        "arch_type": "plainnet",
        "n_value": 18,
        "loss_type": "cross_entropy"
    },
]


# --- Helper Functions ---

def get_model_parameters(model):


    # print(model)
    # print(model.parameters())
    params = []
    for param in model.parameters():
        # Detach and clone to avoid modifying original model's computation graph
        # Although we use torch.no_grad later, this is safer.
        params.append(param.data.clone().detach())
    return params

def set_model_parameters(model, params_list):
    # print("model")
    # print(model)
    # print("params_list")
    # print(params_list)

    with torch.no_grad(): 
        for model_param, new_param in zip(model.parameters(), params_list):
            model_param.data.copy_(new_param.data) 

def get_random_directions(params_list):
    """Generate two random direction vectors matching the structure of params_list."""
    direction1 = []
    direction2 = []
    print("Generating random directions...")
    num_params_total = 0
    for param in params_list:
        # Gaussian random numbers with same shape as parameter tensor
        dir1 = torch.randn_like(param)
        dir2 = torch.randn_like(param)
        direction1.append(dir1)
        direction2.append(dir2)
        num_params_total += param.numel()
        # print(f"  Layer shape: {param.shape}, Num params: {param.numel()}")
    print(f"Total parameters in model: {num_params_total}")
    print("Finished generating directions.")
    return direction1, direction2

def normalize_direction_filterwise(direction, weights):
    # Applyin filter-wise normalization 

    normalized_direction = []
    if len(direction) != len(weights):
        print("Error: Direction and weights must have the same length.")
        return None

    print("Normalizing direction filter-wise")
    processed_params = 0
    for d, w in zip(direction, weights):
        # Check if it's a learnable weight tensor (usually 2D for Linear, 4D for Conv2d)
        # Skip biases (1D) and Batch Norm parameters (usually 1D)
        if d.dim() <= 1:
            # print(f"  Skipping normalization for param shape {d.shape} (likely bias/BN)")
            # Just append the original random direction for biases/BN params
            # Their scale doesn't have the same invariance issue
            normalized_direction.append(d.clone()) # Clone to be safe
            processed_params += d.numel()
            continue

        # Calculate norm filter-wise
        # For Linear layers (dim=2: out_features, in_features), treat each output neuron's weights as a "filter"
        # For Conv2d layers (dim=4: out_channels, in_channels, H, W), treat each output channel as a "filter"
        normalized_filters = []
        # print(f"  Normalizing param shape {d.shape}")
        for i in range(d.shape[0]): # Iterate over output dimension (filters/neurons)
            d_filter = d[i]
            w_filter = w[i]

            # Use Frobenius norm (sqrt of sum of squares)
            d_norm = torch.linalg.norm(d_filter)
            w_norm = torch.linalg.norm(w_filter)

            # Avoid division by zero if a direction filter happens to be all zeros
            if d_norm > 1e-8: # Small epsilon
                scale = w_norm / d_norm
                normalized_filter = d_filter * scale
            else:
                # print(f"    Warning: Zero norm for direction filter {i} of shape {d_filter.shape}. Keeping original random values.")
                normalized_filter = d_filter # Keep original random noise if norm is zero

            normalized_filters.append(normalized_filter)

        # Stack the normalized filters back into a tensor
        normalized_layer_tensor = torch.stack(normalized_filters, dim=0)
        normalized_direction.append(normalized_layer_tensor)
        processed_params += d.numel()

    print(f"Normalization finished. Processed {processed_params} parameters.")
    return normalized_direction

def get_loss(model, criterion, loader, device, subset_size=None):
    """Calculates loss over the dataset (or a subset)."""
    model.eval()
    model.to(device)
    total_loss = 0.0
    total_samples = 0

    data_iter = iter(loader)
    num_batches_to_process = len(loader)
    if subset_size is not None:
        num_batches_to_process = int(np.ceil(subset_size / loader.batch_size))
        print(f"  Calculating loss on subset: {num_batches_to_process} batches...")

    processed_batches = 0
    with torch.no_grad():
        # for data, target in tqdm(loader, desc=" Eval", leave=False): # tqdm inside loop can be noisy
        while processed_batches < num_batches_to_process:
            try:
                data,target = next(data_iter)
                data,target = data.to(device), target.to(device)
                output=model(data)
                loss=criterion(output,target)
                total_loss+= loss.item()*data.size(0)
                total_samples+=data.size(0)
                processed_batches+=1
            except StopIteration:
                break # Reached end of loader

    if total_samples == 0:
        return float('inf') # Avoid division by zero

    return total_loss/total_samples

# --- Main Visualization Logic ---
def visualize_model(model_config):
    # Ensure plot directory exists
    os.makedirs(PLOT_DIR,exist_ok=True)

    # --- Load Data Once ---
    # Use the test transform for evaluating loss during visualization
    print("Loading evaluation data")
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
    ])
    # Use the original get_data or load manually - needs only test set here
    _, eval_loader = get_data(
        'cifar10',
        batch_size=LOADER_BATCH_SIZE,
        transform=eval_transform # Apply the non-augmented transform
    )
    print("Evaluation data loaded.")



    model_name=model_config["name"]
    ckpt_path=model_config["ckpt_path"]
    arch_type=model_config["arch_type"]
    n_value=model_config["n_value"]
    loss_type=model_config["loss_type"]

    print(f"\n\nProcessing hte Model: {model_name} ")
    print(f"Checkpoint:{ckpt_path}")

    if not os.path.exists(ckpt_path):
        print(f"!!! Oh no Checkpoint file not found: {ckpt_path}. Skipping model.")
        return
    
    # was running into issues with the plot error when os not exist 
    # so wanna rerun and create plots of already created ones
    plot3d_save_path = os.path.join(PLOT_DIR, f"{model_name}_3D_landscape_{RESOLUTION}x{RESOLUTION}.png")
    plot2d_save_path = os.path.join(PLOT_DIR, f"{model_name}_2D_contour_{RESOLUTION}x{RESOLUTION}.png")
    if os.path.exists(plot3d_save_path) and os.path.exists(plot2d_save_path):
        print(f"Plots already exist for {model_name} at {RESOLUTION}x{RESOLUTION} resolution. Skipping.")
        return


    # loads the model architecture based on the type
    print("Instantiating the model ")
    if arch_type=="net":
        model=Net(num_classes=10)
        # Ensure model forward pass matches loss_type
        # This assumes you saved the *correct* model variant for base_cnn_best.pth
    elif arch_type=="resnet":
        model=ResNet(n=n_value)
    elif arch_type=="plainnet":
        model=PlainNet(n=n_value)
    else:
        print(f"!!! Unknown architecture type '{arch_type}'. Skipping.")
        return


    print("Loading trained weights...")
    try:
        # Adjust loading based on how checkpoints were saved
        # If saved directly using torch.save(model.state_dict(), ...):
        state_dict = torch.load(ckpt_path, map_location=torch.device('cpu')) # Load to CPU first
        # Handle potential keys like 'state' if saved from techxzen repo run.py
        if 'state' in state_dict and isinstance(state_dict['state'], dict):
                model.load_state_dict(state_dict['state'])
                print(" Loaded  state_dict from 'state' key.")
        else:
                model.load_state_dict(state_dict)
                print(" Loaded  state_dict directly.")

        model.to(DEVICE) # Move model to GPU *after* loading state_dict
    except Exception as e:
        print(f"!!! Error loading checkpoint {ckpt_path}: {e}. Skipping.")
        return

    # --- 3. Get Trained Parameters (theta_star) ---
    theta_star = get_model_parameters(model)
    # print(" Theta_star shapes:", [p.shape for p in theta_star])

    # --- 4. Get Random Directions (delta, eta) ---
    delta,eta=get_random_directions(theta_star)
    # print(" Delta shapes:", [p.shape for p in delta])
    # print(" Eta shapes:", [p.shape for p in eta])

    # --- 5. Normalize Directions Filter-wise ---
    delta_norm = normalize_direction_filterwise(delta, theta_star)
    eta_norm = normalize_direction_filterwise(eta, theta_star)

    # Optional Sanity Check: Calculate norms before/after (can be slow)
    # norm_delta_before = sum([torch.linalg.norm(p).item()**2 for p in delta])**0.5
    # norm_eta_before = sum([torch.linalg.norm(p).item()**2 for p in eta])**0.5
    # norm_delta_after = sum([torch.linalg.norm(p).item()**2 for p in delta_norm])**0.5
    # norm_eta_after = sum([torch.linalg.norm(p).item()**2 for p in eta_norm])**0.5
    # norm_theta_star = sum([torch.linalg.norm(p).item()**2 for p in theta_star])**0.5
    # print(f" Norms: ||theta*||={norm_theta_star:.2f} ||delta||={norm_delta_before:.2f} -> {norm_delta_after:.2f} ||eta||={norm_eta_before:.2f} -> {norm_eta_after:.2f}")


    # --- 6. Define Loss Function ---
    if loss_type.lower() == 'nll':
        criterion = nn.NLLLoss()
        print("Using NLLLoss for evaluation (model should output LogSoftmax).")
    elif loss_type.lower() in ['cross_entropy', 'ce']:
        criterion = nn.CrossEntropyLoss()
        print("Using CrossEntropyLoss for evaluation (model should output logits).")
    else:
        print(f"!!! Invalid loss_type '{loss_type}' specified. Skipping.")
        return

    # --- 7. Calculate Loss over Grid ---
    alpha_coords = np.linspace(ALPHA_RANGE[0], ALPHA_RANGE[1], RESOLUTION)
    beta_coords = np.linspace(BETA_RANGE[0], BETA_RANGE[1], RESOLUTION)
    loss_surface = np.zeros((RESOLUTION, RESOLUTION))

    # Create a temporary model copy ONCE for efficiency
    # We will load parameters into this copy repeatedly
    model_copy = copy.deepcopy(model)
    model_copy.to(DEVICE)

    print(f"Calculating loss on a {RESOLUTION}x{RESOLUTION} grid...")
    total_points = RESOLUTION * RESOLUTION
    processed_points = 0

    for i, alpha in enumerate(alpha_coords):
        for j, beta in enumerate(beta_coords):
            # Calculate temporary parameters
            theta_temp = []
            # Use .data to avoid graph tracking during calculation
            # Also do calculations on CPU potentially if memory is tight, then move tensor
            for ws, d_n, e_n in zip(theta_star, delta_norm, eta_norm):
                # Ensure alpha and beta are treated as scalars
                term = ws.data + float(alpha) * d_n.data + float(beta) * e_n.data
                theta_temp.append(term.to(DEVICE)) # Move to device

            # Load temporary parameters into the model copy
            set_model_parameters(model_copy, theta_temp)

            # Calculate loss
            current_loss = get_loss(model_copy, criterion, eval_loader, DEVICE, subset_size=SUBSET_SIZE)
            loss_surface[i, j] = current_loss

            processed_points += 1
            if processed_points % 10 == 0 or processed_points == total_points: # Print progress
                print(f"  Processed grid point {processed_points}/{total_points}...", end='\r')

    print("\nLoss calculation complete.")

    # --- 8. Save Loss Surface Data ---
    surface_save_path = os.path.join(PLOT_DIR, f"{model_name}_loss_surface_{RESOLUTION}x{RESOLUTION}.npz")
    np.savez(surface_save_path, alpha=alpha_coords, beta=beta_coords, losses=loss_surface)
    print(f"Loss surface data saved to {surface_save_path}")

    # --- 9. Generate Plots ---
    X, Y = np.meshgrid(alpha_coords, beta_coords)

    # 3D Surface Plot
    print("Generating 3D Plot...")
    fig3d = plt.figure(figsize=(10, 8))
    ax3d = fig3d.add_subplot(111, projection='3d')
    # Use log loss for better visualization if values vary a lot, handle potential zeros/negatives
    surf = ax3d.plot_surface(X, Y, np.log(loss_surface+1e-6).T, cmap='viridis', edgecolor='none') # Log scale, transpose Z
    ax3d.set_xlabel('Alpha')
    ax3d.set_ylabel('Beta')
    ax3d.set_zlabel('Log Loss')
    ax3d.set_title(f"Loss Landscape: {model_name} (Log Scale)")
    fig3d.colorbar(surf, shrink=0.5, aspect=5)
    plot3d_save_path = os.path.join(PLOT_DIR, f"{model_name}_3D_landscape_{RESOLUTION}x{RESOLUTION}.png")
    plt.savefig(plot3d_save_path)
    plt.close(fig3d)
    print(f"3D Plot saved to {plot3d_save_path}")

    # 2D Contour Plot
    print("Generating 2D Contour Plot...")
    fig2d, ax2d = plt.subplots(figsize=(8, 7))
    # Adjust levels for contour plot - maybe log scale helps here too
    # Often need to experiment with contour levels
    # levels = np.linspace(np.min(loss_surface), np.min(loss_surface) + 1.0, 15) # Example levels: min loss to min loss + 1
    # Or logarithmic levels: levels = np.logspace(np.log10(np.min(loss_surface)+1e-6), np.log10(np.max(loss_surface)), 15)
    # CS=ax2d.contourf(X, Y, loss_surface.T, levels=levels, cmap='viridis', extend='max') # Filled contour, transpose Z
    n_levels=20
    # CS=ax2d.contour(X, Y, loss_surface.T, levels=n_levels, cmap='viridis', linewidths=0.5) # Add contour lines
    CS=ax2d.contour(X, Y, np.log(loss_surface+1e-6).T, levels=n_levels, cmap='viridis', linewidths=0.5) # Add contour lines
    ax2d.set_xlabel('Alpha')
    ax2d.set_ylabel('Beta')
    ax2d.set_title(f"Loss Contours: {model_name}")
    ax2d.clabel(CS, inline=True, fontsize=8, fmt='%.2f')
    fig2d.colorbar(CS)
    plot2d_save_path = os.path.join(PLOT_DIR, f"{model_name}_2D_contour_{RESOLUTION}x{RESOLUTION}.png")
    plt.savefig(plot2d_save_path)
    plt.close(fig2d)
    print(f"2D Plot saved to {plot2d_save_path}")

    # Clear memory explicitly (might help on some systems)
    del model, model_copy, theta_star, delta, eta, delta_norm, eta_norm, theta_temp, loss_surface
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n\nModel Visualizations Finito")



# CHECK ALL FILES EXIST!!!!
def pre_check():
    print("Performing pre-checks...")

    # Check if required model checkpoint files exist
    for model_config in models_to_visualize:
        ckpt_path = model_config["ckpt_path"]
        if not ckpt_path.endswith('.pth'):
            print(f"!!! Invalid checkpoint file format: {ckpt_path}. Expected a .pth file.")
            return False
        if not os.path.exists(ckpt_path):
            print(f"!!! Required checkpoint file not found: {ckpt_path}. Please ensure it exists.")
            return False

    print("All pre-checks passed.")
    return True

if __name__ == '__main__':
    # if not pre_check():
    #     print("Pre-checks failed. Exiting.")
    #     exit(1)

    parser = argparse.ArgumentParser(description="Visualize loss landscape for a specific model using its index.")
    num_models = len(models_to_visualize)
    parser.add_argument('--model_index', type=int, required=True,
                        help=f'Index of the model to visualize (0 to {num_models-1}).')
    # Removed resolution and subset_size args, use constants defined above

    args = parser.parse_args()

    # --- Validate Model Index ---
    if not 0 <= args.model_index < num_models:
        print(f"Error: Invalid model index {args.model_index}. Please provide an index between 0 and {num_models-1}.")
        # Print available models and their indices
        print("Available models:")
        for idx, config in enumerate(models_to_visualize):
            print(f"  Index {idx}: {config['name']}")
        exit(1)

    # --- Get the configuration for the requested model index ---
    model_config = models_to_visualize[args.model_index]
    visualize_model(model_config)