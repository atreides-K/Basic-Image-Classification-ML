# train_resnet.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR # Or StepLR, CosineAnnealingLR
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from common.utils import get_data, save_training_plot
# --- Import from your project structure ---
from common.utils import get_device # Assuming get_device is in utils
from common.train_utils import train, evaluate # Use your modified train/evaluate
# --- Import the ResNet model ---
from models import ResNet 

def run_resnet_experiment(n_value: int, depth: int):
    """Runs a training experiment for a ResNet with a given n."""

    # --- Standard ResNet Config (as decided previously) ---
    config = {
        "n_value": n_value,
        "depth": depth,
        "lr": 0.1,
        "epochs": 169,
        "batch_size": 64, # Recommended if memory allows, else 64
        "optimizer": "SGD",
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "scheduler": "MultiStepLR",
        "milestones": [80, 120], # For 160 epochs (50%, 75%)
        "gamma": 0.1,
        "loss_type": "cross_entropy", # ResNet model outputs logits
        "save_name": f'resnet_og_v1_5e4_{depth}.pth' # Use f-string for dynamic naming
    }
    print(f"\n--- Running ResNet-{depth} (n={n_value}) Experiment ---")
    print("Config:")
    print(config)
    
     # Data Augmentation from the techxzen repo👍
    train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4, padding_mode='edge'),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255)),
    transforms.Normalize([125., 123., 114.], [1., 1., 1.])
    ])

    # transform
    test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255)),
    transforms.Normalize([125., 123., 114.], [1., 1., 1.])
    ])

    # Use your get_data function
    print("Loading data using get_data...")

    train_loader, test_loader = get_data(
    dataset_name='cifar10',
    batch_size=config["batch_size"],
    train_transform=train_transform,
    test_transform=test_transform  )
    print("Data loaded.")

    # --- Create the ResNet Model ---
    model = ResNet(n=config["n_value"]) # Pass the calculated n_value
    print("\nModel Parameter Count:", sum(p.numel() for p in model.parameters()))

    # --- Create Optimizer ---
    if config["optimizer"] == "SGD":
        optimizer = optim.SGD(model.parameters(),
                              lr=config["lr"],
                              momentum=config["momentum"],
                              weight_decay=config["weight_decay"])
    else:
         raise ValueError("Unsupported optimizer for ResNet experiments") # Typically use SGD

    # --- Create Scheduler ---
    scheduler = None
    if config["scheduler"] == "MultiStepLR":
        scheduler = MultiStepLR(optimizer,
                                milestones=config["milestones"],
                                gamma=config["gamma"])
    # Add other scheduler options if needed (StepLR, CosineAnnealingLR)
    else:
        print("Warning: No scheduler specified or matched.")


    # --- Train the model ---
    # Ensure your train function uses the loss_type parameter
    training_plot=train(model,
          train_loader=train_loader,
          optimizer=optimizer,
          epochs=config["epochs"],
          scheduler=scheduler,
          test_loader=test_loader,
          loss_type=config["loss_type"])

    # --- Final Evaluation ---
    print(f"\nFinal Evaluation for ResNet-{depth}:")
    # Recreate criterion for final eval based on config
    if config["loss_type"].lower() in ['cross_entropy', 'ce']:
        final_criterion = nn.CrossEntropyLoss()
    else:
        final_criterion = nn.NLLLoss() # Should not happen based on config
    final_test_loss, final_test_acc = evaluate(model, test_loader, final_criterion)
    print(f"  Final Test Loss: {final_test_loss:.4f} | Final Test Acc: {final_test_acc:.4f}")

    # --- Save the final model ---
    print(f"Saving model to {config['save_name']}...")
    torch.save(model.state_dict(), config['save_name'])
    print(f"ResNet-{depth} model saved.")
        # save trainin plot
    save_path = config['save_name'].replace(".pth", "_history.npz")
    save_training_plot(save_path,training_plot)

    


if __name__ == '__main__':
    # --- Run experiments for required depths ---
    run_resnet_experiment(n_value=3, depth=20)
    # run_resnet_experiment(n_value=9, depth=56)
    # run_resnet_experiment(n_value=18, depth=110)

    print("\nAll ResNet experiments finished.")