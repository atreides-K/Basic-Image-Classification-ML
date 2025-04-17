# version9_exp0.2.py (Example for Experiment 0.2)

import torch
import torch.nn as nn
import torch.optim as optim
# --- Import the specific scheduler you want ---
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchvision import transforms
from common.utils import get_data, get_device # get_device might not be needed directly here
# --- Import the MODIFIED train function and evaluate ---
from common.train_utils import train, evaluate
import torch.nn.functional as F # Need F for the Net definition

# --- Net class definition stays the same ---
class Net(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding="same")
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding="same")
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding="same")
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, padding="same")
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(1024, 256)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.bn1(x)
        x = torch.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.bn2(x)
        x = torch.relu(F.max_pool2d(self.conv3(x), 2))
        x = self.bn3(x)
        x = torch.relu(F.max_pool2d(self.conv4(x), 2))
        x = self.bn4(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = torch.relu(self.fc1(x))
        # Ensure output matches loss function (NLLLoss needs log-probabilities)
        x = torch.log_softmax(self.fc2(x), dim=1)
        return x

# --- Main execution function for the experiment ---
def main() -> None:
    # --- Experiment Configuration ---
    config = {
        "lr": 0.1,
        "epochs": 150,
        "batch_size": 64,
        "optimizer": "SGD",
        "scheduler": "MultiStepLR",
        "step_size": 20,
        "gamma": 0.1,
        "loss_type": "cross_entropy", # Specify loss type here
        "save_name": 'base_cnn_exp0.10_sgd_lr01_e120_multystepLR_CEloss.pth' # Your descriptive name
    }
    print("Running Experiment with Config:")
    print(config)

    # Load the data
    # Consider adding standard CIFAR10 transforms for augmentation if needed
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # Standard CIFAR-10 normalization
    ])
    train_loader, test_loader = get_data('cifar10', batch_size=config["batch_size"], transform=transform)

    # Create a model
    model = Net()
    print("\nModel Parameter Count:", sum(p.numel() for p in model.parameters()))

    # Create an optimizer
    if config["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    elif config["optimizer"] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=1e-4) # Add params if using SGD
    else:
        raise ValueError("Unsupported optimizer")

    # --- Create the Learning Rate Scheduler ---
    scheduler = None
    if config["scheduler"] == "StepLR":
        scheduler = StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])
    elif config["scheduler"] == "MultiStepLR":
         scheduler = MultiStepLR(optimizer, milestones=[75, 113], gamma=config["gamma"]) # Define milestones if using
         pass # Add other schedulers if needed
    # If config["scheduler"] is None or not matched, scheduler remains None

    # --- Train the model using the modified train function ---
    # Pass the test_loader to see evaluation results each epoch
    train(model,
          train_loader=train_loader,
          optimizer=optimizer,
          epochs=config["epochs"],
          scheduler=scheduler,
          test_loader=test_loader) # Pass scheduler and optional test_loader



    if config["loss_type"].lower() in ['cross_entropy', 'ce']:
        final_criterion = nn.CrossEntropyLoss()
    else:
        final_criterion = nn.NLLLoss()
    # --- Final Evaluation ---
    print("\nFinal Evaluation on Test Set:")
    final_test_loss, final_test_acc = evaluate(model, test_loader,final_criterion)
    print(f"Final Test Loss: {final_test_loss:.4f} | Final Test Acc: {final_test_acc:.4f}")

    # --- Save the final model ---
    print(f"Saving model to {config['save_name']}...")
    torch.save(model.state_dict(), config['save_name'])
    print("Model saved.")

