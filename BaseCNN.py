
import torch
import torch.nn as nn
import torch.optim as optim
# --- Import the specific scheduler you want ---
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from common.utils import get_data, get_device # get_device might not be needed directly here
# --- Import the MODIFIED train function and evaluate ---
from common.train_utils import train, evaluate
import torch.nn.functional as F # Need F for the Net definition


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