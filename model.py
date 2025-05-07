import torch.nn as nn
import torch.nn.functional as F


class PlantCNN(nn.Module):
    def __init__(self, num_classes=47):
        super(PlantCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # -> [B, ?, 4, 4]

        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # -> [B, 32, H/2, W/2]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # -> [B, 64, H/4, W/4]
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # -> [B, 128, H/8, W/8]
        x = self.adaptive_pool(x)  # -> [B, 128, 4, 4]
        x = x.view(x.size(0), -1)  # -> [B, 2048]
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
