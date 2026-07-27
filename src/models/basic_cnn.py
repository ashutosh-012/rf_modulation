import torch
import torch.nn as nn


class BasicCNN(nn.Module):
    def __init__(self, numClasses=8, inputChannels=2, seqLen=128):
        super(BasicCNN, self).__init__()

        self.conv1 = nn.Conv1d(inputChannels, 64, kernel_size=8, padding="same")
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding="same")
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding="same")
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(2)

        flatSize = 128 * (seqLen // 8)
        self.fc1 = nn.Linear(flatSize, 256)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, numClasses)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = self.drop1(torch.relu(self.fc1(x)))
        x = self.fc2(x)

        return x
