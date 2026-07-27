import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    def __init__(self, numClasses=8, inputChannels=2, numBlocks=6, hiddenDim=64):
        super(ResNet1D, self).__init__()

        self.inputConv = nn.Conv1d(inputChannels, hiddenDim, kernel_size=7, padding=3)
        self.inputBn = nn.BatchNorm1d(hiddenDim)
        self.inputRelu = nn.ReLU()

        blocks = []
        for i in range(numBlocks):
            blocks.append(ResBlock(hiddenDim))
        self.resBlocks = nn.Sequential(*blocks)

        self.globalPool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hiddenDim, numClasses)

    def forward(self, x):
        x = self.inputRelu(self.inputBn(self.inputConv(x)))
        x = self.resBlocks(x)
        x = self.globalPool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x
