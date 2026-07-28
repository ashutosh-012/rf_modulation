import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()

        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        squeezed = self.squeeze(x).view(b, c)
        excited = torch.relu(self.fc1(squeezed))
        excited = self.sigmoid(self.fc2(excited))
        excited = excited.view(b, c, 1)
        return x * excited

class CLDNN_SE(nn.Module):
    def __init__(self, numClasses=8, inputChannels=2, seqLen=128):
        super(CLDNN_SE, self).__init__()

        self.conv1 = nn.Conv1d(inputChannels, 64, kernel_size=8, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.se = SEBlock(64, reduction=8)

        self.pool = nn.MaxPool1d(2)

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=False
        )

        self.fc1 = nn.Linear(128 + 64, 256)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, numClasses)

        self.skipPool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        cnnOut = torch.relu(self.bn1(self.conv1(x)))
        cnnOut = torch.relu(self.bn2(self.conv2(cnnOut)))

        cnnOut = self.se(cnnOut)

        skipFeat = self.skipPool(cnnOut).squeeze(-1)
        cnnOut = self.pool(cnnOut)

        lstmInput = cnnOut.permute(0, 2, 1)
        lstmOut, _ = self.lstm(lstmInput)
        lstmFinal = lstmOut[:, -1, :]

        combined = torch.cat([lstmFinal, skipFeat], dim=1)

        out = torch.relu(self.fc1(combined))
        out = self.drop(out)
        out = self.fc2(out)

        return out
