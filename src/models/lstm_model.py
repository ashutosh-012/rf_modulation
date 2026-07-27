import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, numClasses=8, inputSize=2, hiddenSize=128, numLayers=2):
        super(LSTMClassifier, self).__init__()

        self.lstm = nn.LSTM(
            input_size=inputSize,
            hidden_size=hiddenSize,
            num_layers=numLayers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        self.fc1 = nn.Linear(hiddenSize * 2, 128)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, numClasses)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        lstmOut, _ = self.lstm(x)
        lastStep = lstmOut[:, -1, :]

        out = torch.relu(self.fc1(lastStep))
        out = self.drop(out)
        out = self.fc2(out)

        return out
