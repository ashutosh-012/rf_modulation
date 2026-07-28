import torch
import torch.nn as nn
import timm

class EfficientNetWrapper(nn.Module):
    def __init__(self, numClasses=8, inputChannels=1, pretrained=True):
        super(EfficientNetWrapper, self).__init__()

        self.backbone = timm.create_model(
            "tf_efficientnetv2_s",
            pretrained=pretrained,
            num_classes=numClasses,
            in_chans=inputChannels
        )

    def forward(self, x):
        if x.dim() == 3 and x.size(1) == 1:
            pass
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        out = self.backbone(x)
        return out
