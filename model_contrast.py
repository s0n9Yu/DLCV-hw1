import torchvision.models
import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.backbone = torchvision.models.resnext101_64x4d(
            weights=torchvision.models.ResNeXt101_64X4D_Weights.IMAGENET1K_V1
        )
        self.backbone.fc = nn.Identity()  # Remove FC for feature extraction
        self.fc = nn.Sequential(
            nn.Dropout(p=0.7),
            nn.Linear(in_features=2048, out_features=100),
        )

    def forward(self, x, return_features=False):
        features = self.backbone(x)
        logits = self.fc(features)
        if return_features:
            return logits, features  # Output features for contrastive loss
        else:
            return logits
