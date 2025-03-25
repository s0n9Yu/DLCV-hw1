import torch
import torchvision.models
import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        pretrained_model = torchvision.models.resnext101_32x8d(
            weights=torchvision.models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2
        )
        self.feature_extractor = \
            nn.Sequential(*list(pretrained_model.children())[:-1])
        self.fc = nn.Sequential(
            nn.Dropout(p=0.7),
            nn.Linear(in_features=12288, out_features=100),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)

        # Positional Encoding
        tmp = []
        for i in range(3):
            tmp.append(torch.sin((2 ** i) * x))
            tmp.append(torch.cos((2 ** i) * x))
        x = torch.cat(tmp, dim=1)

        x = self.fc(x)  # Trainable part
        return x
