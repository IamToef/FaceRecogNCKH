import torch
import torch.nn as nn

from torchvision.models import resnet34

class FRModel(nn.Module):
    def __init__(self, n_classes):
        super(FRModel, self).__init__()
        resnet = resnet34(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        in_features = resnet.fc.in_features
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

