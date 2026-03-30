import torch
from torchvision import models


class BaselineModelConvNextBase(torch.nn.Module):
    def __init__(self):
        super(BaselineModelConvNextBase, self).__init__()
        self.model = models.convnext_base()
        self.model.classifier[2] = torch.nn.Identity()

    def forward(self, x):
        x = self.model(x)
        return x