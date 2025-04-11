import torch
from torch import nn
from torchvision import models


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.chosen_features = ['0', '5', '10', '19', '28']

        # .features => give us all the conv layers
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)

        return features
