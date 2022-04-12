from torch import nn
import torch
import torch.nn.functional as F


class Fusion(nn.Module):

    def __init__(self):
        
        super(Fusion, self).__init__()
        
        self.interpolate = F.interpolate
        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)

    def forward(self, features, gradient, x_size = (120, 160)):

        gradient = F.interpolate(gradient, x_size,
                            mode='bilinear', align_corners=True)
        features = torch.mul(features, gradient) * 0.2 + features

        return features