"""Model definitions used in experiments: SimCLR backbones and projection heads.

This module provides `SimCLR`, `downsizedSimCLR`, and small utilities used
by training and feature-extraction scripts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet18


class ProjectionHead(nn.Module):
    """Simple MLP projection head used by SimCLR.

    Maps encoder features to a lower-dimensional projection space used for
    contrastive loss computation.
    """
    def __init__(self, in_dim, hid=2048, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, out_dim),
        )

    def forward(self, x):
        # ensure contiguous before Linear
        return self.net(x.contiguous())



def resnet18_cifar_backbone():
    """
    Torchvision ResNet-18 modified for CIFAR-10/32x32:
      - 3x3 conv, stride=1, padding=1
      - remove maxpool
      - keep rest identical
    Returns a backbone that outputs a 512-dim feature vector (before any projection head).
    """
    m = resnet18(weights=None)

    # CIFAR stem
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()

    # Keep the global average pool; remove classifier head
    return m

class downsizedSimCLR(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        base = resnet18_cifar_backbone()
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.encoder = base
        self.proj = ProjectionHead(feat_dim, 512, out_dim)

    def forward(self, x):
        # avoid channels_last; keep contiguous to dodge .view() stride issues
        x = x.contiguous()
        h = self.encoder(x)
        h = h.contiguous()
        z = self.proj(h)
        z = F.normalize(z, dim=1)
        return z


class SimCLR(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        base = resnet50(weights=None)
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.encoder = base
        self.proj = ProjectionHead(feat_dim, 2048, out_dim)

    def forward(self, x):
        # avoid channels_last; keep contiguous to dodge .view() stride issues
        x = x.contiguous()
        h = self.encoder(x)
        h = h.contiguous()
        z = self.proj(h)
        z = F.normalize(z, dim=1)
        return z
    
activations = {}
def save_activation(name):
    """Return a forward-hook that saves detached output under `name`.

    Useful for quick feature extraction in small scripts and debugging.
    """
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

handles = []
layers_to_hook = ['encoder.layer1', 'encoder.layer2', 'encoder.layer3', 'encoder.layer4']
def register_hooks(model):
    """Register forward hooks on standard encoder layers to cache activations.

    Appends hook handles to the global `handles` list so they can be removed
    later if desired.
    """
    for layer_name in layers_to_hook:
        layer = dict([*model.named_modules()])[layer_name]
        handle = layer.register_forward_hook(save_activation(layer_name))
        handles.append(handle)

if __name__ == "__main__":
    model = SimCLR()
    # register_hooks(model)
    for name, param in model.named_parameters():
        print(name, param.shape)
    x = torch.randn(4, 3, 128, 128)
    z = model(x)
    print(z.shape)
