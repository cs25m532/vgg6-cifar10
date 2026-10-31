"""VGG6 with a configurable activation function."""
import math

import torch
import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}

# 4 conv layers + 1 linear layer; 'M' is a 2x2 max-pool.
CFG_VGG6 = [64, 64, "M", 128, 128, "M"]


def make_layers(cfg, activation="relu", batch_norm=True):
    act_cls = ACTIVATIONS[activation]
    layers, in_channels = [], 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
        if batch_norm:
            layers.append(nn.BatchNorm2d(v))
        layers.append(act_cls())
        in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, features, num_classes=10):
        super().__init__()
        self.features = features
        self.classifier = nn.Linear(128, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=(2, 3))  # global average pool (deterministic on CUDA)
        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_model(activation="relu", batch_norm=True, num_classes=10):
    return VGG(make_layers(CFG_VGG6, activation, batch_norm), num_classes)
