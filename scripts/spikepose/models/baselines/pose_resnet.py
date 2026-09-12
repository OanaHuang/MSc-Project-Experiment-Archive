"""MIT-licensed SimpleBaseline Pose-ResNet adapted from the official HRNet repo."""

from __future__ import annotations

import torch.nn as nn


BN_MOMENTUM = 0.1


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(channels, channels * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * 4, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, value):
        residual = value
        value = self.relu(self.bn1(self.conv1(value)))
        value = self.relu(self.bn2(self.conv2(value)))
        value = self.bn3(self.conv3(value))
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(value + residual)


class PoseResNet(nn.Module):
    def __init__(self, layers, num_joints=16, input_color="rgb"):
        super().__init__()
        if input_color not in {"rgb", "bgr"}:
            raise ValueError(f"Unsupported input color: {input_color}")
        self.input_color = input_color
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], 2)
        self.layer3 = self._make_layer(256, layers[2], 2)
        self.layer4 = self._make_layer(512, layers[3], 2)
        modules = []
        for channels in (256, 256, 256):
            modules.extend((
                nn.ConvTranspose2d(self.inplanes, channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            ))
            self.inplanes = channels
        self.deconv_layers = nn.Sequential(*modules)
        self.final_layer = nn.Conv2d(256, num_joints, 1)
        self.model_name = "Pose-ResNet"
        self._initialize()

    def _make_layer(self, channels, blocks, stride=1):
        out_channels = channels * Bottleneck.expansion
        downsample = None
        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            )
        layers = [Bottleneck(self.inplanes, channels, stride, downsample)]
        self.inplanes = out_channels
        layers.extend(Bottleneck(self.inplanes, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(module.weight, std=0.001)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, value):
        if self.input_color == "bgr":
            value = value[:, [2, 1, 0]]
        value = self.maxpool(self.relu(self.bn1(self.conv1(value))))
        value = self.layer4(self.layer3(self.layer2(self.layer1(value))))
        return self.final_layer(self.deconv_layers(value))


def build_pose_resnet(depth=50, num_joints=16, input_color="rgb"):
    specifications = {
        50: (3, 4, 6, 3),
        101: (3, 4, 23, 3),
        152: (3, 8, 36, 3),
    }
    if depth not in specifications:
        raise ValueError(f"Unsupported Pose-ResNet depth: {depth}")
    model = PoseResNet(specifications[depth], num_joints, input_color)
    model.model_name = f"Pose-ResNet-{depth}"
    return model
