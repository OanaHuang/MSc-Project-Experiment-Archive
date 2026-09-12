"""MIT-licensed HRNet pose model adapted from the official HRNet repository."""

from __future__ import annotations

import torch.nn as nn


BN_MOMENTUM = 0.1


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)
        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, value):
        residual = value
        value = self.relu(self.bn1(self.conv1(value)))
        value = self.bn2(self.conv2(value))
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(value + residual)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)
        self.conv2 = conv3x3(channels, channels, stride)
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


class HighResolutionModule(nn.Module):
    def __init__(self, branches, blocks, in_channels, channels,
                 multi_scale_output=True):
        super().__init__()
        self.branches_count = branches
        self.in_channels = list(in_channels)
        self.branches = self._make_branches(blocks, channels)
        self.fuse_layers = self._make_fuse_layers(multi_scale_output)
        self.relu = nn.ReLU(inplace=True)

    def _make_branch(self, index, blocks, channels):
        downsample = None
        if self.in_channels[index] != channels[index]:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels[index], channels[index], 1, bias=False),
                nn.BatchNorm2d(channels[index], momentum=BN_MOMENTUM),
            )
        result = [BasicBlock(self.in_channels[index], channels[index], downsample=downsample)]
        self.in_channels[index] = channels[index]
        result.extend(BasicBlock(channels[index], channels[index])
                      for _ in range(1, blocks[index]))
        return nn.Sequential(*result)

    def _make_branches(self, blocks, channels):
        return nn.ModuleList(
            self._make_branch(index, blocks, channels)
            for index in range(self.branches_count)
        )

    def _make_fuse_layers(self, multi_scale_output):
        if self.branches_count == 1:
            return None
        outputs = self.branches_count if multi_scale_output else 1
        layers = []
        for target in range(outputs):
            row = []
            for source in range(self.branches_count):
                if source > target:
                    row.append(nn.Sequential(
                        nn.Conv2d(self.in_channels[source], self.in_channels[target], 1, bias=False),
                        nn.BatchNorm2d(self.in_channels[target], momentum=BN_MOMENTUM),
                        nn.Upsample(scale_factor=2 ** (source - target), mode="nearest"),
                    ))
                elif source == target:
                    row.append(nn.Identity())
                else:
                    modules = []
                    current = self.in_channels[source]
                    for step in range(target - source):
                        final = step == target - source - 1
                        output = self.in_channels[target] if final else current
                        modules.extend((
                            nn.Conv2d(current, output, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(output, momentum=BN_MOMENTUM),
                        ))
                        if not final:
                            modules.append(nn.ReLU(inplace=True))
                        current = output
                    row.append(nn.Sequential(*modules))
            layers.append(nn.ModuleList(row))
        return nn.ModuleList(layers)

    def get_num_inchannels(self):
        return self.in_channels

    def forward(self, values):
        values = [branch(value) for branch, value in zip(self.branches, values)]
        if self.fuse_layers is None:
            return values
        output = []
        for target, row in enumerate(self.fuse_layers):
            fused = row[0](values[0])
            for source in range(1, self.branches_count):
                fused = fused + row[source](values[source])
            output.append(self.relu(fused))
        return output


class PoseHighResolutionNet(nn.Module):
    def __init__(self, width=32, num_joints=16):
        super().__init__()
        self.inplanes = 64
        self.conv1 = conv3x3(3, 64, 2)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = conv3x3(64, 64, 2)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)
        stage2_channels = [width, width * 2]
        self.transition1 = self._make_transition([256], stage2_channels)
        self.stage2, channels = self._make_stage(1, [4, 4], stage2_channels)
        stage3_channels = [width, width * 2, width * 4]
        self.transition2 = self._make_transition(channels, stage3_channels)
        self.stage3, channels = self._make_stage(4, [4, 4, 4], stage3_channels)
        stage4_channels = [width, width * 2, width * 4, width * 8]
        self.transition3 = self._make_transition(channels, stage4_channels)
        self.stage4, channels = self._make_stage(
            3, [4, 4, 4, 4], stage4_channels, multi_scale_output=False,
        )
        self.final_layer = nn.Conv2d(channels[0], num_joints, 1)
        self.model_name = f"HRNet-W{width}"
        self._initialize()

    def _make_layer(self, block, channels, blocks, stride=1):
        output = channels * block.expansion
        downsample = None
        if stride != 1 or self.inplanes != output:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, output, 1, stride, bias=False),
                nn.BatchNorm2d(output, momentum=BN_MOMENTUM),
            )
        layers = [block(self.inplanes, channels, stride, downsample)]
        self.inplanes = output
        layers.extend(block(self.inplanes, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    @staticmethod
    def _make_transition(previous, current):
        layers = []
        for index, channels in enumerate(current):
            if index < len(previous):
                if channels == previous[index]:
                    layers.append(nn.Identity())
                else:
                    layers.append(nn.Sequential(
                        conv3x3(previous[index], channels),
                        nn.BatchNorm2d(channels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True),
                    ))
            else:
                modules = []
                source = previous[-1]
                for step in range(index + 1 - len(previous)):
                    target = channels if step == index - len(previous) else source
                    modules.extend((
                        conv3x3(source, target, 2),
                        nn.BatchNorm2d(target, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True),
                    ))
                    source = target
                layers.append(nn.Sequential(*modules))
        return nn.ModuleList(layers)

    @staticmethod
    def _make_stage(modules, blocks, channels, multi_scale_output=True):
        values = []
        in_channels = list(channels)
        for index in range(modules):
            output_all = multi_scale_output or index < modules - 1
            module = HighResolutionModule(
                len(channels), blocks, in_channels, channels, output_all,
            )
            values.append(module)
            in_channels = module.get_num_inchannels()
        return nn.Sequential(*values), in_channels

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.001)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, value):
        value = self.relu(self.bn1(self.conv1(value)))
        value = self.relu(self.bn2(self.conv2(value)))
        value = self.layer1(value)
        values = [layer(value) for layer in self.transition1]
        values = self.stage2(values)
        values = [
            layer(values[index] if index < len(values) else values[-1])
            for index, layer in enumerate(self.transition2)
        ]
        values = self.stage3(values)
        values = [
            layer(values[index] if index < len(values) else values[-1])
            for index, layer in enumerate(self.transition3)
        ]
        return self.final_layer(self.stage4(values)[0])


def build_hrnet(width=32, num_joints=16):
    if width not in {32, 48}:
        raise ValueError(f"Unsupported HRNet width: {width}")
    return PoseHighResolutionNet(width, num_joints)
