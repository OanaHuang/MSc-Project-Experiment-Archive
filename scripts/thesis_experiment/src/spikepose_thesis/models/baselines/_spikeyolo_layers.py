# SPDX-License-Identifier: AGPL-3.0-only
"""Unmodified inference/training layers from BICLab/SpikeYOLO.

Source: ultralytics/nn/modules/yolo_spikformer.py, commit
 dc51f35d0b6e970cd157080c98157a41af4392f5.
Only the layers used by the P3 heatmap adapter are retained. Imports for the
unused detection trainer have been removed; layer bodies are unchanged.
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

decay = 0.25


class mem_update(nn.Module):
    def __init__(self, act=False):
        super(mem_update, self).__init__()
        # self.actFun= torch.nn.LeakyReLU(0.2, inplace=False)

        self.act = act
        self.qtrick = MultiSpike4()  # change the max value

    def forward(self, x):

        spike = torch.zeros_like(x[0]).to(x.device)
        output = torch.zeros_like(x)
        mem_old = 0
        time_window = x.shape[0]
        for i in range(time_window):
            if i >= 1:
                mem = (mem_old - spike.detach()) * decay + x[i]

            else:
                mem = x[i]
            spike = self.qtrick(mem)

            mem_old = mem.clone()
            output[i] = spike
        # print(output[0][0][0][0])
        return output


class MultiSpike4(nn.Module):

    class quant4(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=4))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 4] = 0
            return grad_input

    def forward(self, x):
        return self.quant4.apply(x)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class BNAndPadLayer(nn.Module):
    def __init__(
            self,
            pad_pixels,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
    ):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (
                        self.bn.bias.detach()
                        - self.bn.running_mean
                        * self.bn.weight.detach()
                        / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(
                    self.bn.running_var + self.bn.eps
                )
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0: self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0: self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps


class RepConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size = 3,
            bias=False,
            group = 1
    ):
        super().__init__()
        padding = int((kernel_size-1)/2)
        # hidden_channel = in_channel
        conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=group)
        bn = BNAndPadLayer(pad_pixels=padding, num_features=in_channel)
        conv3x3 = nn.Sequential(
            # mem_update(), #11111
            nn.Conv2d(in_channel, in_channel, kernel_size, 1,0, groups=in_channel, bias=False),  # grouped convolution here as well
            # mem_update(),  #11111
            nn.Conv2d(in_channel, out_channel, 1,  1,0, groups=group, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)


class SepRepConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size = 3,
            bias=False,
            group = 1
    ):
        super().__init__()
        padding = int((kernel_size-1)/2)
        # hidden_channel = in_channel
        bn = BNAndPadLayer(pad_pixels=padding, num_features=in_channel)
        conv3x3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1,0, groups=group, bias=False),  # grouped convolution here as well
            # mem_update(), #11111
            nn.Conv2d(out_channel, out_channel, kernel_size,  1,0, groups=out_channel, bias=False),
        )


        self.body = nn.Sequential(bn, conv3x3)

    def forward(self, x):
        return self.body(x)


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(self,
                 dim,
                 expansion_ratio=2,
                 act2_layer=nn.Identity,
                 bias=False,
                 kernel_size=3,  #7,3
                 padding=1):
        super().__init__()
        padding = int((kernel_size -1)/2)
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.dwconv2 = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size, #7*7
            padding=padding, groups=med_channels, bias=bias)  # depthwise conv
#         self.pwconv3 = nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias,groups=1)
        self.pwconv3=SepRepConv(med_channels, dim)  # replace the final SepConv layer with a reparameterized convolution; retained after an approximately 0.5-point gain

        self.bn1 = nn.BatchNorm2d(med_channels)
        self.bn2 = nn.BatchNorm2d(med_channels)
        self.bn3 = nn.BatchNorm2d(dim)


        self.lif1 = mem_update()
        self.lif2 = mem_update()
        self.lif3 = mem_update()

    def forward(self, x):
        T, B, C, H, W = x.shape
#         print("x.shape:",x.shape)
        x = self.lif1(x) # x1_lif:0.2328  x2_lif:0.0493; the smaller x2 mean remains small after BN and LIF, producing a lower firing rate, while the larger x1 mean produces a higher firing rate
        x = self.bn1(self.pwconv1(x.flatten(0, 1))).reshape(T, B, -1, H, W)  # flatten dimensions 0 through 1
        x = self.lif2(x)
        x = self.bn2(self.dwconv2(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        x = self.lif3(x)
        x = self.bn3(self.pwconv3(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        return x


class MS_ConvBlock(nn.Module):
    def __init__(self, input_dim, mlp_ratio=4.,sep_kernel_size = 7 ,full=False):  # in_channels(out_channels), internal expansion ratio
        super().__init__()

        self.full =full
        self.Conv = SepConv(dim=input_dim,kernel_size= sep_kernel_size)  # 2x internal expansion
        self.mlp_ratio = mlp_ratio

        self.lif1 = mem_update()
        self.lif2 = mem_update()

        self.conv1 = RepConv(input_dim, int(input_dim * mlp_ratio)) # models other than 137 do not group the first block

        self.bn1 = nn.BatchNorm2d(int(input_dim * mlp_ratio))  # possible improvement point

        self.conv2 = RepConv(int(input_dim * mlp_ratio), input_dim)
        self.bn2 = nn.BatchNorm2d(input_dim)  # possible improvement point




    # @get_local('x_feat')
    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.Conv(x) + x  #sepconv  pw+dw+pw

        x_feat = x

        x = self.bn1(self.conv1(self.lif1(x).flatten(0, 1))).reshape(T, B, int(self.mlp_ratio * C), H, W)
            # RepConv corresponds to ConvMixer and contains 1x1, 3x3, and 1x1 convolutions equivalent to one 3x3 convolution
        x = self.bn2(self.conv2(self.lif2(x).flatten(0, 1))).reshape(T, B, C, H, W)
        x = x_feat + x

        return x


class MS_AllConvBlock(nn.Module):  # standard conv
    def __init__(self, input_dim, mlp_ratio=4.,sep_kernel_size = 7 ,group=False):  # in_channels(out_channels), internal expansion ratio
        super().__init__()

        self.Conv = SepConv(dim=input_dim,kernel_size= sep_kernel_size)

        self.mlp_ratio = mlp_ratio
        self.conv1 = MS_StandardConv(input_dim, int(input_dim * mlp_ratio),3)
        self.conv2 = MS_StandardConv(int(input_dim * mlp_ratio), input_dim,3)



    # @get_local('x_feat')
    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.Conv(x) + x  #sepconv  pw+dw+pw

        x_feat = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = x_feat + x

        return x


class MS_StandardConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):  # in_channels(out_channels), internal expansion ratio
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.s = s
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.lif = mem_update()

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.bn(self.conv(self.lif(x).flatten(0, 1))).reshape(T, B, self.c2, int(H / self.s), int(W / self.s))
        return x


class MS_DownSampling(nn.Module):
    def __init__(self, in_channels=2, embed_dims=256, kernel_size=3, stride=2, padding=1, first_layer=True):
        super().__init__()

        self.encode_conv = nn.Conv2d(in_channels, embed_dims, kernel_size=kernel_size, stride=stride, padding=padding)
        self.encode_bn = nn.BatchNorm2d(embed_dims)
        if not first_layer:
            self.encode_lif = mem_update()
        # self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        T, B, _, _, _ = x.shape


        if hasattr(self, "encode_lif"): # not the first layer
            # x_pool = self.pool(x)
            x = self.encode_lif(x)


        x = self.encode_conv(x.flatten(0, 1))
        _, C, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguous()

        return x


class MS_GetT(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, T=4):
        super().__init__()
        self.T = T
        self.in_channels = in_channels

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        return x


class SpikeConv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.lif = mem_update()
        self.bn = nn.BatchNorm2d(c2)
        self.s = s
        # self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)
        x = self.lif(x)
        x = self.bn(self.conv(x.flatten(0, 1))).reshape(T, B, -1, H_new, W_new)
        return x


class SpikeSPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = SpikeConv(c1, c_, 1, 1)
        self.cv2 = SpikeConv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            T, B, C, H, W = x.shape
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x.flatten(0, 1)).reshape(T, B, -1, H, W)
            y2 = self.m(y1.flatten(0, 1)).reshape(T, B, -1, H, W)
            y3 = self.m(y2.flatten(0, 1)).reshape(T, B, -1, H, W)
            return self.cv2(torch.cat((x, y1, y2, y3), 2))
