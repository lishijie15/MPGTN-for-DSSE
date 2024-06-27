import torch
import torch.nn as nn


# TKY Final SOTA!


def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DepthSeparableConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, **kwargs):
        super(DepthSeparableConv2d, self).__init__()

        ## DW
        self.depthConv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1, groups=in_channel, **kwargs),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )
        # PW
        self.pointConv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (1, 1)),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthConv(x)
        x = self.pointConv(x)
        return x



class MobileNetV1(nn.Module):

    def __init__(self, in_channels, out_channels, scale=1.0):
        super(MobileNetV1, self).__init__()

        input_channel = make_divisible(in_channels * scale, 8)

        depthSeparableConvSize = [
            # in, out, s
            [input_channel, input_channel // 2, 1],
            [input_channel // 2, input_channel, 1]
        ]

        conv1 = []
        for i, o, s in depthSeparableConvSize:
            output_channel = make_divisible(o * scale, 8)
            conv1.append(DepthSeparableConv2d(input_channel, output_channel, bias=False))
            input_channel = output_channel
        self.conv1 = nn.Sequential(*conv1)

        last_channel = make_divisible(out_channels * scale, 8)
        self.conv2 = ConvBNReLU(input_channel, last_channel, 1, 1, 0, bias=False)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x