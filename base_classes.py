import torch.nn as nn


def convt_transformation(in_shape, kernel_size, stride=1, padding=0, dialation=1):
    out_shape = (in_shape - 1) * stride - 2 * padding + dialation * (kernel_size - 1) + padding + 1
    return out_shape


def conv_transformation(in_shape, kernel_size, stride=1, padding=0):
    out_shape = ((in_shape - kernel_size + 2 * padding) // stride) + 1
    return out_shape


class ConvT(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(0, 0)):
        super(ConvT, self).__init__()

        self.neg_slope = 1e-2
        self.convT_block = nn.Sequential(
            nn.LeakyReLU(self.neg_slope),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
        )

    def forward(self, x):
        return self.convT_block(x)


class Conv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=(1, 1),
                 padding=(0, 0),
                 momentum=0.15):
        super(Conv, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_block(x)
