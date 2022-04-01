import torch
import numpy as np
import torch.nn as nn
from torchvision.transforms import Resize
from base_classes import Conv, ConvT, conv_transformation


class Generator(nn.Module):

    def __init__(self,
                 rand_size,
                 out_channels,
                 num_filter=32,
                 num_transpose=3):

        super(Generator, self).__init__()
        assert int(np.cbrt(rand_size)) ** 3 == rand_size, "Data should be a perfect cube!"
        assert num_filter // 2 ** num_transpose > 0, "Invalid scaling, update num_filter or num_transpose"
        self.base_shape = int(np.cbrt(rand_size))
        self.in_block = ConvT(self.base_shape, num_filter, 1, 1, 0)
        self.convT_block = nn.ModuleList(
            [
                ConvT(
                    num_filter // (2 ** level), num_filter // (2 ** (level + 1)), 3, 2, 0)
                for level in range(num_transpose - 1)
            ])
        self.out_block = ConvT(
            num_filter // (2 ** (num_transpose - 1)), out_channels, 3, 2, 0)

    def forward(self, x):
        x = x.view(x.shape[0], self.base_shape, self.base_shape, self.base_shape)
        x = self.in_block(x)
        for block in self.convT_block:
            x = block(x)
        x = self.out_block(x)
        return x


class Critic(nn.Module):

    def __init__(self,
                 in_channels,
                 h=100,
                 w=100,
                 num_conv=3,
                 base_filter=8):
        super(Critic, self).__init__()
        self.H, self.W = h, w
        self.resize_block = Resize((self.H, self.W))
        self.num_conv = num_conv
        self.shape_transform()
        self.in_block = Conv(in_channels, base_filter, 3, 2)
        self.conv_block = nn.ModuleList(
            [
                Conv(
                    base_filter * (2 ** level), base_filter * (2 ** (level + 1)), 3, 2)
                for level in range(num_conv)
            ]
        )
        self.linear_dim = self.H * self.W * (2 ** num_conv) * base_filter
        self.fc_block = nn.Sequential(
            nn.Linear(self.linear_dim, 1024),
            nn.Linear(1024, 128),
            nn.Linear(128, 1)
        )

    def shape_transform(self):
        for _ in range(self.num_conv + 1):
            self.H = conv_transformation(self.H, 3, 2, 0)
            self.W = conv_transformation(self.W, 3, 2, 0)

    def forward(self, x):
        x = self.resize_block(x)
        x = self.in_block(x)
        for block in self.conv_block:
            x = block(x)
        x = x.view(x.shape[0], -1)
        x = self.fc_block(x)
        return x


def main():
    batch_size = 2
    rand_size = 512
    image_height = image_width = 224
    in_channels = 3

    gen_model = Generator(rand_size, in_channels, 32, 5)
    critic_model = Critic(in_channels, h=image_height, w=image_width, num_conv=4)

    generated_patch = gen_model(torch.rand(batch_size, rand_size))
    assert generated_patch.shape[0] == batch_size
    assert generated_patch.shape[1] == in_channels

    output_patch = critic_model(generated_patch)
    assert output_patch.shape[0] == batch_size
    assert output_patch.shape[1] == 1

    print("Success!!!")


if __name__ == "__main__":
    main()
