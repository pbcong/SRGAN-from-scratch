import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, discriminator: bool = False, use_act: bool = True, use_bn: bool = True, **kwargs):
        super().__init__()
        self.use_act = use_act
        self.conv = nn.Conv2d(in_channels, out_channels,
                              **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True) if discriminator else nn.PReLU(
            num_parameters=out_channels)

    def forward(self, x):
        return self.act(self.bn(self.conv(x))) if self.use_act else self.bn(self.conv(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels *
                              scale_factor**2, 3, 1, 1)
        self.ps = nn.PixelShuffle(scale_factor)
        self.act = nn.PReLU(num_parameters=in_channels)

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block1 = conv_block(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.block2 = conv_block(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1, use_act=False)

    def forward(self, x):
        output = self.block1(x)
        output = self.block2(output)
        return output + x


class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=16):
        super().__init__()
        self.input = conv_block(in_channels, num_channels,
                                use_bn=False, kernel_size=9, stride=1, padding=4)
        self.residuals = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.conv = conv_block(num_channels, num_channels,
                               use_act=False, kernel_size=3, stride=1, padding=1)
        self.upsamples = nn.Sequential(UpsampleBlock(
            num_channels, 2), UpsampleBlock(num_channels, 2))
        self.final = nn.Conv2d(num_channels, in_channels,
                               kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        initial = self.input(x)
        x = self.residuals(initial)
        x = self.conv(x) + initial
        x = self.upsamples(x)
        return torch.tanh(self.final(x))


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(
                conv_block(
                    in_channels,
                    feature,
                    kernel_size=3,
                    stride=1 + idx % 2,
                    padding=1,
                    discriminator=True,
                    use_act=True,
                    use_bn=False if idx == 0 else True,
                )
            )
            in_channels = feature

        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            # make sure the output is what we expect, if the model works properly suppose to do nothing
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.blocks(x)
        return self.classifier(x)


def test():
    low_resolution = 24  # 96x96 -> 24x24
    with torch.cuda.amp.autocast():
        x = torch.randn((5, 3, low_resolution, low_resolution))
        gen = Generator()
        gen_out = gen(x)
        disc = Discriminator()
        disc_out = disc(gen_out)

        print(gen_out.shape)
        print(disc_out.shape)


if __name__ == "__main__":
    test()
