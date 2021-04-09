import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.block(x)

class Discriminator(nn.Module):
    def __init__(self, config, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.config = config

        # concat x and y across rgb
        self.start = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=1, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        
        blocks = list()
        for idx in range(len(features)):
            if idx == 0:
                continue
            blocks.append(ConvBlock(features[idx-1], features[idx]))
        self.blocks = nn.Sequential(*blocks)

        self.end = nn.Conv2d(features[-1], 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")

    def forward(self, x, y):
        # either collapse across channels or dimensions
        d = 3 if self.config.should_collapse else 1
        x = torch.cat((x, y), dim=d)
        filename = f"D:\Senior Year Northeastern University\DS Capstone\image_outpainting\Pix2PixCollapse\disc.txt"
        with open(filename, "a") as f:
            f.write(f"x: {x}\n")
        x = self.start(x)
        with open(filename, "a") as f:
            f.write(f"start: {x}\n")
        x = self.blocks(x)
        with open(filename, "a") as f:
            f.write(f"blocks: {x}\n")
        x = self.end(x)
        with open(filename, "a") as f:
            f.write(f"end: {x}\n\n")
        return x
        

        