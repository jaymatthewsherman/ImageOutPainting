import torch
import torch.nn as nn

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.blocks(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.blocks(x)

class Pix2Pix(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 248, 512]):
        super().__init__()
        assert len(features) == 4, "must use 4 features"

        self.start = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        self.down1 = DownBlock(features[0], features[1])
        self.down2 = DownBlock(features[1], features[2])
        self.down3 = DownBlock(features[2], features[3])
        self.down4 = DownBlock(features[3], features[3])
        self.down5 = DownBlock(features[3], features[3])
        self.down6 = DownBlock(features[3], features[3])

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[3], features[3], 4, 2, 1, padding_mode="reflect"),
            nn.ReLU()
        )

        self.up1 = UpBlock(features[3]*1, features[3])
        self.up2 = UpBlock(features[3]*2, features[3])
        self.up3 = UpBlock(features[3]*2, features[3])
        self.up4 = UpBlock(features[3]*2, features[3])
        self.up5 = UpBlock(features[3]*2, features[2])
        self.up6 = UpBlock(features[2]*2, features[1])
        self.up7 = UpBlock(features[1]*2, features[0])

        self.end = nn.Sequential(
            nn.ConvTranspose2d(features[0]*2, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.start(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        x = self.bottleneck(d7)
        x = self.up1(x)
        x = self.up2(torch.cat((x, d7), dim=1))
        del d7
        x = self.up3(torch.cat((x, d6), dim=1))
        del d6
        x = self.up4(torch.cat((x, d5), dim=1))
        del d5
        x = self.up5(torch.cat((x, d4), dim=1))
        del d4
        x = self.up6(torch.cat((x, d3), dim=1))
        del d3
        x = self.up7(torch.cat((x, d2), dim=1))
        del d2
        return self.end(torch.cat((x, d1), dim=1))

class Collapse(nn.Module):
    def __init__(self, config, channels=3):
        super().__init__()
        if config.should_collapse:
            assert(config.stripe_width == 12)

        blocks = []
        kernel_width = 21
        num_blocks = config.pic_width // kernel_width
        for _ in range(num_blocks):
            blocks.append(nn.Sequential(
                nn.Conv2d(channels, channels, 
                    kernel_size=(3, kernel_width), bias=True, padding=(1, 0), stride=1, padding_mode="reflect"),
                nn.BatchNorm2d(channels),
                nn.ReLU()
            ))
        
        self.blocks = nn.Sequential(*blocks)
        self.end = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=(3, 7), bias=True, padding=1, stride=1, padding_mode="reflect"),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.blocks(x)
        return self.end(x)

class Generator(nn.Module):
    def __init__(self, config, in_channels=3, out_channels=3):
        super().__init__()
        self.config = config
        self.pix2pix = Pix2Pix(in_channels, out_channels)
        self.collapse = Collapse(config, out_channels)

    def forward(self, x):
        x = self.pix2pix(x)
        if self.config.should_collapse:
            return self.collapse(x)
        return x