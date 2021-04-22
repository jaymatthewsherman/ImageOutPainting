import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UNET(nn.Module):
    def __init__(self, config, in_channels=4, out_channels=3, intermediate_channels=[64,128,256,512]):
        super(UNET, self).__init__()
        self.config = config
        self.downward_blocks = nn.ModuleList()
        self.upward_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # All downward blocks
        for i in range(len(intermediate_channels)):
            in_c = intermediate_channels[i-1] if i > 0 else in_channels
            out_c = intermediate_channels[i]

            self.downward_blocks.append(ConvBlock(in_c, out_c))

        # All upward blocks
        for i in reversed(range(len(intermediate_channels))):
            in_c = intermediate_channels[i] * 2 # accounts for skip connection
            out_c = intermediate_channels[i]

            self.upward_blocks.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2))
            self.upward_blocks.append(ConvBlock(in_c, out_c))
        
        self.bottleneck = ConvBlock(intermediate_channels[-1], intermediate_channels[-1]*2)

        # resize to the stripe width
        # (nw−kw+pw+1) = sw
        kernel_width = config.pic_width + 3 - config.stripe_width 
        self.final_conv = nn.Conv2d(intermediate_channels[0], out_channels, kernel_size=(3, kernel_width), padding=1)
    
    def forward(self, x):
        skip_connections = []
        
        for down_block in self.downward_blocks:
            x = down_block(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections.reverse()
        for idx in range(0, len(self.upward_blocks), 2):
            x = self.upward_blocks[idx](x)
            skip_connection = skip_connections[idx//2]
            x = self.upward_blocks[idx+1](torch.cat((skip_connection, x), dim=1))
        
        return self.final_conv(x)