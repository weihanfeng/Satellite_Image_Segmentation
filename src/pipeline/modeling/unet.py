import torch
import torch.nn as nn


class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, feature_nums=[64, 128, 256, 512]):
        super().__init__()
        self.down_conv = nn.ModuleList()
        self.up_conv = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Down convolution
        for feature_num in feature_nums:
            self.down_conv.append(DoubleConvolution(in_channels, feature_num))
            in_channels = feature_num
        
        # Up convolution
        for feature_num in reversed(feature_nums):
            self.up_conv.append(
                nn.ConvTranspose2d(
                    feature_num * 2, feature_num, kernel_size=2, stride=2
                )
            )
            self.up_conv.append(DoubleConvolution(feature_num * 2, feature_num))
        
        # Bottom convolution
        self.bottom_conv = DoubleConvolution(feature_nums[-1], feature_nums[-1] * 2)

        # Output convolution
        self.output_conv = nn.Conv2d(feature_nums[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Down convolution and save skip connections to be used in up convolution
        skip_connections = []
        for down_conv in self.down_conv:
            x = down_conv(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottom_conv(x)
        skip_connections = skip_connections[::-1] # reverse the list

        # Up convolution
        for idx in range(0, len(self.up_conv), 2):
            x = self.up_conv[idx](x)
            skip_connection = skip_connections[idx//2] # because we have 2 layers in up convolution for each skip connection
            if x.shape != skip_connection.shape:
                # bilinear interpolation to match the shape of skip connection
                # due to maxpooling, the shape of x may be smaller than skip connection due to floor division
                # e.g. input shape (also skip connection) = [161, 161], output shape = [80, 80], but up_conv shape = [160, 160]
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.up_conv[idx+1](concat_skip)

        return self.output_conv(x)