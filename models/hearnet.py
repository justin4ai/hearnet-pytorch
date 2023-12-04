import torch
import torch.nn as nn
import torch.nn.functional as F

class HearNet():
    
    def __init__(self, in_channels, out_channels):
        super(HearNet, self).__init__()

        # Contracting path
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Expansive path
        self.decoder4 = self.conv_block(1024, 512)
        self.decoder3 = self.conv_block(512, 256)
        self.decoder2 = self.conv_block(256, 128)
        self.decoder1 = self.conv_block(128, 64)

        # Final layer
        self.final_layer = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels): # Two consecutive conv layers maintaining #(channel)
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, yhat_st, h_error):
        """
        Forward pass of the network.

        Args:
        - yhat_st (torch.Tensor): The swapped image.
        - h_error (torch.Tensor): The heuristic error, calculated as x_t - yhat_tt (reconstructed x_t by AEI-Net).

        Returns:
        - torch.Tensor: The output of the network.
        """
        # Contracting path
        enc1 = self.encoder1(yhat_st, h_error)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Expansive path
        dec4 = F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=True)
        dec4 = torch.cat([enc4, dec4], 1)
        dec4 = self.decoder4(dec4)

        dec3 = F.interpolate(dec4, scale_factor=2, mode='bilinear', align_corners=True)
        dec3 = torch.cat([enc3, dec3], 1)
        dec3 = self.decoder3(dec3)

        dec2 = F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True)
        dec2 = torch.cat([enc2, dec2], 1)
        dec2 = self.decoder2(dec2)

        dec1 = F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True)
        dec1 = torch.cat([enc1, dec1], 1)
        dec1 = self.decoder1(dec1)

        # Final layer
        output = self.final_layer(dec1)

        return output



