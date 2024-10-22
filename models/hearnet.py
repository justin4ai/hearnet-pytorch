import torch
import torch.nn as nn
import torch.nn.functional as F

class HearNet(nn.Module):
    
    def __init__(self, in_channels = 6, out_channels = 3):
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

        self.unpool4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        self.unpool3 = nn.ConvTranspose2d(in_channels=512, out_channels=256,
                                    kernel_size=2, stride=2, padding=0, bias=True)
        self.unpool2 = nn.ConvTranspose2d(in_channels=256, out_channels=128,
                                    kernel_size=2, stride=2, padding=0, bias=True)
        self.unpool1 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                    kernel_size=2, stride=2, padding=0, bias=True)

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

        hearnet_stackedinput = torch.cat([yhat_st, h_error], dim=1)
        #print("hearnet_stackedinput : ", hearnet_stackedinput.shape)

        enc1 = self.encoder1(hearnet_stackedinput)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))
        
        #print("enc4_shape : ", enc4.shape)
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2)) #  #(channel) = 1024
        #print("bottleneck_shape : ", bottleneck.shape)
        # Expansive path
        #dec4 = F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=True)
        
        
        dec4 = torch.cat([enc4, self.unpool4(bottleneck)], 1) # upsampling and skip connection at once
        #print("dec4_shape : ", dec4.shape)
        dec4 = self.decoder4(dec4)

        #dec3 = F.interpolate(dec4, scale_factor=2, mode='bilinear', align_corners=True)
        dec3 = torch.cat([enc3, self.unpool3(dec4)], 1) # skip connection
        dec3 = self.decoder3(dec3)

        #dec2 = F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True)
        dec2 = torch.cat([enc2, self.unpool2(dec3)], 1) # skip connection
        dec2 = self.decoder2(dec2)

        #dec1 = F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True)
        dec1 = torch.cat([enc1, self.unpool1(dec2)], 1) # skip connection
        dec1 = self.decoder1(dec1)

        # Final layer
        output = self.final_layer(dec1)
        
        #print("hearnet_output : ",output.shape)

        return output