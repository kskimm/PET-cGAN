import torch
import torch.nn as nn


def conv2d_block(in_channels, out_channels, norm_layer, activation, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        activation, 
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        norm_layer(out_channels)
    )
            

class Patch_discriminator(nn.Module):
    def __init__(self, cond_channels, real_channels, numLayers = 3, norm_layer = nn.BatchNorm2d, ndf = 64):
        super(Patch_discriminator, self).__init__()

        reLU = nn.LeakyReLU(negative_slope = 0.2, inplace = True) 
        initLayer = nn.Conv2d(in_channels=cond_channels + real_channels, out_channels=ndf, kernel_size=4, stride=2, padding = 1) #256 -> 255
        self.layers = [initLayer]
        
        out_channel = ndf
        for i in range(numLayers):
            in_channel = out_channel
            out_channel = in_channel *2 if in_channel < ndf * 8 else ndf * 8
            if i == numLayers - 1:
                stride = 1
            else:
                stride = 2
            self.layers.append(conv2d_block(in_channels=in_channel, out_channels=out_channel, norm_layer = norm_layer, activation = reLU, stride = stride))


        self.layers.append(nn.Sequential(
            reLU,
            nn.Conv2d(in_channels=ndf*8, out_channels=1, kernel_size=4, stride=1, padding = 1)
        ))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x1, x2):
        return self.layers(torch.cat((x1, x2), dim = 1))
