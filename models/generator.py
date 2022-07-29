import torch
import torch.nn as nn


def conv2d_block(in_channels, out_channels, norm_layer, activation, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        activation, 
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        norm_layer(out_channels)
    )
            
def upconv2d_block(in_channels, out_channels, norm_layer, activation, dropout = None, kernel_size=4, stride=2, padding=1):
    if dropout is None:
        layer = nn.Sequential(
            activation,
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            norm_layer(out_channels)
        )
    else:
        layer = nn.Sequential(
            activation,
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            norm_layer(out_channels),
            dropout
        )
    return layer


class UNet_generator(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout, numFold = 8, ngf = 64, norm_layer = nn.BatchNorm2d):
        super(UNet_generator, self).__init__()
        self.use_dropout = use_dropout
        down_ReLU = nn.LeakyReLU(negative_slope = 0.2, inplace = True) 
        up_ReLU = nn.ReLU(inplace = True) 
        dropout = nn.Dropout(p = 0.5) 

        self.conv2d_blocks = []
        self.conv2d_blocks.append(nn.Conv2d(in_channels=in_channels, out_channels=ngf, kernel_size=4, stride=2, padding=1))
        
        out_channel = ngf
        num_channel_list = [out_channel]
        for _ in range(numFold - 2):
            in_channel = out_channel
            out_channel = out_channel*2 if out_channel < ngf*8 else ngf*8
            self.conv2d_blocks.append(conv2d_block(in_channels = in_channel, out_channels = out_channel, norm_layer = norm_layer, activation = down_ReLU))
            num_channel_list.append(out_channel)
        # 64 128 258 512 512 512 512
        self.conv2d_blocks.append(nn.Sequential(
            down_ReLU,
            nn.Conv2d(in_channels=ngf*8, out_channels=ngf*8, kernel_size=4, stride=2, padding=1)
        ))

        self.conv2d_blocks = nn.Sequential(*self.conv2d_blocks)
        self.upconv2d_blocks = []
        self.upconv2d_blocks.append(upconv2d_block(in_channels = ngf*8, out_channels = ngf*8, norm_layer = norm_layer, activation = up_ReLU))

        for _ in range(numFold - 2):
            in_channel = num_channel_list[-1]*2
            out_channel = num_channel_list[-2]
            if use_dropout and in_channel == ngf*16 and out_channel == ngf*8:
                self.upconv2d_blocks.append(upconv2d_block(in_channels = in_channel, out_channels = out_channel, norm_layer = norm_layer, activation = up_ReLU, dropout = dropout))
            else: 
                self.upconv2d_blocks.append(upconv2d_block(in_channels = in_channel, out_channels = out_channel, norm_layer = norm_layer, activation = up_ReLU))
            num_channel_list.pop()

        self.upconv2d_blocks.append(nn.Sequential(
            up_ReLU,
            nn.ConvTranspose2d(in_channels=128, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        ))

        self.upconv2d_blocks = nn.Sequential(*self.upconv2d_blocks)


    def forward(self, x):
        eList = []
        for layer in self.conv2d_blocks:      
            x = layer(x)
            eList.append(x)
            
        for level, layer in enumerate(self.upconv2d_blocks):
            if level == 0:
                x = layer(x)
            else:
                x = layer(torch.cat((x, eList[-(level+1)]), dim = 1))
        return x

