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


def conv2d_block2(in_channels, out_channels, norm_layer, activation, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        norm_layer(out_channels),
        activation 
    )
            
def upconv2d_block2(in_channels, out_channels, norm_layer, activation, kernel_size=4, stride=2, padding=1):
     return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding =1),
        norm_layer(out_channels),
        activation,
    )


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class Resnet_generator(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout = False, ngf = 64, norm_layer = nn.BatchNorm2d):
        super(Resnet_generator, self).__init__()
        self.use_dropout = use_dropout
        down_ReLU = nn.ReLU(inplace = True) 
        up_ReLU = nn.ReLU(inplace = True) 
        
        model = [nn.ReflectionPad2d(3)]
        model += [conv2d_block2(in_channels = in_channels, out_channels = ngf, norm_layer = norm_layer, activation = down_ReLU, kernel_size =7, stride = 1, padding = 0)]
        model += [conv2d_block2(in_channels = ngf, out_channels = ngf*2, norm_layer = norm_layer, activation = down_ReLU, kernel_size =3, stride = 2, padding = 1)]
        model += [conv2d_block2(in_channels = ngf*2, out_channels = ngf*4, norm_layer = norm_layer, activation = down_ReLU, kernel_size =3, stride = 2, padding = 1)]
        
        for _ in range(6):
            model += [ResnetBlock(dim = ngf*4, padding_type = 'reflect', norm_layer = norm_layer, use_dropout = False, use_bias = False)]
        
        model += [upconv2d_block2(in_channels = ngf*4, out_channels = ngf*2, norm_layer = norm_layer, activation = up_ReLU, kernel_size =3, stride = 2, padding = 1)]
        model += [upconv2d_block2(in_channels = ngf*2, out_channels = ngf*1, norm_layer = norm_layer, activation = up_ReLU, kernel_size =3, stride = 2, padding = 1)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(in_channels = ngf, out_channels = out_channels, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, input):
        return self.model(input)