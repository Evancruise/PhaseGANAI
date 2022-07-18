import torch
import math
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torchvision import models
from collections import namedtuple
import torch.nn.functional as F
import tensorflow as tf
from models.options import ParamOptions
import time

import torch
from torch.nn.modules.module import Module
from torch.autograd import Function

class ResUNet(nn.Module):
    def __init__(self, n_out, in_nc=4, out_nc=3):
        super(ResUNet, self).__init__()
        
        nb=2
        nc=[64, 128, 256, 512]
        act_mode='R'
        downsample_mode='strideconv'
        upsample_mode='convtranspose'

        self.m_head = B.conv(n_out, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body  = B.sequential(*[B.ResBlock(nc[3], nc[3], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x):
        
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h/8)*8-h)
        paddingRight = int(np.ceil(w/8)*8-w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)

        x = x[..., :h, :w]

        return x
        
def conv3x(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class Residual(nn.Module):
    def __init__(self, n_channels=64):
        super(Residual, self).__init__()
        self.n_channels = n_channels
        self.conv1 = nn.Conv2d(in_channels=self.n_channels,
                               out_channels=self.n_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.n_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=self.n_channels,
                               out_channels=self.n_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(self.n_channels)

    def forward(self, x):
        input = x
        output = self.relu(self.conv2(self.relu(self.conv1(x))))
        return output

class UNet(nn.Module):
    """Pretrained U-Net generator based on TernausNet"""
    def __init__(self, n_out =1, batch_size = 28, pretrained=False):
        """
        if pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with VGG11
        """
        super(UNet, self).__init__()
        num_filters=32
        self.channels = n_out
        self.n_out = n_out
        self.batch_size = batch_size
        #self.pool = nn.MaxPool2d(2, 2)
        self.pool1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.pool3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.pool4 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.pool5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        
        self.encoder = models.vgg11(pretrained=pretrained).features
        self.conv321 = ConvRelu(1, 3)
        self.relu = self.encoder[1]
        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]
        self.center = DecoderBlock(num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8)
        self.dec5 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4)
        self.dec3 = DecoderBlock(num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(num_filters * (4 + 2), num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)
        self.final = nn.Conv2d(num_filters, n_out, kernel_size=1)
        
        self.vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), self.vgg_pretrained_features[x])
        
    def forward(self, x):
        conv321 = self.conv321(x[:,0,:,:].unsqueeze(1))
        conv1 = self.relu(self.conv1(conv321))
        
        conv2 = self.relu(self.conv2(self.pool1(conv1)))
        conv3s = self.relu(self.conv3s(self.pool2(conv2)))
        conv3 = self.relu(self.conv3(conv3s))
        conv4s = self.relu(self.conv4s(self.pool3(conv3s)))
        conv4 = self.relu(self.conv4(conv4s))
        conv5s = self.relu(self.conv5s(self.pool4(conv4s)))
        conv5 = self.relu(self.conv5(conv5s))

        center = self.center(self.pool5(conv5s))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        outx = self.final(dec1)
        
        if self.n_out == 2:
            
            conv321m = self.conv321(x[:,1,:,:].unsqueeze(1))
            conv1m = self.relu(self.conv1(conv321m))
            conv2m = self.relu(self.conv2(self.pool1(conv1m)))
            conv3sm = self.relu(self.conv3s(self.pool2(conv2m)))
            conv3m = self.relu(self.conv3(conv3sm))
            conv4sm = self.relu(self.conv4s(self.pool3(conv3m)))
            conv4m = self.relu(self.conv4(conv4s))
            conv5sm = self.relu(self.conv5s(self.pool4(conv4m)))
            conv5m = self.relu(self.conv5(conv5sm))

            centerm = self.center(self.pool5(conv5m))

            dec5m = self.dec5(torch.cat([centerm, conv5m], 1))
            dec4m = self.dec4(torch.cat([dec5m, conv4m], 1))
            dec3m = self.dec3(torch.cat([dec4m, conv3m], 1))
            dec2m = self.dec2(torch.cat([dec3m, conv2m], 1))
            dec1m = self.dec1(torch.cat([dec2m, conv1m], 1))
            outm = self.final(dec1m)
            
            outx = outx+x[:,0,:,:].unsqueeze(1)
            
            out = torch.cat((outx, outm), 1)
        else:
            out = torch.cat((outx+x[:,0,:,:].unsqueeze(1), x[:,1,:,:].unsqueeze(1)), 1)
        return out

class UNet_phasegan(nn.Module):
    """Pretrained U-Net generator based on TernausNet"""
    def __init__(self, n_out=1, batch_size = 28, pretrained=False):
        """
        if pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with VGG11
        """
        super().__init__()

        self.n_out = n_out
        self.num_filters = 32
        self.batch_size = batch_size
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = models.vgg11(pretrained=True).features
        self.conv321 = ConvRelu(1, 3)
        self.relu = self.encoder[1]
        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]
        self.center = DecoderBlock(self.num_filters * 8 * 2, self.num_filters * 8 * 2, self.num_filters * 8)
        self.dec5 = DecoderBlock(self.num_filters * (16 + 8), self.num_filters * 8 * 2, self.num_filters * 8)
        self.dec4 = DecoderBlock(self.num_filters * (16 + 8), self.num_filters * 8 * 2, self.num_filters * 4)
        self.dec3 = DecoderBlock(self.num_filters * (8 + 4), self.num_filters * 4 * 2, self.num_filters * 2)
        self.dec2 = DecoderBlock(self.num_filters * (4 + 2), self.num_filters * 2 * 2, self.num_filters)
        self.dec1 = ConvRelu(self.num_filters * (2 + 1), self.num_filters)
        self.final = nn.Conv2d(self.num_filters, n_out, kernel_size=1)
        
        self.vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), self.vgg_pretrained_features[x])

    def forward(self, x):
        
        conv321 = self.conv321(x[:,0,:,:].unsqueeze(1))
        conv1 = self.relu(self.conv1(conv321))
        
        conv2 = self.relu(self.conv2(self.pool(conv1)))
        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        
        conv3 = self.relu(self.conv3(conv3s))
        conv4s = self.relu(self.conv4s(self.pool(conv3s)))
        conv4 = self.relu(self.conv4(conv4s))
        
        conv5s = self.relu(self.conv5s(self.pool(conv4s)))
        
        conv5 = self.relu(self.conv5(conv5s))
        
        center = self.center(self.pool(conv5s))
        
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        outx = self.final(dec1)
        
        if self.n_out == 2:
            
            conv321m = self.conv321(x[:,1,:,:].unsqueeze(1))
            conv1m = self.relu(self.conv1(conv321m))
            conv2m = self.relu(self.conv2(self.pool(conv1m)))
            conv3sm = self.relu(self.conv3s(self.pool(conv2m)))
            conv3m = self.relu(self.conv3(conv3sm))
            conv4sm = self.relu(self.conv4s(self.pool(conv3m)))
            conv4m = self.relu(self.conv4(conv4s))
            conv5sm = self.relu(self.conv5s(self.pool(conv4m)))
            conv5m = self.relu(self.conv5(conv5sm))

            centerm = self.center(self.pool(conv5m))

            dec5m = self.dec5(torch.cat([centerm, conv5m], 1))
            dec4m = self.dec4(torch.cat([dec5m, conv4m], 1))
            dec3m = self.dec3(torch.cat([dec4m, conv3m], 1))
            dec2m = self.dec2(torch.cat([dec3m, conv2m], 1))
            dec1m = self.dec1(torch.cat([dec2m, conv1m], 1))
            outm = self.final(dec1m)
            
            out = torch.cat((x[:,0,:,:].unsqueeze(1) + outx, x[:,1,:,:].unsqueeze(1) + outm), 1)
            
        else:
            out = torch.cat((x[:,0,:,:].unsqueeze(1) + outx, x[:,1,:,:].unsqueeze(1)), 1)
            
        return out

class BaseNet(nn.Module): # 1 U-net
    def __init__(self, input_channels=1,
    encoder=[64, 128, 256, 512], decoder=[1024, 512, 256], output_channels=1):
        super(BaseNet, self).__init__()
        
        self.debug = True
        self.input_size = 96 # Side length of square image patch
        self.batch_size = 1 # Batch size of patches Note: 11 gig gpu will max batch of 5
        self.val_batch_size = 4 # Number of images shown in progress
        self.test_batch_size = 1 # We only use the first part of the model here (forward_encoder), so it can be larger
        self.verbose_testing = False

        self.k = 64 # Number of classes
        self.num_epochs = 3#250 for real
        self.data_dir = "./datasets/BSDS300/" # Directory of images
        self.showdata = False # Debug the data augmentation by showing the data we're training on.

        self.useInstanceNorm = True # Instance Normalization
        self.useBatchNorm = False # Only use one of either instance or batch norm
        self.useDropout = True
        self.drop = 0.2

        # Each item in the following list specifies a module.
        # Each item is the number of input channels to the module.
        # The number of output channels is 2x in the encoder, x/2 in the decoder.
        self.encoderLayerSizes = [64, 128, 256]
        self.decoderLayerSizes = [512, 256]

        self.showSegmentationProgress = True
        self.segmentationProgressDir = './latent_images/'

        self.variationalTranslation = 0 # Pixels, 0 for off. 1 works fine

        self.saveModel = True

        layers = [
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.drop),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.drop),
        ]

        if not self.useInstanceNorm:
            layers = [layer for layer in layers if not isinstance(layer, nn.InstanceNorm2d)]
        if not self.useBatchNorm:
            layers = [layer for layer in layers if not isinstance(layer, nn.BatchNorm2d)]
        if not self.useDropout:
            layers = [layer for layer in layers if not isinstance(layer, nn.Dropout)]

        self.first_module = nn.Sequential(*layers)


        self.pool = nn.MaxPool2d(2, 2)
        self.enc_modules = nn.ModuleList(
            [ConvModule(channels, 2*channels) for channels in encoder])


        decoder_out_sizes = [int(x/2) for x in decoder]
        self.dec_transpose_layers = nn.ModuleList(
            [nn.ConvTranspose2d(channels, channels, 2, stride=2) for channels in decoder]) # Stride of 2 makes it right size
        self.dec_modules = nn.ModuleList(
            [ConvModule(3*channels_out, channels_out) for channels_out in decoder_out_sizes])
        self.last_dec_transpose_layer = nn.ConvTranspose2d(128, 128, 2, stride=2)

        layers = [
            nn.Conv2d(128+64, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.drop),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.drop),

            nn.Conv2d(64, output_channels, 1), # No padding on pointwise
            nn.ReLU(),
        ]

        if not self.useInstanceNorm:
            layers = [layer for layer in layers if not isinstance(layer, nn.InstanceNorm2d)]
        if not self.useBatchNorm:
            layers = [layer for layer in layers if not isinstance(layer, nn.BatchNorm2d)]
        if not self.useDropout:
            layers = [layer for layer in layers if not isinstance(layer, nn.Dropout)]

        self.last_module = nn.Sequential(*layers)


    def forward(self, x):
        x1 = self.first_module(x)
        activations = [x1]
        for module in self.enc_modules:
            activations.append(module(self.pool(activations[-1])))

        x_ = activations.pop(-1)

        for conv, upconv in zip(self.dec_modules, self.dec_transpose_layers):
            skip_connection = activations.pop(-1)
            x_ = conv(
                torch.cat((skip_connection, upconv(x_)), 1)
            )

        segmentations = self.last_module(
            torch.cat((activations[-1], self.last_dec_transpose_layer(x_)), 1)
        )
        return segmentations


def upsample(ch_coarse, ch_fine):
  return nn.Sequential(
    nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
    nn.ReLU()
  )
  
def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False):
  if useBN:
    return nn.Sequential(
      nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.BatchNorm2d(dim_out),
      nn.LeakyReLU(0.1),
      nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.BatchNorm2d(dim_out),
      nn.LeakyReLU(0.1)
    )
  else:
    return nn.Sequential(
      nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.ReLU(),
      nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.ReLU()
    )
    
class DownConv(nn.Module):
    def __init__(self, in_feat, out_feat, drop_rate=0.4, bn_momentum=0.1):
        super(DownConv, self).__init__()
        self.conv1 = nn.Conv2d(in_feat, out_feat, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(out_feat, momentum=bn_momentum)
        self.conv1_drop = nn.Dropout2d(drop_rate)

        self.conv2 = nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(out_feat, momentum=bn_momentum)
        self.conv2_drop = nn.Dropout2d(drop_rate)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv1_bn(x)
        x = self.conv1_drop(x)

        x = F.relu(self.conv2(x))
        x = self.conv2_bn(x)
        x = self.conv2_drop(x)
        return x
        
class UpConv(nn.Module):
    def __init__(self, in_feat, out_feat, drop_rate=0.4, bn_momentum=0.1):
        super(UpConv, self).__init__()
        self.up1 = nn.functional.interpolate
        self.downconv = DownConv(in_feat, out_feat, drop_rate, bn_momentum)

    def forward(self, x, y):
        x = self.up1(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, y], dim=1)
        x = self.downconv(x)
        return x

class Block(nn.Module):
    def __init__(self, in_filters, out_filters, seperable=False):
        super(Block, self).__init__()
        
        if seperable:
            
            self.spatial1=nn.Conv2d(in_filters, in_filters, kernel_size=3, groups=in_filters, padding=1)
            self.depth1=nn.Conv2d(in_filters, out_filters, kernel_size=1)
            
            self.conv1=lambda x: self.depth1(self.spatial1(x))
            
            self.spatial2=nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1, groups=out_filters)
            self.depth2=nn.Conv2d(out_filters, out_filters, kernel_size=1)
            
            self.conv2=lambda x: self.depth2(self.spatial2(x))
            
        else:
            
            self.conv1=nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)
            self.conv2=nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1)
        
        self.batchnorm1=nn.BatchNorm2d(out_filters)
        self.batchnorm2=nn.BatchNorm2d(out_filters)

    def forward(self, x):
        x=self.conv1(x).clamp(0)
        x=self.conv2(x).clamp(0)
        return x

class UEnc(nn.Module):
    def __init__(self, squeeze, ch_mul=64, in_chans=1):
        super(UEnc, self).__init__()
        
        self.channels = squeeze
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.enc1=Block(in_chans, ch_mul, seperable=True)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.enc2=Block(ch_mul, 2*ch_mul, seperable=True)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.enc3=Block(2*ch_mul, 4*ch_mul, seperable=True)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.enc4=Block(4*ch_mul, 8*ch_mul, seperable=True)
        
        self.middle=Block(8*ch_mul, 16*ch_mul)
        
        self.up1 = nn.ConvTranspose2d(16*ch_mul, 8*ch_mul, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec1=Block(16*ch_mul, 8*ch_mul)
        self.up2=nn.ConvTranspose2d(8*ch_mul, 4*ch_mul, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec2=Block(8*ch_mul, 4*ch_mul)
        self.up3=nn.ConvTranspose2d(4*ch_mul, 2*ch_mul, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec3=Block(4*ch_mul, 2*ch_mul)
        self.up4=nn.ConvTranspose2d(2*ch_mul, ch_mul, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec4=Block(2*ch_mul, ch_mul, seperable=False)
        
        self.final=nn.Conv2d(ch_mul, squeeze, kernel_size=(1, 1))
    
    def inite_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data=torch.ones(m.weight.data.shape)*300
                # nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()
        print('[finishing]:assign weight by inite_weight')
        
    def forward(self, x):
        
        enc1=self.relu(self.enc1(x[:,0,:,:].unsqueeze(1)))
        enc2=self.relu(self.enc2(self.conv1(enc1)))
        enc3=self.relu(self.enc3(self.conv2(enc2)))
        enc4=self.relu(self.enc4(self.conv3(enc3)))
        
        middle=self.relu(self.middle(self.conv4(enc4)))
        
        up1=torch.cat([enc4, self.up1(middle)], 1)
        dec1=self.dec1(up1)
        
        up2=torch.cat([enc3, self.up2(dec1)], 1)
        dec2=self.dec2(up2)
        
        up3=torch.cat([enc2, self.up3(dec2)], 1)
        dec3=self.dec3(up3)
        
        up4=torch.cat([enc1, self.up4(dec3)], 1)
        dec4=self.dec4(up4)
        
        final=self.final(dec4)
        
        if self.channels == 2:
            enc1m=self.relu(self.enc1(x[:,1,:,:].unsqueeze(1)))
        
            enc2m=self.relu(self.enc2(self.conv1(enc1m)))
        
            enc3m=self.relu(self.enc3(self.conv2(enc2m)))
        
            enc4m=self.relu(self.enc4(self.conv3(enc3m)))
        
            middlem=self.relu(self.middle(self.conv4(enc4m)))
        
            up1m=torch.cat([enc4m, self.up1(middlem)], 1)
            dec1m=self.dec1(up1m)
        
            up2m=torch.cat([enc3m, self.up2(dec1m)], 1)
            dec2m=self.dec2(up2m)
        
            up3m=torch.cat([enc2m, self.up3(dec2m)], 1)
            dec3m=self.dec3(up3m)
        
            up4m=torch.cat([enc1m, self.up4(dec3m)], 1)
            dec4m=self.dec4(up4m)
        
            finalm=self.final(dec4m)
            
            return torch.cat((final, finalm), 1)
        
        else:
            return torch.cat((final, x[:,1,:,:].unsqueeze(1)), 1)

class UDec(nn.Module):
    def __init__(self, squeeze, ch_mul=64, in_chans=3):
        super(UDec, self).__init__()
        
        self.channels = squeeze
        self.relu = nn.ReLU(inplace=True)
        self.enc1=Block(squeeze, ch_mul, seperable=False)
        self.enc2=Block(ch_mul, 2*ch_mul)
        self.enc3=Block(2*ch_mul, 4*ch_mul)
        self.enc4=Block(4*ch_mul, 8*ch_mul)
        
        self.middle=Block(8*ch_mul, 16*ch_mul)
        
        self.up1=nn.ConvTranspose2d(16*ch_mul, 8*ch_mul, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.dec1=Block(16*ch_mul, 8*ch_mul)
        self.up2=nn.ConvTranspose2d(8*ch_mul, 4*ch_mul, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.dec2=Block(8*ch_mul, 4*ch_mul)
        self.up3=nn.ConvTranspose2d(4*ch_mul, 2*ch_mul, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.dec3=Block(4*ch_mul, 2*ch_mul)
        self.up4=nn.ConvTranspose2d(2*ch_mul, ch_mul, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.dec4=Block(2*ch_mul, ch_mul, seperable=False)
        
        self.final=nn.Conv2d(ch_mul, in_chans, kernel_size=(1, 1))
    
    def inite_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data=torch.ones(m.weight.data.shape)*300
                
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()
        print('[finishing]:assign weight by inite_weight')
        
    def forward(self, x):
        
        enc1 = self.relu(self.enc1(x[:,0,:,:].unsqueeze(1)))
        enc2 = self.relu(self.enc2(self.conv1(enc1)))
        enc3 = self.relu(self.enc3(self.conv2(enc2)))
        enc4 = self.relu(self.enc4(self.conv3(enc3)))
        
        middle = self.relu(self.middle(self.conv4(enc4)))
        
        up1 = torch.cat([enc4, self.up1(middle)], 1)
        dec1 = self.dec1(up1)
        up2 = torch.cat([enc3, self.up2(dec1)], 1)
        dec2 = self.dec2(up2)
        up3 = torch.cat([enc2, self.up3(dec2)], 1)
        dec3 =self.dec3(up3)
        up4 = torch.cat([enc1, self.up4(dec3)], 1)
        dec4 = self.dec4(up4)
        
        final=self.final(dec4)
        
        if self.channels == 2:
            enc1m = self.relu(self.enc1(x[:,1,:,:].unsqueeze(1)))
            enc2m = self.relu(self.enc2(self.conv1(enc1m)))
            enc3m = self.relu(self.enc3(self.conv2(enc2m)))
            enc4m = self.relu(self.enc4(self.conv3(enc3m)))
        
            middlem = self.middle(self.conv4(enc4m))
        
            up1m = torch.cat([enc4m, self.up1(middlem)], 1)
            dec1m = self.dec1(up1m)
            up2m = torch.cat([enc3m, self.up2(dec1m)], 1)
            dec2m = self.dec2(up2m)
            up3m = torch.cat([enc2m, self.up3(dec2m)], 1)
            dec3m =self.dec3(up3m)
            up4m = torch.cat([enc1m, self.up4(dec3m)], 1)
            dec4m = self.dec4(up4m)
        
            finalm=self.final(dec4m)
            return torch.cat((final, finalm), 1)
        else:   
            return torch.cat((final, x[:,1,:,:].unsqueeze(1)), 1)

class WNet_DC(nn.Module):
    def __init__(self, n_out, batch_size, ch_mul=64, in_chans=1, out_chans=1):
        super(WNet_DC, self).__init__()
        out_chans=in_chans
        self.channels = n_out
        self.batch_size = batch_size
        self.relu = nn.ReLU(inplace=False)
        self.UEnc=UEnc(1, ch_mul, 1)
        self.UDec=UDec(1, ch_mul, 1)
        
        self.vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), self.vgg_pretrained_features[x])
    
    def forward(self, x):
        
        enc = self.UEnc(x)
        dec=self.UDec(enc)
        
        if self.channels == 2:
            hm = self.slice1(torch.cat((a[:,1,:,:].unsqueeze(1), a[:,1,:,:].unsqueeze(1), a[:,1,:,:].unsqueeze(1)), 1))
            h_relu1_2m = hm
            hm = self.slice2(hm)
            h_relu2_2m = hm
            hm = self.slice3(hm)
            h_relu3_3m = hm
            hm = self.slice4(hm)
            h_relu4_3m = hm
            hm = self.slice5(hm)
            h_relu5_3m = hm
            out_x1m = h_relu5_3m
            out = torch.cat((x[:,0,:,:].unsqueeze(1) + dec[:,0,:,:].unsqueeze(1), x[:,1,:,:].unsqueeze(1) + dec[:,1,:,:].unsqueeze(1)), 1)
        else:
            out = torch.cat((x[:,0,:,:].unsqueeze(1) + dec[:,0,:,:].unsqueeze(1), x[:,1,:,:].unsqueeze(1)), 1)
            
        return out #, out_vgg

class WNet(torch.nn.Module):
    def __init__(self, n_out=1, batch_size=28):
        super(WNet, self).__init__()
        self.feature1 = []
        self.feature2 = []
        self.MaxLv = 10
        self.radius = 4
        self.batch_size = batch_size
        bias = True
        #U-Net1
        #module1
        self.module = []
        self.maxpool1 = []
        self.uconv1 = []
        self.module.append(
            self.add_conv_stage(n_out,32,3,padding=1,seperable=False)   
        )
        
        self.module.append(self.add_conv_stage(32,64,3,padding=1))
        self.module.append(self.add_conv_stage(64,128,3,padding=1))  
        self.module.append(self.add_conv_stage(128,256,3,padding=1))  
        self.module.append(self.add_conv_stage(256,512,3,padding=1))  
        self.module.append(self.add_conv_stage(512,1024,3,padding=1)) 
        
        #module1-4
        for i in range(self.MaxLv-1):
            self.maxpool1.append(nn.MaxPool2d(2))
        
        self.uconv1.append(nn.ConvTranspose2d(1024,512,2,2,bias = True))
        self.uconv1.append(nn.ConvTranspose2d(512,256,2,2,bias = True))
        self.uconv1.append(nn.ConvTranspose2d(256,128,2,2,bias = True))
        self.uconv1.append(nn.ConvTranspose2d(128,64,2,2,bias = True))
        self.uconv1.append(nn.ConvTranspose2d(64,32,2,2,bias = True))
        
        self.predconv = nn.Conv2d(32,1,1,bias = bias)
        self.pad = nn.ConstantPad2d(self.radius-1,0)
        self.softmax = nn.Softmax2d()
        self.module = torch.nn.ModuleList(self.module)
        self.maxpool1 = torch.nn.ModuleList(self.maxpool1)
        self.uconv1 = torch.nn.ModuleList(self.uconv1)
        #self.loss = NcutsLoss()
        
        self.vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), self.vgg_pretrained_features[x])
    
    def im2double(self, im):
         min_val = torch.min(im)
         max_val = torch.max(im)
         #print(maximum, minimum)
         out = (im - min_val) / (max_val - min_val) * 255.0
         #out = im
         out = (out - 127.5) / 127.5
         return out
         
    def batch_fftshift2d(self,x):
        # Provided by PyTorchSteerablePyramid
        real, imag = torch.unbind(x, -1)
        for dim in range(1, len(real.size())):
            n_shift = real.size(dim)//2
            if real.size(dim) % 2 != 0:
                n_shift += 1  # for odd-sized images
            real = self.roll_n(real, axis=dim, n=n_shift)
            imag = self.roll_n(imag, axis=dim, n=n_shift)
        return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

    def batch_ifftshift2d(self, x):
        real, imag = torch.unbind(x, -1)
        for dim in range(len(real.size()) - 1, 0, -1):
            real = self.roll_n(real, axis=dim, n=real.size(dim)//2)
            imag = self.roll_n(imag, axis=dim, n=imag.size(dim)//2)
        return torch.stack((real, imag), -1)  # last dim=2 (real&imag)
    
    def add_conv_stage(self,dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, seperable=True):
        if seperable:
            return nn.Sequential(
                nn.Conv2d(dim_in,dim_out,1,bias = bias),
                nn.Conv2d(dim_out,dim_out,kernel_size,padding = padding,groups = dim_out,bias = bias),
                nn.ReLU(),
                nn.BatchNorm2d(dim_out),
                nn.Conv2d(dim_out,dim_out,1,bias = bias),
                nn.Conv2d(dim_out,dim_out,kernel_size,padding = padding,groups = dim_out,bias = bias),
                nn.ReLU(),
                nn.BatchNorm2d(dim_out),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(dim_in,dim_out,kernel_size,padding = padding,bias = bias),
                nn.ReLU(),
                nn.BatchNorm2d(dim_out),
                nn.Conv2d(dim_out,dim_out,kernel_size,padding = padding,bias = bias),
                nn.ReLU(),
                nn.BatchNorm2d(dim_out),
            )
    
    def forward(self,x):
        
        a = x
        self.feature1_phase = [a[:,0,:,:].unsqueeze(1)]
        self.feature1_phase.append(self.module[0](a[:,0,:,:].unsqueeze(1)))
        for i in range(1,6):
            tempf = self.maxpool1[i-1](self.feature1_phase[-1])
            self.feature1_phase.append(self.module[i](tempf))
        
        for i in range(5):
            tempf = self.uconv1[i](self.feature1_phase[6-i])
            tempf = self.feature1_phase[5-i]+tempf # torch.cat((self.feature1[2*config.MaxLv-i-1],tempf),dim=1)
            #if 2-i >= 0:
            #    self.feature1_phase.append(self.module[5+i](tempf))
        
        tempf = self.feature1_phase[1]+tempf
        
        self.feature2_phase = self.predconv(tempf)
        
        self.feature1_absorption = [a[:,1,:,:].unsqueeze(1)]
        #U-Net1
        self.feature1_absorption.append(self.module[0](a[:,1,:,:].unsqueeze(1)))
        for i in range(1,6):
            tempf = self.maxpool1[i-1](self.feature1_absorption[-1])
            self.feature1_absorption.append(self.module[i](tempf))
        
        # for i in range(config.MaxLv,2*config.MaxLv-2):
        for i in range(5):
            '''
            tempf = self.uconv1[i-config.MaxLv](self.feature1_phase[-1])
            tempf = self.feature1_phase[2*config.MaxLv-i-1]+tempf # torch.cat((self.feature1[2*config.MaxLv-i-1],tempf),dim=1)
            self.feature1_phase.append(self.module[i](tempf))
            '''
            tempf = self.uconv1[i](self.feature1_absorption[6-i])
            tempf = self.feature1_absorption[5-i]+tempf # torch.cat((self.feature1[2*config.MaxLv-i-1],tempf),dim=1)
            #if 2-i >= 0:
            #    self.feature1_phase.append(self.module[5+i](tempf))
                
        #tempf = self.uconv1[-1](self.feature1_phase[-1])
        #tempf = self.uconv1[-1](tempf)
        tempf = self.feature1_absorption[1]+tempf #torch.cat((self.feature1[1],tempf),dim=1)
        #tempf = self.module[-1](tempf)
        self.feature2_absorption = self.predconv(tempf)
        #self.feature2_absorption = self.softmax(tempf)
        
        self.phase_output = x[:,0,:,:].unsqueeze(1) + self.feature2_phase
        self.absorption_output = x[:,1,:,:].unsqueeze(1) + self.feature2_absorption
        
        for i in range(10):
            self.feature2_phase[i,0,:,:] = self.im2double(self.feature2_phase[i,0,:,:])
            self.feature2_absorption[i,0,:,:] = self.im2double(self.feature2_absorption[i,0,:,:])
        
        return torch.cat((x[:,0,:,:].unsqueeze(1) + self.feature2_phase, x[:,1,:,:].unsqueeze(1) + self.feature2_absorption), 1)
        
class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(64, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output,identity_data)
        return output 

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),  # I added BN layers compared with the vanilla Unet
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=2, stride=2
    )

class SRResNet(nn.Module):
    def __init__(self, num_out=1, batch_size=28):
        super(SRResNet, self).__init__()
        
        self.channels = num_out
        self.conv1 = double_conv(num_out, 64)
        self.conv2 = double_conv(64, 128)
        self.conv3 = double_conv(128, 256)
        self.conv4 = double_conv(256, 512)
        self.bottleneck = double_conv(512, 1024)
        self.up_conv5 = up_conv(1024, 512)
        self.conv5 = double_conv(1024, 512)
        self.up_conv6 = up_conv(512, 256)
        self.conv6 = double_conv(512, 256)
        self.up_conv7 = up_conv(256, 128)
        self.conv7 = double_conv(256, 128)
        self.up_conv8 = up_conv(128, 64)
        self.conv8 = double_conv(128, 64)
        self.conv9 = nn.Conv2d(64, num_out, kernel_size=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_input = nn.Conv2d(in_channels=num_out, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        self.residual = self.make_layer(_Residual_Block, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)
        
        self.batch_size = batch_size

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=num_out, kernel_size=9, stride=1, padding=4, bias=False)
        self.channel = 1
        self._weights_init()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def process_data(self, data):
        # normalization by channels
        #print('data.shape:', data.shape)
        #data = np.clip(np.fabs(data), np.min(data), np.max(data))
        for i in range(10):
            data[i,:,:] -= torch.min(data[i,:,:])
            data[i,:,:] /= torch.max(data[i,:,:])
        return data
        
    def _weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)
    
    def im2double_inverse(self, x, maximum, minimum):
        
        for i in range(self.batch_size):
            for j in range(self.channels):
                x[i,j,:,:] = (x[i,j,:,:] * (maximum[i,j] - minimum[i,j])) + minimum[i,j]
        
        return x
    
    def im2double(self, x):
        
        maximum = torch.zeros((self.batch_size, self.channels))
        minimum = torch.zeros((self.batch_size, self.channels))
        
        for i in range(self.batch_size):
            for j in range(self.channels):
                maximum[i,j] = (torch.max(x[i,j,:,:]))
                minimum[i,j] = (torch.min(x[i,j,:,:]))
                x[i,j,:,:] = ((x[i,j,:,:] - minimum[i,j])/(maximum[i,j] - minimum[i,j]))
        
        return x, maximum, minimum
    
    def forward(self, x):
        
        out1 = self.relu(self.conv_input(torch.unsqueeze(x[:,0,:,:], 1)))
        residual = out1
        out1 = self.residual(out1)
        out1 = self.bn_mid(self.conv_mid(out1))
        out1 = torch.add(out1,residual)
        out1 = self.upscale4x(out1)
        out1 = self.conv_output(out1)
        
        conv1 = self.conv1(torch.unsqueeze(x[:,0,:,:], 1))
        x = self.maxpool(conv1)
        
        conv2 = self.conv2(x)
        x = self.maxpool(conv2)

        conv3 = self.conv3(x)
        x = self.maxpool(conv3)

        conv4 = self.conv4(x)
        x = self.maxpool(conv4)

        bottleneck = self.bottleneck(x)

        x = self.up_conv5(bottleneck)
        x = torch.cat([x, conv4], dim=1)
        x = self.conv5(x)

        x = self.up_conv6(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv8(x)

        x = self.conv9(x)
        x1 = self.sigmoid(x)
        
        return torch.cat(torch.unsqueeze(x[:,0,:,:] + x1[:,0,:,:], 1), torch.unsqueeze(x[:,1,:,:], 1)), 1)

class _Residual_Block(nn.Module):
    def __init__(self,bn = True):
        super(_Residual_Block, self).__init__()
        
        self.bn = bn
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU(num_parameters=1,init=0.02)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        if bn:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(64)
        
    def forward(self, x):
        output = self.conv1(x)
        if self.bn:
            output =self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        if self.bn:
            output = self.bn2(output)
        output = torch.add(output,x)
        return output 
        
class SRResNet_RGBY(nn.Module):
    def __init__(self, num_out=1):
        """
        in and out channel = io_channels;RGB mode io_channels =3 ,Y channel io_channels = 1
        """
        super(SRResNet_RGBY, self).__init__()
        
        out1_channels = 1
        out2_channels = 1
        bn = True
        
        self.bn = bn
        self.n_channel = num_out
        self.conv_input = nn.Conv2d(in_channels = out1_channels, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        
        #self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.PReLU(num_parameters=1,init=0.02)
        #self.relu = nn.ReLU(inplace=False)
        
        self.residual = self.make_layer(_Residual_Block, bn, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        if self.bn:
            self.bn_mid = nn.BatchNorm2d(64)

        '''
        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.PReLU(num_parameters=1,init=0.2),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.PReLU(num_parameters=1,init=0.2),
        )
        '''
        
        self.upscale4x = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False)

        self.conv_output = nn.Conv2d(in_channels=64, out_channels = out1_channels, kernel_size=3, stride=2, padding=1, bias=False)
        #self.conv_output2 = nn.Conv2d(in_channels = out1_channels, out_channels = out2_channels, kernel_size=1, stride=1, padding=4, bias=False)
        
        # init the weight of conv2d
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #init.orthogonal(m.weight, math.sqrt(2))
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
        
        # vgg16 features
        self.vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), self.vgg_pretrained_features[x])
                
    def make_layer(self, block, bn, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(bn = bn))
        return nn.Sequential(*layers)
    
    def swish(self, x):
        return x * torch.sigmoid(x)
    
    def im2double(self, x):
    
        for i in range(10):
            maximum = torch.max(x[i,0,:,:])
            minimum = torch.min(x[i,0,:,:])
            x[i,0,:,:] = ((x[i,0,:,:] - minimum) / (maximum - minimum)*2-1)
            
        return x

    def forward(self, x):
        
        #plt.imshow(x[0,0,:,:].detach().numpy())
        #plt.show()
        
        a = x
        
        a1 = self.im2double(a[:,0,:,:].unsqueeze(1))
        
        out1 = self.relu(self.conv_input(a1))
        #residual = out.clone()
        out1_1 = self.residual(out1)
        
        out1_2 = self.conv_mid(out1_1)
        if self.bn:
            out1_2 = self.bn_mid(out1_2)
        out1_3 = torch.add(out1_1,out1_2)
        out1_4 = self.upscale4x(out1_3)
        #print('out1_4.shape:', out1_4.shape)
        #rgb = self.conv_output(out)
        #y = self.conv_output2(rgb)
        y = self.conv_output(out1_4)
        # print(y.size())
        # torch.cat((rgb,y), dim = 1)   # channel
        
        #print('y.shape:', y.shape)
        
        '''
        fig, ax = plt.subplots(1,8)
        ax[0].imshow(out1[0,0,:,:].detach().numpy())
        ax[1].imshow(out1_1[0,0,:,:].detach().numpy())
        ax[2].imshow(out1_2[0,0,:,:].detach().numpy())
        ax[3].imshow(out1_3[0,0,:,:].detach().numpy())
        ax[4].imshow(out1_4[0,0,:,:].detach().numpy())
        ax[5].imshow(x[0,0,:,:].detach().numpy())
        ax[6].imshow(y[0,0,:,:].detach().numpy())
        ax[7].imshow(x[0,0,:,:].detach().numpy() + y[0,0,:,:].detach().numpy())
        plt.show()
        '''
        
        h = self.slice1(torch.cat((a1, a1, a1), 1))
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        #vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        x1_out = h_relu5_3 #vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        
        if self.n_channel == 2:
            a2 = self.im2double(a[:,1,:,:].unsqueeze(1))
            out2 = self.relu(self.conv_input(a2))
            #residual = out.clone()
            out2_1 = self.residual(out2)
            out2_2 = self.conv_mid(out2_1)
            
            if self.bn:
                out2_2 = self.bn_mid(out2_2)
            
            out2_3 = torch.add(out2_1,out2_2)
            out2_4 = self.upscale4x(out2_3)
            #rgb = self.conv_output(out)
            #y = self.conv_output2(rgb)
            y2 = self.conv_output(out2_4)
            # print(y.size())
        
            h = self.slice1(torch.cat((a2, a2, a2), 1))
            h_relu1_2 = h
            h = self.slice2(h)
            h_relu2_2 = h
            h = self.slice3(h)
            h_relu3_3 = h
            h = self.slice4(h)
            h_relu4_3 = h
            h = self.slice5(h)
            h_relu5_3 = h
            #vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
            x2_out = h_relu5_3 #vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        
            return torch.cat((x[:,0,:,:].unsqueeze(1) + y[:,0,:,:].unsqueeze(1), x[:,1,:,:].unsqueeze(1) + y2[:,0,:,:].unsqueeze(1)), 1), torch.cat((x1_out[:,0,:,:].unsqueeze(1), x2_out[:,0,:,:].unsqueeze(1)), 1)
        else:
            #print(x)
            #print(y)
            return torch.cat((x[:,0,:,:].unsqueeze(1) + y[:,0,:,:].unsqueeze(1), x[:,1,:,:].unsqueeze(1)), 1), x1_out[:,0,:,:].unsqueeze(1)
        
class NN_Network(nn.Module):
    def __init__(self,in_dim,hid,out_dim):
        super(NN_Network, self).__init__()
        self.linear1 = nn.Linear(in_dim,hid)
        self.linear2 = nn.Linear(hid,out_dim)
        self.linear1.weight = torch.nn.Parameter(torch.zeros(in_dim,hid))
        self.linear1.bias = torch.nn.Parameter(torch.ones(hid))
        self.linear2.weight = torch.nn.Parameter(torch.zeros(in_dim,hid))
        self.linear2.bias = torch.nn.Parameter(torch.ones(hid))

    def forward(self, input_array):
        h = self.linear1(input_array)
        y_pred = self.linear2(h)
        return y_pred

class Noise_cov_reduction(nn.Module):
    def __init__(self, n_out = 1, batch_size=28, useBN=True):
        super(Noise_cov_reduction, self).__init__()
        
        self.batch_size = batch_size
        self.flatten1 = nn.Flatten()
        self.encoder_dense1 = nn.Sequential(
            nn.Linear(128*128, 128*128),
            #nn.BatchNorm2d(128*128),
            nn.PReLU(128*128)
        )
        self.encoder_dense2 = nn.Sequential(
            nn.Linear(128*128, 128*128),
            #nn.BatchNorm2d(128*128),
            nn.PReLU(128*128)
        )
        self.decoder_dense1 = nn.Sequential(
            nn.Linear(128*128, 128*128),
            #nn.BatchNorm2d(128*128),
            nn.PReLU(128*128)
        )
        self.decoder_dense2 = nn.Sequential(
            nn.Linear(128*128, 128*128),
            #nn.BatchNorm2d(128*128),
            nn.PReLU(128*128)
        )
        self.decoder_covar_dense1 = nn.Sequential(
            nn.Linear(128*128, 128*128*2),
            #nn.BatchNorm2d(128*128*2),
            nn.PReLU(128*128*2)
        )
        self.decoder_covar_1_half_dense = nn.Sequential(
            nn.Linear(128*128*2, 128*128*2),
            #nn.BatchNorm2d(128*128*2),
            nn.PReLU(128*128*2)
        )
        # Build the network, one encoder, and two decoders
    
    def inite_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data=torch.ones(m.weight.data.shape)*300
                # nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()
        print('[finishing]:assign weight by inite_weight')
        
    def exponentiate_diag(self, x):
        x0 = torch.exp(x[..., 0:1])
        return torch.cat([x0, x[..., 1:]], axis=-1)
    
    def concat_with_zeros(self, x):
        zeros_shape = x.shape
        #print('x:', x.shape)
        zeros_shape -= np.array([0, 0, 0, 1])
        zeros = torch.zeros((self.batch_size, 128, 128, 2))
        #zeros.set_shape((None, 128, 128, 2))
        return torch.cat([zeros, x], axis=-1)
    
    '''
    def decoder_covar(self, z, n_h=256):
        # Decoder for the covariance of the Gaussian distribution
        # from low dim (n_z) to image space with extra channels (w, h, (nb // 2) + 1)
        h0 = self.decoder_covar_dense1(z) #keras.layers.Dense(n_h, activation=tf.nn.relu)(z)

        chol_half_weights = self.decoder_covar_1_half_dense(h0) #keras.layers.Dense((w * h * ((nb // 2) + 1)), activation=None)(h0)
        chol_half_weights = chol_half_weights.view(10, 128, 128, 2) #keras.layers.Reshape((w, h, (nb // 2) + 1))(chol_half_weights)

        # The first channel contains the log_diagonal of the cholesky matrix
        log_diag_chol_precision = chol_half_weights[:,:,:,0] #keras.layers.Lambda(lambda x: x[..., 0])(chol_half_weights)
        log_diag_chol_precision = log_diag_chol_precision.view(10, 128, 128, 1) #keras.layers.Flatten(input_shape=(w, h), name='log_diag_chol_precision')(log_diag_chol_precision)
        
        # Exponentiate to remove the log from the diagonal
        chol_half_weights = self.exponentiate_diag(chol_half_weights) #keras.layers.Lambda(lambda x: exponentiate_diag(x))(chol_half_weights)

        # Concatenate with zeros to have a nb = k*k kernel per pixel, output size is (w, h, nb)
        chol_precision_weights = self.concat_with_zeros(chol_half_weights) #keras.layers.Lambda(lambda x: concat_with_zeros(x), name='chol_precision_weights')(chol_half_weights)

        return chol_precision_weights, log_diag_chol_precision
    
    def decoder_mean(self, z, n_h=128):
        # Decoder for the means of the Gaussian distribution
        # from low dim (n_z) to image space (w, h), use sigmoid as data is in [0, 1]
        l0 = self.decoder_dense1(z) #keras.layers.Dense(n_h, activation=tf.nn.relu)(z)
        out_mean = self.decoder_dense2(l0) #keras.layers.Dense((w * h), activation=tf.nn.sigmoid)(l0)
        return out_mean.view(10, 1, 128, 128) #keras.layers.Reshape((w, h), name='out_mean')(out_mean)
    
    def encoder(self, input_tensor, n_z=128, n_h=128):
        # Encode image to low dim
        h0 = self.flatten1(input_tensor) #keras.layers.Flatten(input_shape=(w, h))(input_tensor)
        #print('h0:', h0.shape)
        h1 = self.encoder_dense1(h0) #keras.layers.Dense(n_h, activation=tf.nn.relu)(h0)
        #print('h1:', h1.shape)
        z = self.encoder_dense2(h1) #keras.layers.Dense(n_z, activation=tf.nn.relu, name='z')(h1)
        #print('z:', z.shape)
        return z
    '''
    
    def encoder(self, input_tensor, n_z=256, n_h=256):
        # Encode image to low dim
        h0 = self.flatten1(input_tensor) #h0 = keras.layers.Flatten(input_shape=(w, h))(input_tensor)
        h1 = self.encoder_dense1(h0) #h1 = keras.layers.Dense(n_h, activation=tf.nn.relu)(h0)
        z = self.encoder_dense2(h1) #z = keras.layers.Dense(n_z, activation=tf.nn.relu, name='z')(h1)
        return z

    def decoder_mean(self, z, n_h=256):
        # Decoder for the means of the Gaussian distribution
        # from low dim (n_z) to image space (w, h), use sigmoid as data is in [0, 1]
        l0 = self.decoder_dense1(z) #l0 = keras.layers.Dense(n_h, activation=tf.nn.relu)(z)
        out_mean = self.decoder_dense2(l0) #out_mean = keras.layers.Dense((w * h), activation=tf.nn.sigmoid)(l0)
        return out_mean.view(self.batch_size, 1, 128, 128) #keras.layers.Reshape((w, h), name='out_mean')(out_mean)

    
    '''
    def decoder_covar(self, z, n_h=256):
        # Decoder for the diagonal covariance of the Gaussian distribution
        # from low dim (n_z) to number of pixels (w * h)
        h0 = self.decoder_covar_dense1(z) #h0 = keras.layers.Dense(n_h, activation=tf.nn.relu)(z)
        log_diag_precision = self.decoder_covar_1_half_dense(h0) #keras.layers.Dense((w * h), activation=None, name='log_diag_chol_precision')(h0)
        return log_diag_precision
    '''
    
    def decoder_covar(self, z, n_h=256):
        # Decoder for the covariance of the Gaussian distribution
        # from low dim (n_z) to image space with extra channels (w, h, (nb // 2) + 1)
        h0 = self.decoder_covar_dense1(z) #keras.layers.Dense(n_h, activation=tf.nn.relu)(z)

        chol_half_weights = self.decoder_covar_1_half_dense(h0) #keras.layers.Dense((w * h * ((nb // 2) + 1)), activation=None)(h0)
        chol_half_weights = chol_half_weights.view(self.batch_size, 128, 128, 2) #keras.layers.Reshape((w, h, (nb // 2) + 1))(chol_half_weights)

        # The first channel contains the log_diagonal of the cholesky matrix
        log_diag_chol_precision = chol_half_weights[:,:,:,0] #keras.layers.Lambda(lambda x: x[..., 0])(chol_half_weights)
        log_diag_chol_precision = log_diag_chol_precision.view(self.batch_size, 128, 128, 1) #keras.layers.Flatten(input_shape=(w, h), name='log_diag_chol_precision')(log_diag_chol_precision)
        
        # Exponentiate to remove the log from the diagonal
        chol_half_weights = self.exponentiate_diag(chol_half_weights) #keras.layers.Lambda(lambda x: exponentiate_diag(x))(chol_half_weights)

        # Concatenate with zeros to have a nb = k*k kernel per pixel, output size is (w, h, nb)
        chol_precision_weights = self.concat_with_zeros(chol_half_weights) #keras.layers.Lambda(lambda x: concat_with_zeros(x), name='chol_precision_weights')(chol_half_weights)

        return chol_precision_weights, log_diag_chol_precision
        
    def forward(self, x):
        z = self.encoder(torch.reshape(x, (self.batch_size, 1, 128, 128)))
        out_mean = self.decoder_mean(z)
        #print('out_mean:', out_mean)
        chol_precision_weights, log_diag_chol_precision = self.decoder_covar(z)
        #log_diag_precision = self.decoder_covar(z)
        return out_mean, chol_precision_weights, log_diag_chol_precision

'''
class Noise_cov_reduction(nn.Module):
    def __init__(self, n_out = 1, useBN=True):
        super(Noise_cov_reduction, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(128*128, 3, 3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU())
            
        self.fc1 = nn.Linear(128, 128)
        
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(3, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 25, 3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.Conv2d(25, 25, 3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.Conv2d(25, 25, 3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.Conv2d(25, 25, 3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.Conv2d(25, 25, 3, stride=1, padding=1),
            nn.BatchNorm2d(25))
        
    def forward(self, x):
        
        print('x:', x.shape)
        x = x.view(1,128*128,128,128)
        print('x.view:', x.shape)
        
        xn = self.conv(x)
        print('xn:', xn.shape)
        x1 = self.fc1(xn)
        print('x1:', x1.shape)
        x1_f = self.deconv1(x1)
        print('x1_f:', x1_f.shape)
        #xn = x1_f
        x2 = self.conv1(x1_f)
        print('x2:', x2.shape)
        x2o = x2+x1_f
        print('x2o:', x2o.shape)
        
        x3_f = self.deconv2(x2o)
        print('x3_f:', x3_f.shape)
        x3 = self.conv2(x3_f)
        print('x3:', x3.shape)
        x3o = x3+x3_f
        print('x3o:', x3o.shape)
        
        x4_f = self.deconv3(x3o)
        print('x4_f:', x4_f.shape)
        x4 = self.conv3(x3_f)
        print('x4:', x4)
        x4o = x4+x4_f
        print('x4o:', x4o.shape)
        
        x5_f = self.deconv4(x4o)
        print('x5_f:', x5_f.shape)
        
        # zero padding to size 64x64x49
        # sparse reshape to size 128*128x128*128
        # remove the diagonal values to obtain lower traingular matrix
        
        return x
'''

class SLNN(nn.Module):
    def __init__(self, n_out = 1, batch_size = 28, useBN=True):
        super(SLNN, self).__init__()
        #self.conv1 = nn.Conv2d(n_out, 32, 3, stride=1, padding=1)
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(n_out, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
            
        #self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
            
        #self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        #self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.maxpooling3_1 = nn.MaxPool2d((4, 4))
        self.relu3_1 = nn.LeakyReLU(0.2, inplace=True)
        self.flatten3_1 = nn.Flatten()
        self.dense3_1 = nn.Linear(32768,16384)
        self.dense3_2_1 = nn.Linear(16384,1)
        self.activate3_1 = nn.Sigmoid()
        
        #self.conv1 = nn.Conv2d(n_out, 32, 3, stride=1, padding=1)
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(n_out, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
            
        #self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
            
        #self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        #self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.maxpooling3_2 = nn.MaxPool2d((4, 4))
        self.relu3_2 = nn.LeakyReLU(0.2, inplace=True)
        self.flatten3_2 = nn.Flatten()
        self.dense3_2 = nn.Linear(32768,16384)
        self.dense3_2_2 = nn.Linear(16384,1)
        self.activate3_2 = nn.Sigmoid()
        '''
        self.fc3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32768,2),
            nn.Sigmoid())
        '''
        #self.sigmoid = nn.Sigmoid()
  
    def forward(self, x):
    
        x1p = self.conv1_1(torch.unsqueeze(x[:,0,:,:], 1))
        #print(x1p.shape)
        x2p = self.conv2_1(x1p)
        #print(x2p.shape)
        x3p = self.maxpooling3_1(x2p)
        #print(x3p.shape)
        x3p = self.relu3_1(x3p)
        #print(x3p.shape)
        x3p1 = self.flatten3_1(x3p)
        #print(x3p1.shape)
        x3p2 = self.dense3_1(x3p1)
        #print(x3p2.shape)
        x3p3 = self.dense3_2_1(x3p2)
        #print(x3p3.shape)
        outputp = self.activate3_1(x3p3)
        #print(outputp.shape)
        #print(outputp.shape)
        
        x1a = self.conv1_2(torch.unsqueeze(x[:,1,:,:], 1))
        #print(x1a.shape)
        x2a = self.conv2_2(x1a)
        #print(x2a.shape)
        x3a = self.maxpooling3_2(x2a)
        #print(x3a.shape)
        x3a = self.relu3_2(x3a)
        #print(x3a.shape)
        x3a1 = self.flatten3_2(x3a)
        x3a2 = self.dense3_2(x3a1)
        x3a3 = self.dense3_2_2(x3a2)
        outputa = self.activate3_2(x3a3)
        #print(outputa.shape)
        
        return torch.cat((outputp, outputa), axis=1)

def PRNet(pretrained=False, n_out = 1, batch_size = 28, model_name='WNet', **kwargs):
    # model = UNet11(pretrained=pretrained, num_out = n_out, **kwargs)
    
    if model_name == 'SRResNet':
        model = SRResNet(num_out = n_out, batch_size = batch_size)
        #model = SRResNet_RGBY(num_out=n_out)
    elif model_name == 'WNet':
        model = WNet(n_out = n_out, batch_size = batch_size)
    elif model_name == 'WNet_DC':
        model = WNet_DC(n_out = n_out, batch_size = batch_size)
    elif model_name == 'UNet':
        model = UNet(n_out = n_out, batch_size = batch_size, pretrained=False)
        #model = UNet_decoder(n_out=1, useBN=True)
        #model = U_net_restore_reconstruct_Fresnel_diffraction_images(n_out=1, useBN=True)
    elif model_name == 'UNet_phasegan':
        model = UNet_phasegan(n_out=n_out, batch_size = batch_size, pretrained=False)
        
    '''
    elif model_name == 'ResUNet':
        model = ResUNet(n_out = 1, batch_size = batch_size)
    elif model_name == 'Deep_Residual_shrinkage_neural_network':    
        model = Deep_Residual_shrinkage_neural_network(n_out = n_out, batch_size = batch_size, useBN=True)
    '''
    # model = MCFCN(nin = 2, n_out = 2)
    # model = eHoloNet(pretrained=pretrained,is_deconv=True, **kwargs)
    #print(model)
    return model

def NN_model(n_out = 1, batch_size = 28, model_name='SLNN'):
    
    model = SLNN(n_out = n_out, batch_size = batch_size, useBN=True)
    
    return model

def cov_model(n_out = 1, batch_size = 28, model_name='Cov_reduction_model'):
    
    model = Noise_cov_reduction(n_out = n_out, batch_size = batch_size, useBN=True)
    
    return model