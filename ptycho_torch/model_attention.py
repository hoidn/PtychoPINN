import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import numpy as np

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio = 16, pool_types = ['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            # Add other pooling types if needed

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        # Scale output range to [0, 1] with sigmoid
        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * scale
    
class SpatialGate(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialGate, self).__init__()
        padding = kernel_size // 2
        self.compress = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=padding, bias=False)

    def forward(self, x):
        x_compress = torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
        # Scale output range to [0, 1] with sigmoid
        scale = torch.sigmoid(self.compress(x_compress))
        return x * scale
    
class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], spatial_kernel_size=7):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate(kernel_size=spatial_kernel_size)
    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out
    
class ECALayer(nn.Module):
    def __init__(self, channel, k_size=3): # k_size often adapted based on C
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Use 1D Conv to capture local cross-channel interactions
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x) # Squeeze B x C x 1 x 1
        # Apply 1D conv requires B x 1 x C or B x C x 1 view
        y = self.conv(y.squeeze(-1).transpose(-1, -2)) # B x C x 1 -> B x 1 x C -> apply conv1d -> B x 1 x C
        y = y.transpose(-1, -2).unsqueeze(-1) # B x 1 x C -> B x C x 1 -> B x C x 1 x 1
        y = self.sigmoid(y) # Excitation
        return x * y.expand_as(x) # Scale
    
class BasicSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(BasicSpatialAttention, self).__init__()
        # Using the previously defined SpatialGate class directly
        self.spatial_gate = SpatialGate(kernel_size=kernel_size)

    def forward(self, x):
        return self.spatial_gate(x)
    
