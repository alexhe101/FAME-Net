# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .modules import InvertibleConv1x1
from .refine import Refine
import torch.nn.init as init

class MaskPredictor(nn.Module):
    def __init__(self,in_channels):
        super(MaskPredictor,self).__init__()
        self.conv = nn.Conv2d(in_channels=2*in_channels,out_channels=in_channels,kernel_size=3,stride=1,padding=1)
        self.spatial_mask=nn.Conv2d(in_channels=in_channels,out_channels=2,kernel_size=1,bias=False)
        self.relu = nn.LeakyReLU(0.1,False)
    def forward(self,x):
        f_x = self.relu(self.conv(x))
        spa_mask=self.spatial_mask(f_x)
        spa_mask=F.gumbel_softmax(spa_mask,tau=1,hard=True,dim=1)
        return spa_mask
class ConvProce(nn.Module):
    def __init__(self,channel):
        super(ConvProce,self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3,1,1)
        self.relu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(channel,channel,3,1,1)
    def forward(self,x):
        res = x
        x = self.relu1(self.conv1(x))
        x = self.conv2(x)+res
        return x
class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_3 = nn.Conv2d(in_size+in_size,out_size,3,1,1)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        return x+resi

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, d, relu_slope=0.1):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

    def forward(self, x):
        out = self.relu_1(self.conv_1(x))
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out




class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, d = 1, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1, d)
        self.G = subnet_constructor(self.split_len1, self.split_len2, d)
        self.H = subnet_constructor(self.split_len1, self.split_len2, d)

        in_channels = channel_num
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        # if not rev:
        # invert1x1conv
        x, logdet = self.flow_permutation(x, logdet=0, rev=False)

        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out



class GateNetwork(nn.Module):
    def __init__(self, input_size, num_experts, top_k):
        super(GateNetwork, self).__init__()
        self.gap = nn.AdaptiveMaxPool2d(1)
        self.gap2 = nn.AdaptiveAvgPool2d(1)
        self.input_size = input_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.fc0 = nn.Linear(input_size,num_experts)
        self.fc1 = nn.Linear(input_size, num_experts)
        self.relu1 = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=1)
        init.zeros_(self.fc1.weight)
        self.sp = nn.Softplus()
    def forward(self, x):
        # Flatten the input tensor
        x = self.gap(x)+self.gap2(x)
        x = x.view(-1, self.input_size)
        inp = x
        # Pass the input through the gate network layers
        x = self.fc1(x)
        x= self.relu1(x)
        noise = self.sp(self.fc0(inp))
        noise_mean = torch.mean(noise,dim=1)
        noise_mean = noise_mean.view(-1,1)
        std = torch.std(noise,dim=1)
        std = std.view(-1,1)
        noram_noise = (noise-noise_mean)/std
        # Apply topK operation to get the highest K values and indices along dimension 1 (columns)
        topk_values, topk_indices = torch.topk(x+noram_noise, k=self.top_k, dim=1)

        # Set all non-topK values to -inf to ensure they are not selected by softmax
        mask = torch.zeros_like(x).scatter_(dim=1, index=topk_indices, value=1.0)
        x[~mask.bool()] = float('-inf')

        # Pass the masked tensor through softmax to get gating coefficients for each expert network
        gating_coeffs = self.softmax(x)

        return gating_coeffs
class Expert(nn.Module):
    def __init__(self,channels):
        super(Expert, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels,3,1,1)
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(channels, channels,3,1,1)
        self.relu2 = nn.LeakyReLU(0.1)
    def forward(self,x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return x


class LfInstance(nn.Module):
    def __init__(self,channels):
        super(LfInstance, self).__init__()
        self.process = ConvProce(channels)
    def forward(self,x):
        return self.process(x)
class HfInstance(nn.Module):
    def __init__(self,channels):
        super(HfInstance, self).__init__()
        self.process = HinResBlock(channels,channels)
    def forward(self,x):
        return self.process(x)



class LfExpert(nn.Module):
    def __init__(self,channels,num_experts,k):
        super(LfExpert, self).__init__()
        self.gate = GateNetwork(channels,num_experts,k)
        self.pre_fuse = nn.Sequential(InvBlock(HinResBlock, 2*channels, channels),
                                         nn.Conv2d(2*channels,channels,1,1,0))
        self.expert_networks_d = nn.ModuleList(
            [LfInstance(channels) for i in range(num_experts)])

    def forward(self, x):
        x = self.pre_fuse(x)
        cof = self.gate(x)
        out = torch.zeros_like(x).to(x.device)
        for idx in range(self.num_experts):
            if cof[:,idx].all()==0:
                continue
            mask = torch.where(cof[:,idx]>0)[0]
            expert_layer = self.expert_networks_d[idx]
            expert_out = expert_layer(x[mask])
            cof_k = cof[mask,idx].view(-1,1,1,1)
            out[mask]+=expert_out*cof_k
        return out,cof_k

class HfExpert(nn.Module):
    def __init__(self,channels,num_experts,k):
        super(HfExpert, self).__init__()
        self.gate = GateNetwork(channels,num_experts,k)
        self.expert_networks_d = nn.ModuleList(
            [HfInstance(channels) for i in range(num_experts)])
        self.pre_fuse = nn.Sequential(InvBlock(HinResBlock, 2*channels, channels),
                                         nn.Conv2d(2*channels,channels,1,1,0))
    def forward(self, x):
        x = self.pre_fuse(x)
        cof = self.gate(x)
        out = torch.zeros_like(x).to(x.device)
        for idx in range(self.num_experts):
            if cof[:,idx].all()==0:
                continue
            mask = torch.where(cof[:,idx]>0)[0]
            expert_layer = self.expert_networks_d[idx]
            expert_out = expert_layer(x[mask])
            cof_k = cof[mask,idx].view(-1,1,1,1)
            out[mask]+=expert_out*cof_k
        return out,cof_k

class Decoder(nn.Module):
    def __init__(self,channels,num_experts,k):
        super(Decoder, self).__init__()
        self.gate = GateNetwork(channels, num_experts, k)
        self.expert_networks_d = nn.ModuleList(
            [ConvProce(channels) for i in range(num_experts)])
        self.pre_fuse = nn.Sequential(InvBlock(HinResBlock, 4*channels, 2*channels),
                                         nn.Conv2d(4*channels,channels,1,1,0))

    def forward(self,x):
        x = self.pre_fuse(x)
        cof = self.gate(x)
        out = torch.zeros_like(x).to(x.device)
        for idx in range(self.num_experts):
            if cof[:,idx].all()==0:
                continue
            mask = torch.where(cof[:,idx]>0)[0]
            expert_layer = self.expert_networks_d[idx]
            expert_out = expert_layer(x[mask])
            cof_k = cof[mask,idx].view(-1,1,1,1)
            out[mask]+=expert_out*cof_k
        return out,cof_k
def upsample(x, h, w):
    return F.interpolate(x, size=[h, w], mode='bicubic', align_corners=True)

class MOEInstance(nn.Module):
    def __init__(self,channels,num_experts,k):
        super(MOEInstance, self).__init__()
        self.lf_experts = LfExpert(channels,num_experts, k)
        self.hf_experts = HfExpert(channels,num_experts, k)
    def forward(self,panf,msf,hf_mask,lf_mask):
        mlf = msf*lf_mask
        plf = panf*lf_mask

        mhf = msf*hf_mask
        phf = panf*hf_mask

        lf,lfgate = self.lf_experts(torch.cat([mlf,plf],dim=1))
        hf,hfgate = self.hf_experts(torch.cat([mhf,phf],dim=1))

        return lf,hf,lfgate,hfgate
class FeatureEncoder(nn.Module):
    def __init__(self,base_filter):
        super(FeatureEncoder, self).__init__()
        self.conv1 = HinResBlock(base_filter,base_filter)
        self.conv2 = HinResBlock(base_filter,base_filter)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Net(nn.Module):
    def __init__(self, num_channels, base_filter, args):
        super(Net, self).__init__()
        self.nc = base_filter
        channels = base_filter
        self.msconv = nn.Conv2d(4,base_filter,3,1,1)
        self.pconv = nn.Conv2d(1,base_filter,3,1,1)
        self.msencoder = FeatureEncoder(base_filter)
        self.panencoder = FeatureEncoder(base_filter)

        self.maskp = MaskPredictor(base_filter)

        self.moeInstance = MOEInstance(channels,4,2)
        self.decoder = Decoder(channels,4,2)
        self.refine = Refine(base_filter,4)
    def forward(self, ms, _, pan):
        # ms  - low-resolution multi-spectral image [N,C,h,w]
        # pan - high-resolution panchromatic image [N,1,H,W]
        if type(pan) == torch.Tensor:
            pass
        elif pan == None:
            raise Exception('User does not provide pan image!')
        _, _, m, n = ms.shape
        _, _, M, N = pan.shape

        mHR = upsample(ms, M, N)
        msf = self.msconv(mHR)
        panf = self.pconv(pan)
        msf = self.msencoder(msf)
        panf = self.panencoder(panf)
        mask = self.maskp(torch.cat([msf,panf],dim=1))
        high_fre_mask = (mask[:,0, ...]).unsqueeze(1)
        low_fre_mask = (mask[:, 1, ...]).unsqueeze(1)
        lf,hf,lf_gate,hf_gate  = self.moeInstance(panf,msf,high_fre_mask,low_fre_mask)
        dec,dec_gate = self.decoder(torch.cat([msf,hf,panf,lf],dim=1))
        HR = self.refine(dec)+mHR
        return HR,mask,[lf_gate,hf_gate,dec_gate]

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)