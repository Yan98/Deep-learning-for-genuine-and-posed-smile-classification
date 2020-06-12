#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 19:28:32 2020

@author: Yan
"""

import torch
import torch.nn.functional as F
import math
from torch import nn

############################################
######## Define neural networks model ######
############################################


# The implmentation of NonLocalBlock is from https://github.com/AlexHex7/Non-local_pytorch/tree/master/lib
class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)
        

# The implementation of CONVLSTM are based on the code from
# https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=3,
                              padding=1)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next



    def init_hidden(self, batch_size, height, width):
        
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):


    def __init__(self, input_dim, hidden_dim):
        super(ConvLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell  = ConvLSTMCell(input_dim=self.input_dim,
                                  hidden_dim=self.hidden_dim)

    def forward(self, input_tensor,time = None):
        
        b, _, _, h, w = input_tensor.size()
        
        hidden_state = self.cell.init_hidden(b, h, w)

        seq_len = input_tensor.size(1)
        
        h, c = hidden_state
        for t in range(seq_len):
            reset = (time == t).nonzero().view(-1)
            for index in reset:
                h[index] = 0
                c[index] = 0
                
            h, c = self.cell(input_tensor=input_tensor[:, t, :, :, :],cur_state=[h, c])      

        return h
    

# The implmentation of resnet is based on  https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py and
# https://github.com/zhunzhong07/Random-Erasing/blob/master/models/cifar/resnet.py       
    
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, depth):
        super(ResNet, self).__init__()
        
        assert depth  % 6 == 0, 'depth should be 6n'
        n = depth // 6

        block = BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 48x48

        x = self.layer1(x)  # 48x48
        x = self.layer2(x)  # 24x24
        x = self.layer3(x)  # 12x12

        return x


def resnet12(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(12)
    


##The implementation of AlexNet is based on https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py    
class miniAlexNet(nn.Module):

    def __init__(self):
        super(miniAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size= 3, stride= 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2,padding = 1),
            nn.Conv2d(16, 48, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1),
        )
        
    def forward(self, x):
        x = self.features(x)
        return x
    
    
#The implementation of DenseNet is based on https://github.com/kuangliu/pytorch-cifar/blob/master/models/densenet.py        
class DenseBlock(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(DenseBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out

#No stride on Transition
class Transition(nn.Module):
    def __init__(self, in_planes, out_planes,last = False):
        super(Transition, self).__init__()
        self.bn   = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.last = last
        
    def forward(self, x):
        stride = self.last and 1 or 2
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2,stride)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes//2))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes//2))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes//2))
        self.trans3 = Transition(num_planes, out_planes,last = True)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate
        
        self.bn = nn.BatchNorm2d(num_planes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.relu(self.bn(out))
        return out

def miniDensenet():
    return DenseNet(DenseBlock, [6,12,24,16], growth_rate=2)
    

    
class TemporalAttension(nn.Module):
    
    def __init__(self,channels):
        
        super(TemporalAttension,self).__init__()
        
        self.conv = nn.Conv2d(channels,channels,3,padding = 1)
        
        init = torch.zeros((3,3))
        init[1,1] = 1
        
        self.conv.weight.data.copy_(init)
        
        self.conv.bias.data.copy_(torch.zeros(channels))
        
    def forward(self, x):
        
        x1 = x[:,:-1]
        x2 = x[:,1:]
        o = x2 - x1
        o = torch.cat((torch.zeros((x.size(0),1,x.size(2),x.size(3),x.size(4)),device = x.device),o),1)
        o = o.view(-1,x.size(2),x.size(3),x.size(4))
        x = self.conv(o).view(x.size()) * x + x
        
        return x     

#implementation of Conv2d + BN + RELU
class Convbn(nn.Module):
    
    def __init__(self,ins,ous,kernel,padding = 0):
        
        super(Convbn,self).__init__()
        
        self.conv = nn.Conv2d(ins,ous,kernel,padding = padding)
        self.bn   = nn.BatchNorm2d(ous)
        self.relu = nn.ReLU(inplace = True)
        
    def forward(self,x):
        
        return self.relu(self.bn(self.conv(x)))

#implementation of bi_directional CONVLSTM    
class BI_CONVLSTM(nn.Module):
    
    def __init__(self,ins,outs):
        
        super(BI_CONVLSTM,self).__init__()
        
        self.fconv = ConvLSTM(ins,outs)
        self.bconv = ConvLSTM(ins,outs)
        
    def forward(self,x,s):
        xs = x.clone()
        f = self.fconv(x,time = s)
        
        batch,time_step,c,h,w  = x.size()
        
        for idx in range(batch):
            xs[idx,s[idx]:] = xs[idx,s[idx]:].flip(0)
        
        b = self.bconv(xs,time = s)
        
        return torch.cat((f,b),dim=1)

class DeepSmileNet(nn.Module):
    
    def __init__(self, cfg = [4,'M', 6, 'M'],re = "org"):
        
        super(DeepSmileNet,self).__init__()
        
        if re == "resnet":
            self.encoder = resnet12()
            
        elif re == "miniAlexnet":
            self.encoder = miniAlexNet()
            
        elif re == "minidensenet":
            self.encoder = miniDensenet()
            
        else:
            
            self.encoder = self._make_layers([4,'M', 6, 'M'])
            
        if re == "GRU":
            
            self.decoder = nn.GRU(864,256,batch_first = True)
        
        elif re == "LSTM":
            
            self.decoder = nn.LSTM(864, 256,batch_first = True)
        
        elif re == "org":
            self.TA = TemporalAttension(3)
            self.decoder = ConvLSTM(6,8)
            
        else:
            
            self.decoder = ConvLSTM(64,64)
        
        
        if re not in ["GRU","LSTM","org"]:
            
            self.pool = nn.Sequential(
                NONLocalBlock2D(64),
                nn.Conv2d(64, 80, 3),
                nn.BatchNorm2d(80),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((2,2)),
                nn.Flatten(),
                nn.Linear(320,1),
                nn.Sigmoid(),
                )
            
        elif re == "org":
            
            self.pool = nn.Sequential(
                NONLocalBlock2D(8),
                nn.AvgPool2d(kernel_size = 2, stride = 2),
                Convbn(8,10,2),
                nn.Dropout(0.5),
                nn.Flatten(),
                nn.Linear(250,1),
                nn.Sigmoid()
                )
            
        else:
            
            self.pool = nn.Sequential(
                nn.Linear(256, 1),
                nn.Sigmoid()
                )
            
        self.re = re
        
    def _make_layers(self,cfg,in_channels = 3):
        
        layers = [nn.BatchNorm2d(in_channels)]
        for x in cfg:
            if x == 'M':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                layers += [Convbn(in_channels, x, kernel = 3, padding=1)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        layers += [nn.Dropout2d(0.2)]
        return nn.Sequential(*layers)
        
        
    def forward(self,x,s):
        
        if self.re == "org":
            x = self.TA(x)
        
        batch_size, timesteps, C, H, W = x.size()
        input_x = []
        
        for l in range(x.size(0)):
            input_x.append(x[l,s[l]:,:,:,:])
        
        input_x = torch.cat(input_x,0)
        out = self.encoder(input_x)
        
        current = 0
        _,new_c,new_w,new_h = out.size()
        reshape_out = torch.zeros((batch_size, timesteps,new_c,new_w,new_h),device = x.device)   
        
        for index,l in enumerate(s):
            reshape_out[index,l:] = out[current:current + timesteps - l]
            current+= timesteps - l
        x  = reshape_out 
        
        if self.re not in ["GRU","LSTM"]:
            x = self.decoder(x,s)
        
        else:
            x,_ = self.decoder(x.view(batch_size,timesteps,-1))
            x= x[:,-1,:]
        
        x = self.pool(x)
        
        return x 




      