import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np

# import torch.onnx
# import netron
# import onnx
from thop import profile
from torchsummary import summary

# from onnx import shape_inference
import pandas as pd
import os
import matplotlib.pyplot as plt
import random
from collections import OrderedDict
from typing import Dict, Callable
import torchextractor as tx


global num_cla
num_cla = 10

class Downsample(nn.Module): 
    def __init__(self,in_planes,out_planes,stride=2):
        super(Downsample,self).__init__()
        self.downsample=nn.Sequential(
                        nn.BatchNorm2d(in_planes),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_planes, out_planes,
                                kernel_size=1, stride=stride, bias=False)
                        )
    def forward(self,x):
        x=self.downsample(x)
        return x

class Downsample_clean(nn.Module): 
    def __init__(self,in_planes,out_planes,stride=2):
        super(Downsample_clean,self).__init__()
        self.downsample_=nn.Sequential(
                        nn.Conv2d(in_planes, out_planes,
                                kernel_size=1, stride=stride, bias=False)
                        )
    def forward(self,x):
        x=self.downsample_(x)
        return x
 
class Downsample_Fix(nn.Module): 
    def __init__(self,in_planes,out_planes,stride=1):#stride=2):
        super(Downsample_Fix,self).__init__()
        self.downsample_=nn.Sequential(
                    nn.AvgPool2d(2),
                        nn.Conv2d(in_planes, out_planes,
                                kernel_size=1, stride=stride, bias=False)
                        )
    def forward(self,x):
        x=self.downsample_(x)
        return x

class MaskConv(nn.Module): 
    def __init__(self,in_planes, out_planes, kernel_size, stride, padding, bias):
        super(MaskConv,self).__init__()
        self.mask_conv_=nn.Sequential(
                        nn.Conv2d(in_planes, out_planes,
                                kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
                        )        
        # self.FixDS.append( nn.Conv2d(pre_features[i].shape[1], F_x_n.shape[1], kernel_size=1, stride=2, padding=0, bias=False).cuda() )
        # pre_features[i] = self.FixDS[-1](pre_features[i])
        # nn.init.dirac_(self.FixDS[-1].weight, 2)
        # self.FixDS[-1].weight.requires_grad = False
    def forward(self,x):
        x=self.mask_conv_(x)
        return x
    
def _downsample_All_Ele(x):
    out00 = F.pad(x, pad=(0,0,0,0), mode='constant', value=0)
    out01 = F.pad(x, pad=(-1,1,0,0), mode='constant', value=0)
    out10 = F.pad(x, pad=(0,0,-1,1), mode='constant', value=0)
    out11 = F.pad(x, pad=(-1,1,-1,1), mode='constant', value=0)

    out00 = F.avg_pool2d(out00, kernel_size=1, stride=2, padding=0)
    out01 = F.avg_pool2d(out01, kernel_size=1, stride=2, padding=0)
    out10 = F.avg_pool2d(out10, kernel_size=1, stride=2, padding=0)
    out11 = F.avg_pool2d(out11, kernel_size=1, stride=2, padding=0)
    x = torch.cat((out00, out01, out10, out11), dim=1)
    return x 
    


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Fix') != -1:

        nn.init.dirac_(m.downsample_[1].weight.data, 4)

    if classname.find('Mask_') != -1:

        nn.init.sparse_(m.mask_[0].weight.data, sparsity=0.1)

def weights_init_plot(m):
    classname = m.__class__.__name__
    # print('m_name', m)
    if classname.find('Conv2d') != -1:

        nn.init.constant_(m.weight.data, 0.0001)

        # nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
         torch.nn.init.constant_(m.weight, 0.0001)
         torch.nn.init.constant_(m.bias, 0.0)


class ZeroSBlockAnySettings(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, pre_planes, coesA, coesB, DcoesA=None, DcoesB=None, coes_stepsize=1, steps=3, stride=1, coe_ini=1, fix_coe=False, given_coe=[1.0 / 3, 5.0 / 9, 1.0 / 9, 16.0 / 9], Settings='', Layer_idx=0, coesDecay=1, switch=0):        
        super(ZeroSBlockAnySettings, self).__init__()
        self.Settings = Settings 
        self.drop_rate = None      
        if 'drop' in self.Settings:
            self.drop_rate = 0.2
      
        self.Layer_idx = Layer_idx

        self.steps = steps
        if 'DSfirst' in self.Settings:    
            # print('DSfirst')   
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.EndBN = nn.BatchNorm2d(planes)
        # self.relu = nn.ReLU(inplace=True)
        if 'Swish' in self.Settings:
            self.relu = nn.SiLU(inplace=False)
        elif 'Mish' in self.Settings:
            self.relu = nn.Mish(inplace=False)            
        else:
            self.relu = nn.ReLU(inplace=False)
            
        self.identity = nn.Identity()

        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes
        self.pre_planes_b = pre_planes
        # self.coes_stepsize = coes_stepsize
        self.fix_coe = fix_coe
        self.DcoesA_ = DcoesA
        self.DcoesB_ = DcoesB
        self.coesA_ = coesA
        self.coesB_ = coesB        
        self.start=1
        self.has_ini_block = False
        if 'LearnStepSize' in self.Settings:
            self._coes_stepsize = nn.Parameter(torch.ones(1)*coes_stepsize, requires_grad = True)
        else:
            self._coes_stepsize = nn.Parameter(torch.ones(1)*coes_stepsize, requires_grad = False)
        for pre_pl in self.pre_planes_b:
            if pre_pl <=0:
                self.is_ini = True
            else:
                self.is_ini = False
            self.has_ini_block = self.is_ini or self.has_ini_block #wrong
        if 'ConvStride2Fix' in self.Settings:
            # print('ConvStride2Fix')

            if not (self.has_ini_block):
                if self.in_planes != self.planes:
                    self.downsample_x_Fix = Downsample_clean(self.in_planes, self.planes, 2)
                start_DS = 1        
                for i in range(self.steps-1):
                    if self.pre_planes_b[i] != self.planes:
                        if start_DS:
                            self.FixDS = nn.ModuleList([])
                            start_DS = 0
                        self.FixDS.append( Downsample_clean(self.pre_planes_b[i], self.planes) )            
        elif 'ConvStride2Learn' in self.Settings:
            # print('ConvStride2Learn')

            if not (self.has_ini_block):
                if self.in_planes != self.planes:
                    self.downsample_x_Learn = Downsample_clean(self.in_planes, self.planes, 2)
                start_DS = 1        
                for i in range(self.steps-1):
                    if self.pre_planes_b[i] != self.planes:
                        if start_DS:
                            self.LearnDS = nn.ModuleList([])
                            start_DS = 0
                        self.LearnDS.append( Downsample_clean(self.pre_planes_b[i], self.planes) )
                        self.LearnDS.append( Downsample_clean(self.pre_planes_b[i], self.planes) )  
                        
        elif 'ConvStride2ResLikeShare' in self.Settings:
            # print('ConvStride2ResLikeShare')

            if not (self.has_ini_block):
                if self.in_planes != self.planes:
                    self.downsample_x_Learn = Downsample(self.in_planes, self.planes, 2)
                start_DS = 1        
                if self.pre_planes_b[0] != self.planes:
                    if start_DS:
                        self.LearnDS = nn.ModuleList([])
                        start_DS = 0
                    self.LearnDS.append( Downsample(self.pre_planes_b[0], self.planes) )
        elif 'ConvStride2ResLike' in self.Settings:        
            if not (self.has_ini_block):
                if self.in_planes != self.planes:
                    self.downsample_x_Learn = Downsample(self.in_planes, self.planes, 2)
                start_DS = 1        
                for i in range(self.steps-1):
                    if self.pre_planes_b[i] != self.planes:
                        if start_DS:
                            self.LearnDS = nn.ModuleList([])
                            start_DS = 0
                        self.LearnDS.append( Downsample(self.pre_planes_b[i], self.planes) )
                        self.LearnDS.append( Downsample(self.pre_planes_b[i], self.planes) )                                                                            
        elif 'ConvStride2Share' in self.Settings:
            # print('ConvStride2Share')

            if not (self.has_ini_block):
                if self.in_planes != self.planes:
                    self.downsample_x_Learn = Downsample_clean(self.in_planes, self.planes, 2)
                start_DS = 1        
                # for i in range(self.steps-1):
                if self.pre_planes_b[0] != self.planes:
                    if start_DS:
                        self.LearnDS = nn.ModuleList([])
                        start_DS = 0
                    self.LearnDS.append( Downsample_clean(self.pre_planes_b[0], self.planes) )                         
        elif 'AllEle' in self.Settings:
            # print('AllEle')
            AllEle = 1
        else:    
            # print('DiracConvFix')
                    
            if not (self.has_ini_block):
                if self.in_planes != self.planes:
                    self.downsample_x_Fix = Downsample_Fix(self.in_planes, self.planes)
                    # self.downsample_x_Learn = Downsample_clean(self.in_planes, self.planes, 2)
                start_DS = 1  
                      
        if self.in_planes != self.planes:
            self.downsample_x_Learn = Downsample(self.in_planes, self.planes, 2)
                
        if 'LearnBal' in self.Settings:
            self.coesBalance = nn.Parameter(torch.zeros(1), requires_grad = True)
        else:
            self.coesBalance = nn.Parameter(torch.zeros(1), requires_grad = False)
        if 'ShareExpDecay' in self.Settings:
            self.coesDecay_ = coesDecay#nn.Parameter(torch.ones(1)*(coesDecay))
        else:
            self.coesDecay_ = nn.Parameter(torch.ones(1)*coesDecay)
        self.Layer_idx = Layer_idx
        # self.coes_stepsize = 1
        if 'BiasExp' in self.Settings:
            self.coes_bias_inner = nn.Parameter(torch.zeros(1), requires_grad = True)
            self.coes_bias_outer = nn.Parameter(torch.zeros(1), requires_grad = True)
        else:
            self.coes_bias_inner = nn.Parameter(torch.zeros(1), requires_grad = False)
            self.coes_bias_outer = nn.Parameter(torch.zeros(1), requires_grad = False)
        self.switch = switch    
    # def forward(self, x, pre_features, pre_acts, coesA, coesB, DcoesA, DcoesB, coes_stepsize):  
    def forward(self, x, pre_features, pre_acts):  
 
        residual = x

        F_x_n = self.bn1(x)
        F_x_n = self.relu(F_x_n)
        F_x_n = self.conv1(F_x_n)
        F_x_n = self.bn2(F_x_n)
        F_x_n = self.relu(F_x_n)
        F_x_n = self.conv2(F_x_n)

        # self.has_ini_block = False
        # for pre_fea in pre_features:
        #     if isinstance(pre_fea, int):
        #         self.is_ini = True
        #     else:
        #         self.is_ini = False
        #     self.has_ini_block = self.is_ini or self.has_ini_block
        # self.has_ini_block = isini

        # if not (self.has_ini_block):
        if self.Layer_idx >= self.switch:
            # print('self.Layer_idx',self.Layer_idx)

            if self.in_planes != self.planes:
                residual = self.downsample_x_Learn(residual)  
                # for i in range(self.switch):
                for i in range(self.steps-1):
                    pre_features[i] = self.LearnDS[i](pre_features[i])
                    pre_acts[i] = self.LearnDS[self.steps-1+i](pre_acts[i])                                                


            # residual_ExpDecay = residual
       
            sum_features = (self.coesA_[0]+0.01*self.DcoesA_[0])*residual
            
            sum_acts = (self.coesB_[0]+0.01*self.DcoesB_[0])*F_x_n
            # print('-coesB[0]', -coesB[0].data)
            
            for i in range(self.steps-1):
                if not self.coesA_[i+1] in [0, 0.0]:
                    sum_features = torch.add( sum_features, (self.coesA_[i+1]+0.01*self.DcoesA_[i+1])*pre_features[i] )
                if not self.coesB_[i+1] in [0, 0.0]:
                    sum_acts = torch.add( sum_acts, (self.coesB_[i+1]+0.01*self.DcoesB_[i+1])*pre_acts[i] )  

            x =  torch.add( sum_features, torch.mul(self._coes_stepsize, -sum_acts ) )

        else:
            # print('residual')
            if self.in_planes != self.planes:
                residual = self.downsample_x_Learn(residual)  
            # residual_ExpDecay = residual  
         
            x = F_x_n + residual
            
        for i in range(self.steps-2, 0, -1): #steps-2, steps-1, ..., 0 #1, 0 
            pre_features[i] = pre_features[i-1] #pre_features[i-1] = pre_features[i-2]?
            pre_acts[i] = pre_acts[i-1]

        pre_features[0] = residual
        pre_acts[0] = F_x_n
        # if self.training and "FeatureNoise" in self.Settings:
        #     x = x + torch.randn_like(x)* torch.std(x)*0.5
        x = self.identity(x)
        
        return x, pre_features, pre_acts#, coesA, coesB


class ZeroSAnySettings(nn.Module):

    def __init__(self, block, layers, coe_ini=1, givenA=[0.33333333333, 0.55555555556, 0.1111111111], givenB=[0.888888888889, 0.8888888888889, 0], share_coe=False, ini_stepsize=1, pretrain=False, num_classes=num_cla, stochastic_depth=False, PL=1.0, noise_level=0.001,
                 noise=False, Settings='',ini_block_shift=None,IniDecay=-10):
        
        self.Settings = Settings
        steps = len(givenA)        
        self.steps = steps
        if 'EulerSwitch' in self.Settings:
            self.switch = int(self.Settings[self.Settings.find('EulerSwitch')+11:self.Settings.find('EulerSwitch')+14])
            # print('switch',self.switch)
            # print('l', l)
        else:
            self.switch = self.steps-1    
            # print('w/o switch',self.switch)
        self.drop_rate  = None
        if 'dropout' in self.Settings:
            self.drop_rate = 0.2
        if 'Start8' in self.Settings or '8,16,32' in self.Settings:
            # print('Start8')
            self.in_planes = 8
        else:
            self.in_planes = 16
        if '2Chs' in self.Settings:
            self.planes = [16, 32, 64]
        elif 'Start8' in self.Settings:
            self.planes = [8, 32, 128]
        elif '8,16,32' in self.Settings:
            self.planes = [8, 16, 32]
        else:
            self.planes = [16, 64, 256]
        
        self.pre_planes = [] # the smaller, the closer to the current layer
        self.test = []
        # for i in range(self.switch):
        for i in range(self.steps-1):

            self.pre_planes += [-i]
        # for i in range(self.switch):
        for i in range(self.steps-1):

            self.test += [-i]

        self.strides = [1, 2, 2]
        super(ZeroSAnySettings, self).__init__()
        self.block = block
        if 'mnist' in self.Settings:
            print('mnist')
            self.conv1 = nn.Conv2d(1, self.in_planes, kernel_size=3, padding='same', bias=False)
            
        else:
            self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, padding=1, bias=False)   
            # self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, padding='same', bias=False)            
            self.convDS = nn.Conv2d(3, self.in_planes, kernel_size=1, padding='same', bias=False)  

        self.bn1 = nn.BatchNorm2d(16)        
        # if 'mnist' in self.Settings:
        #     print('mnist')
        #     self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding='same', bias=False)
        # else:
        #     self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)            
        # self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # self.pretrain = pretrain
        self.stochastic_depth = stochastic_depth
        self.share_coe = share_coe
        self.noise_level = noise_level
        self.PL = PL
        self.noise = noise
        self.coe_ini= coe_ini
        # self.ini_stepsize = ini_stepsize
        self.givenA = givenA 
        self.givenB = givenB

        if 'LearnCoeA' in self.Settings:   
            print('LearnCoeA')         
            coesA = nn.ParameterList([nn.Parameter(torch.ones(1)*self.givenA[i], requires_grad=False) for i in range(self.steps)]) 
            DcoesA = nn.ParameterList([nn.Parameter(torch.zeros(1), requires_grad=True) for i in range(self.steps)]) 
        else:    
            # self.coes = nn.ParameterList([nn.Parameter(torch.ones(1)*self.given[i], requires_grad=True) for i in range(self.steps+1)])  
            coesA = nn.ParameterList([nn.Parameter(torch.ones(1)*self.givenA[i], requires_grad=False) for i in range(self.steps)])     
            DcoesA = nn.ParameterList([nn.Parameter(torch.zeros(1), requires_grad=False) for i in range(self.steps)]) 
       
        if 'LearnCoeB' in self.Settings:  
            print('LearnCoeB') 
            coesB = nn.ParameterList([nn.Parameter(torch.ones(1)*self.givenB[i], requires_grad=False) for i in range(self.steps)]) 
            DcoesB = nn.ParameterList([nn.Parameter(torch.zeros(1), requires_grad=True) for i in range(self.steps)]) 
        else:    
            # self.coes = nn.ParameterList([nn.Parameter(torch.ones(1)*self.given[i], requires_grad=True) for i in range(self.steps+1)])  
            coesB = nn.ParameterList([nn.Parameter(torch.ones(1)*self.givenB[i], requires_grad=False) for i in range(self.steps)]) 
            DcoesB = nn.ParameterList([nn.Parameter(torch.zeros(1), requires_grad=False) for i in range(self.steps)]) 
        # if self.share_coe == True:
        if 'Step0p1' in self.Settings:
            ini_stepsize = 0.1
        if 'LearnShareStepSize'in self.Settings:
            coes_stepsize =nn.Parameter(torch.Tensor(1).uniform_(ini_stepsize, ini_stepsize), requires_grad=True)
        else:
            coes_stepsize = ini_stepsize
        # self.coesDecay = None
        if 'LearnDecay' in self.Settings:
            LearnDecay = True
        else:
            LearnDecay = False
        # if 'ExpDecay' in self.Settings:   
        coesDecay =nn.Parameter(torch.ones(1)*IniDecay, requires_grad=LearnDecay)
        # else:
        #     self.coesDecay = nn.Parameter(torch.ones(1)*IniDecay, requires_grad=False)
           
        blocks = []
        n = layers[0] + layers[1] + layers[2]
        l = 0
        if not self.stochastic_depth:
            for i in range(3):
                # print('self.pre_planes in net1', self.pre_planes)
                l_ = l                    
                if 'RestaLayerIdx' in self.Settings:
                    l = 0 
                    if self.Settings.split('RestaLayerIdx')[1]:
                        # print(self.Settings.split('RestaLayerIdx')[1])
                        Split = int(self.Settings.split('RestaLayerIdx')[1])
                        l_ = l%Split
                        print('Split',Split)
                Layer_idx = nn.Parameter(torch.ones(1)*l_, requires_grad=False)
                # print('l',l)
                blocks.append(block(self.in_planes, self.planes[i], self.pre_planes, coesA=coesA, coesB=coesB, DcoesA=DcoesA, DcoesB=DcoesB, coes_stepsize=coes_stepsize, steps=self.steps, stride=self.strides[i], coe_ini=self.coe_ini, Settings=self.Settings,Layer_idx=Layer_idx, coesDecay=coesDecay,switch=self.switch))

                # if l < steps-1:
                if l < self.switch:

                    for j in range(steps-2,0,-1): # steps-1, steps-2, ..., 1
                    # for j in range(self.switch-1,0,-1):
                        # print('self.pre_planes[j]',self.pre_planes[j])
                        self.pre_planes[j] =  self.pre_planes[j-1]
                        
                    self.pre_planes[0] = self.in_planes
                else:
                    for j in range(steps-2,0,-1): # steps-2, ..., 1
                    # for j in range(self.switch-1,0,-1): # steps-2, ..., 1

                        self.pre_planes[j] =  self.planes[i] * block.expansion
                    self.pre_planes[0] = self.planes[i] * block.expansion


                self.in_planes = self.planes[i] * block.expansion
                l += 1
                if 'IniEveryStage' in self.Settings:
                    l = 0
                for k in range(1, layers[i]):
                                    
                    l_ = l
                    if 'RestaLayerIdx' in self.Settings:
                        if self.Settings.split('RestaLayerIdx')[1]:
                            Split = int(self.Settings.split('RestaLayerIdx')[1])
                            l_ = l%Split
                            print('Split',Split)

                    print('l2',l_)
                    Layer_idx = nn.Parameter(torch.ones(1)*l_, requires_grad=False)

                    blocks.append(block(self.in_planes, self.planes[i], self.pre_planes, coesA=coesA, coesB=coesB, DcoesA=DcoesA, DcoesB=DcoesB, coes_stepsize=coes_stepsize, steps=self.steps, coe_ini=self.coe_ini, Settings=self.Settings,Layer_idx=Layer_idx, coesDecay=coesDecay,switch=self.switch))

                    if l < steps-1:
                        for j in range(steps-2,0,-1): # steps-2, steps-3, ..., 1
                    # if l < steps-1:
                    #     for j in range(self.switch-1,0,-1): # steps-2, steps-3, ..., 1                    
                            self.pre_planes[j] =  self.pre_planes[j-1]
                        self.pre_planes[0] = self.in_planes
                    else:
                        for j in range(steps-2,0,-1): # steps-2, steps-3, ..., 1
                        # for j in range(self.switch-1,0,-1): # steps-2, steps-3, ..., 1

                            self.pre_planes[j] =  self.planes[i] * block.expansion
                        self.pre_planes[0] = self.planes[i] * block.expansion
                    l += 1
                    
        self.blocks = nn.ModuleList(blocks)
        # self.downsample1 = Downsample(16, 64, stride=1)
        self.downsample1 = Downsample(self.in_planes, 64, stride=1)
        
        if '2Chs' in self.Settings:
            self.bn = nn.BatchNorm2d(64 * block.expansion)
        elif 'Start8' in self.Settings:
            self.bn = nn.BatchNorm2d(128 * block.expansion)
        elif '8,16,32' in self.Settings:
            self.bn = nn.BatchNorm2d(32 * block.expansion)
        else:
            self.bn = nn.BatchNorm2d(256 * block.expansion)

        self.avgpool = nn.AvgPool2d(8)
        if '2Chs' in self.Settings:
            self.fc = nn.Linear(64 * block.expansion, num_classes)
        elif 'Start8' in self.Settings:
            self.fc = nn.Linear(128 * block.expansion, num_classes)
        elif '8,16,32' in self.Settings:
            self.fc = nn.Linear(32 * block.expansion, num_classes)
        else:        
            self.fc = nn.Linear(256 * block.expansion, num_classes)
        self.identityStart0 = nn.Identity()
        self.identityStart1 = nn.Identity()
        self.identityEndM1 = nn.Identity()
        self.identityEnd = nn.Identity()

        for m in self.modules():  # initialization

            if 'PlotTraj' in self.Settings:
                if isinstance(m, nn.Conv2d):
                    m.weight.data.fill_(0.01)
                    
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(0.01)
                    m.bias.data.zero_()
            else:              
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        
                           
    def forward(self, x):
        # print('x.shape', x.shape)
        # if 'mnist' in self.Settings:
        #     # print('mnist_f')
        #     x = F.pad(x, pad=(2,2,2,2), mode='constant', value=0)
        x = self.identityStart0(x)
        # if not 'mnist' in self.Settings:
        #     residual = self.convDS(x)
        # else:
            
        residual = x
        x = self.conv1(x)
        # if 'startRes' in self.Settings:
        #     # print('startRes')
        #     x += residual
        x = self.identityStart1(x)
        pre_features = []
        pre_acts = []
        # for i in range(self.steps-1):
        for i in range(self.switch):
            pre_features.append(-i)
            pre_acts.append(-i)
        if self.block.expansion == 4:
            residual = self.downsample1(x)
        else:
            residual = x

        # traj = [float(x.mean().data)]
        isini = True
        for j in range(self.switch):
        # for j in range(self.steps-1):
            # print('self.blocks[j]',j)
            x, pre_features, pre_acts = self.blocks[j](x, pre_features, pre_acts)

        for i, b in enumerate(self.blocks):  # index and content
            
            # print('b, i',i)
            if i < self.switch:
            # if i < self.steps-1:
                continue
            isini = False
            residual = x 

            x, pre_features, pre_acts = b(x, pre_features, pre_acts)
            # print('b, i_2th',i)
            # traj = traj+[float(x.mean().data)]
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        # residual = self.avgpool(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        # x = x+ residual 
        x = x.view(x.size(0), -1)
        x = self.identityEndM1(x)
        # if self.training and "FeatureNoise" in self.Settings:
        #     x = x + torch.randn_like(x)* torch.std(x)*0.5
        x = self.fc(x)
        x = self.identityEnd(x)

        return x#, coesA, 1

def ZeroSAny20Settings(**kwargs):
    return ZeroSAnySettings(ZeroSBlockAnySettings, [3,3,3], **kwargs)
def ZeroSAny32Settings(**kwargs):
    return ZeroSAnySettings(ZeroSBlockAnySettings, [5,5,5], **kwargs)
def ZeroSAny44Settings(**kwargs):
    return ZeroSAnySettings(ZeroSBlockAnySettings,  [7,7,7], **kwargs)
def ZeroSAny56Settings(**kwargs):
    return ZeroSAnySettings(ZeroSBlockAnySettings, [9,9,9], **kwargs)
def ZeroSAny68Settings(**kwargs):
    return ZeroSAnySettings(ZeroSBlockAnySettings, [11,11,11], **kwargs)
def ZeroSAny80Settings(**kwargs):
    return ZeroSAnySettings(ZeroSBlockAnySettings, [13,13,13], **kwargs)
def ZeroSAny92Settings(**kwargs):
    return ZeroSAnySettings(ZeroSBlockAnySettings, [15,15,15], **kwargs)
def ZeroSAny104Settings(**kwargs):
    return ZeroSAnySettings(ZeroSBlockAnySettings, [17,17,17], **kwargs)
def ZeroSAny110Settings(**kwargs):
    return ZeroSAnySettings(ZeroSBlockAnySettings, [18,18,18], **kwargs)
def ZeroSAny134Settings(**kwargs):
    return ZeroSAnySettings(ZeroSBlockAnySettings, [22,22,22], **kwargs)
def ZeroSAny152Settings(**kwargs):
    return ZeroSAnySettings(ZeroSBlockAnySettings, [25,25,25], **kwargs)
def ZeroSAny170Settings(**kwargs):
    return ZeroSAnySettings(ZeroSBlockAnySettings, [28,28,28], **kwargs)
def ZeroSAny182Settings(**kwargs):
    return ZeroSAnySettings(ZeroSBlockAnySettings, [30,30,30], **kwargs)
def ZeroSAny302Settings(**kwargs):
    return ZeroSAnySettings(ZeroSBlockAnySettings, [50,50,50], **kwargs)
def ZeroSAny482Settings(**kwargs):
    return ZeroSAnySettings(ZeroSBlockAnySettings, [80,80,80], **kwargs)
def ZeroSAny602Settings(**kwargs):
    return ZeroSAnySettings(ZeroSBlockAnySettings, [100,100,100], **kwargs)
def ZeroSAny902Settings(**kwargs):
    return ZeroSAnySettings(ZeroSBlockAnySettings, [150,150,150], **kwargs)
def ZeroSAny1202Settings(**kwargs):
    return ZeroSAnySettings(ZeroSBlockAnySettings, [200,200,200], **kwargs)

# args = parser.parse_args()
def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True   
 
if __name__ == '__main__':
    # w = torch.empty(6, 3, 1, 1)
    # nn.init.dirac_(w,2)
    # # w = torch.empty(3, 24, 5, 5)
    # # nn.init.dirac_(w, 3)
    # print('w', w)
    import cProfile
    import re
    import pstats

    os.environ['CUDA_VISIBLE_DEVICES'] = '5'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # def profileit():

    #     seed_torch()

    #     net = ZeroSAny110Settings(givenA=[1,0
    # ], givenB=[-1,0
    # ], Settings='BnReluConv_ConvStride2ResLike_2Chs_Adam_SamllST')#S4O4
    #     d = 0.5*torch.ones(128, 3, 32, 32).to(device)
    #     net.apply(weights_init) 
    #     net = net.to(device)
    #     out = net(d)
        
    # cProfile.run('profileit()', '/media3/clm/SuppressionConnection/runs/output.prof')

    # # Create a pstats.Stats object
    # p = pstats.Stats('/media3/clm/SuppressionConnection/runs/output.prof')

    # # Sort the statistics by the cumulative time spent in the function
    # p.sort_stats(pstats.SortKey.CUMULATIVE)

    # # Print the statistics
    # p.print_stats()
        
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_torch()


    net = ZeroSAny182Settings(givenA=[1.5, -1.0, 0.5], givenB=[-1.0, 0, 0
], Settings='BnReluConv_ConvStride2ResLike_2Chs_Adam_SamllST_EulerSwitch010')#S4O4

    d = 0.5*torch.ones(1, 3, 32, 32).to(device)

    net.apply(weights_init) 
    net.apply(weights_init_plot)
    net = net.to(device)
    out = net(d)
    print(out)
    