from torchvision.models import resnet50
from thop import profile
import torch
from models import *
import string
import os
import pandas as pd
import numpy as np
models = [
   'ZeroSAny20_Tra',
   'ZeroSAny32_Tra',
   'ZeroSAny44_Tra',
   'ZeroSAny56_Tra',
   'ZeroSAny68_Tra',
   'ZeroSAny80_Tra',
   'ZeroSAny92_Tra',
   'ZeroSAny104_Tra',
#   'ZeroSAny110_Tra',
]
stepsS = ['1',
         '2',
         '2',
         '4',
         '4',
         '4',
         '3',
         '3',
         '3',         
]
ConverOrds = ['1',
             '2',
             '0',
             '4',
             '2',
             '0',
             '2',
             '1', 
             '0',
              ]
givenAs = ['1 0',
          '1 0',
          '1 0',
         '1.49323762497707 -0.574370781405754 0.0855838379295368 -0.00445068150085398',
         '1.49323762497707 -0.574370781405754 0.0855838379295368 -0.00445068150085398',
         '1.49323762497707 -0.574370781405754 0.0855838379295368 -0.00445068150085398',
         '0.3333333333333333 0.5555555555555556 0.1111111111111111',
         '0.3333333333333333 0.5555555555555556 0.1111111111111111',
         '0.3333333333333333 0.5555555555555556 0.1111111111111111',

          ]
givenBs = ['-1 0',
          '-1.5 0.5',
          '0 -2',
           '-2.10313656405320 2.80393876806197 -1.68484817541425 0.400601121454727',
           '-2.10313656405320 5.60787753612393 -7.29272571153819 3.20453988951669',
           '0.400601121454727 -1.68484817541425 1.80393876806197 -2.10313656405320',
          '-1.7777777777777778 0 0',
           '-1.7777777777777778 1.7777777777777778 -1.7777777777777778',
           '0.592592592596 -0.592592592596 -0.592592592596',
           ]
num_classes = [10, 100]
data_arr = []
for NC in num_classes:
    for arch in models:
        for s in range(len(stepsS)):
            givenA = []
            givenB = []
            givenAtem = givenAs[s].split()
            givenBtem = givenBs[s].split()
            for a in givenAtem:
                givenA += [float(a)]
            for b in givenBtem:
                givenB += [float(b)]                
            net = eval(arch)(num_classes=NC, givenA=givenA, givenB=givenB)
            input = torch.randn(1, 3, 32, 32)
            flops, params = profile(net, inputs=(input, ))
            # flops = flops/1000**3
            # params = params/1000**2
            tmp_arr = [arch, NC, givenAs[s], givenBs[s], stepsS[s], ConverOrds[s], flops, params]
            data_arr.append(tmp_arr)

            
df = pd.DataFrame(data_arr, columns=['Model','# classes','givenA','givenB','Steps','Convergence order','FLOPs','# Params'])
print(df)
save_path ='/media/bdc/clm/OverThreeOrders/CIFAR/runs/ParamsAndFLOPs/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

df.to_csv(save_path+'ParamsAndFLOPs.csv')