import os,sys
root_path = os.getcwd()
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
from torchvision import models, datasets
import torch.backends.cudnn as cudnn

from deeprobust.image.attack.deepfool import DeepFool
import deeprobust.image.netmodels.resnet as resnet
# import deeprobust.image.netmodels.AnyOrder as AnyOrder
from models import *
# from deeprobust.image.netmodels import * 
import matplotlib.pyplot as plt

'''
CIFAR10
'''


# load model
# model = resnet.ResNet18().to('cuda')
# model = AnyOrder.ZeroSAny32_Tra().to('cuda')
# model = nn.DataParallel(model)

# cudnn.benchmark = True
# print("Load network")

"""
Change the model directory here
"""
model = torch.load("/media/bdc/clm/OverThreeOrders/CIFAR/runs/cifar10/ZeroSNet/WithRobSGD_noLS/ZeroSAny32_Tra/PL1.0ini_step1.0a0_1.4932a1_-0.574a2_0.0855a3_-0.004_b0_-2.103_b1_2.8039_b2_-1.684_b3_0.4006_sche_stepSGD_miniNone_BS128_LR0.1epoch160warm0CosConverOrder5PreAct1234_Ch4_2022-03-11T:22:57:41/model_best.pth.tar").to('cuda')
model = nn.DataParallel(model)

cudnn.benchmark = True
# args.start_epoch = checkpoint['epoch']
# best_prec1 = checkpoint['best_prec1']
# net.load_state_dict(checkpoint['state_dict'])
# model.load_state_dict(torch.load("/media/bdc/clm/OverThreeOrders/CIFAR/runs/cifar10/ZeroSNet/WithRobSGD_noLS/ZeroSAny32_Tra/PL1.0ini_step1.0a0_1.4932a1_-0.574a2_0.0855a3_-0.004_b0_-2.103_b1_2.8039_b2_-1.684_b3_0.4006_sche_stepSGD_miniNone_BS128_LR0.1epoch160warm0CosConverOrder5PreAct1234_Ch4_2022-03-11T:22:57:41/model_state_best.pth.tar",False))
# model.load_state_dict(torch.load("./trained_models/CIFAR10_ResNet18_epoch_20.pt"))

model.eval()

# load dataset
testloader = torch.utils.data.DataLoader(
    datasets.CIFAR10('/media/bdc/clm/OverThreeOrders/CIFAR/data/cifar10', train = False, download = True,
    transform = transforms.Compose([transforms.ToTensor()])),
    batch_size = 1, shuffle = True)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# choose attack example
X, Y = next(iter(testloader))
X = X.to('cuda').float()

# run deepfool attack
adversary = DeepFool(model)
print('adversary',adversary)
AdvExArray = adversary.generate(X, Y).float()

# predict
pred = model(AdvExArray).cpu().detach()

# print and save result
print('===== RESULT =====')
print("true label:", classes[Y])
print("predict_adv:", classes[np.argmax(pred)])

AdvExArray = AdvExArray.cpu().detach().numpy()
AdvExArray = AdvExArray.swapaxes(1,3).swapaxes(1,2)[0]

plt.imshow(AdvExArray, vmin = 0, vmax = 255)
plt.savefig('./adversary_examples/cifar_advexample_deepfool.png')

