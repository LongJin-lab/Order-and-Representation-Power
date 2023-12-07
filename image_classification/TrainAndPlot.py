from comet_ml import Experiment, OfflineExperiment

from absl import app, flags
from easydict import EasyDict
import numpy as np
import random

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

# import tensorwatch as tw
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import _LRScheduler
import time

import torchvision
import torchvision.transforms as transforms

import os
import argparse
# import torch.onnx
# import netron
# import onnx
# from tensorboardX import SummaryWriter
# import sys
# sys.path.append(r"directory")

from models import *
from noise import *

from datetime import datetime
import errno
import shutil
import pandas as pd

# import homura
from torchvision import transforms
# from timm.loss import LabelSmoothingCrossEntropy
from asam import ASAM, SAM, coesASAM, coesSAM
# from bypass_bn import enable_running_stats, disable_running_stats
from torchsummaryX import summary
from torchvision import transforms
import matplotlib.pyplot as plt

import matplotlib
import torchextractor as tx

from torch.utils.data import DataLoader
import math
import warnings
 
# from robustbench.data import load_cifar10

# from robustbench.utils import load_model 
# import foolbox as fb
from torch.autograd import Variable
# from models import (convert_splitbn_model, create_model, load_checkpoint,
#                          model_parameters, resume_checkpoint, safe_model_name)
warnings.filterwarnings('ignore')

# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cuda:0'

print('device',device)


# parser = argparse.ArgumentParser(description='ZeorSNet CIFAR')
parser = argparse.ArgumentParser(description='ZeorSNet CIFAR')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')#TODO
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--epoch', type=int, default=3, help='training epoch')
parser.add_argument('--warm', type=int, default=0, help='warm up training phase')
parser.add_argument('--data', default='/media/bdc/clm/data', type=str)# /media_HDD_1/lab415/clm/OverThreeOrders/OverThreeOrders/CIFAR/data
parser.add_argument('--dataset', default='cifar10', type=str)#cifar10mnist
parser.add_argument('--arch', '-a', default='convnext_conver_nano_hnf', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--bs', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--weight-decay', '--wd', default=4e-5, type=float,
                    metavar='W', help='weight decay (default: 4e-5 for mobile models)')
parser.add_argument('--opt', default='SGD', type=str)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--seed", default=42, type=int)

parser.add_argument("--ex", default=0, type=int)
parser.add_argument("--notes", default='', type=str)
parser.add_argument('--PL', type=float, default=1.0)
parser.add_argument('--sche', default='cos', type=str)
# parser.add_argument('--coe_ini', type=float, default=1)
parser.add_argument('--share_coe', type=bool, default=False)
# parser.add_argument('--given_coe', default=[1.0 / 3, 5.0 / 9, 1.0 / 9, 16.0 / 9], nargs='+', type=float)
parser.add_argument('--given_coe', default=None, nargs='+', type=float)
parser.add_argument('--steps', type=int, default=3)
parser.add_argument('--ini_stepsize', default=1, type=float)
parser.add_argument('--givenA', default=[1, 0], nargs='+', type=float)#S5
parser.add_argument('--givenB', default=[-1, -1], nargs='+', type=float)#S5O1
parser.add_argument("--ConverOrd", default=4, type=int, help="")


parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')

parser.add_argument("--minimizer", default=None, type=str, help="ASAM or SAM.")
parser.add_argument("--smoothing", default=None, type=float, help="Label smoothing.")#0.1
parser.add_argument("--rho", default=0.5, type=float, help="Rho for ASAM.")
parser.add_argument("--eta", default=0.01, type=float, help="Eta for ASAM.")
    
parser.add_argument("--adv_train", action='store_true')
parser.add_argument("--adv_test", action='store_true')
parser.add_argument("--eps", default=0.03137255, type=float, help="")
parser.add_argument("--eps_iter", default=0.01, type=float, help="step size for each attack iteration")
parser.add_argument("--nb_iter", default=10, type=int, help="Number of attack iterations.")
parser.add_argument("--norm", default=np.inf, type=float, help="Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.")
parser.add_argument("--clip_min", default=None, type=float, help="Minimum float value for adversarial example components.")
parser.add_argument("--clip_max", default=None, type=float, help="Maximum float value for adversarial example components.")
parser.add_argument("--save_path", default='./runs/features/', type=str, help="save path")
parser.add_argument("--Settings", default='_BnReluConvConvAllEle', type=str, help="Settings")#mnistLearnCoeExpDecayLearnDecay
parser.add_argument("--req_ord", default=5, type=int, help="")
parser.add_argument("--mode", default='', type=str, help="")
parser.add_argument("--ini_block_shift", default=None, type=int, help="")
# parser.add_argument("--lr_coe", default=None, type=int, help="")
parser.add_argument("--data_aug", default=None, type=str, help="")
parser.add_argument("--inp_noi", default=None, type=float, help="")

# TODO: print inputs
    
# args = parser.parse_args()
def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False#True
	torch.backends.cudnn.deterministic = True   
    
args =parser.parse_known_args()[0]

seed_torch(args.seed)


givenA_text = ''
givenB_text = ''
if args.givenA is not None:
    for i in range(len(args.givenA)): 
        givenA_text += "a"+str(i)+"_"+str(args.givenA[i])[:6]
        givenB_text += "_b"+str(i)+"_"+str(args.givenB[i])[:6]
else:
    givenA_text = ''
    givenB_text = ''
if args.share_coe:
    share_coe_text = 'share_coe_True'
else:
    share_coe_text = 'share_coe_False'
if args.dataset == "cifar10" or args.dataset == "stl10":
    args.num_classes = 10
    if args.sche == 'step' and args.epoch is None:
        args.epoch = 160
if args.dataset == "cifar100":
    args.num_classes = 100
    if args.sche == 'step'and args.epoch is None:
        args.epoch = 300
if args.dataset == "mnist":
    args.num_classes = 10
    if args.sche == 'step' and args.epoch is None:
        args.epoch = 10
    elif args.sche == 'cos' and args.epoch is None:
        args.epoch = 5        
if args.dataset == "svhn":
    args.num_classes = 10
    if args.sche == 'step' and args.epoch is None:
        args.epoch = 40        
path_base = './runs/' + args.dataset +str(args.adv_train)+'/ZeroSNet/WithRobSGD_noLS_adv/'
if args.adv_train:
    print('adv_train=True')
#    path_base = './adv_train_runs' + path_base.replace('./', '/')+'eps'+str(args.eps)+'/'
    if not args.save_path == None:
        args.save_path = './runs/adv_train_runs'+'eps'+str(args.eps)+'/'+args.save_path.replace('./', '/').replace('runs', '')
try:
    os.makedirs(path_base)
except OSError as exc:  # Python >2.5
    if exc.errno == errno.EEXIST and os.path.isdir(path_base):
        pass
    else:
        raise
try:
    os.makedirs(args.save_path)
except OSError as exc:  # Python >2.5
    if exc.errno == errno.EEXIST and os.path.isdir(args.save_path):
        pass
    else:
        raise      

    
if args.save_path == None:
    args.save_path = path_base + args.arch + '/ini_st' + str(
        args.ini_stepsize)  + givenA_text + givenB_text +'_sche_' + args.sche + str(args.opt) + \
                    '_mini'+args.minimizer+'_BS' + str(args.bs) + '_LR' + \
                    str(args.lr) + 'epoch' + \
                    str(args.epoch) + 'warm' + str(args.warm) + \
                    args.notes +'eps'+str(args.eps)+ 'eps_iter'+str(args.eps_iter)+'nb_iter'+str(args.nb_iter)+\
                        '_G4_'+\
                    "{0:%Y-%m-%dT:%H:%M:%S/}".format(datetime.now())

experiment = Experiment(
# experiment = OfflineExperiment(
    api_key="YourAPIKey",
    project_name="project_name",
    workspace="workspace",
    # auto_histogram_weight_logging=True,
    # offline_directory=path_base+"CometData",
)

# checkpoint
if args.checkpoint is None:
    args.checkpoint = args.save_path+'checkpoint.pth.tar'
    print('args.checkpoint', args.checkpoint)
# print('givenB', args.givenB, 'givenA', args.givenA)

hyper_params = vars(args)
experiment.log_parameters(hyper_params)

def coe_constrains(coe_As, coe_Bs,req_ord=1,prin=False):
    # coe_A0 = [torch.tensor(-1.).to(device).reshape(1)]
    # coe_B0 = [torch.tensor(0.).to(device).reshape(1)]
    # coe_As = coe_A0 + coe_As
    # coe_Bs = coe_B0 + coe_Bs
    # print('coe_Bs',coe_Bs)
    # TODO: a0 should not be trained
    coe_A0 = torch.tensor(-1.).to(device).reshape(1)
    coe_B0 = torch.tensor(0.).to(device).reshape(1)
    coe_As = torch.cat((coe_A0,coe_As),0)
    coe_Bs = torch.cat((coe_B0,coe_Bs),0)
    err = torch.tensor([]).to(device)

    # coe_As = coe_A0 + coe_As
    # coe_Bs = coe_B0 + coe_Bs    
    # err = []
    for i in range(0, req_ord+1):
        if i == 0:
            # err += [sum(coe_As)]
            # err = torch.cat((err, torch.sum(coe_As).reshape(1) ),0)
            err = torch.sum(coe_As).reshape(1)
        else:
            As = 0
            Bs = 0
            for j in range(1,len(coe_As)):
                As += j**i*coe_As[j]
            As *= 1/math.factorial(i)
            for j in range(0,len(coe_Bs)):
                Bs += j**(i-1)*coe_Bs[j]
            Bs *= 1./math.factorial(i-1)
            # err += [(-1)**i*(As+Bs)]
            err = torch.cat((err, ((-1)**i*(As+Bs)).reshape(1) ),0)

    if prin:
        print('ord'+str(i),err)
        print('coe_As,coe_Bs',coe_As,coe_Bs)
    return err

def plot_feature(net, epoch):
    # plt.style.use('ieee')
    plt.style.use(['science','ieee'])  
    # print(matplotlib.rcParams)
    matplotlib.rcParams.update(
        {
            # 'text.usetex': False,
            # 'font.family': 'stixgeneral',
            
            # 'font.size': 24.0,        
            # 'legend.fontsize': 'medium',
            'xtick.labelsize': 'large',
            'ytick.labelsize': 'large',         
        'axes.labelsize': 'x-large',
        'legend.frameon' : True,
        'legend.fontsize' : 'large',
        'legend.fancybox' : False, 
        "legend.facecolor" : 'white',   
        'axes.grid': True,  
        'axes.grid.axis': 'x', 
            # 'axes.titlesize': 'large','mathtext.fontset': 'stix',
            }
    )      
    ax = plt.gca()
    ax.ticklabel_format(style='sci', scilimits=(-1,2),useLocale=True)  
    save_path = args.save_path+args.Settings+args.arch+str(args.epoch)+givenB_text+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    net.eval()
    # with torch.no_grad():         
    for i, (images, labels) in enumerate(testloader):

        if i==0:#(i + 1) % 100 == 0:
            print('batch_number [{}/{}]'.format(i + 1, len(testloader)))
            for j in range(len(images)):
                if j == 1:
                    if args.dataset == 'mnist':                    
                        image = images[j].resize(28, 28).to(device) 
                    elif 'cifar' in args.dataset:                    
                        image = images[j].to(device)                         
                    print('labels[j]',labels[j])                    
                    break
            break


    # plt.rcParams.update({'font.size':14})
    if args.dataset == 'mnist':
        d = image.reshape(1,1,28,28)#.reshape(28,28)
        loc = 2
    elif 'cifar' in args.dataset:
        d = image.reshape(1,3,32,32)#.reshape(28,28)
        loc = 8

    # plt.imshow(d)
    d= d.to(device)

    df = pd.DataFrame()
    plt.xlabel('$n$')#, fontsize=24)
    plt.ylabel('$x_n$')#, fontsize=24)
    out_0 = net(d) 
    MaxDif = []
    MaxDif2 = []
    if 'convnext' in args.arch:
        module_filter_fn = lambda module, name: isinstance(module, torch.nn.Identity) and 'end_block_vis' in name #torch.nn.Identity)
    else:
        module_filter_fn = lambda module, name: isinstance(module, torch.nn.Identity)
    lb = 0
    ub = 50
    for mag in range(lb,ub,1):

        # d = d + mag/1000
        if args.dataset == 'mnist':
            x = d + mag/100*torch.rand(1,1,28,28).to(device)
        elif 'cifar' in args.dataset:
            x = d + mag/100*torch.rand(1,3,32,32).to(device)       

        model = tx.Extractor(net, module_filter_fn=module_filter_fn)
        out, features = model(x) 
        
        fmap = []
        fmap2 = []

        feature_shapes = {name: f.shape for name, f in features.items()}
        for name, f in features.items():
            if len(f.data.shape) == 2:
                # fmap += [float(f.data[f.data.shape[0]//2][f.data.shape[1]//2])]
                # print('f.data',f.data.shape)                
                fmap += [float(f.data[f.data.shape[0]//2][loc])]                
                fmap2 += [float(f.data[f.data.shape[0]//2][0])] 

            elif len(f.data.shape) == 4:
                fmap += [float(f.data[f.data.shape[0]//2][f.data.shape[1]//2][f.data.shape[2]//2][f.data.shape[3]//2])]
                fmap2 += [float(f.data[f.data.shape[0]//2][f.data.shape[1]//2][0][0])]
                #[float(f.data.mean())]
        fmap = np.array(fmap)
        fmap2 = np.array(fmap2)                
        if mag == lb: 
            fmap_ori = fmap
            fmap_ori2 = fmap2
        fmap_gap = fmap-fmap_ori
        fmap2_gap = fmap2-fmap_ori2
        
        MaxDif += [fmap]
        MaxDif2 += [fmap2]
         
        ts = np.arange(0, len(fmap), 1)
        pic1, = plt.plot(ts[0:-3], fmap_gap[0:-3], linewidth = 1,color='indianred',alpha=0.4,linestyle='-', marker = 'o', markersize=1, markeredgecolor='black', markerfacecolor='white') 
        # df = pd.concat([df,df_row])
        pic2, = plt.plot(ts[0:-3], fmap2_gap[0:-3], color='steelblue',linewidth = 1,alpha=0.4,linestyle='-', marker = 'o', markersize=1, markeredgecolor='black', markerfacecolor='white')
        if mag == lb:
            plt.legend(handles=[pic1,pic2], labels=['Center','Corner'])
        plt.xlabel('$n$')#, fontsize=24)
        plt.ylabel('$\Delta x_n$')#,         
        
              
    # plt.show()      

        
    file_name = '/features_normal_ep'+str(epoch)
    # df.to_csv(save_path+file_name+'.csv',index = False)
    ax.figure.savefig(save_path+file_name+'.pdf')
    plt.close()
    MaxDif_adv = []
    MaxDif2_adv = []  
    ax_adv = plt.gca()
    ax_adv.ticklabel_format(style='sci', scilimits=(-1,2),useLocale=True)   
    plt.xlabel('$n$')#, fontsize=24)
    plt.ylabel('$\Delta x_n$')#, fontsize=24)           
    for mag in range(1,10):#10,30,1):
 
        x_adv = projected_gradient_descent(net, d, eps=4.0/255*mag, eps_iter=args.eps_iter, nb_iter=args.nb_iter, norm=args.norm)     
                   
        model = tx.Extractor(net, module_filter_fn=module_filter_fn)
        out, features = model(x_adv) 
        fmap = []
        fmap2 = []
        feature_shapes = {name: f.shape for name, f in features.items()}
        for name, f in features.items():

            if len(f.data.shape) == 2:
                # fmap += [float(f.data[f.data.shape[0]//2][f.data.shape[1]//2])]
                # print('f.data',f.data.shape)                
                fmap += [float(f.data[f.data.shape[0]//2][loc])]                
                fmap2 += [float(f.data[f.data.shape[0]//2][0])] 

            elif len(f.data.shape) == 4:
                fmap += [float(f.data[f.data.shape[0]//2][f.data.shape[1]//2][f.data.shape[2]//2][f.data.shape[3]//2])]
                fmap2 += [float(f.data[f.data.shape[0]//2][f.data.shape[1]//2][0][0])]
                #[float(f.data.mean())]       
        fmap = fmap-fmap_ori
        fmap2 = fmap2-fmap_ori2                         
        MaxDif_adv += [fmap]
        MaxDif2_adv += [fmap2]
        df_row = pd.DataFrame([fmap])
        ts = np.arange(0, len(fmap), 1)

        pic1, = plt.plot(ts[0:-3], fmap[0:-3], linewidth = 1,color='darkorange',alpha=0.4,linestyle='-', marker = 'o', markersize=1, markeredgecolor='black', markerfacecolor='white') 
        df = pd.concat([df,df_row])
        pic2, = plt.plot(ts[0:-3], fmap2[0:-3], color='darkcyan',linewidth = 1,alpha=0.4,linestyle='-', marker = 'o', markersize=1, markeredgecolor='black', markerfacecolor='white')
        if mag == 1:
            plt.legend(handles=[pic1,pic2], labels=['Center$+$AT','Corner$+$AT'])              

        
    out = net(x) 
    out_adv = net(x_adv) 
    
    print('out_diff,out_diff_adv', torch.abs(out-out_0).mean(),torch.abs(out_adv-out_0).mean() )
  
    # plt.tick_params(labelsize=12)
    # print("==>> type(plt.ylim()): ", plt.ylim()

    file_name = '/features_adv_ep'+str(epoch)
    df.to_csv(save_path+file_name+'.csv',index = False)
    ax_adv.figure.savefig(save_path+file_name+'.pdf')
    # plt.show()
    plt.close()
    ax_dif = plt.gca()
    ax_dif.ticklabel_format(style='sci', scilimits=(-1,2),useLocale=True)  
    plt.xlabel('$n$')#, fontsize=24)
    plt.ylabel('$\max{|\Delta x_n|}$')#, fontsize=24)         
    MaxDif = np.array(MaxDif)
    MaxDif = np.max(MaxDif, axis=0)-np.min(MaxDif, axis=0)
    MaxDif = MaxDif.reshape(np.shape(ts))      
    plt.plot(ts[0:-3], MaxDif[0:-3], color='indianred',label='Center') 
    print('MaxOutDif',MaxDif[-1])
    print('MaxOutFeaDif2',MaxDif[-3])


    MaxDif = np.array(MaxDif2)
    MaxDif = np.max(MaxDif, axis=0)-np.min(MaxDif, axis=0)
    MaxDif = MaxDif.reshape(np.shape(ts))      
    plt.plot(ts[0:-3], MaxDif[0:-3],color='steelblue', label='Corner')
    # plt.legend() 
    print('MaxOutDif2',MaxDif[-1])
    print('MaxOutFeaDif2',MaxDif[-3])

    MaxDif = np.array(MaxDif_adv)
    MaxDif = np.max(MaxDif, axis=0)-np.min(MaxDif, axis=0)
    MaxDif = MaxDif.reshape(np.shape(ts))      
    plt.plot(ts[0:-3], MaxDif[0:-3], color='darkorange',label='Center$+$AT') 
    print('MaxOutDif_adv',MaxDif[-1])
    print('MaxOutFeaDif_adv',MaxDif[-3])

    # plt.show()

    MaxDif = np.array(MaxDif2_adv)
    MaxDif = np.max(MaxDif, axis=0)-np.min(MaxDif, axis=0)
    MaxDif = MaxDif.reshape(np.shape(ts))      
    plt.plot(ts[0:-3], MaxDif[0:-3],color='darkcyan', label='Corner$+$AT')
    plt.legend(facecolor='w',framealpha=0.5) 
    print('MaxOutDif2_adv',MaxDif[-1])
    print('MaxOutFeaDif2_adv',MaxDif[-3])    

    # plt.show()
    

    file_name = '/diff_ep'+str(epoch)
    df.to_csv(save_path+file_name+'.csv',index = False)
    ax_dif.figure.savefig(save_path+file_name+'.pdf')
    plt.close()


def train(epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    end = time.time()
    net.train()
    
    loss = 0.
    acc = 0.
    cnt = 0.
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        '''
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer_sgd_sgd.zero_grad()
        if 'zeros' in args.arch or 'ZeroS' in args.arch:
            outputs, coes, stepsize = net(inputs)
        elif 'MResNet' in args.arch:
            outputs, coes = net(inputs)
        else:
            outputs = net(inputs)

        loss = criterion(outputs, targets)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        # print('acc1', acc1, 'acc5', acc5)
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))

        loss.mean().backward()#asam
        # loss.backward() 
        minimizer.ascent_step()#asam
        # optimizer.step()
        
        if 'zeros' in args.arch or 'ZeroS' in args.arch:
            outputs, coes, stepsize = net(inputs)
        elif 'MResNet' in args.arch:
            outputs, coes = net(inputs)
        else:
            outputs = net(inputs)
            
        criterion(outputs, targets).mean().backward()#asam
        minimizer.descent_step()#asam

        if epoch > args.warm:
            train_scheduler_sgd.step()#epoch)
        if epoch <= args.warm:
            warmup_scheduler.step()
        batch_time.update(time.time() - end)
        end = time.time()
'''
        
        # if args.dataset == 'svhn':
        #     inputs, targets = inputs.to(device), targets.to(device)#.long().squeeze()
        # else:
        inputs, targets = inputs.to(device), targets.to(device)            
        # print('inputs.max(), inputs.min()', inputs.max(), inputs.min())
            # Initialize PID parameters
        p_P = 1.0
        p_D = 0.2
        p_I = 0.01

        L_prev = 0.
        L_cum = 0.
        if args.adv_train:
            # Replace clean example with adversarial example for adversarial training
            inputs = projected_gradient_descent(net, inputs, eps=args.eps, eps_iter=args.eps_iter, nb_iter=args.nb_iter, norm=args.norm)     
            # print('inputs.max()adv, inputs.min()adv', inputs.max(), inputs.min())

            # TODO: add args.adv_train, args.eps
        if args.minimizer in ['ASAM', 'SAM']:
            minimizer.optimizer.zero_grad()

            # if 'zeros' in args.arch or 'ZeroS' in args.arch:
            #     outputs, coes, stepsize = net(inputs)
            # elif 'MResNet' in args.arch:
            #     outputs, coes = net(inputs)
            # else:
            #     outputs = net(inputs)
            outputs = net(inputs)
                
            # enable_running_stats(net)

            batch_loss = criterion(outputs, targets)
            # batch_loss.backward()#asam
            
            batch_loss.mean().backward()#asam
            # loss.backward() 
            minimizer.ascent_step()#asam
            # optimizer.step()
            
            # if 'zeros' in args.arch or 'ZeroS' in args.arch:
            #     outputs, coes, stepsize = net(inputs)
            # elif 'MResNet' in args.arch:
            #     outputs, coes = net(inputs)
            # else:
            #     outputs = net(inputs)
            outputs = net(inputs)
                
            # disable_running_stats(net)
        
            # criterion(outputs, targets).back.meanward()#asam
            criterion(outputs, targets).mean().backward()#asam
            
            minimizer.descent_step()#asam
            
            with torch.no_grad():
                loss += batch_loss.sum().item()
                acc += (torch.argmax(outputs, 1) == targets).sum().item()
            cnt += len(targets)
        
            if epoch >= args.warm:
                train_scheduler_sgd.step(epoch)#epoch)
            if epoch < args.warm:
                warmup_scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()
        elif args.minimizer in ['coesASAM', 'coesSAM']:
            minimizer.optimizer.zero_grad()
            
            for n, param in net.named_parameters():
                if not 'coes' in n:
                    param.requires_grad = False
                if 'LearnStepSize' in args.Settings and 'coes_stepsize' in n:
                    param.requires_grad = True
                if 'LearnCoeA' in args.Settings and 'coesA' in n:
                    param.requires_grad = True
                if 'LearnCoeB' in args.Settings and 'coesB' in n:
                    param.requires_grad = True
                    
            outputs = net(inputs)
            batch_loss = criterion(outputs, targets)
            batch_loss.mean().backward()#asam
            minimizer.ascent_step()#asam
            for n, param in net.named_parameters():
                param.requires_grad = True
                if ('LearnStepSize' not in args.Settings and 'coes_stepsize' in n):
                    param.requires_grad = False
                if ('LearnCoeA' not in args.Settings and 'coesA' in n):
                    param.requires_grad = False
                if ('LearnCoeB' not in args.Settings and 'coesB' in n):
                    param.requires_grad = False
                    
            outputs = net(inputs)

            criterion(outputs, targets).mean().backward()#asam
            
            minimizer.descent_step()#asam
            
            with torch.no_grad():
                loss += batch_loss.sum().item()
                acc += (torch.argmax(outputs, 1) == targets).sum().item()
            cnt += len(targets)
        
            if epoch >= args.warm:
                train_scheduler_sgd.step(epoch)#epoch)
            if epoch < args.warm:
                warmup_scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()            
        else:
            optimizer_sgd.zero_grad()
            optimizer_adam.zero_grad()
            if args.inp_noi:
                # print('noise_coff, inputs_pertur.size()',noise_coff, inputs_pertur.size())
                inputs = inputs + args.inp_noi * \
                    torch.autograd.Variable(torch.randn(inputs.size()).cuda(), requires_grad=False)
            # inputs = Variable(inputs,requires_grad=True)                
            outputs = net(inputs)
            loss = criterion(outputs, targets) #+0.*loss_diff
            if "PIDloss" in args.Settings:
                L_PID = p_P * loss + p_D * (loss - L_prev) - p_I * L_cum
                L_PID.backward()
                optimizer_sgd.step()
                optimizer_adam.step()
                L_prev = loss
                L_cum += loss
            else:
                loss.backward()          
                optimizer_sgd.step()
                # if not 'convnext' in args.arch and 'Adam' in args.Settings:
                optimizer_adam.step()
            training_stage = 'Normal training'


                # print('loss_rob',loss_rob)
            #retain_graph=True)
            # optimizer_adam.step()

            # optimizer_sgd.zero_grad()
            # optimizer_adam.zero_grad()
            # loss.backward()
            # optimizer_adam.step()

            # print('loss',loss.mean())
            # handle.remove()
            # del features_in, features_out, f_ins, f_outs
            if epoch > args.warm:
                train_scheduler_sgd.step(epoch)
                train_scheduler_adam.step(epoch)
                
            if epoch <= args.warm:
                warmup_scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()   
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            # print('acc1', acc1, 'acc5', acc5)
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0)) 
            # if args.Settings:    
            #     for name, param in net.named_parameters():
            #         if 'Learn' in name or 'Fix' in name:
            #             print(name, param.data.mean())                    
    # loss /= cnt
    # acc *= 100. / cnt
    # print(f"Epoch: {epoch}, Train accuracy: {acc:6.2f} %, Train loss: {loss:8.5f}")        
    # experiment.log_metric("Train/Average loss", loss, step=epoch)
    # experiment.log_metric("Train/Accuracy-top1", acc, step=epoch)
    # experiment.log_metric("Train/Time", batch_time.sum, step=epoch)
    # plt.legend()
    # plt.show() 
    print("Epoch:", epoch,'Train set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(losses.avg, top1.avg))
    # print('diff_loss',diff_loss)
    # print('training_stage',training_stage)
    # print('crt_ord',crt_ord)
    # print('loss_A',loss_A.mean())
    
    if 'Learn' in args.Settings:
        Dcoes_As = torch.tensor([]).to(device)
        Dcoes_Bs = torch.tensor([]).to(device)
        for name, p in net.named_parameters():
            if p.grad is not None: 
                if name:
                    if 'DcoesA' in name: # and 'bn3' not in name:
                        # print(name)
                        # sum_A = sum_A+p
                        # coes_As = coes_As+[p]
                        Dcoes_As = torch.cat((Dcoes_As, p ),0)
                        print(name,p.data,'grad:',float(p.grad.data))
                    if 'DcoesB' in name: # and 'bn3' not in name:
                        # print(name)
                        # sum_B = sum_B+p
                        # coes_Bs = coes_Bs+[p]
                        Dcoes_Bs = torch.cat((Dcoes_Bs, p ),0)  
                        print(name,p.data,'grad:',p.grad.data)
                    if 'stepsize' in name:
                        print(name,p.data,'grad:',float(p.grad.data))
                        
    #     err = coe_constrains(coes_As, coes_Bs,req_ord=args.req_ord,prin=True)
    # print('err',err )                       
    # return acc, loss, batch_time.sum        
    
        # print('hook removed')
    # print('Epoch: {:.1f}, Train set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(epoch, losses.avg, top1.avg))

    experiment.log_metric("Train/Average loss", losses.avg, step=epoch)
    experiment.log_metric("Train/Accuracy-top1", top1.avg, step=epoch)
    experiment.log_metric("Train/Accuracy-top5", top5.avg, step=epoch)
    experiment.log_metric("Train/Time", batch_time.sum, step=epoch)

    coes_print = []

    return top1.avg, losses.avg, batch_time.sum



def test(epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # losses_fgm = AverageMeter('Loss_fgm', ':.4e')
    # top1_fgm = AverageMeter('fgmAcc@1', ':6.2f')
    # top5_fgm = AverageMeter('fgmAcc@5', ':6.2f')
    
    losses_pgd = AverageMeter('pgdLoss_pgd', ':.4e')
    top1_pgd = AverageMeter('pgdAcc@1', ':6.2f')
    top5_pgd = AverageMeter('pgdAcc@5', ':6.2f')        
    end = time.time()
    net.eval()
    # loss = 0.
    # acc = 0.
    # cnt = 0.
    # with torch.no_grad():
    if 1:
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if args.inp_noi:
                inputs = inputs + args.inp_noi * \
                    torch.autograd.Variable(torch.randn(inputs.size()).cuda(), requires_grad=False)
            # inputs_fgm = fast_gradient_method(net, inputs, eps=args.eps, norm=args.norm)
            # inputs_pgd = projected_gradient_descent(net, inputs, eps=args.eps, eps_iter=args.eps_iter, nb_iter=args.nb_iter, norm=args.norm)            
            # if 'zeros' in args.arch or 'ZeroS' in args.arch:
            #     outputs, coes, stepsize = net(inputs)
            #     outputs_fgm, coes, stepsize = net(inputs_fgm)
            #     outputs_pgd, coes, stepsize = net(inputs_pgd)
            # elif 'MResNet' in args.arch:
            #     outputs, coes = net(inputs)
            # else:
            #     outputs = net(inputs)
            outputs = net(inputs)
            # outputs_fgm = net(inputs_fgm)
            # outputs_pgd = net(inputs_pgd)                
            loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))  
                            
            # loss_fgm = criterion(outputs_fgm, targets)
            # acc1_fgm, acc5_fgm = accuracy(outputs_fgm, targets, topk=(1, 5))
            # losses_fgm.update(loss_fgm.item(), inputs_fgm.size(0))
            # top1_fgm.update(acc1_fgm[0], inputs_fgm.size(0))
            # top5_fgm.update(acc5_fgm[0], inputs_fgm.size(0))
            
            # loss_pgd = criterion(outputs_pgd, targets)
            # acc1_pgd, acc5_pgd = accuracy(outputs_pgd, targets, topk=(1, 5))
            # losses_pgd.update(loss_pgd.item(), inputs_pgd.size(0))
            # top1_pgd.update(acc1_pgd[0], inputs_pgd.size(0))
            # top5_pgd.update(acc5_pgd[0], inputs_pgd.size(0))
                                 
            batch_time.update(time.time() - end)
            end = time.time()
        # loss /= cnt
        # acc *= 100. / cnt
    # print(f"Epoch: {epoch}, Test accuracy:  {acc:6.2f} %, Test loss:  {loss:8.5f}")

    # return acc, loss, batch_time.sum
    #         loss = criterion(outputs, targets)
    #         acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
    #         losses.update(loss.item(), inputs.size(0))
    #         top1.update(acc1[0], inputs.size(0))
    #         top5.update(acc5[0], inputs.size(0))
    #         batch_time.update(time.time() - end)
    #         end = time.time()
    
    
    
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(losses.avg, top1.avg))
    # print('Test set: Average loss_fgm: {:.4f}, Accuracy_fgm: {:.4f}'.format(losses_fgm.avg, top1_fgm.avg))
    print('Test set: Average loss_pgd: {:.4f}, Accuracy_pgd: {:.4f}'.format(losses_pgd.avg, top1_pgd.avg))

    # # writer.add_scalar('Test/Average loss', losses.avg, epoch)
    # # writer.add_scalar('Test/Accuracy-top1', top1.avg, epoch)
    # # writer.add_scalar('Test/Accuracy-top5', top5.avg, epoch)
    # # writer.add_scalar('Test/Time', batch_time.sum, epoch)

    experiment.log_metric("Test/Average loss", losses.avg, step=epoch)
    experiment.log_metric("Test/Accuracy-top1", top1.avg, step=epoch)
    experiment.log_metric("Test/Accuracy-top5", top5.avg, step=epoch)

    # experiment.log_metric("Test/Average loss_fgm", losses_fgm.avg, step=epoch)
    # experiment.log_metric("Test/Accuracy-top1_fgm", top1_fgm.avg, step=epoch)
    # experiment.log_metric("Test/Accuracy-top5_fgm", top5_fgm.avg, step=epoch)

    experiment.log_metric("Test/Average loss_pgd", losses_pgd.avg, step=epoch)
    experiment.log_metric("Test/Accuracy-top1_pgd", top1_pgd.avg, step=epoch)
    experiment.log_metric("Test/Accuracy-top5_pgd", top5_pgd.avg, step=epoch)
    
    experiment.log_metric("Test/Time", batch_time.sum, step=epoch)
        
    return top1.avg, losses.avg, batch_time.sum

    # with torch.no_grad():
    #     for batch_idx, (inputs, targets) in enumerate(testloader):
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         if 'zerosnet' in args.arch or 'ZeroSNet' in args.arch:
    #             outputs, coes, stepsize = net(inputs)
    #         elif 'MResNet' in args.arch:
    #             outputs, coes = net(inputs)
    #         else:
    #             outputs = net(inputs)
    #         loss = criterion(outputs, targets)
    #         acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
    #         losses.update(loss.item(), inputs.size(0))
    #         top1.update(acc1[0], inputs.size(0))
    #         top5.update(acc5[0], inputs.size(0))
    #         batch_time.update(time.time() - end)
    #         end = time.time()
    # print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(losses.avg, top1.avg))

    # writer.add_scalar('Test/Average loss', losses.avg, epoch)
    # writer.add_scalar('Test/Accuracy-top1', top1.avg, epoch)
    # writer.add_scalar('Test/Accuracy-top5', top5.avg, epoch)
    # writer.add_scalar('Test/Time', batch_time.sum, epoch)

    # return top1.avg, losses.avg, batch_time.sum

def eval_adv(net, loader, eps, eps_iter, nb_iter, norm):
    batch_time = AverageMeter('Time', ':6.3f')
    
    losses_fgm = AverageMeter('Loss_fgm', ':.4e')
    top1_fgm = AverageMeter('fgmAcc@1', ':6.2f')
    top5_fgm = AverageMeter('fgmAcc@5', ':6.2f')
    
    losses_pgd = AverageMeter('pgdLoss_pgd', ':.4e')
    top1_pgd = AverageMeter('pgdAcc@1', ':6.2f')
    top5_pgd = AverageMeter('pgdAcc@5', ':6.2f')        
    end = time.time()
    net.eval()
    # loss = 0.
    # acc = 0.
    # cnt = 0.
    # with torch.no_grad():
    if 1:
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            if args.dataset == 'cifar10':
                # print('==> UnNormalize CIFAR10')
                inputs = UnNormalize(
                    inputs, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            elif args.dataset == 'cifar100':
                inputs = UnNormalize(
                    inputs, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            elif args.dataset == 'mnist':
                    inputs = UnNormalize(
                    inputs,  (0.1307,), (0.3081,))            
                            
            inputs_fgm = fast_gradient_method(net, inputs, eps=eps, norm=norm)
            inputs_pgd = projected_gradient_descent(net, inputs, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=norm)   
            inputs_fgm = F.relu(F.relu(inputs_fgm.mul_(-1).add_(1)).mul_(-1).add_(1))      
            inputs_pgd = F.relu(
                F.relu(inputs_pgd.mul_(-1).add_(1)).mul_(-1).add_(1))     
                             
            if args.dataset == 'cifar10':
                inputs_fgm = Normalize(
                    inputs_fgm, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                inputs_pgd = Normalize(
                    inputs_pgd, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))                
            elif args.dataset == 'cifar100':
                inputs_fgm = Normalize(
                    inputs_fgm, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))         
                inputs_pgd = Normalize(
                    inputs_pgd, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))    
            elif args.dataset == 'mnist':
                    inputs_fgm = Normalize(
                    inputs_fgm,  (0.1307,), (0.3081,))     
                    inputs_pgd = Normalize(
                    inputs_pgd,  (0.1307,), (0.3081,))                       
            # if 'zeros' in args.arch or 'ZeroS' in args.arch:
            #     outputs, coes, stepsize = net(inputs)
            #     outputs_fgm, coes, stepsize = net(inputs_fgm)
            #     outputs_pgd, coes, stepsize = net(inputs_pgd)
            # elif 'MResNet' in args.arch:
            #     outputs, coes = net(inputs)
            # else:
            #     outputs = net(inputs)
            
            # outputs = net(inputs)
            outputs_fgm = net(inputs_fgm)
            outputs_pgd = net(inputs_pgd)        
                    
            # loss = criterion(outputs, targets)
            # acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            # losses.update(loss.item(), inputs.size(0))
            # top1.update(acc1[0], inputs.size(0))
            # top5.update(acc5[0], inputs.size(0))  
                            
            loss_fgm = criterion(outputs_fgm, targets)
            acc1_fgm, acc5_fgm = accuracy(outputs_fgm, targets, topk=(1, 5))
            losses_fgm.update(loss_fgm.item(), inputs_fgm.size(0))
            top1_fgm.update(acc1_fgm[0], inputs_fgm.size(0))
            top5_fgm.update(acc5_fgm[0], inputs_fgm.size(0))
            
            loss_pgd = criterion(outputs_pgd, targets)
            acc1_pgd, acc5_pgd = accuracy(outputs_pgd, targets, topk=(1, 5))
            losses_pgd.update(loss_pgd.item(), inputs_pgd.size(0))
            top1_pgd.update(acc1_pgd[0], inputs_pgd.size(0))
            top5_pgd.update(acc5_pgd[0], inputs_pgd.size(0))
                                 
            batch_time.update(time.time() - end)
            end = time.time()

    # print('eps, eps_iter, nb_iter, norm', eps, eps_iter, nb_iter, norm)
    # print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(losses.avg, top1.avg))
    # print('Test set: Average loss_fgm: {:.4f}, Accuracy_fgm: {:.4f}'.format(losses_fgm.avg, top1_fgm.avg))
    # print('Test set: Average loss_pgd: {:.4f}, Accuracy_pgd: {:.4f}'.format(losses_pgd.avg, top1_pgd.avg))

    experiment.log_metric("Test/Average loss_fgm", losses_fgm.avg)
    experiment.log_metric("Test/Accuracy-top1_fgm", top1_fgm.avg)
    # experiment.log_metric("Test/Accuracy-top5_fgm", top5_fgm.avg, step=0)

    experiment.log_metric("Test/Average loss_pgd", losses_pgd.avg)
    experiment.log_metric("Test/Accuracy-top1_pgd", top1_pgd.avg)
    # experiment.log_metric("Test/Accuracy-top5_pgd", top5_pgd.avg, step=0)
    experiment.log_metric("eps", eps)
    experiment.log_metric("nb_iter", nb_iter)

            
    return top1_fgm.avg, top1_pgd.avg, losses_fgm.avg, losses_pgd.avg, batch_time.sum

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


# def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
#     filepath = os.path.join(checkpoint, filename)
#     torch.save(state, filepath)
#     if is_best:
#         shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
        
def save_checkpoint(net, state, is_best,  checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    if is_best:
        torch.save(net, os.path.join(checkpoint, 'model_best.pth.tar'))
        torch.save(state, os.path.join(checkpoint, 'model_state_best.pth.tar'))
        
        # shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
        print('best checkpoint saved.')

def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True



def noise_rob(net, loader, noise_type, noise_coff):

    print('==> Start Robust Evaluation')
    net.eval()
    with torch.no_grad():
        losses_pertur = AverageMeter('Loss', ':.4e')
        top1_pertur = AverageMeter('Acc@1', ':6.2f')
        top5_pertur = AverageMeter('Acc@5', ':6.2f')
        for batch_idx, (inputs, targets) in enumerate(loader):

            inputs_pertur, targets = inputs.to(device), targets.to(device)
            # print('Before UnNormalize, input.max(),input.min()', inputs_pertur.max(),inputs_pertur.min())
            if args.dataset == 'cifar10':
                # print('==> UnNormalize CIFAR10')
                inputs_pertur = UnNormalize(
                    inputs_pertur, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            elif args.dataset == 'cifar100':
                inputs_pertur = UnNormalize(
                    inputs_pertur, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            elif args.dataset == 'mnist':
                    inputs_pertur = UnNormalize(
                    inputs_pertur,  (0.1307,), (0.3081,))
            # print('After UnNormalize, input.max(),input.min()', inputs_pertur.max(),inputs_pertur.min())
            if noise_type == 'randn':
                # print('noise_coff, inputs_pertur.size()',noise_coff, inputs_pertur.size())
                inputs_pertur = inputs_pertur + noise_coff * \
                    torch.autograd.Variable(torch.randn(inputs_pertur.size()).cuda(), requires_grad=False)

            # 均匀分布
            elif noise_type == 'rand':
                inputs_pertur = inputs_pertur + noise_coff * torch.autograd.Variable(torch.rand(
                    inputs_pertur.size()).cuda(), requires_grad=False)
            # 常数
            elif noise_type == 'const':
                inputs_pertur = inputs_pertur + noise_coff 

            # 截断
            inputs_pertur = F.relu(
                F.relu(inputs_pertur.mul_(-1).add_(1)).mul_(-1).add_(1))
            # print('Before Normalize, input.max(),input.min()', inputs_pertur.max(),inputs_pertur.min())
            if args.dataset == 'cifar10':
                inputs_pertur = Normalize(
                    inputs_pertur, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            elif args.dataset == 'cifar100':
                inputs_pertur = Normalize(
                    inputs_pertur, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            elif args.dataset == 'mnist':
                    inputs_pertur = Normalize(
                    inputs_pertur,  (0.1307,), (0.3081,))
            # print('After Normalize, input.max(),input.min()', inputs_pertur.max(),inputs_pertur.min())
            outputs_pertur = net(inputs_pertur)
            loss_pertur = criterion(outputs_pertur, targets)
            acc1_pertur, acc5_pertur = accuracy(
                outputs_pertur, targets, topk=(1, 5))
            losses_pertur.update(loss_pertur.item(), inputs_pertur.size(0))
            top1_pertur.update(acc1_pertur[0], inputs_pertur.size(0))
            top5_pertur.update(acc5_pertur[0], inputs_pertur.size(0))
        # print('top1_pertur.avg', top1_pertur.avg)
        print(noise_type+str(noise_coff)+'loss_pertur: {:.4f}, Accuracy_pertur: {:.4f}'.format(
            losses_pertur.avg, top1_pertur.avg))
        # experiment.log_metric("Test_pertur/Average loss", losses_pertur.avg, step=1)
        # experiment.log_metric("Test_pertur/Acc-top1/"+noise_type, top1_pertur.avg, step=1)
        
        return top1_pertur.avg
    

def log_feature(net, loader, noise_type, noise_coff):

    print('==> Start log_feature')
    net.eval()
    # seed_torch()
    with torch.no_grad():

        for batch_idx, (inputs, targets) in enumerate(loader):
            if batch_idx != 0: 
                break
            print('inputs.shape',inputs.shape)

            inputs_pertur, targets = inputs[0].to(device), targets[0].to(device)
            inputs_pertur = torch.reshape(inputs_pertur,(1,3, 32, 32))
            
            print('inputs_pertur.shape', inputs_pertur.shape)

            # inputs_pertur = inputs_pertur[0]
            # targets = targets[0]
            if args.dataset == 'cifar10':
                # print('==> UnNormalize CIFAR10')
                inputs_pertur = UnNormalize(
                    inputs_pertur, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            elif args.dataset == 'cifar100':
                inputs_pertur = UnNormalize(
                    inputs_pertur, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            elif args.dataset == 'mnist':
                    inputs_pertur = UnNormalize(
                    inputs_pertur,  (0.1307,), (0.3081,))                
            # if noise_type == 'randn':
            #     # print('noise_coff, inputs_pertur.size()',noise_coff, inputs_pertur.size())
            #     inputs_pertur = inputs_pertur + noise_coff * \
            #         torch.autograd.Variable(torch.randn(inputs_pertur.size()).cuda(), requires_grad=False)

            # # 均匀分布
            # elif noise_type == 'rand':
            #     inputs_pertur = inputs_pertur + noise_coff * torch.autograd.Variable(torch.rand(
            #         inputs_pertur.size()).cuda(), requires_grad=False)
            # 常数
            # elif noise_type == 'const':
            inputs_pertur = inputs_pertur + noise_coff 

            # 截断
            inputs_pertur = F.relu(
                F.relu(inputs_pertur.mul_(-1).add_(1)).mul_(-1).add_(1))

            if args.dataset == 'cifar10':
                inputs_pertur = Normalize(
                    inputs_pertur, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            elif args.dataset == 'cifar100':
                inputs_pertur = Normalize(
                    inputs_pertur, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            elif args.dataset == 'mnist':
                    inputs_pertur = Normalize(
                    inputs_pertur,  (0.1307,), (0.3081,))                

            for name, m in net.named_modules():
                # if not isinstance(m, torch.nn.ModuleList) and \
                #         not isinstance(m, torch.nn.Sequential) and \
                #         type(m) in torch.nn.__dict__.values():
                # 这里只对卷积层的feature map进行显示
                if isinstance(m, torch.nn.Conv2d):
                    m.register_forward_pre_hook(print_feature)

            outputs_pertur = net(inputs_pertur)

            # loss_pertur = criterion(outputs_pertur, targets)
            # acc1_pertur, acc5_pertur = accuracy(
            #     outputs_pertur, targets, topk=(1, 5))
            # losses_pertur.update(loss_pertur.item(), inputs_pertur.size(0))
            # top1_pertur.update(acc1_pertur[0], inputs_pertur.size(0))
            # top5_pertur.update(acc5_pertur[0], inputs_pertur.size(0))
        # print('top1_pertur.avg', top1_pertur.avg)
        # print(noise_type+str(noise_coff)+'loss_pertur: {:.4f}, Accuracy_pertur: {:.4f}'.format(
        #     losses_pertur.avg, top1_pertur.avg))
        # experiment.log_metric("Test_pertur/Average loss", losses_pertur.avg, step=1)
        # experiment.log_metric("Test_pertur/Acc-top1/"+noise_type, top1_pertur.avg, step=1)
        
        return outputs_pertur

def print_feature(module, input):
    x = input[0][0]
    print('x.mean()', x.mean())
    #最多显示4张图
    # min_num = np.minimum(4, x.size()[0])
    # for i in range(min_num):
    #     plt.subplot(1, 4, i+1)
    #     plt.imshow(x[i].cpu())
    # plt.show()
    
def target_transform(target):
    return int(target) - 1

# class Hook():
#     def __init__(self, module, backward=False):
#         if backward==False:
#             self.hook = module.register_forward_hook(self.hook_fn)
#         else:
#             self.hook = module.register_backward_hook(self.hook_fn)
#     def hook_fn(self, module, input, output):
#         self.input = input
#         self.output = output
#     def close(self):
#         self.hook.remove()
        
        
class Hooks():
    def __init__(self, layer):
        self.model  = None
        self.input  = None
        self.output = None
        self.grad_input  = None
        self.grad_output = None
        self.forward_hook  = layer.register_forward_hook(self.hook_fn_act)
        self.backward_hook = layer.register_full_backward_hook(self.hook_fn_grad)
    def hook_fn_act(self, module, input, output):
        self.model  = module
        self.input  = input[0]
        self.output = output
    def hook_fn_grad(self, module, grad_input, grad_output):
        self.grad_input  = grad_input[0]
        self.grad_output = grad_output[0]
    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()
        
    
class Cutout(object):
    def __init__(self, size=16, p=0.5):
        self.size = size
        self.p = p

    def __call__(self, image):
        if torch.rand([1]).item() > self.p:
            return image
        
        h, w = image.size(1), image.size(2)
        mask = np.ones((h,w), np.float32)

        x = np.random.randint(w)
        y = np.random.randint(h)

        y1 = np.clip(y - self.size // 2, 0, h)
        y2 = np.clip(y + self.size // 2, 0, h)
        x1 = np.clip(x - self.size // 2, 0, w)
        x2 = np.clip(x + self.size // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(image)
        image = image * mask
        return image
            
if __name__ == '__main__':

    # if 1:
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    
    best_acc = 0
    start_epoch = 0
    if not os.path.isdir(args.save_path) and args.local_rank == 0:
        mkdir_p(args.save_path)

    if args.dataset == 'cifar10':
        print('==> Preparing cifar10 data..')
        if 'cutout' in args.data_aug:
            transform_train = transforms.Compose([
            transforms.RandomCrop(size=(32, 32), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            Cutout(size=16, p=0.5),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # trainset = torchvision.datasets.CIFAR10(
        #     root= args.data +  '/cifar10', train=False, download=True, transform=transform_train)
        
        trainset = torchvision.datasets.CIFAR10(
            root= args.data +  '/cifar10', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True)

        testset = torchvision.datasets.CIFAR10(
            root= args.data + '/cifar10', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)

        # classes = ('plane', 'car', 'bird', 'cat', 'deer',
        #            'dog', 'frog', 'horse', 'ship', 'truck')

    elif args.dataset == 'cifar100':
        print('==> Preparing cifar100 data..')
        
        if 'cutout' in args.data_aug:
            transform_train = transforms.Compose([
            transforms.RandomCrop(size=(32, 32), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            Cutout(size=16, p=0.5),
            ])
        else:        
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        trainset = torchvision.datasets.CIFAR100(
            root=args.data +'/cifar100', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True)

        testset = torchvision.datasets.CIFAR100(
            root=args.data +'/cifar100', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)

    #        classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #                   'dog', 'frog', 'horse', 'ship', 'truck')
    elif args.dataset == 'mnist':
        trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(args.data +'/mnist', train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
        batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True)
        testloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(args.data +'/mnist',train=False,download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
        batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True) 
        
    elif args.dataset == 'svhn':
        if 'cutout' in args.data_aug:
            transform_train = transforms.Compose([
            transforms.RandomCrop(size=(32, 32), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            Cutout(size=16, p=0.5),
            ])
        else:
            transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        trainloader = torch.utils.data.DataLoader(
            torchvision.datasets.SVHN(
                root=args.data +'/svhn', split='train', download=True,
                transform=transform_train,
                # target_transform=target_transform,
            ),
            batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True) 
        testloader = torch.utils.data.DataLoader(
            torchvision.datasets.SVHN(
                root=args.data +'/svhn', split='test', download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
                # target_transform=target_transform
            ),
            batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True) 


        
    print('==> Building model..')
    net_name = args.arch
    model_name = args.arch
    # net = eval(args.arch)()
    # if 'convnext' in args.arch: 
    #     net = create_model(
    #     args.arch,
    #     pretrained=args.pretrained,
    #     num_classes=args.num_classes,
    #     drop_rate=args.drop,
    #     drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
    #     drop_path_rate=args.drop_path,
    #     drop_block_rate=args.drop_block,
    #     global_pool=args.gp,
    #     bn_momentum=args.bn_momentum,
    #     bn_eps=args.bn_eps,
    #     scriptable=args.torchscript,
    #     checkpoint_path=args.initial_checkpoint,
    #     givenA = args.givenA,
    #     givenB = args.givenB,
    #     Settings=args.Settings,
    #     PL=args.PL, 
    #     ini_stepsize=args.ini_stepsize,
    #     ini_block_shift=args.ini_block_shift,
    #     )
    # else:
    net = eval(args.arch)(num_classes=args.num_classes, givenA=args.givenA, givenB=args.givenB, PL=args.PL, ini_stepsize=args.ini_stepsize, Settings=args.Settings,ini_block_shift=args.ini_block_shift)
    # print('net',net)
    # tw.draw_model(net, [1, 3, 32, 32])
    net = net.to(device)
    # net.apply(weights_init) 


    total_params = sum(p.numel() for p in net.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in net.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    if 'mnist' in args.dataset:
        d = torch.rand(1, 1, 28, 28).to(device)
    elif 'cifar' in args.dataset or args.dataset == "svhn":
        d = torch.rand(1, 3, 32, 32).to(device)
    elif args.dataset == "stl10":
        d = torch.rand(1, 3, 96, 96).to(device)
    # elif args.dataset == "imagenet":
    # d = torch.rand(1, 3, 224, 224).to(device)        
    # print('d.device, net.device',d.device,net.device)
    summary(net, d)
    
    # torch.save(net, 'convnext.pth')
    # netron.start( 'convnext.pth')
    # flops, params = profile(net,  inputs=(d, ))
    trainable_params = 0
    for name, p in net.named_parameters():
        if p.requires_grad: 
            # print(name)
            if name:
                if 'Fix' not in name: # and 'bn3' not in name:
                    trainable_params +=p.numel()
                # else:
                #     print('name', name)
            else:
                trainable_params +=p.numel()
    all_params = 0
    all_trainable_params= 0 
    for p in net.parameters():
        all_params +=p.numel()
    for p in net.parameters():
        if p.requires_grad:         
            all_trainable_params +=p.numel()        
    # print('flops, params, trainable_params,all_params,all_trainable_params', flops, params, trainable_params,all_params,all_trainable_params)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True


        
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # optimizer = torch.optim.SGD([{'params':[ param for name, param in net.named_parameters() if ('Fix' not in name and 'coes' not in name)]}, 
    # {'params': (p for name, p in net.named_parameters() if 'coes' in name), 'weight_decay': 0.}
    # ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
#TODO: named_parameters may lose some unnamed parameters
    # optimizer_sgd = torch.optim.SGD([{'params':[ param for name, param in net.named_parameters() if ('Fix' not in name and 'coes' not in name)]},#, 'lr': 0.}, 
    # {'params': (p for name, p in net.named_parameters() if 'coes' in name), 'weight_decay': 0.,'momentum':0.,}
    # ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)




    if args.minimizer in ['ASAM', 'SAM', 'coesASAM', 'coesSAM']:
        # optimizer = torch.optim.SGD([{'params':[ param for name, param in net.named_parameters() if ('Fix' not in name)]},#, 'lr': 0.}, 
        # ], lr=0.1, momentum=args.momentum, weight_decay=args.weight_decay)
        
        optimizer = torch.optim.SGD([{'params': (p for name, p in net.named_parameters() if 'coes' in name), 'weight_decay': 0.,}, {'params':[ param for name, param in net.named_parameters() if ('Fix' not in name and 'coes' not in name)]} ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)     
           
        minimizer = eval(args.minimizer)(optimizer, net, rho=args.rho, eta=args.eta)
        if args.sche == 'cos':
            train_scheduler_sgd = optim.lr_scheduler.CosineAnnealingLR(minimizer.optimizer, args.epoch)
        elif args.sche == 'step':
            if args.dataset == 'cifar100':
                train_scheduler_sgd = optim.lr_scheduler.MultiStepLR(minimizer.optimizer, milestones=[150, 225], gamma=0.1)
            elif args.dataset == 'cifar10' or args.dataset == 'mnist':
                train_scheduler_sgd = optim.lr_scheduler.MultiStepLR(minimizer.optimizer, milestones=[80, 120], gamma=0.1)
    else:
        if 'convnext' in args.arch:
            optimizer_sgd = torch.optim.AdamW([{'params':[ param for name, param in net.named_parameters() if ('Fix' not in name and 'coes' not in name)]},#, 'lr': 0.}, 
        ], lr=args.lr, weight_decay=0.05,eps=5e-9)
        else:
            optimizer_sgd = torch.optim.SGD([{'params':[ param for name, param in net.named_parameters() if ('Fix' not in name and 'coes' not in name)]},#, 'lr': 0.}, 
            ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        optimizer_adam = torch.optim.Adam([
        {'params': (p for name, p in net.named_parameters() if 'coes' in name and 'Bal' not in name), 'weight_decay': 0.,},
        ], lr=args.lr*0.1, weight_decay=args.weight_decay)
            
        if args.sche == 'cos':
            train_scheduler_sgd = optim.lr_scheduler.CosineAnnealingLR(optimizer_sgd, args.epoch)
            train_scheduler_adam = optim.lr_scheduler.CosineAnnealingLR(optimizer_adam, args.epoch)            
        elif args.sche == 'step':
            if args.dataset == 'cifar100':
                train_scheduler_sgd = optim.lr_scheduler.MultiStepLR(optimizer_sgd, milestones=[150, 225], gamma=0.1)
                train_scheduler_adam = optim.lr_scheduler.MultiStepLR(optimizer_adam, milestones=[150, 225], gamma=0.1)                
            elif args.dataset == 'cifar10':# or args.dataset == 'mnist':
                train_scheduler_sgd = optim.lr_scheduler.MultiStepLR(optimizer_sgd, milestones=[80, 120], gamma=0.1)
                train_scheduler_adam = optim.lr_scheduler.MultiStepLR(optimizer_adam, milestones=[80, 120], gamma=0.1)                
            elif args.dataset == 'svhn' or args.dataset == 'mnist':
                train_scheduler_sgd = optim.lr_scheduler.MultiStepLR(optimizer_sgd, milestones=[20, 30], gamma=0.1)
                train_scheduler_adam = optim.lr_scheduler.MultiStepLR(optimizer_adam, milestones=[20, 30], gamma=0.1)                
                
    iter_per_epoch = len(trainloader)
    if args.minimizer in ['ASAM', 'SAM', 'coesASAM', 'coesSAM']:
        warmup_scheduler = WarmUpLR(minimizer.optimizer, iter_per_epoch * args.warm)
    else:
        warmup_scheduler = WarmUpLR(optimizer_sgd, iter_per_epoch * args.warm)

    
    # criterion = nn.CrossEntropyLoss()
    if args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        criterion2 = torch.nn.MSELoss()
    # optionally resume from a checkpoint
    title = 'CIFAR-' + args.arch
    args.lastepoch = -1
    if args.resume:
        if os.path.isfile(args.checkpoint):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.checkpoint)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer_sgd.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            args.lastepoch = checkpoint['epoch']

    train_time = 0.0
    test_time = 0.0
    train_top1_acc = 0.0
    train_min_loss = 100
    test_top1_acc = 0.0
    test_top1_acc_fgm = 0.0
    test_top1_acc_pgd = 0.0
    
    test_min_loss = 100
    best_prec1 = -1
    # lr_list = []

    # writer = SummaryWriter(log_dir=args.save_path)
    # plot_feature(net, 0)

    # features_in = []
    # features_out = []
    # def hook(module, input, output):
    #     features_in.append(input)
    #     features_out.append(output)
    #     return None
    # # for (name, module) in net.named_modules():
    # for module in net.modules():
    #     # print('name,module',name,module)
    #     # print('name',name)
    #     # if name is not None:
    #     # if isinstance(module, torch.nn.ReLU):#isinstance(module, torch.nn.Identity):        
    #     handle = module.register_forward_hook(hook)
    #     handle.remove()

    # hookF = [Hook(layer[1]) for layer in list(net._modules.items())]   
    
    # hookF_back = [Hooks(layer[1]) for layer in list(net._modules.items())]     
    # top1_pertur_test = noise_rob(net, testloader, 'randn', 0.1)
    # print('top1_pertur_test',top1_pertur_test)
    # for epoch in range(1, args.epoch):
    
    # args.epoch = 2
    # with torch.profiler.profile(schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1), 
    #                         on_trace_ready=torch.profiler.tensorboard_trace_handler('/media3/clm/SuppressionConnection/logdir'), 
    #                         record_shapes=True, 
    #                         with_stack=True) as prof:
    for epoch in range(1, args.epoch+1):

        # print('start time: ', "{0:%Y-%m-%dT%H-%M/}".format(datetime.now()))
        train_acc_epoch, train_loss_epoch, train_epoch_time = train(epoch)
        

        # if epoch%10==1 or epoch>= (0.95*args.epoch+1):
            
        #     plot_feature(net, epoch)
        #     # net.eval()
        #     # x_test, y_test = load_cifar10(n_examples=50)
        #     # fmodel = fb.PyTorchModel(net, bounds=(0, 1))
        #     # _, advs, success = fb.attacks.LinfPGD()(fmodel, x_test.to(device), y_test.to(device), epsilons=[8/255])
        #     # print('Robust accuracy: {:.1%}'.format(1 - success.float().mean()))            
        # # print('end time: ', "{0:%Y-%m-%dT%H-%M/}".format(datetime.now()))
        train_top1_acc = max(train_top1_acc, train_acc_epoch)
        train_min_loss = min(train_min_loss, train_loss_epoch)
        train_time += train_epoch_time
        acc, test_loss_epoch, test_epoch_time = test(epoch)
        test_top1_acc = max(test_top1_acc, acc)
        # test_top1_acc_pgd = max(test_top1_acc_pgd, acc_pgd)
        # test_top1_acc_fgm = max(test_top1_acc_fgm, acc_fgm)
        # print('optimizer_sgd_lr',optimizer_sgd.state_dict()['param_groups'][0]['lr'])
        # print('optimizer_adam_lr',optimizer_adam.state_dict()['param_groups'][0]['lr'])
        test_min_loss = min(test_min_loss, test_loss_epoch)
        test_time += test_epoch_time

        if args.local_rank == 0:

            is_best = test_top1_acc > best_prec1
            best_prec1 = max(test_top1_acc, best_prec1)
            
            # if epoch > args.epoch*0.8:
            #     save_checkpoint(net, {
            #         'epoch': epoch + 1,
            #         'arch': args.arch,
            #         'state_dict': net.state_dict(),
            #         'best_prec1': best_prec1,
            #         'optimizer': optimizer.state_dict(),
            #     }, is_best, checkpoint=args.save_path)

           

    # writer.close()
    end_train = train_time // 60
    end_test = test_time // 60
    experiment.log_metric("test_top1_best", test_top1_acc)
    
    print(model_name)
    print("train time: {}D {}H {}M".format(end_train // 1440, (end_train % 1440) // 60, end_train % 60))
    print("test time: {}D {}H {}M".format(end_test // 1440, (end_test % 1440) // 60, end_test % 60))
    print(
        "train_acc_top1:{}, train_min_loss:{}, train_time:{}, test_top1_acc:{}, test_min_loss:{}, test_time:{}".format(
            train_top1_acc, train_min_loss, train_time, test_top1_acc, test_min_loss, test_time))
    print("args.save_path:", args.save_path)
    
    
    givenA_txt = ''
    givenB_txt = ''
    givenA_list = []
    givenB_list = []
    for i in args.givenA:
        givenA_txt += str(i)+'_'
        givenA_list += str(i)
    print('args.givenA', args.givenA)    
    print('givenA_txt', givenA_txt)
    print('givenA_list', givenA_list)

    for i in args.givenB:
        givenB_txt += str(i)+'_'        
        givenB_list += str(i)
    ConverOrder = args.ConverOrd
    if args.adv_train:
        head_list = ['Model',  "Step", "Order", "Alphas", "Betas", 'Noise Type', 'Noise Value', 'Adv. Train Acc.', 'Adv. Test Acc.']#"FLOPs","# Params", "# Trainable Params", 
    else:
        head_list = ['Model',"Step", "Order", "Alphas", "Betas",'Noise Type', 'Noise Value', 'Train Acc.', 'Test Acc.']#"FLOPs","# Params", "# Trainable Params", 
    df = pd.DataFrame(columns=head_list)
    
    # net = torch.load(os.path.join(args.save_path, 'model_state_best.pth.tar')) 
    noise_dict = {'const': [0,0.01,0.05,0.1, 0.2, 0.3, 0.4, 0.45, -0.01, -0.05,-0.1,-0.2, -0.3, -0.4, -0.45],
            'randn': [0.01,  0.02, 0.03, 0.04, 0.05, 0.1, 0.2],
            'rand': [0.04, 0.06, 0.08, 0.1, 0.12, 0.2, -0.04, -0.06, -0.08, -0.1, -0.12,-0.2],
            }
    # exps = ["0", "1", "2"]
    eps_list = [3.0/255, 5.0/255, 8.0/255] #  16.0/255, 32.0/255]
    eps_iter = args.eps_iter
    nb_iter = args.nb_iter
    norm = args.norm        
    

       
  
       
    for noise_type in noise_dict.keys():  
        Noise_Coffs = []
        Top1_pertur_tests = []
        Top1_pertur_trains = []
        for noise_coff in noise_dict.get(noise_type):  

            top1_pertur_test = noise_rob(net, testloader, noise_type, noise_coff)
            Noise_Coffs += [noise_coff]
            Top1_pertur_tests += [top1_pertur_test.item()]
            df_row = pd.DataFrame([[args.arch, args.steps, ConverOrder, givenA_txt, givenB_txt, noise_type, noise_coff, 0, top1_pertur_test.item() ]], columns=head_list)

            df = df.append(df_row)
        print(noise_type+'Noise_Coffs, Top1_pertur_tests',Noise_Coffs, Top1_pertur_tests)

    print('Table \n',df)
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = args.arch+args.notes+'.csv'
    df.to_csv(save_path+file_name)
    experiment.log_table(save_path+file_name)    
experiment.end()

