from comet_ml import Experiment, OfflineExperiment

from absl import app, flags
from easydict import EasyDict
import numpy as np
import random

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)


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

# from tensorboardX import SummaryWriter

from models import *
from noise import *

from datetime import datetime
import errno
import shutil
import pandas as pd

import homura
from torchvision import transforms
from timm.loss import LabelSmoothingCrossEntropy
from asam import ASAM, SAM
from bypass_bn import enable_running_stats, disable_running_stats
from torchsummaryX import summary
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from MISGD import *
parser = argparse.ArgumentParser(description='ZeorSNet CIFAR')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--epoch', type=int, default=None, help='training epoch')
parser.add_argument('--warm', type=int, default=3, help='warm up training phase')
parser.add_argument('--data', default='/media/bdc/clm/OverThreeOrders/CIFAR/data', type=str)# /media_HDD_1/lab415/clm/OverThreeOrders/OverThreeOrders/CIFAR/data
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--arch', '-a', default='ZeroSNet20_Opt', type=str)
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
parser.add_argument("--ex", default=0, type=int)
parser.add_argument("--notes", default='', type=str)
parser.add_argument('--PL', type=float, default=1.0)
parser.add_argument('--sche', default='step', type=str)
# parser.add_argument('--coe_ini', type=float, default=1)
parser.add_argument('--share_coe', type=bool, default=False)
# parser.add_argument('--given_coe', default=[1.0 / 3, 5.0 / 9, 1.0 / 9, 16.0 / 9], nargs='+', type=float)
parser.add_argument('--given_coe', default=None, nargs='+', type=float)
parser.add_argument('--steps', type=int, default=3)
parser.add_argument('--ini_stepsize', default=1, type=float)
parser.add_argument('--givenA', default=None, nargs='+', type=float)
parser.add_argument('--givenB', default=None, nargs='+', type=float)
parser.add_argument("--ConverOrd", default=4, type=int, help="")

parser.add_argument("--minimizer", default='ASAM', type=str, help="ASAM or SAM.")
parser.add_argument("--smoothing", default=None, type=float, help="Label smoothing.")#0.1
parser.add_argument("--rho", default=0.5, type=float, help="Rho for ASAM.")
parser.add_argument("--eta", default=0.01, type=float, help="Eta for ASAM.")
    
parser.add_argument("--adv_train", action='store_true')
parser.add_argument("--eps", default=0.1, type=float, help="")
parser.add_argument("--eps_iter", default=0.01, type=float, help="step size for each attack iteration")
parser.add_argument("--nb_iter", default=40, type=int, help="Number of attack iterations.")
parser.add_argument("--norm", default=np.inf, type=float, help="Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.")
parser.add_argument("--clip_min", default=None, type=float, help="Minimum float value for adversarial example components.")
parser.add_argument("--clip_max", default=None, type=float, help="Maximum float value for adversarial example components.")
parser.add_argument("--save_path", default=None, type=str, help="save path")
parser.add_argument("--ablation", default='', type=str, help="ablation")

# TODO: print inputs
    
args = parser.parse_args()
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
      
# args.save_path = path_base + args.arch + '/PL' + str(args.PL) + 'ini_step' + str(
#     args.ini_stepsize)  + givenA_text + givenB_text +'_sche_' + args.sche + str(args.opt) + \
#                  '_mini'+args.minimizer+'_BS' + str(args.bs) + '_LR' + \
#                  str(args.lr) + 'epoch' + \
#                  str(args.epoch) + 'warm' + str(args.warm) + \
#                  args.notes + \
#                      '_Ch4_'+\
#                  "{0:%Y-%m-%dT:%H:%M:%S/}".format(datetime.now())


    
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
    api_key="KbJPNIfbsNUoZJGJBSX4BofNZ",
    # project_name="OverThreeOrders",
    project_name="MISGD_CIFAR",
    # project_name="overthreeorders-4channels",
    workspace="logichen",
    # auto_histogram_weight_logging=True,
    # offline_directory=path_base+"CometData",
)

# checkpoint
if args.checkpoint is None:
    args.checkpoint = args.save_path+'checkpoint.pth.tar'
    print('args.checkpoint', args.checkpoint)
# print('givenB', args.givenB, 'givenA', args.givenA)
hyper_params = {
    'epoch': args.epoch,
    "learning_rate": args.lr,
    'warmup': args.warm,
    'dataset': args.dataset,
    'arch': args.arch,
    "batch_size": args.bs, 
    'momentum': args.momentum,
    'wd': args.weight_decay,
    'opt': args.opt,
    # 'PL': args.PL,
    'sche': args.sche,
    # 'coe_ini': args.coe_ini,
    # 'share_coe': args.share_coe,
    'ini_stepsize': args.ini_stepsize,
    'givenA': args.givenA,
    'givenB': args.givenB,
    # 'steps': args.steps,
    'notes': args.notes,
    'minimizer': args.minimizer,
    'smoothing': args.smoothing,
    'rho': args.rho,
    'eta': args.eta,
    'adv_train': args.adv_train,
    'ablation': args.ablation,
    }
experiment.log_parameters(hyper_params)


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
        optimizer.zero_grad()
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
            train_scheduler.step()#epoch)
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
        
            # criterion(outputs, targets).backward()#asam
            criterion(outputs, targets).mean().backward()#asam
            
            minimizer.descent_step()#asam
            
            with torch.no_grad():
                loss += batch_loss.sum().item()
                acc += (torch.argmax(outputs, 1) == targets).sum().item()
            cnt += len(targets)
        
            if epoch >= args.warm:
                train_scheduler.step(epoch)#epoch)
            if epoch < args.warm:
                warmup_scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()
        else:
            optimizer.zero_grad()
            # if 'zero' in args.arch or 'ZeroS' in args.arch:
            #     outputs, coes, stepsize = net(inputs)
            # elif 'MRes' in args.arch:
            #     outputs, coes = net(inputs)
            # else:
            #     outputs = net(inputs)
            outputs = net(inputs)

            loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            loss.backward()
            optimizer.step()

            if epoch > args.warm:
                train_scheduler.step(epoch)
            if epoch <= args.warm:
                warmup_scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()   
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            # print('acc1', acc1, 'acc5', acc5)
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0)) 
            # if args.ablation:    
            #     for name, param in net.named_parameters():
            #         if 'Learn' in name or 'Fix' in name:
            #             print(name, param.data.mean())                    
    # loss /= cnt
    # acc *= 100. / cnt
    # print(f"Epoch: {epoch}, Train accuracy: {acc:6.2f} %, Train loss: {loss:8.5f}")        
    # experiment.log_metric("Train/Average loss", loss, step=epoch)
    # experiment.log_metric("Train/Accuracy-top1", acc, step=epoch)
    # experiment.log_metric("Train/Time", batch_time.sum, step=epoch)
    print('Train set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(losses.avg, top1.avg))
       
    # return acc, loss, batch_time.sum        
    
    # print('Epoch: {:.1f}, Train set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(epoch, losses.avg, top1.avg))
    # writer.add_scalar('Train/Average loss', losses.avg, epoch)
    # writer.add_scalar('Train/Accuracy-top1', top1.avg, epoch)
    # writer.add_scalar('Train/Accuracy-top5', top5.avg, epoch)
    # writer.add_scalar('Train/Time', batch_time.sum, epoch)

    experiment.log_metric("Train/Average loss", losses.avg, step=epoch)
    experiment.log_metric("Train/Accuracy-top1", top1.avg, step=epoch)
    experiment.log_metric("Train/Accuracy-top5", top5.avg, step=epoch)
    experiment.log_metric("Train/Time", batch_time.sum, step=epoch)

    coes_print = []
    # if 'zeros' in args.arch or 'ZeroS' in args.arch:
    #     if not isinstance(stepsize, int):
    #         stepsize = stepsize.data.cpu().numpy()
    #     # writer.add_scalar('stepsize', float(stepsize), epoch)
    #     experiment.log_metric("stepsize", float(stepsize), step=epoch)
    #     if not isinstance(coes, int): 
    #         if isinstance(coes, float):
    #             # writer.add_scalar('coes', coes, epoch)
    #             experiment.log_metric("coes", coes, step=epoch)
    #         else:
    #             for i in range(len(coes)):
    #                 if isinstance(coes, torch.nn.ParameterList):
    #                     # print('coes0:',coes[0].data)
    #                     # print('coes.parameters()[i]', coes.parameters()[i])
    #                     coes_print.append(float(coes[i].data.cpu().numpy()))
    #                 if not isinstance(coes[i], float):
    #                     coes_print[i] = float(coes[i].data.cpu().numpy())
    #                 # writer.add_scalar('coes_' + str(i), coes[i], epoch)
    #                 experiment.log_metric("coes_" + str(i), coes_print[i], step=epoch)

    return top1.avg, losses.avg, batch_time.sum

def test(epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # losses_fgm = AverageMeter('Loss_fgm', ':.4e')
    # top1_fgm = AverageMeter('fgmAcc@1', ':6.2f')
    # top5_fgm = AverageMeter('fgmAcc@5', ':6.2f')
    
    # losses_pgd = AverageMeter('pgdLoss_pgd', ':.4e')
    # top1_pgd = AverageMeter('pgdAcc@1', ':6.2f')
    # top5_pgd = AverageMeter('pgdAcc@5', ':6.2f')        
    end = time.time()
    net.eval()
    # loss = 0.
    # acc = 0.
    # cnt = 0.
    # with torch.no_grad():
    if 1:
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
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
    # print('Test set: Average loss_pgd: {:.4f}, Accuracy_pgd: {:.4f}'.format(losses_pgd.avg, top1_pgd.avg))

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

    # experiment.log_metric("Test/Average loss_pgd", losses_pgd.avg, step=epoch)
    # experiment.log_metric("Test/Accuracy-top1_pgd", top1_pgd.avg, step=epoch)
    # experiment.log_metric("Test/Accuracy-top5_pgd", top5_pgd.avg, step=epoch)
    
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
            inputs_fgm = F.relu(
                F.relu(inputs_fgm.mul_(-1).add_(1)).mul_(-1).add_(1))      
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

            if args.dataset == 'cifar10':
                inputs_pertur = Normalize(
                    inputs_pertur, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            elif args.dataset == 'cifar100':
                inputs_pertur = Normalize(
                    inputs_pertur, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            elif args.dataset == 'mnist':
                    inputs_pertur = Normalize(
                    inputs_pertur,  (0.1307,), (0.3081,))                
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
    seed_torch()
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

if __name__ == '__main__':

    # if 1:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0
    start_epoch = 0
    if not os.path.isdir(args.save_path) and args.local_rank == 0:
        mkdir_p(args.save_path)

    if args.dataset == 'cifar10':
        print('==> Preparing cifar10 data..')

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
        trainloader = torch.utils.data.DataLoader(
            torchvision.datasets.SVHN(
                root=args.data +'/svhn', split='train', download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
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
    net = eval(args.arch)(num_classes=args.num_classes, givenA=args.givenA, givenB=args.givenB, PL=args.PL, ini_stepsize=args.ini_stepsize, ablation=args.ablation)

    net = net.to(device)
    net.apply(weights_init) 
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
    # print('d.device, net.device',d.device,net.device)
    # summary(net, d)

    flops, params = profile(net,  inputs=(d, ))
    trainable_params = 0
    for name, p in net.named_parameters():
        if p.requires_grad: 
            if name:
                if 'Fix' not in name: # and 'bn3' not in name:
                    trainable_params +=p.numel()
                # else:
                #     print('name', name)
            else:
                trainable_params +=p.numel()
    print('flops, params, trainable_params', flops, params, trainable_params)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True


        
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.SGD([{'params':[ param for name, param in net.named_parameters() if 'Fix' not in name]}], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # optimizer = MISGD([{'params':[ param for name, param in net.named_parameters() if 'Fix' not in name]}], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)   

    if args.minimizer in ['ASAM', 'SAM']:
        minimizer = eval(args.minimizer)(optimizer, net, rho=args.rho, eta=args.eta)
        if args.sche == 'cos':
            train_scheduler = optim.lr_scheduler.CosineAnnealingLR(minimizer.optimizer, args.epoch)
        elif args.sche == 'step':
            if args.dataset == 'cifar100':
                train_scheduler = optim.lr_scheduler.MultiStepLR(minimizer.optimizer, milestones=[150, 225], gamma=0.1)
            elif args.dataset == 'cifar10' or args.dataset == 'mnist':
                train_scheduler = optim.lr_scheduler.MultiStepLR(minimizer.optimizer, milestones=[80, 120], gamma=0.1)
    else:
        if args.sche == 'cos':
            train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
        elif args.sche == 'step':
            if args.dataset == 'cifar100':
                train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 225], gamma=0.1)
            elif args.dataset == 'cifar10' or args.dataset == 'mnist':
                train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
            elif args.dataset == 'svhn':
                train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30], gamma=0.1)
                
    iter_per_epoch = len(trainloader)
    if args.minimizer in ['ASAM', 'SAM']:
        warmup_scheduler = WarmUpLR(minimizer.optimizer, iter_per_epoch * args.warm)
    else:
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    
    # criterion = nn.CrossEntropyLoss()
    if args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
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
            optimizer.load_state_dict(checkpoint['optimizer'])
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

    # for epoch in range(1, args.epoch):
    for epoch in range(1, args.epoch+1):

        # print('start time: ', "{0:%Y-%m-%dT%H-%M/}".format(datetime.now()))
        train_acc_epoch, train_loss_epoch, train_epoch_time = train(epoch)
        # print('end time: ', "{0:%Y-%m-%dT%H-%M/}".format(datetime.now()))
        train_top1_acc = max(train_top1_acc, train_acc_epoch)
        train_min_loss = min(train_min_loss, train_loss_epoch)
        train_time += train_epoch_time
        acc, test_loss_epoch, test_epoch_time = test(epoch)
        test_top1_acc = max(test_top1_acc, acc)
        # test_top1_acc_pgd = max(test_top1_acc_pgd, acc_pgd)
        # test_top1_acc_fgm = max(test_top1_acc_fgm, acc_fgm)
        
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
    

       
    # for noise_type in noise_dict.keys():  # 噪音类型
  
       
    for noise_type in noise_dict.keys():  # 噪音类型
        Noise_Coffs = []
        Top1_pertur_tests = []
        Top1_pertur_trains = []
        for noise_coff in noise_dict.get(noise_type):  # 噪音值
            # if noise_type == 'const':
            #     loss_per = log_feature(net, testloader, noise_type, noise_coff)
            #     print('loss_per',loss_per) 
            if noise_coff ==0 and noise_type =='const':
                    for eps in eps_list:  # 噪音值
                        print('eps, eps_iter, nb_iter, norm', eps, eps_iter, nb_iter, norm)

                        test_top1_fgm, test_top1_pgd, losses_fgm, losses_pgd, batch_time = eval_adv(net, testloader, eps, eps_iter, nb_iter, norm)
                        
                        # train_top1_fgm, train_top1_pgd, train_losses_fgm, train_losses_pgd, train_batch_time = eval_adv(net, trainloader, eps, eps_iter, nb_iter, norm)
                        print('test_top1_fgm, test_top1_pgd',test_top1_fgm.data, test_top1_pgd.data)
                        # print('train_top1_fgm, train_top1_pgd',train_top1_fgm, train_top1_pgd)    
                        df_row = pd.DataFrame([[args.arch, args.steps, ConverOrder, givenA_txt, givenB_txt, 'PGD', eps, 0, test_top1_pgd.item() ]], columns=head_list)
                        df = df.append(df_row)                        
                        df_row = pd.DataFrame([[args.arch, args.steps, ConverOrder, givenA_txt, givenB_txt, 'FGM', eps, 0, test_top1_fgm.item() ]], columns=head_list)                                             
                        # df_row = pd.DataFrame([[args.arch, args.steps, ConverOrder, givenA_txt, givenB_txt, 'PGD', eps, train_top1_pgd.item(), test_top1_pgd.item() ]], columns=head_list)
                        # df = df.append(df_row)                        
                        # df_row = pd.DataFrame([[args.arch, args.steps, ConverOrder, givenA_txt, givenB_txt, 'FGM', eps, train_top1_fgm.item(), test_top1_fgm.item() ]], columns=head_list)
                        df = df.append(df_row)
            top1_pertur_test = noise_rob(net, testloader, noise_type, noise_coff)
            # top1_pertur_train = noise_rob(net, trainloader, noise_type, noise_coff)
            Noise_Coffs += [noise_coff]
            Top1_pertur_tests += [top1_pertur_test.item()]
            # Top1_pertur_trains += [0]
            # Top1_pertur_trains += [top1_pertur_train.item()]
            df_row = pd.DataFrame([[args.arch, args.steps, ConverOrder, givenA_txt, givenB_txt, noise_type, noise_coff, 0, top1_pertur_test.item() ]], columns=head_list)

            # df_row = pd.DataFrame([[args.arch, args.steps, ConverOrder, givenA_txt, givenB_txt, noise_type, noise_coff, top1_pertur_train.item(), top1_pertur_test.item() ]], columns=head_list)
            df = df.append(df_row)
        # print(noise_type+'Noise_Coffs, Top1_pertur_trains',Noise_Coffs, Top1_pertur_trains)
        print(noise_type+'Noise_Coffs, Top1_pertur_tests',Noise_Coffs, Top1_pertur_tests)

        # experiment.log_curve('Train'+noise_type, x=Noise_Coffs, y=Top1_pertur_trains)
        # experiment.log_curve('Test'+noise_type, x=Noise_Coffs, y=Top1_pertur_tests)
        
    print('Table \n',df)
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = args.arch+args.notes+'.csv'
    df.to_csv(save_path+file_name)
    experiment.log_table(save_path+file_name)

    
    # print('top1_pertur_test', top1_pertur_test.data.cpu(), 'top1_pertur_train', top1_pertur_train.data.cpu().numpy())
    
    # ### add noise 
    # isTest = 0
    # if isTest:
    #     log_save_path_base = '/media/bdc/clm/OverThreeOrders/CIFAR/RobustPreTrain/robust_dychf/test/results_'
    # else:
    #     log_save_path_base = '/media/bdc/clm/OverThreeOrders/CIFAR/RobustPreTrain/robust_dychf/allmodel/results_'

    # model_name_cifar10 = args.arch
    # model_path = '/media/bdc/clm/OverThreeOrders/CIFAR/runs/cifar10/ZeroSNet/'+model_name_cifar10

    # py_flie_path = '/media/bdc/clm/OverThreeOrders/CIFAR/robustness_run_cifar10_lwd_chf.py'


    # def get_given_short(given):
    #     split = given.split('_')
    #     coe = ''
    #     for i in range(len(split)-1):
    #         coe  += split[i][:6]+'_'
    #     return coe

    # def get_notes(file_name):
    #     notes = file_name[file_name.find('warmup')+7: file_name.find('2022-')]
    #     return notes

    # for i in args.givenA:
    #     givenA = str(i)+' '
    # for i in args.givenB:
    #     givenB = str(i)+' '        

    # givenA_txt = givenA.replace(' ', '_')
    # givenB_txt = givenB.replace(' ', '_')

    # # exps = ["0"]
    # exps = ["0", "1", "2"]
    # noise_dict = {'randn': ['0.01'],#,  '0.02', '0.03', '0.035', '0.04'],
    #             'rand': ['0.04'],#, '0.06', '0.08', '0.1', '0.12', '-0.04', '-0.06', '-0.08', '-0.1', '-0.12'],
    #             'const': ['0.2'],# '0.3', '0.4', '0.45', '-0.2', '-0.3', '-0.4', '-0.45', ]
    #             }

    # gpu_flag = True
    # for noise_type in noise_dict.keys():  # 噪音类型
    #     log_save_path = log_save_path_base + noise_type+'/'
    #     if not os.path.exists(log_save_path):
    #         os.makedirs(log_save_path)

            
    #         if isTest: # and (not (a0 == '0.3333333')):
    #             continue

    #         for noise in noise_dict.get(noise_type):  # 噪音值
    #             first_thread = True
    #             for exp in exps:  # 试验次数
    #                 command = ''
    #                 gpu = str('0') if gpu_flag else str('1')
    #                 command += 'CUDA_VISIBLE_DEVICES=' + gpu
    #                 command += ' nohup'
    #                 command += ' python3 '+py_flie_path
    #                 command += ' --arch '+model_name_cifar10
    #                 command += " --noise_coff " + str(noise)
    #                 command += " --dataset cifar10 --opt SGD --resume True"
    #                 command += " --steps " + str(args.steps)
    #                 command += " --ini_stepsize " + str(args.ini_stepsize)
    #                 command += " --notes " + str(args.notes)
    #                 command += " --givenA " + givenA
    #                 command += " --givenB " + givenB
    #                 # command += " --k_ini " + str(args.k_ini) + " "
    #                 # command += ' --given_ks ' + given_ks
    #                 command += " --sche " + args.sche
    #                 command += " --epoch " + str(args.epoch)
    #                 command += " --warm " + str(args.warm)
    #                 # command += " --PL " + str(PL)
    #                 command += " --wd " + str(args.wd)
    #                 command += " --bs " + str(args.bs)
    #                 command += " --lr " + str(args.lr)
    #                 command += " --save_path " + args.save_path
    #                 command += " --workers 4 "
    #                 command += " --noise_type "+noise_type
                    
    #                 log_file_name = ''
    #                 log_file_name += model_name_cifar10
    #                 # log_file_name += '_'+given_ks_txt
    #                 log_file_name += "_noi_"+str(noise)
    #                 log_file_name += "_ord" + str(args.steps)
    #                 log_file_name += "_IniStep" + str(args.ini_stepsize)
    #                 log_file_name += "_notes" + str(args.notes)
    #                 log_file_name += "_A" + givenA_txt
    #                 log_file_name += "_B" + givenB_txt
    #                 # log_file_name += "_tag_" + tag
    #                 # log_file_name += "_k_ini_" + str(args.k_ini)
    #                 # log_file_name += "_PL" + str(args.PL)
    #                 # log_file_name += "_lr" + str(args.lr)
    #                 # log_file_name += "_wd" + str(args.wd)
    #                 log_file_name += "_sche_" + args.sche
    #                 log_file_name += "_ep" + str(args.epoch)
    #                 # log_file_name += "_SGD"
    #                 log_file_name += "_warm" + str(args.warm)
    #                 log_file_name += "_exp" + str(exp)

    #                 command += " > "+log_save_path+log_file_name+".txt 2>&1"
    #                 if not first_thread:
    #                     command += ' &'  # 不阻塞

    #                 print(command)
    #                 os.system(command)
    #                 first_thread = False

    #             gpu_flag = not gpu_flag
    #     log_save_path = ''

# if __name__ == '__main__':
#     d = torch.rand(2, 3, 32, 32)
#     # net = ZeroSNet20_rec()
#     # net = ZeroSNet164_Tra()
#     # net = ZeroSNet650_Tra()
#     net=ResNet_20()
#     # net = MResNet164()
#     # net = ResNet_164()
#     # net = ResNet_650()
#     o = net(d)
#     probs = F.softmax(o).detach().numpy()[0]
#     pred = np.argmax(probs)
#     print('pred, probs', pred, probs)
#
#     total_params = sum(p.numel() for p in net.parameters())
#     print(f'{total_params:,} total parameters.')
#     total_trainable_params = sum(
#         p.numel() for p in net.parameters() if p.requires_grad)
#     print(f'{total_trainable_params:,} training parameters.')
#     for name, parameters in net.named_parameters():
#         print(name, ':', parameters.size())
# #     onnx_path = "onnx_model_name.onnx"
# #     torch.onnx.export(net, d, onnx_path)
# #     #
# #     netron.start(onnx_path)
# # #
