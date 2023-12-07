import os
import numpy as np
import argparse
import os
import sys
import time
from datetime import datetime
import random
import errno
from random import randint

from sympy import sec 

def gpu_info(GpuNum):
    gpu_status = os.popen('nvidia-smi -i '+str(GpuNum)+' | grep %').read().split('|')
    # print('gpu_status', gpu_status)
    gpu_memory = int(gpu_status[2].split('/')[0].split('M')[0].strip())
    gpu_power = int(gpu_status[1].split(
        '   ')[-1].split('/')[0].split('W')[0].strip())
    gpu_utilization = int(gpu_status[3].split('%')[0].split()[-1])
    return gpu_power, gpu_memory, gpu_utilization

def get_A(B_ab):
    A_ab = np.array([0]*len(B_ab))
    A_ab[0] = A_ab[0]+1
    A_ab = ' '+str(A_ab).replace('[', ' ').replace(']', ' ')+' '
    return A_ab

b_ = np.array([4277, -7923, 9982,-7298, 2877,-475])
B_ab6 = -b_/1440
b_ = np.array([1901, -2774, 2616, -1274, 251])
B_ab5 = -b_/720
b_ = np.array([55,-59,37,-9])
B_ab4 = -b_/24
b_ = np.array([23,-16,5])
B_ab3 = -b_/12
b_ = np.array([3,-1])
B_ab2 = -b_/2
b_ = np.array([1, 0])
B_ab1 = -b_
b_ = np.array([-1, 0])
B_ab0 = -b_
A_ab1=get_A(B_ab1)
A_ab6=get_A(B_ab6)
A_ab5=get_A(B_ab5)
A_ab4=get_A(B_ab4)
A_ab3=get_A(B_ab3)
A_ab2=get_A(B_ab2)
A_ab0=get_A(B_ab0)

def SearchAndExe(Gpus, cmd, interval):
    prefix = 'CUDA_VISIBLE_DEVICES='
    foundGPU = 0
    while foundGPU==0:  # set waiting condition

        for u in Gpus: 
            gpu_power, gpu_memory, gpu_utilization = gpu_info(u)      
            cnt = 0   
            first = 0
            second = 0   
            empty = 1
            print('gpu, gpu_power, gpu_memory, cnt', u, gpu_power, gpu_memory, cnt)
            for i in range(5):
                gpu_power, gpu_memory, gpu_utilization = gpu_info(u)   
                print('gpu, gpu_power, gpu_memory, cnt', u, gpu_power, gpu_memory, cnt)
                # if gpu_memory > 2000 or gpu_power > 100: # running
                if gpu_utilization > 40 or gpu_power > 100 or gpu_memory > 4000:
                    empty = 0
                time.sleep(interval)
            if empty == 1:
                foundGPU = 1
                break
            
    if foundGPU == 1:
        prefix += str(u)
        cmd = prefix + ' '+ cmd
        print('\n' + cmd)
        os.system(cmd)
    
def rand_port():
    r = ' '

    r += str(random.randint(1, 5))
    r += str(random.randint(1, 9))
    r += str(random.randint(1, 9))
    r += str(random.randint(1, 9))
    r += str(random.randint(1, 9))
    r += " "
    return r

models = [
  
    ' ZeroSAny32Settings  ',
    ' ZeroSAny68Settings  ',
    ' ZeroSAny80Settings  ',
    ' ZeroSAny110Settings  ',
    ' ZeroSAny92Settings ',
    ' ZeroSAny152Settings ',

]
adv_trains =['']#, ' --adv_train ']#, ['']#, ' --adv_train ']# 
givenAs = [
    ' 1.5 -1.0 0.5 ',
    
    # ' 1 0 0 0 0 0 ',
        A_ab1,
        # ' 2.1667 -3.0000 2.5000  -0.6667'
    #     A_ab0,
        # A_ab2,
        # A_ab3,
        # A_ab4,
        # A_ab5,
        # A_ab6,
        # ' 1 0 '
        #    ' 1 0 ',
           
# #####  #    ' 1 0 ',
# ######   #    ' 1 0 ',
        
# # # #####       ' 0 1 ',
# # # ####          ' 0 1 ',
# #  #####           ' 0 1 ',

#             ' 0.5 0.5 ', 
#             ' 0.5 0.5 ', 
#             ' 0.5 0.5 ', 
            
#             # ' 2 -1 ', 
#             # ' 2 -1 ', 
#             # ' 2 -1 ',                        
    
#            ' 0.3333333333333333 0.5555555555555556 0.1111111111111111 ',
#            ' 0.3333333333333333 0.5555555555555556 0.1111111111111111 ',
#            ' 0.3333333333333333 0.5555555555555556 0.1111111111111111 ', 

#           ' 1.49323762497707 -0.574370781405754 0.0855838379295368 -0.00445068150085398 ',
#           ' 1.49323762497707 -0.574370781405754 0.0855838379295368 -0.00445068150085398 ',
#           ' 1.49323762497707 -0.574370781405754 0.0855838379295368 -0.00445068150085398 ',              
                      
#            ' 1 0 0 0 0 0 ',
#            ' 1 0 0 0 0 0 ',
#            ' 1 0 0 0 0 0 ',

          ]
givenBs = [
    # ' -1.0 0.0 0.0 0.0',
    ' -1.0 0 0 ', 
    # ' -0.8766 0.1319 0.1264 0.0618 -0.0242 -0.0065 ',
        ' '+str(B_ab1).replace('[', ' ').replace(']', ' '),
    #     ' '+str(B_ab0).replace('[', ' ').replace(']', ' '),
        # ' '+str(B_ab2).replace('[', ' ').replace(']', ' '),
        # ' '+str(B_ab3).replace('[', ' ').replace(']', ' '),
        # ' '+str(B_ab4).replace('[', ' ').replace(']', ' '),
    #    ' '+str( B_ab5).replace('[', ' ').replace(']', ' '),
        # ' '+str(B_ab6).replace('[', ' ').replace(']', ' '),
    # ' -1 1 '
#    ' -1.0 0 ',

# #    ###     #    ' -1.5 0.5 ',
# #   ###      #    ' 0 -2 ',

# #  ###       ' -2 0 ',
# # #####        ' -1 -1 ',
# # ####        ' 3 0 ',
        
#         ' -1.75 0.25 ', #S2O2
#         ' -3 1.5 ', #S2O1
#         ' 2 2 ', #S2O0
        
#         # ' -1 1 ',
#         # ' 1 -1 ',
#         # ' -1 -1 ',
        
#             #  '  -1.7777777777777778 0 0 ',#O2
#             # # #### ' 1.7777777777777778 -1.7777777777777778 -1.7777777777777778 ',#O1b
#             #  ' -3.5555555555556 1.7777777777778 0 ',#O1
#             #  ' 1 1 1 ', #O0
#             # #### ' -1.7777777777777778 1.7777777777777778 -1.7777777777777778 ',#O1
#             # #### ' 0.592592592596 -0.592592592596 -0.592592592596 ',
             
#              '  -1.7777777777777778 0 0 ',#O2
#              ' -3.5555555555556 1.7777777777778 0 ',#O1
#              ' -3 -3 3 ', #O0               
             
#             ' -2.10313656405320 2.80393876806197 -1.68484817541425 0.400601121454727 ',
#             ' -2.10313656405320 5.60787753612393 -7.29272571153819 3.20453988951669 ',
#             ' 0.400601121454727 -1.68484817541425 1.80393876806197 -2.10313656405320 ',
            
                    
#     ' -2.9701388888889 5.5020833333333  -6.9319444444444 5.0680555555556 -1.9979166666667 0.32986111111111 ',#S6O6
#            ' 5.5020833333333  -6.9319444444444 5.0680555555556 -1.9979166666667 0.32986111111111 -2.9701388888889 ',#S6O1
#            ' -2.9701388888889 -5.5020833333333  -6.9319444444444 -5.0680555555556 -1.9979166666667 -0.32986111111111 ',#S6O0
#         #    ' 2.9701388888889 5.5020833333333  -6.9319444444444 5.0680555555556 -1.9979166666667 0.32986111111111 ',#S6O0

           ]
# print(givenBs)
notes = [
    # ' Special4 '
    ' Taylor ',
        # ' Rob6 ',
        ' ab1 ',
        # ' ab0 ',
        # ' ab2 ',
        # ' ab3 ',
        # ' ab4 ',
        # ' ab5 ',
        # ' ab6 ',
        # ' ZeroSum_2 '
#         ' Step1Order1PreRes ',
         
#         ' Step2Order2 ',
        # ' Step2Order1 ',
#          ' Step2Order0 ',
         
# # ###################        #  ' Step2Order2Adv ',
# #   ###################      #  ' Step2Order0Adv ',

#           ' Step3Order2 ',
#           ' Step3Order1 ',
#           ' Step3Order0 ',
        
#          ' Step4Order4 ',
#          ' Step4Order2 ',
#          ' Step4Order0 ',
         
#          ' Step6Order6AB6 ',
#          ' Step6Order1AB6 ',
#          ' Step6Order0AB6 ',         

         ]
stepsS = [
    # ' 6 ',
    # ' '+str(len(B_ab1))+' ',
    #         ' '+str(len(B_ab0))+' ',
        # ' '+str(len(B_ab2))+' ',
        # ' '+str(len(B_ab3))+' ',
        # ' '+str(len(B_ab4))+' ',
        # ' '+str(len(B_ab5))+' ',
        # ' '+str(len(B_ab6))+' ',
        # ' 2 ',
        #  ' 1 ',
          
# #     ###########    # #   ' 2 ',
# #   #################      # #   ' 2 ',

        #  ' 2 ',
#          ' 2 ',        
#           ' 2 ',        
          
           ' 3 ',
           ' 1 ',
#            ' 3 ',
#            ' 3 ',    
                   
#           ' 4 ',
        #   ' 4 ',
#           ' 4 ',

#           ' 6 ',
#           ' 6 ',
#           ' 6 ',
          
   
]
ConverOrds = [
    ' 3 ',
    # ' 6 ',
    #     ' 1 ',' 0 ',  
    # ' 2 ', ' 3 ', ' 4 ',  ' 5 ',  ' 6 ', 
    # ' -1 '
            #  ' 5 ',

# # # ### # 
# # # #  ###           #   ' 2 ',
# # # #   ###          #   ' 0 ',

#                ' 2 ',
            #    ' 1 ', 
#                ' 0 ',
            
#              ' 2 ',
             ' 1 ',
#               ' 0 ',
              
            #   ' 4 ',
            #   ' 2 ',
#               ' 0 ',
              
            #   ' 6 ',
            #   ' 1 ',
#               ' 0 ',              
              ]
epsS = [' 0.031 ']
GPUs = [2,3,4,5,6,7,8,9,0]
# GPUs = [2]

datasets = ['svhn','cifar10','cifar100']#,'cifar10','svhn']#['mnist']#['cifar100']#, ['cifar10']svhn
# Settings = ['mnistConvStride2ResLikeExpDecayLearnDecay', 'mnistOriExpDecayLearnDecay', 'mnistAllEleExpDecayLearnDecay']

Settings = ['BnReluConv_ConvStride2ResLike_2Chs_Adam_SamllST_EulerSwitch010_noise']#, 'BnReluConv_ConvStride2ResLike_2Chs_Adam_SamllST_IniEveryStage']#BnReluConv_ConvStride2ResLike_2Chs_Adam_SamllST'Cosep200_0914RobAB']#'0508AllEleStep0p1','0508AllEleExpDecayLearnDecayStep0p1']#'0422AllEleStep0p1']#,'ConvStride2ResLike']
#['ConvStride2ResLikeExpDecayLearnDecay', 'OriExpDecayLearnDecay', 'AllEleExpDecayLearnDecay','AllEle']
#['mnistConvStride2ResLikeStart8', 'mnistOriStart8','mnistConvStride2ResLike8,16,32', 'mnistOri8,16,32']#,'mnistConvStride2ResLikeShare',]#'','mnistOri','mnistAllEle']#,'mnistConvStride2Learn','mnistConvStride2Share', 'mnistConvStride2ResLikeShare']#'ConvStride2Learn','AllEle','Ori']#, '2ChsG4','BnReluConv', 'BnReluConvBn',  'ConvStride2Fix']
data_aug = 'None'#'cutout'
inp_noiS = [0.0,0.001,0.01,0.1,0.2,0.3,0.4,0.5]#0.001,0.01,0.1,0.2,0.3,0.4,0.5]
# inp_noiS = [0.0]

ini_stepsizeS = [0.1]
sche = 'cos'#'cos'#
# opt = 'SGD'
opt = 'RealSGD'

minimizer = 'None'
bs = '128'
lr = '0.1'
# lr = '0.0001'

# bs = '256'
# lr = '0.2'

epoch = '200'
warm = '0'
# datapath = '/media_HDD_1/lab415/clm/OverThreeOrders/OverThreeOrders/CIFAR/data'
cnt=0
for exp in range(0,3):
    for ini_stepsize in ini_stepsizeS:
        for model in models:
            for eps in epsS:
                for dataset in datasets:
                    if dataset == "cifar10":
                        if sche == 'step' and epoch is None:
                            epoch = '160'
                    if dataset == "cifar100":
                        if sche == 'step'and epoch is None:
                            epoch = '300'     
                    if dataset == "mnist":
                        if epoch is None:
                            epoch = '5'                               
                    for adv_train in adv_trains:    
                        for inp_noi in inp_noiS:    
                            for i in range(len(givenBs)):
                                for setting in Settings:
                                    path_base = './runs/' + dataset+str(adv_train).replace(' ', '').replace('--', '') + '/Taylor/RobNoAdvExp_'+str(exp)+setting+'/'

                                # for GpuNum in [1,2,3,4,5,6,7,8,9]:                                
                                    save_path = path_base 
                                    
                                            
                                    cmd = ''
                                    cmd += ' nohup python3 TrainAndPlot.py --arch '+model
                                    cmd += ' --minimizer None --rho 0.5 --eta 0.01 --ini_stepsize '+str(ini_stepsize)
                                    cmd += ' --eps_iter 0.01 --nb_iter 7 '                        
                                    cmd += ' --givenA ' + givenAs[i] + ' --givenB '+ givenBs[i]
                                    cmd += ' --notes ' + notes[i] + adv_train
                                    cmd += ' --lr ' + lr +' --bs '+bs
                                    cmd += ' --dataset '+ dataset + ' --sche '+sche+' --steps '+stepsS[i]  
                                    cmd += ' --ConverOrd '+ConverOrds[i] +' --epoch '+epoch + ' --warm '+warm
                                    cmd += ' --eps '+ eps
                                    cmd += ' --data_aug '+data_aug
                                    cmd += ' --inp_noi '+str(inp_noi)
                                    # if Settings:
                                    save_path =  cmd + "{0:%Y%m%dT%H%M%S/}".format(datetime.now())
                                    save_path = save_path.replace('\n', '#').replace('\r', '#').replace(' ', '#').replace('##', '#').replace('--', '')
                                    save_path = path_base + model.replace(' ', '#')+save_path.split('notes')[1]
                                    save_path = save_path.replace('#','')
                                    if not os.path.exists(save_path):
                                        os.makedirs(save_path)   
                                    cmd += ' --Settings ' + setting                                         
                                    cmd += ' --save_path '+save_path
                                    cmd += ' --seed '+str(exp)
                                    cmd += '   >   '+save_path+'log'+'.txt 2>&1 '
                                    #if i < len(givenBs)-1:
                                    cmd += '&'
                                    SearchAndExe(GPUs, cmd, interval=4)                 
                                    cnt += 1
                                # print('\n', cnt, cmd)
                                # execute_cmd(GPUs[i], cmd, interval=10)
                                # os.system(cmd)

        # break
# for i in range(0,3):
#     command = "CUDA_VISIBLE_DEVICES=0 nohup python3 train_cifar_ZeroSNet.py --arch ZeroSAny20_Tra --dataset cifar10 --sche step --order 2  --minimizer None --rho 0.5 --eta 0.01 --ini_stepsize 1 --warm 0 --givenA 1 0 --givenB -1 0 --notes PreResAdv --adv_train --eps 0.1 --eps_iter 0.01 --nb_iter 7  > tem_u5.txt 2>&1 "
#     os.system(command)
#     command = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar_ZeroSNet.py --arch ZeroSAny56_Tra --dataset cifar100 --sche step --order 2  --minimizer None --rho 0.5 --eta 0.01 --ini_stepsize 1 --warm 0 --givenA 1 0 --givenB -1 0 --notes PreRes > tem_u5.txt 2>&1 "
#     os.system(command)
#     ## 44
#     command = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar_ZeroSNet.py --arch ZeroSAny44_Tra --dataset cifar10 --sche step --order 2  --minimizer None --rho 0.5 --eta 0.01 --ini_stepsize 1 --warm 0 --givenA 1 0 --givenB -1 0 --notes PreRes > tem_u5.txt 2>&1 "
#     os.system(command)
#     command = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar_ZeroSNet.py --arch ZeroSAny44_Tra --dataset cifar100 --sche step --order 2  --minimizer None --rho 0.5 --eta 0.01 --ini_stepsize 1 --warm 0 --givenA 1 0 --givenB -1 0 --notes PreRes > tem_u5.txt 2>&1 "
#     os.system(command)

#     ## 32
#     command = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar_ZeroSNet.py --arch ZeroSAny32_Tra --dataset cifar10 --sche step --order 2  --minimizer None --rho 0.5 --eta 0.01 --ini_stepsize 1 --warm 0 --givenA 1 0 --givenB -1 0 --notes PreRes > tem_u5.txt 2>&1 "
#     os.system(command)
#     command = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar_ZeroSNet.py --arch ZeroSAny32_Tra --dataset cifar100 --sche step --order 2  --minimizer None --rho 0.5 --eta 0.01 --ini_stepsize 1 --warm 0 --givenA 1 0 --givenB -1 0 --notes PreRes > tem_u5.txt 2>&1 "
#     os.system(command)
#     ## 20
#     command = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar_ZeroSNet.py --arch ZeroSAny20_Tra --dataset cifar10 --sche step --order 2  --minimizer None --rho 0.5 --eta 0.01 --ini_stepsize 1 --warm 0 --givenA 1 0 --givenB -1 0 --notes PreRes > tem_u5.txt 2>&1 "
#     os.system(command)
#     command = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar_ZeroSNet.py --arch ZeroSAny20_Tra --dataset cifar100 --sche step --order 2  --minimizer None --rho 0.5 --eta 0.01 --ini_stepsize 1 --warm 0 --givenA 1 0 --givenB -1 0 --notes PreRes > tem_u5.txt 2>&1 "
#     os.system(command)
#     ## 68
#     command = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar_ZeroSNet.py --arch ZeroSAny68_Tra --dataset cifar10 --sche step --order 2  --minimizer None --rho 0.5 --eta 0.01 --ini_stepsize 1 --warm 0 --givenA 1 0 --givenB -1 0 --notes PreRes > tem_u5.txt 2>&1 "
#     os.system(command)
#     command = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar_ZeroSNet.py --arch ZeroSAny68_Tra --dataset cifar100 --sche step --order 2  --minimizer None --rho 0.5 --eta 0.01 --ini_stepsize 1 --warm 0 --givenA 1 0 --givenB -1 0 --notes PreRes > tem_u5.txt 2>&1 "
#     os.system(command)

#     ## 80
#     command = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar_ZeroSNet.py --arch ZeroSAny80_Tra --dataset cifar10 --sche step --order 2  --minimizer None --rho 0.5 --eta 0.01 --ini_stepsize 1 --warm 0 --givenA 1 0 --givenB -1 0 --notes PreRes > tem_u5.txt 2>&1 "
#     os.system(command)
#     command = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar_ZeroSNet.py --arch ZeroSAny80_Tra --dataset cifar100 --sche step --order 2  --minimizer None --rho 0.5 --eta 0.01 --ini_stepsize 1 --warm 0 --givenA 1 0 --givenB -1 0 --notes PreRes > tem_u5.txt 2>&1 "
#     os.system(command)
#     ## 92
#     command = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar_ZeroSNet.py --arch ZeroSAny92_Tra --dataset cifar10 --sche step --order 2  --minimizer None --rho 0.5 --eta 0.01 --ini_stepsize 1 --warm 0 --givenA 1 0 --givenB -1 0 --notes PreRes > tem_u5.txt 2>&1 "
#     os.system(command)
#     command = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar_ZeroSNet.py --arch ZeroSAny92_Tra --dataset cifar100 --sche step --order 2  --minimizer None --rho 0.5 --eta 0.01 --ini_stepsize 1 --warm 0 --givenA 1 0 --givenB -1 0 --notes PreRes > tem_u5.txt 2>&1 "
#     os.system(command)
#     ## 104
#     command = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar_ZeroSNet.py --arch ZeroSAny104_Tra --dataset cifar10 --sche step --order 2  --minimizer None --rho 0.5 --eta 0.01 --ini_stepsize 1 --warm 0 --givenA 1 0 --givenB -1 0 --notes PreRes > tem_u5.txt 2>&1 "
#     os.system(command)
#     command = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar_ZeroSNet.py --arch ZeroSAny104_Tra --dataset cifar100 --sche step --order 2  --minimizer None --rho 0.5 --eta 0.01 --ini_stepsize 1 --warm 0 --givenA 1 0 --givenB -1 0 --notes PreRes > tem_u5.txt 2>&1 "
#     os.system(command)
#     ## 110
#     command = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar_ZeroSNet.py --arch ZeroSAny68_Tra --dataset cifar10 --sche step --order 2  --minimizer None --rho 0.5 --eta 0.01 --ini_stepsize 1 --warm 0 --givenA 1 0 --givenB -1 0 --notes PreRes > tem_u5.txt 2>&1 "
#     os.system(command)
#     command = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar_ZeroSNet.py --arch ZeroSAny68_Tra --dataset cifar100 --sche step --order 2  --minimizer None --rho 0.5 --eta 0.01 --ini_stepsize 1 --warm 0 --givenA 1 0 --givenB -1 0 --notes PreRes > tem_u5.txt 2>&1 "
#     os.system(command)