from nni.experiment import Experiment
import torch
import os
import logging
import os
import nni

logger = logging.getLogger('DebugCudaVisibleDevices')
logger.info(f'''trial {nni.get_trial_id()}: "{os.getenv('CUDA_VISIBLE_DEVICES')}"''')

print(torch.cuda.is_available())
torch.cuda.current_device()
torch.cuda._initialized = True
Experiment = Experiment('local')
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'

Experiment.config.training_service.use_active_gpu = True
Experiment.config.trial_gpu_number = 1
Experiment.config.trial_command = 'python3 TrainAndPlotTuning.py --arch  ZeroSAny20Settings  --minimizer None --rho 0.5 --eta 0.01 --ini_stepsize 1 --eps_iter 0.01 --nb_iter 7  --givenA   1 0   --givenB   -1  0  --notes  SumBase_ab1  --lr 0.1 --bs 128 --opt SGD --dataset cifar100 --sche cos --steps  2  --ConverOrd  1  --epoch 1 --warm 0 --eps  0.031  --Settings BnReluConvConvStride2ResLikeExpDecayLearnDecayLearnBal_Tuning --save_path /media_HDD_1/lab415/clm/OverThreeOrders/OverThreeOrders/CIFAR/runs/cifar100/Tuning/'#CUDA_VISIBLE_DEVICES=5 
Experiment.config.trial_code_directory = '/media_HDD_1/lab415/clm/OverThreeOrders/OverThreeOrders/CIFAR/'
search_space = {
    'IniDecay': {'_type': 'uniform', '_value': [-10.0, 5.0]},
}
Experiment.config.search_space = search_space
Experiment.config.tuner.name = 'Random'#'TPE'#'Random'
Experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
Experiment.config.max_trial_number = 5
Experiment.config.trial_concurrency = 4

Experiment.run(4562)

# Experiment.get_all_experiments_metadata()
input() 