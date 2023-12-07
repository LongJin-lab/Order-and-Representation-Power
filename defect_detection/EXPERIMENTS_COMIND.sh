#!/usr/bin/env bash

source EXPERIMENTS_ROOT.sh

run_COMIND_EXPERIMENTS()
{
    RESULTS_PATH=$1; shift
    SAVE_IMAGES=$1; shift
    # GPUS=($@)
    GPUS_GROUP0=(0)

    train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH N_246 $RESULTS_PATH 15 -1 246 50 0.01 1 1 True  2 3 True  True  True 0.5 ${GPUS_GROUP0[0]}  

    train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH N_246 $RESULTS_PATH 15 -1 246 50 0.01 1 1 True  2 3 True  True  True 0.4 ${GPUS_GROUP0[0]}  

    train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH N_246 $RESULTS_PATH 15 -1 246 50 0.01 1 1 True  2 3 True  True  True 0.3 ${GPUS_GROUP0[0]}  

    train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH N_246 $RESULTS_PATH 15 -1 246 50 0.01 1 1 True  2 3 True  True  True 0.2 ${GPUS_GROUP0[0]}  

    train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH N_246 $RESULTS_PATH 15 -1 246 50 0.01 1 1 True  2 3 True  True  True 0.1 ${GPUS_GROUP0[0]}  
    
    train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH N_246 $RESULTS_PATH 15 -1 246 50 0.01 1 1 True  2 3 True  True  True 0.05 ${GPUS_GROUP0[0]}  

    train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH N_246 $RESULTS_PATH 15 -1 246 50 0.01 1 1 True  2 3 True  True  True 0.01 ${GPUS_GROUP0[0]}  

    train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH N_246 $RESULTS_PATH 15 -1 246 50 0.01 1 1 True  2 3 True  True  True 0.0 ${GPUS_GROUP0[0]}  

    # train_KSDD $SAVE_IMAGES N_ALL  $RESULTS_PATH 7 33 33 50 1 0.01 1 True  2 1 True  True  True 0.5 "${GPUS_GROUP0[@]}"  
    # train_KSDD $SAVE_IMAGES N_ALL  $RESULTS_PATH 7 33 33 50 1 0.01 1 True  2 1 True  True  True 0.4 "${GPUS_GROUP0[@]}"  
    # train_KSDD $SAVE_IMAGES N_ALL  $RESULTS_PATH 7 33 33 50 1 0.01 1 True  2 1 True  True  True 0.3 "${GPUS_GROUP0[@]}"  
    # train_KSDD $SAVE_IMAGES N_ALL  $RESULTS_PATH 7 33 33 50 1 0.01 1 True  2 1 True  True  True 0.2 "${GPUS_GROUP0[@]}"  
    # train_KSDD $SAVE_IMAGES N_ALL  $RESULTS_PATH 7 33 33 50 1 0.01 1 True  2 1 True  True  True 0.1 "${GPUS_GROUP0[@]}"  
    # train_KSDD $SAVE_IMAGES N_ALL  $RESULTS_PATH 7 33 33 50 1 0.01 1 True  2 1 True  True  True 0.05 "${GPUS_GROUP0[@]}"  
    # train_KSDD $SAVE_IMAGES N_ALL  $RESULTS_PATH 7 33 33 50 1 0.01 1 True  2 1 True  True  True 0.01 "${GPUS_GROUP0[@]}"  
    # train_KSDD $SAVE_IMAGES N_ALL  $RESULTS_PATH 7 33 33 50 1 0.01 1 True  2 1 True  True  True 0.0 "${GPUS_GROUP0[@]}"  

}


# Space delimited list of GPU IDs which will be used for training
# GPUS=(0 1 2)
GPUS=(0)

if [ "${#GPUS[@]}" -eq 0 ]; then
  GPUS=(0)
  #GPUS=(0 1 2) # if more GPUs available
fi

run_COMIND_EXPERIMENTS ./results-comind/TunedHyper/ True "${GPUS[@]}"


