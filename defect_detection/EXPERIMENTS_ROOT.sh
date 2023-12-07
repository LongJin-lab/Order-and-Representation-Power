#!/usr/bin/env bash

KSDD_PATH="/media3/datasets/KolektorSDD/"
DAGM_PATH="./datasets/DAGM/"
STEEL_PATH="./datasets/STEEL/"
KSDD2_PATH="/media3/datasets/KolektorSDD2/"


train_KSDD()
{
    SAVE_IMAGES=$1;shift
    RUN_NAME=$1; shift
    RESULTS_PATH=$1; shift
    DILATE=$1; shift
    TRAIN_NUM=$1; shift
    NUM_SEGMENTED=$1; shift

    EPOCHS=$1; shift
    LEARNING_RATE=$1; shift
    DELTA_CLS_LOSS=$1; shift
    BATCH_SIZE=$1; shift
    WEIGHTED_SEG_LOSS=$1; shift
    WEIGHTED_SEG_LOSS_P=$1; shift
    WEIGHTED_SEG_LOSS_MAX=$1; shift
    DYN_BALANCED_LOSS=$1; shift
    GRADIENT_ADJUSTMENT=$1; shift
    FREQUENCY_SAMPLING=$1; shift

    DATASET=KSDD
    TRAIN_NOI_COE=$1; shift  # Add this line, replace N with the correct position

    GPUS=($@)
    GPU1=${GPUS[0]}
    GPU2=${GPUS[1]}
    GPU3=${GPUS[2]}
    N=${#GPUS[*]}
    echo Will evaluate on "$N" GPUS!

    local local_results_path=$RESULTS_PATH

    if [ "$TRAIN_NOI_COE" = "0.0" ]; then
        local_results_path="${local_results_path}_Noi0p0_TayLearnABTau0p1"
    fi 
    if [ "$TRAIN_NOI_COE" = "0.01" ]; then
        local_results_path="${local_results_path}_Noi0p01_TayLearnABTau0p1"
    fi   
    if [ "$TRAIN_NOI_COE" = "0.05" ]; then
        local_results_path="${local_results_path}_Noi0p05_TayLearnABTau0p1"
    fi      
    if [ "$TRAIN_NOI_COE" = "0.1" ]; then
        local_results_path="${local_results_path}_Noi0p1_TayLearnABTau0p1"
    fi
    if [ "$TRAIN_NOI_COE" = "0.2" ]; then
        local_results_path="${local_results_path}_Noi0p2_TayLearnABTau0p1"
    fi
    if [ "$TRAIN_NOI_COE" = "0.3" ]; then
        local_results_path="${local_results_path}_Noi0p3_TayLearnABTau0p1"
    fi   
    if [ "$TRAIN_NOI_COE" = "0.4" ]; then
        local_results_path="${local_results_path}_Noi0p4_TayLearnABTau0p1"
    fi    
    if [ "$TRAIN_NOI_COE" = "0.5" ]; then
        local_results_path="${local_results_path}_Noi0p5_TayLearnABTau0p1"
    fi 
    if [ "$TRAIN_NOI_COE" = "0.6" ]; then
        local_results_path="${local_results_path}_Noi0p6_TayLearnABTau0p1"
    fi     
    if [ "$TRAIN_NOI_COE" = "0.7" ]; then
        local_results_path="${local_results_path}_Noi0p7_TayLearnABTau0p1"
    fi 
    if [ "$TRAIN_NOI_COE" = "0.8" ]; then
        local_results_path="${local_results_path}_Noi0p8_TayLearnABTau0p1"
    fi     
    if [ "$TRAIN_NOI_COE" = "0.9" ]; then
        local_results_path="${local_results_path}_Noi0p9_TayLearnABTau0p1"
    fi    
    if [ "$TRAIN_NOI_COE" = "1.0" ]; then
        local_results_path="${local_results_path}_Noi1p0_TayLearnABTau0p1"
    fi  


    # RUN_ARGS="--DATASET=KSDD --DATASET_PATH=$KSDD_PATH --DILATE=$DILATE --SAVE_IMAGES=$SAVE_IMAGES --RUN_NAME=$RUN_NAME --NUM_SEGMENTED=$NUM_SEGMENTED --RESULTS_PATH=$RESULTS_PATH --TRAIN_NUM=$TRAIN_NUM --EPOCHS=$EPOCHS --LEARNING_RATE=$LEARNING_RATE --DELTA_CLS_LOSS=$DELTA_CLS_LOSS --BATCH_SIZE=$BATCH_SIZE --WEIGHTED_SEG_LOSS=$WEIGHTED_SEG_LOSS --WEIGHTED_SEG_LOSS_P=$WEIGHTED_SEG_LOSS_P --WEIGHTED_SEG_LOSS_MAX=$WEIGHTED_SEG_LOSS_MAX --DYN_BALANCED_LOSS=$DYN_BALANCED_LOSS --GRADIENT_ADJUSTMENT=$GRADIENT_ADJUSTMENT --FREQUENCY_SAMPLING=$FREQUENCY_SAMPLING --TRAIN_NOI_COE=$TRAIN_NOI_COE --VALIDATE=True --VALIDATE_ON_TEST=True"

    RUN_ARGS="--DATASET=KSDD --DATASET_PATH=$KSDD_PATH --DILATE=$DILATE --SAVE_IMAGES=$SAVE_IMAGES --RUN_NAME=$RUN_NAME --NUM_SEGMENTED=$NUM_SEGMENTED --RESULTS_PATH=$local_results_path --TRAIN_NUM=$TRAIN_NUM --EPOCHS=$EPOCHS --LEARNING_RATE=$LEARNING_RATE --DELTA_CLS_LOSS=$DELTA_CLS_LOSS --BATCH_SIZE=$BATCH_SIZE --WEIGHTED_SEG_LOSS=$WEIGHTED_SEG_LOSS --WEIGHTED_SEG_LOSS_P=$WEIGHTED_SEG_LOSS_P --WEIGHTED_SEG_LOSS_MAX=$WEIGHTED_SEG_LOSS_MAX --DYN_BALANCED_LOSS=$DYN_BALANCED_LOSS --GRADIENT_ADJUSTMENT=$GRADIENT_ADJUSTMENT --FREQUENCY_SAMPLING=$FREQUENCY_SAMPLING --TRAIN_NOI_COE=$TRAIN_NOI_COE --VALIDATE=True --VALIDATE_ON_TEST=True"

    if [ "$N" -eq 1 ]
    then
      mkdir -p $local_results_path/$DATASET/$RUN_NAME/FOLD_0 &&  python -u train_net.py --GPU="$GPU1" --FOLD=0 $RUN_ARGS | /usr/bin/tee $local_results_path/$DATASET/$RUN_NAME/FOLD_0/training_log.txt
      mkdir -p $local_results_path/$DATASET/$RUN_NAME/FOLD_1 &&  python -u train_net.py --GPU="$GPU1" --FOLD=1 $RUN_ARGS | /usr/bin/tee $local_results_path/$DATASET/$RUN_NAME/FOLD_1/training_log.txt
      mkdir -p $local_results_path/$DATASET/$RUN_NAME/FOLD_2 &&  python -u train_net.py --GPU="$GPU1" --FOLD=2 $RUN_ARGS | /usr/bin/tee $local_results_path/$DATASET/$RUN_NAME/FOLD_2/training_log.txt

    fi

    if [ "$N" -eq 2 ]
    then
      mkdir -p $local_results_path/$DATASET/$RUN_NAME/FOLD_0 &&  python -u train_net.py --GPU="$GPU1" --FOLD=0 $RUN_ARGS | /usr/bin/tee $local_results_path/$DATASET/$RUN_NAME/FOLD_0/training_log.txt &
      mkdir -p $local_results_path/$DATASET/$RUN_NAME/FOLD_1 &&  python -u train_net.py --GPU="$GPU2" --FOLD=1 $RUN_ARGS | /usr/bin/tee $local_results_path/$DATASET/$RUN_NAME/FOLD_1/training_log.txt &
      wait

      mkdir -p $local_results_path/$DATASET/$RUN_NAME/FOLD_2 &&  python -u train_net.py --GPU="$GPU1" --FOLD=2 $RUN_ARGS | /usr/bin/tee $local_results_path/$DATASET/$RUN_NAME/FOLD_2/training_log.txt
    fi

    if [ "$N" -gt 2 ]
    then
      mkdir -p $local_results_path/$DATASET/$RUN_NAME/FOLD_0 &&  python -u train_net.py --GPU="$GPU1" --FOLD=0 $RUN_ARGS | /usr/bin/tee $local_results_path/$DATASET/$RUN_NAME/FOLD_0/training_log.txt &
      mkdir -p $local_results_path/$DATASET/$RUN_NAME/FOLD_1 &&  python -u train_net.py --GPU="$GPU2" --FOLD=1 $RUN_ARGS | /usr/bin/tee $local_results_path/$DATASET/$RUN_NAME/FOLD_1/training_log.txt &
      mkdir -p $local_results_path/$DATASET/$RUN_NAME/FOLD_2 &&  python -u train_net.py --GPU="$GPU3" --FOLD=2 $RUN_ARGS | /usr/bin/tee $local_results_path/$DATASET/$RUN_NAME/FOLD_2/training_log.txt &
      wait

    fi

    echo Running eval
    python -u join_folds_results.py --RUN_NAME="$RUN_NAME" --RESULTS_PATH="$local_results_path" --DATASET=KSDD | /usr/bin/tee -a $local_results_path/$DATASET/$RUN_NAME/eval_log.txt
}

train_DAGM()
{
    SAVE_IMAGES=$1;shift
    RUN_NAME=$1; shift
    RESULTS_PATH=$1; shift

    DILATE=$1; shift
    NUM_SEGMENTED=$1; shift

    EPOCHS=$1; shift
    LEARNING_RATE=$1; shift
    DELTA_CLS_LOSS=$1; shift
    BATCH_SIZE=$1; shift
    WEIGHTED_SEG_LOSS=$1; shift
    WEIGHTED_SEG_LOSS_P=$1; shift
    WEIGHTED_SEG_LOSS_MAX=$1; shift
    DYN_BALANCED_LOSS=$1; shift
    GRADIENT_ADJUSTMENT=$1; shift
    FREQUENCY_SAMPLING=$1; shift


    GPUS=($@)
    N=${#GPUS[*]}
    echo Will evaluate on "$N" GPUS!

    class=1



    RUN_ARGS="--DILATE=$DILATE --SAVE_IMAGES=$SAVE_IMAGES --DATASET_PATH=$DAGM_PATH --NUM_SEGMENTED=$NUM_SEGMENTED --RUN_NAME=$RUN_NAME --RESULTS_PATH=$RESULTS_PATH --DATASET=DAGM --EPOCHS=$EPOCHS --LEARNING_RATE=$LEARNING_RATE --DELTA_CLS_LOSS=$DELTA_CLS_LOSS --BATCH_SIZE=$BATCH_SIZE --WEIGHTED_SEG_LOSS=$WEIGHTED_SEG_LOSS --WEIGHTED_SEG_LOSS_P=$WEIGHTED_SEG_LOSS_P --WEIGHTED_SEG_LOSS_MAX=$WEIGHTED_SEG_LOSS_MAX --DYN_BALANCED_LOSS=$DYN_BALANCED_LOSS --GRADIENT_ADJUSTMENT=$GRADIENT_ADJUSTMENT --FREQUENCY_SAMPLING=$FREQUENCY_SAMPLING --VALIDATE=True --VALIDATE_ON_TEST=True"

    for (( ;; ));
    do
      for j in $(seq 0 $(( $N - 1 )));
      do
          LOG_REDIRECT=$RESULTS_PATH/DAGM/$RUN_NAME/FOLD_$class/training_log.txt

          mkdir -p $RESULTS_PATH/DAGM/$RUN_NAME/FOLD_$class && python -u train_net.py --GPU=${GPUS[$j]} --FOLD=$class $RUN_ARGS | /usr/bin/tee $LOG_REDIRECT &

          class=$(( $class + 1 ))
          [[ $class -eq 11 ]] && break
      done
      sleep 1
      wait
      [[ $class -eq 11 ]] && break
    done
    wait

}


train_single()
{
    SAVE_IMAGES=$1;shift
    DATASET=$1; shift
    DATASET_PATH=$1; shift
    RUN_NAME=$1; shift
    RESULTS_PATH=$1; shift
    DILATE=$1; shift
    TRAIN_NUM=$1; shift
    NUM_SEGMENTED=$1; shift

    EPOCHS=$1; shift
    LEARNING_RATE=$1; shift
    DELTA_CLS_LOSS=$1; shift
    BATCH_SIZE=$1; shift
    WEIGHTED_SEG_LOSS=$1; shift
    WEIGHTED_SEG_LOSS_P=$1; shift
    WEIGHTED_SEG_LOSS_MAX=$1; shift
    DYN_BALANCED_LOSS=$1; shift
    GRADIENT_ADJUSTMENT=$1; shift
    FREQUENCY_SAMPLING=$1; shift
    TRAIN_NOI_COE=$1; shift

    GPU=$1; shift
    local local_results_path=$RESULTS_PATH
    if [ "$TRAIN_NOI_COE" = "0.0" ]; then
        local_results_path="${local_results_path}_Noi0p0_TayLearnABTau0p1"
    fi 
    if [ "$TRAIN_NOI_COE" = "0.01" ]; then
        local_results_path="${local_results_path}_Noi0p01_TayLearnABTau0p1"
    fi   
    if [ "$TRAIN_NOI_COE" = "0.05" ]; then
        local_results_path="${local_results_path}_Noi0p05_TayLearnABTau0p1"
    fi      
    if [ "$TRAIN_NOI_COE" = "0.1" ]; then
        local_results_path="${local_results_path}_Noi0p1_TayLearnABTau0p1"
    fi
    if [ "$TRAIN_NOI_COE" = "0.2" ]; then
        local_results_path="${local_results_path}_Noi0p2_TayLearnABTau0p1"
    fi
    if [ "$TRAIN_NOI_COE" = "0.3" ]; then
        local_results_path="${local_results_path}_Noi0p3_TayLearnABTau0p1"
    fi   
    if [ "$TRAIN_NOI_COE" = "0.4" ]; then
        local_results_path="${local_results_path}_Noi0p4_TayLearnABTau0p1"
    fi    
    if [ "$TRAIN_NOI_COE" = "0.5" ]; then
        local_results_path="${local_results_path}_Noi0p5_TayLearnABTau0p1"
    fi 
    if [ "$TRAIN_NOI_COE" = "0.6" ]; then
        local_results_path="${local_results_path}_Noi0p6_TayLearnABTau0p1"
    fi     
    if [ "$TRAIN_NOI_COE" = "0.7" ]; then
        local_results_path="${local_results_path}_Noi0p7_TayLearnABTau0p1"
    fi 
    if [ "$TRAIN_NOI_COE" = "0.8" ]; then
        local_results_path="${local_results_path}_Noi0p8_TayLearnABTau0p1"
    fi     
    if [ "$TRAIN_NOI_COE" = "0.9" ]; then
        local_results_path="${local_results_path}_Noi0p9_TayLearnABTau0p1"
    fi    
    if [ "$TRAIN_NOI_COE" = "1.0" ]; then
        local_results_path="${local_results_path}_Noi1p0_TayLearnABTau0p1"
    fi  
         
    RUN_ARGS="--SAVE_IMAGES=$SAVE_IMAGES --DATASET_PATH=$DATASET_PATH --DILATE=$DILATE --NUM_SEGMENTED=$NUM_SEGMENTED --TRAIN_NUM=$TRAIN_NUM --RUN_NAME=$RUN_NAME --RESULTS_PATH=$local_results_path --DATASET=$DATASET --EPOCHS=$EPOCHS --LEARNING_RATE=$LEARNING_RATE --DELTA_CLS_LOSS=$DELTA_CLS_LOSS --BATCH_SIZE=$BATCH_SIZE --WEIGHTED_SEG_LOSS=$WEIGHTED_SEG_LOSS --WEIGHTED_SEG_LOSS_P=$WEIGHTED_SEG_LOSS_P --WEIGHTED_SEG_LOSS_MAX=$WEIGHTED_SEG_LOSS_MAX --DYN_BALANCED_LOSS=$DYN_BALANCED_LOSS --GRADIENT_ADJUSTMENT=$GRADIENT_ADJUSTMENT --FREQUENCY_SAMPLING=$FREQUENCY_SAMPLING --TRAIN_NOI_COE=$TRAIN_NOI_COE --VALIDATE=True"

    LOG_REDIRECT=$local_results_path/$DATASET/$RUN_NAME/training_log.txt

    mkdir -p $local_results_path/$DATASET/$RUN_NAME/ && python -u train_net.py --GPU=$GPU $RUN_ARGS | /usr/bin/tee $LOG_REDIRECT


}

