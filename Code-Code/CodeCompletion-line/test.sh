#!/usr/bin/bash

# token completion

## getting and preprocessing the data
cd Code-Code/CodeCompletion-token/dataset/javaCorpus/ || return;
bash download.sh;
python preprocess.py --base_dir=token_completion --output_dir=token_completion

## fine-tune the model
cd Code-Code/CodeCompletion-token/code || return;
LANG=java
DATADIR=../dataset/javaCorpus/token_completion
LITFILE=../dataset/javaCorpus/literals.json
OUTPUTDIR=../save/javaCorpus
PRETRAINDIR=microsoft/CodeGPT-small-java
LOGFILE=completion_javaCorpus.log
PER_NODE_GPU=1
python -m torch.distributed.launch --nproc_per_node=$PER_NODE_GPU run_lm.py \
        --data_dir=$DATADIR \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=1024 \
        --do_train \
        --gpu_per_node $PER_NODE_GPU \
        --learning_rate=8e-5 \
        --weight_decay=0.01 \
        --evaluate_during_training \
        --per_gpu_train_batch_size=2 \
        --per_gpu_eval_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --num_train_epochs=5 \
        --logging_steps=100 \
        --save_steps=1000 \
        --seed=42 \
        --overwrite_output_dir \
        --not_pretrain

## evaluation
export CUDA_VISIBLE_DEVICES=0
LANG=java
DATADIR=../dataset/javaCorpus/token_completion
LITFILE=../dataset/javaCorpus/literals.json
OUTPUTDIR=../save/javaCorpus
PRETRAINDIR=../save/javaCorpus/checkpoint-last
LOGFILE=completion_javaCorpus_eval.log
python -u run_lm.py \
        --data_dir=$DATADIR \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=1024 \
        --do_eval \
        --per_gpu_eval_batch_size=16 \
        --logging_steps=100 \
        --seed=42 

# line completion
cd Code-Code/CodeCompletion-line/code || return;

## evaluation
export CUDA_VISIBLE_DEVICES=0
LANG=java
DATADIR=../dataset/javaCorpus/line_completion
LITFILE=../dataset/javaCorpus/literals.json
OUTPUTDIR=../../CodeCompletion-token/save/javaCorpus
PRETRAINDIR=../../CodeCompletion-token/save/javaCorpus/checkpoint-last
LOGFILE=completion_javaCorpus_eval.log
python -u run_lm.py \
        --data_dir=$DATADIR \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=1024 \
        --eval_line \
        --logging_steps=100 \
        --seed=42 
