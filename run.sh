#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py --exp_id 000 --run_id 001 --skip_wandb --epochs 1 --input_feats MelSpec --model minerva --p_factor 1 --ex_per_class 384 --skip_train
#CUDA_VISIBLE_DEVICES=0 python main.py --exp_id 001 --run_id 001 --skip_wandb --epochs 1 --input_feats MelSpec --model ffnn_init --feat_embed_dim 1024 --class_embed_dim 1024 --lr 5e-4 --wd 1e-5 --do_class 0.3
CUDA_VISIBLE_DEVICES=0 python main.py --exp_id 002 --run_id 001 --skip_wandb --epochs 1 --input_feats MelSpec --model minerva --feat_embed_dim 64 --lr 1e-3 --wd 1e-4 --do_feats 0 --do_class 0.1 --p_factor 9
#CUDA_VISIBLE_DEVICES=0 python main.py --exp_id 003 --run_id 001 --skip_wandb --epochs 1 --input_feats MelSpec --model minerva --feat_embed_dim 64 --lr 1e-3 --wd 1e-6 --lr_ex 1e-3 --wd_ex 0 --do_feats 0 --do_class 0 --p_factor 7 --train_ex_classes
#CUDA_VISIBLE_DEVICES=0 python main.py --exp_id 004 --run_id 001 --skip_wandb --epochs 1 --input_feats hubert --model minerva --feat_embed_dim 64 --lr 1e-4 --wd 1e-6 --lr_ex 1e-3 --wd_ex 0 --lr_cr 1e-6 --wd_cr 1 --do_feats 0.1 --do_class 0 --p_factor 5 --train_class_reps --train_ex_classes
