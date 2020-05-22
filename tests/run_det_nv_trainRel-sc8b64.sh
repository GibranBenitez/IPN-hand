#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python main.py \
	--root_path /host/space0/gibran/\
	--video_path dataset/Nvidia/nvgesture_arch \
	--annotation_path scripts/Real-time-GesRec/annotation_nvGesture/nvbinary.json \
	--result_path scripts/Real-time-GesRec/results \
	--dataset nv \
	--sample_duration 8 \
    --learning_rate 0.01 \
    --model resnetl \
	--model_depth 10 \
	--resnet_shortcut A \
	--batch_size 64 \
	--n_classes 2 \
	--n_finetune_classes 2 \
	--n_threads 32 \
	--checkpoint 1 \
	--modality RGB-flo \
	--train_crop random \
	--n_val_samples 1 \
	--test_subset test \
 	--n_epochs 150 \
 	--lr_steps 30 60 90 120 \
    --store_name nvDetRfl_sc8b64 \
