#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python main.py \
	--root_path /host/space0/gibran/\
	--video_path dataset/HandGestures/IPN_dataset \
	--annotation_path scripts/Real-time-GesRec/annotation_ipnGesture/ipnbinary.json \
	--result_path scripts/Real-time-GesRec/results_ipn \
	--dataset ipn \
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
    --store_name ipnDetRfl_sc8b64 \
