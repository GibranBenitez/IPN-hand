#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python offline_test.py \
	--root_path /host/space0/gibran/\
	--video_path dataset/Nvidia/nvgesture_arch \
	--annotation_path scripts/Real-time-GesRec/annotation_nvGesture/nvall_but_None.json \
	--result_path scripts/Real-time-GesRec/results \
	--resume_path scripts/Real-time-GesRec/report_ipn/nvClf_jes32rb32_resnext-101_7946.pth \
    --store_name nvClf_jes32r_b32 \
	--dataset nv \
	--sample_duration 32 \
    --model resnext \
	--model_depth 101 \
	--resnet_shortcut B \
	--batch_size 64 \
	--n_classes 25 \
	--n_finetune_classes 25 \
	--modality RGB \
	--n_val_samples 1 \
	--test_subset test \
    --n_epochs 100 \
    --no_train \
    --no_val \
    --test \