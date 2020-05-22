import argparse
import time
import os
import glob 
import sys
import json
import shutil
import itertools
import numpy as np
import pandas as pd 
import csv

from opts import parse_opts_online

import pdb
import numpy as np
import scipy.io as sio

def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)

opt = parse_opts_online()
opt.result_path = os.path.join(opt.result_path, opt.store_name)
opt.video_path = os.path.join(opt.root_path, opt.video_path)

## Get list of videos to test
if opt.dataset == 'ipn':
    file_set = os.path.join(opt.video_path, 'ValList.txt')
    test_names = []
    num_frames = []
    buf = 0
    with open(file_set,'rb') as f:
        for line in f:
            vid_name = line.decode().split('\t')[0]
            test_names.append(vid_name)
            num_frames.append(line.decode().split('\t')[1])

print('Start Evaluation')
sys.stdout.flush()
rgb = load_annotation_data(os.path.join(opt.root_path, opt.result_path+'.json'))
rgbf = load_annotation_data(os.path.join(opt.root_path, opt.result_path.replace('RGB','RGB-flo')+'.json'))
rgbs = load_annotation_data(os.path.join(opt.root_path, opt.result_path.replace('RGB','RGB-seg')+'.json'))
assert len(test_names) == len(rgb['all_pred']) == len(rgbf['all_pred']) == len(rgbs['all_pred'])
rgb_idxs = []
rgbf_idxs = []
rgbs_idxs = []
gt_idxs = []

videoidx = 0
for idx, path in enumerate(test_names[buf:]):
    videoidx += 1

    print('[{}/{}] {} |=========='.format(videoidx,len(test_names),path))
    sys.stdout.flush()
    n_frames = num_frames[idx]

    rgb_idx = np.zeros(6000)
    rgb_pred = np.array(rgb['all_pred'][idx]) + 1
    rgbf_idx = np.zeros(6000)
    rgbf_pred = np.array(rgbf['all_pred'][idx]) + 1
    rgbs_idx = np.zeros(6000)
    rgbs_pred = np.array(rgbs['all_pred'][idx]) + 1
    gt_idx = np.zeros(6000)
    gt_true = np.array(rgb['all_true'][idx]) + 1
    
    rgb_idx[0] = n_frames
    for i, predicted in enumerate(rgb_pred):
        pred_end = rgb['all_pred_frames'][idx][i]
        pred_start = rgb['all_pred_starts'][idx][i]

        if pred_end-pred_start > (opt.clf_queue_size+opt.sample_duration_clf):
            pred_start = pred_end - (opt.clf_queue_size+opt.sample_duration_clf)
        rgb_idx[pred_start:pred_end] = predicted
    rgb_idxs.append(rgb_idx)

    rgbf_idx[0] = n_frames
    for i, predicted in enumerate(rgbf_pred):
        pred_end = rgbf['all_pred_frames'][idx][i]
        pred_start = rgbf['all_pred_starts'][idx][i]

        if pred_end-pred_start > (opt.clf_queue_size+opt.sample_duration_clf):
            pred_start = pred_end - (opt.clf_queue_size+opt.sample_duration_clf)
        rgbf_idx[pred_start:pred_end] = predicted
    rgbf_idxs.append(rgbf_idx)

    rgbs_idx[0] = n_frames
    for i, predicted in enumerate(rgbs_pred):
        pred_end = rgbs['all_pred_frames'][idx][i]
        pred_start = rgbs['all_pred_starts'][idx][i]

        if pred_end-pred_start > (opt.clf_queue_size+opt.sample_duration_clf):
            pred_start = pred_end - (opt.clf_queue_size+opt.sample_duration_clf)
        rgbs_idx[pred_start:pred_end] = predicted
    rgbs_idxs.append(rgbs_idx)

    gt_idx[0] = n_frames
    for i, predicted in enumerate(gt_true):
        pred_end = rgb['all_true_frames'][idx][i]
        pred_start = rgb['all_true_starts'][idx][i]

        gt_idx[pred_start:pred_end] = predicted
    gt_idxs.append(gt_idx)

rgb_preds = np.array(rgb_idxs)
rgbf_preds = np.array(rgbf_idxs)
rgbs_preds = np.array(rgbs_idxs)
gt_preds = np.array(gt_idxs)
sio.savemat(os.path.join(opt.root_path,opt.result_path+'.mat'), {'rgb_preds':rgb_preds, 'rgbf_preds':rgbf_preds, 'rgbs_preds':rgbs_preds, 'gt_preds':gt_preds})