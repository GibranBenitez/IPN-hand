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
import torch
from torch import nn
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from torch.nn import functional as F

from opts import parse_opts_online
from model import generate_model, _modify_first_conv_layer, _construct_depth_model
from mean import get_mean, get_std
from spatial_transforms import *
from temporal_transforms import *
from target_transforms import ClassLabel
from dataset import get_online_data 
from utils import Logger, AverageMeter, LevenshteinDistance, Queue

import pdb
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scipy.io as sio
import pickle


def weighting_func(x):
    return (1 / (1 + np.exp(-0.2*(x-9))))


opt = parse_opts_online()

def load_models(opt):
    opt.resume_path = opt.resume_path_clf
    opt.pretrain_path = opt.pretrain_path_clf
    opt.sample_duration = opt.sample_duration_clf
    opt.model = opt.model_clf
    opt.model_depth = opt.model_depth_clf
    opt.modality = opt.modality_clf
    opt.resnet_shortcut = opt.resnet_shortcut_clf
    opt.n_classes = opt.n_classes_clf
    opt.n_finetune_classes = opt.n_finetune_classes_clf
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value)
    opt.std = get_std(opt.norm_value)

    print(opt)
    with open(os.path.join(opt.result_path, 'opts_clf_{}.json'.format(opt.store_name)), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)
    if opt.modality == 'Depth':
        opt.modality = 'RGB'

    classifier, parameters = generate_model(opt)

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']
        classifier.load_state_dict(checkpoint['state_dict'])
        if opt.sample_duration_clf < 32 and opt.model_clf != 'c3d':
            classifier = _modify_first_conv_layer(classifier,3,3)
        classifier = _construct_depth_model(classifier)
        classifier = classifier.cuda()

    if not opt.modality == opt.modality_clf:
        opt.modality = opt.modality_clf
    print('Model \n', classifier)
    pytorch_total_params = sum(p.numel() for p in classifier.parameters() if
                               p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params)

    return classifier



classifier = load_models(opt)

if opt.no_mean_norm and not opt.std_norm:
    norm_method = Normalize([0, 0, 0], [1, 1, 1])
elif not opt.std_norm:
    norm_method = Normalize(opt.mean, [1, 1, 1])
else:
    norm_method = Normalize(opt.mean, opt.std)


spatial_transform = Compose([
    Scale(112),
    CenterCrop(112),
    ToTensor(opt.norm_value), norm_method
    ])

target_transform = ClassLabel()


## Get list of videos to test
if opt.dataset == 'egogesture':
    subject_list = ['Subject{:02d}'.format(i) for i in [2, 9, 11, 14, 18, 19, 28, 31, 41, 47]]
    test_paths = []
    buf = 4
    for subject in subject_list:
        for x in glob.glob(os.path.join(opt.video_path,subject,'*/*/rgb*')):
            test_paths.append(x)
elif opt.dataset == 'nv':
    df = pd.read_csv(os.path.join(opt.video_path,'nvgesture_test_correct_cvpr2016_v2.lst'), delimiter = ' ', header = None)
    test_paths = []
    buf = 4
    for x in df[0].values:
        if opt.modality_det == 'RGB':
            test_paths.append(os.path.join(opt.video_path, x.replace('path:', ''), 'sk_color_all'))
        elif opt.modality_det == 'Depth':
            test_paths.append(os.path.join(opt.video_path, x.replace('path:', ''), 'sk_depth_all'))
elif opt.dataset == 'AHG':
    data = sio.loadmat(os.path.join(opt.root_path,'bega/datasets/AHG/splitfiles/testlist01.mat'))['raw_list'][0]
    test_paths = []
    true_classes_all = []
    true_frames_all = []
    buf = 0
    for i in range(data.shape[0]):
        test_paths.append(str(data[i][0][0]))
        true_classes_all.append(np.array(data[i][1][0]))
        true_frames_all.append(np.array(data[i][-2][0]))
elif opt.dataset == 'denso':
    if opt.test_subset == 'val':
        print('Feature extraction of Validation set with {}ns'.format(opt.sample_duration))
        data = sio.loadmat(os.path.join(opt.root_path,'bega/datasets/Pointing/train_sets/valid_list3.mat'))['raw_list'][0]
    elif opt.test_subset == 'test':
        print('Feature extraction of Testing set with {}ns'.format(opt.sample_duration))
        data = sio.loadmat(os.path.join(opt.root_path,'bega/datasets/Pointing/train_sets/test_list3.mat'))['raw_list'][0]
    elif opt.test_subset == 'train':
        print('Feature extraction of Training set with {}ns'.format(opt.sample_duration))
        data = sio.loadmat(os.path.join(opt.root_path,'bega/datasets/Pointing/train_sets/train_list3.mat'))['raw_list'][0]
    else:
        print('ERROR: chose val or test set for online evaluation')
        assert(opt.test_subset == 1)
    test_paths = []
    true_classes_all = []
    true_frames_all = []
    buf = 0
    for i in range(data.shape[0]):                          #All videos
        test_paths.append(str(data[i][0][0]))               #path
        true_classes_all.append(np.array(data[i][1][0]))    #classes
        true_frames_all.append(np.array(data[i][-1][0]))    #gef

print('Start Feature Extraction')
classifier.eval()

sta_frames = []
end_frames = []
pre_classes = []
videoidx = 0
for idx, path in enumerate(test_paths[buf:]):
    if opt.dataset == 'egogesture':
        opt.whole_path = path.split(os.sep, 4)[-1]
    elif opt.dataset == 'nv':
        opt.whole_path = path.split(os.sep, 6)[-1]
    elif opt.dataset == 'AHG':
        opt.whole_path = path
    elif opt.dataset == 'denso':
        opt.whole_path = path
    
    videoidx += 1
    active_index = 0
    passive_count = 0
    active = False
    prev_active = False
    finished_prediction = None
    pre_predict = False

    cum_sum = np.zeros(opt.n_classes_clf,)
    clf_selected_queue = np.zeros(opt.n_classes_clf,)
    det_selected_queue = np.zeros(opt.n_classes_det,)
    myqueue_det = Queue(opt.det_queue_size ,  n_classes = opt.n_classes_det)
    myqueue_clf = Queue(opt.clf_queue_size, n_classes = opt.n_classes_clf )

    opt.sample_duration = opt.sample_duration_clf
    vid_path = os.path.join(opt.result_path,'features_{}_{}ns'.format(opt.model,opt.sample_duration),path)
    if not os.path.exists(vid_path):
        os.makedirs(vid_path)
    print('[{}/{}]============{}'.format(videoidx,len(test_paths),path))
    test_data = get_online_data(
        opt, spatial_transform, None, target_transform)

    test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)


    results = []
    pred_frames = []
    init_frames = []
    sta_fra = np.zeros(15)
    end_fra = np.zeros(15)
    pre_cla = np.zeros(15)
    feats_time = AverageMeter()

    for i, (inputs, targets) in enumerate(test_loader):
        if not opt.no_cuda:
            targets = targets.cuda(async=True)
        with torch.no_grad():
            inputs = Variable(inputs)
            targets = Variable(targets)
            if opt.modality_clf == 'RGB':
                inputs_clf = inputs[:,:-1,:,:,:]
            elif opt.modality_clf == 'Depth':
                inputs_clf = inputs[:,-1,:,:,:].unsqueeze(1)
            elif opt.modality_clf =='RGB-D':
                inputs_clf = inputs[:,:,:,:,:]

            frame_idx = test_data.data[i]['frame_indices'][-1]
            # feat_path = os.path.join(vid_path,test_data.data[i]['vid_name']+'_{:06d}.pkl'.format(frame_idx))
            feat_path = os.path.join(vid_path,test_data.data[i]['vid_name']+'_{:06d}.mat'.format(frame_idx))
            if os.path.exists(feat_path):
                continue

            s_time = time.time()
            clf_feats = classifier(inputs_clf, phase='features')
            feats_time.update((time.time() - s_time)*1000)
            if i % 500 == 0:
                print('\t[{0}/{1}],\t'
                      'Frame: {2:06d},\t'
                      'seg({3}, {4}),\t'
                      'time: {feats_time.val:.0f}ms ({feats_time.avg:.0f}ms)'.format(
                          i + 1,
                          len(test_loader),
                          frame_idx,
                          test_data.data[i]['segment'][0],
                          test_data.data[i]['segment'][1],
                          feats_time=feats_time))
                sys.stdout.flush()
        clf_feats = clf_feats.data.cpu().numpy()
        sio.savemat(feat_path, {'feats':clf_feats})
        # with open(feat_path, 'wb') as f:
        #     pickle.dump({'feats': clf_feats}, f, pickle.HIGHEST_PROTOCOL)
    # pdb.set_trace()
    print('Finished extraction from frames of: {}'.format(test_data.data[i]['vid_name']))
    print('\n')
print('\n')    
print('-----Feature extraction of {} is finished------'.format(opt.test_subset))
