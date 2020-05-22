import argparse
import time
import os
import sys
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
import itertools
import torch
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from torch.nn import functional as F

from opts import parse_opts_offline
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import *
from temporal_transforms import *
# from temporal_transforms_adap import *
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set, get_online_data
from utils import Logger
from train import train_epoch
from validation import val_epoch
import test
from utils import AverageMeter, calculate_precision, calculate_recall
import pdb
from sklearn.metrics import confusion_matrix

def plot_cm(cm, classes, normalize = True):
    import seaborn as sns
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    ax= plt.subplot()
    sns.heatmap(cm, annot=False, ax = ax); #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='horizontal')
    


def calculate_accuracy(outputs, targets, topk=(1,)):
    maxk = max(topk)
    batch_size = targets.size(0)
    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    ret = []
    for k in topk:
        correct_k = correct[:k].float().sum().item()
        ret.append(correct_k / batch_size)

    return ret


opt = parse_opts_offline()
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
opt.store_name = '{}_{}'.format(opt.store_name, opt.arch)
opt.mean = get_mean(opt.norm_value)
opt.std = get_std(opt.norm_value)

torch.manual_seed(opt.manual_seed)

model, parameters = generate_model(opt)
pytorch_total_params = sum(p.numel() for p in model.parameters() if
                           p.requires_grad)

if opt.no_mean_norm and not opt.std_norm:
    norm_method = Normalize([0, 0, 0], [1, 1, 1])
elif not opt.std_norm:
    norm_method = Normalize(opt.mean, [1, 1, 1])
else:
    norm_method = Normalize(opt.mean, opt.std)


spatial_transform = Compose([
    #Scale(opt.sample_size),
    Scale(112),
    CenterCrop(112),
    ToTensor(opt.norm_value), norm_method
    ])
if opt.true_valid:
    test_batch = 1
    opt.batch_size = test_batch
    temporal_transform = Compose([
        TemporalBeginCrop(opt.sample_duration)
        ])
else:
    test_batch = opt.batch_size
    temporal_transform = Compose([
        TemporalCenterCrop(opt.sample_duration)
        ])
target_transform = ClassLabel()
test_data = get_test_set(
    opt, spatial_transform, temporal_transform, target_transform)

test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=test_batch,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
test_logger = Logger(os.path.join(opt.result_path, 'test-{}_{}.log'.format(opt.test_subset,opt.store_name)), 
    [ 'top1', 'top5', 'precision', 'recall', 'time', 'cm', 'class_names', 'y_true', 'y_pred'])

print(opt)
with open(os.path.join(opt.result_path, 'optsTest_{}.json'.format(opt.store_name)), 'w') as opt_file:
    json.dump(vars(opt), opt_file)
print(model)
sys.stdout.flush()
print("Total number of trainable parameters: ", pytorch_total_params)

if opt.resume_path:
    print('loading checkpoint {}'.format(opt.resume_path))
    checkpoint = torch.load(opt.resume_path)
    assert opt.arch == checkpoint['arch']

    opt.begin_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])


#test.test(test_loader, model, opt, test_data.class_names)
sys.stdout.flush()
recorder = []

print('run with {} samples'.format(len(test_loader.dataset.data)))
sys.stdout.flush()
model.eval()

batch_time = AverageMeter()
top1 = AverageMeter()
top5 = AverageMeter()
precisions = AverageMeter() #
recalls = AverageMeter()

y_true = []
y_pred = []
inst_ids = list(range(len(test_loader.dataset.data)))
inst_ids.append(-1)
out_queue = []
for i, (inputs, targets) in enumerate(test_loader):
    if not opt.no_cuda:
        targets = targets.cuda(async=True)
        inputs = inputs.cuda()
    #inputs = Variable(torch.squeeze(inputs), volatile=True)
    end_time = time.time()
    with torch.no_grad():
        inputs = Variable(inputs)
        targets = Variable(targets)
        outputs = model(inputs)
        if not opt.no_softmax_in_test:
            outputs = F.softmax(outputs)
        recorder.append(outputs.data.cpu().numpy().copy())
    out_queue.append(outputs.cpu().numpy()[0].reshape(-1,))

    #outputs = torch.unsqueeze(torch.mean(outputs, 0), 0)
    # pdb.set_trace()
    # print(outputs.shape, targets.shape)
    if outputs.size(1) <= 4:

        if not opt.true_valid:
            batch_time.update(time.time() - end_time)
            prec1= calculate_accuracy(outputs, targets, topk=(1,))
            precision = calculate_precision(outputs, targets) #
            recall = calculate_recall(outputs,targets)
            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(outputs.argmax(1).cpu().numpy().tolist())

            top1.update(prec1[0], inputs.size(0))
            precisions.update(precision, inputs.size(0))
            recalls.update(recall,inputs.size(0))

            print('[{0}/{1}]\t'
                  'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
                  'acc@1 {top1.avg:.5f} \t'
                  'precision {precision.val:.5f} ({precision.avg:.5f})\t'
                  'recall {recall.val:.5f} ({recall.avg:.5f})'.format(
                      i + 1,
                      len(test_loader),
                      batch_time=batch_time,
                      top1 =top1,
                      precision = precisions,
                      recall = recalls))

        elif inst_ids[i+1] != inst_ids[i]:
            batch_time.update(time.time() - end_time)
            output = np.mean(out_queue,0)
            if opt.clf_threshold > 0.1 and output.max(0) < opt.clf_threshold:
                output = np.append(output, 1.0)
            outputs = torch.from_numpy(output).float().unsqueeze_(0).cuda()
            prec1= calculate_accuracy(outputs, targets, topk=(1,))
            precision = calculate_precision(outputs, targets) #
            recall = calculate_recall(outputs,targets)
            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(outputs.argmax(1).cpu().numpy().tolist())

            top1.update(prec1[0], inputs.size(0))
            precisions.update(precision, inputs.size(0))
            recalls.update(recall,inputs.size(0))

            out_queue = []

            print('[{0}/{1}]\t'
                  'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
                  'acc@1 {top1.avg:.5f} \t'
                  '{5}, '
                  'true, {2}, pred, {3},\t'
                  'score, {4:.2f},\t'
                  '{6},\t'
                  '{7},\t'.format(
                      i + 1,
                      len(test_loader),
                      targets.cpu().numpy().tolist()[0],
                      outputs.argmax(1).cpu().numpy().tolist()[0],
                      outputs.max(1)[0].cpu().numpy().tolist()[0],
                      test_data.data[i]['vid_name'],
                      test_data.data[i]['segment'][0],
                      test_data.data[i]['segment'][1],
                      batch_time=batch_time,
                      top1 =top1))
    else:
        if not opt.true_valid:
            batch_time.update(time.time() - end_time)
            prec1, prec5 = calculate_accuracy(outputs, targets, topk=(1,5))
            precision = calculate_precision(outputs, targets) #
            recall = calculate_recall(outputs,targets)
            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(outputs.argmax(1).cpu().numpy().tolist())

            top1.update(prec1, inputs.size(0))
            top5.update(prec5, inputs.size(0))
            precisions.update(precision, inputs.size(0))
            recalls.update(recall,inputs.size(0))

            print('[{0}/{1}]\t'
                  'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
                  'acc@1 {top1.avg:.5f} acc@5 {top5.avg:.5f}\t'
                  'precision {precision.val:.5f} ({precision.avg:.5f})\t'
                  'recall {recall.val:.5f} ({recall.avg:.5f})'.format(
                      i + 1,
                      len(test_loader),
                      batch_time=batch_time,
                      top1 =top1,
                      top5=top5,
                      precision = precisions,
                      recall = recalls))
            sys.stdout.flush()

        elif inst_ids[i+1] != inst_ids[i]:
            batch_time.update(time.time() - end_time)
            output = np.mean(out_queue,0)
            if opt.clf_threshold > 0.1 and output.max(0) < opt.clf_threshold:
                output = np.append(output, 1.0)
            outputs = torch.from_numpy(output).float().unsqueeze_(0).cuda()
            prec1, prec5 = calculate_accuracy(outputs, targets, topk=(1,5))
            precision = calculate_precision(outputs, targets) #
            recall = calculate_recall(outputs,targets)
            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(outputs.argmax(1).cpu().numpy().tolist())

            top1.update(prec1, inputs.size(0))
            top5.update(prec5, inputs.size(0))
            precisions.update(precision, inputs.size(0))
            recalls.update(recall,inputs.size(0))

            # pdb.set_trace()
            out_queue = []

            print('[{0}/{1}],\t'
                  'Time {batch_time.val:.5f} ({batch_time.avg:.5f}),\t'
                  'acc@1 {top1.avg:.5f}, acc@5 {top5.avg:.5f},\t'
                  '{5}, '
                  'true, {2}, pred, {3},\t'
                  'score, {4:.2f},\t'
                  '{6},\t'
                  '{7},\t'.format(
                      i + 1,
                      len(test_loader),
                      targets.cpu().numpy().tolist()[0],
                      outputs.argmax(1).cpu().numpy().tolist()[0],
                      outputs.max(1)[0].cpu().numpy().tolist()[0],
                      test_data.data[i]['vid_name'],
                      test_data.data[i]['segment'][0],
                      test_data.data[i]['segment'][1],
                      batch_time=batch_time,
                      top1 =top1,
                      top5=top5))
            sys.stdout.flush()
    # pdb.set_trace()

print('-----Evaluation is finished------')
print('Overall Acc@1 {:.03f}% Acc@5 {:.03f}%  Avg Time {:.02f}ms'.format(top1.avg*100, top5.avg*100, batch_time.avg*1000))
cm = confusion_matrix(y_true, y_pred)
class_names = [x for x in test_data.class_names.values()]
print(class_names)
print(cm)
test_logger.log({
        'top1': top1.avg,
        'top5': top5.avg,
        'precision':precisions.avg,
        'recall':recalls.avg,
        'time':batch_time.avg,
        'cm':cm,
        'class_names':class_names,
        'y_true':y_true,
        'y_pred':y_pred
    })