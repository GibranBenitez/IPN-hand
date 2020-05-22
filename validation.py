import torch
from torch.autograd import Variable
from torch.nn import functional as F
import time
import sys
import pdb
import numpy as np

from utils import AverageMeter, calculate_accuracy, calculate_precision, calculate_recall, Queue


def val_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    precisions = AverageMeter() #
    recalls = AverageMeter()
    printer = 50 if opt.dataset == 'jester' else 5

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda(async=True)
        with torch.no_grad():
            inputs = Variable(inputs)
            targets = Variable(targets)
            outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)
        precision = calculate_precision(outputs, targets) #
        recall = calculate_recall(outputs,targets)

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        precisions.update(precision, inputs.size(0))
        recalls.update(recall,inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if i % printer ==0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
                  'Data {data_time.val:.5f} ({data_time.avg:.5f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch,
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      acc=accuracies))

    logger.log({'epoch': epoch,
                'loss': losses.avg, 
                'acc': accuracies.avg,
                'precision':precisions.avg,
                'recall':recalls.avg
                })

    return losses.avg, accuracies.avg

def val_epoch_true(epoch, data_loader, model, criterion, opt, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    precisions = AverageMeter() #
    recalls = AverageMeter()

    end_time = time.time()
    vid_ids = data_loader.dataset.ids
    vid_ids.append(-1)
    out_queue = []
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda(async=True)
        with torch.no_grad():
            inputs = Variable(inputs)
            targets = Variable(targets)
            outputs = model(inputs)
            outputs = F.softmax(outputs,dim=1)
            outputs = outputs.cpu().numpy()[0].reshape(-1,)
            out_queue.append(outputs)

        if vid_ids[i+1] != vid_ids[i]:
          output = np.mean(out_queue,0)
          outputs = torch.from_numpy(output).float().unsqueeze_(0).cuda()
          loss = criterion(outputs, targets)
          acc = calculate_accuracy(outputs, targets)
          precision = calculate_precision(outputs, targets)
          recall = calculate_recall(outputs,targets)

          losses.update(loss.item(), inputs.size(0))
          accuracies.update(acc, inputs.size(0))
          precisions.update(precision, inputs.size(0))
          recalls.update(recall,inputs.size(0))

          batch_time.update(time.time() - end_time)
          end_time = time.time()
          out_queue = []

          if i % 100 ==0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
                  'Data {data_time.val:.5f} ({data_time.avg:.5f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch,
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      acc=accuracies))

    logger.log({'epoch': epoch,
                'loss': losses.avg, 
                'acc': accuracies.avg,
                'precision':precisions.avg,
                'recall':recalls.avg
                })

    return losses.avg, accuracies.avg