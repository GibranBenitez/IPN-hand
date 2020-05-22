import torch
from torch import nn
import pdb

from models import resnet, resnext, resnetl, c3d, mobilenetv2, shufflenetv2


def generate_model(opt):
    assert opt.model in [
        'resnet', 'resnetl', 'resnext', 'c3d', 'mobilenetv2', 'shufflenetv2'
    ]

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 50]

        from models.resnet import get_fine_tuning_parameters

        if opt.model_depth == 10:
            model = resnet.resnet10(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'resnetl':
        assert opt.model_depth in [10, 18]

        from models.resnetl import get_fine_tuning_parameters

        if opt.model_depth == 10:
            model = resnetl.resnetl10(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 18:
            model = resnetl.resnetl10(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        
    elif opt.model == 'resnext':
        assert opt.model_depth in [101]

        from models.resnext import get_fine_tuning_parameters

        if opt.model_depth == 101:
            model = resnext.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'c3d':
        assert opt.model_depth in [10]

        from models.c3d import get_fine_tuning_parameters
        if opt.model_depth == 10:

            model = c3d.c3d_v1(
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                num_classes=opt.n_classes)
    elif opt.model == 'mobilenetv2':

        from models.mobilenetv2 import get_fine_tuning_parameters
        model = mobilenetv2.mob_v2(
            num_classes=opt.n_classes,
            sample_size=opt.sample_size,
            width_mult=opt.width_mult)
    elif opt.model == 'shufflenetv2':

        from models.shufflenetv2 import get_fine_tuning_parameters
        model = shufflenetv2.shf_v2(
            num_classes=opt.n_classes,
            sample_size=opt.sample_size,
            width_mult=opt.width_mult)
    
    if not opt.no_cuda:
        model = nn.DataParallel(model, device_ids=None)

        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            assert opt.arch == pretrain['arch']
            if opt.pretrain_dataset == 'jester':
                if opt.sample_duration < 32 and opt.model != 'c3d':
                    model = _modify_first_conv_layer(model,3,3)
                if opt.model in  ['mobilenetv2', 'shufflenetv2']:
                    del pretrain['state_dict']['module.classifier.1.weight']
                    del pretrain['state_dict']['module.classifier.1.bias']
                else:
                    del pretrain['state_dict']['module.fc.weight']
                    del pretrain['state_dict']['module.fc.bias']
                model.load_state_dict(pretrain['state_dict'],strict=False)

        if opt.modality in ['RGB', 'flo'] and opt.model != 'c3d':
            print("[INFO]: RGB model is used for init model")
            if opt.dataset != 'jester' and not opt.no_first_lay:
                model = _modify_first_conv_layer(model,3,3) ##### Check models trained (3,7,7) or (7,7,7)
        elif opt.modality in ['Depth', 'seg']:
            print("[INFO]: Converting the pretrained model to Depth init model")
            model = _construct_depth_model(model)
            print("[INFO]: Done. Flow model ready.")
        elif opt.modality in ['RGB-D', 'RGB-flo', 'RGB-seg']:
            print("[INFO]: Converting the pretrained model to RGB+D init model")
            model = _construct_rgbdepth_model(model)
            if opt.no_first_lay:
                model = _modify_first_conv_layer(model,3,4) ##### Check models trained (3,7,7) or (7,7,7)
            print("[INFO]: Done. RGB-D model ready.")
        if opt.pretrain_dataset == opt.dataset:
            model.load_state_dict(pretrain['state_dict'])
        elif opt.pretrain_dataset in ['egogesture', 'nv', 'denso']:
            del pretrain['state_dict']['module.fc.weight']
            del pretrain['state_dict']['module.fc.bias']
            model.load_state_dict(pretrain['state_dict'],strict=False)

        # Check first kernel size 
        modules = list(model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv3d),
                                                   list(range(len(modules)))))[0]

        conv_layer = modules[first_conv_idx]
        if conv_layer.kernel_size[0]> opt.sample_duration:
            print("[INFO]: RGB model is used for init model")
            model = _modify_first_conv_layer(model,int(opt.sample_duration/2),1) 


        if opt.model == 'c3d':# CHECK HERE
            model.module.fc = nn.Linear(
                model.module.fc[0].in_features, model.module.fc[0].out_features)
            model.module.fc = model.module.fc.cuda()
        elif opt.model in  ['mobilenetv2', 'shufflenetv2']:
            model.module.classifier = nn.Sequential(
                nn.Dropout(0.9),
                nn.Linear(model.module.classifier[1].in_features, opt.n_finetune_classes))
            model.module.classifier = model.module.classifier.cuda()
        else:
            model.module.fc = nn.Linear(model.module.fc.in_features,
                                        opt.n_finetune_classes)
            model.module.fc = model.module.fc.cuda()

        parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
        model = model.cuda()
        return model, parameters

    else:
        print('ERROR no cuda')

    return model, model.parameters()

def _construct_depth_model(base_model):
    # modify the first convolution kernels for Depth input
    modules = list(base_model.modules())

    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv3d),
                                 list(range(len(modules)))))[0]
    conv_layer = modules[first_conv_idx]
    container = modules[first_conv_idx - 1]

    # modify parameters, assume the first blob contains the convolution kernels
    motion_length = 1
    params = [x.clone() for x in conv_layer.parameters()]
    kernel_size = params[0].size()
    new_kernel_size = kernel_size[:1] + (1*motion_length,  ) + kernel_size[2:]
    new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()




    new_conv = nn.Conv3d(1, conv_layer.out_channels, conv_layer.kernel_size, conv_layer.stride,
                         conv_layer.padding, bias=True if len(params) == 2 else False)
    new_conv.weight.data = new_kernels
    if len(params) == 2:
        new_conv.bias.data = params[1].data # add bias if neccessary
    layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

    # replace the first convlution layer
    setattr(container, layer_name, new_conv)

    return base_model

def _construct_rgbdepth_model(base_model):
    # modify the first convolution kernels for RGB-D input
    modules = list(base_model.modules())

    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv3d),
                           list(range(len(modules)))))[0]
    conv_layer = modules[first_conv_idx]
    container = modules[first_conv_idx - 1]
    # modify parameters, assume the first blob contains the convolution kernels
    motion_length = 1
    params = [x.clone() for x in conv_layer.parameters()]
    kernel_size = params[0].size()
    new_kernel_size = kernel_size[:1] + (1 * motion_length,) + kernel_size[2:]
    new_kernels = torch.mul(torch.cat((params[0].data, params[0].data.mean(dim=1,keepdim=True).expand(new_kernel_size).contiguous()), 1), 0.6)
    new_kernel_size = kernel_size[:1] + (3 + 1 * motion_length,) + kernel_size[2:]

    new_conv = nn.Conv3d(4, conv_layer.out_channels, conv_layer.kernel_size, conv_layer.stride,
                         conv_layer.padding, bias=True if len(params) == 2 else False)
    new_conv.weight.data = new_kernels
    if len(params) == 2:
        new_conv.bias.data = params[1].data  # add bias if neccessary
    layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

    # replace the first convolution layer
    setattr(container, layer_name, new_conv)
    return base_model

def _modify_first_conv_layer(base_model, new_kernel_size1, new_filter_num):
    modules = list(base_model.modules())
    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv3d),
                                               list(range(len(modules)))))[0]
    conv_layer = modules[first_conv_idx]
    container = modules[first_conv_idx - 1]
 
    new_conv = nn.Conv3d(new_filter_num, conv_layer.out_channels, kernel_size=(new_kernel_size1,7,7),
                         stride=(1,2,2), padding=(1,3,3), bias=False)
    layer_name = list(container.state_dict().keys())[0][:-7]

    setattr(container, layer_name, new_conv)
    return base_model


