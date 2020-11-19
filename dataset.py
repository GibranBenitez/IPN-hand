from datasets.jester import Jester
from datasets.egogesture import EgoGesture
from datasets.nv import NV
from datasets.ipn import IPN
from datasets.nv_online import NVOnline
from datasets.ipn_online import IPNOnline

def get_training_set(opt, spatial_transform, temporal_transform,
                     target_transform):
    assert opt.dataset in ['jester', 'nv', 'ipn']

    if opt.train_validate:
        subset = ['training', 'validation']
    else:
        subset = 'training'
    if opt.dataset == 'jester':
        training_data = Jester(
            opt.video_path,
            opt.annotation_path,
            subset,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            modality=opt.modality)
    elif opt.dataset == 'egogesture':
        training_data = EgoGesture(
            opt.video_path,
            opt.annotation_path,
            subset,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            modality=opt.modality)
    elif opt.dataset == 'nv':
        training_data = NV(
            opt.video_path,
            opt.annotation_path,
            subset,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            modality=opt.modality)
    elif opt.dataset == 'ipn':
        training_data = IPN(
            opt.video_path,
            opt.annotation_path,
            subset,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            modality=opt.modality)
    elif opt.dataset == 'denso':
        training_data = Denso(
            opt.video_path,
            opt.annotation_path,
            subset,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            modality=opt.modality,
            no_subject_crop=opt.no_scrop)
    elif opt.dataset == 'AHG':
        training_data = AHG(
            opt.video_path,
            opt.annotation_path,
            subset,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            modality=opt.modality,
            no_subject_crop=opt.no_scrop)
    return training_data


def get_validation_set(opt, spatial_transform, temporal_transform,
                       target_transform):
    assert opt.dataset in ['jester', 'nv', 'ipn']

    if opt.dataset == 'jester':
        validation_data = Jester(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            modality=opt.modality,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'egogesture':
        validation_data = EgoGesture(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            modality=opt.modality,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'nv':
        validation_data = NV(
            opt.video_path,
            opt.annotation_path,
            'validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            modality=opt.modality)
    elif opt.dataset == 'ipn':
        validation_data = IPN(
            opt.video_path,
            opt.annotation_path,
            'validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            modality=opt.modality)
    elif opt.dataset == 'denso':
        validation_data = Denso(
            opt.video_path,
            opt.annotation_path,
            'validation',
            true_valid=opt.true_valid,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            modality=opt.modality,
            no_subject_crop=opt.no_scrop)
    elif opt.dataset == 'AHG':
        validation_data = AHG(
            opt.video_path,
            opt.annotation_path,
            'validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            modality=opt.modality,
            no_subject_crop=opt.no_scrop)
    return validation_data


def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.dataset in ['jester', 'nv', 'ipn']
    assert opt.test_subset in ['val', 'test']

    if opt.test_subset == 'val':
        subset = 'validation'
    else:
        subset = 'testing'

    if opt.dataset == 'jester':
        test_data = Jester(
            opt.video_path,
            opt.annotation_path,
            subset,
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            modality=opt.modality,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'egogesture':
        test_data = EgoGesture(
            opt.video_path,
            opt.annotation_path,
            subset,
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            modality=opt.modality,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'nv':
        test_data = NV(
            opt.video_path,
            opt.annotation_path,
            'validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            modality=opt.modality)
    elif opt.dataset == 'ipn':
        test_data = IPN(
            opt.video_path,
            opt.annotation_path,
            'validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            modality=opt.modality)
    elif opt.dataset == 'denso':
        test_data = Denso(
            opt.video_path,
            opt.annotation_path,
            subset,
            true_valid=opt.true_valid,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            modality=opt.modality,
            no_subject_crop=opt.no_scrop)
    elif opt.dataset == 'AHG':
        test_data = AHG(
            opt.video_path,
            opt.annotation_path,
            subset,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            modality=opt.modality,
            no_subject_crop=opt.no_scrop)
    return test_data

def get_online_data(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.dataset in [ 'nv', 'ipn']
    whole_path = opt.whole_path
    if opt.dataset == 'egogesture':
        online_data = EgoGestureOnline(
            opt.annotation_path,  
            opt.video_path,
            opt.whole_path,  
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            modality=opt.modality,
            stride_len = opt.stride_len,
            sample_duration=opt.sample_duration)
    if opt.dataset == 'nv':
        online_data = NVOnline(
            opt.annotation_path,  
            opt.video_path,
            opt.whole_path,  
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            modality=opt.modality,
            stride_len = opt.stride_len,
            sample_duration=opt.sample_duration)
    if opt.dataset == 'ipn':
        online_data = IPNOnline(
            opt.annotation_path,  
            opt.video_path,
            opt.whole_path,  
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            modality=opt.modality,
            stride_len = opt.stride_len,
            sample_duration=opt.sample_duration)
    if opt.dataset == 'AHG':
        fill_ = True if opt.model_clf == 'c3d' else False
        online_data = AHGOnline(
            opt.annotation_path,  
            opt.video_path,
            opt.whole_path,  
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            modality=opt.modality,
            stride_len = opt.stride_len,
            fill = fill_,
            sample_duration=opt.sample_duration)
    if opt.dataset == 'denso':
        fill_ = True if opt.model_clf == 'c3d' else False
        online_data = densOnline(
            opt.annotation_path,  
            opt.video_path,
            opt.whole_path,  
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            modality=opt.modality,
            stride_len = opt.stride_len,
            fill = fill_,
            no_subject_crop=opt.no_scrop,
            sample_duration=opt.sample_duration)
    
    return online_data
