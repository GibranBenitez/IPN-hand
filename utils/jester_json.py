from __future__ import print_function, division
import os
import sys
import json
import pandas as pd
import glob

def convert_csv_to_dict(csv_path, subset, labels, vid_data):
    data = pd.read_csv(csv_path, delimiter=' ', header=None)
    keys = []
    key_labels = []
    for i in range(data.shape[0]):
        row = data.ix[i, :]
        class_name = labels[row[1]-1]
        basename = str(row[0])
        
        keys.append(basename)
        key_labels.append(class_name)
        
    database = {}
    for i in range(len(keys)):
        key = keys[i]
        video_path = os.path.join(vid_data, key)
        n_frames = len(glob.glob(os.path.join(video_path, '*.jpg')))
        database[key] = {}
        database[key]['subset'] = subset
        label = key_labels[i]
        database[key]['annotations'] = {'label': label, 'end_frame': n_frames}
    
    return database

def load_labels(label_csv_path):
    data = pd.read_csv(label_csv_path, delimiter=' ', header=None)
    labels = []
    for i in range(data.shape[0]):
        labels.append(data.ix[i, 1])
    return labels

def convert_jester_csv_to_activitynet_json(label_csv_path, train_csv_path, 
                                           val_csv_path, dst_json_path, vid_data):
    labels = load_labels(label_csv_path)
    train_database = convert_csv_to_dict(train_csv_path, 'training', labels, vid_data)
    val_database = convert_csv_to_dict(val_csv_path, 'validation', labels, vid_data)
    
    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    dst_data['database'].update(train_database)
    dst_data['database'].update(val_database)

    with open(dst_json_path, 'w') as dst_file:
        json.dump(dst_data, dst_file)

if __name__ == '__main__':
    csv_dir_path = sys.argv[1]
    vid_data = '/export/space0/gibran/dataset/20BN-jester/20bn-jester-v1'

    label_csv_path = os.path.join(csv_dir_path, 'classInd.txt')
    train_csv_path = os.path.join(csv_dir_path, 'trainlist01.txt')
    val_csv_path = os.path.join(csv_dir_path, 'vallist01.txt')
    dst_json_path = os.path.join(csv_dir_path, 'jester.json')

    convert_jester_csv_to_activitynet_json(label_csv_path, train_csv_path,
                                               val_csv_path, dst_json_path, vid_data)
