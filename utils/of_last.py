import os
import pdb
import numpy as np
import glob
import sys
from shutil import copyfile

def create_list(folder_path):

    n_images = len(glob.glob(os.path.join(folder_path, '*.jpg')))
    src = '{:s}_{:06d}.jpg'.format(folder_path.split('/')[-1], n_images)
    dst = '{:s}_{:06d}.jpg'.format(folder_path.split('/')[-1], n_images+1)
    # pdb.set_trace()

    copyfile(os.path.join(folder_path, src), os.path.join(folder_path, dst))

    
if __name__ == "__main__":
    
    list_file = "/host/space0/gibran/dataset/HandGestures/IPN_dataset/ListVideos.txt"
    dataset_path = "/host/space0/gibran/dataset/HandGestures/IPN_dataset/flow"

    with open(list_file,'rb') as f:
        for line in f:
            video = line.decode().rstrip('\n')
            create_list(os.path.join(dataset_path, video))
            print(video)
