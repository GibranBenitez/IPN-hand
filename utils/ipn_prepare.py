import os
import cv2
import numpy as np
import glob
import sys
from subprocess import call

dataset_path = "/host/space0/gibran/dataset/HandGestures/IPN_dataset"
def load_split_nvgesture(file_with_split = './Annot_TestList.txt',list_split = list()):
    file_with_split = os.path.join(dataset_path,file_with_split)
    params_dictionary = dict()
    with open(file_with_split,'rb') as f:
          dict_name  = file_with_split[file_with_split.rfind('/')+1 :]
          dict_name  = dict_name[:dict_name.find('_')]

          for line in f:
            params = line.decode().split(',')
            params_dictionary = dict()

            params_dictionary['dataset'] = dict_name

            for sens in ['frames','segment','flow']:
                    path = os.path.join('./'+sens, params[0])
                    key = sens

                    label = int(params[2]) - 1 
                    params_dictionary['label'] = label

                    #first store path
                    params_dictionary[key] = path
                    #store start frame
                    params_dictionary[key+'_start'] = int(params[3])
                    params_dictionary[key+'_end'] = int(params[4])

            list_split.append(params_dictionary)
 
    return list_split

def create_list(example_config, sensor,  class_types = 'all'):

    folder_path = example_config[sensor]# aqui debe ir nada en +X, porque voy a mneter el pad correcto desde la funcion anterior
    n_images = len(glob.glob(os.path.join(folder_path, '*.jpg')))

    label = example_config['label']+1
    start_frame = example_config[sensor+'_start']
    end_frame = example_config[sensor+'_end']

    
    frame_indices = np.array([[start_frame,end_frame]])
    len_lines = frame_indices.shape[0]
    start = 1
    for i in range(len_lines):
        line = frame_indices[i,:]
        if class_types == 'all':
            new_lines.append(folder_path + ' ' + str(label)+ ' ' + str(line[0])+ ' ' + str(line[1]))
        elif class_types == 'all_but_None':
            if label == 1:
                continue
            new_lines.append(folder_path + ' ' + str(label-1)+ ' ' + str(line[0])+ ' ' + str(line[1]))
        elif class_types == 'binary':
            if label == 1:
                new_lines.append(folder_path + ' ' + str(label)+ ' ' + str(line[0])+ ' ' + str(line[1]))
            else:
                new_lines.append(folder_path + ' ' + '2' + ' ' + str(line[0])+ ' ' + str(line[1]))
        elif class_types == 'group':
            if label < 4:
                new_lines.append(folder_path + ' ' + str(label)+ ' ' + str(line[0])+ ' ' + str(line[1]))
            else:
                new_lines.append(folder_path + ' ' + '4' + ' ' + str(line[0])+ ' ' + str(line[1]))
        elif class_types == 'gests_only':
            if label > 3:
                new_lines.append(folder_path + ' ' + str(label-3)+ ' ' + str(line[0])+ ' ' + str(line[1]))
        start = line[1]+1

    
if __name__ == "__main__":
    subset = sys.argv[1]          # [training, validation] 
    file_name = sys.argv[2]       # [trainlistall.txt, trainlistall_but_None.txt, trainlistbinary.txt, vallistall.txt, ...]
    class_types = sys.argv[3]     # [all, all_but_None, binary, group]

    sensors = ['frames','segment','flow']

    file_lists = dict()
    if subset == 'training':
        file_list = "./Annot_TrainList.txt"
    elif subset == 'validation':
        file_list = "./Annot_TestList.txt"
    

    subset_list = list()
    load_split_nvgesture(file_with_split = file_list,list_split = subset_list)

    for idx in range(len(sensors)):
        new_lines = [] 
        print("Processing Traing List")
        for sample_name in subset_list:
            create_list(example_config = sample_name, sensor = sensors[idx], class_types = class_types)


        print("Writing to the file ...")
        if idx > 0:
            f_name = file_name.split('.')[0] + '_' + sensors[idx][0:3] + '.' + file_name.split('.')[1]
        else:
            f_name = file_name
        file_path = os.path.join('annotation_ipnGesture',f_name)
        with open(file_path, 'w') as myfile:
            for new_line in new_lines:
                myfile.write(new_line)
                myfile.write('\n')
        print("Scuccesfully wrote file to:",file_path)

# HOW TO RUN:
# python utils/ipn_prepare.py validation vallistall_but_None.txt all_but_None
# python utils/ipn_prepare.py validation vallistall.txt all
# python utils/ipn_prepare.py validation vallistbinary.txt binary
# python utils/ipn_prepare.py training trainlistall_but_None.txt all_but_None
# python utils/ipn_prepare.py training trainlistall.txt all
# python utils/ipn_prepare.py training trainlistbinary.txt binary