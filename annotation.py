import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

#-------------------------------------------------------#
# Want to increase the test set modify trainval_percent
# Modify train_percent to change the ratio of validation set 9:1
#
# Currently the library uses test sets as validation sets and does not divide test sets separately
#-------------------------------------------------------#
from config_file import config

trainval_percent    = 1
train_percent       = 0.9
#-------------------------------------------------------#
# Point to the folder where the dataset is located
# Default to dataset in the root directory
#-------------------------------------------------------#
dataset_path      = config["dataset_path"]

if __name__ == "__main__":
    random.seed(0)

    segfilepath     = os.path.join(dataset_path, config["label_data_path"])
    saveBasePath    = os.path.join(dataset_path, config["Image_txt_Sets"])
    
    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".tif"):
            total_seg.append(seg)

    num     = len(total_seg)  
    list    = range(num)  
    tv      = int(num*trainval_percent)  
    tr      = int(tv*train_percent)  
    trainval= random.sample(list,tv)  
    train   = random.sample(trainval,tr)  

    ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
    ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
    ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
    fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
    
    for i in list:  
        name = total_seg[i][:-4]+'\n'  
        if i in trainval:  
            ftrainval.write(name)  
            if i in train:  
                ftrain.write(name)  
            else:  
                fval.write(name)  
        else:  
            ftest.write(name)  
    
    ftrainval.close()  
    ftrain.close()  
    fval.close()  
    ftest.close()
