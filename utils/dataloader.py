import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from config_file import config
from utils.utils import cvtColor, preprocess_input, read_tif, dataPreprocess


class DeeplabDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path, colorDict_GRAY):
        super(DeeplabDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path
        self.colorDict_GRAY = colorDict_GRAY

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]
        _, _, im_data = read_tif(os.path.join(os.path.join(self.dataset_path, config["source_image_path"]), name + ".tif"))
        _, _, label_data = read_tif(os.path.join(os.path.join(self.dataset_path, config["label_data_path"]), name + ".tif"))
        if (len(label_data.shape) == 3):
            label_data = label_data.swapaxes(1, 0)
            label_data = label_data.swapaxes(1, 2)
            label_data = cv2.cvtColor(label_data, cv2.COLOR_RGB2GRAY)
        img, png = dataPreprocess(im_data, label_data, self.num_classes, self.colorDict_GRAY)
        png[png >= self.num_classes] = self.num_classes  # 不是num_classes类，则多加一类
        seg_labels = np.eye(self.num_classes)[np.array(png, dtype=np.int).reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[1]), int(self.input_shape[0]), self.num_classes))
        return img, png, seg_labels

def deeplab_dataset_collate(batch):
    images      = []
    pngs        = []
    seg_labels  = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs        = torch.from_numpy(np.array(pngs)).long()
    seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels
