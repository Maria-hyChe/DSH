import os

import cv2
import numpy as np
import pandas as pd

from config_file import config
from utils.utils import color_dict, read_tif
from utils.utils_metrics import compute_mIoU

if __name__ == '__main__':

    num_classes = config["num_classes"]
    name_classes = ["Class "+chr(i+65) for i in range(num_classes)]

    dataset_path = config["dataset_path"]
    colorDict_RGB, colorDict_GRAY = color_dict(os.path.join(dataset_path, config["predicted_img_label_dir"]),
                                               num_classes)

    gt_dir = os.path.join(dataset_path, config["predicted_img_label_dir"])
    miou_out_path = "miou_out"
    pred_dir = os.path.join(dataset_path, config["predicted_img_out_dir"])
    image_name = "img"
    hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_name, num_classes,colorDict_GRAY, name_classes)  # 执行计算mIoU的函数
    print(hist.shape)
    f_dict = {}
    for i,each in enumerate(name_classes):
        f_dict[each]= hist[:,i]
    df = pd.DataFrame(f_dict)
    df.to_csv('./logs/confusion_matrix_voting_label.csv', index=False)
    print("Get miou done.")
