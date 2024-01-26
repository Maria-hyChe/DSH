import os

import numpy as np

from utils.utils import read_tif

if __name__ == '__main__':
    img_file_dir = r"G:\hlj\image_qs"
    label_file_dir = r"G:\hlj\label_qs"

    threshold = 0.90

    label_file_Name_List = os.listdir(label_file_dir)

    for each_file in label_file_Name_List:
        label_file_path = os.path.join(label_file_dir, each_file)
        img_file_path = os.path.join(img_file_dir, each_file)
        _, _, label_data = read_tif(label_file_path)
        white_ratio = np.sum(label_data == 255) / label_data.size
        if white_ratio > threshold:
            print("{} 文件的背景占比为{}，超过设定的阈值{}，文件将被删除！".format(label_file_path,white_ratio,threshold))
            try:
                # 删除label文件
                os.remove(label_file_path)
            except FileNotFoundError:
                print("{} does not exist...".format(label_file_path))
            except:
                print("Unknown error...")

            try:
                # 删除image文件
                os.remove(img_file_path)
            except FileNotFoundError:
                print("{} does not exist...".format(img_file_path))
            except:
                print("Unknown error...")

