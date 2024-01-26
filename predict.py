import time

import cv2
import numpy as np
from PIL import Image
from osgeo import gdal
import os
from config_file import config
from deeplab import DeeplabV3
from utils.utils import TifCroppingArray, Result, write_tif, read_tif, color_dict, writeTiff, merge_img

if __name__ == "__main__":
    num_classes = config["num_classes"]
    dataset_path = config["dataset_path"]
    train_label_path = os.path.join(dataset_path, config["label_data_path"])
    colorDict_RGB, colorDict_GRAY = color_dict(train_label_path, num_classes)

    deeplab = DeeplabV3()
    #----------------------------------------------------------------------------------------------------------#
    #   'predict'           Represents a single image prediction
    #----------------------------------------------------------------------------------------------------------#
    mode = "predict"
    #-------------------------------------------------------------------------#
    name_classes = ["Class " + chr(i + 65) for i in range(num_classes)]
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    dir_origin_path = os.path.join(dataset_path, config["predicted_img_dir"])
    dir_save_path = os.path.join(dataset_path, config["predicted_img_out_dir"])
    input_shape = config["input_shape"]  # [w,h]
    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    if mode == "predict":
        img = config["predicted_img_name"]
        image_path = os.path.join(dir_origin_path,img)
        im_proj, im_Geotrans, image = read_tif(image_path)

        RepetitiveLength = 20
        if RepetitiveLength >= input_shape[0] / 2 or RepetitiveLength >= input_shape[1] / 2:
            print("RepetitiveLength is set too large. Please reset the repetitivelength")
            exit()

        print("Crop the overlap size of the picture:", RepetitiveLength)
        # Call clipping function
        TifArray, RowOver, ColumnOver = TifCroppingArray(image, input_shape, RepetitiveLength)
        _, ih, iw = image.shape

        pre_labelArray = []
        pre_Array = []
        for i in range(len(TifArray)):
            for j in range(len(TifArray[i])):
                image = TifArray[i][j]
                image = image / 255.0
                r_image,image_prb = deeplab.detect_image(image, colorDict_GRAY)
                pre_labelArray.append(r_image)
                pre_Array.append(image_prb)
        # Call merge_img function
        result_data = merge_img((ih, iw), TifArray, pre_labelArray, input_shape, RepetitiveLength, RowOver, ColumnOver)
        image_prb_data = merge_img((ih, iw), TifArray, pre_Array, input_shape, RepetitiveLength, RowOver, ColumnOver)

        result_png = Image.fromarray(result_data)
        result_png.save(os.path.join(dir_save_path , config["predicted_img_out_name"]+".png"))

        if len(result_data.shape) == 3:
            result_tif = result_data.swapaxes(1, 2)
            result_tif = result_tif.swapaxes(0, 1)
        else:
            result_tif = result_data
            image_prb_tif = image_prb_data

        writeTiff(os.path.join(dir_save_path , config["predicted_img_out_name"]+".tif"), result_tif, im_Geotrans, im_proj)
        writeTiff(os.path.join(dir_save_path, config["predicted_img_prb_out_name"] + ".tif"), image_prb_tif, im_Geotrans,
                  im_proj)

        print("After running, go to the {} folder to view the file.".format(dir_save_path))
        
    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = deeplab.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
    elif mode == "export_onnx":
        deeplab.convert_to_onnx(simplify, onnx_save_path)
        
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
