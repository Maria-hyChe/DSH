import os
from typing import List, Optional
import  cv2
import numpy as np
from PIL import Image

from numpy import ndarray
from osgeo import gdal
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 


def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh
    

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def preprocess_input(image):
    image /= 255.0
    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        'mobilenet' : 'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar',
        'xception'  : 'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/xception_pytorch_imagenet.pth',
    }
    url = download_urls[backbone]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)

def color_dict(labelFolder, classNum):
    colorDict: List[Optional[ndarray]] = []
    #  获取文件夹内的文件名
    ImageNameList = os.listdir(labelFolder)
    for i in range(len(ImageNameList)):
        ImagePath = labelFolder + "/" + ImageNameList[i]
        img = cv2.imread(ImagePath).astype(np.uint32)

        if (len(img.shape) == 2):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.uint32)
        img_new = img[:, :, 0] * 1000000 + img[:, :, 1] * 1000 + img[:, :, 2]
        unique = np.unique(img_new)

        for j in range(unique.shape[0]):
            colorDict.append(unique[j])

        colorDict = sorted(set(colorDict))

        if (len(colorDict) == classNum):
            break

    colorDict_RGB = []
    for k in range(len(colorDict)):
        color = str(colorDict[k]).rjust(9, '0')
        color_RGB = [int(color[0: 3]), int(color[3: 6]), int(color[6: 9])]
        colorDict_RGB.append(color_RGB)
    colorDict_RGB = np.array(colorDict_RGB)
    colorDict_GRAY = colorDict_RGB.reshape((colorDict_RGB.shape[0], 1, colorDict_RGB.shape[1])).astype(np.uint8)
    colorDict_GRAY = cv2.cvtColor(colorDict_GRAY, cv2.COLOR_BGR2GRAY)
    return colorDict_RGB, colorDict_GRAY



def saveResult(test_image_path, test_predict_path, model_predict, color_dict, output_size):
    imageList = os.listdir(test_image_path)
    for i, img in enumerate(model_predict):
        channel_max = np.argmax(img, axis=-1)
        img_out = np.uint8(color_dict[channel_max.astype(np.uint8)])
        img_out = cv2.resize(img_out, (output_size[0], output_size[1]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(test_predict_path + "\\" + imageList[i][:-4] + ".png", img_out)


def read_tif(file_name):
    dataset = gdal.Open(file_name)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    im_proj = (dataset.GetProjection())
    im_Geotrans = (dataset.GetGeoTransform())
    im_data = dataset.ReadAsArray(0, 0, width, height)
    del dataset
    return im_proj, im_Geotrans, im_data

def writeTiff(img_path, im_data, im_geotrans, im_proj):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(img_path, im_width, im_height, im_bands, datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


def write_tif(newpath, im_data, im_Geotrans, im_proj, datatype):
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
    diver = gdal.GetDriverByName('GTiff')
    new_dataset = diver.Create(newpath, im_width, im_height, im_bands, datatype)
    new_dataset.SetGeoTransform(im_Geotrans)
    new_dataset.SetProjection(im_proj)

    if im_bands == 1:
        new_dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            new_dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del new_dataset


def TifCroppingArray(img, size, SideLength):
    _, ih, iw = img.shape
    w, h = size
    TifArrayReturn = []
    ColumnNum = int((ih - SideLength * 2) / (h - SideLength * 2))
    RowNum = int((iw - SideLength * 2) / (w - SideLength * 2))

    for i in range(ColumnNum):
        TifArray = []
        for j in range(RowNum):
            cropped = img[:, i * (h - SideLength * 2): i * (h - SideLength * 2) + h,
                      j * (w - SideLength * 2): j * (w - SideLength * 2) + w]
            TifArray.append(cropped)
        TifArrayReturn.append(TifArray)

    for i in range(ColumnNum):
        cropped = img[:, i * (h - SideLength * 2): i * (h - SideLength * 2) + h,
                  (iw - w): iw]
        TifArrayReturn[i].append(cropped)
    TifArray = []
    for j in range(RowNum):
        cropped = img[:, (ih - h): ih,
                  j * (w - SideLength * 2): j * (w - SideLength * 2) + w]
        TifArray.append(cropped)

    cropped = img[:, (ih - h): ih, (iw - w): iw]
    TifArray.append(cropped)
    TifArrayReturn.append(TifArray)

    ColumnOver = (ih - SideLength * 2) % (h - SideLength * 2) + SideLength

    RowOver = (iw - SideLength * 2) % (w - SideLength * 2) + SideLength
    return TifArrayReturn, RowOver, ColumnOver



def Result(img_shape, TifArray, npyfile, size, RepetitiveLength, RowOver, ColumnOver):
    result = np.zeros(img_shape + (3,), np.uint8)
    w, h = size
    j = 0
    for i, item in enumerate(npyfile):
        img = item
        if (i % len(TifArray[0]) == 0):
            if (j == 0):
                result[0: h - RepetitiveLength, 0: w - RepetitiveLength, :] = img[0: h - RepetitiveLength,
                                                                              0: w - RepetitiveLength, :]
            elif (j == len(TifArray) - 1):
                result[img_shape[0] - ColumnOver - RepetitiveLength: img_shape[0], 0: w - RepetitiveLength, :] = img[
                                                                                                                 h - ColumnOver - RepetitiveLength: h,
                                                                                                                 0: w - RepetitiveLength,
                                                                                                                 :]
            else:
                result[j * (h - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                        h - 2 * RepetitiveLength) + RepetitiveLength,
                0:w - RepetitiveLength, :] = img[RepetitiveLength: h - RepetitiveLength, 0: w - RepetitiveLength, :]
        elif (i % len(TifArray[0]) == len(TifArray[0]) - 1):
            if (j == 0):
                result[0: h - RepetitiveLength, img_shape[1] - RowOver: img_shape[1], :] = img[0: h - RepetitiveLength,
                                                                                           w - RowOver: w, :]
            elif (j == len(TifArray) - 1):
                result[img_shape[0] - ColumnOver: img_shape[0], img_shape[1] - RowOver: img_shape[1], :] = img[
                                                                                                           h - ColumnOver: h,
                                                                                                           w - RowOver: w,
                                                                                                           :]
            else:
                result[j * (h - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                        h - 2 * RepetitiveLength) + RepetitiveLength,
                img_shape[1] - RowOver: img_shape[1], :] = img[RepetitiveLength: h - RepetitiveLength, w - RowOver: w,
                                                           :]
            j = j + 1
        else:
            if (j == 0):
                result[0: h - RepetitiveLength,
                (i - j * len(TifArray[0])) * (w - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                    TifArray[0]) + 1) * (w - 2 * RepetitiveLength) + RepetitiveLength
                , :] = img[0: h - RepetitiveLength, RepetitiveLength: w - RepetitiveLength, :]
            if (j == len(TifArray) - 1):
                result[img_shape[0] - ColumnOver: img_shape[0],
                (i - j * len(TifArray[0])) * (w - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                    TifArray[0]) + 1) * (w - 2 * RepetitiveLength) + RepetitiveLength
                , :] = img[h - ColumnOver: h, RepetitiveLength: w - RepetitiveLength, :]
            else:
                result[j * (h - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                        h - 2 * RepetitiveLength) + RepetitiveLength,
                (i - j * len(TifArray[0])) * (w - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                    TifArray[0]) + 1) * (w - 2 * RepetitiveLength) + RepetitiveLength, :] = img[
                                                                                            RepetitiveLength: h - RepetitiveLength,
                                                                                            RepetitiveLength: w - RepetitiveLength,
                                                                                            :]
    return result


def merge_img(img_shape, TifArray, npyfile, size, RepetitiveLength, RowOver, ColumnOver):
    data_type = npyfile[0].dtype
    if data_type == "uint8":
        result = np.zeros(img_shape, np.uint8)
    else:
        result = np.zeros(img_shape)
    w, h = size
    j = 0
    for i, item in enumerate(npyfile):
        img = item
        if (i % len(TifArray[0]) == 0):
            if (j == 0):
                result[0: h - RepetitiveLength, 0: w - RepetitiveLength] = img[0: h - RepetitiveLength,
                                                                              0: w - RepetitiveLength]
            elif (j == len(TifArray) - 1):
                result[img_shape[0] - ColumnOver - RepetitiveLength: img_shape[0], 0: w - RepetitiveLength] = img[h - ColumnOver - RepetitiveLength: h,
                                                                                                                 0: w - RepetitiveLength]
            else:
                result[j * (h - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                        h - 2 * RepetitiveLength) + RepetitiveLength,
                0:w - RepetitiveLength] = img[RepetitiveLength: h - RepetitiveLength, 0: w - RepetitiveLength]
        elif (i % len(TifArray[0]) == len(TifArray[0]) - 1):
            if (j == 0):
                result[0: h - RepetitiveLength, img_shape[1] - RowOver: img_shape[1]] = img[0: h - RepetitiveLength,
                                                                                           w - RowOver: w]
            elif (j == len(TifArray) - 1):
                result[img_shape[0] - ColumnOver: img_shape[0], img_shape[1] - RowOver: img_shape[1]] = img[
                                                                                                           h - ColumnOver: h,
                                                                                                           w - RowOver: w]
            else:
                result[j * (h - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                        h - 2 * RepetitiveLength) + RepetitiveLength,
                img_shape[1] - RowOver: img_shape[1]] = img[RepetitiveLength: h - RepetitiveLength, w - RowOver: w]
            j = j + 1
        else:
            if (j == 0):
                result[0: h - RepetitiveLength,
                (i - j * len(TifArray[0])) * (w - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                    TifArray[0]) + 1) * (w - 2 * RepetitiveLength) + RepetitiveLength] = img[0: h - RepetitiveLength, RepetitiveLength: w - RepetitiveLength]
            if (j == len(TifArray) - 1):
                result[img_shape[0] - ColumnOver: img_shape[0],
                (i - j * len(TifArray[0])) * (w - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                    TifArray[0]) + 1) * (w - 2 * RepetitiveLength) + RepetitiveLength] = img[h - ColumnOver: h, RepetitiveLength: w - RepetitiveLength]
            else:
                result[j * (h - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                        h - 2 * RepetitiveLength) + RepetitiveLength,
                (i - j * len(TifArray[0])) * (w - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                    TifArray[0]) + 1) * (w - 2 * RepetitiveLength) + RepetitiveLength] = img[
                                                                                            RepetitiveLength: h - RepetitiveLength,
                                                                                            RepetitiveLength: w - RepetitiveLength]
    return result


def dataPreprocess(img, label, classNum, colorDict_GRAY):
    img = img / 255.0

    for i in range(colorDict_GRAY.shape[0]):
        label[label == colorDict_GRAY[i][0]] = i
    return (img, label)
