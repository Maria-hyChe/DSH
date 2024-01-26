import csv
import os
import cv2
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score

from config_file import config
from utils.utils import read_tif


def f_score(inputs, target, beta=1, smooth = 1e-5, threhold = 0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target, axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score

def metrics(inputs, target):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)  # (n,h,w,c)
    temp_target = target.view(n, -1, ct)

    auc = roc_auc_score(temp_target.cpu().numpy().flatten(), temp_inputs.cpu().numpy().flatten())

    temp_inputs = torch.gt(temp_inputs, 0.5).float()
    label = temp_target.argmax(dim=-1)
    label_pre = temp_inputs.argmax(dim=-1)
    label = label.reshape(-1)
    label_pre = label_pre.reshape(-1)

    metric_values = confusion_matrix_to_accuraccies(confusion_matrix(label.cpu().numpy(), label_pre.cpu().numpy()))
    overall_accuracy, kappa, precision, recall, f1, cl_acc = metric_values

    return overall_accuracy, kappa, precision.mean(), recall.mean(), f1.mean(), cl_acc, auc


def confusion_matrix_to_accuraccies(confusion_matrix):
    confusion_matrix = confusion_matrix.astype(float)

    total = np.sum(confusion_matrix)
    n_classes, _ = confusion_matrix.shape
    overall_accuracy = np.sum(np.diag(confusion_matrix)) / total

    N = total
    p0 = np.sum(np.diag(confusion_matrix)) / N
    pc = np.sum(np.sum(confusion_matrix, axis=0) * np.sum(confusion_matrix, axis=1)) / N ** 2
    kappa = (p0 - pc) / (1 - pc)

    recall = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + 1e-12)
    precision = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=0) + 1e-12)
    f1 = (2 * precision * recall) / ((precision + recall) + 1e-12)

    cl_acc = np.diag(confusion_matrix) / (confusion_matrix.sum(1) + 1e-12)

    return overall_accuracy, kappa, precision, recall, f1, cl_acc

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1) 

def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1) 

def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, colorDict_GRAY,name_classes=None):
    hist = np.zeros((num_classes, num_classes))

    if "detection-results" in pred_dir:
        gt_imgs     = [join(gt_dir, x + ".tif") for x in png_name_list]
        pred_imgs   = [join(pred_dir, x + ".tif") for x in png_name_list]
    else:
        gt_imgs = [join(gt_dir, config["predicted_img_label_name"])]
        pred_imgs = [join(pred_dir, config["predicted_img_out_name"] + ".tif")]

    for ind in range(len(gt_imgs)):
        _,_,pred = read_tif(pred_imgs[ind])
        _, _, label = read_tif(gt_imgs[ind])

        for i in range(colorDict_GRAY.shape[0]):
            pred[pred == colorDict_GRAY[i][0]] = i
            label[label == colorDict_GRAY[i][0]] = i

        if len(label.flatten()) != len(pred.flatten()):
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        prec = precision_score(label.flatten(), pred.flatten(),average='macro')  # average : {'micro', 'macro', 'samples', 'weighted', 'binary'}
        rec = recall_score(label.flatten(), pred.flatten(),average='macro')
        if name_classes is not None and ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: mIou:{:0.2f}%\t| mPA:{:0.2f}%\t| Accuracy:{:0.2f}% |'.format(
                    ind,
                    len(gt_imgs),
                    100 * np.nanmean(per_class_iu(hist)),
                    100 * np.nanmean(per_class_PA_Recall(hist)),
                    100 * per_Accuracy(hist)
                )
            )
    IoUs        = per_class_iu(hist)
    PA_Recall   = per_class_PA_Recall(hist)
    Precision   = per_class_Precision(hist)
    if name_classes is not None:
        for ind_class in range(num_classes):
            print("-"*75)
            print(r"{:8}:  Iou:{:5.2f}% | Recall (equal to the PA): {:5.2f}% | Precision: {:5.2f}% |".format(name_classes[ind_class],IoUs[ind_class] * 100,PA_Recall[ind_class] * 100,Precision[ind_class] * 100))
        print("-" * 75)

    print(r'mIoU: {:5.2f}% | mPrecision: {:5.2f}% |  mPA: {:5.2f}% | Accuracy: {:5.2f}% |'.format(np.nanmean(IoUs) * 100,np.nanmean(Precision) * 100,np.nanmean(PA_Recall) * 100,per_Accuracy(hist) * 100))
    return np.array(hist, np.int), IoUs, PA_Recall, Precision

def adjust_axes(r, t, fig, axes):
    bb                  = t.get_window_extent(renderer=r)
    text_width_inches   = bb.width / fig.dpi
    current_fig_width   = fig.get_figwidth()
    new_fig_width       = current_fig_width + text_width_inches
    propotion           = new_fig_width / current_fig_width
    x_lim               = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])

def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size = 12, plt_show = True):
    fig     = plt.gcf() 
    axes    = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val) 
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values)-1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()

def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size = 12):
    draw_plot_func(IoUs, name_classes, "mIoU = {0:.2f}%".format(np.nanmean(IoUs)*100), "Intersection over Union", \
        os.path.join(miou_out_path, "mIoU.png"), tick_font_size = tick_font_size, plt_show = True)
    print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))

    draw_plot_func(PA_Recall, name_classes, "mPA = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Pixel Accuracy", \
        os.path.join(miou_out_path, "mPA.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))
    
    draw_plot_func(PA_Recall, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Recall", \
        os.path.join(miou_out_path, "Recall.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(Precision, name_classes, "mPrecision = {0:.2f}%".format(np.nanmean(Precision)*100), "Precision", \
        os.path.join(miou_out_path, "Precision.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer          = csv.writer(f)
        writer_list     = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))
def IntersectionOverUnion(confusionMatrix):
    intersection = np.diag(confusionMatrix)
    union = np.sum(confusionMatrix, axis=0) + np.sum(confusionMatrix, axis=1) - np.diag(confusionMatrix)
    IoU = intersection / union
    return IoU


def MeanIntersectionOverUnion(confusionMatrix):
    intersection = np.diag(confusionMatrix)
    union = np.sum(confusionMatrix, axis=0) + np.sum(confusionMatrix, axis=1) - np.diag(confusionMatrix)
    IoU = intersection / union
    mIoU = np.nanmean(IoU)
    return mIoU


def Frequency_Weighted_Intersection_over_Union(confusionMatrix):
    freq = np.sum(confusionMatrix, axis=0) / np.sum(confusionMatrix)
    iu = np.diag(confusionMatrix) / (
            np.sum(confusionMatrix, axis=0) +
            np.sum(confusionMatrix, axis=1) -
            np.diag(confusionMatrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU
            