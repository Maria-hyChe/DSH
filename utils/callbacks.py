import os
import datetime
import matplotlib
import torch
import torch.nn.functional as F

from config_file import config

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal

import cv2
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .utils import cvtColor, preprocess_input, resize_image, read_tif, writeTiff
from .utils_metrics import compute_mIoU


class LossHistory():
    def __init__(self, log_dir, model, init_batch_size, input_shape):
        # self.log_dir    = log_dir
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        self.log_dir = os.path.join(log_dir, "loss_" + str(time_str))
        self.losses = []
        self.val_loss = []
        self.train_metrics = []
        self.val_metrics = []

        os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        try:
            # dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
            dummy_input = torch.randn(init_batch_size, input_shape[2], input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def append_metric(self, epoch, train_metric, val_metric):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.train_metrics.append(train_metric)
        self.val_metrics.append(val_metric)

        with open(os.path.join(self.log_dir, "epoch_train_metrics.txt"), 'a') as f:
            f.write(str(train_metric))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_metrics.txt"), 'a') as f:
            f.write(str(val_metric))
            f.write("\n")

        self.writer.add_scalar('AUC', train_metric[1], epoch)
        self.writer.add_scalar('val_AUC', val_metric[1], epoch)
        # self.loss_plot()
        self.metric_plot()

    def metric_plot(self):
        iters = range(len(self.train_metrics))
        train_metrics = self.train_metrics
        val_metrics = self.val_metrics
        train_metrics = np.array(train_metrics)
        val_metrics = np.array(val_metrics)
        plt.figure()
        plt.plot(iters, train_metrics[:, 1], 'red', linewidth=2, label='train AUC')
        plt.plot(iters, val_metrics[:, 1], 'coral', linewidth=2, label='val AUC')
        plt.plot(iters, train_metrics[:, 0], 'blue', linewidth=2, label='train ACC')
        plt.plot(iters, val_metrics[:, 0], 'brown', linewidth=2, label='val ACC')
        try:
            if len(self.train_metrics) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(train_metrics[:, 1], num, 3), 'green', linestyle='--',
                     linewidth=2,
                     label='smooth train AUC')
            plt.plot(iters, scipy.signal.savgol_filter(val_metrics[:, 1], num, 3), '#8B4513', linestyle='--',
                     linewidth=2,
                     label='smooth val AUC')
            plt.plot(iters, scipy.signal.savgol_filter(train_metrics[:, 0], num, 3), 'cyan', linestyle='--',
                     linewidth=2,
                     label='smooth train ACC')
            plt.plot(iters, scipy.signal.savgol_filter(val_metrics[:, 0], num, 3), 'darkorange', linestyle='--',
                     linewidth=2,
                     label='smooth val ACC')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('metric')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_metric.png"))

        plt.cla()
        plt.close("all")

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2,
                     label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")


class EvalCallback():
    def __init__(self, net, input_shape, num_classes, image_ids, dataset_path, log_dir, cuda, \
                 miou_out_path=".temp_miou_out", eval_flag=True, period=1):
        super(EvalCallback, self).__init__()

        self.net = net
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.image_ids = image_ids
        self.dataset_path = dataset_path
        self.log_dir = log_dir
        self.cuda = cuda
        self.miou_out_path = miou_out_path
        self.eval_flag = eval_flag
        self.period = period

        self.image_ids = [image_id.split()[0] for image_id in image_ids]
        self.mious = [0]
        self.epoches = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def get_miou_png(self, image, colorDict_GRAY):

        orininal_h = np.array(image).shape[2]
        orininal_w = np.array(image).shape[1]

        image_data = np.expand_dims(np.array(image, np.float32), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr.argmax(axis=-1)

        image = np.reshape(np.array(colorDict_GRAY, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
        image = np.squeeze(image)
        image = np.uint8(image)
        return image

    def on_epoch_end(self, epoch, model_eval, colorDict_GRAY):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            gt_dir = os.path.join(self.dataset_path, config["label_data_path"])
            pred_dir = os.path.join(self.miou_out_path, 'detection-results')
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            print("Get miou.")
            for image_id in tqdm(self.image_ids):
                image_path = os.path.join(self.dataset_path, config["source_image_path"] , image_id + ".tif")
                im_proj, im_Geotrans, image = read_tif(image_path)
                image = self.get_miou_png(image, colorDict_GRAY)
                writeTiff(os.path.join(pred_dir, image_id + ".tif"), image, im_Geotrans, im_proj)

            print("Calculate miou.")
            _, IoUs, _, _ = compute_mIoU(gt_dir, pred_dir, self.image_ids, self.num_classes,colorDict_GRAY, None)  # 执行计算mIoU的函数
            temp_miou = np.nanmean(IoUs) * 100

            self.mious.append(temp_miou)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(temp_miou))
                f.write("\n")

            plt.figure()
            plt.plot(self.epoches, self.mious, 'red', linewidth=2, label='train miou')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Miou')
            plt.title('A Miou Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
            plt.cla()
            plt.close("all")

            print("Get miou done.")
            shutil.rmtree(self.miou_out_path)
