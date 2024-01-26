import os
import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim

from torch.utils.data import DataLoader

from config_file import config
from nets.deeplabv3_plus import DeepLab
from nets.deeplabv3_training import (get_lr_scheduler, set_optimizer_lr,
                                     weights_init)
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils.utils import download_weights, show_config, color_dict
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":

    # ---------------------------------#
    #   Cuda    是否使用Cuda
    #           没有GPU可以设置成False
    # ---------------------------------#
    Cuda = False
    # ---------------------------------------------------------------------#
    #   distributed     Specifies whether to use single-node multi-card distributed operation
    #   DP mode：
    #       setup :           distributed = False
    #       command:    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP mode：
    #       setup            distributed = True
    #       command:    CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m torch.distributed.launch --nproc_per_node=5 train.py
    # ---------------------------------------------------------------------#
    distributed = False
    # distributed = True
    # ---------------------------------------------------------------------#
    #   sync_bn     If sync_bn is used, the DDP mode is available for multiple cards
    # ---------------------------------------------------------------------#
    # sync_bn         = False
    sync_bn = True
    # ---------------------------------------------------------------------#
    #   fp16        Whether to use mixed precision training
    # ---------------------------------------------------------------------#
    fp16 = False
    # -----------------------------------------------------#
    #   num_classes     Training your own data set, it must be modified
    # -----------------------------------------------------#
    num_classes = config["num_classes"]
    # ---------------------------------#
    #   Backbone network used:
    #   mobilenet
    #   xception
    # ---------------------------------#
    backbone = config["backbone"]
    # --------------------------------------------------------------#
    pretrained = False
    # ---------------------------------------------------------------#
    model_path = ""
    # ---------------------------------------------------------#
    #   downsample_factor   Multiple of downsampling 8 and 16
    #                       8： The subsampling multiple is smaller and the theoretical effect is better.
    #                       But it also requires more memory
    # ---------------------------------------------------------#
    downsample_factor = config["downsample_factor"]
    # ------------------------------#
    #   Size of picture
    # ------------------------------#
    input_shape = config["input_shape"]

    Init_Epoch = 0
    Freeze_Epoch = config["Freeze_Epoch"]
    Freeze_batch_size = config["Freeze_batch_size"]
    # ------------------------------------------------------------------#
    UnFreeze_Epoch = config["UnFreeze_Epoch"]
    Unfreeze_batch_size = config["Unfreeze_batch_size"]
    Freeze_Train = config["Freeze_Train"]
    # ------------------------------------------------------------------#
    #   Init_lr         The maximum learning rate of the model
    #                   Recommended setting when using the Adam optimizer  Init_lr=5e-4
    #                   Recommended setting when using the SGD optimizer   Init_lr=7e-3
    #   Min_lr          The minimum learning rate of the model defaults to 0.01 of the maximum learning rate
    # ------------------------------------------------------------------#
    Init_lr = 7e-3
    Min_lr = Init_lr * 0.01
    # ------------------------------------------------------------------#
    #   optimizer_type  The types of optimizers: adam and sgd
    #   momentum        The momentum parameter used inside the optimizer
    #   weight_decay    Weight attenuation prevents overfitting
    # ------------------------------------------------------------------#
    optimizer_type = "sgd"
    momentum = 0.9
    weight_decay = 1e-4
    # ------------------------------------------------------------------#
    #   lr_decay_type   The learning rate reduction method used is 'step' and 'cos'.
    # ------------------------------------------------------------------#
    lr_decay_type = 'cos'
    # ------------------------------------------------------------------#
    #   save_period     How many epochs to store a weight at a time
    # ------------------------------------------------------------------#
    save_period = config["save_period"]
    # ------------------------------------------------------------------#
    #   save_dir        The weights and log files are stored in the folder
    # ------------------------------------------------------------------#
    save_dir = config["save_dir"]
    # ------------------------------------------------------------------#
    #   eval_flag       Whether to evaluate during training, the evaluation object is the verification set
    #   eval_period     This represents how many epochs are evaluated once. Frequent evaluation is not recommended
    # ------------------------------------------------------------------#
    eval_flag = config["eval_flag"]
    eval_period = config["eval_period"]

    # ------------------------------------------------------------------#
    #   dataset_path  Data set path
    # ------------------------------------------------------------------#
    dataset_path = config["dataset_path"]
    # ------------------------------------------------------------------#
    dice_loss = False
    # ------------------------------------------------------------------#
    #   Whether to use focal loss to prevent positive and negative sample imbalance
    # ------------------------------------------------------------------#
    focal_loss = False
    # ------------------------------------------------------------------#
    cls_weights = np.ones([num_classes], np.float32)
    # ------------------------------------------------------------------#
    #   num_workers     Set whether to use multiple threads to read data.
    # ------------------------------------------------------------------#
    num_workers = config["num_workers"]

    # ------------------------------------------------------#
    #   Set the graphics card to be used
    # ------------------------------------------------------#
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0

    # ----------------------------------------------------#
    #   Download pre-training weights
    # ----------------------------------------------------#
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)
            dist.barrier()
        else:
            download_weights(backbone)

    model = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor,
                    pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        # ------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))

    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        init_batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        loss_history = LossHistory(save_dir, model, init_batch_size=init_batch_size, input_shape=input_shape)
    else:
        loss_history = None

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()

    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            model_train = torch.nn \
                .DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    with open(os.path.join(dataset_path, "train_set/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(dataset_path, "train_set/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes=num_classes, backbone=backbone, model_path=model_path, input_shape=input_shape, \
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train, \
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type, \
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )

        wanted_step = 1.5e4 if optimizer_type == "sgd" else 0.5e4
        total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print(
                "\n\033[1;33;44m[Warning] When using the %s optimizer, it is recommended to set the total training step size above %d.\033[0m" % (
                optimizer_type, wanted_step))
            print(
                "\033[1;33;44m[Warning] The total training data amount of this run is %d, the Unfreeze_batch_size is %d, a total of %d epochs are trained, and the total training step size is %d.\033[0m" % (
                num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print(
                "\033[1;33;44m[Warning] Since the total training step size is %d, which is less than the recommended total step size %d, it is recommended to set the total generation to %d.\033[0m" % (
                total_step, wanted_step, wanted_epoch))

    if True:
        UnFreeze_flag = False
        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs = 16
        lr_limit_max = 5e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        if backbone == "xception":
            lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
            lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                             weight_decay=weight_decay)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The data set is too small to continue training. Please expand the data set.")
        colorDict_RGB, colorDict_GRAY = color_dict(os.path.join(dataset_path, config["label_data_path"]),
                                                   num_classes)

        train_dataset = DeeplabDataset(train_lines, input_shape, num_classes, True, dataset_path, colorDict_GRAY)
        val_dataset = DeeplabDataset(val_lines, input_shape, num_classes, False, dataset_path, colorDict_GRAY)

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, )
            batch_size = batch_size // ngpus_per_node
            shuffle = False

        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=deeplab_dataset_collate, sampler=train_sampler)
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=deeplab_dataset_collate, sampler=val_sampler)

        if local_rank == 0:
            eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, dataset_path, log_dir, Cuda, \
                                         eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback = None

        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                nbs = 16
                lr_limit_max = 5e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
                if backbone == "xception":
                    lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
                    lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("The data set is too small to continue training. Please expand the data set.")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=deeplab_dataset_collate, sampler=train_sampler)
                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=deeplab_dataset_collate, sampler=val_sampler)

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val,
                          gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16,
                          scaler, save_period, save_dir, colorDict_GRAY, local_rank)

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
