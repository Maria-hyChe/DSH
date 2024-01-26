## DSH: a new transferable deep learning approach for crop mapping
## Introduction
This is a PyTorch implementation of the research: [DSH: a new transferable deep learning approach for crop mapping](https://github.com/happay-ending/graphsformerCPI)

This repository contains a brief description of the paper, source code, data and program run instructions.

---

---

## Dependencies
```
scipy==1.2.1
numpy==1.17.0
matplotlib==3.1.2
opencv_python==4.1.2.30
torch==1.2.0
torchvision==0.4.0
tqdm==4.60.0
Pillow==8.2.0
h5py==2.10.0
```
## Using
### training steps
1. Put the images and labels into the corresponding directory under "./dataset/train_set/".
2. Generate training set and validation set annotations using the following commands,
```
python annotation.py
```
3. To specify the number of GPUs to be used you can use the following command. For example, to use the 1st and 4th GPU for training:
```
CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 train.py
```
### predict steps
Run predict.py for predict.
```
python predict.py
```
