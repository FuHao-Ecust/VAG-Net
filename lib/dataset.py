import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.sampler import RandomSampler
from copy import deepcopy
import numpy as np
import cv2
from tqdm import tqdm
import os
from config import *
from PIL import Image
from torchvision import transforms

class SaltDataset(Dataset):
    def __init__(self, image_list, mask_list):
        self.imagelist = image_list
        self.masklist = mask_list
    def __len__(self):
        return len(self.imagelist)
    def __getitem__(self, idx):
        image = deepcopy(self.imagelist[idx])
        mask = deepcopy(self.masklist[idx])
        image = image.reshape(1, image.shape[1], image.shape[1])
        return image, mask

def ImageFetch(images_id,Train_mask_t1,Train_mask_t2, get_T1outphase=False):
    T1inphase_ls = np.zeros((images_id.shape[0], 256, 256), dtype=np.float32)
    T1outphase_ls = np.zeros((images_id.shape[0], 256, 256), dtype=np.float32)
    mask_train = np.zeros((images_id.shape[0], 5, 256, 256), dtype=np.float32)

    for idx, image_id in tqdm(enumerate(images_id), total=images_id.shape[0]):
        image_path = image_id
        fpath, fname = os.path.split(image_path)
        if fname.split('_')[1]=='T1DUAL':
            Mask_path = Train_mask_t1
        elif fname.split('_')[1]=='T2SPIR':
            Mask_path = Train_mask_t2 
            
        mask_path = os.path.join(Mask_path, fname.replace('.png', '_m.png'))
        T1inphase = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        label = Image.open(mask_path)
        resize = transforms.Resize(size=256,interpolation=0)
        label = resize(label)
        label = np.array(label)

        if get_T1outphase:
            T1outphase_path = os.path.join(T1outphase_DIR, fname)
            T1outphase = cv2.imread(T1outphase_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
            T1outphase_ls[idx] = T1outphase

        mask = get_onehot(label)
        T1inphase_ls[idx] = T1inphase
        mask_train[idx] = mask
    return T1inphase_ls, T1outphase_ls, mask_train


def get_onehot(mask, label2trainid=label2trainid):
    gt_copy = mask.copy()
    for k, v in label2trainid.items():
        gt_copy[mask == k] = v
    gt_with_trainid = gt_copy.astype(np.uint8)
    gt_onehot = mask2onehot(gt_with_trainid, len(label2trainid))
    return gt_onehot


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def mask2onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)


def onehot2mask(mask):
    """
    Converts a mask (K, H, W) to (H,W)
    """
    _mask = np.argmax(mask, axis=1).astype(np.uint8)
    return _mask

