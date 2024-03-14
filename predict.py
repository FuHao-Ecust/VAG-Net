import os
import numpy as np
import pandas as pd
import argparse
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from lib.CHAOSmetrics import png_series_reader,png_4class_reader,evaluate
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
from PIL import Image
from itertools import cycle
from tqdm import tqdm
from lib.utils import mkdirs,mean_arr
from vag_net import VAG_Net
import time
import sys
import logging
from os.path import join as pjoin
from lib.dataset import onehot2mask
from lib.utils import *
from config import *
from vag_net import VAG_Net
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
device_ids = [0, 1]
device = 'cuda'

parser = argparse.ArgumentParser()
parser.add_argument('--T1_test', type=str, default='./test_t1', help='T1 test image path')
parser.add_argument('--T2_test', type=str, default='./test_t2', help='T2 test image path')
parser.add_argument('--GROUND_4class_T1', type=str, default='./test/mask_4class_T1', help='T1 test mask path')
parser.add_argument('--GROUND_4class_T2', type=str, default='./test/mask_4class_T2', help='T2 test mask path')

def get_img(img, train2label=train2label):
    img_copy = img.copy()
    for k, v in train2label.items():
        img_copy[img == k] = v
    img_with_train = img_copy.astype(np.uint8)
    return img_with_train

def save_fig(pred, path, pred_save_dir, threshold=True):
    fpath, fname = os.path.split(path)
    if not threshold:
        arr = pred
        pred_save_dir = pred_save_dir.replace('class4', 'class4_no_threshold')
        mkdirs(pred_save_dir)
    else:
        arr = torch.where(pred >= 0.5, 1, 0)

    save_path = os.sep.join([pred_save_dir, fname])
    arr_img = torch.argmax(arr, axis=1)
    arr_img = arr_img.data.cpu().numpy()
    xx = get_img(arr_img.squeeze())
    arr_img = Image.fromarray(np.uint8(xx))
    arr_img.save(save_path)

def read_img(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    image = image.reshape(1, 1, image.shape[0], image.shape[1])
    image = torch.from_numpy(image)
    image = image.to(device)
    return image

def predict_4class_img(model, weight, test_ls, save_dir, train_2gpu=True):
    param = torch.load(weight)
    if train_2gpu:
        model = nn.DataParallel(model).cuda()
    model.load_state_dict(param)
    model.to(device)
    model.eval()

    T1_ls = test_ls[0]
    T2_ls = test_ls[1]

    save_path_t1 = os.sep.join([save_dir, MRI_ls[0]])
    save_path_t2 = os.sep.join([save_dir, MRI_ls[1]])
    mkdirs([save_path_t1, save_path_t2])

    for t2_img,t1_img in tqdm(zip(cycle(T2_ls),T1_ls)):
        s_img = read_img(t2_img)
        t_img = read_img(t1_img)
        s_logit, _, _, _, _, t_logit, _, _, _, _ = model(s_img, t_img)
        s_logit = nn.Softmax2d()(s_logit)
        t_logit = nn.Softmax2d()(t_logit)
        mkdirs([save_path_t1, save_path_t2])
        save_fig(s_logit,t2_img,save_path_t2)
        save_fig(t_logit,t1_img,save_path_t1)

class Network(nn.Module):
    def __init__(self, n_class):
        super(Network, self).__init__()
        self.s_model = VAG_Net(n_class)
        self.t_model = VAG_Net(n_class)

    def forward(self, s_img, t_img):

        s_logit, s_side8, s_side7, s_side6, s_side5 = self.s_model(s_img)
        t_logit, t_side8, t_side7, t_side6, t_side5 = self.t_model(t_img)
        return s_logit, s_side8, s_side7, s_side6, s_side5, t_logit, t_side8, t_side7, t_side6, t_side5


if __name__ == '__main__':
    args = parser.parse_args()

    TEST_T1 = args.T1_test
    TEST_T2 = args.T2_test
    GROUND_4class_T1 = args.GROUND_4class_T1
    GROUND_4class_T2 = args.GROUND_4class_T2

    test_t1_data = glob.glob(TEST_T1 + '/*.png')
    test_t2_data = glob.glob(TEST_T2 + '/*.png')

    weight = './weights/class4/VAGNet_JSloss_soft_tuning_03-14_05-50'  # weight path
    pred_save_dir = weight.replace('weights', 'result')
    param = glob.glob(weight + '/*.pth')

    model = Network(num_classes)
    for i in range(len(param)):
        weight_path = param[i]
        fpath, fname = os.path.split(weight_path)
        file_name = fname.replace('.pth', '')
        pred_dir = os.sep.join([pred_save_dir, file_name])
        mkdirs(pred_dir)
        predict_4class_img(model, weight=weight_path, test_ls=[test_t1_data, test_t2_data], save_dir=pred_dir)



