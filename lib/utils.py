import os 
import glob
import cv2
import numpy as np
from skimage.morphology import remove_small_holes, remove_small_objects
import logging

def mkdirs(dirs):
    if isinstance(dirs, list):
        for dir_path in dirs:
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
    elif isinstance(dirs, str):
        if not os.path.isdir(dirs):
            os.makedirs(dirs)

def found_dir(all_dir,typename,npy_list=[]):
    sub_list = [dI for dI in os.listdir(all_dir)]
    stype = '/*.'+str(typename)
    for sub in sub_list:
        new_dir = os.sep.join([all_dir, sub])
        #print(new_dir)
        if os.path.isdir(new_dir):
            img_dir = glob.glob(new_dir+ stype)
            if img_dir ==[]:
                found_dir(new_dir,npy_list)
            else:
                npy_list.extend(img_dir)
    return npy_list 


def do_resize2(image, mask, H, W):
    image = cv2.resize(image, dsize=(W,H))
    mask = cv2.resize(mask, dsize=(W,H))
    return image, mask

def do_center_pad(image, pad_left, pad_right):
    return np.pad(image, (pad_left, pad_right), 'edge')

def do_center_pad2(image, mask, pad_left, pad_right):
    image = do_center_pad(image, pad_left, pad_right)
    mask = do_center_pad(mask, pad_left, pad_right)
    return image, mask

def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def get_arr(arr_ls):
    V = []
    for img in arr_ls:
        V.append(img)
    
    V = np.array(V,order='A')
    Vt = V > 0.5
    Vt = Vt.astype(bool)
    return Vt 

def DICE(predict,truth):
    Vseg = get_arr(predict)
    Vref = get_arr(truth)
    dice = 2*(Vref & Vseg).sum() / (Vref.sum()+Vseg.sum())
    return dice

def plot_countours(img_path,mask_path,show=True,output=False):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path,0)
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    rgb_ls = [(255,0,0),(0,255,0),(0,0,255),(0,255,255)]
    for i in range(len(contours)):
        cnt = [contours[i]]
        rgb = rgb_ls[i]
        new_img = cv2.drawContours(img,cnt,-1,rgb,thickness=1)
        continue
    if show:
        plt.axis('off') 
        plt.imshow(new_img)
        plt.show()
    if output:
        return new_img

def mask_post_process(np_array):
    """
    mask post-process
    """
    min_area = 30
    min_hole = 15

    yuce01_mask_array = np_array
    yuce01_mask_array = remove_small_objects(yuce01_mask_array.astype(bool), min_area, connectivity=2)
    yuce01_mask_array = remove_small_holes(yuce01_mask_array.astype(int), area_threshold=min_hole, connectivity=2)

    return yuce01_mask_array

def mean_arr(ls):
    ls_mean = np.mean(ls)
    ls_var =  np.std(ls, ddof=1)
    return ls_mean,ls_var

def get_onehot(mask,label2trainid):
    gt_copy = mask.copy()
    for k, v in label2trainid.items():
        gt_copy[mask == k] = v
    gt_with_trainid = gt_copy.astype(np.uint8)
    gt_onehot = mask2onehot(gt_with_trainid, len(label2trainid))
    return gt_onehot

def mask2onehot(mask, num_classes):
    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)

def onehot2mask(mask):
    _mask = np.argmax(mask, axis=1).astype(np.uint8)
    return _mask


def logger(file_log):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # mode:a write，w cover write
    fh = logging.FileHandler(filename=file_log, encoding='utf-8', mode='w')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s-%(filename)s-%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # 给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
