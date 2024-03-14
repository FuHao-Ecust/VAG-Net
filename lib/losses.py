import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from copy import deepcopy
import segmentation_models_pytorch as smp
import scipy.stats
from config import *

def get_segmentation_cost(seg_logits, seg_gt):
    '''
    segmentation loss, BCE+DICE
    '''
    lossdice = smp.utils.losses.DiceLoss()
    lossCE = smp.utils.losses.BCEWithLogitsLoss()
    dice_loss = 0.0
    ce_loss = 0.0
    seg_logits = nn.Softmax2d()(seg_logits)
    for i in range(num_classes):
        if i == 0:
            continue
        predi = seg_logits[:, i, :, :]
        gti = seg_gt[:, i, :, :]
        dice_loss += lossdice(predi, gti)
        ce_loss += lossCE(predi, gti)
    return dice_loss / (num_classes - 1), ce_loss / (num_classes - 1)


def JS_divergence(P, Q):
    M = (P + Q) / 2
    return 0.5 * scipy.stats.entropy(P, M) + 0.5 * scipy.stats.entropy(Q, M)


def get_js_cost(source_logits, source_gt, target_logits, target_gt, output=False):
    '''
    get JS divergences of two models' outputs
    '''
    js_loss = 0.0
    source_logits = source_logits
    target_logits = target_logits

    source_prob = []
    target_prob = []

    temperature = 2.0

    for i in range(0, num_classes):
        eps = 1e-6
        s_mask = torch.tile(torch.unsqueeze(source_gt[:, i, :, :], 1), [1, num_classes, 1, 1])
        s_logits_mask_out = source_logits * s_mask
        s_logits_avg = torch.sum(s_logits_mask_out, [0, 2, 3]) / (
                torch.sum(source_gt[:, i, :, :]) + eps)
        s_soft_prob = F.softmax(s_logits_avg / temperature)
        s_soft_prob = s_soft_prob.detach().cpu().numpy()
        source_prob.append(s_soft_prob)

        t_mask = torch.tile(torch.unsqueeze(target_gt[:, i, :, :], 1), [1, num_classes, 1, 1])
        t_logits_mask_out = target_logits * t_mask
        t_logits_avg = torch.sum(t_logits_mask_out, [0, 2, 3]) / (
                torch.sum(target_gt[:, i, :, :]) + eps)
        t_soft_prob = F.softmax(t_logits_avg / temperature)
        t_soft_prob = t_soft_prob.detach().cpu().numpy()
        target_prob.append(t_soft_prob)

        loss = JS_divergence(s_soft_prob, t_soft_prob)
        js_loss += loss.item()
    js_loss = js_loss / num_classes
    if output:
        return js_loss, source_prob, target_prob
    else:
        return js_loss


class Dice_num(nn.Module):
    def __init__(self):
        super(Dice_num, self).__init__()

    def forward(self, input, target):
        N = target.shape[0]
        smooth = 1
        input_flat = input.reshape(N, -1)
        target_flat = target.reshape(N, -1)
        intersection = (input_flat * target_flat).sum()
        dice = (2 * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
        return dice


class MulticlassDice(nn.Module):
    def __init__(self):
        super(MulticlassDice, self).__init__()

    def forward(self, input, target, weights=None):
        C = target.shape[1]
        dice = Dice_num()
        totaldice = 0

        for i in range(C):
            dicenum = dice(input[:, i, :, :], target[:, i, :, :])
            if weights is not None:
                dicenum *= weights[i]
            totaldice += dicenum
        re_dice = totaldice / (C)
        return re_dice

def get_seg_side_loss(side8,side7,side6,side5,mask):
    '''
    get segmentation loss of side-outputs.
    '''
    seg_dice_side8,seg_ce_side8 = get_segmentation_cost(seg_logits=side8, seg_gt=mask)
    seg_dice_side7,seg_ce_side7 = get_segmentation_cost(seg_logits=side7, seg_gt=mask)
    seg_dice_side6,seg_ce_side6 = get_segmentation_cost(seg_logits=side6, seg_gt=mask)
    seg_dice_side5,seg_ce_side5 = get_segmentation_cost(seg_logits=side5, seg_gt=mask)
    seg_dice_side_all = seg_dice_side8+seg_dice_side7+seg_dice_side6+seg_dice_side5
    seg_ce_side_all = seg_ce_side8+seg_ce_side7+seg_ce_side6+seg_ce_side5
    return seg_dice_side_all,seg_ce_side_all

