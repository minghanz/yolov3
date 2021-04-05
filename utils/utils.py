import glob
import math
import os
import random
import shutil
import subprocess
import time
from copy import copy
from pathlib import Path
from sys import platform

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

from . import torch_utils  # , google_utils

import torch
# from d3d.box import box2d_iou, box2d_nms, box2d_crop
import d3d

import torchsnooper

# import sys
# sys.path.append("../../")
# from bev.rbox_torch import *

# Set printoptions
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
matplotlib.rc('font', **{'size': 11})

# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
cv2.setNumThreads(0)


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch_utils.init_seeds(seed=seed)


def check_git_status():
    if platform in ['linux', 'darwin']:
        # Suggest 'git pull' if repo is out of date
        s = subprocess.check_output('if [ -d .git ]; then git fetch && git status -uno; fi', shell=True).decode('utf-8')
        if 'Your branch is behind' in s:
            print(s[s.find('Your branch is behind'):s.find('\n\n')] + '\n')


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurences per class

    # Prepend gridpoint count (for uCE trianing)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # Produces image weights based on class mAPs
    n = len(labels)
    class_counts = np.array([np.bincount(labels[i][:, 0].astype(np.int), minlength=nc) for i in range(n)])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)

    if len(x) == 0:
        print("empty input in xyxy2xywh")
        return y

    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)

    if len(x) == 0:
        print("empty input in xywh2xyxy")
        return y

    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def v2yaw(x):
    ### atan2(sin, cos)
    y = torch.atan2(x[:,0], x[:,1]) if isinstance(x, torch.Tensor) else np.arctan2(x[:,0], x[:,1])
    return y

def yaw2v(x):
    y = torch.stack((torch.sin(x), torch.cos(x)), dim=1) if isinstance(x, torch.Tensor) else np.stack((np.sin(x), np.cos(x)), axis=1)
    return y

def yaw2mat(x):

    x = x.reshape(-1,1)
    try:
        if isinstance(x, torch.Tensor):
            # y = torch.stack([torch.tensor([[torch.cos(x[i]), torch.sin(x[i]) ], [-torch.sin(x[i]), torch.cos(x[i])]]) for i in range(x.shape[0])], dim=0).to(device=x.device) # n*2*2
            y = torch.cat([torch.cos(x), torch.sin(x), -torch.sin(x), torch.cos(x)], dim=1).reshape(-1,2,2)
        else:
            # y = np.stack([np.array([[np.cos(x[i]), np.sin(x[i]) ], [-np.sin(x[i]), np.cos(x[i])]]) for i in range(x.shape[0])], axis=0) # n*2*2
            y = np.concatenate([np.cos(x), np.sin(x), -np.sin(x), np.cos(x)], axis=1).reshape(-1,2,2)
    except:
        print("yaw2mat x", x)
        raise ValueError("empty x?")
    return y

def xywh2xyxy_r(x, external_aa=False):
    # convert n*5 boxes from [x,y,w,h,yaw] to [x1, y1, x2, y2] if external_aa, 
    # else [x1, y1, x2, y2, x3, y3, x4, y4] from [x_min, y_min] [x_min, y_max] [x_max, y_max], [x_max, y_min]
    if external_aa:
        y = torch.zeros((x.shape[0], 4), dtype=x.dtype, device=x.device) if isinstance(x, torch.Tensor) else np.zeros((x.shape[0], 4), dtype=x.dtype)
    else:
        y = torch.zeros((x.shape[0], 8), dtype=x.dtype, device=x.device) if isinstance(x, torch.Tensor) else np.zeros((x.shape[0], 8), dtype=x.dtype)

    if len(x) == 0:
        print("empty input in xywh2xyxy_r")
        return y

    # xyxy_ur = xywh2xyxy(x[:,:4])
    y[:, 0] = - x[:, 2] / 2  # top left x
    y[:, 1] = - x[:, 3] / 2  # top left y
    y[:, 2] = - x[:, 2] / 2  # bottom left x
    y[:, 3] =   x[:, 3] / 2  # bottom left y
    y[:, 4] =   x[:, 2] / 2  # bottom right x
    y[:, 5] =   x[:, 3] / 2  # bottom right y
    y[:, 6] =   x[:, 2] / 2  # top right x
    y[:, 7] = - x[:, 3] / 2  # top right y

    if isinstance(x, torch.Tensor):
        y = y.reshape(-1, 4, 2).transpose(1,2) # n*2*4      ### the rotation matrix is consistent with random_affine_xywhr() in datasets.py
        # rot_mat = torch.stack([torch.tensor([[torch.cos(x[i,4]), torch.sin(x[i,4]) ], [-torch.sin(x[i,4]), torch.cos(x[i,4])]]) for i in range(x.shape[0])], dim=0) # n*2*2
        rot_mat = yaw2mat(x[:,4])
        y = torch.matmul(rot_mat, y)
        y = y.transpose(1,2).reshape(-1, 8)
    else:
        y = y.reshape(-1, 4, 2).swapaxes(1,2) # n*2*4
        # rot_mat = np.stack([np.array([[np.cos(x[i,4]), np.sin(x[i,4]) ], [-np.sin(x[i,4]), np.cos(x[i,4])]]) for i in range(x.shape[0])], axis=0) # n*2*2
        rot_mat = yaw2mat(x[:,4])
        y = np.matmul(rot_mat, y)
        y = y.swapaxes(1,2).reshape(-1, 8)

    y += x[:, [0,1,0,1,0,1,0,1]]    # n*8
    
    if not external_aa:
        return y
    else:
        if isinstance(x, torch.Tensor):
            x_min = y[:,[0,2,4,6]].min(dim=1) # n*1
            x_max = y[:,[0,2,4,6]].max(dim=1) # n*1
            y_min = y[:,[1,3,5,7]].min(dim=1) # n*1
            y_max = y[:,[1,3,5,7]].max(dim=1) # n*1
            y = torch.stack([x_min, y_min, x_max, y_max], dim=1)
        else:
            x_min = y[:,[0,2,4,6]].min(axis=1) # n*1
            x_max = y[:,[0,2,4,6]].max(axis=1) # n*1
            y_min = y[:,[1,3,5,7]].min(axis=1) # n*1
            y_max = y[:,[1,3,5,7]].max(axis=1) # n*1
            y = np.stack([x_min, y_min, x_max, y_max], axis=1)
        return y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def scale_coords_r(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, 0] -= pad[0]  # x padding
    coords[:, 1] -= pad[1]  # y padding
    coords[:, :4] /= gain
    # clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls: Predicted object classes (nparray).
        target_cls: True object classes (nparray).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    ### sort detections from max conf to min
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
    s = [unique_classes.shape[0], tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 or n_gt == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_gt + 1e-16)  # recall curve
            # r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])  # r at pr_score, negative x, xp because xp decreases
            ### use "-conf[i]" because np.interp(x, xp, fp) must accept increasing xp (https://numpy.org/doc/stable/reference/generated/numpy.interp.html)

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            # p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

                # Recall
                r[ci, j] = np.interp(-pr_score, -conf[i], recall[:, j])  # r at pr_score, negative x, xp because xp decreases
                ### use "-conf[i]" because np.interp(x, xp, fp) must accept increasing xp (https://numpy.org/doc/stable/reference/generated/numpy.interp.html)

                # Precision
                p[ci, j] = np.interp(-pr_score, -conf[i], precision[:, j])  # p at pr_score

            # Plot
            # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            # ax.plot(recall, precision)
            # ax.set_xlabel('Recall')
            # ax.set_ylabel('Precision')
            # ax.set_xlim(0, 1.01)
            # ax.set_ylim(0, 1.01)
            # fig.tight_layout()
            # fig.savefig('PR_curve.png', dpi=300)

    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap

def lin_iou(box1, box2):

    # inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)

    dist = (box1[:, None, :2] - box2[:, :2]).norm(dim=2)

    dist_ratio = (1 - dist / box2[:, 3]).clamp(0)
    return dist_ratio

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    if box1.shape[0] == 4:
        assert box1.shape[0] == box2.shape[1]
        return bbox_iou_aa(box1, box2, x1y1x2y2, GIoU, DIoU, CIoU)
    else:
        assert box1.shape[0] == 5
        return bbox_iou_r(box1, box2, GIoU, DIoU, CIoU)

def bbox_iou_r(box1, box2, GIoU=False, DIoU=False, CIoU=False):
    ### box is of n*5 with each row: xywhr
    pass

def bbox_iou_aa(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)

def r_align(r1, r2):
    r1 = r1[:, None]
    r2 = r2[None]
    cos = torch.cos(r1 - r2) # if cos > 0, then the angle difference is less than 90 degree 
    return cos

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

# @torchsnooper.snoop()
def compute_loss(p, targets, model, use_giou_loss=False, half_angle=False, tail_inv=False):  # predictions, targets, model
    ### input p here is training output

    ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor
    lcls, lbox, lobj = ft([0]), ft([0]), ft([0])
    lxy, lwh = ft([0]), ft([0])
    lr = ft([0])
    ltt = ft([0])
    rotated = targets.shape[1] == 7 or targets.shape[1] == 9
    tail = targets.shape[1] == 9
    tcls, tbox, indices, anchors, twh, txy, tyaw, ttt = build_targets(p, targets, model, half_angle, tail_inv)  # targets  ### twh, txy are added for bbox l1/MSE loss 
    rotated_anchor = len(tyaw)>0
    tail = len(ttt) > 0

    h = model.hyp  # hyperparameters
    red = 'mean'  # Loss reduction (sum or mean)

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([h['cls_pw']]), reduction=red)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=ft([h['obj_pw']]), reduction=red)
    MSE = nn.MSELoss(reduction=red)
    MSE_full = nn.MSELoss(reduction='none')

    # class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0)

    # focal loss
    g = h['fl_gamma']  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    # per output
    nt = 0  # targets
    ### For each yolo layers (there should be 3), find the list of ground truth in the form of corresponding grid locations, anchors, and regression target. (indexed by b, a, gj, gi)
    ### Then retrieve the predictions at these anchors. (ps)
    ### pi is the predictions in grid. ps is flattened predictions indexed by target location and anchors. 
    ### Most losses are calculated by comparing targets and ps (positive targets), while pi only appears at BCEobj to encourage correct classification of positive and negative targets. 
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0])  # target obj

        nb = b.shape[0]  # number of targets
        if nb:
            nt += nb  # cumulative targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
            ### b, a, gj, gi are all 1D VECTOR OF THE SAME LENGTH, combining together to form the indices in pi

            # GIoU
            # print(b,a,gj,gi)
            # print("ps", ps)
            # print("giou", i, )
            # print("anchors", anchors[i].shape)
            # print("pswh", ps[:, 2:4].shape)
            pxy0 = ps[:, :2].sigmoid()
            if tail:
                if not rotated_anchor:
                    if tail_inv:
                        ptt = ((ps[:, 6:8]/10).sigmoid()-0.5)*20 * anchors[i][:,[1]]
                        pxy = pxy0.detach() - ptt        ### tail_inv
                    else:
                        ptt = ps[:, 6:8] * anchors[i][:,[1]]
                        pxy = pxy0
                else:
                    if tail_inv:
                        ptt = ((ps[:, 5:7]/10).sigmoid()-0.5)*20 * anchors[i][:,[1]]
                        pxy = pxy0.detach() - ptt        ### tail_inv
                    else:
                        ptt = ps[:, 5:7] * anchors[i][:,[1]]
                        pxy = pxy0

                # print("pxy", pxy)
                # print("tbox[i]", tbox[i])
            else:
                pxy = pxy0
            # pwh = ps[:, 2:4].exp().clamp(max=1E3) * anchors[i][:,:2]    ### Minghan: trying to fix the nan in grad
            pwh = torch.nn.functional.softplus(ps[:, 2:4]) * anchors[i][:,:2]
            if not rotated:
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, GIoU=True)  # giou(prediction, target)
            else:
                ### here it is actually iou instead of giou
                if not rotated_anchor:
                    prvec = nn.functional.normalize(ps[:, 4:6], dim=-1)
                    pyaw = v2yaw(prvec)
                    ### when calculating iou, rotation difference of 180 does not matter.
                else:
                    if half_angle:
                        pyaw_local = (torch.sigmoid(ps[:, 4])-0.5) * math.pi * 0.5
                    else:
                        pyaw_local = (torch.sigmoid(ps[:, 4])-0.5) * math.pi # * 1.5 # using range larger than [-0.5pi, 0.5pi] could result in dead lock
                    pyaw = pyaw_local + anchors[i][:,2]

                pbox = torch.cat((pxy, pwh, -pyaw.unsqueeze(1)), 1)  # predicted box
                tbox_i = tbox[i][:,:5].clone()
                tbox_i[:, 4] = - tbox_i[:, 4]
                giou = d3d.box.box2d_iou(pbox, tbox_i, method="rbox")
                giou = torch.diag(giou)     # d3d.box.box2d_iou produces iou of every pair of boxes
            if use_giou_loss:
                lbox += (1.0 - giou).sum() if red == 'sum' else (1.0 - giou).mean()  # giou loss
            else:
                lxy += MSE(pxy, txy[i])  # xy loss
                lwh += MSE(ps[:, 2:4], twh[i])  # wh yolo loss
                if rotated:
                    ### box coordinate regression?
                    tbox_i_xy8 = xywh2xyxy_r(tbox[i][:,:5])
                    if (not rotated_anchor) and half_angle:
                        pbox_i_xywhr = torch.cat((pxy, pwh, pyaw.unsqueeze(1)), 1)
                        pbox_i_xy8_1 = xywh2xyxy_r(pbox_i_xywhr)
                        coord_loss_1 = MSE_full(pbox_i_xy8_1, tbox_i_xy8)
                        pbox_i_xywhr2 = torch.cat((pxy, pwh, pyaw.unsqueeze(1)+math.pi), 1)
                        pbox_i_xy8_2 = xywh2xyxy_r(pbox_i_xywhr2)
                        coord_loss_2 = MSE_full(pbox_i_xy8_2, tbox_i_xy8)
                        lbox += torch.min(coord_loss_1, coord_loss_2).mean() if red == 'mean' else torch.min(coord_loss_1, coord_loss_2).sum()
                    else:
                        pbox_i_xywhr = torch.cat((pxy, pwh, pyaw.unsqueeze(1)), 1)
                        # pbox_i_xywhr.register_hook(lambda grad: print("pbox_i_xywhr grad:", grad))
                        pbox_i_xy8 = xywh2xyxy_r(pbox_i_xywhr)
                        # pbox_i_xy8.register_hook(lambda grad: print("pbox_i_xy8 grad:", grad))
                        lbox += MSE(pbox_i_xy8, tbox_i_xy8)

                    if not rotated_anchor:
                        tyaw = tbox[i][:,4]
                        trvec = yaw2v(tyaw)
                        # trvec = torch.cat((torch.sin(tyaw), torch.cos(tyaw)), dim=1)  ### rotated vec consistent with YOLOLayer in models.py
                        if half_angle:
                            vec_loss_1 = MSE_full(prvec, trvec)
                            vec_loss_2 = MSE_full(-prvec, trvec)
                            lr += torch.min(vec_loss_1, vec_loss_2).mean() if red == 'mean' else torch.min(vec_loss_1, vec_loss_2).sum()
                        else:
                            lr += MSE(prvec, trvec)  # wh yolo loss
                    else:
                        if half_angle:
                            # lr += MSE(pyaw_local, tyaw[i])
                            lr += torch.sin(pyaw_local - tyaw[i]).pow(2).sum() if red == 'sum' else torch.sin(pyaw_local - tyaw[i]).pow(2).mean()  # giou loss
                        else:
                            # lr += torch.sin((pyaw_local - tyaw[i])/2).abs().sum() if red == 'sum' else torch.sin((pyaw_local - tyaw[i])/2).abs().mean()  # giou loss
                            lr += torch.sin((pyaw_local - tyaw[i])/2).pow(2).sum() if red == 'sum' else torch.sin((pyaw_local - tyaw[i])/2).pow(2).mean()  # giou loss
            if tail:
                # if not rotated_anchor:
                #     ptt = ps[:, 6:8] * anchors[i][:,:2]
                # else:
                #     ptt = ps[:, 5:7] * anchors[i][:,:2]
                
                # x1y1_gt = txy[i] + ttt[i]
                # x1y1 = pxy.detach() + ptt
                # ltt += MSE(x1y1, x1y1_gt)  # tail loss
                x1y1_gt = txy[i] + ttt[i]   
                if tail_inv:
                    ltt += MSE(pxy0, x1y1_gt)   ### tail_inv
                else:
                    x1y1 = pxy0.detach() + ptt
                    ltt += MSE(x1y1, x1y1_gt)

                #     ltt += MSE(ps[:, 6:8], ttt[i])  # tail loss
                # else:
                #     ltt += MSE(ps[:, 5:7], ttt[i])  # tail loss

            # Obj
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * giou.detach().clamp(0).type(tobj.dtype)  # giou ratio

            # Class
            if model.nc > 1:  # cls loss (only if multiple classes)
                if rotated:
                    if not tail:
                        if not rotated_anchor:
                            t = torch.full_like(ps[:, 7:], cn)  # targets
                            t[range(nb), tcls[i]] = cp
                            lcls += BCEcls(ps[:, 7:], t)  # BCE
                        else:
                            t = torch.full_like(ps[:, 6:], cn)  # targets
                            t[range(nb), tcls[i]] = cp
                            lcls += BCEcls(ps[:, 6:], t)  # BCEvvvvvvv
                    else:
                        if not rotated_anchor:
                            t = torch.full_like(ps[:, 9:], cn)  # targets
                            t[range(nb), tcls[i]] = cp
                            lcls += BCEcls(ps[:, 9:], t)  # BCE
                        else:
                            t = torch.full_like(ps[:, 8:], cn)  # targets
                            t[range(nb), tcls[i]] = cp
                            lcls += BCEcls(ps[:, 8:], t)  # BCEvvvvvvv

                else:
                    t = torch.full_like(ps[:, 5:], cn)  # targets
                    t[range(nb), tcls[i]] = cp
                    lcls += BCEcls(ps[:, 5:], t)  # BCE

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        if rotated:
            if not tail:
                if not rotated_anchor:
                    lobj += BCEobj(pi[..., 6], tobj)  # obj loss
                else:
                    lobj += BCEobj(pi[..., 5], tobj)  # obj loss
            else:
                if not rotated_anchor:
                    lobj += BCEobj(pi[..., 8], tobj)  # obj loss
                else:
                    lobj += BCEobj(pi[..., 7], tobj)  # obj loss
        else:
            lobj += BCEobj(pi[..., 4], tobj)  # obj loss

    lbox *= h['giou']
    lobj *= h['obj']
    lcls *= h['cls']
    lxy *= h['xy']
    lwh *= h['wh']
    lr *= h['r']
    ltt *= h['tt']
    if red == 'sum':
        bs = tobj.shape[0]  # batch size
        g = 3.0  # loss gain
        lobj *= g / bs
        if nt:
            lcls *= g / nt / model.nc
            lbox *= g / nt
            lxy *= g / nt
            lwh *= g / nt
            lr *= g / nt
            ltt *= g / nt

    # loss = lbox + lobj + lcls
    # return loss, torch.cat((lbox, lobj, lcls, loss)).detach()
    loss = lbox + lobj + lcls + lxy + lwh + lr + ltt
    # loss = lobj + lxy + lwh + lr + ltt
    return loss, torch.cat((lbox, lobj, lcls, lxy, lwh, lr, ltt, loss)).detach()


def build_targets(p, targets, model, half_angle, tail_inv):
    """
    Given targets (the ground truth objects), find the corresponding location in grid, 
    and the corresponding regression target w.r.t. the matched anchors at the grid. 
    p is only used for shape (grid size).
    """
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    # or (image,class,x,y,w,h,y) for rotated boxes
    ### note that we only use wh to filter out some anchor-detection matches. Not using location (xy) or yaw.
    nt = targets.shape[0]
    tcls, tbox, indices, anch = [], [], [], []
    twh, txy, tyaw = [], [], [] ### add back this because we want to use l2 (mseloss) of rotated boxes, because giou is not available for rotated iou/giou yet.
    ttt = []

    rotated = targets.shape[1] == 7 or targets.shape[1] == 9
    tail = targets.shape[1] == 9
    
    if not rotated:
        gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain
    else:
        if tail:
            gain = torch.ones(9, device=targets.device)  # normalized to gridspace gain (for rotated case)
        else:
            gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain (for rotated case)
    off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float()  # overlap offsets

    style = None
    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    for i, j in enumerate(model.yolo_layers):
        anchors = model.module.module_list[j].anchor_vec if multi_gpu else model.module_list[j].anchor_vec
        rotated_anchor = anchors.shape[1] == 3
        ### lower level layer, denser grid, smaller stride, smaller anchors, targeting smaller objects
        ### three layers, each with three anchors        
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
        if tail:
            gain[7:9] = torch.tensor(p[i].shape)[[3, 2]]
        ### note that p[i].shape == (bs, anchors, ny, nx, classes + xywh)
        ### now gain == [1,1,nx, ny, nx, ny]
        na = anchors.shape[0]  # number of anchors, anchors are n*2 for non-rotated cfg (see .cfg files)
        at = torch.arange(na).view(na, 1).repeat(1, nt)  # anchor tensor, same as .repeat_interleave(nt)

        # Match targets to anchors
        a, t, offsets = [], targets * gain, 0
        ### targets recovered to t=[image, class, xpx, ypx, wpx, hpx, [yaw]] in grid coordinate

        if nt:
            # r = t[None, :, 4:6] / anchors[:, None]  # wh ratio
            # j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare
            ### this is to filter out those detection-anchor pairs meeting wt_iou criteria
            if not rotated_anchor:
                whiou = wh_iou(anchors, t[:, 4:6]) #> model.hyp['iou_t']  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
                j = whiou > model.hyp['iou_t']      ### a bool matrix of na(3)*nt
            else:
                assert anchors.shape[1] == 3
                assert rotated
                if half_angle:
                    # whiou = wh_iou(anchors[:, :2], t[:, 4:6]) #> model.hyp['iou_t']  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
                    # j = whiou > model.hyp['iou_t']      ### a bool matrix of na(3)*nt
                    angle_cos = r_align(anchors[:, 2], t[:, 6])
                    j = torch.abs(angle_cos) > 0.71                      ### 08162020: 0.707 -> pi/4
                else:
                    angle_cos = r_align(anchors[:, 2], t[:, 6])
                    j = angle_cos > 0.1                      ### 08162020: slightly larger than 0 to avoid dead lock
            a, t = at[j], t.repeat(na, 1, 1)[j]  # filter
            ### a is n_filtered index, and t is now n_filtered * 6 or 7 matrix, both are in terms of n*m

            # overlaps
            gxy = t[:, 2:4]  # grid xy
            z = torch.zeros_like(gxy)
            if style == 'rect2':
                g = 0.2  # offset
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                a, t = torch.cat((a, a[j], a[k]), 0), torch.cat((t, t[j], t[k]), 0)
                offsets = torch.cat((z, z[j] + off[0], z[k] + off[1]), 0) * g

            elif style == 'rect4':
                g = 0.5  # offset
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]] - 1.))).T
                a, t = torch.cat((a, a[j], a[k], a[l], a[m]), 0), torch.cat((t, t[j], t[k], t[l], t[m]), 0)
                offsets = torch.cat((z, z[j] + off[0], z[k] + off[1], z[l] + off[2], z[m] + off[3]), 0) * g

        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        if rotated:
            gtt = t[:, 7:9]
            if tail and tail_inv:
                # tbox.append(torch.cat((gxy - gij, gwh, gyaw, gtt), 1))  # box
                gtail = gxy + gtt
                gtail[:, 0] = gtail[:, 0].clamp(min=1e-2, max=p[i].shape[3]-1-1e-2)
                gtail[:, 1] = gtail[:, 1].clamp(min=1e-2, max=p[i].shape[2]-1-1e-2)
                gtt = gtail - gxy
                gtail_ij = (gtail - offsets).long()
                gi, gj = gtail_ij.T       ### for tail_inv

            gyaw = t[:, 6:7]
            if half_angle:
                if rotated_anchor:
                    ### to guarantee the gt is achievable from anchor, need to ensure gt is within pi/2 to anchors
                    need_flip_idx = (gyaw[:, 0] - anchors[a][:, 2]).abs() > math.pi/2
                    anch_yaw = anchors[a][need_flip_idx, 2]
                    gyaw[need_flip_idx, 0] = anch_yaw + torch.atan(torch.tan( gyaw[need_flip_idx, 0]-anch_yaw ))
                else:
                    ### directional vector use both signs and take the min, no need to do things here
                    pass 

        # Append
        indices.append((b, a, gj, gi))  # image, anchor, grid indices
        if rotated:
            if tail:
                if tail_inv:
                    tbox.append(torch.cat((gxy - gtail_ij, gwh, gyaw, gtt), 1))  # box   ### for tail_inv
                else:
                    tbox.append(torch.cat((gxy - gij, gwh, gyaw, gtt), 1))  # box
            else:
                tbox.append(torch.cat((gxy - gij, gwh, gyaw), 1))  # box
        else:
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class

        # twh.append(torch.log(gwh / anchors[a][:, :2]))
        ### Minghan: trying to fix the nan in grad  (inv of softplus)
        twh.append( torch.log(torch.exp(gwh / anchors[a][:, :2]) - 1) )         # compare relative value
        
        if rotated and tail and tail_inv:
            txy.append(gxy - gtail_ij)     ### for tail_inv
        else:
            txy.append(gxy - gij)

        if rotated_anchor:
            # tyaw.append(gyaw - anchors[a][:, 2])
            tyaw.append(gyaw[:,0] - anchors[a][:, 2])     # compare relative value
            # tyaw.append(gyaw[:,0] )                         # compare absolute value
        
        if tail:
            # ttt.append(gtt / anchors[a][:, :2])           # compare relative value
            ttt.append(gtt )                                # compare absolute value

        if c.shape[0]:  # if any targets
            assert c.max() < model.nc, 'Model accepts %g classes labeled from 0-%g, however you labelled a class %g. ' \
                                       'See https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data' % (
                                           model.nc, model.nc - 1, c.max())
                                           
    # print("len(tbox)", len(tbox))
    # for i in range(len(tbox)):
    #     print("tbox[%d].shape"%i, tbox[i].shape)
    # for i in range(len(tbox)):
    #     if tbox[i].numel() > 0:
    #         print("tbox[%d].min(dim=0)"%i, tbox[i].min(dim=0))
    #         print("tbox[%d].max(dim=0)"%i, tbox[i].max(dim=0))

    return tcls, tbox, indices, anch, twh, txy, tyaw, ttt


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, multi_label=True, classes=None, agnostic=False, rotated=False, rotated_anchor=False, tail=False, invalid_masks=None):
    """
    Performs  Non-Maximum Suppression on inference results
    Returns detections with shape:
        nx6 (x1, y1, x2, y2, conf, cls)
        or n*7 (x1, y1, x2, y2, yaw, conf, cls) for rotated bbox
        or n*9 (x1, y1, x2, y2, yaw, dx, dy, conf, cls) for rotated bbox with tail
    """

    # Settings
    merge = True  # merge for best mAP
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    time_limit = 10.0  # seconds to quit after

    t = time.time()
    if rotated:
        if rotated_anchor:
            if tail:
                nc = prediction[0].shape[1] - 8  # number of classes
                conf_idx = 7
                conf_x_idx = 7
            else:
                nc = prediction[0].shape[1] - 6  # number of classes
                conf_idx = 5
                conf_x_idx = 5
        else:
            if tail:
                nc = prediction[0].shape[1] - 9  # number of classes
                conf_idx = 8
                conf_x_idx = 7
            else:
                nc = prediction[0].shape[1] - 7  # number of classes
                conf_idx = 6
                conf_x_idx = 5
    else:
        nc = prediction[0].shape[1] - 5  # number of classes
        conf_idx = 4
        conf_x_idx = 4
    multi_label &= nc > 1  # multiple labels per box
    output = [None] * prediction.shape[0]

    if invalid_masks is None:
        invalid_masks = [None] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        ### this is in terms of batch
        # vnorm = x[:,4:6].norm(dim=1)
        # print("vnorm.min()", vnorm.min())
        # print("vnorm.max()", vnorm.max())
        # print("pred[%d][:,4].min()"%xi, x[:,4].min())
        # print("pred[%d][:,4].max()"%xi, x[:,4].max())
        # print("pred[%d][:,5].min()"%xi, x[:,5].min())
        # print("pred[%d][:,5].max()"%xi, x[:,5].max())
        # print("pred[%d][:,6].min()"%xi, x[:,6].min())
        # print("pred[%d][:,6].max()"%xi, x[:,6].max())
        # print("pred[%d][0,4:6]"%xi, x[0,4:7])
        # print("pred[%d][1,4:6]"%xi, x[1,4:7])

        # Apply constraints
        x = x[x[:, conf_idx] > conf_thres]  # confidence
        x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]  # width-height
        # x = x[(x[:, 0]>208) | (x[:, 1] > 272)]        ### only applicable to Jackson lturn

        invalid_mask = invalid_masks[xi]
        if invalid_mask is not None:
            valid_idx = [ invalid_mask[0, int(x[i, 1]), int(x[i, 0])] == 0 for i in range(x.shape[0]) ]
            x = x[valid_idx, :]

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[..., conf_idx+1:] *= x[..., conf_idx:conf_idx+1]  # conf = obj_conf * cls_conf

        if not rotated:
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])
        else:
            if tail:
                box = torch.zeros((x.shape[0], 7), dtype=x.dtype, device=x.device)
                if rotated_anchor:
                    box[:, 5:7] = x[:, 5:7]
                else:
                    box[:, 5:7] = x[:, 6:8]
            else:
                box = torch.zeros((x.shape[0], 5), dtype=x.dtype, device=x.device)
            box[:, :4] = x[:, :4]
            if rotated_anchor:
                box[:, 4] = x[:, 4]
            else:
                box[:, 4] = v2yaw(x[:, 4:6])
        ############# vx vy are transformed to yaw here

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, conf_idx+1:] > conf_thres).nonzero().t()
            x_cc = torch.cat((box[i], x[i, j + conf_idx+1].unsqueeze(1), j.float().unsqueeze(1)), 1)
            x = x[i] 
        else:  # best class only
            conf, j = x[:, conf_idx+1:].max(1)
            x_cc = torch.cat((box, conf.unsqueeze(1), j.float().unsqueeze(1)), 1)[conf > conf_thres]
            x = x[conf > conf_thres]

        # Filter by class
        if classes:
            x_cc = x_cc[(j.view(-1, 1) == torch.tensor(classes, device=j.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x_cc.shape[0]  # number of boxes
        if not n:
            # print("NMS skipped because no left after conf and class filtering", xi)
            # print("x_cc", x_cc)
            # print("conf.max()", conf.max(), conf_thres)
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x_cc[:, conf_x_idx+1] * 0 if agnostic else x_cc[:, conf_x_idx+1]  # classes
        if not rotated:
            boxes, scores = x_cc[:, :4].clone() + c.view(-1, 1) * max_wh, x_cc[:, conf_x_idx]  # boxes (offset by class), scores
            i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
            if len(i) ==0:
                continue
        else:
            scores = x_cc[:, conf_x_idx]
            boxes = x_cc[:, :5].clone()
            boxes[:, :2] += c.view(-1, 1) * max_wh
            boxes_d3d = boxes.clone()
            boxes_d3d[:, 4] = - boxes[:, 4]
            i_mask = d3d.box.box2d_nms(boxes_d3d, scores, iou_method="rbox", iou_threshold=iou_thres)
            if i_mask.sum() == 0:
                continue
            
        if merge and (1 < n < 3E4):  # Merge NMS (boxes merged using weighted mean)
        # if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            if not rotated:
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x_cc[i, :4] = torch.mm(weights, x_cc[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                ### xyxy
            else:
                boxes_d3d_i = boxes_d3d[i_mask]#.clone().contiguous()
                # print(i_mask)
                # print(i_mask.shape)
                # print(boxes_d3d_i.shape, boxes_d3d_i.dtype, boxes_d3d_i.device)
                # print(boxes_d3d.shape, boxes_d3d.dtype, boxes_d3d.device)
                iou = d3d.box.box2d_iou(boxes_d3d_i, boxes_d3d, method="rbox") #> iou_thres  # iou matrix                
                iou = iou > iou_thres
                weights = iou * scores[None]  # box weights
                if rotated_anchor:
                    x_xywh = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes 
                    v = yaw2v(x[:, 4])
                    x_v = torch.mm(weights, v).float() / weights.sum(1, keepdim=True)  # merged boxes 
                    x_r = v2yaw(x_v)
                    x_cc[i_mask, :4] = x_xywh
                    x_cc[i_mask, 4] = x_r
                    if tail:
                        dxdy = x[:, 5:7]
                        x0y0 = x[:, :2]
                        x1y1 = x0y0 + dxdy
                        x_tt = torch.mm(weights, x1y1).float() / weights.sum(1, keepdim=True)  # merged boxes 
                        x_cc[i_mask, 5:7] = x_tt - x_xywh[:, :2]
                else:
                    x_vcc = torch.mm(weights, x[:, :6]).float() / weights.sum(1, keepdim=True)  # merged boxes      
                    ### note that here we are averaging xywh and rot vec while in non-rotated case the average is on xyxy
                    x_r = v2yaw(x_vcc[:, 4:6])
                    x_cc[i_mask, :4] = x_vcc[:, :4]
                    x_cc[i_mask, 4] = x_r
                    if tail:
                        dxdy = x[:, 6:8]
                        x0y0 = x[:, :2]
                        x1y1 = x0y0 + dxdy
                        x_tt = torch.mm(weights, x1y1).float() / weights.sum(1, keepdim=True)  # merged boxes 
                        x_cc[i_mask, 5:7] = x_tt - x_vcc[:, :2]

                ### xywhr

                # i = i[iou.sum(1) > 1]  # require redundancy
            # except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
            #     if not rotated:
            #         print(x_cc, i, x_cc.shape, i.shape)
            #     else:
            #         print(x_cc, i_mask, x_cc.shape, i_mask.shape)
            #     pass
        if not rotated:
            output[xi] = x_cc[i]
        else:
            output[xi] = x_cc[i_mask]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def get_yolo_layers(model):
    bool_vec = [x['type'] == 'yolo' for x in model.module_defs]
    return [i for i, x in enumerate(bool_vec) if x]  # [82, 94, 106] for yolov3


def print_model_biases(model):
    # prints the bias neurons preceding each yolo layer
    print('\nModel Bias Summary: %8s%18s%18s%18s' % ('layer', 'regression', 'objectness', 'classification'))
    try:
        multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
        for l in model.yolo_layers:  # print pretrained biases
            if multi_gpu:
                na = model.module.module_list[l].na  # number of anchors
                b = model.module.module_list[l - 1][0].bias.view(na, -1)  # bias 3x85
            else:
                na = model.module_list[l].na
                b = model.module_list[l - 1][0].bias.view(na, -1)  # bias 3x85
            print(' ' * 20 + '%8g %18s%18s%18s' % (l, '%5.2f+/-%-5.2f' % (b[:, :4].mean(), b[:, :4].std()),
                                                   '%5.2f+/-%-5.2f' % (b[:, 4].mean(), b[:, 4].std()),
                                                   '%5.2f+/-%-5.2f' % (b[:, 5:].mean(), b[:, 5:].std())))
    except:
        pass


def strip_optimizer(f='weights/best.pt'):  # from utils.utils import *; strip_optimizer()
    # Strip optimizer from *.pt files for lighter files (reduced by 2/3 size)
    x = torch.load(f, map_location=torch.device('cpu'))
    x['optimizer'] = None
    print('Optimizer stripped from %s' % f)
    torch.save(x, f)


def create_backbone(f='weights/best.pt'):  # from utils.utils import *; create_backbone()
    # create a backbone from a *.pt file
    x = torch.load(f, map_location=torch.device('cpu'))
    x['optimizer'] = None
    x['training_results'] = None
    x['epoch'] = -1
    for p in x['model'].parameters():
        p.requires_grad = True
    s = 'weights/backbone.pt'
    print('%s saved as %s' % (f, s))
    torch.save(x, s)


def coco_class_count(path='../coco/labels/train2014/'):
    # Histogram of occurrences per class
    nc = 80  # number classes
    x = np.zeros(nc, dtype='int32')
    files = sorted(glob.glob('%s/*.*' % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        x += np.bincount(labels[:, 0].astype('int32'), minlength=nc)
        print(i, len(files))


def coco_only_people(path='../coco/labels/train2017/'):  # from utils.utils import *; coco_only_people()
    # Find images with only people
    files = sorted(glob.glob('%s/*.*' % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        if all(labels[:, 0] == 0):
            print(labels.shape[0], file)


def crop_images_random(path='../images/', scale=0.50):  # from utils.utils import *; crop_images_random()
    # crops images into random squares up to scale fraction
    # WARNING: overwrites images!
    for file in tqdm(sorted(glob.glob('%s/*.*' % path))):
        img = cv2.imread(file)  # BGR
        if img is not None:
            h, w = img.shape[:2]

            # create random mask
            a = 30  # minimum size (pixels)
            mask_h = random.randint(a, int(max(a, h * scale)))  # mask height
            mask_w = mask_h  # mask width

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random color mask
            cv2.imwrite(file, img[ymin:ymax, xmin:xmax])


def coco_single_class_labels(path='../coco/labels/train2014/', label_class=43):
    # Makes single-class coco datasets. from utils.utils import *; coco_single_class_labels()
    if os.path.exists('new/'):
        shutil.rmtree('new/')  # delete output folder
    os.makedirs('new/')  # make new output folder
    os.makedirs('new/labels/')
    os.makedirs('new/images/')
    for file in tqdm(sorted(glob.glob('%s/*.*' % path))):
        with open(file, 'r') as f:
            labels = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
        i = labels[:, 0] == label_class
        if any(i):
            img_file = file.replace('labels', 'images').replace('txt', 'jpg')
            labels[:, 0] = 0  # reset class to 0
            with open('new/images.txt', 'a') as f:  # add image to dataset list
                f.write(img_file + '\n')
            with open('new/labels/' + Path(file).name, 'a') as f:  # write label
                for l in labels[i]:
                    f.write('%g %.6f %.6f %.6f %.6f\n' % tuple(l))
            shutil.copyfile(src=img_file, dst='new/images/' + Path(file).name.replace('txt', 'jpg'))  # copy images


def kmean_anchors(path='./data/coco64.txt', n=9, img_size=(640, 640), thr=0.20, gen=1000):
    # Creates kmeans anchors for use in *.cfg files: from utils.utils import *; _ = kmean_anchors()
    # n: number of anchors
    # img_size: (min, max) image size used for multi-scale training (can be same values)
    # thr: IoU threshold hyperparameter used for training (0.0 - 1.0)
    # gen: generations to evolve anchors using genetic algorithm
    from utils.datasets import LoadImagesAndLabels

    def print_results(k):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        iou = wh_iou(wh, torch.Tensor(k))
        max_iou = iou.max(1)[0]
        bpr, aat = (max_iou > thr).float().mean(), (iou > thr).float().mean() * n  # best possible recall, anch > thr
        print('%.2f iou_thr: %.3f best possible recall, %.2f anchors > thr' % (thr, bpr, aat))
        print('n=%g, img_size=%s, IoU_all=%.3f/%.3f-mean/best, IoU>thr=%.3f-mean: ' %
              (n, img_size, iou.mean(), max_iou.mean(), iou[iou > thr].mean()), end='')
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg
        return k

    def fitness(k):  # mutation fitness
        iou = wh_iou(wh, torch.Tensor(k))  # iou
        max_iou = iou.max(1)[0]
        return (max_iou * (max_iou > thr).float()).mean()  # product

    # Get label wh
    wh = []
    dataset = LoadImagesAndLabels(path, augment=True, rect=True)
    nr = 1 if img_size[0] == img_size[1] else 10  # number augmentation repetitions
    for s, l in zip(dataset.shapes, dataset.labels):
        wh.append(l[:, 3:5] * (s / s.max()))  # image normalized to letterbox normalized wh
    wh = np.concatenate(wh, 0).repeat(nr, axis=0)  # augment 10x
    wh *= np.random.uniform(img_size[0], img_size[1], size=(wh.shape[0], 1))  # normalized to pixels (multi-scale)
    wh = wh[(wh > 2.0).all(1)]  # remove below threshold boxes (< 2 pixels wh)

    # Kmeans calculation
    from scipy.cluster.vq import kmeans
    print('Running kmeans for %g anchors on %g points...' % (n, len(wh)))
    s = wh.std(0)  # sigmas for whitening
    k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
    k *= s
    wh = torch.Tensor(wh)
    k = print_results(k)

    # # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.tight_layout()
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    npr = np.random
    f, sh, mp, s = fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    for _ in tqdm(range(gen), desc='Evolving anchors'):
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            print_results(k)
    k = print_results(k)

    return k


def print_mutation(hyp, results, bucket=''):
    # Print mutation results to evolve.txt (for use with train.py --evolve)
    a = '%10s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%10.3g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%10.4g' * len(results) % results  # results (P, R, mAP, F1, test_loss)
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))

    if bucket:
        os.system('gsutil cp gs://%s/evolve.txt .' % bucket)  # download evolve.txt

    with open('evolve.txt', 'a') as f:  # append result
        f.write(c + b + '\n')
    x = np.unique(np.loadtxt('evolve.txt', ndmin=2), axis=0)  # load unique rows
    np.savetxt('evolve.txt', x[np.argsort(-fitness(x))], '%10.3g')  # save sort by fitness

    if bucket:
        os.system('gsutil cp evolve.txt gs://%s' % bucket)  # upload evolve.txt


def apply_classifier(x, model, img, im0):
    # applies a second stage classifier to yolo outputs
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            # Classes
            pred_cls1 = d[:, 5].long()
            ims = []
            for j, a in enumerate(d):  # per item
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR
                # cv2.imwrite('test%i.jpg' % j, cutout)

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # classifier prediction
            x[i] = x[i][pred_cls1 == pred_cls2]  # retain matching class detections

    return x


def fitness(x):
    # Returns fitness (for use with results.txt or evolve.txt)
    w = [0.0, 0.01, 0.99, 0.00]  # weights for [P, R, mAP, F1]@0.5 or [P, R, mAP@0.5, mAP@0.5:0.95]
    return float((x[:, :4] * w).sum(1))


def output_to_target(output, width, height):
    """
    Convert a YOLO model output to target format
    [batch_id, class_id, x, y, w, h, conf]
    input is xyxy, conf, class
    """
    if isinstance(output, torch.Tensor):
        output = output.cpu().numpy()

    targets = []
    for i, o in enumerate(output):
        if o is not None:
            for pred in o:
                box = pred[:4]
                w = (box[2] - box[0]) / width
                h = (box[3] - box[1]) / height
                x = box[0] / width + w / 2
                y = box[1] / height + h / 2
                conf = pred[4]
                cls = int(pred[5])

                targets.append([i, cls, x, y, w, h, conf])

    return np.array(targets)

def output_to_target_r(output, width, height):
    """
    Convert a YOLO model output to target format
    [batch_id, class_id, x, y, w, h, r, conf]
    input is xywhr, conf, class
    """

    if isinstance(output, torch.Tensor):
        output = output.cpu().numpy()
    if isinstance(output, list):
        if any(isinstance(output[i], torch.Tensor) for i in range(len(output))):
            try:
                output = [output[i].cpu().numpy() if output[i] is not None else None for i in range(len(output)) ]
            except:
                print("output_to_target_r", output)
                raise ValueError("something happened")
        else:
            ### if all are none, return a 0*8 np array
            return np.ones((0,8))

    ### if we are here, then at least some item of output is not zero and is a valid numpy array
    targets = []
    for i, o in enumerate(output):
        if o is not None:
            for pred in o:
                box = pred[:4]
                w = box[2] / width
                h = box[3] / height
                x = box[0] / width
                y = box[1] / height
                yaw = pred[4]
                conf = pred[5]
                cls = int(pred[6])

                targets.append([i, cls, x, y, w, h, yaw, conf])

    return np.array(targets)


def output_to_target_rtt(output, width, height):
    """
    Convert a YOLO model output to target format
    [batch_id, class_id, x, y, w, h, r, dx, dy, conf]
    input is xywhr, conf, class
    """

    if isinstance(output, torch.Tensor):
        output = output.cpu().numpy()
    if isinstance(output, list):
        if any(isinstance(output[i], torch.Tensor) for i in range(len(output))):
            try:
                output = [output[i].cpu().numpy() if output[i] is not None else None for i in range(len(output)) ]
            except:
                print("output_to_target_rtt", output)
                raise ValueError("something happened")
        else:
            ### if all are none, return a 0*8 np array
            return np.ones((0,10))

    ### if we are here, then at least some item of output is not zero and is a valid numpy array
    targets = []
    for i, o in enumerate(output):
        if o is not None:
            for pred in o:
                box = pred[:4]
                w = box[2] / width
                h = box[3] / height
                x = box[0] / width
                y = box[1] / height
                yaw = pred[4]

                dx = pred[5] / width
                dy = pred[6] / height
                
                conf = pred[7]
                cls = int(pred[8])

                targets.append([i, cls, x, y, w, h, yaw, dx, dy, conf])

    return np.array(targets)

# Plotting functions ---------------------------------------------------------------------------------------------------
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    if isinstance(x, list):
        rotated = len(x) in [8, 10]
        tail = len(x) == 10
    elif isinstance(x, np.ndarray):
        rotated = x.shape[0] in [8, 10]
        tail = x.shape[0] == 10
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    if not rotated:
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    else:
        pts = x[:8].reshape(4,2).round().astype(int)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=tl, lineType=cv2.LINE_AA)
        if tail:
            pts0 = (pts[0] + pts[2])*0.5
            dudv = x[8:10]
            pts1 = pts0 + dudv
            pts_tail = np.stack([pts0, pts1], axis=0).round().astype(int)[None]
            cv2.polylines(img, pts_tail, isClosed=False, color=color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        if not rotated:
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        else:
            c1 = tuple(pts[0])
            c2 = (c1[0] + t_size[0]), (c1[1] - t_size[1] - 3)
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot_wh_methods():  # from utils.utils import *; plot_wh_methods()
    # Compares the two methods for width-height anchor multiplication
    # https://github.com/ultralytics/yolov3/issues/168
    x = np.arange(-4.0, 4.0, .1)
    ya = np.exp(x)
    yb = torch.sigmoid(torch.from_numpy(x)).numpy() * 2

    fig = plt.figure(figsize=(6, 3), dpi=150)
    plt.plot(x, ya, '.-', label='yolo method')
    plt.plot(x, yb ** 2, '.-', label='^2 power method')
    plt.plot(x, yb ** 2.5, '.-', label='^2.5 power method')
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=0, top=6)
    plt.xlabel('input')
    plt.ylabel('output')
    plt.legend()
    fig.tight_layout()
    fig.savefig('comparison.png', dpi=200)


def plot_images(images, targets, paths=None, fname='images.jpg', names=None, max_size=640, max_subplots=16, gt=True):
    ### targets: [batch_id, class_id, x, y, w, h, r, conf] (xywh are all normalized by width and height)
    ### no r if not rotated, no conf if gt

    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness
    if os.path.isfile(fname):  # do not overwrite
        return None

    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()

    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # if len(targets.shape) <2:
    #     # if gt:
    #     #     print("This batch does not have any target")
    #     #     print(targets)
    #     # else:
    #     #     print("This batch does not have any predicted targets")
    #     #     print(targets)
    #     #     ### this happens when output_to_target_r outputs a []
                ### now it should not happen at all
    #     return


    rotated = targets.shape[1] in [7,9] if gt else targets.shape[1] in [8,10]
    tail = targets.shape[1] == 9 if gt else targets.shape[1] == 10

    # un-normalise
    if np.max(images[0]) <= 1:
        images *= 255

    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    # Empty array for output
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)

    # Fix class - colour map
    prop_cycle = plt.rcParams['axes.prop_cycle']
    # https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    hex2rgb = lambda h: tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
    color_lut = [hex2rgb(h) for h in prop_cycle.by_key()['color']]

    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img
        if len(targets) > 0:
            image_targets = targets[targets[:, 0] == i].copy()

            # if len(image_targets) <= 0:
            #     print("current image does not have targets detected", i)
            #     # print(targets)
            #     # print(image_targets)
            if len(image_targets) > 0:
                
                ### scaling should happen before rotation, if rotated
                image_targets[:, [2,4]] *= w
                image_targets[:, 2] += block_x
                image_targets[:, [3,5]] *= h
                image_targets[:, 3] += block_y

                if rotated:
                    boxes = xywh2xyxy_r(image_targets[:, 2:7]).T    # conf is the last dim so not included here # 8*n
                    ### the return is n*8 xyxyxyxy
                    if tail:
                        image_targets[:, 7] *= w
                        image_targets[:, 8] *= h
                        boxes = np.concatenate([boxes, image_targets[:, 7].reshape(1,-1), image_targets[:, 8].reshape(1,-1)], axis=0)   # 10*n
                else:
                    boxes = xywh2xyxy(image_targets[:, 2:6]).T  # T means now 4*n
                classes = image_targets[:, 1].astype('int')
                # gt = image_targets.shape[1] == 6  # ground truth if no conf column
                conf = None if gt else image_targets[:, -1]  # check for confidence presence (gt vs pred)
                
                for j, box in enumerate(boxes.T):
                    cls = int(classes[j])
                    color = color_lut[cls % len(color_lut)]
                    if names is not None and len(names) > 1:
                        cls = names[cls] if names else cls
                        # if gt or conf[j] > 0.3:  # 0.3 conf thresh    ## we do not use conf thresh here because the predictions are already filtered by nms
                        label = '%s' % cls if gt else '%s %.1f' % (cls, conf[j])
                    else:
                        label = None if gt else '%.1f'%conf[j]
                    plot_one_box(box, mosaic, label=label, color=color, line_thickness=tl)

        # Draw image filename labels
        if paths is not None:
            label = os.path.basename(paths[i])[:40]  # trim to 40 char
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            cv2.putText(mosaic, label, (block_x + 5, block_y + t_size[1] + 5), 0, tl / 3, [220, 220, 220], thickness=tf,
                        lineType=cv2.LINE_AA)

        # Image border
        cv2.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h), (255, 255, 255), thickness=3)

    if fname is not None:
        mosaic = cv2.resize(mosaic, (int(ns * w * 0.5), int(ns * h * 0.5)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))

    return mosaic


def plot_lr_scheduler(optimizer, scheduler, epochs=300):
    # Plot LR simulating training for full epochs
    optimizer, scheduler = copy(optimizer), copy(scheduler)  # do not modify originals
    y = []
    for _ in range(epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]['lr'])
    plt.plot(y, '.-', label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.tight_layout()
    plt.savefig('LR.png', dpi=200)


def plot_test_txt():  # from utils.utils import *; plot_test()
    # Plot test.txt histograms
    x = np.loadtxt('test.txt', dtype=np.float32)
    box = xyxy2xywh(x[:, :4])
    cx, cy = box[:, 0], box[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    ax.hist2d(cx, cy, bins=600, cmax=10, cmin=0)
    ax.set_aspect('equal')
    plt.savefig('hist2d.png', dpi=300)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax[0].hist(cx, bins=600)
    ax[1].hist(cy, bins=600)
    plt.savefig('hist1d.png', dpi=200)


def plot_targets_txt():  # from utils.utils import *; plot_targets_txt()
    # Plot targets.txt histograms
    x = np.loadtxt('targets.txt', dtype=np.float32).T
    s = ['x targets', 'y targets', 'width targets', 'height targets']
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    for i in range(4):
        ax[i].hist(x[i], bins=100, label='%.3g +/- %.3g' % (x[i].mean(), x[i].std()))
        ax[i].legend()
        ax[i].set_title(s[i])
    plt.savefig('targets.jpg', dpi=200)


def plot_labels(labels):
    # plot dataset labels
    c, b = labels[:, 0], labels[:, 1:].transpose()  # classees, boxes

    def hist2d(x, y, n=100):
        xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
        hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
        xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
        yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
        return hist[xidx, yidx]

    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    ax[0].hist(c, bins=int(c.max() + 1))
    ax[0].set_xlabel('classes')
    ax[1].scatter(b[0], b[1], c=hist2d(b[0], b[1], 90), cmap='jet')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[2].scatter(b[2], b[3], c=hist2d(b[2], b[3], 90), cmap='jet')
    ax[2].set_xlabel('width')
    ax[2].set_ylabel('height')
    plt.savefig('labels.png', dpi=200)


def plot_evolution_results(hyp):  # from utils.utils import *; plot_evolution_results(hyp)
    # Plot hyperparameter evolution results in evolve.txt
    x = np.loadtxt('evolve.txt', ndmin=2)
    f = fitness(x)
    # weights = (f - f.min()) ** 2  # for weighted results
    fig = plt.figure(figsize=(12, 10), tight_layout=True)
    matplotlib.rc('font', **{'size': 8})
    for i, (k, v) in enumerate(hyp.items()):
        y = x[:, i + 7]
        # mu = (y * weights).sum() / weights.sum()  # best weighted result
        mu = y[f.argmax()]  # best single result
        plt.subplot(4, 5, i + 1)
        plt.plot(mu, f.max(), 'o', markersize=10)
        plt.plot(y, f, '.')
        plt.title('%s = %.3g' % (k, mu), fontdict={'size': 9})  # limit to 40 characters
        print('%15s: %.3g' % (k, mu))
    plt.savefig('evolve.png', dpi=200)


def plot_results_overlay(start=0, stop=0):  # from utils.utils import *; plot_results_overlay()
    # Plot training results files 'results*.txt', overlaying train and val losses
    s = ['train', 'train', 'train', 'Precision', 'mAP@0.5', 'val', 'val', 'val', 'Recall', 'F1']  # legends
    t = ['GIoU', 'Objectness', 'Classification', 'P-R', 'mAP-F1']  # titles
    for f in sorted(glob.glob('results*.txt') + glob.glob('../../Downloads/results*.txt')):
        results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
        n = results.shape[1]  # number of rows
        x = range(start, min(stop, n) if stop else n)
        fig, ax = plt.subplots(1, 5, figsize=(14, 3.5), tight_layout=True)
        ax = ax.ravel()
        for i in range(5):
            for j in [i, i + 5]:
                y = results[j, x]
                if i in [0, 1, 2]:
                    y[y == 0] = np.nan  # dont show zero loss values
                ax[i].plot(x, y, marker='.', label=s[j])
            ax[i].set_title(t[i])
            ax[i].legend()
            ax[i].set_ylabel(f) if i == 0 else None  # add filename
        fig.savefig(f.replace('.txt', '.png'), dpi=200)


def plot_results(start=0, stop=0, bucket='', id=()):  # from utils.utils import *; plot_results()
    # Plot training 'results*.txt' as seen in https://github.com/ultralytics/yolov3#training
    # fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ### for compatible with rotated mode
    # fig, ax = plt.subplots(2, 8, figsize=(20, 6), tight_layout=True)
    fig, ax = plt.subplots(2, 7, figsize=(16, 6), tight_layout=True)
    ### figsize is the overall size instead of subsize. This size times dpi at the end is the rendered figure size
    ax = ax.ravel()
    # s = ['GIoU', 'Objectness', 'Classification', 'Precision', 'Recall',
    #      'val GIoU', 'val Objectness', 'val Classification', 'mAP@0.5', 'F1']
    ### for compatible with rotated mode
    # s = ['Box', 'Objectness', 'Classification', 'Position','Dimension','Direction', 'Precision', 'Recall',
    #      'val Box', 'val Objectness', 'val Classification', 'val Position','val Dimension','val Direction', 'AP', 'F1']
    s = ['Box', 'Objectness', 'Position','Dimension','Direction', 'Precision', 'Recall',
         'val Box', 'val Objectness', 'val Position','val Dimension','val Direction', 'AP', 'F1']
    if bucket:
        os.system('rm -rf storage.googleapis.com')
        files = ['https://storage.googleapis.com/%s/results%g.txt' % (bucket, x) for x in id]
    else:
        files = glob.glob('results/results*.txt') + glob.glob('../../Downloads/results*.txt')
    for f in sorted(files):
        # try:
        # results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
        ### for compatible with rotated mode
        # results = np.loadtxt(f, usecols=[2, 3, 4, 5, 6, 7, 11, 12, 15, 16, 17, 18, 19, 20, 13, 14], ndmin=2).T
        results = np.loadtxt(f, usecols=[2, 3, 5, 6, 7, 12, 13, 16, 17, 19, 20, 21, 14, 15], ndmin=2, skiprows=1).T

        n = results.shape[1]  # number of rows
        x = range(start, min(stop, n) if stop else n)
        # for i in range(10):
        ### for compatible with rotated mode
        # for i in range(16):
        for i in range(14):
            y = results[i, x]
            # if i in [0, 1, 2, 5, 6, 7]:
            ### for compatible with rotated mode
            # if i in [0, 1, 2, 8, 9, 10]:
            if i in [0, 1, 8, 9]:
                y[y == 0] = np.nan  # dont show zero loss values
                # y /= y[0]  # normalize
            # y[y == 0] = np.nan  # dont show zero loss values
            ax[i].plot(x, y, marker='.', label=Path(f).stem[8:], linewidth=2, markersize=8)
            ### [8:] is to remove "results_" substring
            ax[i].set_title(s[i])
            # if i in [5, 6, 7]:  # share train and val loss y axes
            #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        # except:
        #     print('Warning: Plotting error for %s, skipping file' % f)

    ax[1].legend()
    # fig.savefig('results.png', dpi=200)
    fig.savefig('results.png', dpi=100)

def choose_cfg_by_args(args):
    """
    Set opt.cfg according to other options, so that when running the program, you do not need to specify the cfg file to use manually. 
    """

    assert "yolov3-spp" in args.cfg, "Only yolov3-spp*.cfg is tested with rotated bounding box detection. Now {} is given".format(args.cfg)
    args.cfg = "yolov3-spp"
    if args.rotated_anchor:
        args.cfg += "-rbox"
    if args.tail:
        args.cfg += "tt"
    if args.dual_view:
        args.cfg += "dv"
    if args.half_angle:
        args.cfg += "180"
    args.cfg += ".cfg"
    return args