#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
from sklearn.cluster import DBSCAN
from torch import Tensor

### depth loss ###
def normalize_depth(depth: Tensor, max_depth: float = 80.0):
    return torch.clamp(depth / max_depth, 0.0, 1.0)

def compute_depth(
    loss_type,
    pred_depth: Tensor,
    gt_depth: Tensor,
    max_depth: float = 80,
):
    pred_depth = pred_depth.squeeze()
    gt_depth = gt_depth.squeeze()
    valid_mask = (gt_depth > 0.01) & (gt_depth < max_depth)
    pred_depth = normalize_depth(pred_depth[valid_mask], max_depth=max_depth)
    gt_depth = normalize_depth(gt_depth[valid_mask], max_depth=max_depth)
    if loss_type == "smooth_l1":
        loss =  F.smooth_l1_loss(pred_depth, gt_depth, reduction="none")
        return loss.mean()
    elif loss_type == "l1":
        loss = F.l1_loss(pred_depth, gt_depth, reduction="none")
        return loss.mean()
    elif loss_type == "l2":
        loss = F.mse_loss(pred_depth, gt_depth, reduction="none")
        return loss.mean()
    else:
        raise NotImplementedError(f"Unknown loss type: {loss_type}")

def l1_loss_withmask(network_output, gt, mask):
    return torch.abs((network_output - gt) * mask).mean()

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def print_info(x, name='x'):
    check_nan(x, x.grad, name=name)

def check_nan(x, grad, name='x'):
    # check nan for single tensor
    if torch.isnan(x).any():
        print(f"\n{name} has nan")
    else:
        print(f"\n{name} norm: {torch.norm(x)}")
    # grad
    if grad is None:
        print(f"{name} has no grad")
    else:
        if torch.isnan(grad).any():
            print(f"{name} has nan grad")
        else:
            print(f"{name} grad-norm: {torch.norm(grad)}")

def has_hook(x):
    if x._backward_hooks is None:
        return False
    else:
        return len(x._backward_hooks) > 0

def register_grad_hook(x, name='x'):
    #def hook(grad):
    #    print(f"attr {name} grad-norm: {torch.norm(grad)}")
    hook = lambda grad: check_nan(x, grad, name=name)
    # 检查是否已经注册过 hook
    if not has_hook(x):
        #print(f"attr {name} has registered hook")
        handle = x.register_hook(hook)
        # handle.remove() # 用于移除 hook

def check_gs_nan(gaussian):
    # check nan for gaussian
    for group in gaussian.optimizer.param_groups:
        # check nan
        for p in group["params"]:
            # value
            if torch.isnan(p).any():
                print(f"\nattr {group['name']} has nan grad")
                #continue
            else:
                print(f"\nattr {group['name']} norm: {torch.norm(p)}")
            # grad
            if p.grad is None:
                print(f"attr {group['name']} has no grad")
            else:
                if torch.isnan(p.grad).any():
                    print(f"attr {group['name']} has nan grad")
                    #continue
                else:
                    #print(f"attr {group['name']} is normal")     
                    #print(f"\nattr {group['name']} is normal, norm: {torch.norm(p)}, grad-norm: {torch.norm(p.grad)}")  
                    print(f"attr {group['name']} grad-norm: {torch.norm(p.grad)}")             
    print(" --------------- check nan done --------------- ") 
