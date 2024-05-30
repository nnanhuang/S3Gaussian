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
import sys
from datetime import datetime
import numpy as np
import random
import open3d as o3d

def visualize_points(points, aabb_center=None, aabb_size=None):
    # Create a PointCloud object
    pcd = o3d.geometry.PointCloud()

    # Set the points
    pcd.points = o3d.utility.Vector3dVector(points)

    if aabb_center is not None:
        # 可视化aabb  o3d 的aabb 并不是这么用的 输入应该是点云
        #aabb = o3d.geometry.AxisAlignedBoundingBox(aabb_center - aabb_size / 2, aabb_center + aabb_size / 2)
        # 可视化aabb 用立方体
        aabb = o3d.geometry.OrientedBoundingBox(aabb_center, np.eye(3), aabb_size)
        o3d.visualization.draw_geometries([pcd, aabb])
        #o3d.visualization.draw_geometries([aabb])
    else:
        ## Visualize the points
        o3d.visualization.draw_geometries([pcd])

def get_OccGrid(pts, aabb, occ_voxel_size):
    # 计算网格的大小
    grid_size = np.ceil((aabb[1] - aabb[0]) / occ_voxel_size).astype(int)
    assert pts.min() >= aabb[0].min() and pts.max() <= aabb[1].max(), "Points are outside the AABB"

    # 创建一个空的网格
    voxel_grid = np.zeros(grid_size, dtype=np.uint8)

    # 将点云转换为网格坐标
    grid_pts = ((pts - aabb[0]) / occ_voxel_size).astype(int)

    # 将网格中的点设置为1
    voxel_grid[grid_pts[:, 0], grid_pts[:, 1], grid_pts[:, 2]] = 1

    # check
    #voxel_coords = np.floor((pts - aabb[0]) / occ_voxel_size).astype(int)
    #occ = voxel_grid[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]]

    return voxel_grid


def sample_on_aabb_surface(aabb_center, aabb_size, n_pts=1000, above_half=False):
    """
    0:立方体的左面(x轴负方向)
    1:立方体的右面(x轴正方向)
    2:立方体的下面(y轴负方向)
    3:立方体的上面(y轴正方向)
    4:立方体的后面(z轴负方向)
    5:立方体的前面(z轴正方向)
    """
    # Choose a face randomly
    faces = np.random.randint(0, 6, size=n_pts)

    # Generate two random numbers
    r_ = np.random.random((n_pts, 2))

    # Create an array to store the points
    points = np.zeros((n_pts, 3))

    # Define the offsets for each face
    offsets = np.array([
        [-aabb_size[0]/2, 0, 0],
        [aabb_size[0]/2, 0, 0],
        [0, -aabb_size[1]/2, 0],
        [0, aabb_size[1]/2, 0],
        [0, 0, -aabb_size[2]/2],
        [0, 0, aabb_size[2]/2]
    ])

    # Define the scales for each face
    scales = np.array([
        [aabb_size[1], aabb_size[2]],
        [aabb_size[1], aabb_size[2]],
        [aabb_size[0], aabb_size[2]],
        [aabb_size[0], aabb_size[2]],
        [aabb_size[0], aabb_size[1]],
        [aabb_size[0], aabb_size[1]]
    ])

    # Define the positions of the zero column for each face
    zero_column_positions = [0, 0, 1, 1, 2, 2]
    # Define the indices of the aabb_size components for each face
    aabb_size_indices = [[1, 2], [1, 2], [0, 2], [0, 2], [0, 1], [0, 1]]
    # Calculate the coordinates of the points for each face
    for i in range(6):
        mask = faces == i
        r_scaled = r_[mask] * scales[i]
        r_scaled = np.insert(r_scaled, zero_column_positions[i], 0, axis=1)
        aabb_size_adjusted = np.insert(aabb_size[aabb_size_indices[i]] / 2, zero_column_positions[i], 0)
        points[mask] = aabb_center + offsets[i] + r_scaled - aabb_size_adjusted
        #visualize_points(points[mask], aabb_center, aabb_size)
    #visualize_points(points, aabb_center, aabb_size)
        
    # 提取上半部分的点
    if above_half:
        points = points[points[:, -1] > aabb_center[-1]]
    return points

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def DepthMaptoTorch(depth_map):
    resized_image = torch.from_numpy(depth_map) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
def ObjectPILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) 
    #max_val = resized_image.max()
    #if max_val > 0:
    #    resized_image = resized_image / max_val
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func_after_iter(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000,
    after_iter=0,
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0) or step < after_iter:
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def get_piecewise_lr_func(
    lr_init, zero_intervals = [(0, 500), (500, 5000)],
):
    """
        分段常数 学习率, 控制在 特定区间内 学习率为0
    """
    def helper(step):
        if len(zero_intervals) == 0:
            return lr_init
        for start, end in zero_intervals:
            if start <= step < end:
                return 0
            else:
                return lr_init

    return helper


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
