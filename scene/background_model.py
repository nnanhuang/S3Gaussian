""" from EMER-NeRF """
from torch import Tensor
import torch.nn as nn
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from .encodings import SinusoidalEncoder
from typing import Dict, Optional, Union
import math
from scene.gaussian_model import GaussianModel

from utils.general_utils import sample_on_aabb_surface
from utils.sh_utils import SH2RGB
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import matplotlib.pyplot as plt

def min_max(x, name='x', print_info=True):
    print(f"{name} range : {x.min().item():.7f} ~ {x.max().item():.7f}")
    # 如果x 是一维 
    if x.dim() == 1:
        x = x.unsqueeze(1)
    out = torch.cat((x.min(dim=0).values, x.max(dim=0).values), dim=0)
    return out
def get_rays(
    x: Tensor, y: Tensor, c2w: Tensor, intrinsic: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Args:
        x: the horizontal coordinates of the pixels, shape: (num_rays,)
        y: the vertical coordinates of the pixels, shape: (num_rays,)
        c2w: the camera-to-world matrices, shape: (num_cams, 4, 4)
        intrinsic: the camera intrinsic matrices, shape: (num_cams, 3, 3)
    Returns:
        origins: the ray origins, shape: (num_rays, 3)
        viewdirs: the ray directions, shape: (num_rays, 3)
        direction_norm: the norm of the ray directions, shape: (num_rays, 1)
    """
    if len(intrinsic.shape) == 2:
        intrinsic = intrinsic[None, :, :]
    if len(c2w.shape) == 2:
        c2w = c2w[None, :, :]
    camera_dirs = torch.nn.functional.pad(
        torch.stack(
            [
                (x - intrinsic[:, 0, 2] + 0.5) / intrinsic[:, 0, 0],
                (y - intrinsic[:, 1, 2] + 0.5) / intrinsic[:, 1, 1],
            ],
            dim=-1,
        ),
        (0, 1),
        value=1.0,
    )  # [num_rays, 3]

    # rotate the camera rays w.r.t. the camera pose
    directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
    origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
    # TODO: not sure if we still need direction_norm
    direction_norm = torch.linalg.norm(directions, dim=-1, keepdims=True)
    # normalize the ray directions
    viewdirs = directions / (direction_norm + 1e-8)
    return origins, viewdirs, direction_norm

def get_directions(
    x: Tensor, y: Tensor, c2w: Tensor, intrinsic: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Args:
        x: the horizontal coordinates of the pixels, shape: (num_rays,)
        y: the vertical coordinates of the pixels, shape: (num_rays,)
        c2w: the camera-to-world matrices, shape: (num_cams, 4, 4)
        intrinsic: the camera intrinsic matrices, shape: (num_cams, 3, 3)
    Returns:
        origins: the ray origins, shape: (num_rays, 3)
        viewdirs: the ray directions, shape: (num_rays, 3)
        direction_norm: the norm of the ray directions, shape: (num_rays, 1)
    """
    if len(intrinsic.shape) == 2:
        intrinsic = intrinsic[None, :, :]
    if len(c2w.shape) == 2:
        c2w = c2w[None, :, :]
    camera_dirs = torch.nn.functional.pad(
        torch.stack(
            [
                (x - intrinsic[:, 0, 2] + 0.5) / intrinsic[:, 0, 0],
                (y - intrinsic[:, 1, 2] + 0.5) / intrinsic[:, 1, 1],
            ],
            dim=-1,
        ),
        (0, 1),
        value=1.0,
    )  # [num_rays, 3]

    # rotate the camera rays w.r.t. the camera pose
    directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
    # TODO: not sure if we still need direction_norm
    direction_norm = torch.linalg.norm(directions, dim=-1, keepdims=True)
    # normalize the ray directions
    viewdirs = directions / (direction_norm + 1e-8)
    return viewdirs

def ray_box_intersect(rays_o: torch.Tensor, rays_d: torch.Tensor, r: Union[float, torch.Tensor]) -> torch.Tensor:
    """ Calculate intersections of each rays and each boxes

    Args:
        rays_o (torch.Tensor): [B,3]
        rays_d (torch.Tensor): [B,3]
        r (Union[float, torch.Tensor]): [(B,)P] half side-lengths of boxes

    Returns:
        torch.Tensor: [B,P] depths of intersections along rays
    """
    # Expand to (B, 1, 3)
    rays_o = rays_o.unsqueeze(1)
    rays_d = rays_d.unsqueeze(1)

    # Expand to ([B, ]P, 1)
    r = r[..., None]

    # t_min, t_max: (B, P, 3)
    t_min = (-r - rays_o) / rays_d
    t_max = (r - rays_o) / rays_d

    # t_near, t_far: (B, P)
    t_near = torch.minimum(t_min, t_max).max(dim=-1).values
    t_far = torch.maximum(t_min, t_max).min(dim=-1).values

    # Check if rays are inside boxes and boxes are in front of the ray origin
    mask_intersect = (t_far > t_near) & (t_far > 0)
    t_far[~mask_intersect] = math.nan
    return t_far


class MLP(nn.Module):
    """A simple MLP with skip connections."""

    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        num_layers: int = 3,
        hidden_dims: Optional[int] = 256,
        skip_connections: Optional[Tuple[int]] = [0],
    ) -> None:
        super().__init__()
        self.in_dims = in_dims
        self.hidden_dims = hidden_dims
        self.n_output_dims = out_dims
        self.num_layers = num_layers
        self.skip_connections = skip_connections
        layers = []
        if self.num_layers == 1:
            layers.append(nn.Linear(in_dims, out_dims))
        else:
            for i in range(self.num_layers - 1):
                if i == 0:
                    layers.append(nn.Linear(in_dims, hidden_dims))
                elif i in skip_connections:
                    layers.append(nn.Linear(in_dims + hidden_dims, hidden_dims))
                else:
                    layers.append(nn.Linear(hidden_dims, hidden_dims))
            layers.append(nn.Linear(hidden_dims, out_dims))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        input = x
        for i, layer in enumerate(self.layers):
            if i in self.skip_connections:
                x = torch.cat([x, input], -1)
            x = layer(x)
            if i < len(self.layers) - 1:
                x = nn.functional.relu(x)
        return x

class ShellModel(nn.Module):
    """
        find_intersection + grid_interpolation --> sky_feat
    """
    def __init__(self, model_cfg: dict):
        super().__init__()
        pass

class SkyModel(nn.Module):
    """
        dir_encoding + MLP --> sky_feat
    """
    def __init__(self, model_cfg: dict = {"mlp_width": 256,}):
        super().__init__()
        self.model_cfg = model_cfg
        self.direction_encoding = SinusoidalEncoder(
            n_input_dims=3, min_deg=0, max_deg=4
        )
        self.sky_head = MLP(
            in_dims=self.direction_encoding.n_output_dims,
            out_dims=3,
            num_layers=3,
            hidden_dims=model_cfg['mlp_width'],
            skip_connections=[1],
        )
    def forward(self, directions):
        dd = self.direction_encoding(directions)
        sky_feat = self.sky_head(dd)
        #rgb_sky = F.sigmoid(sky_feat)
        return sky_feat


class BackgroundGaussianModel(GaussianModel):
    """ 
        Gaussian model for background
    - 输入 aabb
    - 增加了 xyz 的映射 ,  将坐标 固定在指定范围内: aabb 之外
    - 
    """

    def print_gs_info(self, hist_bins='auto', hist_density=True, hist_img_path=None):
        pass
        print('------------------ GS info ------------------')
        print("Number of points : ", self.get_xyz.shape[0])
        # -------------------------- get data -----------------------------------
        # scale
        scale_before_activation = self._scaling
        scale = self.get_scaling
        # rotation: 四元数 用 norm 来归一化
        rotation_before_activation = self._rotation
        rotation = self.get_rotation
        # opacity
        opacity_before_activation = self._opacity
        opacity = self.get_opacity
        # xyz
        xyz = self.get_xyz
        # -------------------------- static-info -----------------------------------
        # ----------------- min-max range -----------------
        #scale_before_activation_range = min_max(scale_before_activation, 'scale_before_activation')
        scale_x_before_activation_range = min_max(scale_before_activation[:, 0:1], 'scale_x_before_activation')
        scale_y_before_activation_range = min_max(scale_before_activation[:, 1:2], 'scale_y_before_activation')
        scale_z_before_activation_range = min_max(scale_before_activation[:, 2:3], 'scale_z_before_activation')
        #scale_range = min_max(scale, 'scale')
        scale_x_range = min_max(scale[:, 0:1], 'scale_x')
        scale_y_range = min_max(scale[:, 1:2], 'scale_y')
        scale_z_range = min_max(scale[:, 2:3], 'scale_z')
        # scale ratio # 长轴 与 短轴之比
        min_axis = torch.min(scale, dim=1)[0]
        max_axis = torch.max(scale, dim=1)[0]
        scale_ratio = max_axis / min_axis
        scale_ratio_range = min_max(scale_ratio, 'scale_ratio')
        # rot 暂不考虑
        #rotation_before_activation_range = min_max(rotation_before_activation, 'rotation_before_activation')
        #rotation_range = min_max(rotation, 'rotation')
        opacity_before_activation_range = min_max(opacity_before_activation, 'opacity_before_activation')
        opacity_range = min_max(opacity, 'opacity')
        #xyz_range = min_max(xyz, 'xyz')
        x_range = min_max(xyz[:, 0:1], 'x')
        y_range = min_max(xyz[:, 1:2], 'y')
        z_range = min_max(xyz[:, 2:3], 'z')
        # ----------------- hist -----------------
        #scale_before_activation_hist = np.histogram(scale_before_activation.cpu().numpy(), bins=hist_bins, density=hist_density)
        scale_x_before_activation_hist = np.histogram(scale_before_activation[:, 0:1].cpu().numpy(), bins=hist_bins, density=hist_density)
        scale_y_before_activation_hist = np.histogram(scale_before_activation[:, 1:2].cpu().numpy(), bins=hist_bins, density=hist_density)
        scale_z_before_activation_hist = np.histogram(scale_before_activation[:, 2:3].cpu().numpy(), bins=hist_bins, density=hist_density)
        #scale_hist = np.histogram(scale.cpu().numpy(), bins=hist_bins, density=hist_density)
        scale_x_hist = np.histogram(scale[:, 0:1].cpu().numpy(), bins=hist_bins, density=hist_density)
        scale_y_hist = np.histogram(scale[:, 1:2].cpu().numpy(), bins=hist_bins, density=hist_density)
        scale_z_hist = np.histogram(scale[:, 2:3].cpu().numpy(), bins=hist_bins, density=hist_density)
        scale_ratio_hist = np.histogram(scale_ratio.cpu().numpy(), bins=hist_bins, density=hist_density)
        #rotation_before_activation_hist = np.histogram(rotation_before_activation.cpu().numpy(), bins=hist_bins, density=hist_density)
        #rotation_hist = np.histogram(rotation.cpu().numpy(), bins=hist_bins, density=hist_density)
        opacity_before_activation_hist = np.histogram(opacity_before_activation.cpu().numpy(), bins=hist_bins, density=hist_density)
        opacity_hist = np.histogram(opacity.cpu().numpy(), bins=hist_bins, density=hist_density)
        #xyz_hist = np.histogram(xyz.cpu().numpy(), bins=hist_bins, density=hist_density)
        x_hist = np.histogram(xyz[:, 0:1].cpu().numpy(), bins=hist_bins, density=hist_density)
        y_hist = np.histogram(xyz[:, 1:2].cpu().numpy(), bins=hist_bins, density=hist_density)
        z_hist = np.histogram(xyz[:, 2:3].cpu().numpy(), bins=hist_bins, density=hist_density)
        # collect_info
        info_dict = {
            'scale_x_before_activation_range': scale_x_before_activation_range,
            'scale_y_before_activation_range': scale_y_before_activation_range,
            'scale_z_before_activation_range': scale_z_before_activation_range,
            'scale_x_range': scale_x_range,
            'scale_y_range': scale_y_range,
            'scale_z_range': scale_z_range,
            'scale_ratio_range': scale_ratio_range,
            'opacity_before_activation_range': opacity_before_activation_range,
            'opacity_range': opacity_range,
            'x_range': x_range,
            'y_range': y_range,
            'z_range': z_range,
            'scale_x_before_activation_hist': scale_x_before_activation_hist,
            'scale_y_before_activation_hist': scale_y_before_activation_hist,
            'scale_z_before_activation_hist': scale_z_before_activation_hist,
            'scale_x_hist': scale_x_hist,
            'scale_y_hist': scale_y_hist,
            'scale_z_hist': scale_z_hist,
            'scale_ratio_hist': scale_ratio_hist,
            'opacity_before_activation_hist': opacity_before_activation_hist,
            'opacity_hist': opacity_hist,
            'x_hist': x_hist,
            'y_hist': y_hist,
            'z_hist': z_hist,
        }
        
        if hist_img_path is not None:
            # plot hist
            fig, axes = plt.subplots(4, 3, figsize=(15, 15))
            axes[0, 0].hist(scale_x_before_activation_hist[1][:-1], scale_x_before_activation_hist[1], weights=scale_x_before_activation_hist[0])
            axes[0, 0].set_title('scale_x_before_activation_hist')
            axes[0, 1].hist(scale_y_before_activation_hist[1][:-1], scale_y_before_activation_hist[1], weights=scale_y_before_activation_hist[0])
            axes[0, 1].set_title('scale_y_before_activation_hist')
            axes[0, 2].hist(scale_z_before_activation_hist[1][:-1], scale_z_before_activation_hist[1], weights=scale_z_before_activation_hist[0])
            axes[0, 2].set_title('scale_z_before_activation_hist')
            axes[1, 0].hist(scale_x_hist[1][:-1], scale_x_hist[1], weights=scale_x_hist[0])
            axes[1, 0].set_title('scale_x_hist')
            axes[1, 1].hist(scale_y_hist[1][:-1], scale_y_hist[1], weights=scale_y_hist[0])
            axes[1, 1].set_title('scale_y_hist')
            axes[1, 2].hist(scale_z_hist[1][:-1], scale_z_hist[1], weights=scale_z_hist[0])
            axes[1, 2].set_title('scale_z_hist')
            axes[2, 0].hist(scale_ratio_hist[1][:-1], scale_ratio_hist[1], weights=scale_ratio_hist[0])
            axes[2, 0].set_title('scale_ratio_hist')
            axes[2, 1].hist(opacity_before_activation_hist[1][:-1], opacity_before_activation_hist[1], weights=opacity_before_activation_hist[0])
            axes[2, 1].set_title('opacity_before_activation_hist')
            axes[2, 2].hist(opacity_hist[1][:-1], opacity_hist[1], weights=opacity_hist[0])
            axes[2, 2].set_title('opacity_hist')
            axes[3, 0].hist(x_hist[1][:-1], x_hist[1], weights=x_hist[0])
            axes[3, 0].set_title('x_hist')
            axes[3, 1].hist(y_hist[1][:-1], y_hist[1], weights=y_hist[0])
            axes[3, 1].set_title('y_hist')
            axes[3, 2].hist(z_hist[1][:-1], z_hist[1], weights=z_hist[0])
            axes[3, 2].set_title('z_hist')
            plt.savefig(hist_img_path)
            plt.close()
            print(f'hist_img_path: {hist_img_path}')

        return info_dict
    def training_setup_no_feat(self, training_args):
        self.percent_dense = training_args.bg_percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # 相当于是 基于相机运动轨迹 or 基于场景大小  的一定比例 来确定 xyz 的学习率
        # 但如果 我对 xyz 进行了归一化, 那么就应该避免 场景尺度的学习
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def training_setup(self, training_args):
        self.percent_dense = training_args.bg_percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        if self._language_feature is None or self._language_feature.shape[0] != self._xyz.shape[0]:
            # 开始feature训练的时候，往模型中加入language feature参数
            language_feature = torch.zeros((self._xyz.shape[0], training_args.feat_dim), device="cuda")
            self._language_feature = nn.Parameter(language_feature.requires_grad_(True))
            # 初始化 language feature
            self._language_feature.data.uniform_(-1.0 / self.num_panoptic_objects,
                                                  1.0 / self.num_panoptic_objects)

        # 与 codebook-embedding 的 dim-norm 的情况 保持一致
        self.with_dim_norm = training_args.gs_dim_norm
        self.with_render_dim_norm = training_args.render_gs_dim_norm
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._language_feature], 'lr': training_args.language_feature_lr, "name": "language_feature"}, # TODO: training_args.language_feature_lr
            # bg-gs 不需要 codebook
            #{'params': self.feat_conv.parameters(), 'lr': training_args.feat_conv_lr, "name": "feat_conv"},
        ]
        # bg-gs 不需要 背景模型
        self.bg_model = self.bg_gs = None
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # bg-gs 不需要 multi-scheduler
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, init_opacity = 0.1):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of BackGround points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(init_opacity * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

def min_max(x, name='x', print_info=True):
    print(f"{name} range : {x.min().item():.7f} ~ {x.max().item():.7f}")
    # 如果x 是一维 
    if x.dim() == 1:
        x = x.unsqueeze(1)
    out = torch.cat((x.min(dim=0).values, x.max(dim=0).values), dim=0)
    return out

    
class Background_Model(nn.Module):
    """ NeRF-like bg-model """
    def __init__(self, 
                 img_width = 960, img_height = 640,
                 aabb = None, cam_norm = None,
                 model_type = "mlp", # mlp, rec_shell, sphere_shell
                 model_cfg: dict = {"head_mlp_layer_width": 256,},
                 model_args: dict = {},
                ) -> None:
        super().__init__()
        self.aabb = aabb
        self.img_width, self.img_height = img_width, img_height
        self.model_type = model_type
        self.model_cfg = model_cfg
        if model_type == 'mlp':
            self.sky_model = SkyModel(model_cfg)
        elif model_type == 'rec_shell' or model_type == 'sphere_shell':
            # TODO: has not been implemented
            self.sky_model = ShellModel(model_cfg)
        elif model_type == 'gs':
            self.sky_model = BackgroundGaussianModel(**model_cfg)
        else:
            raise NotImplementedError("model_type {} not implemented".format(model_type))
        if model_type == 'gs':
            # 先构造高斯点
            fg_aabb_center, fg_aabb_size = (aabb[0] + aabb[1]) / 2, aabb[1] - aabb[0] # cam-frustum aabb
            # use bg_scale to scale the aabb
            bg_gs_aabb = np.stack([fg_aabb_center - fg_aabb_size * model_args.bg_aabb_scale / 2, 
                        fg_aabb_center + fg_aabb_size * model_args.bg_aabb_scale / 2], axis=0)
            bg_aabb_center, bg_aabb_size = (bg_gs_aabb[0] + bg_gs_aabb[1]) / 2, bg_gs_aabb[1] - bg_gs_aabb[0]
            # add bg_gs_aabb SURFACE points
            bg_points = sample_on_aabb_surface(bg_aabb_center, bg_aabb_size, model_args.bg_gs_num, above_half=True)
            print("bg_gs_points min:",np.min(bg_points,axis=0))
            print("bg_gs_points max:",np.max(bg_points,axis=0))
            # DO NOT add bg_gs_points to points
            #points = np.concatenate([points, bg_points], axis=0)
            #shs = np.concatenate([shs, np.random.random((len(bg_points), 3)) / 255.0], axis=0)
            bg_shs = np.random.random((len(bg_points), 3)) / 255.0
            # visualize
            #from utils.general_utils import visualize_points
            #visualize_points(points, fg_aabb_center, fg_aabb_size)
            #bg_ply_path = os.path.join(data_root, "ds-bg-points3d.ply")
            #storePly(bg_ply_path, bg_points, SH2RGB(bg_shs) * 255)
            bg_pcd = BasicPointCloud(points=bg_points, colors=SH2RGB(bg_shs), normals=np.zeros((len(bg_points), 3)))

            self.sky_model.create_from_pcd(bg_pcd, 
                                spatial_lr_scale = cam_norm['radius'] * model_args.bg_aabb_scale,
                                #spatial_lr_scale = 1.0 # TODO  设置 xyz 在 0-1之内变化 ， 后续映射到 inf 空间里
                                ) # cam 距离 轨迹中心的 最大距离
            # prepare bg-hyper-params: 暂且与 fg-gs 保持一致
            #bg_model_args = model_args['']
            # gs-training-setup
            self.sky_model.training_setup(model_args) 
        else:
            # pixel coords
            x = torch.arange(0, img_width, dtype=torch.float32, device="cuda")
            y = torch.arange(0, img_height, dtype=torch.float32, device="cuda")
            x, y = x.long(), y.long()
            self.pixel_coord = torch.stack(torch.meshgrid(x, y, indexing='xy')).float().cuda()
    
    def get_ray_directions(self, c2w, intrinsic):
        x, y= self.pixel_coord[0].flatten(), self.pixel_coord[1].flatten()
        viewdirs = get_directions(x, y, c2w, intrinsic)
        viewdirs = viewdirs.reshape(self.img_height, self.img_width, 3)
        return viewdirs
        
    def forward(self, c2w, intrinsic, img_idx=None) -> Tensor:
        directions = self.get_ray_directions(c2w, intrinsic)
        sky_feat = self.sky_model(directions)
        rgb_sky = torch.sigmoid(sky_feat)
        return rgb_sky.permute(2, 0, 1)
