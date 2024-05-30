"""
    用 voxel 坐标 表示 gaussian 的 mean 和 scale
    - 用于 第二阶段训练
"""
from scene.gaussian_model import GaussianModel, Grid
from utils.general_utils import strip_symmetric, build_scaling_rotation
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation

def build_from_global_gaussian(global_gaussian: GaussianModel, grid: Grid,
                scale_ratio=1.0 ):
    """
        global_gaussian: [N, 6]
        grid: Grid
    """
    # init new-gaussian
    sh_degree = global_gaussian.max_sh_degree
    gs = GridGaussianModel(sh_degree, grid=grid)
    #gaussians : create from old-gs <-- create_from_pcd

    # get mean and scale
    mean = global_gaussian.get_xyz.clone().detach()
    scale = global_gaussian.get_scaling.clone().detach()

    # filter xyz by aabb
    valid_gs_mask = grid.is_in_aabb(mean)
    mean = mean[valid_gs_mask]
    scale = scale[valid_gs_mask]
    # 打印 gs range 
    value = mean.min(dim=0)[0].cpu().detach().numpy().tolist()
    print('gs min = ', ', '.join("{:.2f}".format(v) for v in value))
    value = mean.max(dim=0)[0].cpu().detach().numpy().tolist()
    print('gs max = ', ', '.join("{:.2f}".format(v) for v in value))
    # 打印 aabb range
    value = grid.aabb[0].cpu().detach().numpy().tolist()
    print('aabb min = ', ', '.join("{:.2f}".format(v) for v in value))
    value = grid.aabb[1].cpu().detach().numpy().tolist()
    print('aabb max = ', ', '.join("{:.2f}".format(v) for v in value))
    # other properties: 不进行修改 直接返回 activate 之前的值即可
    rot = global_gaussian._rotation[valid_gs_mask].clone().detach()
    features_dc = global_gaussian._features_dc[valid_gs_mask].clone().detach()
    features_rest = global_gaussian._features_rest[valid_gs_mask].clone().detach()
    opacity = global_gaussian._opacity[valid_gs_mask].clone().detach()

    # global-gs to local-grid-gs
    grid_xyz, voxel_xyz = grid.get_grid_and_voxel_coords(mean) 
    voxel_scale = grid.norm_scale_by_voxel_size(scale, scale_ratio=scale_ratio)

    ## save grid_xyz for : voxel -> global
    gs._grid_xyz = grid_xyz
    ## build gaussian
    gs._xyz = voxel_xyz
    gs._scaling = voxel_scale
    # other properties
    gs._rotation = rot
    gs._features_dc = features_dc
    gs._features_rest = features_rest
    gs._opacity = opacity
    gs.max_radii2D = torch.zeros((gs.get_xyz.shape[0]), device="cuda")
    return gs
class GridGaussianModel(GaussianModel):
    """ 修改 基于 voxel-size 的坐标表示 
        _xyz 和 _scale 都是 [0,1] 之间的 归一化坐标
    """
    def __init__(self, sh_degree, grid: Grid=None) -> None:
        super().__init__(sh_degree)
        self.grid = grid
        # 需要维护 voxel_xyz(voxel内部归一化坐标) 和 grid_xyz(grid的网格索引)
        self._grid_xyz = None

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_xyz(self):
        # norm to global
        return self.grid.normVoxel_to_global(self._xyz, self._grid_xyz)


    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        new_grid_xyz = self._grid_xyz[selected_pts_mask].repeat(N,1)
        self.densification_postfix(new_xyz, new_grid_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_grid_xyz = self._grid_xyz[selected_pts_mask]
        self.densification_postfix(new_xyz, new_grid_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
    
    def densification_postfix(self, new_xyz, new_grid_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._grid_xyz = torch.cat([self._grid_xyz, new_grid_xyz], dim=0)

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        ## 用于约束 xyz 和 scale 都在 [0,1] 之间
        self.scaling_activation = torch.sigmoid #torch.exp
        self.scaling_inverse_activation = inverse_sigmoid #torch.log
        self.xyz_activation = torch.sigmoid
        self.xyz_inverse_activation = inverse_sigmoid

        # 替换为 恒等函数 : 优化时 会导致 scale 为负数 然后 高斯采样函数出错
        #self.scaling_activation = lambda x: x
        #self.scaling_inverse_activation = lambda x: x

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def prune_points(self, mask):
        valid_points_mask = ~mask   # 保留 剩余的节点
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self._grid_xyz = self._grid_xyz[valid_points_mask]
