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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.debug_test = False
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = True
        # test/ train split
        self.stride = 0
        # visual
        self.render_process=True
        # waymo
        self.start_time = 0 # now hard-coded
        self.end_time = 49
        self.num_objs = 256 
        self.num_pts = 1500000 
        # mask loading options
        self.load_sky_mask = False
        self.load_panoptic_mask = False
        self.load_sam_mask = False
        self.load_dynamic_mask = True
        self.load_feat_map = True
        # waymo
        self.n_views = 100 
        self.random_init = False
        self.train_split = False
        self.num_classes = 200
        self.load_intrinsic = False
        self.load_c2w = False
        # occ grid
        self.save_occ_grid = True
        self.occ_voxel_size = 0.4
        self.recompute_occ_grid = False 

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = True
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.vis_step = 2000
        self.batch_size=1
        
        self.iterations = 50_000 # 30_000
        self.coarse_iterations = 5000

        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000

        self.deformation_lr_init = 0.000016
        self.deformation_lr_final = 0.0000016
        self.deformation_lr_delay_mult = 0.01
        self.grid_lr_init = 0.00016
        self.grid_lr_final = 0.000016

        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_depth = 0.5
        self.densification_interval = 100   # 100
        self.opacity_reset_interval = 3000
        self.pruning_interval = 100
        self.pruning_from_iter = 500
        self.densify_until_iter = 25_000
        # self.densify_grad_threshold = 0.0002
        self.densify_grad_threshold_coarse = 0.0002
        self.densify_grad_threshold_fine_init = 0.0002
        self.densify_grad_threshold_after = 0.0002

        # self.min_opacity_threshold = 0.005
        self.opacity_threshold_coarse = 0.005
        self.opacity_threshold_fine_init = 0.005
        self.opacity_threshold_fine_after = 0.005

        self.random_background = False
        # for waymo
        self.max_points = 500_000
        self.prune_from_iter = 500
        self.prune_interval = 100
        
        self.scale_ratio = 1.0 #   global-scale = local-norm-scale * voxel_size * scale_ratio
        # feat
        self.include_feature = True
        self.language_feature_lr = 0.0025 # TODO: update
        self.feat_dim = 8 #12  #  recomplie-cuda   SET DISTUTILS_USE_SDK=1
        self.feat_conv_lr = 0.0001 

        self.lambda_feat = 0.001
        self.dx_reg = False
        self.lambda_dx = 0.001
        self.lambda_dshs = 0.001
        # TODO: don't use, clean
        self.use_bg_gs = True
        self.use_bg_model = False
        self.bg_aabb_scale = 20.0 #2
        self.bg_gs_num = 5000
        self.bg_percent_dense = 0.01 #0.01
        self.bg_model_type = 'gs' #'mlp'
        self.mlp_width = 256
        self.bg_grid_res = 10 #　aabb/grid_res = grid_size
        self.bg_model_lr = 0.0025
        self.custom_xyz_scheduler = False
                
        # deprecated
        self.densify_from_iter = 500   # 调整至与position_lr_after_iter 一致  # 500
        self.position_lr_after_iter = 500
        self.scale_ratio_threshold = 5.0 
        self.hard_alpha_composite = True
        self.alpha_mask_threshold = 0.8

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

class ModelHiddenParams(ParamGroup):
    def __init__(self, parser):
        self.net_width = 64
        self.timebase_pe = 4
        self.defor_depth = 1
        self.posebase_pe = 10
        self.scale_rotation_pe = 2
        self.opacity_pe = 2
        self.timenet_width = 64
        self.timenet_output = 32
        self.bounds = 1.6
        self.plane_tv_weight = 0.0001
        self.time_smoothness_weight = 0.01
        self.l1_time_planes = 0.0001
        self.kplanes_config = {
                             'grid_dimensions': 2,
                             'input_coordinate_dim': 4,
                             'output_coordinate_dim': 32,
                             'resolution': [64, 64, 64, 25]
                            }
        self.multires = [1, 2, 4, 8]
        self.no_dx=False
        self.no_grid=False
        self.no_ds=True 
        self.no_dr=True
        self.no_do=True
        self.no_dshs=False
        self.feat_head=True
        self.empty_voxel=False
        self.grid_pe=0
        self.static_mlp=False
        self.apply_rotation=False

        
        super().__init__(parser, "ModelHiddenParams")

