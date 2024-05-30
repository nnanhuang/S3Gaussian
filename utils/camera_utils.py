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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch, DepthMaptoTorch, ObjectPILtoTorch
from utils.graphics_utils import fov2focal
import torch 

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    # for waymo
    sky_mask = None
    if cam_info.sky_mask is not None:
        sky_mask = PILtoTorch(cam_info.sky_mask, resolution)
    depth_map = None
    if cam_info.depth_map is not None:
        depth_map = DepthMaptoTorch(cam_info.depth_map)
    semantic_mask = None
    if cam_info.semantic_mask is not None:
        semantic_mask = ObjectPILtoTorch(cam_info.semantic_mask, resolution)
    instance_mask = None
    if cam_info.instance_mask is not None:
        instance_mask = ObjectPILtoTorch(cam_info.instance_mask, resolution)
    sam_mask = None
    if cam_info.sam_mask is not None:
        sam_mask = ObjectPILtoTorch(cam_info.sam_mask, resolution)
    feat_map = None
    if cam_info.feat_map is not None:
        feat_map = cam_info.feat_map
    dynamic_mask = None
    if cam_info.dynamic_mask is not None:
        dynamic_mask = ObjectPILtoTorch(cam_info.dynamic_mask, resolution)
    intrinsic = None
    if cam_info.intrinsic is not None:
        intrinsic = torch.from_numpy(cam_info.intrinsic).to(dtype=torch.float32)
    c2w = None
    if cam_info.c2w is not None:
        c2w = torch.from_numpy(cam_info.c2w).to(dtype=torch.float32)
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  # for waymo
                  sky_mask = sky_mask, depth_map = depth_map,
                  semantic_mask = semantic_mask, instance_mask = instance_mask,
                  sam_mask = sam_mask,
                  dynamic_mask = dynamic_mask,
                  feat_map = feat_map,
                  objects=torch.from_numpy(np.array(cam_info.objects)) if cam_info.objects is not None else None,
                  intrinsic = intrinsic, 
                  c2w = c2w,
                  time = cam_info.time
                  )

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
