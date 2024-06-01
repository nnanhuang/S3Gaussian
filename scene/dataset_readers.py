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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from tqdm import trange
from utils.general_utils import PILtoTorch
from tqdm import tqdm
import cv2
from utils.general_utils import sample_on_aabb_surface, get_OccGrid
from utils.segmentation_utils import get_panoptic_id
import torch
from utils.feature_extractor import extract_and_save_features
from utils.image_utils import get_robust_pca

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    # for waymo
    sky_mask: np.array = None
    depth_map: np.array = None
    time: float = None
    semantic_mask: np.array = None
    instance_mask: np.array = None
    sam_mask: np.array = None
    dynamic_mask: np.array = None
    feat_map: np.array = None
    # grouping
    objects: np.array = None
    # 
    intrinsic: np.array = None
    c2w: np.array = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    # for waymo
    full_cameras:list
    bg_point_cloud: BasicPointCloud = None
    bg_ply_path: str = None
    cam_frustum_aabb: np.array = None
    num_panoptic_objects: int = 0
    panoptic_id_to_idx: dict = None
    panoptic_object_ids: list = None
    occ_grid: np.array = None

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, objects_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path) if os.path.exists(image_path) else None

        #object_path = os.path.join(objects_folder, image_name + '.png')
        #objects = Image.open(object_path) if os.path.exists(object_path) else None
        if 'test' not in image_name:
            # For Training, we use SAM-auto-mask
            if os.path.exists(os.path.join(objects_folder, image_name + '.png')):
                object_path = os.path.join(objects_folder, image_name + '.png')
                objects = Image.open(object_path)
            elif os.path.exists(os.path.join(objects_folder, image_name + '.jpg')):
                object_path = os.path.join(objects_folder, image_name + '.jpg')
                objects = Image.open(object_path)
            else:
                objects = None
        else:
            # For Testing, we use labeled-mask
            object_path = os.path.join(os.path.dirname(objects_folder),'object_mask', image_name + '.png')
            objects = Image.open(object_path) if os.path.exists(object_path) else None



        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, objects=objects)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    
    green_color = [0, 255, 0]  # [N,3] array
    rgb = np.array([green_color for _ in range(xyz.shape[0])])

    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, object_path, llffhold=8, n_views=100, random_init=False, train_split=False):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    object_dir = 'object_mask' if object_path == None else object_path
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), objects_folder=os.path.join(path, object_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        if train_split:
            train_dir = os.path.join(path, "images_train")
            train_names = sorted(os.listdir(train_dir))
            train_names = [train_name.split('.')[0] for train_name in train_names]
            train_cam_infos = []
            test_cam_infos = []
            for cam_info in cam_infos:
                if cam_info.image_name in train_names:
                    train_cam_infos.append(cam_info)
                else:
                    test_cam_infos.append(cam_info)

        else:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]

            if n_views == 100:
                pass 
            elif n_views == 50:
                idx_sub = np.linspace(0, len(train_cam_infos)-1, round(len(train_cam_infos)*0.5)) # 50% views
                idx_sub = [round(i) for i in idx_sub]
                train_cam_infos = [train_cam_infos[i_sub] for i_sub in idx_sub]
            elif isinstance(n_views,int):
                idx_sub = np.linspace(0, len(train_cam_infos)-1, n_views) # 3views
                idx_sub = [round(i) for i in idx_sub]
                train_cam_infos = [train_cam_infos[i_sub] for i_sub in idx_sub]
                print(train_cam_infos)
            else:
                raise NotImplementedError
        print("Training images:     ", len(train_cam_infos))
        print("Testing images:     ", len(test_cam_infos))

    else:
        if train_split:
            train_dir = os.path.join(path, "images_train")
            train_names = sorted(os.listdir(train_dir))
            train_names = [train_name.split('.')[0] for train_name in train_names]
            train_cam_infos = []
            for cam_info in cam_infos:
                if cam_info.image_name in train_names:
                    train_cam_infos.append(cam_info)
            test_cam_infos = []
        else:
            train_cam_infos = cam_infos
            test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if random_init:
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        
        ply_path = os.path.join(path, "sparse/0/points3D_randinit.ply")
        storePly(ply_path, xyz, SH2RGB(shs) * 255)

    else:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def constructCameras_waymo(frames_list, white_background, mapper = {},
                           load_intrinsic=False, load_c2w=False, start_time = 50, original_start_time = 0):
    cam_infos = []
    for idx, frame in enumerate(frames_list):
        # current frame time
        time = mapper[frame["time"] + start_time - original_start_time]
        # ------------------
        # load c2w
        # ------------------
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to OpenCV/COLMAP (Y down, Z forward)
        #c2w[:3, 1:3] *= -1
        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        # ------------------
        # load image
        # ------------------
        cam_name = image_path = frame['file_path']
        image_name = Path(cam_name).stem
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))
        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0]) # d-nerf 透明背景
        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
        load_size = frame["load_size"]
        #image = PILtoTorch(image, load_size) #(800,800))
        # resize to load_size
        image = image.resize(load_size, Image.BILINEAR)
        # save pil image
        # image.save(os.path.join("debug", image_name + ".png"))
        # ------------------
        # load depth-map
        # ------------------
        depth_map = frame.get('depth_map', None)
        
        # # visualize depth map with rgb
        # mask = depth_map > 0
        # # normalize depth map to [0, 255]
        # depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map)) * 255
        # np_depth_map = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=1.0), cv2.COLORMAP_JET)
        # # mask empty depth map: depth_map(h,w) , np_depth_map(h,w,3)
        # np_depth_map[~mask] = [255, 255, 255]
        # depth_map_colored = Image.fromarray(np_depth_map)
        # #image_depth = Image.blend(image, depth_map_colored, 0.5)
        # image_np = np.array(image)
        # image_np[mask] = np_depth_map[mask]
        # image_depth = Image.fromarray(image_np)
        # image_depth.save(os.path.join("exp/debug-0", image_name + "_depth.png"))
        # depth_map_colored.save(os.path.join("exp/debug-0", image_name + "_depth_colored.png"))
        
        # ------------------
        # load sky-mask
        # ------------------
        sky_mask_path, sky_mask = frame["sky_mask_path"], None
        if sky_mask_path is not None:
            sky_mask = Image.open(sky_mask_path)
            sky_mask = sky_mask.resize(load_size, Image.BILINEAR)
        # ------------------
        # load intrinsic
        # ------------------
        # intrinsic to fov: intrinsic 已经被 scale
        intrinsic = frame["intrinsic"]
        fx, fy, cx, cy = intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]
        # get fov
        fovx = focal2fov(fx, image.size[0])
        fovy = focal2fov(fy, image.size[1])
        FovY = fovy
        FovX = fovx

        # ------------------
        # load semantic mask
        # ------------------
        semantic_mask_path, semantic_mask = frame["semantic_mask_path"], None
        if semantic_mask_path is not None:
            semantic_mask = np.load(semantic_mask_path)
            semantic_mask = Image.fromarray(semantic_mask.squeeze(-1))
            semantic_mask = semantic_mask.resize(load_size, Image.NEAREST)
            # to numpy
            #semantic_mask = np.array(semantic_mask)#  .unsqueeze(-1)

        # ------------------
        # load instance mask
        # ------------------
        instance_mask_path, instance_mask = frame["instance_mask_path"], None
        if instance_mask_path is not None:
            instance_mask = np.load(instance_mask_path)
            instance_mask = Image.fromarray(instance_mask.squeeze(-1))
            instance_mask = instance_mask.resize(load_size, Image.NEAREST)
            # to numpy
            #instance_mask = np.array(instance_mask) #.unsqueeze(-1)

        # ------------------
        # load sam mask
        # ------------------
        sam_mask_path, sam_mask = frame["sam_mask_path"], None
        if sam_mask_path is not None:
            sam_mask = Image.open(sam_mask_path)
            sam_mask = sam_mask.resize(load_size, Image.NEAREST)
            # to numpy
            #sam_mask = np.array(sam_mask) #.unsqueeze(-1)

        # ------------------
        # load dynamic mask
        # ------------------
        dynamic_mask_path, dynamic_mask = frame["dynamic_mask_path"], None
        if dynamic_mask_path is not None:
            dynamic_mask = Image.open(dynamic_mask_path)
            dynamic_mask = dynamic_mask.resize(load_size, Image.NEAREST)
            # to numpy
            #dynamic_mask = np.array(dynamic_mask) #.unsqueeze(-1)

        # ------------------
        # load feat map
        # ------------------
        feat_map_path, feat_map = frame["feat_map_path"], None
        if feat_map_path is not None:
            # mmap_mode="r" is to avoid memory overflow when loading features
            # but it only slightly helps... do we have a better way to load features?
            features = np.load(feat_map_path, mmap_mode="r").squeeze()
            features = torch.from_numpy(features).unsqueeze(0).float()

            # shape: (num_imgs, num_patches_h, num_patches_w, C)
            # featmap_downscale_factor is used to convert the image coordinates to ViT feature coordinates.
            # resizing ViT features to (H, W) using bilinear interpolation is infeasible.
            # imagine a feature array of shape (num_timesteps x num_cams, 640, 960, 768). it's too large to fit in GPU memory.
            featmap_downscale_factor = (
                features.shape[1] / 640,
                features.shape[2] / 960,
            )
            # print(
            #     f"Loaded {features.shape} dinov2_vitb14 features."
            # )
            # print(f"Feature scale: {featmap_downscale_factor}")
            # print(f"Computing features PCA...")
            # compute feature visualization matrix
            C = features.shape[-1]
            # no need to compute PCA on the entire set of features, we randomly sample 100k features
            temp_feats = features.reshape(-1, C)
            max_elements_to_compute_pca = min(100000, temp_feats.shape[0])
            selected_features = temp_feats[
                np.random.choice(
                    temp_feats.shape[0], max_elements_to_compute_pca, replace=False
                )
            ]
            target_feature_dim = 3
            device='cuda'
            if target_feature_dim is not None:
                # print(
                #     f"Reducing features to {target_feature_dim} dimensions."
                # )
                # compute PCA to reduce the feature dimension to target_feature_dim
                U, S, reduce_to_target_dim_mat = torch.pca_lowrank(
                    selected_features, q=target_feature_dim, niter=20
                )
                # compute the fraction of variance explained by target_feature_dim
                variances = S**2
                fraction_var_explained = variances / variances.sum()
                # print(f"[PCA] fraction_var_explained: \n{fraction_var_explained}")
                # print(
                #     f"[PCA] fraction_var_explained sum: {fraction_var_explained.sum()}",
                # )
                reduce_to_target_dim_mat = reduce_to_target_dim_mat

                # reduce the features to target_feature_dim
                selected_features = selected_features @ reduce_to_target_dim_mat
                features =  features @ reduce_to_target_dim_mat
                C = features.shape[-1]

                # normalize the reduced features to [0, 1] along each dimension
                feat_min = features.reshape(-1, C).min(dim=0)[0]
                feat_max = features.reshape(-1, C).max(dim=0)[0]
                features = (features - feat_min) / (feat_max - feat_min)
                selected_features = (selected_features - feat_min) / (feat_max - feat_min)
                feat_min = feat_min.to(device)
                feat_max = feat_max.to(device)
                reduce_to_target_dim_mat = reduce_to_target_dim_mat.to(device)
            # we compute the first 3 principal components of the ViT features as the color
            reduction_mat, feat_color_min, feat_color_max = get_robust_pca(
                selected_features
            )
            # final features are of shape (num_imgs, num_patches_h, num_patches_w, target_feature_dim)
            features = features

            # save visualization parameters
            feat_dimension_reduction_mat = reduction_mat
            feat_color_min = feat_color_min
            feat_color_max = feat_color_max
            del temp_feats, selected_features

            # print(
            #     f"Feature PCA computed, shape: {feat_dimension_reduction_mat.shape}"
            # )
            # tensor: [91, 137, 64]
            x, y = torch.meshgrid(
                torch.arange(image.size[0]),
                torch.arange(image.size[1]),
                indexing="xy",
            )
            x, y = x.flatten(), y.flatten()
            x, y = x.to(device), y.to(device)

            # we compute the nearest DINO feature for each pixel
            # map (x, y) in the (W, H) space to (x * dino_scale[0], y * dino_scale[1]) in the (W//patch_size, H//patch_size) space
            dino_y = (y * featmap_downscale_factor[0]).long()
            dino_x = (x * featmap_downscale_factor[1]).long()
            # dino_feats are in CPU memory (because they are huge), so we need to move them to GPU
            features = features.squeeze()
            dino_feat = features[dino_y.cpu(), dino_x.cpu()]

            features = dino_feat.reshape(image.size[1], image.size[0], -1)
            feat_map = features.float()

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],
                        # for waymo
                        sky_mask=sky_mask, depth_map=depth_map, time=time,
                        semantic_mask=semantic_mask, instance_mask=instance_mask, 
                        sam_mask=sam_mask, 
                        dynamic_mask=dynamic_mask, 
                        feat_map=feat_map, # [640,960,3]
                        intrinsic=intrinsic if load_intrinsic else None,
                        c2w=c2w if load_c2w else None,
                         ))
            
    return cam_infos

def readWaymoInfo(path, white_background, eval, extension=".png", use_bg_gs=False, 
                  load_sky_mask = False, load_panoptic_mask = True, load_sam_mask = False,load_dynamic_mask = False,
                  load_feat_map = False,
                  load_intrinsic = False, load_c2w = False,
                  start_time = 0, end_time = -1, num_pts = 5000, 
                  save_occ_grid = False, occ_voxel_size = 0.4, recompute_occ_grid=True,
                  stride = 10 , original_start_time = 0
                  ):
    ORIGINAL_SIZE = [[1280, 1920], [1280, 1920], [1280, 1920], [884, 1920], [884, 1920]]
    OPENCV2DATASET = np.array(
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    )
    load_size = [640, 960]
    # modified from emer-nerf
    data_root = path
    image_folder = os.path.join(data_root, "images")
    num_seqs = len(os.listdir(image_folder))/5
    start_time = start_time
    if end_time == -1:
        end_time = int(num_seqs)
    else:
        end_time += 1
    camera_list = [1,0,2]
    truncated_min_range, truncated_max_range = -2, 80
    cam_frustum_range = [0.01, 80]
    # set img_list
    load_sky_mask = load_sky_mask
    load_panoptic_mask = load_panoptic_mask
    load_sam_mask = load_sam_mask
    load_dynamic_mask = load_dynamic_mask
    load_feat_map = load_feat_map
    img_filepaths = []
    dynamic_mask_filepaths, sky_mask_filepaths = [], []
    semantic_mask_filepaths, instance_mask_filepaths = [], []
    sam_mask_filepaths = []
    feat_map_filepaths = []
    dynamic_mask_filepaths = []
    lidar_filepaths = []
    for t in range(start_time, end_time):
        for cam_idx in camera_list:
            img_filepaths.append(os.path.join(data_root, "images", f"{t:03d}_{cam_idx}.jpg"))
            #dynamic_mask_filepaths.append(os.path.join(data_root, "dynamic_masks", f"{t:03d}_{cam_idx}.png"))
            sky_mask_filepaths.append(os.path.join(data_root, "sky_masks", f"{t:03d}_{cam_idx}.png"))
            #semantic_mask_filepaths.append(os.path.join(data_root, "semantic_masks", f"{t:03d}_{cam_idx}.png"))
            #instance_mask_filepaths.append(os.path.join(data_root, "instance_masks", f"{t:03d}_{cam_idx}.png"))
            if os.path.exists(os.path.join(data_root, "semantic_segs", f"{t:03d}_{cam_idx}.npy")):
                semantic_mask_filepaths.append(os.path.join(data_root, "semantic_segs", f"{t:03d}_{cam_idx}.npy"))
            else:
                semantic_mask_filepaths.append(None)
            if os.path.exists(os.path.join(data_root, "instance_segs", f"{t:03d}_{cam_idx}.npy")):
                instance_mask_filepaths.append(os.path.join(data_root, "instance_segs", f"{t:03d}_{cam_idx}.npy"))
            else:
                instance_mask_filepaths.append(None)
            if os.path.exists(os.path.join(data_root, "sam_masks", f"{t:03d}_{cam_idx}.jpg")):
                sam_mask_filepaths.append(os.path.join(data_root, "sam_masks", f"{t:03d}_{cam_idx}.jpg"))
            if os.path.exists(os.path.join(data_root, "dynamic_masks", f"{t:03d}_{cam_idx}.png")):
                dynamic_mask_filepaths.append(os.path.join(data_root, "dynamic_masks", f"{t:03d}_{cam_idx}.png"))
            if load_feat_map:
                feat_map_filepaths.append(os.path.join(data_root, "dinov2_vitb14", f"{t:03d}_{cam_idx}.npy"))
        lidar_filepaths.append(os.path.join(data_root, "lidar", f"{t:03d}.bin"))

    if load_feat_map:
        return_dict = extract_and_save_features(
                input_img_path_list=img_filepaths,
                saved_feat_path_list=feat_map_filepaths,
                img_shape=[644, 966],
                stride=7,
                model_type='dinov2_vitb14',
            )
    img_filepaths = np.array(img_filepaths)
    dynamic_mask_filepaths = np.array(dynamic_mask_filepaths)
    sky_mask_filepaths = np.array(sky_mask_filepaths)
    lidar_filepaths = np.array(lidar_filepaths)
    semantic_mask_filepaths = np.array(semantic_mask_filepaths)
    instance_mask_filepaths = np.array(instance_mask_filepaths)
    sam_mask_filepaths = np.array(sam_mask_filepaths)
    feat_map_filepaths = np.array(feat_map_filepaths)
    dynamic_mask_filepaths = np.array(dynamic_mask_filepaths)
    # ------------------
    # construct timestamps
    # ------------------
    # original_start_time = 0
    idx_list = range(original_start_time, end_time)
    # map time to [0,1]
    timestamp_mapper = {}
    time_line = [i for i in idx_list]
    time_length = end_time - original_start_time - 1
    for index, time in enumerate(time_line):
        timestamp_mapper[time] = (time-original_start_time)/time_length
    max_time = max(timestamp_mapper.values())
    # ------------------
    # load poses: intrinsic, c2w, l2w
    # ------------------
    _intrinsics = []
    cam_to_egos = []
    for i in range(len(camera_list)):
        # load intrinsics
        intrinsic = np.loadtxt(os.path.join(data_root, "intrinsics", f"{i}.txt"))
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        # scale intrinsics w.r.t. load size
        fx, fy = (
            fx * load_size[1] / ORIGINAL_SIZE[i][1],
            fy * load_size[0] / ORIGINAL_SIZE[i][0],
        )
        cx, cy = (
            cx * load_size[1] / ORIGINAL_SIZE[i][1],
            cy * load_size[0] / ORIGINAL_SIZE[i][0],
        )
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        _intrinsics.append(intrinsic)
        # load extrinsics
        cam_to_ego = np.loadtxt(os.path.join(data_root, "extrinsics", f"{i}.txt"))
        # opencv coordinate system: x right, y down, z front
        # waymo coordinate system: x front, y left, z up
        cam_to_egos.append(cam_to_ego @ OPENCV2DATASET) # opencv_cam -> waymo_cam -> waymo_ego
    # compute per-image poses and intrinsics
    cam_to_worlds, ego_to_worlds = [], []
    intrinsics, cam_ids = [], []
    lidar_to_worlds = []
    # ===! for waymo, we simplify timestamps as the time indices
    timestamps, timesteps = [], []
    # we tranform the camera poses w.r.t. the first timestep to make the translation vector of
    # the first ego pose as the origin of the world coordinate system.
    ego_to_world_start = np.loadtxt(os.path.join(data_root, "ego_pose", f"{start_time:03d}.txt"))
    for t in range(start_time, end_time):
        ego_to_world_current = np.loadtxt(os.path.join(data_root, "ego_pose", f"{t:03d}.txt"))
        # ego to world transformation: cur_ego -> world -> start_ego(world)
        ego_to_world = np.linalg.inv(ego_to_world_start) @ ego_to_world_current
        ego_to_worlds.append(ego_to_world)
        for cam_id in camera_list:
            cam_ids.append(cam_id)
            # transformation:
            # opencv_cam -> waymo_cam -> waymo_cur_ego -> world -> start_ego(world)
            cam2world = ego_to_world @ cam_to_egos[cam_id]
            cam_to_worlds.append(cam2world)
            intrinsics.append(_intrinsics[cam_id])
            # ===! we use time indices as the timestamp for waymo dataset for simplicity
            # ===! we can use the actual timestamps if needed
            # to be improved
            timestamps.append(t - start_time)
            timesteps.append(t - start_time)
        # lidar to world : lidar = ego in waymo
        lidar_to_worlds.append(ego_to_world)
    # convert to numpy arrays
    intrinsics = np.stack(intrinsics, axis=0)
    cam_to_worlds = np.stack(cam_to_worlds, axis=0)
    ego_to_worlds = np.stack(ego_to_worlds, axis=0)
    lidar_to_worlds = np.stack(lidar_to_worlds, axis=0)
    cam_ids = np.array(cam_ids)
    timestamps = np.array(timestamps)
    timesteps = np.array(timesteps)
    # ------------------
    # get aabb: c2w --> frunstums --> aabb
    # ------------------
    # compute frustums
    frustums = []
    pix_corners = np.array( # load_size : [h, w]
        [[0,0],[0,load_size[0]],[load_size[1],load_size[0]],[load_size[1],0]]
    )
    for c2w, intri in zip(cam_to_worlds, intrinsics):
        frustum = []
        for cam_extent in cam_frustum_range:
            # pix_corners to cam_corners
            cam_corners = np.linalg.inv(intri) @ np.concatenate(
                [pix_corners, np.ones((4, 1))], axis=-1
            ).T * cam_extent
            # cam_corners to world_corners
            world_corners = c2w[:3, :3] @ cam_corners + c2w[:3, 3:4]
            # compute frustum
            frustum.append(world_corners)
        frustum = np.stack(frustum, axis=0)
        frustums.append(frustum)
    frustums = np.stack(frustums, axis=0)
    # compute aabb
    aabbs = []
    for frustum in frustums:
        flatten_frustum = frustum.transpose(0,2,1).reshape(-1,3)
        aabb_min = np.min(flatten_frustum, axis=0)
        aabb_max = np.max(flatten_frustum, axis=0)
        aabb = np.stack([aabb_min, aabb_max], axis=0)
        aabbs.append(aabb)
    aabbs = np.stack(aabbs, axis=0).reshape(-1,3)
    aabb = np.stack([np.min(aabbs, axis=0), np.max(aabbs, axis=0)], axis=0)
    print('cam frustum aabb min: ', aabb[0])
    print('cam frustum aabb max: ', aabb[1])
    # ------------------
    # get split: train and test splits from timestamps
    # ------------------
    # mask
    if stride != 0 :
        train_mask = (timestamps % int(stride) != 0) | (timestamps == 0)
    else:
        train_mask = np.ones(len(timestamps), dtype=bool)
    test_mask = ~train_mask
    # mask to index                                                                    
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    full_idx = np.arange(len(timestamps))
    train_timestamps = timestamps[train_mask]
    test_timestamps = timestamps[test_mask]
    # ------------------
    # load points and depth map
    # ------------------
    pts_path = os.path.join(data_root, "lidar")
    load_lidar, load_depthmap = True, True
    depth_maps = None
    # bg-gs settings
    #use_bg_gs = False
    bg_scale = 2.0 # used to scale fg-aabb
    if not os.path.exists(pts_path) or not load_lidar:
        # random sample
        # Since this data set has no colmap data, we start with random points
        #num_pts = 2000
        print(f"Generating random point cloud ({num_pts})...")
        aabb_center = (aabb[0] + aabb[1]) / 2
        aabb_size = aabb[1] - aabb[0]
        # We create random points inside the bounds of the synthetic Blender scenes
        random_xyz = np.random.random((num_pts, 3)) 
        print('normed xyz min: ', np.min(random_xyz, axis=0))
        print('normed xyz max: ', np.max(random_xyz, axis=0))
        xyz = random_xyz * aabb_size + aabb[0]
        print('xyz min: ', np.min(xyz, axis=0))
        print('xyz max: ', np.max(xyz, axis=0))
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    # storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        # load lidar points
        origins, directions, points, ranges, laser_ids = [], [], [], [], []
        depth_maps = []
        accumulated_num_original_rays = 0
        accumulated_num_rays = 0
        for t in trange(0, len(lidar_filepaths), desc="loading lidar", dynamic_ncols=True):
            lidar_info = np.memmap(
                lidar_filepaths[t],
                dtype=np.float32,
                mode="r",
            ).reshape(-1, 10) 
            #).reshape(-1, 14)
            original_length = len(lidar_info)
            accumulated_num_original_rays += original_length
            lidar_origins = lidar_info[:, :3]
            lidar_points = lidar_info[:, 3:6]
            lidar_ids = lidar_info[:, -1]
            # select lidar points based on a truncated ego-forward-directional range
            # make sure most of lidar points are within the range of the camera
            valid_mask = lidar_points[:, 0] < truncated_max_range
            valid_mask = valid_mask & (lidar_points[:, 0] > truncated_min_range)
            lidar_origins = lidar_origins[valid_mask]
            lidar_points = lidar_points[valid_mask]
            lidar_ids = lidar_ids[valid_mask]
            # transform lidar points to world coordinate system
            lidar_origins = (
                lidar_to_worlds[t][:3, :3] @ lidar_origins.T
                + lidar_to_worlds[t][:3, 3:4]
            ).T
            lidar_points = (
                lidar_to_worlds[t][:3, :3] @ lidar_points.T
                + lidar_to_worlds[t][:3, 3:4]
            ).T
            if load_depthmap:
                # transform world-lidar to pixel-depth-map
                for cam_idx in range(len(camera_list)):
                    # world-lidar-pts --> camera-pts : w2c
                    c2w = cam_to_worlds[int(len(camera_list))*t + cam_idx]
                    w2c = np.linalg.inv(c2w)
                    cam_points = (
                        w2c[:3, :3] @ lidar_points.T
                        + w2c[:3, 3:4]
                    ).T
                    # camera-pts --> pixel-pts : intrinsic @ (x,y,z) = (u,v,1)*z
                    pixel_points = (
                        intrinsics[int(len(camera_list))*t + cam_idx] @ cam_points.T
                    ).T
                    # select points in front of the camera
                    pixel_points = pixel_points[pixel_points[:, 2]>0]
                    # normalize pixel points : (u,v,1)
                    image_points = pixel_points[:, :2] / pixel_points[:, 2:]
                    # filter out points outside the image
                    valid_mask = (
                        (image_points[:, 0] >= 0)
                        & (image_points[:, 0] < load_size[1])
                        & (image_points[:, 1] >= 0)
                        & (image_points[:, 1] < load_size[0])
                    )
                    pixel_points = pixel_points[valid_mask]     # pts_cam : (x,y,z)
                    image_points = image_points[valid_mask]     # pts_img : (u,v)
                    # compute depth map
                    depth_map = np.zeros(load_size)
                    depth_map[image_points[:, 1].astype(np.int32), image_points[:, 0].astype(np.int32)] = pixel_points[:, 2]
                    depth_maps.append(depth_map)
            # compute lidar directions
            lidar_directions = lidar_points - lidar_origins
            lidar_ranges = np.linalg.norm(lidar_directions, axis=-1, keepdims=True)
            lidar_directions = lidar_directions / lidar_ranges
            # time indices as timestamp
            #lidar_timestamps = np.ones_like(lidar_ranges).squeeze(-1) * t
            accumulated_num_rays += len(lidar_ranges)

            origins.append(lidar_origins)
            directions.append(lidar_directions)
            points.append(lidar_points)
            ranges.append(lidar_ranges)
            laser_ids.append(lidar_ids)

        #origins = np.concatenate(origins, axis=0)
        #directions = np.concatenate(directions, axis=0)
        points = np.concatenate(points, axis=0)
        #ranges = np.concatenate(ranges, axis=0)
        #laser_ids = np.concatenate(laser_ids, axis=0)
        shs = np.random.random((len(points), 3)) / 255.0
        # filter points by cam_aabb 
        cam_aabb_mask = np.all((points >= aabb[0]) & (points <= aabb[1]), axis=-1)
        points = points[cam_aabb_mask]
        shs = shs[cam_aabb_mask]
        # construct occupancy grid to aid densification
        if save_occ_grid:
            #occ_grid_shape = (int(np.ceil((aabb[1, 0] - aabb[0, 0]) / occ_voxel_size)),
            #                    int(np.ceil((aabb[1, 1] - aabb[0, 1]) / occ_voxel_size)),
            #                    int(np.ceil((aabb[1, 2] - aabb[0, 2]) / occ_voxel_size)))
            if not os.path.exists(os.path.join(data_root, "occ_grid.npy")) or recompute_occ_grid:
                occ_grid = get_OccGrid(points, aabb, occ_voxel_size)
                np.save(os.path.join(data_root, "occ_grid.npy"), occ_grid)
            else:
                occ_grid = np.load(os.path.join(data_root, "occ_grid.npy"))
            print(f'Lidar points num : {len(points)}')
            print("occ_grid shape : ", occ_grid.shape)
            print(f'occ voxel num :{occ_grid.sum()} from {occ_grid.size} of ratio {occ_grid.sum()/occ_grid.size}')
        
        # downsample points
        points,shs = GridSample3D(points,shs)

        if len(points)>num_pts:
            downsampled_indices = np.random.choice(
                len(points), num_pts, replace=False
            )
            points = points[downsampled_indices]
            shs = shs[downsampled_indices]
        
        # check
        #voxel_coords = np.floor((points - aabb[0]) / occ_voxel_size).astype(int)
        #occ = occ_grid[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]]
        #origins = origins[downsampled_indices] 
        
        ## 计算 points xyz 的范围
        xyz_min = np.min(points,axis=0)
        xyz_max = np.max(points,axis=0)
        print("init lidar xyz min:",xyz_min)
        print("init lidar xyz max:",xyz_max)        # lidar-points aabb (range)
        ## 设置 背景高斯点
        if use_bg_gs:
            fg_aabb_center, fg_aabb_size = (aabb[0] + aabb[1]) / 2, aabb[1] - aabb[0] # cam-frustum aabb
            # use bg_scale to scale the aabb
            bg_gs_aabb = np.stack([fg_aabb_center - fg_aabb_size * bg_scale / 2, 
                        fg_aabb_center + fg_aabb_size * bg_scale / 2], axis=0)
            bg_aabb_center, bg_aabb_size = (bg_gs_aabb[0] + bg_gs_aabb[1]) / 2, bg_gs_aabb[1] - bg_gs_aabb[0]
            # add bg_gs_aabb SURFACE points
            bg_points = sample_on_aabb_surface(bg_aabb_center, bg_aabb_size, 1000)
            print("bg_gs_points min:",np.min(bg_points,axis=0))
            print("bg_gs_points max:",np.max(bg_points,axis=0))
            # DO NOT add bg_gs_points to points
            #points = np.concatenate([points, bg_points], axis=0)
            #shs = np.concatenate([shs, np.random.random((len(bg_points), 3)) / 255.0], axis=0)
            bg_shs = np.random.random((len(bg_points), 3)) / 255.0
            # visualize
            #from utils.general_utils import visualize_points
            #visualize_points(points, fg_aabb_center, fg_aabb_size)
        # save ply
        ply_path = os.path.join(data_root, "ds-points3d.ply")
        storePly(ply_path, points, SH2RGB(shs) * 255)
        pcd = BasicPointCloud(points=points, colors=SH2RGB(shs), normals=np.zeros((len(points), 3)))  
        if use_bg_gs:
            bg_ply_path = os.path.join(data_root, "ds-bg-points3d.ply")
            storePly(bg_ply_path, bg_points, SH2RGB(bg_shs) * 255)
            bg_pcd = BasicPointCloud(points=bg_points, colors=SH2RGB(bg_shs), normals=np.zeros((len(bg_points), 3)))
        else:
            bg_pcd, bg_ply_path = None, None
        # load depth maps
        if load_depthmap:
            assert depth_maps is not None, "should not use random-init-gs, ans set load_depthmap=True"
            depth_maps = np.stack(depth_maps, axis=0)
    # ------------------
    # prepare cam-pose dict
    # ------------------
    train_frames_list = [] # time, transform_matrix(c2w), img_path
    test_frames_list = []
    full_frames_list = []
    for idx, t in enumerate(train_timestamps):
        frame_dict = dict(  time = t,   # 保存 相对帧索引
                            transform_matrix = cam_to_worlds[train_idx[idx]],
                            file_path = img_filepaths[train_idx[idx]],
                            intrinsic = intrinsics[train_idx[idx]],
                            load_size = [load_size[1], load_size[0]],   # [w, h] for PIL.resize
                            sky_mask_path = sky_mask_filepaths[train_idx[idx]] if load_sky_mask else None,
                            depth_map = depth_maps[train_idx[idx]] if load_depthmap else None,
                            semantic_mask_path = semantic_mask_filepaths[train_idx[idx]] if load_panoptic_mask else None,
                            instance_mask_path = instance_mask_filepaths[train_idx[idx]] if load_panoptic_mask else None,
                            sam_mask_path = sam_mask_filepaths[train_idx[idx]] if load_sam_mask else None,
                            feat_map_path = feat_map_filepaths[train_idx[idx]] if load_feat_map else None,
                            dynamic_mask_path = dynamic_mask_filepaths[train_idx[idx]] if load_dynamic_mask else None,
        )
        train_frames_list.append(frame_dict)
    for idx, t in enumerate(test_timestamps):
        frame_dict = dict(  time = t,   # 保存 相对帧索引 
                            transform_matrix = cam_to_worlds[test_idx[idx]],
                            file_path = img_filepaths[test_idx[idx]],
                            intrinsic = intrinsics[test_idx[idx]],
                            load_size = [load_size[1], load_size[0]],   # [w, h] for PIL.resize
                            sky_mask_path = sky_mask_filepaths[test_idx[idx]] if load_sky_mask else None,
                            depth_map = depth_maps[test_idx[idx]] if load_depthmap else None,
                            semantic_mask_path = semantic_mask_filepaths[test_idx[idx]] if load_panoptic_mask else None,
                            instance_mask_path = instance_mask_filepaths[test_idx[idx]] if load_panoptic_mask else None,
                            sam_mask_path = sam_mask_filepaths[test_idx[idx]] if load_sam_mask else None,
                            feat_map_path = feat_map_filepaths[test_idx[idx]] if load_feat_map else None,
                            dynamic_mask_path = dynamic_mask_filepaths[test_idx[idx]] if load_dynamic_mask else None,
        )
        test_frames_list.append(frame_dict)
    if len(test_timestamps)==0:
        full_frames_list = train_frames_list
    else:
        for idx, t in enumerate(timestamps):
            frame_dict = dict(  time = t,   # 保存 相对帧索引 
                                transform_matrix = cam_to_worlds[full_idx[idx]],
                                file_path = img_filepaths[full_idx[idx]],
                                intrinsic = intrinsics[full_idx[idx]],
                                load_size = [load_size[1], load_size[0]],   # [w, h] for PIL.resize
                                sky_mask_path = sky_mask_filepaths[full_idx[idx]] if load_sky_mask else None,
                                depth_map = depth_maps[full_idx[idx]] if load_depthmap else None,
                                semantic_mask_path = semantic_mask_filepaths[full_idx[idx]] if load_panoptic_mask else None,
                                instance_mask_path = instance_mask_filepaths[full_idx[idx]] if load_panoptic_mask else None,
                                sam_mask_path = sam_mask_filepaths[full_idx[idx]] if load_sam_mask else None,
                                feat_map_path = feat_map_filepaths[full_idx[idx]] if load_feat_map else None,
                                dynamic_mask_path = dynamic_mask_filepaths[full_idx[idx]] if load_dynamic_mask else None,
            )
            full_frames_list.append(frame_dict)
    
    # ------------------
    # load cam infos: image, c2w, intrinsic, load_size
    # ------------------
    print("Reading Training Transforms")
    train_cam_infos = constructCameras_waymo(train_frames_list, white_background, timestamp_mapper, 
                                             load_intrinsic=load_intrinsic, load_c2w=load_c2w,start_time=start_time,original_start_time=original_start_time)
    print("Reading Test Transforms")
    test_cam_infos = constructCameras_waymo(test_frames_list, white_background, timestamp_mapper,
                                            load_intrinsic=load_intrinsic, load_c2w=load_c2w,start_time=start_time,original_start_time=original_start_time)
    print("Reading Full Transforms")
    full_cam_infos = constructCameras_waymo(full_frames_list, white_background, timestamp_mapper,
                                            load_intrinsic=load_intrinsic, load_c2w=load_c2w,start_time=start_time,original_start_time=original_start_time)
    # full_cam_infos = train_cam_infos
    
    #print("Generating Video Transforms")
    #video_cam_infos = generateCamerasFromTransforms_waymo(test_frames_list, max_time)
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
    nerf_normalization = getNerfppNorm(train_cam_infos)


    # ------------------
    # find panoptic-objec numbers
    # ------------------
    num_panoptic_objects = 0
    panoptic_object_ids = None
    panoptic_id_to_idx = {}
    if load_panoptic_mask:
        panoptic_object_ids_list = []
        for cam in train_cam_infos+test_cam_infos:
            if cam.semantic_mask is not None and cam.instance_mask is not None:
                panoptic_object_ids = get_panoptic_id(cam.semantic_mask, cam.instance_mask).unique()
                panoptic_object_ids_list.append(panoptic_object_ids)
        # get unique panoptic_objects_ids
        panoptic_object_ids = torch.cat(panoptic_object_ids_list).unique().sort()[0].tolist()
        num_panoptic_objects = len(panoptic_object_ids)
        # map panoptic_id to idx
        for idx, panoptic_id in enumerate(panoptic_object_ids):
            panoptic_id_to_idx[panoptic_id] = idx

    scene_info = SceneInfo(point_cloud=pcd,
                           bg_point_cloud=bg_pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           full_cameras=full_cam_infos,
                           #video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           # background settings
                           ply_path=pts_path,
                           bg_ply_path=bg_ply_path,
                           cam_frustum_aabb=aabb,
                           # panoptic segs
                           num_panoptic_objects=num_panoptic_objects,
                           panoptic_object_ids=panoptic_object_ids,
                           panoptic_id_to_idx=panoptic_id_to_idx,
                           # occ grid
                           occ_grid=occ_grid if save_occ_grid else None,
                           )

    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Waymo" : readWaymoInfo,
}

def GridSample3D(in_pc,in_shs, voxel_size=0.013):
    in_pc_ = in_pc[:,:3].copy()
    quantized_pc = np.around(in_pc_ / voxel_size)
    quantized_pc -= np.min(quantized_pc, axis=0)
    pc_boundary = np.max(quantized_pc, axis=0) - np.min(quantized_pc, axis=0)
    
    voxel_index = quantized_pc[:,0] * pc_boundary[1] * pc_boundary[2] + quantized_pc[:,1] * pc_boundary[2] + quantized_pc[:,2]
    
    split_point, index = get_split_point(voxel_index)
    
    in_points = in_pc[index,:]
    out_points = in_points[split_point[:-1],:]
    
    in_colors = in_shs[index]
    out_colors = in_colors[split_point[:-1]]
    
    # 创建一个新的BasicPointCloud实例作为输出
    # out_pc =out_points
    # #remap index in_pc to out_pc
    # remap = np.zeros(in_pc.points.shape[0])
        
    # for ind in range(len(split_point)-1):
    #     cur_start = split_point[ind]
    #     cur_end = split_point[ind+1]
    #     remap[cur_start:cur_end] = ind
    
    # remap_back = remap.copy()
    # remap_back[index] = remap
    
    # remap_back = remap_back.astype(np.int64)
    return out_points,out_colors

def get_split_point(labels):
    index = np.argsort(labels)
    label = labels[index]
    label_shift = label.copy()
    
    label_shift[1:] = label[:-1]
    remain = label - label_shift
    step_index = np.where(remain > 0)[0].tolist()
    step_index.insert(0,0)
    step_index.append(labels.shape[0])
    return step_index,index
