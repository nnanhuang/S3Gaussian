import logging
import os
from typing import Callable, Dict, List, Optional
from plyfile import PlyData, PlyElement

import imageio
import numpy as np
import torch
from utils.loss_utils import l1_loss
from skimage.metrics import structural_similarity as ssim
from lpipsPyTorch import lpips
from torch import Tensor
from tqdm import tqdm, trange
from gaussian_renderer import render

from utils.image_utils import psnr
# from gs_renderer import Renderer,GaussianModel,TrainCam
# from trainer import Trainer
# from utils.misc import get_robust_pca
from utils.visualization_tools import (
    resize_five_views,
    scene_flow_to_rgb,
    to8b,
    visualize_depth,
)

depth_visualizer = lambda frame, opacity: visualize_depth(
    frame,
    opacity,
    lo=4.0,
    hi=120,
    depth_curve_fn=lambda x: -np.log(x + 1e-6),
)
flow_visualizer = (
    lambda frame: scene_flow_to_rgb(
        frame,
        background="bright",
        flow_max_radius=1.0,
    )
    .cpu()
    .numpy()
)
get_numpy: Callable[[Tensor], np.ndarray] = lambda x: x.squeeze().cpu().numpy()
non_zero_mean: Callable[[Tensor], float] = (
    lambda x: sum(x) / len(x) if len(x) > 0 else -1
)

def get_robust_pca(features: torch.Tensor, m: float = 2, remove_first_component=False):
    # features: (N, C)
    # m: a hyperparam controlling how many std dev outside for outliers
    assert len(features.shape) == 2, "features should be (N, C)"
    reduction_mat = torch.pca_lowrank(features, q=3, niter=20)[2]
    colors = features @ reduction_mat
    if remove_first_component:
        colors_min = colors.min(dim=0).values
        colors_max = colors.max(dim=0).values
        tmp_colors = (colors - colors_min) / (colors_max - colors_min)
        fg_mask = tmp_colors[..., 0] < 0.2
        reduction_mat = torch.pca_lowrank(features[fg_mask], q=3, niter=20)[2]
        colors = features @ reduction_mat
    else:
        fg_mask = torch.ones_like(colors[:, 0]).bool()
    d = torch.abs(colors[fg_mask] - torch.median(colors[fg_mask], dim=0).values)
    mdev = torch.median(d, dim=0).values
    s = d / mdev
    rins = colors[fg_mask][s[:, 0] < m, 0]
    gins = colors[fg_mask][s[:, 1] < m, 1]
    bins = colors[fg_mask][s[:, 2] < m, 2]

    rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
    rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])
    return reduction_mat, rgb_min.to(reduction_mat), rgb_max.to(reduction_mat)

def render_pixels(
    viewpoint_stack,
    gaussians,
    bg,
    pipe,
    compute_metrics: bool = True,
    return_decomposition: bool = True,
    debug:bool = False
):
    """
    Render pixel-related outputs from a model.

    Args:
        ....skip obvious args
        compute_metrics (bool, optional): Whether to compute metrics. Defaults to False.
        vis_indices (Optional[List[int]], optional): Indices to visualize. Defaults to None.
        return_decomposition (bool, optional): Whether to visualize the static-dynamic decomposition. Defaults to True.
    """
    # set up render function
    render_results = render_func(
        viewpoint_stack,
        gaussians,
        pipe,
        bg,
        compute_metrics=compute_metrics,
        return_decomposition=return_decomposition,
        debug = debug
    )
    if compute_metrics:
        num_samples = len(viewpoint_stack)
        print(f"Eval over {num_samples} images:")
        print(f"\tPSNR: {render_results['psnr']:.4f}")
        print(f"\tSSIM: {render_results['ssim']:.4f}")
        print(f"\tLPIPS: {render_results['lpips']:.4f}")
        # print(f"\tFeature PSNR: {render_results['feat_psnr']:.4f}")
        print(f"\tMasked PSNR: {render_results['masked_psnr']:.4f}")
        print(f"\tMasked SSIM: {render_results['masked_ssim']:.4f}")
        # print(f"\tMasked Feature PSNR: {render_results['masked_feat_psnr']:.4f}")

    return render_results


def render_func(
    viewpoint_stack,
    gaussians,
    pipe,
    bg,
    compute_metrics: bool = False,
    return_decomposition:bool = False,
    num_cams: int = 3,
    debug: bool = False,
    save_seperate_pcd = False
):
    """
    Renders a dataset utilizing a specified render function.
    For efficiency and space-saving reasons, this function doesn't store the original features; instead, it keeps
    the colors reduced via PCA.
    TODO: clean up this function

    Parameters:
        dataset: Dataset to render.
        render_func: Callable function used for rendering the dataset.
        compute_metrics: Optional; if True, the function will compute and return metrics. Default is False.
    """
    # rgbs
    rgbs, gt_rgbs = [], []
    static_rgbs, dynamic_rgbs = [], []
    shadow_reduced_static_rgbs, shadow_only_static_rgbs = [], []

    # depths
    depths, median_depths = [], []
    static_depths, static_opacities = [], []
    dynamic_depths, dynamic_opacities = [], []

    # sky
    opacities, sky_masks = [], []

    # features
    pred_dinos, gt_dinos = [], []
    pred_dinos_pe_free, pred_dino_pe = [], []
    static_dinos, dynamic_dinos = [], []  # should we also render this?

    # cross-rendering results
    dynamic_dino_on_static_rgbs, dynamic_rgb_on_static_dinos = [], []

    # flows
    forward_flows, backward_flows = [], []
    dx_list = []

    if compute_metrics:
        psnrs, ssim_scores, feat_psnrs = [], [], []
        masked_psnrs, masked_ssims = [], []
        masked_feat_psnrs = [],
        lpipss = []

    with torch.no_grad():
        for i in tqdm(range(len(viewpoint_stack)), desc=f"rendering full data", dynamic_ncols=True):
            viewpoint_cam = viewpoint_stack[i]

            render_pkg = render(viewpoint_cam, gaussians, pipe, bg,return_decomposition = return_decomposition,return_dx=True)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            
            # ------------- rgb ------------- #
            rgb = image
            gt_rgb = viewpoint_cam.original_image.cuda()

            rgbs.append(get_numpy(rgb.permute(1, 2, 0)))
            gt_rgbs.append(get_numpy(gt_rgb.permute(1, 2, 0)))
            
            if "render_s" in render_pkg:
                static_rgbs.append(get_numpy(render_pkg["render_s"].permute(1, 2, 0)))
                visibility_filter_s = render_pkg['visibility_filter_s']
            if "render_d" in render_pkg:
                # green screen blending for better visualization
                # green_background = torch.tensor([0.0, 177, 64]) / 255.0
                # green_background = green_background.to(render_pkg["render_d"].device)
                dy_rgb = render_pkg["render_d"].permute(1, 2, 0)
                # dy_rgb = dy_rgb * 0.8 + green_background * 0.2
                dynamic_rgbs.append(get_numpy(dy_rgb))
                visibility_filter_d = render_pkg['visibility_filter_d']
            
            # ------------- depth ------------- #
            depth = render_pkg["depth"]
            depth_np = depth.permute(1, 2, 0).cpu().numpy()
            depth_np /= depth_np.max()
            # depth_np = np.repeat(depth_np, 3, axis=2)

            depths.append(depth_np)

            # ------------- flow ------------- #
            if "dx" in render_pkg and render_pkg['dx'] is not None:
                dx = render_pkg['dx']
                dx = torch.tensor(dx)
                dx_max = torch.max(dx)
                dx_min = torch.min(dx)
                dx_list.append(dx)     
            if compute_metrics:
                psnrs.append(psnr(rgb, gt_rgb).mean().double().item())
                # ssim_scores.append(ssim(rgb, gt_rgb).mean().item())
                ssim_scores.append(
                    ssim(
                        get_numpy(rgb),
                        get_numpy(gt_rgb),
                        data_range=1.0,
                        channel_axis=0,
                    )
                )
                lpipss.append(torch.tensor(lpips(rgb, gt_rgb,net_type='alex')).mean().item())
                
                dynamic_mask = get_numpy(viewpoint_cam.dynamic_mask).astype(bool)
                if dynamic_mask.sum() > 0:
                    rgb_d = rgb.permute(1, 2, 0)[dynamic_mask]
                    rgb_d = rgb_d.permute(1, 0)
                    gt_rgb_d = gt_rgb.permute(1, 2, 0)[dynamic_mask]
                    gt_rgb_d = gt_rgb_d.permute(1, 0)

                    masked_psnrs.append(
                    psnr(rgb_d, gt_rgb_d).mean().double().item()
                    )
                    masked_ssims.append(
                        ssim(
                            get_numpy(rgb.permute(1, 2, 0)),
                            get_numpy(gt_rgb.permute(1, 2, 0)),
                            data_range=1.0,
                            channel_axis=-1,
                            full=True,
                        )[1][dynamic_mask].mean()
                    )

        if save_seperate_pcd and len(dx_list)>1:    
            # 首先根据visibility_filter 选出所有的可见范围内的点
            # 然后得到dynamic 和 static 的mask，把点保存
                
            dynamic_pcd_path = os.path.join('test','dynamic.ply')
            static_pcd_path = os.path.join('test','static.ply')

            gaussians.save_ply_split(dynamic_pcd_path, static_pcd_path, dx_list, visibility_filter)

        if len(dx_list)>1:
            # deformation flow -> forward & backward flow
            bf_color_first = []
            ff_color_last = []
            for t in range(len(dx_list)): # 防止越界
                if t < len(dx_list)-num_cams:
                    # forward_flow_t 归一化一下
                    forward_flow_t = dx_list[t + num_cams] - dx_list[t]
                    ff_color = flow_visualizer(forward_flow_t)
                    ff_color = torch.from_numpy(ff_color).to("cuda") 
                    if debug:
                        ff_color = (ff_color - torch.min(ff_color)) / (torch.max(ff_color) - torch.min(ff_color) + 1e-6)  # 归一化，避免除零错误

                    if t == len(dx_list)-num_cams-1 or t == len(dx_list)-num_cams-2 or t == len(dx_list)-num_cams-3: 
                        ff_color_last.append(ff_color)              
                    render_pkg2 = render(viewpoint_stack[t], gaussians, pipe, bg, override_color=ff_color)
                    ff_map = render_pkg2['render'].permute(1, 2, 0).cpu().numpy()

                    # print(ff_map.max())
                    # print(ff_map.min())
                    forward_flows.append(ff_map)
                
                # 同时处理 backward flow，除第一个时刻外
                if t > num_cams-1:
                    backward_flow_t = dx_list[t] - dx_list[t - num_cams]
                    bf_color = flow_visualizer(backward_flow_t)
                    bf_color = torch.from_numpy(bf_color).to("cuda") 
                    if debug:
                        bf_color = (bf_color - torch.min(bf_color)) / (torch.max(bf_color) - torch.min(bf_color) + 1e-6)  # 归一化，避免除零错误
                    if t == num_cams or t == num_cams+1 or t == num_cams+2: 
                        bf_color_first.append(bf_color)                 
                    # viewpoint_cam 要变化
                    render_pkg2 = render(viewpoint_stack[t], gaussians, pipe, bg, override_color=bf_color)
                    bf_map = render_pkg2['render'].permute(1, 2, 0).cpu().numpy()

                    backward_flows.append(bf_map)

            for i, bf_color in enumerate(bf_color_first):
                render_pkg3 = render(viewpoint_stack[i], gaussians, pipe, bg, override_color=bf_color)            
                bf_map_first = render_pkg3['render'].permute(1, 2, 0).cpu().numpy()       
                # 对于 backward flow 的第一个时刻，复制第一个计算的 forward flow
                backward_flows.insert(i, bf_map_first)

            for i, ff_color in enumerate(ff_color_last):
                # 对于 forward flow 的最后一个时刻，复制最后一个计算的 backward flow
                render_pkg4 = render(viewpoint_stack[len(viewpoint_stack)-num_cams+i], gaussians, pipe, bg, override_color=ff_color)            
                ff_map_last = render_pkg4['render'].permute(1, 2, 0).cpu().numpy()       
                forward_flows.append(ff_map_last)           

    # messy aggregation...
    results_dict = {}
    results_dict["psnr"] = non_zero_mean(psnrs) if compute_metrics else -1
    results_dict["ssim"] = non_zero_mean(ssim_scores) if compute_metrics else -1
    results_dict["lpips"] = non_zero_mean(lpipss) if compute_metrics else -1
    results_dict["masked_psnr"] = non_zero_mean(masked_psnrs) if compute_metrics else -1
    results_dict["masked_ssim"] = non_zero_mean(masked_ssims) if compute_metrics else -1

    results_dict["rgbs"] = rgbs
    results_dict["depths"] = depths
    results_dict["opacities"] = opacities

    if len(gt_rgbs) > 0:
        results_dict["gt_rgbs"] = gt_rgbs
    if len(static_rgbs)>0:
        results_dict["static_rgbs"] = static_rgbs
    if len(dynamic_rgbs)>0:
        results_dict["dynamic_rgbs"] = dynamic_rgbs
    if len(sky_masks) > 0:
        results_dict["gt_sky_masks"] = sky_masks
    if len(pred_dinos) > 0:
        results_dict["dino_feats"] = pred_dinos
    if len(gt_dinos) > 0:
        results_dict["gt_dino_feats"] = gt_dinos
    if len(pred_dinos_pe_free) > 0:
        results_dict["dino_feats_pe_free"] = pred_dinos_pe_free
    if len(pred_dino_pe) > 0:
        results_dict["dino_pe"] = pred_dino_pe
    if len(static_dinos) > 0:
        results_dict["static_dino_feats"] = static_dinos
    if len(dynamic_dinos) > 0:
        results_dict["dynamic_dino_feats"] = dynamic_dinos
    if len(dynamic_dino_on_static_rgbs) > 0:
        results_dict["dynamic_dino_on_static_rgbs"] = dynamic_dino_on_static_rgbs
    if len(dynamic_rgb_on_static_dinos) > 0:
        results_dict["dynamic_rgb_on_static_dinos"] = dynamic_rgb_on_static_dinos
    if len(shadow_reduced_static_rgbs) > 0:
        results_dict["shadow_reduced_static_rgbs"] = shadow_reduced_static_rgbs
    if len(shadow_only_static_rgbs) > 0:
        results_dict["shadow_only_static_rgbs"] = shadow_only_static_rgbs
    if len(forward_flows) > 0:
        results_dict["forward_flows"] = forward_flows
    if len(backward_flows) > 0:
        results_dict["backward_flows"] = backward_flows
    if len(median_depths) > 0:
        results_dict["median_depths"] = median_depths
    if len(dx_list) > 0:
        results_dict['dx_list'] = dx_list
    return results_dict


def save_videos(
    render_results: Dict[str, List[Tensor]],
    save_pth: str,
    num_timestamps: int,
    keys: List[str] = ["gt_rgbs", "rgbs", "depths"],
    num_cams: int = 3,
    save_seperate_video: bool = False,
    save_images: bool = False,
    fps: int = 10,
    verbose: bool = True,
):
    if save_seperate_video:
        return_frame = save_seperate_videos(
            render_results,
            save_pth,
            num_timestamps=num_timestamps,
            keys=keys,
            num_cams=num_cams,
            save_images=save_images,
            fps=fps,
            verbose=verbose,
        )
    else:
        return_frame = save_concatenated_videos(
            render_results,
            save_pth,
            num_timestamps=num_timestamps,
            keys=keys,
            num_cams=num_cams,
            save_images=save_images,
            fps=fps,
            verbose=verbose,
        )
    return return_frame


def save_concatenated_videos(
    render_results: Dict[str, List[Tensor]],
    save_pth: str,
    num_timestamps: int,
    keys: List[str] = ["gt_rgbs", "rgbs", "depths"],
    num_cams: int = 3,
    save_images: bool = False,
    fps: int = 10,
    verbose: bool = True,
):
    if num_timestamps == 1:  # it's an image
        writer = imageio.get_writer(save_pth, mode="I")
        return_frame_id = 0
    else:
        return_frame_id = num_timestamps // 2
        writer = imageio.get_writer(save_pth, mode="I", fps=fps)
    for i in trange(num_timestamps, desc="saving video", dynamic_ncols=True):
        merged_list = []
        for key in keys:
            if key == "sky_masks":
                frames = render_results["opacities"][i * num_cams : (i + 1) * num_cams]
            else:
                if key not in render_results or len(render_results[key]) == 0:
                    continue
                frames = render_results[key][i * num_cams : (i + 1) * num_cams]
            if key == "gt_sky_masks":
                frames = [np.stack([frame, frame, frame], axis=-1) for frame in frames]
            elif key == "sky_masks":
                frames = [
                    1 - np.stack([frame, frame, frame], axis=-1) for frame in frames
                ]
            # elif "depth" in key:

            #     frames = [
            #         depth_visualizer(frame, opacity)
            #         for frame, opacity in zip(frames, opacities)
            #     ]
            frames = resize_five_views(frames)
            frames = np.concatenate(frames, axis=1)
            merged_list.append(frames)
        merged_frame = to8b(np.concatenate(merged_list, axis=0))
        if i == return_frame_id:
            return_frame = merged_frame
        writer.append_data(merged_frame)
    writer.close()
    if verbose:
        print(f"saved video to {save_pth}")
    del render_results
    return {"concatenated_frame": return_frame}


def save_seperate_videos(
    render_results: Dict[str, List[Tensor]],
    save_pth: str,
    num_timestamps: int,
    keys: List[str] = ["gt_rgbs", "rgbs", "depths"],
    num_cams: int = 3,
    fps: int = 10,
    verbose: bool = False,
    save_images: bool = False,
):
    return_frame_id = num_timestamps // 2
    return_frame_dict = {}
    for key in keys:
        tmp_save_pth = save_pth.replace(".mp4", f"_{key}.mp4")
        tmp_save_pth = tmp_save_pth.replace(".png", f"_{key}.png")
        if num_timestamps == 1:  # it's an image
            writer = imageio.get_writer(tmp_save_pth, mode="I")
        else:
            writer = imageio.get_writer(tmp_save_pth, mode="I", fps=fps)
        if key not in render_results or len(render_results[key]) == 0:
            continue
        for i in range(num_timestamps):
            if key == "sky_masks":
                frames = render_results["opacities"][i * num_cams : (i + 1) * num_cams]
            else:
                # 这里取3个，得到3个视角
                frames = render_results[key][i * num_cams : (i + 1) * num_cams]
            if key == "gt_sky_masks":
                frames = [np.stack([frame, frame, frame], axis=-1) for frame in frames]
            elif key == "sky_masks":
                frames = [
                    1 - np.stack([frame, frame, frame], axis=-1) for frame in frames
                ]
            # elif "depth" in key:
            #     opacities = render_results[key.replace("depths", "opacities")][
            #         i * num_cams : (i + 1) * num_cams
            #     ]
            #     frames = [
            #         depth_visualizer(frame, opacity)
            #         for frame, opacity in zip(frames, opacities)
            #     ]
            frames = resize_five_views(frames)
            if save_images:
                if i == 0:
                    os.makedirs(tmp_save_pth.replace(".mp4", ""), exist_ok=True)
                for j, frame in enumerate(frames):
                    imageio.imwrite(
                        tmp_save_pth.replace(".mp4", f"_{i*3 + j:03d}.png"),
                        to8b(frame),
                    )
            frames = to8b(np.concatenate(frames, axis=1))
            writer.append_data(frames) # [H,W,3]
            if i == return_frame_id:
                return_frame_dict[key] = frames
        # close the writer
        writer.close()
        del writer
        if verbose:
            print(f"saved video to {tmp_save_pth}")
    del render_results
    return return_frame_dict
