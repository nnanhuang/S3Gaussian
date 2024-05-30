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
import gc
import random
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, compute_depth
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
# import lpips
from utils.scene_utils import render_training_image
from time import time
import copy

import numpy as np
import time
import json
from utils.video_utils import render_pixels, save_videos
from utils.visualization_tools import compute_optical_flow_and_save
from scene.gaussian_model import merge_models

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

# try:
#     from torch.utils.tensorboard import SummaryWriter
#     TENSORBOARD_FOUND = True
# except ImportError:
TENSORBOARD_FOUND = False
   
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

render_keys = [
    "gt_rgbs",
    "rgbs",
    "depths",
    "dynamic_rgbs",
    "static_rgbs",
    # "forward_flows",
    # "backward_flows",
]

@torch.no_grad()
def do_evaluation(
    viewpoint_stack_full,
    viewpoint_stack_test,
    viewpoint_stack_train,
    gaussians,
    bg,
    pipe,
    eval_dir,
    render_full,
    step: int = 0,
    args = None,
):
    if len(viewpoint_stack_test) != 0:
        print("Evaluating Test Set Pixels...")
        render_results = render_pixels(
            viewpoint_stack_test,
            gaussians,
            bg,
            pipe,
            compute_metrics=True,
            return_decomposition=True,
            debug=args.debug_test
        )
        eval_dict = {}
        for k, v in render_results.items():
            if k in [
                "psnr",
                "ssim",
                "lpips",
                # "feat_psnr",
                "masked_psnr",
                "masked_ssim",
                # "masked_feat_psnr",
            ]:
                eval_dict[f"pixel_metrics/test/{k}"] = v
                
        os.makedirs(f"{eval_dir}/metrics", exist_ok=True)
        os.makedirs(f"{eval_dir}/test_videos", exist_ok=True)
        
        test_metrics_file = f"{eval_dir}/metrics/{step}_images_test_{current_time}.json"
        with open(test_metrics_file, "w") as f:
            json.dump(eval_dict, f)
        print(f"Image evaluation metrics saved to {test_metrics_file}")

        video_output_pth = f"{eval_dir}/test_videos/{step}.mp4"

        vis_frame_dict = save_videos(
            render_results,
            video_output_pth,
            num_timestamps=int(len(viewpoint_stack_test)//3),
            keys=render_keys,
            num_cams=3,
            save_seperate_video=True,
            fps=24,
            verbose=True,
        )

        del render_results, vis_frame_dict
        torch.cuda.empty_cache()
    if len(viewpoint_stack_train) != 0 and len(viewpoint_stack_test) != 0:
        print("Evaluating train Set Pixels...")
        render_results = render_pixels(
            viewpoint_stack_train,
            gaussians,
            bg,
            pipe,
            compute_metrics=True,
            return_decomposition=False,
            debug=args.debug_test
        )
        eval_dict = {}
        for k, v in render_results.items():
            if k in [
                "psnr",
                "ssim",
                "lpips",
                # "feat_psnr",
                "masked_psnr",
                "masked_ssim",
                # "masked_feat_psnr",
            ]:
                eval_dict[f"pixel_metrics/train/{k}"] = v
                
        os.makedirs(f"{eval_dir}/metrics", exist_ok=True)
        os.makedirs(f"{eval_dir}/train_videos", exist_ok=True)
        
        train_metrics_file = f"{eval_dir}/metrics/{step}_images_train.json"
        with open(train_metrics_file, "w") as f:
            json.dump(eval_dict, f)
        print(f"Image evaluation metrics saved to {train_metrics_file}")

        video_output_pth = f"{eval_dir}/train_videos/{step}.mp4"

        vis_frame_dict = save_videos(
            render_results,
            video_output_pth,
            num_timestamps=int(len(viewpoint_stack_train)//3),
            keys=render_keys,
            num_cams=3,
            save_seperate_video=True,
            fps=24,
            verbose=True,
        )

        del render_results
        torch.cuda.empty_cache()

    if render_full:
        print("Evaluating Full Set...")
        render_results = render_pixels(
            viewpoint_stack_full,
            gaussians,
            bg,
            pipe,
            compute_metrics=True,
            return_decomposition=True,
            debug=args.debug_test
        )
        eval_dict = {}
        for k, v in render_results.items():
            if k in [
                "psnr",
                "ssim",
                "lpips",
                # "feat_psnr",
                "masked_psnr",
                "masked_ssim",
                # "masked_feat_psnr",
            ]:
                eval_dict[f"pixel_metrics/full/{k}"] = v
                
        os.makedirs(f"{eval_dir}/metrics", exist_ok=True)
        os.makedirs(f"{eval_dir}/full_videos", exist_ok=True)
        
        test_metrics_file = f"{eval_dir}/metrics/{step}_images_full_{current_time}.json"
        with open(test_metrics_file, "w") as f:
            json.dump(eval_dict, f)
        print(f"Image evaluation metrics saved to {test_metrics_file}")

        # if render_video_postfix is None:
        video_output_pth = f"{eval_dir}/full_videos/{step}.mp4"
        vis_frame_dict = save_videos(
            render_results,
            video_output_pth,
            num_timestamps=int(len(viewpoint_stack_full)//3),
            keys=render_keys,
            num_cams=3,
            save_seperate_video=True,
            fps=24,
            verbose=True,
        )
        
        del render_results, vis_frame_dict
        torch.cuda.empty_cache()

def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter,timer):
    first_iter = 0

    gaussians.training_setup(opt)
    if checkpoint:
        # breakpoint()
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            # process is in the coarse stage, but start from fine stage
            return
        if stage in checkpoint: 
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if args.eval_only:
        torch.save(gaussians._deformation.state_dict(),os.path.join(args.model_path, "deformation.pth"))

        eval_dir = os.path.join(args.model_path,"eval")
        os.makedirs(eval_dir,exist_ok=True)
        viewpoint_stack_full = scene.getFullCameras().copy()
        viewpoint_stack_test = scene.getTestCameras().copy()
        viewpoint_stack_train = scene.getTrainCameras().copy()

        # TODO：可视化光流 and 静动态点云分离
        do_evaluation(
            viewpoint_stack_full,
            viewpoint_stack_test,
            viewpoint_stack_train,
            gaussians,
            background,
            pipe,
            eval_dir,
            render_full=True,
            step=first_iter,
            args=args
        )
        # save 静动态点云分离
        # pcd_dir = os.path.join(eval_dir, "split_pcd")
        # os.makedirs(eval_dir,exist_ok=True)

        # gaussians.save_ply_split(pcd_dir)
        exit()

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter
    
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    # lpips_model = lpips.LPIPS(net="alex").cuda()
    # video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()

    if not viewpoint_stack:
        
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)
    
    batch_size = opt.batch_size
    print("data loading done")    
        
    count = 0
    psnr_dict = {}
    for iteration in range(first_iter, final_iter+1):        
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             count +=1
        #             viewpoint_index = (count ) % len(video_cams)
        #             if (count //(len(video_cams))) % 2 == 0:
        #                 viewpoint_index = viewpoint_index
        #             else:
        #                 viewpoint_index = len(video_cams) - viewpoint_index - 1
        #             # print(viewpoint_index)
        #             viewpoint = video_cams[viewpoint_index]
        #             custom_cam.time = viewpoint.time
        #             # print(custom_cam.time, viewpoint_index, count)
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, stage=stage)["render"]

        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive) :
        #             break
        #     except Exception as e:
        #         print(e)
        #         network_gui.conn = None

        iter_start.record()

        position_lr = gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # batch size
        idx = 0
        viewpoint_cams = []

        while idx < batch_size :    
            
            viewpoint_cam = viewpoint_stack.pop(randint(0,len(viewpoint_stack)-1))
            if not viewpoint_stack :
                viewpoint_stack =  temp_list.copy()
                # print("find the worst viewpoint")
                # 对 PSNR 字典按值进行排序，找出最低的 PSNR 值对应的 UID, 最后一个psnr 没办法得到，所以实际上比较 n*3 -1 个psnr
                # with torch.no_grad():
                #     if 'fine' in stage:
                #         psnr_dict = sorted(psnr_dict.items(), key=lambda x: x[1])

                #         # 将最低 PSNR 值对应的 UID 添加到列表中，直到列表的长度达到 args.end_time / 5
                #         lowest_psnr_uids = []
                #         for uid, _ in psnr_dict[:(args.end_time+1)]:
                #             lowest_psnr_uids.append(uid)
                            
                #         psnr_dict = {}

                #         # 将 lowest_psnr_uids 中 UID 对应的 Camera 对象加到 viewpoint_stack 的末尾
                #         for uid in lowest_psnr_uids:
                #             for cam in viewpoint_stack:
                #                 if cam.uid == int(uid):
                #                     viewpoint_stack.append(cam)
                #                     break                
                
            viewpoint_cams.append(viewpoint_cam)
            idx +=1
        if len(viewpoint_cams) == 0:
            continue
        # print(len(viewpoint_cams))     
        # breakpoint()   
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        gt_images = []
        depth_preds = []
        gt_depths = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage,return_dx=True,render_feat = True if ('fine' in stage and args.feat_head) else False)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            depth_pred = render_pkg["depth"]
            depth_preds.append(depth_pred.unsqueeze(0))
            images.append(image.unsqueeze(0))
            gt_image = viewpoint_cam.original_image.cuda()
            gt_depth = viewpoint_cam.depth_map.cuda()

            gt_images.append(gt_image.unsqueeze(0))
            gt_depths.append(gt_depth.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)

        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        depth_pred_tensor = torch.cat(depth_preds,0)
        gt_image_tensor = torch.cat(gt_images,0)
        gt_depth_tensor = torch.cat(gt_depths,0).float()
        # Loss
        # breakpoint()
        Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])

        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        # if 'fine' in stage:
        #     psnr_dict.update({f"{viewpoint_cam.uid}": psnr_})
        # norm        
        loss = Ll1
        # dx loss
        if 'fine' in stage and not args.no_dx and opt.lambda_dx !=0:
            dx_abs = torch.abs(render_pkg['dx'])
            dx_loss = torch.mean(dx_abs) * opt.lambda_dx
            loss += dx_loss
        if 'fine' in stage and not args.no_dshs and opt.lambda_dshs != 0:
            dshs_abs = torch.abs(render_pkg['dshs'])
            dshs_loss = torch.mean(dshs_abs) * opt.lambda_dshs
            loss += dshs_loss
        if opt.lambda_depth != 0:
            depth_loss = compute_depth("l2", depth_pred_tensor, gt_depth_tensor) * opt.lambda_depth
            loss += depth_loss
        if stage == "fine" and hyper.time_smoothness_weight != 0:
            # tv_loss = 0
            tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)
            loss += tv_loss
        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor,gt_image_tensor)
            loss += opt.lambda_dssim * (1.0-ssim_loss)
        if stage == 'fine' and args.feat_head:
            feat = render_pkg['feat'].to('cuda') # [3,640,960]
            gt_feat = viewpoint_cam.feat_map.permute(2,0,1).to('cuda')
            loss_feat = l2_loss(feat, gt_feat) * opt.lambda_feat
            loss += loss_feat
            
        # if opt.lambda_lpips !=0:
        #     lpipsloss = lpips_loss(image_tensor,gt_image_tensor,lpips_model)
        #     loss += opt.lambda_lpips * lpipsloss
        
        loss.backward()
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 100 == 0:
                dynamic_points = 0
                if 'fine' in stage and not args.no_dx:
                    dx_abs = torch.abs(render_pkg['dx']) # [N,3]
                    max_values = torch.max(dx_abs, dim=1)[0] # [N]
                    thre = torch.mean(max_values)                    
                    mask = (max_values > thre)
                    dynamic_points = torch.sum(mask).item()

                print_dict = {
                    "step": f"{iteration}",
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "psnr": f"{psnr_:.{2}f}",
                    "dynamic point": f"{dynamic_points}",
                    "point":f"{total_point}",
                    }
                progress_bar.set_postfix(print_dict)
                metrics_file = f"{scene.model_path}/logger.json"
                with open(metrics_file, "a") as f:
                    json.dump(print_dict, f)
                    f.write('\n')

                progress_bar.update(100)
            if iteration == final_iter:
                progress_bar.close()

            # Log and save
            timer.pause()
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background], stage)
            # if (iteration in saving_iterations):
            #     print("\n[ITER {}] Saving Gaussians".format(iteration))
            #     scene.save(iteration, stage)
            if dataset.render_process:
                if (iteration < 10000 and iteration % 1000 == 999) \
                    or (iteration < 30000 and iteration % 2000 == 1999) \
                        or (iteration < 60000 and iteration %  3000 == 2999) :
                    # breakpoint()
                        if len(test_cams) != 0:
                            render_training_image(scene, gaussians, [test_cams[iteration%len(test_cams)]], render, pipe, background, stage+"test", iteration,timer.get_elapsed_time())
                        render_training_image(scene, gaussians, [train_cams[iteration%len(train_cams)]], render, pipe, background, stage+"train", iteration,timer.get_elapsed_time())

                    # total_images.append(to8b(temp_image).transpose(1,2,0))
            timer.start()
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  

                if  iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<2000000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, iteration, stage)
                if  iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 : # and gaussians.get_xyz.shape[0]>200000
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                # if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000 and opt.add_point:
                #     gaussians.grow(5,5,scene.model_path,iteration,stage)
                    # torch.cuda.empty_cache()
                if iteration % opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    gaussians.reset_opacity()
                    
            
            # Optimizer step
            if iteration < final_iter+1:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                save_path= "chkpnt" +f"_{stage}_" + str(30000) + ".pth"
                for file in os.listdir(scene.model_path):
                    if file.endswith(".pth") and file != save_path:
                        os.remove(os.path.join(scene.model_path, file))

                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" +f"_{stage}_" + str(iteration) + ".pth")

            if (iteration == 30000):
                eval_dir = os.path.join(args.model_path,"eval")
                os.makedirs(eval_dir,exist_ok=True)
                viewpoint_stack_full = scene.getFullCameras().copy()
                viewpoint_stack_test = scene.getTestCameras().copy()
                viewpoint_stack_train = scene.getTrainCameras().copy()

                do_evaluation(
                    viewpoint_stack_full,
                    viewpoint_stack_test,
                    viewpoint_stack_train,
                    gaussians,
                    background,
                    pipe,
                    eval_dir,
                    render_full=True,
                    step=iteration,
                    args=args
                )

def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname):
    # first_iter = 0
    tb_writer = prepare_output_and_logger(expname)        

    gaussians = GaussianModel(dataset.sh_degree, hyper)
        
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=None)
    timer.start()
    
    # eval
    eval_dir = os.path.join(args.model_path,"eval")
    os.makedirs(eval_dir,exist_ok=True)
    viewpoint_stack_full = scene.getFullCameras().copy()
    viewpoint_stack_test = scene.getTestCameras().copy()
    viewpoint_stack_train = scene.getTrainCameras().copy()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # if args.merge and args.prior_checkpoint and args.prior_checkpoint2:
    #     # 这个是最新的，deformation的网络要用这个
    #     gaussians_new = GaussianModel(dataset.sh_degree, hyper)
    #     (model_params, first_iter) = torch.load(args.prior_checkpoint2)
    #     gaussians_new.restore(model_params, opt)
    #     deformation_net = gaussians_new._deformation
    #     del gaussians_new
    #     gc.collect()
    #     torch.cuda.empty_cache()
        
    #     # 这个是上一个
    #     gaussians_prev = GaussianModel(dataset.sh_degree, hyper)
    #     (model_params, first_iter) = torch.load(args.prior_checkpoint)
    #     gaussians_prev.restore(model_params, opt)
               
    #     gaussians_prev._deformation = deformation_net.to('cuda')

        
    #     do_evaluation(
    #         viewpoint_stack_full,
    #         viewpoint_stack_test,
    #         gaussians_prev,
    #         background,
    #         pipe,
    #         eval_dir,
    #         render_full=True,
    #         step=99999,
    #         args=args
    #     )
        
        # merge
        # gaussians = merge_models(gaussians_new, gaussians_prev, hyper, gaussians)
           
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                             checkpoint_iterations, checkpoint, debug_from,
                             gaussians, scene, "coarse", tb_writer, opt.coarse_iterations,timer)

    if args.prior_checkpoint:
        assert 'fine' in args.prior_checkpoint

        gaussians_prev = GaussianModel(dataset.sh_degree, hyper)
        
        (model_params, first_iter) = torch.load(args.prior_checkpoint)
        gaussians_prev.restore(model_params, opt)
            
        deformation_net = gaussians_prev._deformation
        del gaussians_prev
        gc.collect()
        torch.cuda.empty_cache()
        
        gaussians._deformation = deformation_net.to('cuda')
    
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, "fine", tb_writer, opt.iterations,timer)
    
    do_evaluation(
        viewpoint_stack_full,
        viewpoint_stack_test,
        viewpoint_stack_train,
        gaussians,
        background,
        pipe,
        eval_dir,
        render_full=True,
        step=opt.iterations,
        args=args
    )

def prepare_output_and_logger(expname):    
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, stage):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)
        
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        if len(scene.getTestCameras()) != 0: 
            validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]})
        else:
            validation_configs = ({'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    except:
                        pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    # mask=viewpoint.mask
                    
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                # print("sh feature",scene.gaussians.get_features.shape)
                if tb_writer:
                    tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            
            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate', scene.gaussians._deformation_table.sum()/scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_histogram(f"{stage}/scene/motion_histogram", scene.gaussians._deformation_accum.mean(dim=-1)/100, iteration,max_bins=500)
        
        torch.cuda.empty_cache()
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000,7000,14000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[ 14000, 20000, 30_000, 45000, 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[10_000,20_000,30_000,40_000,50_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "waymo")
    parser.add_argument("--configs", type=str, default = "")
    parser.add_argument("--eval_only", action="store_true", help="perform evaluation only")
    parser.add_argument("--prior_checkpoint", type=str, default = None)
    parser.add_argument("--merge", action="store_true", help="merge gaussians")
    parser.add_argument("--prior_checkpoint2", type=str, default = None)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname)

    # All done
    print("\nTraining complete.")
