import logging
import os
from collections import namedtuple
from itertools import accumulate
from typing import List, Optional, Union

import matplotlib.cm as cm
import numpy as np
import plotly.graph_objects as go
import torch
# from omegaconf import OmegaConf
from scipy import ndimage
from tqdm import tqdm
import json
import cv2

# from datasets import SceneDataset
# from datasets.utils import voxel_coords_to_world_coords, world_coords_to_voxel_coords
# from radiance_fields import DensityField, RadianceField
# from radiance_fields.render_utils import render_rays
# from third_party.nerfacc_prop_net import PropNetEstimator
# from utils.misc import get_robust_pca
# from utils.misc import NumpyEncoder

DEFAULT_TRANSITIONS = (15, 6, 4, 11, 13, 6)

logger = logging.getLogger()
turbo_cmap = cm.get_cmap("turbo")

# 定义函数用于将光流可视化为RGB颜色
def flow_to_color(flow):
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# 定义函数用于计算光流并保存可视化结果
def compute_optical_flow_and_save(frames, output_path):
    # 初始化光流计算器
    prev_frame = frames[0]
    hsv = np.zeros_like(prev_frame)
    hsv[..., 1] = 255

    # 逐帧计算光流并保存可视化结果
    for i in range(1, len(frames)):
        next_frame = frames[i]

        # 计算光流
        flow = cv2.calcOpticalFlowFarneback(
            cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY),
            None, 0.5, 5, 15, 5, 7, 1.5, 0)

        # 将光流转换为RGB颜色
        flow_rgb = flow_to_color(flow)

        # 保存可视化结果
        cv2.imwrite(f"{output_path}/optical_flow_{i}.jpg", flow_rgb)

        # 更新前一帧
        prev_frame = next_frame


def to8b(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def resize_five_views(imgs: np.array):
    if len(imgs) != 5:
        return imgs
    for idx in [0, -1]:
        img = imgs[idx]
        new_shape = [int(img.shape[1] * 0.46), img.shape[1], 3]
        new_img = np.zeros_like(img)
        new_img[-new_shape[0] :, : new_shape[1], :] = ndimage.zoom(
            img, [new_shape[0] / img.shape[0], new_shape[1] / img.shape[1], 1]
        )
        # clip the image to 0-1
        new_img = np.clip(new_img, 0, 1)
        imgs[idx] = new_img
    return imgs


def sinebow(h):
    """A cyclic and uniform colormap, see http://basecase.org/env/on-rainbows."""
    f = lambda x: np.sin(np.pi * x) ** 2
    return np.stack([f(3 / 6 - h), f(5 / 6 - h), f(7 / 6 - h)], -1)


def matte(vis, acc, dark=0.8, light=1.0, width=8):
    """Set non-accumulated pixels to a Photoshop-esque checker pattern."""
    bg_mask = np.logical_xor(
        (np.arange(acc.shape[0]) % (2 * width) // width)[:, None],
        (np.arange(acc.shape[1]) % (2 * width) // width)[None, :],
    )
    bg = np.where(bg_mask, light, dark)
    return vis * acc[:, :, None] + (bg * (1 - acc))[:, :, None]


def weighted_percentile(x, w, ps, assume_sorted=False):
    """Compute the weighted percentile(s) of a single vector."""
    x = x.reshape([-1])
    w = w.reshape([-1])
    if not assume_sorted:
        sortidx = np.argsort(x)
        x, w = x[sortidx], w[sortidx]
    acc_w = np.cumsum(w)
    return np.interp(np.array(ps) * (acc_w[-1] / 100), acc_w, x)


def visualize_cmap(
    value,
    weight,
    colormap,
    lo=None,
    hi=None,
    percentile=99.0,
    curve_fn=lambda x: x,
    modulus=None,
    matte_background=True,
):
    """Visualize a 1D image and a 1D weighting according to some colormap.
    from mipnerf

    Args:
      value: A 1D image.
      weight: A weight map, in [0, 1].
      colormap: A colormap function.
      lo: The lower bound to use when rendering, if None then use a percentile.
      hi: The upper bound to use when rendering, if None then use a percentile.
      percentile: What percentile of the value map to crop to when automatically
        generating `lo` and `hi`. Depends on `weight` as well as `value'.
      curve_fn: A curve function that gets applied to `value`, `lo`, and `hi`
        before the rest of visualization. Good choices: x, 1/(x+eps), log(x+eps).
      modulus: If not None, mod the normalized value by `modulus`. Use (0, 1]. If
        `modulus` is not None, `lo`, `hi` and `percentile` will have no effect.
      matte_background: If True, matte the image over a checkerboard.

    Returns:
      A colormap rendering.
    """
    # Identify the values that bound the middle of `value' according to `weight`.
    if lo is None or hi is None:
        lo_auto, hi_auto = weighted_percentile(
            value, weight, [50 - percentile / 2, 50 + percentile / 2]
        )
        # If `lo` or `hi` are None, use the automatically-computed bounds above.
        eps = np.finfo(np.float32).eps
        lo = lo or (lo_auto - eps)
        hi = hi or (hi_auto + eps)

    # Curve all values.
    value, lo, hi = [curve_fn(x) for x in [value, lo, hi]]

    # Wrap the values around if requested.
    if modulus:
        value = np.mod(value, modulus) / modulus
    else:
        # Otherwise, just scale to [0, 1].
        value = np.nan_to_num(
            np.clip((value - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1)
        )
    if weight is not None:
        value *= weight
    else:
        weight = np.ones_like(value)
    if colormap:
        colorized = colormap(value)[..., :3]
    else:
        assert len(value.shape) == 3 and value.shape[-1] == 3
        colorized = value

    return matte(colorized, weight) if matte_background else colorized


def visualize_depth(
    x, acc=None, lo=None, hi=None, depth_curve_fn=lambda x: -np.log(x + 1e-6)
):
    """Visualizes depth maps."""
    return visualize_cmap(
        x,
        acc,
        cm.get_cmap("turbo"),
        curve_fn=depth_curve_fn,
        lo=lo,
        hi=hi,
        matte_background=False,
    )


def _make_colorwheel(transitions: tuple = DEFAULT_TRANSITIONS) -> torch.Tensor:
    """Creates a colorwheel (borrowed/modified from flowpy).
    A colorwheel defines the transitions between the six primary hues:
    Red(255, 0, 0), Yellow(255, 255, 0), Green(0, 255, 0), Cyan(0, 255, 255), Blue(0, 0, 255) and Magenta(255, 0, 255).
    Args:
        transitions: Contains the length of the six transitions, based on human color perception.
    Returns:
        colorwheel: The RGB values of the transitions in the color space.
    Notes:
        For more information, see:
        https://web.archive.org/web/20051107102013/http://members.shaw.ca/quadibloc/other/colint.htm
        http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    """
    colorwheel_length = sum(transitions)
    # The red hue is repeated to make the colorwheel cyclic
    base_hues = map(
        np.array,
        (
            [255, 0, 0],
            [255, 255, 0],
            [0, 255, 0],
            [0, 255, 255],
            [0, 0, 255],
            [255, 0, 255],
            [255, 0, 0],
        ),
    )
    colorwheel = np.zeros((colorwheel_length, 3), dtype="uint8")
    hue_from = next(base_hues)
    start_index = 0
    for hue_to, end_index in zip(base_hues, accumulate(transitions)):
        transition_length = end_index - start_index
        colorwheel[start_index:end_index] = np.linspace(
            hue_from, hue_to, transition_length, endpoint=False
        )
        hue_from = hue_to
        start_index = end_index
    return torch.FloatTensor(colorwheel)


WHEEL = _make_colorwheel()
N_COLS = len(WHEEL)
WHEEL = torch.vstack((WHEEL, WHEEL[0]))  # Make the wheel cyclic for interpolation


def scene_flow_to_rgb(
    flow: torch.Tensor,
    flow_max_radius: Optional[float] = None,
    background: Optional[str] = "dark",
) -> torch.Tensor:
    """Creates a RGB representation of an optical flow (borrowed/modified from flowpy).
    Adapted from https://github.com/Lilac-Lee/Neural_Scene_Flow_Prior/blob/main/visualize.py
    Args:
        flow: scene flow.
            flow[..., 0] should be the x-displacement
            flow[..., 1] should be the y-displacement
            flow[..., 2] should be the z-displacement
        flow_max_radius: Set the radius that gives the maximum color intensity, useful for comparing different flows.
            Default: The normalization is based on the input flow maximum radius.
        background: States if zero-valued flow should look 'bright' or 'dark'.
    Returns: An array of RGB colors.
    """
    flow_min = flow.min()  # 找到最小值
    flow_max = flow.max()  # 找到最大值
    eps = 1e-6  # 一个小常数，防止除以零
    flow = (flow - flow_min) / (flow_max - flow_min + eps)  # 归一化，避免除零错误
    # flow = flow * 100

    valid_backgrounds = ("bright", "dark")
    if background not in valid_backgrounds:
        raise ValueError(
            f"background should be one the following: {valid_backgrounds}, not {background}."
        )

    # For scene flow, it's reasonable to assume displacements in x and y directions only for visualization pursposes.
    complex_flow = flow[..., 0] + 1j * flow[..., 1]
    radius, angle = torch.abs(complex_flow), torch.angle(complex_flow)
    if flow_max_radius is None:
        # flow_max_radius = torch.max(radius)
        flow_max_radius = torch.quantile(radius, 0.99)
    if flow_max_radius > 0:
        radius /= flow_max_radius
    # Map the angles from (-pi, pi] to [0, 2pi) to [0, ncols - 1)
    angle[angle < 0] += 2 * np.pi
    angle = angle * ((N_COLS - 1) / (2 * np.pi))

    # Interpolate the hues
    angle_fractional, angle_floor, angle_ceil = (
        torch.fmod(angle, 1),
        angle.trunc(),
        torch.ceil(angle),
    )
    angle_fractional = angle_fractional.unsqueeze(-1)
    wheel = WHEEL.to(angle_floor.device)
    float_hue = (
        wheel[angle_floor.long()] * (1 - angle_fractional)
        + wheel[angle_ceil.long()] * angle_fractional
    )
    ColorizationArgs = namedtuple(
        "ColorizationArgs",
        ["move_hue_valid_radius", "move_hue_oversized_radius", "invalid_color"],
    )

    def move_hue_on_V_axis(hues, factors):
        return hues * factors.unsqueeze(-1)

    def move_hue_on_S_axis(hues, factors):
        return 255.0 - factors.unsqueeze(-1) * (255.0 - hues)

    if background == "dark":
        parameters = ColorizationArgs(
            move_hue_on_V_axis, move_hue_on_S_axis, torch.FloatTensor([255, 255, 255])
        )
    else:
        parameters = ColorizationArgs(
            move_hue_on_S_axis, move_hue_on_V_axis, torch.zeros(3)
        )
    colors = parameters.move_hue_valid_radius(float_hue, radius)
    oversized_radius_mask = radius > 1
    colors[oversized_radius_mask] = parameters.move_hue_oversized_radius(
        float_hue[oversized_radius_mask], 1 / radius[oversized_radius_mask]
    )
    # print(colors.max())
    # print(colors.min())

    return colors / 255.0


def vis_occ_plotly(
    vis_aabb: List[Union[int, float]],
    coords: np.array = None,
    colors: np.array = None,
    dynamic_coords: List[np.array] = None,
    dynamic_colors: List[np.array] = None,
    x_ratio: float = 1.0,
    y_ratio: float = 1.0,
    z_ratio: float = 0.125,
    size: int = 5,
    black_bg: bool = False,
    title: str = None,
) -> go.Figure:  # type: ignore
    fig = go.Figure()  # start with an empty figure

    if coords is not None:
        # Add static trace
        static_trace = go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers",
            marker=dict(
                size=size,
                color=colors,
                symbol="square",
            ),
        )
        fig.add_trace(static_trace)

    # Add temporal traces
    if dynamic_coords is not None:
        for i in range(len(dynamic_coords)):
            fig.add_trace(
                go.Scatter3d(
                    x=dynamic_coords[i][:, 0],
                    y=dynamic_coords[i][:, 1],
                    z=dynamic_coords[i][:, 2],
                    mode="markers",
                    marker=dict(
                        size=size,
                        color=dynamic_colors[i],
                        symbol="diamond",
                    ),
                )
            )
        steps = []
        if coords is not None:
            for i in range(len(dynamic_coords)):
                step = dict(
                    method="restyle",
                    args=[
                        "visible",
                        [False] * (len(dynamic_coords) + 1),
                    ],  # Include the static trace
                    label=f"Second {i}",
                )
                step["args"][1][0] = True  # Make the static trace always visible
                step["args"][1][i + 1] = True  # Toggle i'th temporal trace to "visible"
                steps.append(step)
        else:
            for i in range(len(dynamic_coords)):
                step = dict(
                    method="restyle",
                    args=[
                        "visible",
                        [False] * (len(dynamic_coords)),
                    ],
                    label=f"Second {i}",
                )
                step["args"][1][i] = True  # Toggle i'th temporal trace to "visible"
                steps.append(step)

        sliders = [
            dict(
                active=0,
                pad={"t": 1},
                steps=steps,
                font=dict(color="white") if black_bg else {},  # Update for font color
            )
        ]
        fig.update_layout(sliders=sliders)
    title_font_color = "white" if black_bg else "black"
    if not black_bg:
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title="x",
                    showspikes=False,
                    range=[vis_aabb[0], vis_aabb[3]],
                ),
                yaxis=dict(
                    title="y",
                    showspikes=False,
                    range=[vis_aabb[1], vis_aabb[4]],
                ),
                zaxis=dict(
                    title="z",
                    showspikes=False,
                    range=[vis_aabb[2], vis_aabb[5]],
                ),
                aspectmode="manual",
                aspectratio=dict(x=x_ratio, y=y_ratio, z=z_ratio),
            ),
            margin=dict(r=0, b=10, l=0, t=10),
            hovermode=False,
            title=dict(
                text=title,
                font=dict(color=title_font_color),
                x=0.5,
                y=0.95,
                xanchor="center",
                yanchor="top",
            )
            if title
            else None,  # Title addition
        )
    else:
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title="x",
                    showspikes=False,
                    range=[vis_aabb[0], vis_aabb[3]],
                    backgroundcolor="rgb(0, 0, 0)",
                    gridcolor="gray",
                    showbackground=True,
                    zerolinecolor="gray",
                    tickfont=dict(color="gray"),
                ),
                yaxis=dict(
                    title="y",
                    showspikes=False,
                    range=[vis_aabb[1], vis_aabb[4]],
                    backgroundcolor="rgb(0, 0, 0)",
                    gridcolor="gray",
                    showbackground=True,
                    zerolinecolor="gray",
                    tickfont=dict(color="gray"),
                ),
                zaxis=dict(
                    title="z",
                    showspikes=False,
                    range=[vis_aabb[2], vis_aabb[5]],
                    backgroundcolor="rgb(0, 0, 0)",
                    gridcolor="gray",
                    showbackground=True,
                    zerolinecolor="gray",
                    tickfont=dict(color="gray"),
                ),
                aspectmode="manual",
                aspectratio=dict(x=x_ratio, y=y_ratio, z=z_ratio),
            ),
            margin=dict(r=0, b=10, l=0, t=10),
            hovermode=False,
            paper_bgcolor="black",
            plot_bgcolor="rgba(0,0,0,0)",
            title=dict(
                text=title,
                font=dict(color=title_font_color),
                x=0.5,
                y=0.95,
                xanchor="center",
                yanchor="top",
            )
            if title
            else None,  # Title addition
        )
    eye = np.array([-1, 0, 0.5])
    eye = eye.tolist()
    fig.update_layout(
        scene_camera=dict(
            eye=dict(x=eye[0], y=eye[1], z=eye[2]),
        ),
    )
    return fig