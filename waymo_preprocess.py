# Acknowledgement:
#   1. https://github.com/open-mmlab/mmdetection3d/blob/main/tools/dataset_converters/waymo_converter.py
#   2. https://github.com/leolyj/DCA-SRSFE/blob/main/data_preprocess/Waymo/generate_flow.py
try:
    from waymo_open_dataset import dataset_pb2
except ImportError:
    raise ImportError(
        'Please run "pip install waymo-open-dataset-tf-2-6-0" '
        ">1.4.5 to install the official devkit first."
    )

import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import camera_segmentation_pb2 as cs_pb2
from waymo_open_dataset.utils import box_utils, range_image_utils, transform_utils
from waymo_open_dataset.utils.frame_utils import parse_range_image_and_camera_projection
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import camera_segmentation_utils

import sys
import time
from collections.abc import Iterable
from multiprocessing import Pool
from shutil import get_terminal_size
import matplotlib.cm as cm

class ProgressBar:
    """A progress bar which can print the progress."""

    def __init__(self, task_num=0, bar_width=50, start=True, file=sys.stdout):
        self.task_num = task_num
        self.bar_width = bar_width
        self.completed = 0
        self.file = file
        if start:
            self.start()

    @property
    def terminal_width(self):
        width, _ = get_terminal_size()
        return width

    def start(self):
        if self.task_num > 0:
            self.file.write(
                f'[{" " * self.bar_width}] 0/{self.task_num}, ' "elapsed: 0s, ETA:"
            )
        else:
            self.file.write("completed: 0, elapsed: 0s")
        self.file.flush()
        self.start_time = time.time()

    def update(self, num_tasks=1):
        assert num_tasks > 0
        self.completed += num_tasks
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            fps = self.completed / elapsed
        else:
            fps = float("inf")
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            msg = (
                f"\r[{{}}] {self.completed}/{self.task_num}, "
                f"{fps:.1f} task/s, elapsed: {int(elapsed + 0.5)}s, "
                f"ETA: {eta:5}s"
            )

            bar_width = min(
                self.bar_width,
                int(self.terminal_width - len(msg)) + 2,
                int(self.terminal_width * 0.6),
            )
            bar_width = max(2, bar_width)
            mark_width = int(bar_width * percentage)
            bar_chars = ">" * mark_width + " " * (bar_width - mark_width)
            self.file.write(msg.format(bar_chars))
        else:
            self.file.write(
                f"completed: {self.completed}, elapsed: {int(elapsed + 0.5)}s,"
                f" {fps:.1f} tasks/s"
            )
        self.file.flush()


def init_pool(process_num, initializer=None, initargs=None):
    if initializer is None:
        return Pool(process_num)
    elif initargs is None:
        return Pool(process_num, initializer)
    else:
        if not isinstance(initargs, tuple):
            raise TypeError('"initargs" must be a tuple')
        return Pool(process_num, initializer, initargs)


def track_parallel_progress(
    func,
    tasks,
    nproc,
    initializer=None,
    initargs=None,
    bar_width=50,
    chunksize=1,
    skip_first=False,
    keep_order=True,
    file=sys.stdout,
):
    """Track the progress of parallel task execution with a progress bar.

    The built-in :mod:`multiprocessing` module is used for process pools and
    tasks are done with :func:`Pool.map` or :func:`Pool.imap_unordered`.

    Args:
        func (callable): The function to be applied to each task.
        tasks (list or tuple[Iterable, int]): A list of tasks or
            (tasks, total num).
        nproc (int): Process (worker) number.
        initializer (None or callable): Refer to :class:`multiprocessing.Pool`
            for details.
        initargs (None or tuple): Refer to :class:`multiprocessing.Pool` for
            details.
        chunksize (int): Refer to :class:`multiprocessing.Pool` for details.
        bar_width (int): Width of progress bar.
        skip_first (bool): Whether to skip the first sample for each worker
            when estimating fps, since the initialization step may takes
            longer.
        keep_order (bool): If True, :func:`Pool.imap` is used, otherwise
            :func:`Pool.imap_unordered` is used.

    Returns:
        list: The task results.
    """
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]
    elif isinstance(tasks, Iterable):
        task_num = len(tasks)
    else:
        raise TypeError('"tasks" must be an iterable object or a (iterator, int) tuple')
    pool = init_pool(nproc, initializer, initargs)
    start = not skip_first
    task_num -= nproc * chunksize * int(skip_first)
    prog_bar = ProgressBar(task_num, bar_width, start, file=file)
    results = []
    if keep_order:
        gen = pool.imap(func, tasks, chunksize)
    else:
        gen = pool.imap_unordered(func, tasks, chunksize)
    for result in gen:
        results.append(result)
        if skip_first:
            if len(results) < nproc * chunksize:
                continue
            elif len(results) == nproc * chunksize:
                prog_bar.start()
                continue
        prog_bar.update()
    prog_bar.file.write("\n")
    pool.close()
    pool.join()
    return results

def get_ground_np(pts):
    """
    This function performs ground removal on a point cloud.
    Modified from https://github.com/tusen-ai/LiDAR_SOT/blob/main/waymo_data/data_preprocessing/ground_removal.py

    Args:
        pts (numpy.ndarray): The input point cloud.

    Returns:
        numpy.ndarray: A boolean array indicating whether each point is ground or not.
    """
    th_seeds_ = 1.2
    num_lpr_ = 20
    n_iter = 10
    th_dist_ = 0.3
    pts_sort = pts[pts[:, 2].argsort(), :]
    lpr = np.mean(pts_sort[:num_lpr_, 2])
    pts_g = pts_sort[pts_sort[:, 2] < lpr + th_seeds_, :]
    normal_ = np.zeros(3)
    for i in range(n_iter):
        mean = np.mean(pts_g, axis=0)[:3]
        xx = np.mean((pts_g[:, 0] - mean[0]) * (pts_g[:, 0] - mean[0]))
        xy = np.mean((pts_g[:, 0] - mean[0]) * (pts_g[:, 1] - mean[1]))
        xz = np.mean((pts_g[:, 0] - mean[0]) * (pts_g[:, 2] - mean[2]))
        yy = np.mean((pts_g[:, 1] - mean[1]) * (pts_g[:, 1] - mean[1]))
        yz = np.mean((pts_g[:, 1] - mean[1]) * (pts_g[:, 2] - mean[2]))
        zz = np.mean((pts_g[:, 2] - mean[2]) * (pts_g[:, 2] - mean[2]))
        cov = np.array(
            [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]],
            dtype=np.float32,
        )
        U, S, V = np.linalg.svd(cov)
        normal_ = U[:, 2]
        d_ = -normal_.dot(mean)
        th_dist_d_ = th_dist_ - d_
        result = pts[:, :3] @ normal_[..., np.newaxis]
        pts_g = pts[result.squeeze(-1) < th_dist_d_]
    ground_label = result < th_dist_d_
    return ground_label

def weighted_percentile(x, w, ps, assume_sorted=False):
    """Compute the weighted percentile(s) of a single vector."""
    x = x.reshape([-1])
    w = w.reshape([-1])
    if not assume_sorted:
        sortidx = np.argsort(x)
        x, w = x[sortidx], w[sortidx]
    acc_w = np.cumsum(w)
    return np.interp(np.array(ps) * (acc_w[-1] / 100), acc_w, x)
def matte(vis, acc, dark=0.8, light=1.0, width=8):
    """Set non-accumulated pixels to a Photoshop-esque checker pattern."""
    bg_mask = np.logical_xor(
        (np.arange(acc.shape[0]) % (2 * width) // width)[:, None],
        (np.arange(acc.shape[1]) % (2 * width) // width)[None, :],
    )
    bg = np.where(bg_mask, light, dark)
    return vis * acc[:, :, None] + (bg * (1 - acc))[:, :, None]
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

depth_visualizer = lambda frame, opacity: visualize_depth(
    frame,
    opacity,
    lo=None,
    hi=None,
    depth_curve_fn=lambda x: x,
)

MOVEABLE_OBJECTS_IDS = [
    cs_pb2.CameraSegmentation.TYPE_CAR,
    cs_pb2.CameraSegmentation.TYPE_TRUCK,
    cs_pb2.CameraSegmentation.TYPE_BUS,
    cs_pb2.CameraSegmentation.TYPE_OTHER_LARGE_VEHICLE,
    cs_pb2.CameraSegmentation.TYPE_BICYCLE,
    cs_pb2.CameraSegmentation.TYPE_MOTORCYCLE,
    cs_pb2.CameraSegmentation.TYPE_TRAILER,
    cs_pb2.CameraSegmentation.TYPE_PEDESTRIAN,
    cs_pb2.CameraSegmentation.TYPE_CYCLIST,
    cs_pb2.CameraSegmentation.TYPE_MOTORCYCLIST,
    cs_pb2.CameraSegmentation.TYPE_BIRD,
    cs_pb2.CameraSegmentation.TYPE_GROUND_ANIMAL,
    cs_pb2.CameraSegmentation.TYPE_PEDESTRIAN_OBJECT,
]

camera_left_to_right_order = [open_dataset.CameraName.SIDE_LEFT,
                              open_dataset.CameraName.FRONT_LEFT,
                              open_dataset.CameraName.FRONT,
                              open_dataset.CameraName.FRONT_RIGHT,
                              open_dataset.CameraName.SIDE_RIGHT]

def project_vehicle_to_image(vehicle_pose, calibration, points):
    """Projects from vehicle coordinate system to image with global shutter.

    Arguments:
      vehicle_pose: Vehicle pose transform from vehicle into world coordinate
        system.
      calibration: Camera calibration details (including intrinsics/extrinsics).
      points: Points to project of shape [N, 3] in vehicle coordinate system.

    Returns:
      Array of shape [N, 3], with the latter dimension composed of (u, v, ok).
    """
    # Transform points from vehicle to world coordinate system (can be
    # vectorized).
    pose_matrix = np.array(vehicle_pose.transform).reshape(4, 4)
    world_points = np.zeros_like(points)
    for i, point in enumerate(points):
        cx, cy, cz, _ = np.matmul(pose_matrix, [*point, 1])
        world_points[i] = (cx, cy, cz)

    # Populate camera image metadata. Velocity and latency stats are filled with
    # zeroes.
    extrinsic = tf.reshape(
        tf.constant(list(calibration.extrinsic.transform), dtype=tf.float32), [4, 4]
    )
    intrinsic = tf.constant(list(calibration.intrinsic), dtype=tf.float32)
    metadata = tf.constant(
        [
            calibration.width,
            calibration.height,
            dataset_pb2.CameraCalibration.GLOBAL_SHUTTER,
        ],
        dtype=tf.int32,
    )
    camera_image_metadata = list(vehicle_pose.transform) + [0.0] * 10

    # Perform projection and return projected image coordinates (u, v, ok).
    return py_camera_model_ops.world_to_image(
        extrinsic, intrinsic, metadata, camera_image_metadata, world_points
    ).numpy()


def compute_range_image_cartesian(
    range_image_polar,
    extrinsic,
    pixel_pose=None,
    frame_pose=None,
    dtype=tf.float32,
    scope=None,
):
    """Computes range image cartesian coordinates from polar ones.

    Args:
      range_image_polar: [B, H, W, 3] float tensor. Lidar range image in polar
        coordinate in sensor frame.
      extrinsic: [B, 4, 4] float tensor. Lidar extrinsic.
      pixel_pose: [B, H, W, 4, 4] float tensor. If not None, it sets pose for each
        range image pixel.
      frame_pose: [B, 4, 4] float tensor. This must be set when pixel_pose is set.
        It decides the vehicle frame at which the cartesian points are computed.
      dtype: float type to use internally. This is needed as extrinsic and
        inclination sometimes have higher resolution than range_image.
      scope: the name scope.

    Returns:
      range_image_cartesian: [B, H, W, 3] cartesian coordinates.
    """
    range_image_polar_dtype = range_image_polar.dtype
    range_image_polar = tf.cast(range_image_polar, dtype=dtype)
    extrinsic = tf.cast(extrinsic, dtype=dtype)
    if pixel_pose is not None:
        pixel_pose = tf.cast(pixel_pose, dtype=dtype)
    if frame_pose is not None:
        frame_pose = tf.cast(frame_pose, dtype=dtype)

    with tf.compat.v1.name_scope(
        scope,
        "ComputeRangeImageCartesian",
        [range_image_polar, extrinsic, pixel_pose, frame_pose],
    ):
        azimuth, inclination, range_image_range = tf.unstack(range_image_polar, axis=-1)

        cos_azimuth = tf.cos(azimuth)
        sin_azimuth = tf.sin(azimuth)
        cos_incl = tf.cos(inclination)
        sin_incl = tf.sin(inclination)

        # [B, H, W].
        x = cos_azimuth * cos_incl * range_image_range
        y = sin_azimuth * cos_incl * range_image_range
        z = sin_incl * range_image_range

        # [B, H, W, 3]
        range_image_points = tf.stack([x, y, z], -1)
        range_image_origins = tf.zeros_like(range_image_points)
        # [B, 3, 3]
        rotation = extrinsic[..., 0:3, 0:3]
        # translation [B, 1, 3]
        translation = tf.expand_dims(tf.expand_dims(extrinsic[..., 0:3, 3], 1), 1)

        # To vehicle frame.
        # [B, H, W, 3]
        range_image_points = (
            tf.einsum("bkr,bijr->bijk", rotation, range_image_points) + translation
        )
        range_image_origins = (
            tf.einsum("bkr,bijr->bijk", rotation, range_image_origins) + translation
        )
        if pixel_pose is not None:
            # To global frame.
            # [B, H, W, 3, 3]
            pixel_pose_rotation = pixel_pose[..., 0:3, 0:3]
            # [B, H, W, 3]
            pixel_pose_translation = pixel_pose[..., 0:3, 3]
            # [B, H, W, 3]
            range_image_points = (
                tf.einsum("bhwij,bhwj->bhwi", pixel_pose_rotation, range_image_points)
                + pixel_pose_translation
            )
            range_image_origins = (
                tf.einsum("bhwij,bhwj->bhwi", pixel_pose_rotation, range_image_origins)
                + pixel_pose_translation
            )

            if frame_pose is None:
                raise ValueError("frame_pose must be set when pixel_pose is set.")
            # To vehicle frame corresponding to the given frame_pose
            # [B, 4, 4]
            world_to_vehicle = tf.linalg.inv(frame_pose)
            world_to_vehicle_rotation = world_to_vehicle[:, 0:3, 0:3]
            world_to_vehicle_translation = world_to_vehicle[:, 0:3, 3]
            # [B, H, W, 3]
            range_image_points = (
                tf.einsum(
                    "bij,bhwj->bhwi", world_to_vehicle_rotation, range_image_points
                )
                + world_to_vehicle_translation[:, tf.newaxis, tf.newaxis, :]
            )
            range_image_origins = (
                tf.einsum(
                    "bij,bhwj->bhwi", world_to_vehicle_rotation, range_image_origins
                )
                + world_to_vehicle_translation[:, tf.newaxis, tf.newaxis, :]
            )

        range_image_points = tf.cast(range_image_points, dtype=range_image_polar_dtype)
        range_image_origins = tf.cast(
            range_image_origins, dtype=range_image_polar_dtype
        )
        return range_image_points, range_image_origins


def extract_point_cloud_from_range_image(
    range_image,
    extrinsic,
    inclination,
    pixel_pose=None,
    frame_pose=None,
    dtype=tf.float32,
    scope=None,
):
    """Extracts point cloud from range image.

    Args:
      range_image: [B, H, W] tensor. Lidar range images.
      extrinsic: [B, 4, 4] tensor. Lidar extrinsic.
      inclination: [B, H] tensor. Inclination for each row of the range image.
        0-th entry corresponds to the 0-th row of the range image.
      pixel_pose: [B, H, W, 4, 4] tensor. If not None, it sets pose for each range
        image pixel.
      frame_pose: [B, 4, 4] tensor. This must be set when pixel_pose is set. It
        decides the vehicle frame at which the cartesian points are computed.
      dtype: float type to use internally. This is needed as extrinsic and
        inclination sometimes have higher resolution than range_image.
      scope: the name scope.

    Returns:
      range_image_points: [B, H, W, 3] with {x, y, z} as inner dims in vehicle frame.
      range_image_origins: [B, H, W, 3] with {x, y, z}, the origin of the range image
    """
    with tf.compat.v1.name_scope(
        scope,
        "ExtractPointCloudFromRangeImage",
        [range_image, extrinsic, inclination, pixel_pose, frame_pose],
    ):
        range_image_polar = range_image_utils.compute_range_image_polar(
            range_image, extrinsic, inclination, dtype=dtype
        )
        (
            range_image_points_cartesian,
            range_image_origins_cartesian,
        ) = compute_range_image_cartesian(
            range_image_polar,
            extrinsic,
            pixel_pose=pixel_pose,
            frame_pose=frame_pose,
            dtype=dtype,
        )
        return range_image_origins_cartesian, range_image_points_cartesian


def parse_range_image_flow_and_camera_projection(frame):
    range_images = {}
    camera_projections = {}
    range_image_top_pose = None
    for laser in frame.lasers:
        if (
            len(laser.ri_return1.range_image_flow_compressed) > 0
        ):  # pylint: disable=g-explicit-length-test
            range_image_str_tensor = tf.io.decode_compressed(
                laser.ri_return1.range_image_flow_compressed, "ZLIB"
            )
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
            range_images[laser.name] = [ri]

            if laser.name == dataset_pb2.LaserName.TOP:
                range_image_top_pose_str_tensor = tf.io.decode_compressed(
                    laser.ri_return1.range_image_pose_compressed, "ZLIB"
                )
                range_image_top_pose = dataset_pb2.MatrixFloat()
                range_image_top_pose.ParseFromString(
                    bytearray(range_image_top_pose_str_tensor.numpy())
                )

            camera_projection_str_tensor = tf.io.decode_compressed(
                laser.ri_return1.camera_projection_compressed, "ZLIB"
            )
            cp = dataset_pb2.MatrixInt32()
            cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
            camera_projections[laser.name] = [cp]
        if (
            len(laser.ri_return2.range_image_flow_compressed) > 0
        ):  # pylint: disable=g-explicit-length-test
            range_image_str_tensor = tf.io.decode_compressed(
                laser.ri_return2.range_image_flow_compressed, "ZLIB"
            )
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
            range_images[laser.name].append(ri)

            camera_projection_str_tensor = tf.io.decode_compressed(
                laser.ri_return2.camera_projection_compressed, "ZLIB"
            )
            cp = dataset_pb2.MatrixInt32()
            cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
            camera_projections[laser.name].append(cp)
    return range_images, camera_projections, range_image_top_pose


def convert_range_image_to_point_cloud_flow(
    frame,
    range_images,
    range_images_flow,
    camera_projections,
    range_image_top_pose,
    ri_index=0,
):
    """
    Modified from the codes of Waymo Open Dataset.
    Convert range images to point cloud.
    Convert range images flow to scene flow.
    Args:
        frame: open dataset frame
        range_images: A dict of {laser_name, [range_image_first_return, range_image_second_return]}.
        range_imaages_flow: A dict similar to range_images.
        camera_projections: A dict of {laser_name,
            [camera_projection_from_first_return, camera_projection_from_second_return]}.
        range_image_top_pose: range image pixel pose for top lidar.
        ri_index: 0 for the first return, 1 for the second return.

    Returns:
        points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
        points_flow: {[N, 3]} list of scene flow vector of each point.
        cp_points: {[N, 6]} list of camera projections of length 5 (number of lidars).
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    origins, points, cp_points = [], [], []
    points_intensity = []
    points_elongation = []
    points_flow = []
    laser_ids = []

    frame_pose = tf.convert_to_tensor(
        np.reshape(np.array(frame.pose.transform), [4, 4])
    )
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(range_image_top_pose.data), range_image_top_pose.shape.dims
    )
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0],
        range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2],
    )
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation, range_image_top_pose_tensor_translation
    )
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        #range_image_flow = range_images_flow[c.name][ri_index]
        if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
            beam_inclinations = range_image_utils.compute_inclination(
                tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                height=range_image.shape.dims[0],
            )
        else:
            beam_inclinations = tf.constant(c.beam_inclinations)

        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims
        )
        #range_image_flow_tensor = tf.reshape(
        #    tf.convert_to_tensor(range_image_flow.data), range_image_flow.shape.dims
        #)
        pixel_pose_local = None
        frame_pose_local = None
        if c.name == dataset_pb2.LaserName.TOP:
            pixel_pose_local = range_image_top_pose_tensor
            pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
            frame_pose_local = tf.expand_dims(frame_pose, axis=0)
        range_image_mask = range_image_tensor[..., 0] > 0
        range_image_intensity = range_image_tensor[..., 1]
        range_image_elongation = range_image_tensor[..., 2]

        #flow_x = range_image_flow_tensor[..., 0]
        #flow_y = range_image_flow_tensor[..., 1]
        #flow_z = range_image_flow_tensor[..., 2]
        #flow_class = range_image_flow_tensor[..., 3]

        mask_index = tf.where(range_image_mask)

        (origins_cartesian, points_cartesian,) = extract_point_cloud_from_range_image(
            tf.expand_dims(range_image_tensor[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
            pixel_pose=pixel_pose_local,
            frame_pose=frame_pose_local,
        )
        origins_cartesian = tf.squeeze(origins_cartesian, axis=0)
        points_cartesian = tf.squeeze(points_cartesian, axis=0)

        origins_tensor = tf.gather_nd(origins_cartesian, mask_index)
        points_tensor = tf.gather_nd(points_cartesian, mask_index)

        points_intensity_tensor = tf.gather_nd(range_image_intensity, mask_index)
        points_elongation_tensor = tf.gather_nd(range_image_elongation, mask_index)

        #points_flow_x_tensor = tf.expand_dims(tf.gather_nd(flow_x, mask_index), axis=1)
        #points_flow_y_tensor = tf.expand_dims(tf.gather_nd(flow_y, mask_index), axis=1)
        #points_flow_z_tensor = tf.expand_dims(tf.gather_nd(flow_z, mask_index), axis=1)
        #points_flow_class_tensor = tf.expand_dims(
        #    tf.gather_nd(flow_class, mask_index), axis=1
        #)

        origins.append(origins_tensor.numpy())
        points.append(points_tensor.numpy())
        points_intensity.append(points_intensity_tensor.numpy())
        points_elongation.append(points_elongation_tensor.numpy())
        laser_ids.append(np.full_like(points_intensity_tensor.numpy(), c.name - 1))

        #points_flow.append(
        #    tf.concat(
        #        [
        #            points_flow_x_tensor,
        #            points_flow_y_tensor,
        #            points_flow_z_tensor,
        #            points_flow_class_tensor,
        #        ],
        #        axis=-1,
        #    ).numpy()
        #)

    return (
        origins,
        points,
        points_flow,
        cp_points,
        points_intensity,
        points_elongation,
        laser_ids,
    )


class WaymoProcessor(object):
    """Process Waymo dataset.

    Args:
        load_dir (str): Directory to load waymo raw data.
        save_dir (str): Directory to save data in KITTI format.
        prefix (str): Prefix of filename.
        workers (int, optional): Number of workers for the parallel process.
            Defaults to 64.
            Defaults to False.
        save_cam_sync_labels (bool, optional): Whether to save cam sync labels.
            Defaults to True.
    """

    def __init__(
        self,
        load_dir,
        save_dir,
        prefix,
        process_keys=[
            "images",
            "lidar",
            "calib",
            "pose",
            "dynamic_masks",
        ],
        process_id_list=None,
        workers=64,
    ):
        self.filter_no_label_zone_points = True

        # Only data collected in specific locations will be converted
        # If set None, this filter is disabled
        # Available options: location_sf (main dataset)
        self.selected_waymo_locations = None
        self.save_track_id = False
        self.process_id_list = process_id_list
        self.process_keys = process_keys
        print("will process keys: ", self.process_keys)

        # turn on eager execution for older tensorflow versions
        if int(tf.__version__.split(".")[0]) < 2:
            tf.enable_eager_execution()

        # keep the order defined by the official protocol
        self.cam_list = [
            "_FRONT",
            "_FRONT_LEFT",
            "_FRONT_RIGHT",
            "_SIDE_LEFT",
            "_SIDE_RIGHT",
        ]
        self.lidar_list = ["TOP", "FRONT", "SIDE_LEFT", "SIDE_RIGHT", "REAR"]

        self.load_dir = load_dir
        self.save_dir = f"{save_dir}/{prefix}"
        self.workers = int(workers)
        # a list of tfrecord pathnames
        training_files = open("data/waymo/waymo_train_list.txt").read().splitlines()
        self.tfrecord_pathnames = [
            f"{self.load_dir}/{f}.tfrecord" for f in training_files
        ]
        # self.tfrecord_pathnames = sorted(glob(join(self.load_dir, "*.tfrecord")))
        self.create_folder()

    def convert(self):
        """Convert action."""
        print("Start converting ...")
        if self.process_id_list is None:
            id_list = range(len(self))
        else:
            id_list = self.process_id_list
        track_parallel_progress(self.convert_one, id_list, self.workers)
        print("\nFinished ...")

    def convert_one(self, file_idx):
        """Convert action for single file.

        Args:
            file_idx (int): Index of the file to be converted.
        """
        pathname = self.tfrecord_pathnames[file_idx]
        dataset = tf.data.TFRecordDataset(pathname, compression_type="")
        num_frames = sum(1 for _ in dataset)
        if "panoptic_segs" in self.process_keys:
            self.save_panoptic_segs(dataset, num_frames, file_idx)

        for frame_idx, data in enumerate(
            tqdm(dataset, desc=f"File {file_idx}", total=num_frames, dynamic_ncols=True)
        ):
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            if (
                self.selected_waymo_locations is not None
                and frame.context.stats.location not in self.selected_waymo_locations
            ):
                continue
            if "images" in self.process_keys:
                self.save_image(frame, file_idx, frame_idx)
            if "calib" in self.process_keys:
                self.save_calib(frame, file_idx, frame_idx)
            if "lidar" in self.process_keys:
                self.save_lidar(frame, file_idx, frame_idx)
            if "pose" in self.process_keys:
                self.save_pose(frame, file_idx, frame_idx)
            if "dynamic_masks" in self.process_keys:
                self.save_dynamic_mask(frame, file_idx, frame_idx)
            if frame_idx == 0:
                self.save_interested_labels(frame, file_idx)



    def __len__(self):
        """Length of the filename list."""
        return len(self.tfrecord_pathnames)

    def save_interested_labels(self, frame, file_idx):
        """
        Saves the interested labels of a given frame to a JSON file.

        Args:
            frame: A `Frame` object containing the labels to be saved.
            file_idx: An integer representing the index of the file to be saved.

        Returns:
            None
        """
        frame_data = {
            "time_of_day": frame.context.stats.time_of_day,
            "location": frame.context.stats.location,
            "weather": frame.context.stats.weather,
        }
        object_type_name = lambda x: label_pb2.Label.Type.Name(x)
        object_counts = {
            object_type_name(x.type): x.count
            for x in frame.context.stats.camera_object_counts
        }
        frame_data.update(object_counts)
        # write as json
        with open(
            f"{self.save_dir}/{str(file_idx).zfill(3)}/frame_info.json",
            "w",
        ) as fp:
            json.dump(frame_data, fp)

    def save_image(self, frame, file_idx, frame_idx):
        """Parse and save the images in jpg format.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        for img in frame.images:
            img_path = (
                f"{self.save_dir}/{str(file_idx).zfill(3)}/images/"
                + f"{str(frame_idx).zfill(3)}_{str(img.name - 1)}.jpg"
            )
            with open(img_path, "wb") as fp:
                fp.write(img.image)

    def save_calib(self, frame, file_idx, frame_idx):
        """Parse and save the calibration data.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        # waymo front camera to kitti reference camera
        extrinsics = []
        intrinsics = []
        for camera in frame.context.camera_calibrations:
            # extrinsic parameters
            extrinsic = np.array(camera.extrinsic.transform).reshape(4, 4)
            intrinsic = list(camera.intrinsic)
            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)
        # all camera ids are saved as id-1 in the result because
        # camera 0 is unknown in the proto
        for i in range(5):
            np.savetxt(
                f"{self.save_dir}/{str(file_idx).zfill(3)}/extrinsics/"
                + f"{str(i)}.txt",
                extrinsics[i],
            )
            np.savetxt(
                f"{self.save_dir}/{str(file_idx).zfill(3)}/intrinsics/"
                + f"{str(i)}.txt",
                intrinsics[i],
            )

    def save_lidar(self, frame, file_idx, frame_idx):
        """Parse and save the lidar data in psd format.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        (
            range_images,
            camera_projections,
            seg_labels,
            range_image_top_pose,
        ) = parse_range_image_and_camera_projection(frame)

        # https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/segmentation.proto
        if range_image_top_pose is None:
            # the camera only split doesn't contain lidar points.
            return

        # collect first return only
        #range_images_flow, _, _ = parse_range_image_flow_and_camera_projection(frame)
        (
            origins,
            points,
            flows,
            cp_points,
            intensity,
            elongation,
            laser_ids,
        ) = convert_range_image_to_point_cloud_flow(
            frame,
            range_images,
            None, #range_images_flow,
            camera_projections,
            range_image_top_pose,
            ri_index=0,
        )
        origins = np.concatenate(origins, axis=0)
        points = np.concatenate(points, axis=0)
        ground_label = get_ground_np(points)
        intensity = np.concatenate(intensity, axis=0)
        elongation = np.concatenate(elongation, axis=0)
        laser_ids = np.concatenate(laser_ids, axis=0)

        #  -1: no-flow-label, the point has no flow information.
        #   0:  unlabeled or "background,", i.e., the point is not contained in a
        #       bounding box.
        #   1: vehicle, i.e., the point corresponds to a vehicle label box.
        #   2: pedestrian, i.e., the point corresponds to a pedestrian label box.
        #   3: sign, i.e., the point corresponds to a sign label box.
        #   4: cyclist, i.e., the point corresponds to a cyclist label box.
        #flows = np.concatenate(flows, axis=0)

        point_cloud = np.column_stack(
            (
                origins,        # n, 3
                points,         # n, 3
                #flows,
                ground_label,   # n, 1
                intensity,      # n, 1
                elongation,     # n, 1
                laser_ids,      # n, 1
            )
        )
        pc_path = (
            f"{self.save_dir}/"
            + f"{str(file_idx).zfill(3)}/lidar/{str(frame_idx).zfill(3)}.bin"
        )
        point_cloud.astype(np.float32).tofile(pc_path)

    def save_pose(self, frame, file_idx, frame_idx):
        """Parse and save the pose data.

        Note that SDC's own pose is not included in the regular training
        of KITTI dataset. KITTI raw dataset contains ego motion files
        but are not often used. Pose is important for algorithms that
        take advantage of the temporal information.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        pose = np.array(frame.pose.transform).reshape(4, 4)
        np.savetxt(
            f"{self.save_dir}/{str(file_idx).zfill(3)}/ego_pose/"
            + f"{str(frame_idx).zfill(3)}.txt",
            pose,
        )

    def save_panoptic_segs(self, dataset, num_frames, file_idx):
        """Parse and save the segmentation data.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        frames_with_seg = []
        frameidx_list = []
        for frame_idx, data in enumerate(
            tqdm(dataset, desc=f"File {file_idx}", total=num_frames, dynamic_ncols=True)
        ):
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            # Save frames which contain CameraSegmentationLabel messages. We assume that
            # if the first image has segmentation labels, all images in this frame will.
            if frame.images[0].camera_segmentation_label.panoptic_label:
                frames_with_seg.append(frame)
                frameidx_list.append(frame_idx)
                #if sequence_id is None:
                #  sequence_id = frame.images[0].camera_segmentation_label.sequence_id
                ## Collect 3 frames for this demo. However, any number can be used in practice.
                #if frame.images[0].camera_segmentation_label.sequence_id != sequence_id or len(frames_with_seg) > 2:
                #  break
        print(f" {len(frameidx_list)} frames of total {num_frames} has panoptic segmentation labels. ")
        # Organize the segmentation labels in order from left to right for viz later.
        camera_front_order = camera_left_to_right_order[1:4] # use: front-left, front, front-right only
        segmentation_protos_ordered = []
        for frame in frames_with_seg:
          segmentation_proto_dict = {image.name : image.camera_segmentation_label for image in frame.images}
          segmentation_protos_ordered.append([segmentation_proto_dict[name] for name in camera_front_order])
        
        # The dataset provides tracking for instances between cameras and over time.
        # By setting remap_to_global=True, this function will remap the instance IDs in
        # each image so that instances for the same object will have the same ID between
        # different cameras and over time.
        segmentation_protos_flat = sum(segmentation_protos_ordered, [])
        panoptic_labels, num_cameras_covered, is_tracked_masks, panoptic_label_divisor = camera_segmentation_utils.decode_multi_frame_panoptic_labels_from_segmentation_labels(
            segmentation_protos_flat, remap_to_global=True
        )
        # We can further separate the semantic and instance labels from the panoptic
        # labels.
        NUM_CAMERA_FRAMES = 3 #5
        semantic_labels_multiframe = []
        instance_labels_multiframe = []
        for i in range(0, len(segmentation_protos_flat), NUM_CAMERA_FRAMES):
            semantic_labels = []
            instance_labels = []
            for j in range(NUM_CAMERA_FRAMES):
                semantic_label, instance_label = camera_segmentation_utils.decode_semantic_and_instance_labels_from_panoptic_label(
                  panoptic_labels[i + j], panoptic_label_divisor)
                semantic_labels.append(semantic_label)
                instance_labels.append(instance_label)
            semantic_labels_multiframe.append(semantic_labels)
            instance_labels_multiframe.append(instance_labels)
        ### visualize 
        ##instance_labels = semantic_labels_multiframe[0:3]
        ##semantic_labels = semantic_labels_multiframe[0:3]
        ##instance_labels = [np.concatenate(label, axis=1) for label in instance_labels]
        ##semantic_labels = [np.concatenate(label, axis=1) for label in semantic_labels]
        ##instance_label_concat = np.concatenate(instance_labels, axis=0)
        ##semantic_label_concat = np.concatenate(semantic_labels, axis=0)
        #semantic_label_concat = semantic_labels_multiframe[0][0]
        #instance_label_concat = instance_labels_multiframe[0][0]
        #panoptic_label_rgb = camera_segmentation_utils.panoptic_label_to_rgb(
        #    semantic_label_concat, instance_label_concat)
        #semantic_rgb = camera_segmentation_utils.semantic_label_to_rgb(
        #    semantic_label_concat,
        #)
        #instance_rgb = instance_label_concat.astype(np.uint8).squeeze(-1)
        ## subplot  3x1
        #import matplotlib.pyplot as plt
        #plt.figure(figsize=(10, 30))
        #plt.subplot(3, 1, 1)
        #plt.imshow(panoptic_label_rgb)
        #plt.grid(False)
        #plt.axis('off')
        #plt.subplot(3, 1, 2)
        #plt.imshow(semantic_rgb)
        #plt.grid(False)
        #plt.axis('off')
        #plt.subplot(3, 1, 3)
        #plt.imshow(instance_rgb)
        #plt.grid(False)
        #plt.axis('off')
        ##plt.show()
        ## save img
        #plt.savefig(f"panoptic_label_rgb.png")

        # Save the panoptic labels to disk.
        for key_frame_idx, (semantic_labels, instance_labels) in enumerate(
            zip(semantic_labels_multiframe, instance_labels_multiframe)
        ):
            frame_idx = frameidx_list[key_frame_idx]
            for camera_idx, semantic_label, instance_label in zip(
                camera_front_order, semantic_labels, instance_labels
            ):
                semantic_label_path = (
                    f"{self.save_dir}/{str(file_idx).zfill(3)}/semantic_segs/"
                    + f"{str(frame_idx).zfill(3)}_{str(camera_idx-1)}.npy"
                )
                instance_label_path = (
                    f"{self.save_dir}/{str(file_idx).zfill(3)}/instance_segs/"
                    + f"{str(frame_idx).zfill(3)}_{str(camera_idx-1)}.npy"
                )
                # save mask
                np.save(semantic_label_path, semantic_label)
                np.save(instance_label_path, instance_label)
        

    def save_dynamic_mask(self, frame, file_idx, frame_idx):
        """Parse and save the segmentation data.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        for img in frame.images:
            # dynamic_mask
            img_path = (
                f"{self.save_dir}/{str(file_idx).zfill(3)}/images/"
                + f"{str(frame_idx).zfill(3)}_{str(img.name - 1)}.jpg"
            )
            img_shape = np.array(Image.open(img_path))
            dynamic_mask = np.zeros_like(img_shape, dtype=np.float32)[..., 0]

            filter_available = any(
                [label.num_top_lidar_points_in_box > 0 for label in frame.laser_labels]
            )
            calibration = next(
                cc for cc in frame.context.camera_calibrations if cc.name == img.name
            )
            for label in frame.laser_labels:
                # camera_synced_box is not available for the data with flow.
                # box = label.camera_synced_box
                box = label.box
                meta = label.metadata
                speed = np.linalg.norm([meta.speed_x, meta.speed_y])
                if not box.ByteSize():
                    continue  # Filter out labels that do not have a camera_synced_box.
                if (filter_available and not label.num_top_lidar_points_in_box) or (
                    not filter_available and not label.num_lidar_points_in_box
                ):
                    continue  # Filter out likely occluded objects.

                # Retrieve upright 3D box corners.
                box_coords = np.array(
                    [
                        [
                            box.center_x,
                            box.center_y,
                            box.center_z,
                            box.length,
                            box.width,
                            box.height,
                            box.heading,
                        ]
                    ]
                )
                corners = box_utils.get_upright_3d_box_corners(box_coords)[
                    0
                ].numpy()  # [8, 3]

                # Project box corners from vehicle coordinates onto the image.
                projected_corners = project_vehicle_to_image(
                    frame.pose, calibration, corners
                )
                u, v, ok = projected_corners.transpose()
                ok = ok.astype(bool)

                # Skip object if any corner projection failed. Note that this is very
                # strict and can lead to exclusion of some partially visible objects.
                if not all(ok):
                    continue
                u = u[ok]
                v = v[ok]

                # Clip box to image bounds.
                u = np.clip(u, 0, calibration.width)
                v = np.clip(v, 0, calibration.height)

                if u.max() - u.min() == 0 or v.max() - v.min() == 0:
                    continue

                # Draw projected 2D box onto the image.
                xy = (u.min(), v.min())
                width = u.max() - u.min()
                height = v.max() - v.min()
                # max pooling
                dynamic_mask[
                    int(xy[1]) : int(xy[1] + height),
                    int(xy[0]) : int(xy[0] + width),
                ] = np.maximum(
                    dynamic_mask[
                        int(xy[1]) : int(xy[1] + height),
                        int(xy[0]) : int(xy[0] + width),
                    ],
                    speed,
                )
            # thresholding, use 1.0 m/s to determine whether the pixel is moving
            dynamic_mask = np.clip((dynamic_mask > 1.0) * 255, 0, 255).astype(np.uint8)
            dynamic_mask = Image.fromarray(dynamic_mask, "L")
            dynamic_mask_path = (
                f"{self.save_dir}/{str(file_idx).zfill(3)}/dynamic_masks/"
                + f"{str(frame_idx).zfill(3)}_{str(img.name - 1)}.png"
            )
            dynamic_mask.save(dynamic_mask_path)

    def create_folder(self):
        """Create folder for data preprocessing."""
        if self.process_id_list is None:
            id_list = range(len(self))
        else:
            id_list = self.process_id_list
        for i in id_list:
            os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/images", exist_ok=True)
            os.makedirs(
                f"{self.save_dir}/{str(i).zfill(3)}/ego_pose",
                exist_ok=True,
            )
            os.makedirs(
                f"{self.save_dir}/{str(i).zfill(3)}/extrinsics",
                exist_ok=True,
            )
            os.makedirs(
                f"{self.save_dir}/{str(i).zfill(3)}/intrinsics",
                exist_ok=True,
            )
            os.makedirs(
                f"{self.save_dir}/{str(i).zfill(3)}/sky_masks",
                exist_ok=True,
            )
            if "lidar" in self.process_keys:
                os.makedirs(
                    f"{self.save_dir}/{str(i).zfill(3)}/lidar",
                    exist_ok=True,
                )
            if "dynamic_masks" in self.process_keys:
                os.makedirs(
                    f"{self.save_dir}/{str(i).zfill(3)}/dynamic_masks",
                    exist_ok=True,
                )
            if "panoptic_segs" in self.process_keys:
                os.makedirs(
                    f"{self.save_dir}/{str(i).zfill(3)}/semantic_segs",
                    exist_ok=True,
                )
                os.makedirs(
                    f"{self.save_dir}/{str(i).zfill(3)}/instance_segs",
                    exist_ok=True,
                )
