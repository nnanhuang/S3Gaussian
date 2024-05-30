import torch
import numpy as np
import PIL
import torch.nn.functional as F
import torch.nn as nn
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

# RGB colors used to visualize each semantic segmentation class.
SEGMENTATION_COLOR_MAP = dict(
    TYPE_UNDEFINED=[0, 0, 0],
    TYPE_EGO_VEHICLE=[102, 102, 102],
    TYPE_CAR=[0, 0, 142],
    TYPE_TRUCK=[0, 0, 70],
    TYPE_BUS=[0, 60, 100],
    TYPE_OTHER_LARGE_VEHICLE=[61, 133, 198],
    TYPE_BICYCLE=[119, 11, 32],
    TYPE_MOTORCYCLE=[0, 0, 230],
    TYPE_TRAILER=[111, 168, 220],
    TYPE_PEDESTRIAN=[220, 20, 60],
    TYPE_CYCLIST=[255, 0, 0],
    TYPE_MOTORCYCLIST=[180, 0, 0],
    TYPE_BIRD=[127, 96, 0],
    TYPE_GROUND_ANIMAL=[91, 15, 0],
    TYPE_CONSTRUCTION_CONE_POLE=[230, 145, 56],
    TYPE_POLE=[153, 153, 153],
    TYPE_PEDESTRIAN_OBJECT=[234, 153, 153],
    TYPE_SIGN=[246, 178, 107],
    TYPE_TRAFFIC_LIGHT=[250, 170, 30],
    TYPE_BUILDING=[70, 70, 70],
    TYPE_ROAD=[128, 64, 128],
    TYPE_LANE_MARKER=[234, 209, 220],
    TYPE_ROAD_MARKER=[217, 210, 233],
    TYPE_SIDEWALK=[244, 35, 232],
    TYPE_VEGETATION=[107, 142, 35],
    TYPE_SKY=[70, 130, 180],
    TYPE_GROUND=[102, 102, 102],
    TYPE_DYNAMIC=[102, 102, 102],
    TYPE_STATIC=[102, 102, 102],
)

def _generate_color_map(
    color_map_dict: Optional[
        Mapping[int, Sequence[int]]] = None
) -> np.ndarray:
  """Generates a mapping from segmentation classes (rows) to colors (cols).

  Args:
    color_map_dict: An optional dict mapping from semantic classes to colors. If
      None, the default colors in SEGMENTATION_COLOR_MAP will be used.
  Returns:
    A np array of shape [max_class_id + 1, 3], where each row encodes the color
      for the corresponding class id.
  """
  if color_map_dict is None:
    color_map_dict = SEGMENTATION_COLOR_MAP
  classes = list(color_map_dict.keys())
  colors = list(color_map_dict.values())
  color_map = np.zeros([#np.amax(classes) + 1
                        len(classes)
                        , 3], dtype=np.uint8)
  for idx, color in enumerate(colors):
      color_map[idx] = color
  #color_map[classes] = colors
  return color_map

DEFAULT_COLOR_MAP = _generate_color_map()

def get_panoptic_id(semantic_id, instance_id, semantic_interval=1000):
    if isinstance(semantic_id, np.ndarray):
        semantic_id = torch.from_numpy(semantic_id)
        instance_id = torch.from_numpy(instance_id)
    elif isinstance(semantic_id, PIL.Image.Image):
        semantic_id = torch.from_numpy(np.array(semantic_id))
        instance_id = torch.from_numpy(np.array(instance_id))
    elif isinstance(semantic_id, torch.Tensor):
        pass
    else:
        raise ValueError("semantic_id type is not supported!")

    return semantic_id * semantic_interval + instance_id

def get_panoptic_encoding(semantic_id, instance_id, ):
    # 将 semantic-id 和 instance-id 编码成 panoptic one-hot编码
    panoptic_id = get_panoptic_id(semantic_id, instance_id)
    unique_panoptic_classes = panoptic_id.unique()
    num_panoptic_classes = unique_panoptic_classes.shape[0]
    # construct id map dict: panoptic_id -> num_class_idx
    id_to_idx_dict = {}
    for i in range(num_panoptic_classes):
        id_to_idx_dict[unique_panoptic_classes[i]] = i

    # convert to one-hot encoding
    panoptic_encoding = torch.zeros((num_panoptic_classes, ), dtype=torch.float32)


def feat_encode(obj_id, id_to_idx, gt_label_embedding: nn.Embedding = None, output_both=False, only_idx=False):
    """ 根据 obj_id 和 id_to_idx_dict 编码成 one-hot """

    map_ids = torch.zeros_like(obj_id) #obj_id.clone()
    # 将 gt-obj-id 替换成 global-obj-idx ，然后转成 one-hot
    for key, value in id_to_idx.items():
        map_ids[obj_id == key] = value

    # query embedding
    if gt_label_embedding is not None:
        gt_label = gt_label_embedding(map_ids.flatten().long())
    else:
        gt_label = None

    if output_both:
        return map_ids, gt_label
    else:
        if only_idx:
            return map_ids.long().flatten()
        else:
            return gt_label

        

