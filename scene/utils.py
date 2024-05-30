import open3d as o3d
import numpy as np
import torch

def visualize_octree_space(pts, node_info_list, depth_colors, offscreen=False):
    if pts is not None:
        if type(pts)==torch.Tensor:
            points = pts.cpu().numpy()
        elif type(pts)==torch.nn.Parameter:
            points = pts.detach().cpu().numpy()
        else:
            points = pts
        N = points.shape[0]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))
        pcd_list = [pcd]
    else:
        pcd_list = []
    # 为每个深度下叶子节点 利用open3d 可视化 占用网格
    lvls_list = []
    for node_info in node_info_list:
        lvl_voxel_list = []
        if len(node_info['node_list']) > 0:
            for node in node_info['node_list']:
                # 设定当前level的颜色
                color = depth_colors[node['depth']]
                # 可视化包围盒
                if type(node['bounds'])==torch.Tensor:
                    assert node['bounds'].shape[0]==2, 'bounds shouuld be 2x3'
                    bounds = node['bounds'].cpu().numpy()
                elif type(node['bounds'])==np.ndarray:
                    assert node['bounds'].shape[0]==2, 'bounds shouuld be 2x3'
                    bounds = node['bounds']
                else:
                    raise NotImplementedError
                assert (bounds[0]<bounds[1]).all(), 'bounds[0]<bounds[1] failed'
                bbox = o3d.geometry.AxisAlignedBoundingBox(np.array(bounds[0]), np.array(bounds[1]))
                bbox.color = color
                lvl_voxel_list.append(bbox)
            lvls_list.append(lvl_voxel_list)
    if not offscreen:
        o3d.visualization.draw_geometries([item for sublist in lvls_list for item in sublist]+pcd_list)
    num_boxes = sum([len(lvl_voxel_list) for lvl_voxel_list in lvls_list])
    print('total boxes num: {}'.format(num_boxes))
    ## debug
    #o3d.visualization.draw_geometries([bbox, pcd])

    # return per-level boxes
    #return lvls_list

    # return all visible boxes + pts
    return [item for sublist in lvls_list for item in sublist]+pcd_list



def check_num_points(N, node_info_list):
    print('check points number == {}'.format(N))
    val_num_pts_dict = {}
    for lvl, depth_i_node_list in enumerate(node_info_list):
        cur_lvl_pts_list = []
        if depth_i_node_list['count'] > 0:
            for node in depth_i_node_list['node_list']:
                cur_lvl_pts_list.append(len(node['points']))
        val_num_pts_dict[lvl] = cur_lvl_pts_list
        print('level {} leaf-node num: {}'.format(lvl, depth_i_node_list['count']))
        print('level {} points num: {}'.format(lvl, sum(cur_lvl_pts_list)))
    num_pts_perlvl = [sum(val_num_pts_dict[lvl]) for lvl in val_num_pts_dict.keys()]
    print('total points num: {}'.format(sum(num_pts_perlvl)))
    if sum(num_pts_perlvl) == N:
        print('check points number passed')
    else:
        print('check points number failed !!!!!!!!')

def collect_nodeinfo_perdepth(node, node_info_list, current_depth):
    # 按照 depth 在字典中 存储 node 信息
    if node is not None:
        node_info = {
            'depth': current_depth,
            'child_index': node_info_list[current_depth]['count'],
            'bounds': node.bounds
        }
        node_info_list[current_depth]['count'] += 1
        node_info_list[current_depth]['node_list'].append(node_info)

        for child in node.children:
            collect_nodeinfo_perdepth(child, node_info_list, current_depth + 1)
def collect_leafnodeinfo_perdepth(node, node_info_list, current_depth):
    # 按照 depth 在字典中 存储 叶子节点的信息
    if node is not None:
        if node.is_leaf():
            node_info = {
                'depth': current_depth,
                #'child_index': node_info_list[current_depth]['count'],
                'bounds': node.bounds,
                'points': node.points
            }
            node_info_list[current_depth]['leaf_count'] += 1
            node_info_list[current_depth]['node_list'].append(node_info)
        else:
            for child in node.children:
                collect_leafnodeinfo_perdepth(child, node_info_list, current_depth + 1)