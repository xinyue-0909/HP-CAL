import shutil
import time
import pickle

import numpy as np

from base_op import *
from gcn import *
from semantic3d_dataset_sampling import *
from fps_gcn_cuda import GCN_FPS_sampling
import os
from numpy.linalg import svd
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform


def compute_region_uncertainty(pixel_uncertainty, pixel_class, class_num, sampler_args):
    """
        计算superpoint的不确定度
        pixel_uncertainty: 每个point的不确定度
        pexel_class: 每个point的类别
    """
    if "mean" in sampler_args:
        return np.mean(pixel_uncertainty)
    elif "sum_weight" in sampler_args:
        # pixel_weights = np.exp(weights_percentage(list_class=prob_class[point_ids], class_num=class_num) - 1)
        pixel_weights = weights_percentage(list_class=pixel_class, class_num=class_num)
        return np.sum(np.multiply(pixel_weights, pixel_uncertainty))
    elif "WetSU" in sampler_args:
        # majority class
        d_label, _ = _dominant_label(pixel_class)
        # superpoint中的 majority class 中的每个点为1，其余class为0
        equal_id = np.where(pixel_class == d_label, 1.0, 0.0)
        equal_num = np.sum(equal_id)
        pixel_num = len(pixel_uncertainty)
        # region_uncertainty = np.sum(np.multiply(pixel_uncertainty, equal_id) * equal_num / pixel_num) - np.max(pixel_uncertainty)*(pixel_num-equal_num)
        # region_uncertainty = np.sum(np.multiply(pixel_uncertainty, equal_id)) - np.max(pixel_uncertainty)*(pixel_num-equal_num)

        # superpoint的uncertainty  dominant_label的点的不确定度之和 —  非dominant_label的点的不确定度之和
        region_uncertainty = np.sum(np.multiply(pixel_uncertainty, equal_id)) - np.sum(np.multiply(pixel_uncertainty, 1 - equal_id))
        return region_uncertainty

def compute_point_uncertainty(prob_logits, sampler_args):
    """
        计算点的不确定度
        prob_logits: 预测这个点的所有class的概率
    """
    if "lc" in sampler_args:
        """
        least confidence
        An analysis of active learning strategies for sequence labeling tasks
        """
        prob_max = np.max(prob_logits, axis=-1)  # [batch_size * point_num]
        point_uncertainty = 1.0 - prob_max  # [batch_size * point_num]
    elif "entropy" in sampler_args:
        """
        entropy
        An analysis of active learning strategies for sequence labeling tasks
        点的预测熵
        """
        point_uncertainty = compute_entropy(prob_logits)  # [batch_size * point_num]
    elif "sb" in sampler_args:
        """second best / best
            次大概率/最大概率作为点的不确定度
        """
        prob_sorted = np.sort(prob_logits, axis=-1)  # 升序  [batch_size * point_num, class_num]
        point_uncertainty = prob_sorted[:, -2] / prob_sorted[:, -1]  # [batch_size * point_num]

    return point_uncertainty

def farthest_superpoint_sample(superpoint_list, superpoint_centroid_list, sample_number, trigger_idx):
    """
    在DR最后一步,从 diversity space中利用FPS算法对superpoint采样
    Input:
        superpoint_list: pointcloud data, [sp_num, each_sp_p_num, 3]
        superpoint_centroid_list: pointcloud centroid xyz [sp_num, 3]
        sample_number: number of samples
    Return:
        centroids: sampled superpoint index, [sample_number]
    """
    sp_num = len(superpoint_list)
    align_superpoint_list = []
    tree_list = []
    for i in range(sp_num):
        # 中心对齐
        align_superpoint = superpoint_list[i] - superpoint_centroid_list[i]
        align_superpoint_list.append(align_superpoint)
        tree_list.append(KDTree(align_superpoint))


    centroids = np.zeros([sample_number], dtype=np.int32)
    centroids[0] = trigger_idx

    distance = np.ones([sp_num]) * 1e10

    for i in range(sample_number - 1):

        current_superpoint_center = superpoint_centroid_list[centroids[i]]
        euclidean_dist = np.sum((superpoint_centroid_list - current_superpoint_center) ** 2, axis=-1)

        cd_dist = chamfer_distance(align_superpoint_list, tree_list, centroids[i])


        dist = np.add(euclidean_dist, cd_dist)


        mask = dist < distance
        distance[mask] = dist[mask]

        centroids[i + 1] = np.argmax(distance)
    return centroids

def weights_softmax(list_class, class_num):
    class_distribution = np.zeros([class_num], dtype=np.float128)
    for cls in list_class:
        class_distribution[cls] = class_distribution[cls] + 1
    class_distribution = np.true_divide(np.exp(class_distribution), np.sum(np.exp(class_distribution)))  # softmax
    weights = []
    for cls in list_class:
        weights.append(class_distribution[cls])
    return np.asarray(weights)

def weights_percentage(list_class, class_num):
    class_distribution = np.zeros([class_num])
    for c in list_class:
        class_distribution[c] = class_distribution[c] + 1
    class_distribution = class_distribution / len(list_class)
    weights = []
    for c in list_class:
        weights.append(class_distribution[c])
    return np.asarray(weights)

def _dominant_label(ary):
    ssdr = np.zeros([np.max(ary) + 1], dtype=np.int32)
    for a in ary:
        ssdr[a] = ssdr[a] + 1
    return np.argmax(ssdr), np.amax(ssdr) / len(ary)

def _dominant_2(ary):
    ary = np.array(ary)
    ssdr = np.zeros([np.max(ary) + 1], dtype=np.int32)
    for a in ary:
        ssdr[a] = ssdr[a] + 1
    label = np.argmax(ssdr)
    ids = np.where(ary == label)
    return label, ids

def _get_sub_region_from_superpoint(prob_class, point_inds):
    #将超点区域范围内的点根据类别进行子区域划分
    #当前超点对应的预测类别选取最大值
    #prob_class 预测类别
    #point_inds
    ssdr = [[] for _ in range(np.max(prob_class[point_inds]) + 1)]#选取当前这些点中类别最大值的点
    for pid in point_inds:
        cls = prob_class[pid]
        ssdr[cls].append(pid)
    return ssdr

def calculate_similarity(vector1, vector2):
    # 计算两个向量的单位向量
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)
    # 计算两个单位向量的点积
    dot_product = np.dot(unit_vector1, unit_vector2)
    # 计算夹角的余弦值
    cosine_similarity = dot_product
    return cosine_similarity

param_13 = 1.0 / 3.0
param_16116 = 16.0 / 116.0
 
Xn = 0.950456
Yn = 1.0
Zn = 1.088754

def RGB2XYZ(r, g, b):
    x = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z = 0.019334 * r + 0.119193 * g + 0.950227 * b
    return x, y, z
 
 
def XYZ2Lab(x, y, z):
    x /=  255 * Xn
    y /= 255 * Yn
    z /= 255 * Zn
    if y > 0.008856:
        fy = pow(y, param_13)
        l = 116.0 * fy - 16.0
    else:
        fy = 7.787 * y + param_16116
        l = 903.3 * fy
    if l < 0:
        l = 0.0
    if x>0.008856:
        fx = pow(x,param_13)
    else:
        fx = 7.787 * x + param_16116
    if z>0.008856:
        fz = pow(z,param_13)
    else:
        fz = 7.787 * z + param_16116
    a = 500.0*(fx-fy)
    b = 200.0*(fy-fz)
    return [round(l,2), round(a,2), round(b,2)]

def rgb2lab(rgb):
    # r = rgb[0] / 255.0  # rgb range: 0 ~ 1
    # g = rgb[1] / 255.0
    # b = rgb[2] / 255.0
    r = rgb[0]   # rgb range: 0 ~ 1
    g = rgb[1] 
    b = rgb[2] 
 
    # gamma 2.2
    if r > 0.04045:
        r = pow((r + 0.055) / 1.055, 2.4)
    else:
        r = r / 12.92
 
    if g > 0.04045:
        g = pow((g + 0.055) / 1.055, 2.4)
    else:
        g = g / 12.92
 
    if b > 0.04045:
        b = pow((b + 0.055) / 1.055, 2.4)
    else:
        b = b / 12.92
 
    # sRGB
    X = r * 0.436052025 + g * 0.385081593 + b * 0.143087414
    Y = r * 0.222491598 + g * 0.716886060 + b * 0.060621486
    Z = r * 0.013929122 + g * 0.097097002 + b * 0.714185470
 
    # XYZ range: 0~100
    X = X * 100.000
    Y = Y * 100.000
    Z = Z * 100.000
 
    # Reference White Point
 
    ref_X = 96.4221
    ref_Y = 100.000
    ref_Z = 82.5211
 
    X = X / ref_X
    Y = Y / ref_Y
    Z = Z / ref_Z
 
    # Lab
    if X > 0.008856:
        X = pow(X, 1 / 3.000)
    else:
        X = (7.787 * X) + (16 / 116.000)
 
    if Y > 0.008856:
        Y = pow(Y, 1 / 3.000)
    else:
        Y = (7.787 * Y) + (16 / 116.000)
 
    if Z > 0.008856:
        Z = pow(Z, 1 / 3.000)
    else:
        Z = (7.787 * Z) + (16 / 116.000)
 
    Lab_L = round((116.000 * Y) - 16.000, 2)
    Lab_a = round(500.000 * (X - Y), 2)
    Lab_b = round(200.000 * (Y - Z), 2)
 
    return Lab_L, Lab_a, Lab_b
 

def oracle_labeling(superpoint_inds, components, input_gt, pseudo_gt, cloud_name, w, sampler_args, prob_class, threshold, budget, min_size, total_obj, ply_data):
    '''
    对于选择出来的超点idx，先看是否还有剩余标注预算，有则根据取出这个sp的所有p，从.ply中读取得到这些点的真实label，根据dominant原则，将pseudo_gt中对应点act置1,label置大多数label
    '''
    sp_neighbour_path = '/data/home/dxy/dataset/Semantic3D/0.012/spNeighbour'#邻居超点路径
    sp_neighbour_file = os.path.join(sp_neighbour_path, cloud_name + '_spNeighbour.pkl')#邻居超点文件
    with open(sp_neighbour_file, "rb") as f:
        sp_neighbour_obj = pickle.load(f)
    sub_xyz = np.vstack((ply_data['x'], ply_data['y'], ply_data['z'])).T#邻居超点xyz值
    sub_rgb = np.vstack((ply_data['red'], ply_data['green'], ply_data['blue'])).T#邻居超点颜色值
    '''# selected_top_sp_witth_neighbours = 
    {
        center_sp_idx: [valid_neighbours_sp_idx]
        10:[1,2,3...],
        100:[101,102,103...],
        972:[122,354,1213,35321...],
        ...
    }
    '''
    selected_top_sp_with_neighbours = {}
    # print("superpoint_inds length:",len(superpoint_inds))
    used_superpoint_inds = []
    print("sampler_args: ",sampler_args)

    if "NAIL" in sampler_args:
        if "divide" in sampler_args:
            ignore_merge_vector_sp_count = 0
            ignore_merge_rgb_sp_count = 0
            ignore_purity_sp_count = 0
            for superpoint_idx in superpoint_inds:#遍历超点
                if budget["click"] > 0:#如果还有标注预算
                    point_inds = components[superpoint_idx]#属于当前超点区域覆盖范围内的点的索引
                    new_center_point_inds = []
                    if len(point_inds) >= min_size:
                        #对于中心超点的处理
                        ignore = True   # 这个ingore有什么用？
                        used_superpoint_inds.append(superpoint_idx)
                        budget["click"] = budget["click"] - 1  # click + 1 标注预算-1
                        # 直接用ground_truth来查明其中的do_label和纯度--------------------------实际场景没有ground_truth应该怎么办?:人为标注
                        do_label, purity = _dominant_label(input_gt[point_inds])#真值主导类别和纯度
                        sp_pred_do_label, _ = _dominant_label(prob_class[point_inds])#预测主导类别
                        if purity < threshold:
                            # pseudo_gt[0][point_inds] = 0.0
                            # 对中心超点划分
                            ignore_purity_sp_count += 1
                            sub_region_list = _get_sub_region_from_superpoint(prob_class=prob_class, point_inds=point_inds)
                            for sub_id in range(len(sub_region_list)):
                                sub_region_pids = sub_region_list[sub_id]
                                if len(sub_region_pids) > min_size:
                                    # 对于这个sp中，每一种预测类别的点，对其根据预测类别判断真实纯度，若纯度达标，对这种预测类别的点进行标注
                                    sub_do_label, sub_do_rate = _dominant_label(input_gt[sub_region_pids])
                                    if sub_do_rate >= threshold and do_label==sub_do_label: 
                                        new_center_point_inds += sub_region_pids 
                                        pseudo_gt[0][sub_region_pids] = 1.0
                                        pseudo_gt[1][sub_region_pids] = sub_do_label * 1.0
                                        total_obj["selected_class_list"].append(sub_do_label)

                                        w["sub_num"] = w["sub_num"] + 1
                                        w["sub_p_num"] = w["sub_p_num"] + len(sub_region_pids)
                                        ignore = False
                        else:
                            pseudo_gt[0][point_inds] = 1.0
                            pseudo_gt[1][point_inds] = do_label * 1.0
                            total_obj["selected_class_list"].append(do_label)

                            w["sp_num"] = w["sp_num"] + 1
                            w["p_num"] = w["p_num"] + len(point_inds)
                            ignore = False

                        # 先划分：将每个中心超点和相邻超点的主导类别筛选出来。
                        # if 1<0:     # no merge
                        if 0<1:   #  merge

                        # 对其邻居节点进行合并标注
                            #计算中心sp的法向量
                            if new_center_point_inds != []:
                                center_sp_points_xyz = sub_xyz[new_center_point_inds]
                                center_sp_points_mean = center_sp_points_xyz.mean(axis=0)
                                center_sp_points_relative = center_sp_points_xyz - center_sp_points_mean

                                center_sp_points_rgb = sub_rgb[new_center_point_inds]
                                center_sp_mean_rgb = np.mean(center_sp_points_rgb, axis=0)
                                center_sp_mean_lab = rgb2lab(center_sp_mean_rgb)#rgb->lab
                                center_sp_weight_lab = center_sp_mean_lab[0] * 0.5 +center_sp_mean_lab[1] * 1 +center_sp_mean_lab[2] * 1#修改lab特征各分量权重占比
                            else:
                                center_sp_points_xyz = sub_xyz[point_inds]
                                center_sp_points_mean = center_sp_points_xyz.mean(axis=0)
                                center_sp_points_relative = center_sp_points_xyz - center_sp_points_mean

                                center_sp_points_rgb = sub_rgb[point_inds]
                                center_sp_mean_rgb = np.mean(center_sp_points_rgb, axis=0)
                                center_sp_mean_lab = rgb2lab(center_sp_mean_rgb)#rgb->lab
                                center_sp_weight_lab = center_sp_mean_lab[0] * 0.5 +center_sp_mean_lab[1] * 1 +center_sp_mean_lab[2] * 1#修改lab特征各分量权重占比
                            
                            U, S, Vt = svd(center_sp_points_relative)#计算中心超点的法向量
                            center_sp_normal_vector = Vt[-1]
                            if center_sp_normal_vector[2] < 0:
                                center_sp_normal_vector = -center_sp_normal_vector
                                
                            # 处理邻居超点
                            # 计算邻居节点的法向量
                            sp_neighbours_idx = sp_neighbour_obj[superpoint_idx]
                            for sp_neighbour_idx in sp_neighbours_idx:
                                # 在此处还应该判断是否 该sp_neighbour_idx 已经被使用过
                                # 检查是否使用过，是否位于同一平面，是则"合并"进行标注
                                neighbour_point_inds = components[sp_neighbour_idx]
                                # 不合并特别大的sp,因为本身选的时候也没有选这么大的
                                if len(neighbour_point_inds) > 1000:
                                    continue
                                
                                #邻居超点纯度计算
                                neighbour_do_label, neighbour_purity = _dominant_label(input_gt[neighbour_point_inds])#计算邻居节点纯度
                                if(neighbour_do_label != do_label or purity - neighbour_purity>0.05):
                                    continue
                                #邻居超点子区域划分
                                new_neighbour_point_inds = []
                                if neighbour_purity < threshold:
                                    pseudo_gt[0][neighbour_point_inds] = 0.0
                                    #对邻居超点进行子区域划分
                                    #sub_region_list是类别值对应的列表
                                    sub_region_list = _get_sub_region_from_superpoint(prob_class=prob_class, point_inds=neighbour_point_inds)
                                    #遍历划分子区域，保留提纯后的子区域（最好还是合并下，这样统一计算，节省计算量和时间）
                                    for sub_id in range(len(sub_region_list)):
                                        sub_region_pids = sub_region_list[sub_id]
                                        if len(sub_region_pids) > min_size:
                                            #计算子区域纯度
                                            #保留大于阈值的区域，并合并在一起
                                            sub_do_label, sub_do_rate = _dominant_label(input_gt[sub_region_pids])
                                            if sub_do_rate >= threshold and do_label==sub_do_label:  
                                                new_neighbour_point_inds += sub_region_pids 
                                            
                                if new_neighbour_point_inds!=[]:
                                    neighbour_sp_points_xyz = sub_xyz[new_neighbour_point_inds]
                                else:
                                    neighbour_sp_points_xyz = sub_xyz[neighbour_point_inds]
                                neighbour_sp_points_mean = neighbour_sp_points_xyz.mean(axis=0)
                                neighbour_points_relative = neighbour_sp_points_xyz - neighbour_sp_points_mean
                                U, S, Vt = svd(neighbour_points_relative)
                                # 相邻sp的法向量
                                neighbour_sp_normal_vector = Vt[-1]
                                if neighbour_sp_normal_vector[2] < 0:
                                    neighbour_sp_normal_vector = -neighbour_sp_normal_vector
                                #计算中心超点和邻居超点的法向量相似度，判断二者是否处于同一平面
                                vector_cosine_similarity =  calculate_similarity(center_sp_normal_vector, neighbour_sp_normal_vector)
                                if(abs(vector_cosine_similarity)<0.97):#小于法向量相似度阈值的该邻居超点被丢弃
                                        ignore_merge_vector_sp_count+=1
                                        continue
                                
                                # 邻居sp的颜色衡量度
                                if new_neighbour_point_inds!=[]:
                                    neighbour_sp_points_rgb = sub_rgb[new_neighbour_point_inds]
                                else:
                                    neighbour_sp_points_rgb = sub_rgb[neighbour_point_inds]
                                neighbour_sp_mean_rgb = np.mean(neighbour_sp_points_rgb, axis=0)
                                neighbour_sp_mean_lab = rgb2lab(neighbour_sp_mean_rgb)
                                neighbour_sp_weight_lab = neighbour_sp_mean_lab[0]*0.5+neighbour_sp_mean_lab[1]*1+neighbour_sp_mean_lab[2]*1
                                    
                                if (neighbour_sp_weight_lab<center_sp_weight_lab*0.9 or neighbour_sp_weight_lab>center_sp_weight_lab*1.1):
                                    ignore_merge_rgb_sp_count+=1
                                    continue
                                if new_neighbour_point_inds!=[]:
                                        
                                    pseudo_gt[0][new_neighbour_point_inds] = 1.0
                                    pseudo_gt[1][new_neighbour_point_inds] = neighbour_do_label * 1.0
                                    total_obj["selected_class_list"].append(neighbour_do_label)
                                            
                                    used_superpoint_inds.append(sp_neighbour_idx)  #极小的概率会标错，待试验确定标记错误的影响 // 不对，这里不会标错，因为上面使用的input_gt获得的neigh_do_label和purity
                                    w["sp_num"] = w["sp_num"] + 1
                                    w["p_num"] = w["p_num"] + len(new_neighbour_point_inds)
                                else:
                                    pseudo_gt[0][neighbour_point_inds] = 1.0
                                    pseudo_gt[1][neighbour_point_inds] = neighbour_do_label * 1.0
                                    total_obj["selected_class_list"].append(neighbour_do_label)
                                            
                                    used_superpoint_inds.append(sp_neighbour_idx)  #极小的概率会标错，待试验确定标记错误的影响 // 不对，这里不会标错，因为上面使用的input_gt获得的neigh_do_label和purity
                                    w["sp_num"] = w["sp_num"] + 1
                                    w["p_num"] = w["p_num"] + len(neighbour_point_inds)    

                else:
                    break
        elif "fix_new" in sampler_args:
            ignore_merge_vector_sp_count = 0
            ignore_merge_rgb_sp_count = 0
            ignore_purity_sp_count = 0
            for superpoint_idx in superpoint_inds:#遍历超点
                if budget["click"] > 0:#如果还有标注预算
                    point_inds = components[superpoint_idx]#属于当前超点区域覆盖范围内的点的索引
                    if len(point_inds) >= min_size:
                        ignore = True   # 这个ingore有什么用？

                        used_superpoint_inds.append(superpoint_idx)
                        budget["click"] = budget["click"] - 1  # click + 1 标注预算-1
                        # 直接用ground_truth来查明其中的do_label和纯度--------------------------实际场景没有ground_truth应该怎么办?:人为标注
                        do_label, purity = _dominant_label(input_gt[point_inds])#真值主导类别和纯度
                        sp_pred_do_label, _ = _dominant_label(prob_class[point_inds])#预测主导类别
                        # if purity >= threshold:#如果超点纯度大于设定阈值
                        # if 0<1 :
                        pseudo_gt[0][point_inds] = 1.0
                        pseudo_gt[1][point_inds] = do_label * 1.0
                        total_obj["selected_class_list"].append(do_label)

                        w["sp_num"] = w["sp_num"] + 1
                        w["p_num"] = w["p_num"] + len(point_inds)
                        ignore = False

                        # if 1<0:     # no merge
                        if 0<1:   #  merge

                        # 对其邻居节点进行合并标注
                        #   计算中心sp的法向量
                            center_sp_points_xyz = sub_xyz[point_inds]
                            center_sp_points_mean = center_sp_points_xyz.mean(axis=0)
                            center_sp_points_relative = center_sp_points_xyz - center_sp_points_mean

                            center_sp_points_rgb = sub_rgb[point_inds]
                            center_sp_mean_rgb = np.mean(center_sp_points_rgb, axis=0)
                            center_sp_mean_lab = rgb2lab(center_sp_mean_rgb)#rgb->lab
                            center_sp_weight_lab = (center_sp_mean_lab[0] * 0.5 + center_sp_mean_lab[1] * 1 + center_sp_mean_lab[2] * 1)#修改lab特征各分量权重占比

                            U, S, Vt = svd(center_sp_points_relative)#计算中心超点的法向量
                            center_sp_normal_vector = Vt[-1]
                            if center_sp_normal_vector[2] < 0:
                                center_sp_normal_vector = -center_sp_normal_vector
                            
                            #   计算邻居节点的法向量
                            sp_neighbours_idx = sp_neighbour_obj[superpoint_idx]
                            for sp_neighbour_idx in sp_neighbours_idx:
                                # 在此处还应该判断是否 该sp_neighbour_idx 已经被使用过------------------------------------
                                # continue 
                                # 检查是否使用过，是否位于同一平面，是则"合并"进行标注
                                neighbour_point_inds = components[sp_neighbour_idx]
                                # 不合并特别大的sp,因为本身选的时候也没有选这么大的
                                if len(neighbour_point_inds) > 1000:
                                    continue

                                neighbour_do_label, neighbour_purity = _dominant_label(input_gt[neighbour_point_inds])#计算邻居节点纯度
                                if(neighbour_do_label != do_label or purity - neighbour_purity>0.05):
                                    continue
                                # print("neighbour_sp_idx:",sp_neighbour_idx," do_label:",neighbour_do_label," purity:",neighbour_purity," sp_points_count:",len(neighbour_point_inds))

                                neighbour_sp_points_xyz = sub_xyz[neighbour_point_inds]
                                neighbour_sp_points_mean = neighbour_sp_points_xyz.mean(axis=0)
                                neighbour_points_relative = neighbour_sp_points_xyz - neighbour_sp_points_mean
                                U, S, Vt = svd(neighbour_points_relative)
                                # 相邻sp的法向量
                                neighbour_sp_normal_vector = Vt[-1]
                                if neighbour_sp_normal_vector[2] < 0:
                                    neighbour_sp_normal_vector = -neighbour_sp_normal_vector
                                #计算中心超点和邻居超点的法向量相似度，判断二者是否处于同一平面
                                vector_cosine_similarity =  calculate_similarity(center_sp_normal_vector, neighbour_sp_normal_vector)
                                # print(cloud_name," : ", sp_neighbour_idx," 和 ",superpoint_idx , " 相似度: ",vector_cosine_similarity)
                                if(abs(vector_cosine_similarity)<0.97):#小于法向量相似度阈值的该邻居超点被丢弃
                                    ignore_merge_vector_sp_count+=1
                                    continue
                                
                                # 邻居sp的颜色衡量度
                        
                                neighbour_sp_points_rgb = sub_rgb[neighbour_point_inds]
                                neighbour_sp_mean_rgb = np.mean(neighbour_sp_points_rgb, axis=0)
                                neighbour_sp_mean_lab = rgb2lab(neighbour_sp_mean_rgb)
                                neighbour_sp_weight_lab = (neighbour_sp_mean_lab[0] * 0.5 + neighbour_sp_mean_lab[1] * 1 + neighbour_sp_mean_lab[2] * 1)

                                if (neighbour_sp_weight_lab<center_sp_weight_lab*0.8 or neighbour_sp_weight_lab>center_sp_weight_lab*1.2):
                                    ignore_merge_rgb_sp_count+=1
                                    # print("颜色特征不匹配筛掉")
                                    continue
                                
                                # 合并进行标注
                                # print("合并1个邻居---------------------------------------------------,相似度:",vector_cosine_similarity)
                                pseudo_gt[0][neighbour_point_inds] = 1.0
                                pseudo_gt[1][neighbour_point_inds] = do_label * 1.0
                                used_superpoint_inds.append(sp_neighbour_idx)  #极小的概率会标错，待试验确定标记错误的影响 // 不对，这里不会标错，因为上面使用的input_gt获得的neigh_do_label和purity
                                w["sp_num"] = w["sp_num"] + 1
                                w["p_num"] = w["p_num"] + len(point_inds)
                                if neighbour_purity < threshold:
                                    pseudo_gt[0][neighbour_point_inds] = 0.0
                                    # 对邻居超点划分
                                    ignore_purity_sp_count += 1
                                    sub_region_list = _get_sub_region_from_superpoint(prob_class=prob_class, point_inds=neighbour_point_inds)
                                    for sub_id in range(len(sub_region_list)):
                                        sub_region_pids = sub_region_list[sub_id]
                                        if len(sub_region_pids) > min_size:
                                            # 对于这个sp中，每一种预测类别的点，对其根据预测类别判断真实纯度，若纯度达标，对这种预测类别的点进行标注
                                            sub_do_label, sub_do_rate = _dominant_label(input_gt[sub_region_pids])
                                            if sub_do_rate >= threshold and do_label==sub_do_label:                                 # 
                                                pseudo_gt[0][sub_region_pids] = 1.0
                                                pseudo_gt[1][sub_region_pids] = sub_do_label * 1.0
                                                total_obj["selected_class_list"].append(sub_do_label)

                                                w["sub_num"] = w["sub_num"] + 1
                                                w["sub_p_num"] = w["sub_p_num"] + len(sub_region_pids)
                                                ignore = False

                        if purity < threshold:
                            pseudo_gt[0][point_inds] = 0.0
                            # 对中心超点划分
                            ignore_purity_sp_count += 1
                            sub_region_list = _get_sub_region_from_superpoint(prob_class=prob_class, point_inds=point_inds)
                            for sub_id in range(len(sub_region_list)):
                                sub_region_pids = sub_region_list[sub_id]
                                if len(sub_region_pids) > min_size:
                                    # 对于这个sp中，每一种预测类别的点，对其根据预测类别判断真实纯度，若纯度达标，对这种预测类别的点进行标注
                                    sub_do_label, sub_do_rate = _dominant_label(input_gt[sub_region_pids])
                                    if sub_do_rate >= threshold and do_label==sub_do_label:                                 # 
                                        pseudo_gt[0][sub_region_pids] = 1.0
                                        pseudo_gt[1][sub_region_pids] = sub_do_label * 1.0
                                        total_obj["selected_class_list"].append(sub_do_label)

                                        w["sub_num"] = w["sub_num"] + 1
                                        w["sub_p_num"] = w["sub_p_num"] + len(sub_region_pids)
                                        ignore = False


                            if not ignore:
                                # 有一个sp纯度不达标，但其子sub_region中有达标的
                                w["split_sp_num"] = w["split_sp_num"] + 1

                        if ignore:
                            # 当前sp由于纯度不达标，不可利用，且其划分成sub_region后也无法没有任何一个可以利用
                            w["ignore_sp_num"] = w["ignore_sp_num"] + 1

                else:
                    break
        elif "change_sim1" in sampler_args:
            ignore_merge_vector_sp_count = 0
            ignore_merge_rgb_sp_count = 0
            ignore_purity_sp_count = 0
            for superpoint_idx in superpoint_inds:#遍历超点
                if budget["click"] > 0:#如果还有标注预算
                    point_inds = components[superpoint_idx]#属于当前超点区域覆盖范围内的点的索引
                    new_center_point_inds = []
                    if len(point_inds) >= min_size:
                        #对于中心超点的处理
                        ignore = True   # 这个ingore有什么用？
                        used_superpoint_inds.append(superpoint_idx)
                        budget["click"] = budget["click"] - 1  # click + 1 标注预算-1
                        # 直接用ground_truth来查明其中的do_label和纯度--------------------------实际场景没有ground_truth应该怎么办?:人为标注
                        do_label, purity = _dominant_label(input_gt[point_inds])#真值主导类别和纯度
                        sp_pred_do_label, _ = _dominant_label(prob_class[point_inds])#预测主导类别
                        if purity < threshold:
                            # pseudo_gt[0][point_inds] = 0.0
                            # 对中心超点划分
                            ignore_purity_sp_count += 1
                            sub_region_list = _get_sub_region_from_superpoint(prob_class=prob_class, point_inds=point_inds)
                            for sub_id in range(len(sub_region_list)):
                                sub_region_pids = sub_region_list[sub_id]
                                if len(sub_region_pids) > min_size:
                                    # 对于这个sp中，每一种预测类别的点，对其根据预测类别判断真实纯度，若纯度达标，对这种预测类别的点进行标注
                                    sub_do_label, sub_do_rate = _dominant_label(input_gt[sub_region_pids])
                                    if sub_do_rate >= threshold and do_label==sub_do_label: 
                                        new_center_point_inds += sub_region_pids 
                                        pseudo_gt[0][sub_region_pids] = 1.0
                                        pseudo_gt[1][sub_region_pids] = sub_do_label * 1.0
                                        total_obj["selected_class_list"].append(sub_do_label)

                                        w["sub_num"] = w["sub_num"] + 1
                                        w["sub_p_num"] = w["sub_p_num"] + len(sub_region_pids)
                                        ignore = False
                        else:
                            pseudo_gt[0][point_inds] = 1.0
                            pseudo_gt[1][point_inds] = do_label * 1.0
                            total_obj["selected_class_list"].append(do_label)

                            w["sp_num"] = w["sp_num"] + 1
                            w["p_num"] = w["p_num"] + len(point_inds)
                            ignore = False

                        # 先划分：将每个中心超点和相邻超点的主导类别筛选出来。
                        # if 1<0:     # no merge
                        if 0<1:   #  merge

                        # 对其邻居节点进行合并标注
                            #计算中心sp的法向量
                            if new_center_point_inds != []:
                                center_sp_points_xyz = sub_xyz[new_center_point_inds]
                                center_sp_points_mean = center_sp_points_xyz.mean(axis=0)
                                center_sp_points_relative = center_sp_points_xyz - center_sp_points_mean

                                center_sp_points_rgb = sub_rgb[new_center_point_inds]
                                center_sp_mean_rgb = np.mean(center_sp_points_rgb, axis=0)
                                center_sp_mean_lab = rgb2lab(center_sp_mean_rgb)#rgb->lab
                                center_sp_weight_lab = (center_sp_mean_lab[0]*0.5,center_sp_mean_lab[1]*1,center_sp_mean_lab[2]*1)#修改lab特征各分量权重占比
                            else:
                                center_sp_points_xyz = sub_xyz[point_inds]
                                center_sp_points_mean = center_sp_points_xyz.mean(axis=0)
                                center_sp_points_relative = center_sp_points_xyz - center_sp_points_mean

                                center_sp_points_rgb = sub_rgb[point_inds]
                                center_sp_mean_rgb = np.mean(center_sp_points_rgb, axis=0)
                                center_sp_mean_lab = rgb2lab(center_sp_mean_rgb)#rgb->lab
                                center_sp_weight_lab = (center_sp_mean_lab[0]*0.5,center_sp_mean_lab[1]*1,center_sp_mean_lab[2]*1)#修改lab特征各分量权重占比
                            
                            U, S, Vt = svd(center_sp_points_relative)#计算中心超点的法向量
                            center_sp_normal_vector = Vt[-1]
                            if center_sp_normal_vector[2] < 0:
                                center_sp_normal_vector = -center_sp_normal_vector
                                
                            
                            #计算邻居节点的法向量
                            sp_neighbours_idx = sp_neighbour_obj[superpoint_idx]
                            for sp_neighbour_idx in sp_neighbours_idx:
                                # 在此处还应该判断是否 该sp_neighbour_idx 已经被使用过
                                # 检查是否使用过，是否位于同一平面，是则"合并"进行标注
                                neighbour_point_inds = components[sp_neighbour_idx]
                                # 不合并特别大的sp,因为本身选的时候也没有选这么大的
                                if len(neighbour_point_inds) > 1000:
                                    continue
                                
                                #邻居超点纯度计算
                                neighbour_do_label, neighbour_purity = _dominant_label(input_gt[neighbour_point_inds])#计算邻居节点纯度
                                if(neighbour_do_label != do_label or purity - neighbour_purity>0.05):
                                    continue
                                #邻居超点子区域划分
                                new_neighbour_point_inds = []
                                if neighbour_purity < threshold:
                                    pseudo_gt[0][neighbour_point_inds] = 0.0
                                    #对邻居超点进行子区域划分
                                    #sub_region_list是类别值对应的列表
                                    sub_region_list = _get_sub_region_from_superpoint(prob_class=prob_class, point_inds=neighbour_point_inds)
                                    #遍历划分子区域，保留提纯后的子区域（最好还是合并下，这样统一计算，节省计算量和时间）
                                    for sub_id in range(len(sub_region_list)):
                                        sub_region_pids = sub_region_list[sub_id]
                                        if len(sub_region_pids) > min_size:
                                            #计算子区域纯度
                                            #保留大于阈值的区域，并合并在一起
                                            sub_do_label, sub_do_rate = _dominant_label(input_gt[sub_region_pids])
                                            if sub_do_rate >= threshold and do_label==sub_do_label:  
                                                new_neighbour_point_inds += sub_region_pids 
                                if new_neighbour_point_inds!=[]:
                                    neighbour_sp_points_xyz = sub_xyz[new_neighbour_point_inds]
                                else:
                                    neighbour_sp_points_xyz = sub_xyz[neighbour_point_inds]
                                neighbour_sp_points_mean = neighbour_sp_points_xyz.mean(axis=0)
                                neighbour_points_relative = neighbour_sp_points_xyz - neighbour_sp_points_mean
                                U, S, Vt = svd(neighbour_points_relative)
                                # 相邻sp的法向量
                                neighbour_sp_normal_vector = Vt[-1]
                                if neighbour_sp_normal_vector[2] < 0:
                                    neighbour_sp_normal_vector = -neighbour_sp_normal_vector
                                #计算中心超点和邻居超点的法向量相似度，判断二者是否处于同一平面
                                vector_cosine_similarity =  calculate_similarity(center_sp_normal_vector, neighbour_sp_normal_vector)
                                if(abs(vector_cosine_similarity)<0.97):#小于法向量相似度阈值的该邻居超点被丢弃
                                        ignore_merge_vector_sp_count+=1
                                        continue
                                
                                # 邻居sp的颜色衡量度
                                if new_neighbour_point_inds!=[]:
                                    neighbour_sp_points_rgb = sub_rgb[new_neighbour_point_inds]
                                else:
                                    neighbour_sp_points_rgb = sub_rgb[neighbour_point_inds]
                                neighbour_sp_mean_rgb = np.mean(neighbour_sp_points_rgb, axis=0)
                                neighbour_sp_mean_lab = rgb2lab(neighbour_sp_mean_rgb)    
                                neighbour_sp_weight_lab = (neighbour_sp_mean_lab[0]*0.5,neighbour_sp_mean_lab[1]*1,neighbour_sp_mean_lab[2]*1)
                                    
                                distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(neighbour_sp_weight_lab, center_sp_weight_lab)))
                                rgb_sim = 1 / (1 + math.exp(-distance*0.5))
                                if (abs(rgb_sim) < 0.94):
                                        ignore_merge_rgb_sp_count+=1
                                        continue
                                if new_neighbour_point_inds!=[]:
                                        
                                    pseudo_gt[0][new_neighbour_point_inds] = 1.0
                                    pseudo_gt[1][new_neighbour_point_inds] = neighbour_do_label * 1.0
                                    total_obj["selected_class_list"].append(neighbour_do_label)
                                            
                                    used_superpoint_inds.append(sp_neighbour_idx)  #极小的概率会标错，待试验确定标记错误的影响 // 不对，这里不会标错，因为上面使用的input_gt获得的neigh_do_label和purity
                                    w["sp_num"] = w["sp_num"] + 1
                                    w["p_num"] = w["p_num"] + len(new_neighbour_point_inds)
                                else:
                                    pseudo_gt[0][neighbour_point_inds] = 1.0
                                    pseudo_gt[1][neighbour_point_inds] = neighbour_do_label * 1.0
                                    total_obj["selected_class_list"].append(neighbour_do_label)
                                            
                                    used_superpoint_inds.append(sp_neighbour_idx)  #极小的概率会标错，待试验确定标记错误的影响 // 不对，这里不会标错，因为上面使用的input_gt获得的neigh_do_label和purity
                                    w["sp_num"] = w["sp_num"] + 1
                                    w["p_num"] = w["p_num"] + len(neighbour_point_inds)    
                else:
                    break
        elif "change_sim2" in sampler_args:
            ignore_merge_vector_sp_count = 0
            ignore_merge_rgb_sp_count = 0
            ignore_merge_vector_and_rgb_sp_count = 0
            ignore_purity_sp_count = 0
            for superpoint_idx in superpoint_inds:#遍历超点
                if budget["click"] > 0:#如果还有标注预算
                    point_inds = components[superpoint_idx]#属于当前超点区域覆盖范围内的点的索引
                    new_center_point_inds = []
                    if len(point_inds) >= min_size:
                        #对于中心超点的处理
                        ignore = True   # 这个ingore有什么用？
                        used_superpoint_inds.append(superpoint_idx)
                        budget["click"] = budget["click"] - 1  # click + 1 标注预算-1
                        # 直接用ground_truth来查明其中的do_label和纯度--------------------------实际场景没有ground_truth应该怎么办?:人为标注
                        do_label, purity = _dominant_label(input_gt[point_inds])#真值主导类别和纯度
                        sp_pred_do_label, _ = _dominant_label(prob_class[point_inds])#预测主导类别
                        if purity < threshold:
                            # pseudo_gt[0][point_inds] = 0.0
                            # 对中心超点划分
                            ignore_purity_sp_count += 1
                            sub_region_list = _get_sub_region_from_superpoint(prob_class=prob_class, point_inds=point_inds)
                            for sub_id in range(len(sub_region_list)):
                                sub_region_pids = sub_region_list[sub_id]
                                if len(sub_region_pids) > min_size:
                                    # 对于这个sp中，每一种预测类别的点，对其根据预测类别判断真实纯度，若纯度达标，对这种预测类别的点进行标注
                                    sub_do_label, sub_do_rate = _dominant_label(input_gt[sub_region_pids])
                                    if sub_do_rate >= threshold and do_label==sub_do_label: 
                                        new_center_point_inds += sub_region_pids 
                                        pseudo_gt[0][sub_region_pids] = 1.0
                                        pseudo_gt[1][sub_region_pids] = sub_do_label * 1.0
                                        total_obj["selected_class_list"].append(sub_do_label)

                                        w["sub_num"] = w["sub_num"] + 1
                                        w["sub_p_num"] = w["sub_p_num"] + len(sub_region_pids)
                                        ignore = False
                        else:
                            pseudo_gt[0][point_inds] = 1.0
                            pseudo_gt[1][point_inds] = do_label * 1.0
                            total_obj["selected_class_list"].append(do_label)

                            w["sp_num"] = w["sp_num"] + 1
                            w["p_num"] = w["p_num"] + len(point_inds)
                            ignore = False

                        # 先划分：将每个中心超点和相邻超点的主导类别筛选出来。
                        # if 1<0:     # no merge
                        if 0<1:   #  merge

                        # 对其邻居节点进行合并标注
                            #计算中心sp的法向量
                            if new_center_point_inds != []:
                                center_sp_points_xyz = sub_xyz[new_center_point_inds]
                                center_sp_points_mean = center_sp_points_xyz.mean(axis=0)
                                center_sp_points_relative = center_sp_points_xyz - center_sp_points_mean

                                center_sp_points_rgb = sub_rgb[new_center_point_inds]
                                center_sp_mean_rgb = np.mean(center_sp_points_rgb, axis=0)
                                center_sp_mean_lab = rgb2lab(center_sp_mean_rgb)#rgb->lab
                                center_sp_weight_lab = (center_sp_mean_lab[0]*0.5,center_sp_mean_lab[1]*1,center_sp_mean_lab[2]*1)#修改lab特征各分量权重占比
                            else:
                                center_sp_points_xyz = sub_xyz[point_inds]
                                center_sp_points_mean = center_sp_points_xyz.mean(axis=0)
                                center_sp_points_relative = center_sp_points_xyz - center_sp_points_mean

                                center_sp_points_rgb = sub_rgb[point_inds]
                                center_sp_mean_rgb = np.mean(center_sp_points_rgb, axis=0)
                                center_sp_mean_lab = rgb2lab(center_sp_mean_rgb)#rgb->lab
                                center_sp_weight_lab = (center_sp_mean_lab[0]*0.5,center_sp_mean_lab[1]*1,center_sp_mean_lab[2]*1)#修改lab特征各分量权重占比
                            
                            U, S, Vt = svd(center_sp_points_relative)#计算中心超点的法向量
                            center_sp_normal_vector = Vt[-1]
                            if center_sp_normal_vector[2] < 0:
                                center_sp_normal_vector = -center_sp_normal_vector
                                
                            
                            #计算邻居节点的法向量
                            sp_neighbours_idx = sp_neighbour_obj[superpoint_idx]
                            for sp_neighbour_idx in sp_neighbours_idx:
                                # 在此处还应该判断是否 该sp_neighbour_idx 已经被使用过
                                # 检查是否使用过，是否位于同一平面，是则"合并"进行标注
                                neighbour_point_inds = components[sp_neighbour_idx]
                                # 不合并特别大的sp,因为本身选的时候也没有选这么大的
                                if len(neighbour_point_inds) > 1000:
                                    continue
                                
                                #邻居超点纯度计算
                                neighbour_do_label, neighbour_purity = _dominant_label(input_gt[neighbour_point_inds])#计算邻居节点纯度
                                if(neighbour_do_label != do_label or purity - neighbour_purity>0.05):
                                    continue
                                #邻居超点子区域划分
                                new_neighbour_point_inds = []
                                if neighbour_purity < threshold:
                                    pseudo_gt[0][neighbour_point_inds] = 0.0
                                    #对邻居超点进行子区域划分
                                    #sub_region_list是类别值对应的列表
                                    sub_region_list = _get_sub_region_from_superpoint(prob_class=prob_class, point_inds=neighbour_point_inds)
                                    #遍历划分子区域，保留提纯后的子区域（最好还是合并下，这样统一计算，节省计算量和时间）
                                    for sub_id in range(len(sub_region_list)):
                                        sub_region_pids = sub_region_list[sub_id]
                                        if len(sub_region_pids) > min_size:
                                            #计算子区域纯度
                                            #保留大于阈值的区域，并合并在一起
                                            sub_do_label, sub_do_rate = _dominant_label(input_gt[sub_region_pids])
                                            if sub_do_rate >= threshold and do_label==sub_do_label:  
                                                new_neighbour_point_inds += sub_region_pids 
                                if new_neighbour_point_inds!=[]:
                                    neighbour_sp_points_xyz = sub_xyz[new_neighbour_point_inds]
                                else:
                                    neighbour_sp_points_xyz = sub_xyz[neighbour_point_inds]
                                neighbour_sp_points_mean = neighbour_sp_points_xyz.mean(axis=0)
                                neighbour_points_relative = neighbour_sp_points_xyz - neighbour_sp_points_mean
                                U, S, Vt = svd(neighbour_points_relative)
                                # 相邻sp的法向量
                                neighbour_sp_normal_vector = Vt[-1]
                                if neighbour_sp_normal_vector[2] < 0:
                                    neighbour_sp_normal_vector = -neighbour_sp_normal_vector
                                #计算中心超点和邻居超点的法向量相似度，判断二者是否处于同一平面
                                vector_cosine_similarity =  calculate_similarity(center_sp_normal_vector, neighbour_sp_normal_vector)
                                # if(abs(vector_cosine_similarity)<0.97):#小于法向量相似度阈值的该邻居超点被丢弃
                                #         ignore_merge_vector_sp_count+=1
                                #         continue
                                
                                # 邻居sp的颜色衡量度
                                if new_neighbour_point_inds!=[]:
                                    neighbour_sp_points_rgb = sub_rgb[new_neighbour_point_inds]
                                else:
                                    neighbour_sp_points_rgb = sub_rgb[neighbour_point_inds]
                                neighbour_sp_mean_rgb = np.mean(neighbour_sp_points_rgb, axis=0)
                                neighbour_sp_mean_lab = rgb2lab(neighbour_sp_mean_rgb)    
                                neighbour_sp_weight_lab = (neighbour_sp_mean_lab[0]*0.5,neighbour_sp_mean_lab[1]*1,neighbour_sp_mean_lab[2]*1)
                                    
                                distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(neighbour_sp_weight_lab, center_sp_weight_lab)))
                                rgb_sim = 1 / (1 + math.exp(-distance*0.5))
                                # if (abs(rgb_sim) < 0.94):
                                #         ignore_merge_rgb_sp_count+=1
                                #         continue
                                w1 = 1/ (1 + abs(vector_cosine_similarity))
                                w2 = 1/ (1+abs(rgb_sim))
                                if (w1 * abs(vector_cosine_similarity) + w2 * abs(rgb_sim) < 0.99 ):
                                    ignore_merge_vector_and_rgb_sp_count += 1
                                    continue
                                if new_neighbour_point_inds!=[]:
                                        
                                    pseudo_gt[0][new_neighbour_point_inds] = 1.0
                                    pseudo_gt[1][new_neighbour_point_inds] = neighbour_do_label * 1.0
                                    total_obj["selected_class_list"].append(neighbour_do_label)
                                            
                                    used_superpoint_inds.append(sp_neighbour_idx)  #极小的概率会标错，待试验确定标记错误的影响 // 不对，这里不会标错，因为上面使用的input_gt获得的neigh_do_label和purity
                                    w["sp_num"] = w["sp_num"] + 1
                                    w["p_num"] = w["p_num"] + len(new_neighbour_point_inds)
                                else:
                                    pseudo_gt[0][neighbour_point_inds] = 1.0
                                    pseudo_gt[1][neighbour_point_inds] = neighbour_do_label * 1.0
                                    total_obj["selected_class_list"].append(neighbour_do_label)
                                            
                                    used_superpoint_inds.append(sp_neighbour_idx)  #极小的概率会标错，待试验确定标记错误的影响 // 不对，这里不会标错，因为上面使用的input_gt获得的neigh_do_label和purity
                                    w["sp_num"] = w["sp_num"] + 1
                                    w["p_num"] = w["p_num"] + len(neighbour_point_inds)    
                else:
                    break   
    else:
        print("not find oracle_mode==" + get_sampler_args_str(sampler_args))
        1 / 0
    print("ignore_merge_vector_sp_count:  ",ignore_merge_vector_sp_count)
    print("ignore_merge_rgb_sp_count:  ",ignore_merge_rgb_sp_count)
    print("ignore_purity_sp_count: ",ignore_purity_sp_count)
    print("ignore_merge_vector_and_rgb_sp_count:", ignore_merge_vector_and_rgb_sp_count)

    return pseudo_gt, used_superpoint_inds

def _help(input_path, data_path, total_obj, current_path, cloud_name, superpoint_inds, w, sampler_args, prob_class, threshold, budget, min_size):

    with open(os.path.join(data_path, "superpoint", cloud_name + ".superpoint"), "rb") as f:
        sp = pickle.load(f)
    components = sp["components"]
    # pseudo gt
    pseudo_gt_path = os.path.join(current_path, cloud_name + ".gt")
    with open(pseudo_gt_path, "rb") as f:
        pseudo_gt = pickle.load(f)
        pseudo_gt = np.asarray(pseudo_gt)
    # input gt
    data = read_ply(os.path.join(input_path, cloud_name + ".ply"))
    input_gt = np.asarray(data['class'])

    pseudo_gt, used_superpoint_inds = oracle_labeling(superpoint_inds=superpoint_inds, components=components, input_gt=input_gt, pseudo_gt=pseudo_gt,
                    cloud_name=cloud_name, w=w, sampler_args=sampler_args, prob_class=prob_class, threshold=threshold,
                    budget=budget, min_size=min_size, total_obj=total_obj, ply_data=data)

    with open(os.path.join(pseudo_gt_path), "wb") as f:
        pickle.dump(pseudo_gt, f)

    total_obj["unlabeled"][cloud_name] = list(set(total_obj["unlabeled"][cloud_name]) - set(used_superpoint_inds))
    if len(total_obj["unlabeled"][cloud_name]) == 0:
        del total_obj["unlabeled"][cloud_name]

def _help_seed(input_path, data_path, total_obj, current_path, cloud_name, superpoint_inds, w):
    # input_path,data_path:           /home/ncl/dataset/Semantic3D/input_0.060       ,  /home/ncl/dataset/Semantic3D/0.012
    # current_path:                   /home/ncl/dataset/Semantic3D/0.012/sampling/baseline-30-0.92/round_1
    # w = {"sp_num": 0, "p_num": 0, "p_num_list": [], "sp_id_list": [], "sub_num": 0, "sub_p_num": 0}
    with open(os.path.join(data_path, "superpoint", cloud_name + ".superpoint"), "rb") as f:
        sp = pickle.load(f)
    # components表示这个场景的superpoint，假设这个场景有10000个超点，则components有10000个array,每个array包含所有属于该超点的point_idx
    components = sp["components"]
    # pseudo gt gt表示每个sp的伪标签
    #   [[0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]]
    pseudo_gt_path = os.path.join(current_path, cloud_name + ".gt")
    with open(pseudo_gt_path, "rb") as f:
        # [2, len(xyz)]
        #    [[0. 0. 0. ... 0. 0. 0.]
        #     [0. 0. 0. ... 0. 0. 0.]]
        pseudo_gt = pickle.load(f)
        pseudo_gt = np.asarray(pseudo_gt)
    # input gt
    # 将gt赋值
    # 读取这个场景下采样后的.ply文件
    data = read_ply(os.path.join(input_path, cloud_name + ".ply"))
    input_gt = np.asarray(data['class'])

    for superpoint_idx in superpoint_inds:
        # 每个point_inds为一个superpoint包含的 所有点 的索引
        point_inds = components[superpoint_idx]
        pseudo_gt[0][point_inds] = 1.0
        pseudo_gt[1][point_inds] = input_gt[point_inds]  # precise label
        # pseudo_gt包含这个sub_ply的所有点，这个循环会将从每个场景选出的sp中包含的点对应的[0]置1，[1]为实际label
        w["sp_num"] = w["sp_num"] + 1
        w["p_num"] = w["p_num"] + len(point_inds)
    # w为所有场景本次选出的sp数量以及选出的sp包含的point数量

    with open(os.path.join(pseudo_gt_path), "wb") as f:
        pickle.dump(pseudo_gt, f)
    # 重新写回.gt文件，已经将选出的点对应位置  置 1和 真实label

    # 将每个场景中本次选择出来的sp从total_obj["unlabeled"][cloud_name]中删除
    total_obj["unlabeled"][cloud_name] = list(set(total_obj["unlabeled"][cloud_name]) - set(superpoint_inds))
    if len(total_obj["unlabeled"][cloud_name]) == 0:
        del total_obj["unlabeled"][cloud_name]

def compute_entropy(x):

    class_num = np.array(x).shape[-1]
    x = np.reshape(x, [-1, class_num])
    k = np.log2(x)
    where_are_inf = np.isinf(k)
    k[where_are_inf] = 0
    entropy = np.sum(np.multiply(x, k), axis=-1)
    return -1 * entropy  # [sampler_batch_size * point_number]

def  add_classbal(class_num, region_class, region_uncertainty):
    weights = weights_percentage(list_class=region_class, class_num=class_num)
    class_bal_region_uncertainty = np.multiply(region_uncertainty, np.exp(-np.asarray(weights)))
    return class_bal_region_uncertainty

def add_clsbal(class_num, region_class, region_uncertainty, total_obj):
    list_class = list(region_class) + list(total_obj["selected_class_list"])  # concat
    weights = weights_percentage(list_class=list_class, class_num=class_num)
    class_bal_region_uncertainty = np.multiply(region_uncertainty, np.exp(-np.asarray(weights[0:len(region_uncertainty)])))
    return class_bal_region_uncertainty

def get_labeled_selection_cloudname_spidx_pointidx(input_path, data_path, labeled_region_reference_dict, class_num, round_num):
    dominant_label_list = []
    labeled_region_reference = []  # ele {cloud_name:, sp_idx:, dominant_point_ids:}

    for cloud_name in labeled_region_reference_dict:
        with open(join(data_path, "superpoint",
                       cloud_name + ".superpoint"), "rb") as f:
            sp = pickle.load(f)
        components = sp["components"]

        sub_ply_file = join(input_path, '{:s}.ply'.format(cloud_name))
        data = read_ply(sub_ply_file)
        cloud_point_label = data['class']  # shape=[point_number]

        sp_idx_list = labeled_region_reference_dict[cloud_name]
        for sp_idx in sp_idx_list:
            point_ids = components[sp_idx]
            dominant_label, idns = _dominant_2(cloud_point_label[point_ids])
            dominant_point_ids = np.array(point_ids)[idns]
            # dominant_label_list:已标注的所有超点的dominant_label
            dominant_label_list.append(dominant_label)
            labeled_region_reference.append({"cloud_name": cloud_name, "sp_idx": sp_idx, "dominant_point_ids": dominant_point_ids})

    labeled_region_reference = np.array(labeled_region_reference, dtype='object')
    # 用dominant_label_list  计算所有已标注的sp的label权重
    # weight=[sp1的label所占权重，sp2的label所占权重...]  每个超点的类别的权重
    weights = weights_percentage(list_class=dominant_label_list, class_num=class_num)
    probability = weights / np.sum(weights)
    labeled_all_num = len(probability)
    batch = (round_num - 1) * 1000
    if batch > labeled_all_num:
        batch = labeled_all_num
    # 占比多的点选出来的就多，占比少的点选出来的就少
    
    selection = np.random.choice(labeled_all_num, batch, replace=False, p=probability)

    labeled_select_region = {}
    for item in labeled_region_reference[selection]:
        cloud_name = item["cloud_name"]
        sp_idx = item["sp_idx"]
        dominant_point_ids = item["dominant_point_ids"]
        if cloud_name not in labeled_select_region:
            labeled_select_region[cloud_name] = {}
        labeled_select_region[cloud_name][sp_idx] = dominant_point_ids
    # 根据加权概率选择出的已标注的sp，返回batch个其dominant_point的点的索引
    return labeled_select_region, batch  # {cloud_name: {sp_idx: dominant_point_ids}}

def compute_features(dataset_name, test_area_idx, sampler_args, round_num, reg_strength, model, labeled_select_regions, unlabeled_candidate_regions):
    labeled_select_features = []
    labeled_select_ref = []
    unlabeled_candidate_features = []
    unlabeled_candidate_ref = []

    if dataset_name == "Semantic3D":
        sample_data = Semantic3D_Dataset_Sampling(sampler_args=sampler_args,
                                         round_num=round_num, mode="sampling", reg_strength=reg_strength)

    sample_loader = DataLoader(sample_data, batch_size=1, shuffle=False)

    for i, dat in enumerate(sample_loader):
        print("2", "sample_loader", i)
        total_num = 0
        for dat_sub in dat:
            total_num += len(dat_sub[-2][0][0])

        # 存储该场景下所有点的特征 32维向量
        total_last_second_features = np.zeros([total_num, 32])
        cloud_idx = 0
        for dat_sub in dat:
            last_second_features, cloud_inds, point_idx = model.sess.run([model.last_second_features, model.input_cloud_inds, model.input_input_inds],
                                                                         feed_dict=model.get_feed_dict_sub(dat_sub=dat_sub))
            total_last_second_features[point_idx[0]] = last_second_features
            cloud_idx = cloud_inds[0]

        cloud_name = sample_data.input_cloud_names[cloud_idx]
        # 该场景下所有已标注sp的特征：其do_label_point点的平均特征
        if cloud_name in labeled_select_regions:
            for sp_idx in labeled_select_regions[cloud_name]:
                dominant_point_ids = labeled_select_regions[cloud_name][sp_idx]
                labeled_select_features.append(np.mean(total_last_second_features[dominant_point_ids], axis=0))
                labeled_select_ref.append({"cloud_name": cloud_name, "sp_idx": sp_idx})
        # 该场景下top 2k 个 未标注sp的特征：其do_label_point点的平均特征
        if cloud_name in unlabeled_candidate_regions:
            for sp_idx in unlabeled_candidate_regions[cloud_name]:
                dominant_point_ids = unlabeled_candidate_regions[cloud_name][sp_idx]
                unlabeled_candidate_features.append(np.mean(total_last_second_features[dominant_point_ids], axis=0))
                unlabeled_candidate_ref.append({"cloud_name": cloud_name, "sp_idx": sp_idx})

    return labeled_select_features, labeled_select_ref, unlabeled_candidate_features, unlabeled_candidate_ref

class SeedSampler:
    """
    use precise labeling
    """
    # Sampler = SeedSampler("/home/ncl/dataset/" +dataset_name + "/" + input_, "/home/ncl/dataset/" + dataset_name + "/" + str(reg_strength), total_sp_num, sampler_args)
    def __init__(self, input_path, data_path, total_num, sampler_args):
        #  /home/ncl/dataset/Semantic3D/input_0.060       ,  /home/ncl/dataset/Semantic3D/0.012                          ,  434044     , seed
        self.input_path = input_path
        self.data_path = data_path
        self.total_num = total_num  # sp_number 434044
        self.sampler_args = sampler_args

    def _iteration(self, current_path, total_obj, number, w):
        #   /home/ncl/dataset/Semantic3D/0.012/sampling/baseline-30-0.92/round_1
        #   file num 13 sp num 434044 point num 25921310 total_obj["unlabeled"][cloud_name] 
        #   3472
        #   w = {"sp_num": 0, "p_num": 0, "p_num_list": [], "sp_id_list": [], "sub_num": 0, "sub_p_num": 0}
       
        remain_number = 0
        # 从范围 [0, 1, 2, ..., self.total_num-1] 中随机选择 number 个元素，而且不允许重复选择，number参数为sp_batch_size=3472
        rand_inds = np.random.choice(range(self.total_num), int(number), replace=False)
        length = len(total_obj["unlabeled"])  # 这里length=13,即划分了超点的训练集场景数量，每次可能会变,total_obj一直被修改，每次选择出一部分sp后会将这部分sp删除
        cloud_name_list = []
        # # 为每个场景创建一个从0-num(count_of_sp)的索引
        # total_obj["unlabeled"][cloud_name] = np.arange(len(components))
        for cloud_name in total_obj["unlabeled"]:
            cloud_name_list.append(cloud_name)

        # [0,0,0,...] len=13
        # 434044个超点
        each_file_number = np.zeros([length], dtype=np.int32)
        # 有点像将3472平均分到13个each_file_number里,each_file_number[d]表示要从第d个场景中选出each_file_number[d]个超点的索引并打乱顺序，
        # 作用貌似是：想在13个场景中，从每个场景尽可能选出相同个数的superpoint   × 若number<434044，则不平均
        print("rand_inds的length: {}".format(len(rand_inds)))
        for ind in rand_inds:
            d = ind % length
            each_file_number[d] = each_file_number[d] + 1
        

        for i in range(length):
            if each_file_number[i] > 0:
                cloud_name = cloud_name_list[i]
                # len(total_obj["unlabeled"][cloud_name]) : 每个场景的超点数量
                if len(total_obj["unlabeled"][cloud_name]) >= each_file_number[i]:
                    # 从对应的场景中随机选出的 each_file_number[i]个超点的索引
                    superpoint_inds = np.random.choice(list(total_obj["unlabeled"][cloud_name]), int(each_file_number[i]), replace=False).tolist()
                    _help_seed(input_path=self.input_path, data_path=self.data_path, total_obj=total_obj, current_path=current_path, cloud_name=cloud_name,
                               superpoint_inds=superpoint_inds, w=w)
                else:
                    # 在这个场景中，超点个数不足以选出原本想每个场景选的相同个数，就选择这个场景的全部超点
                    superpoint_inds = total_obj["unlabeled"][cloud_name]
                    # 累加在每个场景选sp时不足的sp个数
                    remain_number = remain_number + each_file_number[i] - len(superpoint_inds) # 没选到的superpoint的个数
                    _help_seed(input_path=self.input_path, data_path=self.data_path, total_obj=total_obj, current_path=current_path, cloud_name=cloud_name,
                               superpoint_inds=superpoint_inds, w=w)
        # 如果所有的点都被选择过了，则清空total_obj["unlabeled"]，并重新写回total.pkl
        if remain_number == 0 or len(total_obj["unlabeled"]) == 0:
            # save total_obj
            with open(os.path.join(current_path, "total.pkl"), "wb") as f:
                pickle.dump(total_obj, f)
        else:
            return self._iteration(current_path, total_obj, remain_number, w)

    def sampling(self, model, batch_size, last_round, w):
        #               None, 3472    , 0         , w
        # w = {"sp_num": 0, "p_num": 0, "p_num_list": [], "sp_id_list": [], "sub_num": 0, "sub_p_num": 0}
        if last_round == 0:
            # /home/ncl/dataset/Semantic3D/0.012/superpoint ，其中包含了每个场景的.superpoint文件和 .gt文件
            current_path = os.path.join(self.data_path, "superpoint")
        else:
            current_path = os.path.join(self.data_path, "sampling", get_sampler_args_str(self.sampler_args), "round_" + str(last_round))

        round_num = last_round+1
        # /home/ncl/dataset/Semantic3D/0.012/sampling/baseline-30-0.92/round_1 在这里创建的samping文件夹及其目录下/baseline-30-0.92/round_*目录
        next_round_path = os.path.join(self.data_path, "sampling", get_sampler_args_str(self.sampler_args), "round_" + str(round_num))
        os.makedirs(next_round_path) if not os.path.exists(next_round_path) else None
        # copy content to next round 只复制.gt和total.pkl
        list1 = os.listdir(current_path)
        for file1 in list1:
            p = os.path.join(current_path, file1)
            if os.path.isfile(p) and ".superpoint" not in file1:
                shutil.copyfile(p, os.path.join(next_round_path, file1))
        
        # 这里在total.pkl时好像有点问题，没有将total_obj["unlabeled"] 复制进去

        # read total_obj
        with open(os.path.join(next_round_path, "total.pkl"), "rb") as f:
            # file num 13 sp num 434044 point num 25921310 unlabeled={.......}
            total_obj = pickle.load(f)
        # batch_size = 3472
        # 在_iteration中会根据选出的sp,其中的所有点p，修改.gt文件,被选过的点[0]=1,[1]=label,同时将每个场景中本次选择出来的sp从total_obj["unlabeled"][cloud_name]中删除
        self._iteration(current_path=next_round_path, total_obj=total_obj, number=batch_size, w=w)

class AllSampler:
    def __init__(self, input_path, data_path, total_num, sampler_args):
        self.input_path = input_path
        self.data_path = data_path
        self.total_num = total_num
        self.sampler_args = sampler_args

    def sampling(self, model, batch_size, last_round, w, threshold):
        budget = {}
        budget["click"] = batch_size

        if last_round == 1:
            current_path = os.path.join(self.data_path, "superpoint")
        else:
            current_path = os.path.join(self.data_path, "sampling", get_sampler_args_str(self.sampler_args), "round_" + str(last_round))

        round_num = last_round+1
        next_round_path = os.path.join(self.data_path, "sampling", get_sampler_args_str(self.sampler_args), "round_" + str(round_num))
        os.makedirs(next_round_path) if not os.path.exists(next_round_path) else None
        # copy content to next round
        list1 = os.listdir(current_path)
        for file1 in list1:
            p = os.path.join(current_path, file1)
            if os.path.isfile(p) and ".superpoint" not in file1:
                shutil.copyfile(p, os.path.join(next_round_path, file1))

        # read total_obj
        with open(os.path.join(next_round_path, "total.pkl"), "rb") as f:
            total_obj = pickle.load(f)
            if "selected_class_list" not in total_obj:
                total_obj["selected_class_list"] = []

        cloud_name_list = []
        for cloud_name in total_obj["unlabeled"]:
            cloud_name_list.append(cloud_name)

        for cloud_name in cloud_name_list:
            superpoint_inds = total_obj["unlabeled"][cloud_name]
            _help(input_path=self.input_path, data_path=self.data_path, total_obj=total_obj, current_path=next_round_path, cloud_name=cloud_name,
                  superpoint_inds=superpoint_inds, w=w, sampler_args=self.sampler_args, prob_class=None, threshold=threshold, budget=budget, min_size=1)

        # save total_obj
        with open(os.path.join(next_round_path, "total.pkl"), "wb") as f:
            pickle.dump(total_obj, f)

class RandomSampler:
    def __init__(self, input_path, data_path, total_num, sampler_args, min_size):
        self.input_path = input_path
        self.data_path = data_path
        self.total_num = total_num
        self.sampler_args = sampler_args
        self.min_size = min_size

    def _iteration(self, current_path, total_obj, w, threshold, budget):
        '''
        budget = {}
        budget["click"] = batch_size = 10000
        '''
        # 从所有sp中选出10000个idx
        rand_inds = np.random.choice(range(self.total_num), budget["click"], replace=False)
        # 训练场景个数
        length = len(total_obj["unlabeled"])

        cloud_name_list = []
        for cloud_name in total_obj["unlabeled"]:
            cloud_name_list.append(cloud_name)

        each_file_number = np.zeros([length], dtype=np.int32)
        for ind in rand_inds:
            d = ind % length
            each_file_number[d] = each_file_number[d] + 1

        for i in range(length):
            if each_file_number[i] > 0:
                cloud_name = cloud_name_list[i]
                if len(total_obj["unlabeled"][cloud_name]) >= each_file_number[i]:
                    superpoint_inds = np.random.choice(list(total_obj["unlabeled"][cloud_name]), int(each_file_number[i]), replace=False).tolist()
                    _help(input_path=self.input_path, data_path=self.data_path, total_obj=total_obj, current_path=current_path, cloud_name=cloud_name,
                               superpoint_inds=superpoint_inds, w=w, sampler_args=self.sampler_args, prob_class=None, threshold=threshold, budget=budget, min_size=self.min_size)
                else:
                    superpoint_inds = total_obj["unlabeled"][cloud_name]
                    _help(input_path=self.input_path, data_path=self.data_path, total_obj=total_obj, current_path=current_path, cloud_name=cloud_name,
                               superpoint_inds=superpoint_inds, w=w, sampler_args=self.sampler_args, prob_class=None, threshold=threshold, budget=budget, min_size=self.min_size)

        if budget["click"] == 0 or len(total_obj["unlabeled"]) == 0:
            # save total_obj
            with open(os.path.join(current_path, "total.pkl"), "wb") as f:
                pickle.dump(total_obj, f)
        else:
            return self._iteration(current_path, total_obj, w, threshold, budget)

    def sampling(self, model, batch_size, last_round, w, threshold, gcn_gpu):
        # Sampler.sampling(model=model, batch_size=10000, last_round=r-1(0~33), w=w, threshold=0.9, gcn_gpu=1)
        '''
        w = {"sp_num": 0, "p_num": 0, "p_num_list": [], "sp_id_list": [], "sub_num": 0,
                 "sub_p_num": 0, "ignore_sp_num": 0, "split_sp_num": 0}
        '''
        budget = {}
        budget["click"] = batch_size

        if last_round == 1:
            current_path = os.path.join(self.data_path, "sampling", "seed", "round_1")
        else:
            current_path = os.path.join(self.data_path, "sampling", get_sampler_args_str(self.sampler_args), "round_" + str(last_round))

        round_num = last_round+1
        next_round_path = os.path.join(self.data_path, "sampling", get_sampler_args_str(self.sampler_args), "round_" + str(round_num))
        os.makedirs(next_round_path) if not os.path.exists(next_round_path) else None
        # copy content to next round
        list1 = os.listdir(current_path)
        for file1 in list1:
            p = os.path.join(current_path, file1)
            if os.path.isfile(p) and ".superpoint" not in file1:
                # 拷贝.gt的时候还是上次seed选择出来的哪些点
                shutil.copyfile(p, os.path.join(next_round_path, file1))

        # read total_obj
        with open(os.path.join(next_round_path, "total.pkl"), "rb") as f:
            total_obj = pickle.load(f)
            if "selected_class_list" not in total_obj:
                total_obj["selected_class_list"] = []

        self._iteration(current_path=next_round_path, total_obj=total_obj, w=w, threshold=threshold, budget=budget)

class TSampler:
    def __init__(self, input_path, data_path, total_num, test_area_idx, sampler_args, reg_strength, min_size, dataset_name):
        self.input_path = input_path
        self.data_path = data_path
        self.total_num = total_num
        self.test_area_idx = test_area_idx
        self.sampler_args = sampler_args
        self.reg_strength = reg_strength
        self.min_size = min_size
        self.dataset_name = dataset_name

    def create_file_top_and_all(self, region_reference, sorted_inds, batch_size):
        file_list_top = {}
        file_list_all = {}
        for i in range(len(sorted_inds)):
            idx = sorted_inds[i]
            cloud_name, sp_idx, dominant_point_ids = region_reference[idx]["cloud_name"], region_reference[idx]["sp_idx"], region_reference[idx]["dominant_point_ids"]
            if i < batch_size:
                if cloud_name not in file_list_top:
                    file_list_top[cloud_name] = {}
                    file_list_top[cloud_name]["sp_idx_list"] = []
                file_list_top[cloud_name][sp_idx] = dominant_point_ids
                file_list_top[cloud_name]["sp_idx_list"].append(sp_idx)

            if cloud_name not in file_list_all:
                file_list_all[cloud_name] = {}
                file_list_all[cloud_name]["sp_idx_list"] = []
            file_list_all[cloud_name][sp_idx] = dominant_point_ids
            file_list_all[cloud_name]["sp_idx_list"].append(sp_idx)

        return file_list_top, file_list_all

    def create_sp_inds_with_position(self, file_list_top, file_list_all, cloud_name):

        selected_num = len(file_list_top[cloud_name]["sp_idx_list"])
        candicate_sp_inds = np.asarray(file_list_all[cloud_name]["sp_idx_list"][:2 * selected_num])
        with open(join(self.data_path, "superpoint",
                       cloud_name + ".superpoint"), "rb") as f:
            sp = pickle.load(f)
        components = sp["components"]
        data = read_ply(
            join(self.input_path, '{:s}.ply'.format(cloud_name)))
        xyz = np.vstack((data['x'], data['y'], data['z'])).T
        candicate_superpoints = []
        candicate_superpoints_centroid = []
        for si in range(len(candicate_sp_inds)):
            sp_idx = candicate_sp_inds[si]
            x_y_z = xyz[components[sp_idx]]
            center_x = (np.min(x_y_z[:, 0]) + np.max(x_y_z[:, 0]))/2.0
            center_y = (np.min(x_y_z[:, 1]) + np.max(x_y_z[:, 1])) / 2.0
            center_z = (np.min(x_y_z[:, 2]) + np.max(x_y_z[:, 2])) / 2.0
            candicate_superpoints_centroid.append(np.asarray([center_x, center_y, center_z]))
            candicate_superpoints.append(x_y_z)
        candicate_superpoints_centroid = np.asarray(candicate_superpoints_centroid)

        selected_ids = farthest_superpoint_sample(candicate_superpoints, candicate_superpoints_centroid, selected_num, 0)
        return candicate_sp_inds[selected_ids]

    def prediction(self, model, total_obj, round_num):
        print("prediction round_num:",round_num)
        # raise ValueError("program terminated")
        region_uncertainty = []#区域不确定度
        region_class = []#区域类别

        unlabeled_region_reference = []#未标注区域索引
        labeled_region_reference_dict = {}#标注区域索引字典

        # 存放了所有场景的每个点的预测类别
        prob_class_dict = {}

        if self.dataset_name == "Semantic3D":
            sample_data = Semantic3D_Dataset_Sampling(sampler_args=self.sampler_args,
                                        round_num=round_num, mode="sampling", reg_strength=self.reg_strength)

        sample_loader = DataLoader(sample_data, batch_size=1, shuffle=False)
        print("sample_loader初始化完毕,准备迭代读取----------------")
        class_num = None
        for i, dat in enumerate(sample_loader):
        # if 0<1:
            # with open('/home/ncl/dataset/Semantic3D/analyse_data/input_part_list.pkl', 'rb') as f:
            #     dat = pickle.load(f)
            # print(len(dat))
            '''
            每次迭代即调用一次Semantic3D_Dataset_Sampling的__getitem__方法,返回一个训练场景的数据,dat的长度为ombination后的part区域,每个dat_sub是1个子区域
            bildstein_station1_xyz_intensity_rgb KD_Tree中点的个数: 1037198
            这个part的点的数量: 205176
            这个part的点的数量: 0
            这个part的点的数量: 19706
            这个part的点的数量: 0
            这个part的点的数量: 520194
            这个part的点的数量: 0
            这个part的点的数量: 292122
            这个part的点的数量: 0
            dat_sub是一个长度26的列表 4*5  +  6

            这里dat返回了4个非0的part
            '''
            # print("1", "sample_loader", i)
            # print("sampler2中prediction(),",i,len(dat))  # sampler2中prediction(), 0 4
            # print("dat length:", len(dat))
            #计算该场景点云点数
            total_num = 0
            for dat_sub in dat: #对输入数据根据划分的子区域进行遍历
                # print("dat_sub len:",len(dat_sub[-2][0][0])) # 非空part的点的数量
                # print("dat_sub :",dat_sub[-2])
                '''
                dat_sub len: 205118
                dat_sub : tensor([[[360951, 604651, 384743,  ..., 700752,  88754, 237132]]])
                dat_sub len: 19652
                dat_sub : tensor([[[ 78374,  73978, 905032,  ..., 679136,  97487, 122744]]])
                dat_sub len: 520252
                dat_sub : tensor([[[317612, 236260, 925731,  ..., 869047, 307993, 829274]]])
                dat_sub len: 292176
                dat_sub : tensor([[[618585, 329240,  41807,  ..., 159745, 571836, 156508]]])
                total_num shape: 1037198
                '''

                total_num += len(dat_sub[-2][0][0])  #该part的点的个数
                # print(len(dat_sub[-2][0][0]))
            #     print("dat_sub length:", len(dat_sub))
            #     input_list=dat_sub
            #     print("input_list len:", len(input_list))
            #     for sub_input_lists in input_list:
            #         print("len of input_sub_points:", len(sub_input_lists))
            #         for sub_sub_input_lists in sub_input_lists:
            #             print("shape of sub_sub_input_lists:",sub_sub_input_lists.shape)
            # raise ValueError("program terminated")
                
            #这个是每个场景的局部变量，主要用于辅助计算这个场景中sp的不确定度
            total_pixel_uncertainty = np.zeros([total_num])#逐点不确定度
            total_prob_class = np.zeros([total_num], dtype=np.int)#逐点类别
            # print("total_num shape:",total_num) # total_num shape: 1037198
            
            
            cloud_idx = 0
            total_class_counts = np.zeros(8, dtype=int)   #用于统计每类又多少个样本
            # 用于统计每个点的预测概率，用于计算后面的sp的do_label,因为标的时候是根据do_label
            points_label = {}
            for dat_sub in dat:#遍历该场景下每个划分子区域
                '''
                1.计算每个点的prob_logits,并得到其所属类别 2.计算每个点的不确定度
                '''

                prob_logits, cloud_idns, point_idx = model.sess.run([model.prob_logits, model.input_cloud_inds, model.input_input_inds], feed_dict=model.get_feed_dict_sub(dat_sub=dat_sub))
                class_num = prob_logits.shape[-1]  # 代表类别的数量 
                predicted_labels = np.argmax(prob_logits, axis=1)
                for i, point_id in enumerate(point_idx[0]):
                    points_label[point_id] = predicted_labels[i]
                class_counts = np.bincount(predicted_labels, minlength=8)
                total_class_counts += class_counts
                
                # print()
                # print("倒数第1个点的概率:",prob_logits[prob_logits.shape[0]-1])
                # print("倒数第1个点的下标:",point_idx[0][prob_logits.shape[-1]-1])
                # print()
                # print("倒数第2个点的概率:",prob_logits[prob_logits.shape[0]-2])
                # print("倒数第2个点的下标:",point_idx[0][prob_logits.shape[-1]-2])
                # print()
                # print("倒数第3个点的概率:",prob_logits[prob_logits.shape[0]-3])
                # print("倒数第3个点的下标:",point_idx[0][prob_logits.shape[-1]-3])
                # print()
                # print(prob_logits)
                # print()
                # print("dat_sub shape:", dat_sub.shape)
                # print("prob_logits shape:",prob_logits.shape)           
                # print("cloud_idns shape:",cloud_idns.shape)             
                # print("point_idx shape:",point_idx.shape)               
                '''

                print("prob_logits shape:",prob_logits.shape)           prob_logits shape: (205176, 8)
                print("cloud_idns shape:",cloud_idns.shape)             cloud_idns shape: (1,)
                print("point_idx shape:",point_idx.shape)               point_idx shape: (1, 205176)
                
                '''
                # print(prob_logits)
                # 统计每一个点预测概率最高的 label_index
                total_prob_class[point_idx[0]] = np.argmax(prob_logits, axis=-1)  # [batch_size * point_num] 所有点的total_prob_class
                # print("total_prob_class shape:",total_prob_class.shape)
                # 计算每一个点的不确定度，second best / best
                total_pixel_uncertainty[point_idx[0]] = compute_point_uncertainty(prob_logits=prob_logits, sampler_args=self.sampler_args)  # [batch_size * point_num]
                cloud_idx = cloud_idns[0]
                
                '''
                print("total_prob_class shape:",total_prob_class.shape)             total_prob_class shape: (1037198,)
                print("cloud_idns shape:",total_pixel_uncertainty.shape)            cloud_idns shape: (1037198,)
                print("cloud_idx:",cloud_idx)                                       cloud_idx: 0                
                '''
                # raise ValueError("program terminated")
            # raise ValueError("program terminated")
            # for i, count in enumerate(class_counts):
            #     print(f"类别 {i} 包含 {count} 个点")
            # 1/0


            # save points_label = {}
            # with open('/home/ncl/dataset/Semantic3D/analyse_data/bildstein_station1_xyz_intensity_rgb_points_label.pkl', 'wb') as f:
            #     pickle.dump(points_label, f)
            data_obj={}
            # data_obj
            data_obj["point_pred_label"] = total_prob_class
            data_obj["point_uncertainty"] = total_pixel_uncertainty

            cloud_name = sample_data.input_cloud_names[cloud_idx]
            prob_class_dict[cloud_name] = total_prob_class

            # 取这个场景的.superpoint文件读取sp bildstein_station1_xyz_intensity_rgb 场景中point:1037198 ,sp:9222
            with open(join(self.data_path, "superpoint",
                                       cloud_name + ".superpoint"), "rb") as f:
                sp = pickle.load(f)
            components = sp["components"]
            data_obj["components"] = components
            data_obj["in_component"] = sp["components"]
            '''
                total_obj["unlabeled"][cloud_name] = np.arange(len(components))

            total_obj["file_num"] = file_num
            total_obj["sp_num"] = sp_num
            total_obj["point_num"] = point_num
            total_obj["selected_class_list"] = []
            '''
            ignored_sp_idx=[]
            valid_sp_idx=[]
            data_obj["sp_idx"] = []
            data_obj["all_sp_uncertainty"] = []
            # 对于这个场景的所有sp
            for sp_idx in range(len(components)):
                # #计算所有sp的不确定度
                # data_obj["sp_idx"].append(sp_idx)
                # point_ids = components[sp_idx]
                # data_obj["all_sp_uncertainty"].append(
                #                 compute_region_uncertainty(pixel_uncertainty=total_pixel_uncertainty[point_ids],
                #                                            pixel_class=total_prob_class[point_ids],
                #                                            class_num=class_num, sampler_args=self.sampler_args))
                # data_obj["all_sp_dolabel"], data_obj["all_sp_purity"] = _dominant_label(total_prob_class[point_ids])
                # 当前sp未被使用过
                point_ids = components[sp_idx]
                #计算所有sp的不确定度
                if len(point_ids) >= self.min_size and len(point_ids) <= 1000:
                    data_obj["sp_idx"].append(sp_idx)
                    # point_ids = components[sp_idx]
                    data_obj["all_sp_uncertainty"].append(
                                    compute_region_uncertainty(pixel_uncertainty=total_pixel_uncertainty[point_ids],
                                                            pixel_class=total_prob_class[point_ids],
                                                            class_num=class_num, sampler_args=self.sampler_args))
                    data_obj["all_sp_dolabel"], data_obj["all_sp_purity"] = _dominant_label(total_prob_class[point_ids])
                # 当前sp未被使用过
                if cloud_name in total_obj["unlabeled"] and sp_idx in total_obj["unlabeled"][cloud_name]:
                    # 不选择那些过大的超点，简单粗暴的防止初始划分效果比较差的sp
                    if len(point_ids) >= self.min_size and len(point_ids) <= 1000:
                        valid_sp_idx.append(sp_idx)
                        # 对于每个sp中点的数量进行限制
                        region_uncertainty.append(
                                compute_region_uncertainty(pixel_uncertainty=total_pixel_uncertainty[point_ids],
                                                           pixel_class=total_prob_class[point_ids],
                                                           class_num=class_num, sampler_args=self.sampler_args))
                        
                        # 这个超点中的所有点的占比最高的label,以及对应的点的下标 ids 
                        do_label, idns = _dominant_2(total_prob_class[point_ids])
                        # 找到这些点
                        dominant_point_ids = np.array(point_ids)[idns]
                        # unlabeled_region_reference 添加的是一个字典,包含场景名称、超点索引，这个超点中占dominant的点的索引
                        # unlabeled_region_reference.append({"cloud_name": cloud_name, "sp_idx": sp_idx, "dominant_point_ids": dominant_point_ids})
                        unlabeled_region_reference.append({"cloud_name": cloud_name, "sp_idx": sp_idx, "dominant_point_ids": dominant_point_ids, "do_label":do_label})
                        do_label, _ = _dominant_label(total_prob_class[point_ids])
                        # 这个超点的dominant label
                        region_class.append(do_label)
                        # unlabeled_region_reference 和 region_class元素坐标一一对应
                else:
                    ignored_sp_idx.append(sp_idx)
                    # region_uncertainty.append(-1000000)
                    #当前场景已经是lebeled, 或者这个当前超点不属于unlabeled
                    if len(point_ids) >= self.min_size and len(point_ids) <= 1000:
                        if cloud_name not in labeled_region_reference_dict:
                            labeled_region_reference_dict[cloud_name] = []
                        labeled_region_reference_dict[cloud_name].append(sp_idx)
            data_obj["valid_sp_idx"] = valid_sp_idx
            data_obj["ignored_sp_idx"] = ignored_sp_idx
            data_obj["valid_sp_uncertainty"] = region_uncertainty
            data_obj["valid_sp_dolabel"] = region_class
            data_obj["cloud_name"] = cloud_name
            '''保存数据'''
            # if round_num == 3:

            # if cloud_name == "bildstein_station1_xyz_intensity_rgb":
            #     prediction_data_path = "/home/ncl/dataset/Semantic3D/sp_prediction_data_round2"
            #     round_file_name = "new_bildstein_station1_" + str(round_num) + ".data"
            #     with open(os.path.join(prediction_data_path, round_file_name), "wb") as f:
            #         pickle.dump(data_obj, f)
            #     print("写入一个场景的数据完成")
            # 1/0
            # raise ValueError("program terminated")
            # raise ValueError("program terminated")
            

        print("\n############\n compute uncertaintly successfully \n###############\n")
        if "classbal" in self.sampler_args: 
            # 会重新计算每个sp的不确定度 
            region_uncertainty = add_classbal(class_num=class_num, region_class=region_class, region_uncertainty=region_uncertainty)
        elif "clsbal" in self.sampler_args:
            region_uncertainty = add_clsbal(class_num=class_num, region_class=region_class,
                                              region_uncertainty=region_uncertainty, total_obj=total_obj)

        sorted_inds = np.argsort(-np.asarray(region_uncertainty))#降序排序
        # data_obj["sorted_inds_sp_uncertainty"] = sorted_inds
        print("\n############\n compute class balance successfully \n###############\n")
        
        # 返回了 场景+spid+其中do_label的点, 将sp不确定度由高到低排序后的sp,该场景所有点的label,labeled_region_reference_dict???
        # print("unlabeled_region_reference length:",len(unlabeled_region_reference))
        # 需要在此额外返回region_uncertainty
        return unlabeled_region_reference, sorted_inds, prob_class_dict, labeled_region_reference_dict, class_num, region_uncertainty 

    def sampling(self, model, batch_size, last_round, w, threshold, gcn_gpu):
        budget = {}
        budget["click"] = batch_size
        print("last_round:",last_round)
        if last_round == 1:
            current_path = os.path.join(self.data_path, "sampling", "seed", "round_1")
        else:
            current_path = os.path.join(self.data_path, "sampling", get_sampler_args_str(self.sampler_args), "round_" + str(last_round))

        round_num = last_round + 1
        next_round_path = os.path.join(self.data_path, "sampling", get_sampler_args_str(self.sampler_args), "round_" + str(round_num))
        os.makedirs(next_round_path) if not os.path.exists(next_round_path) else None
        # copy content to next round
        list1 = os.listdir(current_path)
        for file1 in list1:
            p = os.path.join(current_path, file1)
            if os.path.isfile(p) and ".superpoint" not in file1:
                shutil.copyfile(p, os.path.join(next_round_path, file1))

        # read total_obj
        '''
            total_obj["unlabeled"][cloud_name] = np.arange(len(components))

        total_obj["file_num"] = file_num
        total_obj["sp_num"] = sp_num
        total_obj["point_num"] = point_num
        '''
        with open(os.path.join(next_round_path, "total.pkl"), "rb") as f:
            total_obj = pickle.load(f)
            if "selected_class_list" not in total_obj:
                total_obj["selected_class_list"] = []
        
        #  self.prediction ::::   return unlabeled_region_reference, sorted_inds, prob_class_dict, labeled_region_reference_dict, class_num
        '''
        region_reference 列表,每个元素是一个字典,包含场景名称、一个超点索引,这个超点中占dominant的点的索引
        sorted_inds 中包含了按照 region_uncertainty 的值降序排列的索引序列。
        prob_class_dict 字典,包含每个场景所有点的预测类别
        labeled_region_reference_dict 包含每个场景已标注的sp_idx
        class_num 8
        '''
        '''
        # 返回了 场景+spid+其中do_label的点, 将sp不确定度由高到低排序后的sp,该场景所有点的label,labeled_region_reference_dict???
        return unlabeled_region_reference, sorted_inds, prob_class_dict, labeled_region_reference_dict, class_num
        '''
        region_reference, sorted_inds, prob_class_dict, labeled_region_reference_dict, class_num ,region_uncertainty= self.prediction(model=model, total_obj=total_obj, round_num=round_num)
        
        '''
        
        在这里对于返回的某个场景内的超点不确定度进行不确定度衰减。
        unlabeled_region_reference.append({"cloud_name": cloud_name, "sp_idx": sp_idx, "dominant_point_ids": dominant_point_ids, "do_label":do_label})
        region_uncertainty。
        1.根据region_reference中的cloud_name进行分类,拿到一个场景中本轮主动学习周期的可用的所有超点的不确定度

        2.读取.superpoint文件,根据sp_idx, 读取该超点的所有点的位置,计算该超点的位置sp_pos,拿到该场景中所有超点的位置。->所有场景
        多个场景 设置比例 or 全局最高
        每个场景要一个字典列表存所有的超点的,sp_idx,pos_xyz,sp_uncertainty,prob_class,所有场景放到一个列表中，


        '''
        # print("region_reference len")
        # print(len(region_reference))
        # 初始化结果存储结构
        cloud_sp_uncertainty = defaultdict(list)

        # 遍历unlabeled_region_reference，关联cloud_name、sp_idx和uncertainty
        for i, region in enumerate(region_reference):
            cloud_name = region["cloud_name"]
            sp_idx = region["sp_idx"]  # 当前超点的索引
            uncertainty = region_uncertainty[i]  # 按索引取对应不确定度
            prob_class = region["do_label"]
            # 将每个超点及其不确定度存储到对应的点云组中
            cloud_sp_uncertainty[cloud_name].append({"sp_idx": sp_idx, "uncertainty": uncertainty,"class": prob_class})

        # for cloud_name, superpoints in cloud_sp_uncertainty.items():
                # print("cloud_sp_uncertainty len ")
                # print(len(superpoints)) #8095

        # 转换为常规dict（可选）
        cloud_sp_uncertainty = dict(cloud_sp_uncertainty)
        # print(cloud_sp_uncertainty)
        # 打印结果（仅供验证）
        # for cloud, sp_list in cloud_sp_uncertainty.items():
        #     print(f"Cloud: {cloud}")
        #     for sp in sp_list:
        #         print(f"  SP Index: {sp['sp_idx']}, Uncertainty: {sp['uncertainty']}")
        data_path = "/data/home/dxy/dataset/Semantic3D/"
        input_path = "/data/home/dxy/dataset/Semantic3D/input_0.060"

        cloud_max_distance = defaultdict(list)

        for cloud_name, superpoints in cloud_sp_uncertainty.items():
            print(cloud_name,len(superpoints))

        for cloud_name, superpoints in cloud_sp_uncertainty.items():#场景-场景对应超点
            with open(os.path.join(data_path+"/0.012", "superpoint", cloud_name + ".superpoint"), "rb") as f:
                sp = pickle.load(f)
            components = sp["components"]  #某场景中超点映射到每个点的索引

            sub_ply_file = join(input_path, '{:s}.ply'.format(cloud_name))
            data = read_ply(sub_ply_file)#降采样后的点云输入数据
            # cloud_point_label = data['class']  # shape=[point_number]
            xyz = np.vstack((data['x'], data['y'], data['z'])).T#xyz数据

            sp_centers = []


            for sp_data in superpoints:#遍历某场景中超点
                sp_idx = sp_data["sp_idx"]  # 超点索引
                point_ids = components[sp_idx]#点索引
                sp_points_xyz = xyz[point_ids]#超点区域对应的xyz坐标
                pos_x, pos_y, pos_z = np.mean(sp_points_xyz, axis=0)#计算超点区域的中心/质心
            
                # 将平均坐标添加到sp_data中 保存中心坐标
                sp_data["pos_x"] = pos_x
                sp_data["pos_y"] = pos_y
                sp_data["pos_z"] = pos_z
                # 统计出来该场景中距离最远的两个超点坐标作为衰减系数
                sp_centers.append((pos_x, pos_y, pos_z))

            # 计算所有超点中心的两两距离
            distances = pdist(np.array(sp_centers))  # 快速计算两点之间的距离
            max_distance = np.max(distances)  # 获取最大距离
            cloud_max_distance[cloud_name] = max_distance


        # for cloud_name, max_distance in cloud_max_distance.items():
        #     print(f"Cloud Name: {cloud_name}, Max Distance: {max_distance}")
        '''
        利用
        cloud_sp_uncertainty[cloud_name].append({"sp_idx": sp_idx, "uncertainty": uncertainty,sp_data["pos_x"] = pos_x,pos_y,pos_z,"class": prob_class})
        cloud_max_distance[cloud_name] = max_distance
        衰减
        '''

        # 遍历每个场景
        cloud_max_uncertainty = defaultdict(list)

        for cloud_name, superpoints in cloud_sp_uncertainty.items():
            try:
                # 初始化当前场景的最大不确定度
                max_uncertainty = -float('inf')  # 确保初始值很小
                
                # 遍历该场景的超点
                for sp_data in superpoints:
                    sp_uncertainty = sp_data["uncertainty"]  # 获取当前超点的不确定度
                    
                    # 更新最大不确定度
                    if sp_uncertainty > max_uncertainty:
                        max_uncertainty = sp_uncertainty
                        cloud_max_uncertainty[cloud_name]={"sp_idx": sp_data["sp_idx"], "uncertainty": max_uncertainty,"pos_x" : sp_data["pos_x"],"pos_y" : sp_data["pos_y"],"pos_z" : sp_data["pos_z"],"class": sp_data["class"]}
                        # cloud_max_uncertainty[cloud_name]["uncertainty"]=max_uncertainty
                        # cloud_max_uncertainty[cloud_name]["sp_idx"]=sp_data["sp_idx"]
                        # cloud_max_uncertainty[cloud_name]["pos_x"]=sp_data["pos_x"]
                        # cloud_max_uncertainty[cloud_name]["pos_y"]=sp_data["pos_y"]
                        # cloud_max_uncertainty[cloud_name]["pos_z"]=sp_data["pos_z"]
                        # cloud_max_uncertainty[cloud_name]["class"]=sp_data["class"]
                # 将最大不确定度记录到 cloud_max_uncertainty 中
                # cloud_max_uncertainty[cloud_name] = max_uncertainty
                
                print(f"Scene: {cloud_name}, Max Uncertainty: {max_uncertainty:.2f}")
            
            except Exception as e:
                print(f"Error processing scene {cloud_name}: {e}")
        
        for cloud_name, max_uncertainty in cloud_max_uncertainty.items():
            print(1382,cloud_name,max_uncertainty)

        max_uncertainty_scene = None
        max_uncertainty_value = -float('inf')  # 确保初始值很小
        max_uncertainty_point = None  # 存储最大不确定度超点的xyz坐标
        max_uncertainty_class = None  # 存储最大不确定度超点的预测类别
        max_uncertainty_sp = None  # 

        # 遍历 cloud_max_uncertainty 找到最大不确定度的场景
        for cloud_name, max_uncertainty in cloud_max_uncertainty.items():
            if max_uncertainty["uncertainty"] > max_uncertainty_value:
                max_uncertainty_value = max_uncertainty["uncertainty"]
                max_uncertainty_scene = cloud_name
                
                max_uncertainty_point = (max_uncertainty["pos_x"], max_uncertainty["pos_y"], max_uncertainty["pos_z"])  # xyz 坐标
                max_uncertainty_class = max_uncertainty["class"]  # 预测类别
                max_uncertainty_sp = max_uncertainty["sp_idx"]
                
        # 找到那个点
        print("max_uncertainty_sp:")
        print(max_uncertainty_sp)
        print(max_uncertainty_value)
        # 标注预算 batch_size
        label_budget = batch_size
        print(label_budget)
        file_list = {}
        
        count0=0
        for iteration in range(label_budget):
            # print(count0)
            # print(" ")
            # count0 += 1
            # print("max uncertainty sp:",max_uncertainty_sp)
            # 要剔除本轮的最大不确定度超点
            for cloud_name, superpoints in cloud_sp_uncertainty.items():
                # 找到目标场景
                if cloud_name == max_uncertainty_scene:
                    # 删除满足条件的超点
                    cloud_sp_uncertainty[cloud_name] = [
                        sp for sp in superpoints if sp["sp_idx"] != max_uncertainty_sp
                    ]
                    # print(f"Removed SP Index: {max_uncertainty_sp} from Scene: {cloud_name}")
                    break
            # 加入file_list
            # print("cloud_sp_uncertainty[max_uncertainty_scene] length: ",len(cloud_sp_uncertainty[max_uncertainty_scene])," max_uncertainty_sp: ",max_uncertainty_sp)
            # print(len(cloud_sp_uncertainty[max_uncertainty_scene]))
            # print("max_uncertainty_sp:")
            # print(max_uncertainty_sp)
            # print()

            if max_uncertainty_scene not in file_list:
                file_list[max_uncertainty_scene] = []
            file_list[max_uncertainty_scene].append(max_uncertainty_sp)

            # 动态衰减其他sp_uncertainty
            for sp_data in cloud_sp_uncertainty[max_uncertainty_scene]:
                if max_uncertainty_class == sp_data["class"]:
                    cur_sp_point = np.array([sp_data["pos_x"], sp_data["pos_y"], sp_data["pos_z"]])
                    declay_factor = (np.linalg.norm(cur_sp_point - max_uncertainty_point))/cloud_max_distance[max_uncertainty_scene]
                    sp_data["uncertainty"]=sp_data["uncertainty"] * (1-math.exp(-declay_factor))
            # 遍历本场景的超点，更新  cloud_max_uncertainty
            cloud_max_uncertainty[max_uncertainty_scene]["uncertainty"] = 0
            for sp_data in cloud_sp_uncertainty[max_uncertainty_scene]:
                # print(cloud_max_uncertainty[max_uncertainty_scene])
                if sp_data["uncertainty"] > cloud_max_uncertainty[max_uncertainty_scene]["uncertainty"]:
                    cloud_max_uncertainty[max_uncertainty_scene]={"sp_idx": sp_data["sp_idx"], "uncertainty": sp_data["uncertainty"],"pos_x" : sp_data["pos_x"],"pos_y" : sp_data["pos_y"],"pos_z" : sp_data["pos_z"],"class": sp_data["class"]}

            
            # 遍历 cloud_max_uncertainty 找到最大不确定度的场景
            max_uncertainty_scene = None
            max_uncertainty_value = -float('inf')  # 确保初始值很小
            max_uncertainty_point = None  # 存储最大不确定度超点的xyz坐标
            max_uncertainty_class = None  # 存储最大不确定度超点的预测类别
            max_uncertainty_sp = None  # 

            for cloud_name, max_uncertainty in cloud_max_uncertainty.items():
                if max_uncertainty["uncertainty"] > max_uncertainty_value:
                    max_uncertainty_value = max_uncertainty["uncertainty"]
                    max_uncertainty_scene = cloud_name
                    
                    max_uncertainty_point = (max_uncertainty["pos_x"], max_uncertainty["pos_y"], max_uncertainty["pos_z"])  # xyz 坐标
                    max_uncertainty_class = max_uncertainty["class"]  # 预测类别
                    max_uncertainty_sp = max_uncertainty["sp_idx"]
        print("本轮标注超点数量:")
        for cloud_name, superpoints in file_list.items():
            print(cloud_name,": ",len(superpoints))
        for cloud_name in file_list:
            _help(input_path=self.input_path, data_path=self.data_path, total_obj=total_obj, current_path=next_round_path, cloud_name=cloud_name,
                    superpoint_inds=file_list[cloud_name], w=w, sampler_args=self.sampler_args, prob_class=prob_class_dict[cloud_name],
                    threshold=threshold, budget=budget, min_size=self.min_size)
        # save total_obj
        with open(os.path.join(next_round_path, "total.pkl"), "wb") as f:
            pickle.dump(total_obj, f)

        return
        




        print("prediction round_num:",round_num, "finished")
        if "edcd" in self.sampler_args:
            if batch_size > len(region_reference):
                batch_size = len(region_reference)
            file_list_top, file_list_all = self.create_file_top_and_all(region_reference=region_reference, sorted_inds=sorted_inds, batch_size=batch_size)
            # oracle
            for cloud_name in file_list_top:
                begin_time = time.time()
                superpoint_idns = self.create_sp_inds_with_position(file_list_top=file_list_top,
                                                                    file_list_all=file_list_all,
                                                                    cloud_name=cloud_name)
                print("create_sp_inds_with_position. class_name= " + cloud_name + ", cost_time=", time.time() - begin_time)
                _help(input_path=self.input_path, data_path=self.data_path, total_obj=total_obj, current_path=next_round_path, cloud_name=cloud_name,
                      superpoint_inds=superpoint_idns, w=w, sampler_args=self.sampler_args, prob_class=prob_class_dict[cloud_name],
                      threshold=threshold, budget=budget, min_size=self.min_size)
            print("\n############\n compute distance successfully \n###############\n")

        elif "gcn" in self.sampler_args:
            if batch_size > len(region_reference):
                batch_size = len(region_reference)
            file_list_top, file_list_all = self.create_file_top_and_all(region_reference=region_reference,
                                                                        sorted_inds=sorted_inds, batch_size=batch_size)

            labeled_select_regions, _ = get_labeled_selection_cloudname_spidx_pointidx(input_path=self.input_path, data_path=self.data_path,
                                                                                       labeled_region_reference_dict=labeled_region_reference_dict,
                                                                                       class_num=class_num, round_num=round_num)
            unlabeled_candidate_regions = {}
            # unlabeled_candidate_count = 0
            sampling_batch = 0
            for cloud_name in file_list_top:
                unlabeled_candidate_regions[cloud_name] = {}
                selected_num = len(file_list_top[cloud_name]["sp_idx_list"])
                sampling_batch = sampling_batch + selected_num
                candicate_sp_inds = file_list_all[cloud_name]["sp_idx_list"][:2 * selected_num]
                for sp_idx in candicate_sp_inds:
                    unlabeled_candidate_regions[cloud_name][sp_idx] = file_list_all[cloud_name][sp_idx]
                # unlabeled_candidate_count = unlabeled_candidate_count + len(candicate_sp_inds)
            labeled_select_features, labeled_select_ref, unlabeled_candidate_features, unlabeled_candidate_ref = compute_features(dataset_name=self.dataset_name, test_area_idx=self.test_area_idx, sampler_args=self.sampler_args,
                                                                                                              round_num=round_num, reg_strength=self.reg_strength, model=model,
                                                                                                              labeled_select_regions=labeled_select_regions, unlabeled_candidate_regions=unlabeled_candidate_regions)
            print("\n############\n compute gcn features V successfully \n###############\n")

            file_list = GCN_sampling(labeled_select_features=labeled_select_features, labeled_select_ref=labeled_select_ref,
                                          unlabeled_candidate_features=unlabeled_candidate_features, unlabeled_candidate_ref=unlabeled_candidate_ref,
                                          input_path=self.input_path, data_path=self.data_path,
                                          sampling_batch=sampling_batch, gcn_gpu=gcn_gpu)

            print("\n############\n compute gcn total successfully \n###############\n")
            # oracle
            for cloud_name in file_list:
                _help(input_path=self.input_path, data_path=self.data_path, total_obj=total_obj, current_path=next_round_path, cloud_name=cloud_name,
                      superpoint_inds=file_list[cloud_name], w=w, sampler_args=self.sampler_args, prob_class=prob_class_dict[cloud_name],
                      threshold=threshold, budget=budget, min_size=self.min_size)

        elif "gcn_fps" in self.sampler_args:
            if batch_size > len(region_reference):
                batch_size = len(region_reference)
            #   存储了每个场景内选出的sp，以及该sp中 dominant_label 的点的下标
            #   file_list_top[cloud_name][sp_idx] = dominant_point_ids
            #   file_list_top[cloud_name]["sp_idx_list"].append(sp_idx)
            file_list_top, file_list_all = self.create_file_top_and_all(region_reference=region_reference,
                                                                        sorted_inds=sorted_inds, batch_size=batch_size)
            # 从已标注的sp(根据prediction的labeled_region_reference_dict),以等比例(原本各个do_label的占比)选择出batch = (round_num - 1) * 1000 个sp,return labeled_select_region[cloud_name][sp_idx] = dominant_point_ids
            labeled_select_regions, _ = get_labeled_selection_cloudname_spidx_pointidx(input_path=self.input_path, data_path=self.data_path,
                                                                                       labeled_region_reference_dict=labeled_region_reference_dict,
                                                                                       class_num=class_num, round_num=round_num)
            # 保存每个场景中选出的 前2k个top uncertainty的 sp 中的 dominant_point_ids
            unlabeled_candidate_regions = {}
            # 全部场景中
            sampling_batch = 0
            for cloud_name in file_list_top:
                unlabeled_candidate_regions[cloud_name] = {}
                # 该场景中 前 k 个 top uncertainty的 sp的个数 :k 
                selected_num = len(file_list_top[cloud_name]["sp_idx_list"])
                sampling_batch = sampling_batch + selected_num
                # 该场景中 前 2k 个 top uncertainty的 sp   Q:为什么这里是2倍的?????  A:给最远点采样一些富余选项？
                candicate_sp_inds = file_list_all[cloud_name]["sp_idx_list"][:2 * selected_num]
                for sp_idx in candicate_sp_inds:
                    unlabeled_candidate_regions[cloud_name][sp_idx] = file_list_all[cloud_name][sp_idx]   # file_list_all[cloud_name][sp_idx] = dominant_point_ids
            #  labeled_select_features.append(np.mean(total_last_second_features[dominant_point_ids], axis=0))
            #  labeled_select_ref.append({"cloud_name": cloud_name, "sp_idx": sp_idx})
                #     # batch = (round_num - 1) * 1000 个sp已标注sp的特征：其do_label_point点的平均特征
                # if cloud_name in labeled_select_regions:
                #     for sp_idx in labeled_select_regions[cloud_name]:
                #         dominant_point_ids = labeled_select_regions[cloud_name][sp_idx]
                #         labeled_select_features.append(np.mean(total_last_second_features[dominant_point_ids], axis=0))
                #         labeled_select_ref.append({"cloud_name": cloud_name, "sp_idx": sp_idx})
                # # 该场景下top 2k 个 未标注sp的特征：其do_label_point点的平均特征
            labeled_select_features, labeled_select_ref, unlabeled_candidate_features, unlabeled_candidate_ref = compute_features(dataset_name=self.dataset_name, test_area_idx=self.test_area_idx, sampler_args=self.sampler_args,
                                                                                                              round_num=round_num, reg_strength=self.reg_strength, model=model,
                                                                                                              labeled_select_regions=labeled_select_regions, unlabeled_candidate_regions=unlabeled_candidate_regions)
            print("sampling_batch: ", sampling_batch," \n")
            print("\n############\n compute gcn features V successfully \n###############\n")

            file_list = GCN_FPS_sampling(labeled_select_features=labeled_select_features, labeled_select_ref=labeled_select_ref, unlabeled_candidate_features=unlabeled_candidate_features, unlabeled_candidate_ref=unlabeled_candidate_ref,
                                          input_path=self.input_path, data_path=self.data_path,
                                          sampling_batch=sampling_batch
                                         )
            # with open('/home/ncl/dataset/Semantic3D/analyse_data/gcn_fps_single_sp_100.pkl', 'wb') as f:
            #     pickle.dump(file_list, f)
            # 1/0

            w["gcn_file_num"] = len(file_list)
            w["gcn_sp_num"] = 0
            w["gcn_unlabel_num"] = 0
            for cloud_name in file_list:
                w["gcn_sp_num"] += len(file_list[cloud_name])
                for spid in file_list[cloud_name]:
                    if cloud_name in total_obj["unlabeled"] and spid in total_obj["unlabeled"][cloud_name]:
                        w["gcn_unlabel_num"] += 1

            print("\n############\n compute gcn total successfully \n###############\n")
            # oracle
            for cloud_name in file_list:
                _help(input_path=self.input_path, data_path=self.data_path, total_obj=total_obj,
                      current_path=next_round_path, cloud_name=cloud_name,
                      superpoint_inds=file_list[cloud_name], w=w, sampler_args=self.sampler_args,
                      prob_class=prob_class_dict[cloud_name],
                      threshold=threshold, budget=budget, min_size=self.min_size)

        else: 
            # batch_size=10000
            if batch_size > len(region_reference):
                batch_size = len(region_reference)
            file_list = {}
            # uncertainty 从高到低的sp 索引 找出10000个sp 
            print(len(region_reference))
            for i in sorted_inds[:batch_size]:
                # print("current i:",i)
                # with open("/home/ncl/dataset/Semantic3D/analyse_data/sorted_inds-sp1000-nomerge-new-5.txt", "a") as file:
                # # 写入 i 到文件
                #     file.write(str(i) + " ")
                cloud_name, sp_idx ,do_label= region_reference[i]["cloud_name"], region_reference[i]["sp_idx"],region_reference[i]["do_label"]
                if cloud_name not in file_list:
                    file_list[cloud_name] = []
                # 
                file_list[cloud_name].append(sp_idx)
                # with open("/home/ncl/dataset/Semantic3D/analyse_data/select-sp1000-1.txt", "a") as file:
                #         # 将字典的内容以自定义格式写入文件
                #     # file.write(f"Cloud Name: {cloud_name}, SP Index: {sp_idx}, dp_label: {do_label}\n")
                #     pickle.dump()
            # 1/0    
            # oracle
            # with open("/home/ncl/dataset/Semantic3D/analyse_data/select-sp1000-merge-2-allscene.pkl", "wb") as file:
            #             # 将字典的内容以自定义格式写入文件
            #         # file.write(f"Cloud Name: {cloud_name}, SP Index: {sp_idx}, dp_label: {do_label}\n")
            #     pickle.dump(file_list,file)
            for cloud_name in file_list:
                _help(input_path=self.input_path, data_path=self.data_path, total_obj=total_obj, current_path=next_round_path, cloud_name=cloud_name,
                      superpoint_inds=file_list[cloud_name], w=w, sampler_args=self.sampler_args, prob_class=prob_class_dict[cloud_name],
                      threshold=threshold, budget=budget, min_size=self.min_size)

        # save total_obj
        with open(os.path.join(next_round_path, "total.pkl"), "wb") as f:
            pickle.dump(total_obj, f)