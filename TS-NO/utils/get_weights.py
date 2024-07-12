import torch
import numpy as np
import torch.nn.functional as F



def get_weights_norm(X):
    # 最简单的基于范数的加权
    norms = X[:, :, 0, 0]
    norms = torch.norm(norms, p=1, dim=1)  # 计算每个样本的1范数
    norms = norms.unsqueeze(1)

    max_norms = norms.max()
    min_norms = norms.min()
    
    normalized = (norms - min_norms) / (max_norms - min_norms)
    re_weight = 1 - normalized # 反转，使得最大的最小，最小的最大
    
    return re_weight


def get_weights_attention(X, data_center):
    """
    X: 未标记数据的坐标
    data_center: 标记数据的算术平均点，作为代表
    
    0411工作节点：
    现在代表点计算完成了，下面要计算X和data_center的点积，用来返回权重
    """
    # 基于交叉注意力的加权
    X = X[:,:,0,0]
    _, dim = X.shape
    scale_factor = dim**0.5   # 引入一个放缩
    attention_scores = torch.matmul(X, data_center)
    
    norm_au = torch.norm(X)
    norm_ap = torch.norm(data_center)
    
    attention_scores = (scale_factor * attention_scores) / (norm_au * norm_ap)
    
    attention_scores = F.relu(attention_scores)
    
    
    #attention_weights = F.softmax(attention_scores, dim=0)   # 不要用softmax，不是分类问题
    
    return attention_scores

def get_weights_distance(X, data_center):
    """
    基于与典型点的距离
    X: 未标记数据的坐标
    data_center: 标记数据的算术平均点，作为代表
    This one is not considered in the paper
    """

    X = X[:,:,0,0]
    _, dim = X.shape
    
    data_center = data_center.squeeze().unsqueeze(0)  # 维持算法的共同性
    
    distances = torch.norm(X - data_center, dim=1).reshape(-1,1)
    #beta = 0.5
    #distance_scores = torch.exp(-beta * distances).reshape(-1,1)
    min_distances = distances.min()
    max_distances = distances.max()
    distance_scores = (distances - min_distances) / (max_distances - min_distances)

    #distance_scores = distance_scores / distance_scores.sum() # 归一化之后都太小了
   
    
    return distance_scores