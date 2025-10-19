import numpy as np
from scipy.spatial.distance import cdist

def compute_rank1_map(query_feats, gallery_feats, query_labels, gallery_labels, metric="euclidean"):
    """
    计算Rank-1准确率和mAP（复现原论文3.2节评价标准）
    Args:
        query_feats: 查询集特征（Nq, D），Nq为查询样本数，D为特征维度
        gallery_feats: gallery集特征（Ng, D），Ng为gallery样本数
        query_labels: 查询集标签（Nq,）
        gallery_labels: gallery集标签（Ng,）
        metric: 距离度量（原论文用欧氏距离，"euclidean"或"cosine"）
    Returns:
        rank1: Rank-1准确率（%）
        mAP: 平均准确率（%）
    """
    # 1. 计算距离矩阵（原论文用欧氏距离，式(7)(8)中权重距离的基础）
    if metric == "euclidean":
        dist_matrix = cdist(query_feats, gallery_feats, metric="euclidean")  # (Nq, Ng)
    elif metric == "cosine":
        dist_matrix = 1 - cdist(query_feats, gallery_feats, metric="cosine")  # 余弦相似度→距离（越小越相似）
    else:
        raise ValueError("Metric must be 'euclidean' or 'cosine' (as in paper)")

    # 2. 计算Rank-1
    rank1 = 0
    # 计算每个查询的mAP中间值
    ap_list = []

    for q_idx in range(len(query_feats)):
        # 2.1 对当前查询的距离排序，获取gallery索引
        dist_sorted = np.argsort(dist_matrix[q_idx])  # 从小到大排序（距离越小越相似）
        # 2.2 获取排序后的gallery标签
        gallery_labels_sorted = gallery_labels[dist_sorted]
        # 2.3 计算Rank-1：Top-1是否为同一身份
        if gallery_labels_sorted[0] == query_labels[q_idx]:
            rank1 += 1

        # 2.4 计算当前查询的AP（平均准确率）
        # 找到所有正样本的位置
        pos_mask = (gallery_labels_sorted == query_labels[q_idx])
        pos_indices = np.where(pos_mask)[0]
        if len(pos_indices) == 0:
            ap_list.append(0.0)
            continue
        # 计算每个正样本的Precision@k
        ap = 0.0
        for i, pos_idx in enumerate(pos_indices):
            # 前pos_idx+1个样本中的正样本数（i+1）
            precision = (i + 1) / (pos_idx + 1)
            ap += precision
        # 平均准确率
        ap /= len(pos_indices)
        ap_list.append(ap)

    # 3. 计算最终指标（转为百分比）
    rank1 = (rank1 / len(query_feats)) * 100
    mAP = (np.mean(ap_list)) * 100

    return rank1, mAP